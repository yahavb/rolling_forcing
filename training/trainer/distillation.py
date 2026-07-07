import gc
import logging
import shutil

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import os


def _maybe_empty_cache():
    # Neuron has no torch.cuda.empty_cache(); only call it on CUDA.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        # Neuron: TF32 matmul flags are CUDA-only; guarded so they no-op off-GPU.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        # Neuron: all trainable / frozen modules live on the "neuron" device
        # (matches this repo's inference: torch.device("neuron")). Upstream used
        # torch.cuda.current_device().
        self.device = torch.device("neuron") if not torch.cuda.is_available() \
            else torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process:
            self.writer = SummaryWriter(
                log_dir=os.path.join(config.logdir, "tensorboard"),
                flush_secs=10
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # ── Optional precomputed T5 embeds (frees ~9.6GB HBM) ─────────────────
        # If PRECOMPUTED_EMBEDS points at a .pt produced by precompute_embeds.py,
        # we drop the text encoder from the training graph entirely: it is never
        # fsdp_wrapped / moved to the Neuron device, and fwdbwd_one_step looks up
        # embeds by prompt string instead of running T5. When the env var is
        # unset, behavior is IDENTICAL to upstream (in-loop T5) — no regression.
        self.precomputed_embeds_path = os.environ.get("PRECOMPUTED_EMBEDS")
        self.precomputed_prompt_embeds = None
        self.precomputed_negative_embeds = None
        if self.precomputed_embeds_path:
            if self.is_main_process:
                print(f"[precomputed-embeds] loading {self.precomputed_embeds_path}; "
                      f"T5 text encoder will NOT be placed on device")
            payload = torch.load(self.precomputed_embeds_path, map_location="cpu")
            self.precomputed_prompt_embeds = payload["prompt_embeds"]          # {prompt: [1,512,4096]}
            self.precomputed_negative_embeds = payload["negative_prompt_embeds"]  # [1,512,4096]
            # Free the CPU-resident text encoder before it is ever moved/wrapped.
            self.model.text_encoder = None

        # Neuron fp32-master fix: generator and fake_score (critic) are TRAINABLE
        # and MUST keep fp32 master params (see fsdp_wrap docstring). real_score
        # (the frozen 14B teacher) and text_encoder are frozen and stay bf16.
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            fp32_master=True
        )

        # ALL models stay on Neuron — NO CPU offload. The frozen 14B teacher is
        # FULL_SHARD across all 16 cores like the others (~1/16 of 14B bf16 ≈
        # 1.75GB/core). The HBM fix is sharding (full, ÷16), not offload.
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            fp32_master=True
        )

        # Skip wrapping/placing the text encoder entirely when using precomputed
        # embeds (it was set to None above) — this is where the ~9.6GB HBM is freed.
        if self.precomputed_embeds_path is None:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
            )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        # ── fp32-master weight-change verification ────────────────────────────
        # Snapshot one trainable generator parameter at init. After
        # `weight_change_check_step` optimizer steps we assert its value ACTUALLY
        # changed. With a bf16-only master and lr=1.5e-6 the update rounds to zero
        # and this snapshot would stay bit-identical — this catches that silent
        # no-train failure early. Set config.weight_change_check_step<=0 to skip.
        self.weight_change_check_step = getattr(config, "weight_change_check_step", 50)
        self._wc_param_name = None
        self._wc_param_ref = None
        self._wc_param_init = None
        if self.weight_change_check_step and self.weight_change_check_step > 0:
            for n, p in self.model.generator.named_parameters():
                if p.requires_grad:
                    self._wc_param_name = n
                    self._wc_param_ref = p
                    self._wc_param_init = p.detach().float().clone()
                    break
            if self.is_main_process and self._wc_param_name is not None:
                print(f"[fp32-master check] tracking generator param "
                      f"'{self._wc_param_name}' dtype={self._wc_param_ref.dtype}; "
                      f"will verify it changed after {self.weight_change_check_step} steps")

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            ckpt_dirname = f"checkpoint_model_{self.step:06d}"
            ckpt_dir = os.path.join(self.output_path, ckpt_dirname)
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "model.pt")
            torch.save(state_dict, ckpt_path)
            print("Model saved to", ckpt_path)

            # Neuron/k8s: mirror the just-saved checkpoint to an S3-backed dir so
            # checkpoints land on the PVC live and the local logdir (often /tmp)
            # stays bounded. Guarded on CKPT_MIRROR_DIR, rank0 only.
            mirror_dir = os.environ.get("CKPT_MIRROR_DIR")
            if mirror_dir:
                try:
                    dst = os.path.join(mirror_dir, ckpt_dirname)
                    shutil.copytree(ckpt_dir, dst, dirs_exist_ok=True)
                    print("Checkpoint mirrored to", dst)
                except Exception as e:  # never let mirroring kill training
                    print(f"WARNING: failed to mirror checkpoint to {mirror_dir}: {e}")

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            _maybe_empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            if self.precomputed_embeds_path is not None:
                # Precomputed-embeds path: look up each prompt's embed by string
                # (no T5 on device). A prompt missing from the cache fails LOUDLY.
                embeds = []
                for pr in text_prompts:
                    assert pr in self.precomputed_prompt_embeds, (
                        f"prompt not found in precomputed embeds "
                        f"({self.precomputed_embeds_path}): {pr!r}. Re-run "
                        f"precompute_embeds.py with the same --config_path.")
                    embeds.append(self.precomputed_prompt_embeds[pr])
                prompt_embeds = torch.cat(embeds, dim=0).to(  # [B, 512, 4096]
                    device=self.device, dtype=self.dtype)
                conditional_dict = {"prompt_embeds": prompt_embeds}

                if not getattr(self, "unconditional_dict", None):
                    neg = self.precomputed_negative_embeds.to(
                        device=self.device, dtype=self.dtype)
                    neg = neg.repeat(batch_size, 1, 1)  # [B, 512, 4096]
                    self.unconditional_dict = {"prompt_embeds": neg.detach()}
                unconditional_dict = self.unconditional_dict
            else:
                # Upstream in-loop T5 (unchanged when PRECOMPUTED_EMBEDS unset).
                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)

                if not getattr(self, "unconditional_dict", None):
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size)
                    unconditional_dict = {k: v.detach()
                                          for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
                else:
                    unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        # Neuron: use self.device (torch.device("neuron")) instead of the
        # hard-coded "cuda" device strings from upstream.
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=self.device, dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # ── fp32-master weight-change verification ────────────────────────
            # After weight_change_check_step steps, assert the tracked generator
            # param actually moved. If it is bit-identical the trainable network
            # is silently not learning (the bf16-master rounding bug) — fail loud.
            if (self._wc_param_init is not None
                    and self.step == self.weight_change_check_step):
                with torch.no_grad():
                    delta = (self._wc_param_ref.detach().float()
                             - self._wc_param_init).abs().max().item()
                if self.is_main_process:
                    print(f"[fp32-master check] after {self.step} steps, "
                          f"max|Δ| of '{self._wc_param_name}' = {delta:.3e}")
                assert delta > 0.0, (
                    f"[fp32-master check] generator param '{self._wc_param_name}' "
                    f"did NOT change after {self.step} steps (max|Δ|={delta:.3e}). "
                    f"This is the bf16-master no-train bug — ensure trainable "
                    f"modules use fp32 master params (fsdp_wrap fp32_master=True)."
                )
                # free the init snapshot
                self._wc_param_init = None
                self._wc_param_ref = None

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                _maybe_empty_cache()
                self.save()
                _maybe_empty_cache()

            # Logging
            if self.is_main_process:

                if TRAIN_GENERATOR:
                    self.writer.add_scalar(
                        "generator_loss",
                        generator_log_dict["generator_loss"].mean().item(),
                        self.step
                    )
                    self.writer.add_scalar(
                        "generator_grad_norm",
                        generator_log_dict["generator_grad_norm"].mean().item(),
                        self.step
                    )
                    self.writer.add_scalar(
                        "dmdtrain_gradient_norm",
                        generator_log_dict["dmdtrain_gradient_norm"].mean().item(),
                        self.step
                    )

                self.writer.add_scalar(
                    "critic_loss",
                    critic_log_dict["critic_loss"].mean().item(),
                    self.step
                )
                self.writer.add_scalar(
                    "critic_grad_norm",
                    critic_log_dict["critic_grad_norm"].mean().item(),
                    self.step
                )

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                _maybe_empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    self.writer.add_scalar(
                        "per iteration time",
                        current_time - self.previous_time,
                        self.step
                    )
                    print(
                        f"Step {self.step} | "
                        f"Iteration time: {current_time - self.previous_time:.2f} seconds | "
                    )
                    self.previous_time = current_time