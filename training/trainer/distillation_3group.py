"""Three-group DMD distillation trainer (flat loop, ported from StreamDiffusionV2's
distill_sdv2.py main()). Replaces the encapsulated single-rank DMD path when
DISTILL_THREE_GROUP=1.

WHY: with all three nets (14B teacher + 1.3B student + 1.3B critic) FSDP-sharded onto
ALL 16 ranks, each core holds ~11.2GB of co-resident model shards; the rollout's fused
NEFF then needs ~1.5GB more scratchpad than the ~24GB/core budget -> OOM. SD's proven
fix (three_group=True): give EACH net its own tp-rank group so a core holds ONE model.
Cross-group transfer is via GLOBAL broadcast (Neuron supports broadcast, not P2P):
  (a) student rolls out x0 (its group)          -> bcast x_t, t, x0 to all
  (c) teacher scores x_t (its group)            -> bcast real_pred back
  (d) critic scores x_t (its group)             -> bcast fake_pred back
  (e) student DMD update (its group)
  (f) critic diffusion update (its group)
Every rank calls every broadcast in lockstep (collective requirement); only the src
group provides real data, the rest send/recv zeros.

This uses RF's OWN model APIs (WanDiffusionWrapper scorers + RollingForcingTrainingPipeline
rollout + RF's DMD normalizer), NOT SD's single-block causal scorer — only the placement
+ broadcast skeleton is from SD.
"""
import gc
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf

from utils.distributed import fsdp_wrap, make_distill_groups, launch_distributed_job
from utils.misc import set_seed
from utils.wan_wrapper import WanDiffusionWrapper
from pipeline import RollingForcingTrainingPipeline
from wan.modules.causal_model import CausalWanAttentionBlock
from wan.modules.model import WanAttentionBlock

_WAN_BLOCKS = {CausalWanAttentionBlock, WanAttentionBlock}


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        launch_distributed_job()
        # ASYMMETRIC groups: the 14B teacher needs more ranks than the 1.3B student/critic
        # (RF's FSDP teacher holds full activations per rank, so 4 OOMs). Default:
        # teacher=8, student=4, critic=4 -> all 16 cores, 14B bf16/8 = 3.5GB/core.
        self.groups = make_distill_groups(
            int(getattr(config, "tp_degree", 4)),
            teacher_tp=int(getattr(config, "teacher_tp", 8)),
            student_tp=int(getattr(config, "student_tp", 4)),
            fake_tp=int(getattr(config, "fake_tp", 4)))
        g = self.groups
        self.my_rank = g["my_rank"]
        self.world_size = g["world_size"]
        self.in_teacher, self.in_student, self.in_fake = g["in_teacher"], g["in_student"], g["in_fake"]
        self.tsrc, self.ssrc, self.fsrc = g["tsrc"], g["ssrc"], g["fsrc"]
        self.is_main_process = self.my_rank == 0
        self.device = torch.device("neuron")
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32

        if config.seed == 0:
            rs = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(rs, src=0)
            config.seed = rs.item()
        set_seed(config.seed + self.my_rank)

        self._log(f"placement: world={self.world_size} "
                  f"teacher_ranks={g['teacher_ranks']} student_ranks={g['student_ranks']} fake_ranks={g['fake_ranks']} "
                  f"rank={self.my_rank} teacher={self.in_teacher} student={self.in_student} fake={self.in_fake}")

        # ── precomputed embeds (no T5 on device) ──
        self.embeds_by_prompt = None
        self.neg_embed = None
        emb_path = os.environ.get("PRECOMPUTED_EMBEDS")
        assert emb_path and os.path.exists(emb_path), (
            "three-group trainer requires PRECOMPUTED_EMBEDS (no in-loop T5)")
        payload = torch.load(emb_path, map_location="cpu")
        self.embeds_by_prompt = payload["prompt_embeds"]
        self.neg_embed = payload["negative_prompt_embeds"]

        # ── prompts ──
        with open(config.data_path) as f:
            self.prompts = [ln.strip() for ln in f if ln.strip()]
        self._log(f"{len(self.prompts)} prompt(s) from {config.data_path}")

        # ── DMD hyperparameters ──
        self.num_frame_per_block = getattr(config, "num_frame_per_block", 3)
        self.num_training_frames = getattr(config, "num_training_frames", 21)
        self.guidance_scale = getattr(config, "guidance_scale", 3.0)
        self.dfake_gen_update_ratio = getattr(config, "dfake_gen_update_ratio", 5)
        self.warmup = int(getattr(config, "warmup", 10))
        self.grad_accum = max(1, int(getattr(config, "grad_accum", 4)))
        self.iters = int(getattr(config, "iters", 10000))
        self.save_every = int(getattr(config, "save_every", 200))
        self.timestep_shift = getattr(config, "timestep_shift", 5.0)
        self.num_train_timestep = getattr(config, "num_train_timestep", 1000)
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        _b, _f, _c, _h, _w = config.image_or_video_shape
        self.lat_shape = (1, self.num_training_frames, _c, _h, _w)
        # frame_seq = patchified tokens/frame (patch (1,2,2) -> (H//2)*(W//2)). The
        # non-causal scorers (WanModel) PAD the input up to self.seq_len and run one
        # full-sequence SDPA at that size. The wrapper hardcodes seq_len=32760 (=21*1560,
        # the 480x832 default) -> our 15*1200=18000-token x_t gets padded to 32760 and
        # the SDPA compile fails at bf16[1,32760,40,128]. Set seq_len to the ACTUAL
        # num_training_frames * frame_seq so there is no oversized pad.
        self.frame_seq = (_h // 2) * (_w // 2)
        self.score_seq_len = self.num_training_frames * self.frame_seq

        # ── build ONLY this rank's model, sharded within its own group ──
        self.generator = self.real_score = self.fake_score = None
        self.opt_g = self.opt_f = None
        self.scheduler = None
        self._build_group_model(config)

        self._g_since_step = 0
        self._dmdnorm_hist = []
        self.mirror_dir = os.environ.get("CKPT_MIRROR_DIR", "").strip()
        self.output_path = config.logdir

    def _log(self, msg):
        if self.is_main_process:
            print(msg, flush=True)

    def _build_group_model(self, config):
        real_name = getattr(config, "real_name", "Wan2.1-T2V-1.3B")
        if self.in_student:
            self._log("building student generator (1.3B causal, trainable)...")
            self.generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
            self.generator.model.requires_grad_(True)
            if getattr(config, "gradient_checkpointing", False):
                self.generator.enable_gradient_checkpointing()
            self.scheduler = self.generator.get_scheduler()
            self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
            if getattr(config, "alphas_cumprod", None) is None and getattr(self.scheduler, "alphas_cumprod", None) is not None:
                self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
            if getattr(config, "generator_ckpt", False):
                self._log(f"loading student init from {config.generator_ckpt}")
                sd = torch.load(config.generator_ckpt, map_location="cpu")
                # ode_init.pt's "generator" dict is keyed for the WRAPPER
                # (model.patch_embedding.weight, ...), so load into self.generator
                # (WanDiffusionWrapper), NOT self.generator.model — exactly as the
                # original trainer (distillation.py) does.
                if "generator" in sd:
                    sd = sd["generator"]
                elif "model" in sd:
                    sd = sd["model"]
                self.generator.load_state_dict(sd, strict=True)
            self.generator = fsdp_wrap(
                self.generator, sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision, wrap_strategy="transformer",
                transformer_module=_WAN_BLOCKS, process_group=self.groups["student_pg"])
            self.opt_g = torch.optim.AdamW(
                [p for p in self.generator.parameters() if p.requires_grad],
                lr=config.lr, betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay)

        if self.in_fake:
            self._log("building fake_score critic (1.3B, trainable)...")
            self.fake_score = WanDiffusionWrapper(model_name=getattr(config, "fake_name", "Wan2.1-T2V-1.3B"), is_causal=False)
            self.fake_score.seq_len = self.score_seq_len  # actual res, not the 32760 default (avoids pad-to-32760 SDPA)
            self.fake_score.model.requires_grad_(True)
            if getattr(config, "gradient_checkpointing", False):
                self.fake_score.enable_gradient_checkpointing()
            self.scheduler_fake = self.fake_score.get_scheduler()
            self.scheduler_fake.timesteps = self.scheduler_fake.timesteps.to(self.device)
            self.fake_score = fsdp_wrap(
                self.fake_score, sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision, wrap_strategy="transformer",
                transformer_module=_WAN_BLOCKS, process_group=self.groups["fake_pg"])
            self.opt_f = torch.optim.AdamW(
                [p for p in self.fake_score.parameters() if p.requires_grad],
                lr=getattr(config, "lr_critic", config.lr),
                betas=(config.beta1_critic, config.beta2_critic),
                weight_decay=config.weight_decay)

        if self.in_teacher:
            self._log(f"building real_score teacher ({real_name}, FROZEN)...")
            self.real_score = WanDiffusionWrapper(model_name=real_name, is_causal=False)
            self.real_score.seq_len = self.score_seq_len  # actual res, not the 32760 default (avoids pad-to-32760 SDPA)
            self.real_score.model.requires_grad_(False)
            # The 14B teacher loads fp32 (Wan weights are fp32 on disk). Sharded across
            # only its 4-rank group that is 14B*4B/4 = 14GB/core of PARAMS + scoring
            # activation -> 21GB Tensors -> OOM on the teacher scoring x_t. It is FROZEN,
            # so it needs NO fp32 master (that only matters for trainable nets whose tiny
            # updates would round away in bf16). Cast to bf16 BEFORE wrapping -> 14B*2B/4 =
            # 7GB/core. SD builds its 14B teacher bf16 for exactly this reason. The
            # student/critic keep fp32 masters (they train).
            self.real_score.model.to(torch.bfloat16)
            self.real_score = fsdp_wrap(
                self.real_score, sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision, wrap_strategy="transformer",
                transformer_module=_WAN_BLOCKS, process_group=self.groups["teacher_pg"])

        # student needs the rollout pipeline; wire its collectives to the student group
        self.pipeline = None
        if self.in_student:
            dsl = torch.tensor(config.denoising_step_list, dtype=torch.long)
            if getattr(config, "warp_denoising_step", False):
                ts = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                dsl = ts[1000 - dsl]
            self.pipeline = RollingForcingTrainingPipeline(
                denoising_step_list=dsl, scheduler=self.scheduler, generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=getattr(config, "independent_first_frame", False),
                same_step_across_blocks=getattr(config, "same_step_across_blocks", True),
                last_step_only=getattr(config, "last_step_only", False),
                num_max_frames=self.num_training_frames,
                context_noise=getattr(config, "context_noise", 0),
                sync_group=self.groups["student_pg"], sync_src=self.ssrc)

    # ── broadcast helper: every rank calls in lockstep ──
    def _bcast(self, t, src):
        t = t.contiguous()
        dist.broadcast(t, src=src)
        return t

    def _cond(self, prompt):
        e = self.embeds_by_prompt[prompt].to(device=self.device, dtype=self.dtype)
        return {"prompt_embeds": e}

    def _sample_timestep(self, b, num_frame):
        t = torch.randint(self.min_step, self.max_step, (b, 1), device=self.device, dtype=torch.long).repeat(1, num_frame)
        if self.timestep_shift > 1:
            t = self.timestep_shift * (t / 1000) / (1 + (self.timestep_shift - 1) * (t / 1000)) * 1000
        return t.clamp(self.min_step, self.max_step)

    def train(self):
        for it in range(self.iters):
            prompt = self.prompts[it % len(self.prompts)]
            cond = self._cond(prompt)
            emb_shape = cond["prompt_embeds"].shape

            def zeros_lat():
                return torch.zeros(self.lat_shape, dtype=self.dtype, device=self.device)

            # (a) student rollout under no_grad -> detached x0 + its x_t/timestep
            if self.in_student:
                x0_det, x_t, tt, num_frame = self._student_rollout(it, cond)
            else:
                x0_det = x_t = tt = num_frame = None

            # (b) broadcast x_t/t/x0/embeds to all
            if self.in_student:
                embeds = cond["prompt_embeds"]
                x0_send = x0_det
            else:
                x_t = zeros_lat(); tt = torch.zeros((1, self.num_training_frames), dtype=torch.int64, device=self.device)
                embeds = torch.zeros(emb_shape, dtype=self.dtype, device=self.device); x0_send = zeros_lat()
            x_t = self._bcast(x_t, self.ssrc)
            tt = self._bcast(tt.to(torch.int64), self.ssrc)
            embeds = self._bcast(embeds, self.ssrc)
            x0_send = self._bcast(x0_send, self.ssrc)
            condb = {"prompt_embeds": embeds}
            neg = self.neg_embed.to(device=self.device, dtype=self.dtype).repeat(x_t.shape[0], 1, 1)
            uncondb = {"prompt_embeds": neg}

            # (c) teacher scores x_t -> real_pred (CFG)
            real_pred = zeros_lat()
            if self.in_teacher:
                with torch.no_grad():
                    _, real_cond = self.real_score(noisy_image_or_video=x_t, conditional_dict=condb, timestep=tt)
                    _, real_unc = self.real_score(noisy_image_or_video=x_t, conditional_dict=uncondb, timestep=tt)
                    real_pred = real_cond + (real_cond - real_unc) * self.guidance_scale
            real_pred = self._bcast(real_pred, self.tsrc)

            # (d) critic scores x_t -> fake_pred
            fake_pred = zeros_lat()
            if self.in_fake:
                with torch.no_grad():
                    _, fake_pred = self.fake_score(noisy_image_or_video=x_t, conditional_dict=condb, timestep=tt)
            fake_pred = self._bcast(fake_pred, self.fsrc)

            # (e) student DMD update — recompute forward WITH grad
            dmdnorm = float("nan")
            do_g = (it >= self.warmup) and (it % self.dfake_gen_update_ratio == 0)
            if self.in_student and do_g:
                x0_grad, _, _, _ = self._student_rollout(it, cond, with_grad=True)
                grad = (fake_pred - real_pred)
                normalizer = torch.abs(x0_grad - real_pred).mean(
                    dim=list(range(1, x0_grad.dim())), keepdim=True)
                grad = torch.nan_to_num(grad / (normalizer + 1e-8))
                dmdnorm = float(torch.mean(torch.abs(grad)).detach())
                target = (x0_grad - grad).detach()
                if self._g_since_step == 0:
                    self.opt_g.zero_grad(set_to_none=True)
                loss_g = 0.5 * F.mse_loss(x0_grad.double(), target.double()) / self.grad_accum
                loss_g.backward()
                self._g_since_step += 1
                if self._g_since_step >= self.grad_accum:
                    gn = float(self.generator.clip_grad_norm_(10.0))
                    if gn != gn or gn == float("inf"):
                        self._log(f"  [grad] it {it}: NON-FINITE grad_norm={gn} -> SKIP step")
                        self.opt_g.zero_grad(set_to_none=True)
                    else:
                        self.opt_g.step()
                    self._g_since_step = 0
                del grad, target, x0_grad, loss_g

            # (f) critic diffusion (flow) update on x0_send
            lf = float("nan")
            if self.in_fake:
                bb, nf = x0_send.shape[0], x0_send.shape[1]
                tf = self._sample_timestep(bb, nf)
                noise_f = torch.randn_like(x0_send)
                xtf = self.scheduler_fake.add_noise(
                    x0_send.flatten(0, 1), noise_f.flatten(0, 1), tf.flatten(0, 1)
                ).unflatten(0, (bb, nf))
                flow_pred, _ = self.fake_score(noisy_image_or_video=xtf, conditional_dict=condb, timestep=tf)
                # flow-matching target: noise - x0 (velocity)
                flow_tgt = (noise_f - x0_send)
                loss_f = F.mse_loss(flow_pred.float(), flow_tgt.float())
                self.opt_f.zero_grad(set_to_none=True)
                loss_f.backward()
                fn = float(self.fake_score.clip_grad_norm_(10.0))
                if fn != fn or fn == float("inf"):
                    self._log(f"  [grad] it {it}: NON-FINITE critic grad_norm={fn} -> SKIP step")
                    self.opt_f.zero_grad(set_to_none=True)
                else:
                    self.opt_f.step()
                lf = float(loss_f.detach())
                del flow_pred, flow_tgt, loss_f, xtf, noise_f

            # logging
            if self.in_student and dmdnorm == dmdnorm:
                self._dmdnorm_hist.append(dmdnorm)
                if len(self._dmdnorm_hist) > 50:
                    self._dmdnorm_hist.pop(0)
            if self.my_rank == self.ssrc:
                avg = sum(self._dmdnorm_hist) / len(self._dmdnorm_hist) if self._dmdnorm_hist else float("nan")
                phase = "warmup" if it < self.warmup else ("G-step" if do_g else "critic-only")
                print(f"it {it}/{self.iters} [{phase}] dmdnorm={dmdnorm:.4f} dmdnorm_avg50={avg:.4f}", flush=True)
            if self.my_rank == self.fsrc:
                print(f"it {it}/{self.iters}  loss_fake={lf:.4f}", flush=True)

            self.step = it
            if it > 0 and it % self.save_every == 0:
                self._save_ckpt(it)

            # free per-iter graph/tensors before the next iter allocates
            x0_det = x_t = real_pred = fake_pred = x0_send = None
            gc.collect()
            if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
                try:
                    torch.neuron.synchronize()
                except Exception:
                    pass

        self._save_ckpt(self.iters)
        self._log("done.")

    def _student_rollout(self, it, cond, with_grad=False):
        """Run the RF rollout on the student group. Returns (x0, x_t, tt, num_frame).
        no_grad by default; with_grad=True rebuilds the graph for the single backward."""
        noise = torch.randn(self.lat_shape, dtype=self.dtype, device=self.device)
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            out, _, _ = self.pipeline.inference_with_self_forcing(noise=noise, **cond)
        # out: [B, F, C, H, W] x0 prediction (last-21 handled inside for >21; here ==21)
        x0 = out
        num_frame = x0.shape[1]
        b = x0.shape[0]
        tt = self._sample_timestep(b, num_frame)
        if with_grad:
            return x0, None, None, num_frame
        # build x_t from detached x0 for the scorers
        x0d = x0.detach()
        noise_t = torch.randn_like(x0d)
        x_t = self.scheduler.add_noise(
            x0d.flatten(0, 1), noise_t.flatten(0, 1), tt.flatten(0, 1)
        ).unflatten(0, (b, num_frame))
        return x0d, x_t, tt, num_frame

    def _save_ckpt(self, it):
        from utils.distributed import fsdp_state_dict
        if not self.in_student:
            if dist.is_initialized():
                dist.barrier()
            return
        sd = fsdp_state_dict(self.generator)
        if dist.is_initialized():
            dist.barrier()
        if self.my_rank == self.ssrc:
            def _clean(k):
                return (k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
                        .replace("_orig_mod.", ""))
            payload = {"generator": {_clean(k): v for k, v in sd.items()}, "distill_iter": it}
            out = os.path.join(self.output_path, f"model.iter{it}.pt")
            torch.save(payload, out)
            print(f"[ckpt] wrote {out} ({len(payload['generator'])} tensors)", flush=True)
            if self.mirror_dir:
                import subprocess
                dst = os.path.join(self.mirror_dir, os.path.basename(out))
                subprocess.Popen(["bash", "-c", f"mkdir -p {self.mirror_dir} && cp '{out}' '{dst}' && rm -f '{out}'"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
