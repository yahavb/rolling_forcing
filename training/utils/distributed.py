from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp2_wrap_student(model, student_ranks, transformer_layer_cls, student_pg=None):
    """FSDP2 fully_shard for the STUDENT generator (SD 5d90c6b path). FSDP1's backward
    unshard buffers were NOT resharded/freed after the DMD G-step backward -> the rollout
    activation graph accumulated +~10GB per G-step (proven by memprobe across del+gc+sync).
    FSDP2 fully_shard(reshard_after_forward=True) reshards params after BOTH forward and
    backward, so the graph frees. Per-block + root, matching SD build_student_pipeline.

    Only the student needs this — teacher is frozen (no backward) and critic does a single
    forward+backward (no multi-block rollout), neither OOM'd. Both stay FSDP1.

    student_ranks: the global ranks in the student group (e.g. [8,9,10,11]).
    transformer_layer_cls: set of block classes to shard per-block.
    Returns the model (fully_shard mutates in place)."""
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing)
    from torch.distributed.device_mesh import DeviceMesh

    # Build the student mesh WITHOUT any new collective. Only the student ranks enter this
    # function (teacher/critic never call it), so any world-level collective here deadlocks.
    #
    # DEADLOCK FIX (trn3-dev1 / torch-neuronx 1417): the old form
    #   DeviceMesh("neuron", torch.tensor(student_ranks))
    # does NOT reuse student_pg despite the prior comment — the DeviceMesh(device_type, mesh)
    # constructor CREATES ITS OWN process group internally (_init_process_groups). On the
    # newer torch-neuronx stack that internal PG creation is world-order-sensitive and hangs a
    # subset of the student ranks (observed: 26/32 ranks passed fully_shard, 6 stuck here).
    # Instead wrap the ALREADY-created student_pg (built in lockstep by ALL ranks in
    # make_distill_groups via dist.new_group) with DeviceMesh.from_group() — no new collective.
    if student_pg is not None:
        local_mesh = DeviceMesh.from_group(
            student_pg, "neuron", mesh=torch.tensor(student_ranks, dtype=torch.int))
    else:
        # Fallback (pre-fix behavior) if the caller didn't pass the pre-created group.
        local_mesh = DeviceMesh("neuron", torch.tensor(student_ranks, dtype=torch.int))

    # per-block NO_REENTRANT activation checkpointing (SD: mandatory — OOMs without it)
    m = model.model  # WanDiffusionWrapper.model = the CausalWanModel (has .blocks)
    # WanDiffusionWrapper.__init__ set model.eval() -> self.training=False. The functional
    # training attention path (causal_model self-attn `elif self.training`) needs train mode.
    # SD calls m.train().requires_grad_(True) here for exactly this.
    m.train().requires_grad_(True)
    apply_activation_checkpointing(
        m,
        checkpoint_wrapper_fn=lambda mod: checkpoint_wrapper(mod, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda mod: any(isinstance(mod, c) for c in transformer_layer_cls),
    )
    # DEADLOCK FIX (trn3-dev1): shard-init on CPU, not the neuron device.
    # py-spy showed a subset of student ranks wedged in a C call at
    # _fsdp_param.py:394 `.copy_(sharded_param)` — FSDP2's per-parameter shard init
    # (new_zeros + copy_) runs ON THE NEURON DEVICE because the generator was built on
    # device("neuron"). 32 ranks issuing per-block device allocations/copies at once
    # wedges the Neuron runtime on a random ~6 ranks (not a collective — no dist.* in
    # the frame). Move params to CPU so new_zeros/copy_ happen in host memory during
    # wrap; FSDP2 places the sharded params on the neuron device via `mesh`.
    m.to("cpu")
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in m.blocks:
        fully_shard(blk, mesh=local_mesh, mp_policy=mp, reshard_after_forward=True)
    fully_shard(m, mesh=local_mesh, mp_policy=mp, reshard_after_forward=True)
    return model


def make_distill_groups(tp_degree, teacher_tp=None, student_tp=None, fake_tp=None):
    """SD three-group placement (distill_sdv2.py three_group=True): give EACH net its
    own rank group so each core holds ONE model (teacher | student | fake), not all
    three co-resident. Cross-group transfer is via GLOBAL broadcast (Neuron supports
    broadcast, not P2P): student bcasts x_t/t/x0; teacher & fake each score and bcast
    their pred back. Ranks beyond the assigned groups are IDLE but MUST still join every
    WORLD collective in lockstep.

    ASYMMETRIC by default: the 14B teacher is ~10x the 1.3B student/critic and, unlike
    SD's custom-TP teacher (splits activations too), RF's FSDP teacher holds FULL
    activations per rank — so 4 ranks OOMs. Give the teacher MORE ranks. Default on a
    16-core box: teacher=8, student=4, critic=4 (uses all 16; 14B bf16/8 = 3.5GB/core
    vs 7GB at /4). Override via per-group *_tp; falls back to symmetric tp_degree.

    Returns a dict with per-group ProcessGroup handles + this rank's membership +
    the GLOBAL src rank of each group (for world-broadcasts)."""
    ws = dist.get_world_size()
    my_rank = dist.get_rank()
    t_tp = teacher_tp if teacher_tp is not None else tp_degree
    s_tp = student_tp if student_tp is not None else tp_degree
    f_tp = fake_tp if fake_tp is not None else tp_degree
    assert ws >= t_tp + s_tp + f_tp, (
        f"three-group placement needs world>=teacher_tp+student_tp+fake_tp "
        f"({t_tp}+{s_tp}+{f_tp}={t_tp+s_tp+f_tp}); got world={ws}")
    teacher_ranks = list(range(0, t_tp))
    student_ranks = list(range(t_tp, t_tp + s_tp))
    fake_ranks = list(range(t_tp + s_tp, t_tp + s_tp + f_tp))
    # new_group is COLLECTIVE: every rank must call for all groups in the same order.
    teacher_pg = dist.new_group(ranks=teacher_ranks)
    student_pg = dist.new_group(ranks=student_ranks)
    fake_pg = dist.new_group(ranks=fake_ranks)
    return {
        "teacher_ranks": teacher_ranks, "student_ranks": student_ranks, "fake_ranks": fake_ranks,
        "teacher_pg": teacher_pg, "student_pg": student_pg, "fake_pg": fake_pg,
        "tsrc": teacher_ranks[0], "ssrc": student_ranks[0], "fsrc": fake_ranks[0],
        "in_teacher": my_rank in teacher_ranks,
        "in_student": my_rank in student_ranks,
        "in_fake": my_rank in fake_ranks,
        "my_rank": my_rank, "world_size": ws, "tp_degree": tp_degree,
    }


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False, fp32_master=False, process_group=None):
    # ── Neuron fp32-master-weight fix ─────────────────────────────────────────
    # THE critical distillation fix. For TRAINABLE networks (generator +
    # fake_score/critic) we MUST keep fp32 master parameters. A bf16-only param
    # with lr=1.5e-6 rounds every update to zero (bf16 resolution ~3.9e-3 >>
    # Δp/p ~1.5e-4), so the model silently never trains.
    #
    # BUT the fp32 master must NOT mean fp32 COMPUTE. FSDP MixedPrecision keeps the
    # sharded flat parameter in its LOADED dtype (fp32 here — ode_init loads fp32,
    # confirmed by the [fp32-master check]) as the optimizer master, and casts to
    # `param_dtype` only for forward/backward. Setting param_dtype=float32 forced
    # fp32 COMPUTE while activations stay bf16 (self.dtype), so the DiT's nn.Linear
    # matmuls hit "aten.mm: input datatypes mismatched" and failed to compile on
    # Neuron. The fp32-master-ness comes from the fp32 flat param, NOT from the
    # compute dtype. So compute in bf16 (param_dtype=bfloat16) to match activations,
    # while FSDP retains the fp32 master for the optimizer step — exactly SD's
    # MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32), which trains
    # the same lr=1.5e-6 for 1000s of iters. reduce_dtype=float32 keeps grad reduction
    # full-precision so the tiny updates survive. RF's [fp32-master check] at step 50
    # still fires loudly if the master somehow isn't retained. The fp32_master arg is
    # now inert (kept for call-site compatibility): every net computes in bf16, and the
    # fp32-master-ness is provided by FSDP retaining the fp32-loaded flat param.
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    # Neuron uses its own collectives (backend="neuron"); NCCL knobs are a no-op
    # but harmless. Left as-is to minimize divergence from upstream.
    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        process_group=process_group,   # None -> WORLD; a group -> shard within that group only
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        # Neuron: FSDP device_id is the local NeuronCore ordinal (torchrun
        # LOCAL_RANK), not a CUDA device. Matches this repo's inference which
        # runs one rank per NeuronCore.
        device_id=int(os.environ.get("LOCAL_RANK", 0)),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "neuron"):
    # Neuron: default backend is "neuron" (matches this repo's inference
    # generate_latents.py / serve.py which call dist.init_process_group(
    # backend="neuron")). torch.cuda.set_device is not applicable on Neuron;
    # the runtime binds one NeuronCore per rank via LOCAL_RANK, so we do not
    # call any set_device here.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])  # noqa: F841 (kept for parity)
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
