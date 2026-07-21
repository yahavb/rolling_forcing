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

    # Build the mesh over the student ranks WITHOUT creating a new communicator.
    #
    # DEADLOCK FIX (trn3, traced from run kp2wh): the bare `DeviceMesh("neuron", tensor)`
    # constructor does NOT reuse student_pg — it calls _init_process_groups() and bootstraps
    # a SECOND communicator over the 16 student ranks. That second bootstrap is order-
    # sensitive on the neuron backend: 13/16 ranks completed it and finished fully_shard,
    # 3 (r37/r39/r44) never reached the matching bootstrap step -> `nccl init comm … 0 out
    # of 16` + `Timeout waiting for RX` doubling forever. The student_pg built in lockstep
    # by ALL ranks in make_distill_groups (dist.new_group) ALREADY bootstrapped cleanly;
    # wrap THAT with DeviceMesh.from_group() — no second bootstrap, nothing to desync on.
    if student_pg is not None:
        local_mesh = DeviceMesh.from_group(
            student_pg, "neuron", mesh=torch.tensor(student_ranks, dtype=torch.int))
    else:
        # Fallback (only if the caller didn't pass the pre-created group): the bare
        # constructor, which spawns the second communicator (the deadlock above).
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
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    # ── WEDGE FIX (traced from run vgnr2) — stage params to HOST during the wrap only ──
    # faulthandler dumped all 4 stuck ranks frozen at the IDENTICAL frame:
    #   _fsdp_param.py:389  param_data.new_zeros(padded_sharded_size)   # alloc shard ON NEURON
    #   _fsdp_param.py:394  ....narrow(...).copy_(sharded_param)        # copy ON NEURON
    # No dist.* in the frame; 12/16 ranks finished while 4 hung at DIFFERENT blocks (3/19/
    # 31/37) -> NOT a collective/ordering issue, it's a nondeterministic neuron-runtime race
    # in the on-device shard alloc+copy under 16-way concurrency. param_data is on neuron
    # because the 14B student was built there, so new_zeros/copy_ execute on-device.
    # Move params to CPU FIRST: new_zeros/copy_ then run in HOST memory (no device race),
    # and fully_shard places only the 1/16 shard onto neuron via `mesh`. Wrap-time staging
    # ONLY — all training runs on neuron afterward. (cpushard variant reached 30/32 at
    # student=32; here combined with tp16 + from_group.)
    m.to("cpu")

    # ── INSTRUMENTATION (probe branch) — locate the fully_shard wedge, no behavior change ──
    # Run kp2wh/dvm4l: 12-13/16 student ranks finish fully_shard, 3-4 hang between
    # START and DONE with NO error. 12 finishing while 4 hang PROVES fully_shard is not
    # blocking on a 16-way collective (that would hang all 16) -> the stuck ranks are on
    # LOCAL per-rank work. These markers show WHICH block each rank reaches; faulthandler
    # prints the exact C/Python frame of whatever is still stuck after the timeout.
    import time as _t, faulthandler as _fh, sys as _sys
    _r = dist.get_rank() if dist.is_initialized() else -1

    def _dbg(msg):
        print(f"[dbg r{_r} {_t.strftime('%H:%M:%S')}] {msg}", flush=True)

    # If this rank is still inside the shard loop 600s from now, dump ALL thread stacks
    # (repeat every 300s) so the stuck rank's frame lands in the log. cancel() on success.
    _fh.dump_traceback_later(600, repeat=True, file=_sys.stderr)

    # ── STAGGER (traced from runs vgnr2/njq75) — spread the on-device shard placement ──
    # faulthandler proved the wedge is _fsdp_param.py:390 _init_sharded_param (new_zeros/
    # copy_ that place the shard ON NEURON). m.to("cpu") cut it 4->2 but the frame is
    # unchanged: FSDP2 still does that on-device op, and 16 ranks firing it SIMULTANEOUSLY
    # race the neuron runtime (nondeterministic: different ranks/blocks each run, no dist.*
    # in the frame). fully_shard here is NOT a cross-rank barrier (12/16 finished while
    # others hung), so offsetting each rank's ENTRY is safe -> the per-block device
    # placements interleave instead of colliding. Offset by local student index.
    _local_idx = student_ranks.index(_r) if _r in student_ranks else 0
    _stagger_s = _local_idx * 1.5
    _dbg(f"student: stagger {_stagger_s:.1f}s before fully_shard (local idx {_local_idx})")
    _t.sleep(_stagger_s)

    nblk = len(m.blocks)
    for i, blk in enumerate(m.blocks):
        fully_shard(blk, mesh=local_mesh, mp_policy=mp, reshard_after_forward=True)
        _dbg(f"student: fully_shard block {i+1}/{nblk} DONE")
    fully_shard(m, mesh=local_mesh, mp_policy=mp, reshard_after_forward=True)
    _dbg("student: fully_shard root DONE")
    _fh.cancel_dump_traceback_later()
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
