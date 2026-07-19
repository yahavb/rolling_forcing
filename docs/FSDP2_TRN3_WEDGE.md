# 14B Student FSDP2 `fully_shard` Wedge on trn3 / SDK 2.31 — Consolidated Debug Log

Single source of truth for the `fsdp2_wrap_student` hang that blocks the **14B** DMD distill
from running end-to-end on a full 64-rank trn3 instance. Consolidates the six `rf-distill-14b-*`
branches. **The 1.3B distill is unaffected** (16-core, student=4 — the proven layout in
`PAPER_parallelism_neuron_training.md` §3.2). This is a scale-up (student=32) + SDK-2.31 problem.

## The symptom

64-core LNC2 layout (`configs/rolling_forcing_dmd_t4.yaml`: teacher_tp/student_tp/fake_tp =
16/32/16): teacher ranks 0-15, student 16-47 (**32 ranks**), critic 48-63. The student's FSDP2
`fully_shard` (per Wan block) **hangs a subset of the 32 student ranks** — no error, no
traceback, silent. The finished ranks then block downstream (student-PG all-gather in the first
rollout forward), so the other 48 ranks eventually time out on the world `bcast x_t` with
`CCOM WARN … Timeout waiting for RX` doubling 480→960→…→15360 s. One root cause (stuck student
ranks), one big downstream symptom (world collective timeout).

py-spy on the stuck ranks (cpushard run): wedged in a C call at `_fsdp_param.py:394
.copy_(sharded_param)` — FSDP2 placing each param's shard **onto the neuron device**. **No
`dist.*` in the frame** → this is LOCAL device-runtime contention (concurrent per-block
shard-copies on one rank), NOT a collective desync. That distinction drives the whole fix arc.

## Branch-by-branch (each tip is one hypothesis tested)

| Branch | tip | Change to `fsdp2_wrap_student` | Result (stuck / 32) |
|---|---|---|---|
| `rf-distill-14b-fsdp2-mesh-deadlock` | `94866fb` | `DeviceMesh.from_group(student_pg)` instead of `DeviceMesh("neuron", tensor(ranks))` | 6 stuck |
| `…-cpushard` | `d64f3da` | + `m.to("cpu")` before `fully_shard` (shard-copy in host mem) | **2 stuck, 30/32 through all blocks — best** |
| `…-barrier` | `e3c1b3a` | + `dist.barrier(group=student_pg)` per block | ~0 — but `barrier` UNSUPPORTED on neuron backend |
| `rf-distill-14b-student16` | `42b9fea` | student_tp 32→16, **dropped** the unsupported barrier | 2 stuck (of 16) — regressed |
| `…-fsdp2-allreduce` | `88ab567`* | replace barrier with in-place `all_reduce(student_pg)` per block | **0/32 past block 1 — WORSE** |
| `…-fsdp2-drain` (current fix) | `bf7f03d` | replace barrier with per-rank device **drain** (`p.to_local().cpu().item()`), off cpushard | not yet confirmed |

\* `88ab567` is the allreduce branch's tip (a docs commit); the code change is the commit under it.

### What each step established

1. **`DeviceMesh(device_type, mesh)` builds its OWN internal process group** (`_init_process_groups`),
   which on SDK 2.31 is world-order-sensitive and hangs a subset. Fix: wrap the ALREADY-created
   `student_pg` (built in lockstep by all ranks in `make_distill_groups` via `dist.new_group`)
   with `DeviceMesh.from_group()` — no new collective. (6/32 residual remained.) This is the same
   trap `PAPER_parallelism_neuron_training.md` §4 flags for `init_device_mesh` over the world.
2. **The residual is an on-device shard-copy, not a collective.** Building the model on
   `device("neuron")` means FSDP2's per-param `new_zeros + copy_` runs on-device; 32 ranks issuing
   per-block device allocs/copies at once wedges the neuron runtime on a random ~6. `m.to("cpu")`
   moves the *source* to host → 6→2 stuck, and 30/32 complete all 40 blocks + reach rollout.
   **cpushard is the high-water mark.**
3. **A cross-rank barrier drove the 2 residual to ~0** — but `dist.barrier(group=student_pg)` is
   **unsupported** on the neuron backend ("Barrier is implemented only for default group").
4. **Dropping the barrier + halving TP (student16) regressed to 2 stuck** — smaller TP alone does
   not fix it; the barrier was doing the work.
5. **KEY NEGATIVE RESULT — do NOT add a cross-rank collective here.** Replacing the unsupported
   barrier with `all_reduce(student_pg)` per block made it **dramatically worse: 0/32 past block 1.**
   `fully_shard` **is itself a collective** (all-gather over the student mesh); interleaving a
   foreign `all_reduce` on the same group between blocks corrupts the neuron collective ordering →
   block-2 deadlock for all 32. (Corollary: you also cannot serialize `fully_shard` one-rank-at-a-time
   — it's collective, all ranks must call together.) The lockstep barrier also made the wedge
   *unanimous* (all 32 at once) instead of 2 stragglers — worse-looking, but it localized the defect.

## Current fix — per-rank device drain (branch `rf-distill-14b-fsdp2-drain`)

Off cpushard (the 30/32 base). After each `fully_shard`, force **this rank** to finish its
shard-copy before issuing the next block's, via a host read of one just-sharded param:

```python
def _drain(mod):
    for p in mod.parameters():
        t = p.to_local() if hasattr(p, "to_local") else p
        t.detach().float().sum().cpu().item()   # host read -> device copy must complete
        return
```

Purely **per-rank, no `dist.*`** → cannot desync or corrupt `fully_shard`'s own collectives.
It targets the actual residual (concurrent per-block device copies piling up). Per-block
`[dbg r.. fully_shard block i/N + drain DONE]` markers localize any remaining wedge by block index.

**How to read the next run:**
- All 32 reach `block 40/40 + drain DONE` → `FSDP2 fully_shard DONE` → all reach
  `it0 (b): ENTER world bcast x_t` = **fixed**.
- Wedges deeper than block 1 but not all 40 = drain helped, residual localized to a block.
- Back to 2 stuck / 30 through = we're at the cpushard floor; the residual is a genuine SDK 2.31
  on-device shard-copy bug to **escalate to the Neuron team** (not a userland fix).

## Rules for anyone touching `fsdp2_wrap_student` on neuron

- **Never** add a cross-rank collective (`barrier`/`all_reduce`/`all_gather`) between the per-block
  `fully_shard` calls — it corrupts `fully_shard`'s own collective ordering (proven: allreduce → 0/32).
- Build the mesh with `DeviceMesh.from_group(student_pg)`, never `DeviceMesh(dev, mesh)` or
  `init_device_mesh` over the world (both spawn a world-order-sensitive PG → subset hang).
- Keep `m.to("cpu")` before `fully_shard` (shard-copy in host memory, not on-device).
- Only per-rank, non-collective ops are safe to interleave (device drain, host reads).

## Cross-refs

- `PAPER_parallelism_neuron_training.md` — §2.2 (unmatched collective = silent deadlock), §3.2
  (proven 16-core teacher8/student4/critic4), §4 (FSDP2 subgroup mesh, `init_device_mesh`
  deadlocks). The 1.3B stack that works.
- `DISTILL_DIVERGENCE_ROOTCAUSE.md` — separate problem: even once 14B RUNS, the drifted recipe
  (grad_accum 1 vs 4, ema 0.999 vs 0.99) diverges. 14B can't take grad_accum=4 without a real
  memory fix (that batch=1 was the OOM dodge). "Runs on 64 ranks" ≠ "converges."
- Manifests: `rf-distill-14b-job.yaml` (BRANCH env = runtime code pointer, currently
  `rf-distill-14b-fsdp2-drain`). Cluster dir: `~/k8s/clusters/KEEP-trn3pds-dev1/`.
