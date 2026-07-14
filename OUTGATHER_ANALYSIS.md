# rf-drop-outgather: the merged-path world all_gather is LOAD-BEARING, not redundant

## What was investigated
forward_merged output tail (dit_attention.py ~712-738): reduce_scatter(attn-tp) ->
all_gather(world) -> restore_layout -> slice this rank's L_full_N. Hypothesis: the world
all_gather is redundant (result mostly discarded) and could be dropped/shrunk to a subgroup
to cut a collective + NEFF per layer = launch-count win.

## Derivation (CPU, topology-exact: world16, tp4, sp4)
- Each rank's FINAL output = its natural-contiguous slice [r*L_full_N:(r+1)*L_full_N]. ✓
- reduce_scatter over attn-tp SUMS heads + SPLITS tokens; token content identical across the
  tp group, so `rearranged` layout controls which slice each tp_rank gets.
- KEY TEST: is rank r's natural slice a SUBSET of its own sp-group's tokens (cu[g*L_cu_sp:]
  ++ dn[g*L_dn_sp:])?  NO — most ranks need tokens from OTHER sp-groups (rank1: 1350/1350
  missing from its group). So RS-alone CANNOT deliver the natural slice.
- attn-sp subgroup gather also insufficient (rank0 missing 900/1350): RS already mixed cu/dn
  across tp-chunks, so needed tokens aren't in one subgroup.

## Verdict
The world all_gather MOVES data that genuinely lives on other sp-groups — it is LOAD-BEARING,
not redundant. It cannot be dropped or shrunk to a subgroup without a full redesign of the CP
output token layout (and any such redesign must still move the same cross-group data). This
launch-reduction lever is CLOSED.

## Net (serving fps optimization, this session)
- RoPE Win1 (batch swap) + Win2 (cache padded grid): +0.8 fps (14.18 -> ~15.0), bit-identical. REAL. -> branch rope-swap-batch.
- Win3 restore_layout rank-slice: wash (launch-bound, not bandwidth-bound).
- Win4 RMSNorm bf16: wash (1.6% gpsimd too small; not bit-identical).
- Collective/launch reduction: no redundant collective exists (derived above).
- Fusion (qkv gather/matmul), KV-window sharding: previously measured-dead.
CONFIRMED SHIPPABLE WIN = RoPE (+0.8 fps).
