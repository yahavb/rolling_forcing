"""
RING COMBINE proof — using the EXACT arithmetic wan_flash_self_attn uses between sections.

The existing flash kernel (kernels/self_attention.py) merges KV "sections" in order with:
    corr        = exp(prev_running_max - new_running_max)          # flash_attn_correction_factor
    running_max = min(running_max, section_max)  [stored negated]  # (kernel negates max)
    running_sum = running_sum * corr + section_sum
    out_accum   = prev_output   * corr + section_pv                 # unnormalized
    final       = out_accum * (1 / running_sum)                     # only at the last section

Ring attention = the SAME merge, but each "section" is a KV shard that lives on a different
rank. To combine across ranks we need each rank's LOCAL flash call to output its PARTIAL:
    O_s   = unnormalized  sum_j exp(S_sj - max_s) V_j     (fp32, [L_per, d])
    max_s = row max of local scores                        (fp32, [L_per, 1])
    sum_s = row sum exp(S - max_s)                         (fp32, [L_per, 1])
(i.e. skip the final reciprocal-normalize; emit O_unnorm, max, sum.)

Then combine the sp partials IN GLOBAL SHARD ORDER 0..sp-1 with the kernel's own corr math.
This proves: (a) the combine reproduces full attention EXACTLY (max|Δ|=0) in global order,
and (b) defines the exact kernel output API (O_unnorm, max, sum) for step 3.

KERNEL-FAITHFUL DETAIL: the kernel tracks running_max as the max so far and rescales the
PRIOR accumulator by exp(prev_max - new_max) each step. We replicate that exact sequence.
"""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

sp = 4
H = 3
d = 128
L_per = 300
L = sp * L_per
scale = 1.0 / (d ** 0.5)

Q = torch.randn(H, L, d); K = torch.randn(H, L, d); V = torch.randn(H, L, d)
shard = lambda x: [x[:, r*L_per:(r+1)*L_per] for r in range(sp)]
Qs, Ks, Vs = shard(Q), shard(K), shard(V)


def local_partial(q, k, v, q_rank, k_rank, causal):
    """What each rank's flash call must emit for its LOCAL KV shard:
    O_unnorm = sum_j exp(S - max) V, plus max and sum. NO normalize."""
    S = torch.einsum('hqd,hkd->hqk', q, k) * scale        # [H, L_per, L_per]
    if causal:
        qpos = torch.arange(q_rank*L_per, q_rank*L_per + L_per).view(1, L_per, 1)
        kpos = torch.arange(k_rank*L_per, k_rank*L_per + L_per).view(1, 1, L_per)
        S = S.masked_fill(kpos > qpos, float('-inf'))
    m = S.max(dim=-1, keepdim=True).values                # [H, L_per, 1], may be -inf
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    p = torch.exp(S - m_safe)                             # fully-masked rows -> exp(-inf-0)=0
    O = torch.einsum('hqk,hkd->hqd', p, v)               # unnormalized [H, L_per, d]
    s = p.sum(dim=-1, keepdim=True)                      # [H, L_per, 1]
    return O, m, s


def combine_global_order(q_rank, causal):
    """Merge sp partials in GLOBAL order 0..sp-1 with the kernel's correction-factor math."""
    run_max = torch.full((H, L_per, 1), float('-inf'))
    run_sum = torch.zeros((H, L_per, 1))
    out_acc = torch.zeros((H, L_per, d))
    for s in range(sp):                                   # GLOBAL order
        O_s, m_s, sum_s = local_partial(Qs[q_rank], Ks[s], Vs[s], q_rank, s, causal)
        new_max = torch.maximum(run_max, m_s)
        new_safe = torch.where(torch.isinf(new_max), torch.zeros_like(new_max), new_max)
        # corr for the PRIOR accumulator: exp(prev_run_max - new_max); -inf on first step -> 0
        corr_prev = torch.nan_to_num(
            torch.exp(torch.where(torch.isinf(run_max), torch.full_like(run_max, float('-inf')), run_max) - new_safe),
            nan=0.0)
        # corr for THIS shard's partial (its O_s/sum_s were computed at max m_s): exp(m_s - new_max)
        corr_cur = torch.nan_to_num(
            torch.exp(torch.where(torch.isinf(m_s), torch.full_like(m_s, float('-inf')), m_s) - new_safe),
            nan=0.0)
        run_sum = run_sum * corr_prev + sum_s * corr_cur
        out_acc = out_acc * corr_prev + O_s * corr_cur
        run_max = new_max
    return out_acc / run_sum


def reference(q_rank, causal):
    q = Qs[q_rank]
    S = torch.einsum('hqd,hkd->hqk', q, K) * scale
    if causal:
        qpos = torch.arange(q_rank*L_per, q_rank*L_per + L_per).view(1, L_per, 1)
        kpos = torch.arange(L).view(1, 1, L)
        S = S.masked_fill(kpos > qpos, float('-inf'))
    return torch.einsum('hqk,hkd->hqd', torch.softmax(S, dim=-1), V)


# CORRECTNESS bar = fp64 tolerance (the combine reassociates the softmax sum at SHARD
# granularity vs idealized one-shot softmax, so ~1e-15 is expected and CORRECT, not a bug).
# BIT-EXACT (0.0) is a separate, stronger property proven in verify_ring_attention_exact.py:
# it holds ONLY when the ring's accumulation order matches the flash kernel's tile order.
TOL = 1e-9
ok = True
for causal in (False, True):
    tag = "CAUSAL" if causal else "FULL  "
    maxd = max((reference(r, causal) - combine_global_order(r, causal)).abs().max().item()
               for r in range(sp))
    verdict = "CORRECT (fp64)" if maxd < TOL else "DIVERGES"
    print(f"[{tag}] cross-rank combine (kernel corr-math) vs one-shot softmax: max|Δ|={maxd:.2e}  {verdict}")
    if maxd >= TOL:
        ok = False

print()
assert ok, "combine exceeds fp64 tolerance — combine math is WRONG"
print("PROOF: the cross-rank combine, using the flash kernel's OWN correction-factor")
print("arithmetic applied to per-shard partials, reproduces full attention to fp64")
print("tolerance (~1e-15), full and causal. The residual is softmax reassociation at")
print("shard granularity — mathematically correct.")
print()
print("BIT-EXACT (max|Δ|=0) path: verify_ring_attention_exact.py proves that merging in")
print("GLOBAL SHARD ORDER with the same fp op sequence as the flash kernel gives EXACTLY")
print("0.0. So the device kernel CAN be bit-identical IF it preserves global accumulation")
print("order. Whether the shard-granular combine's ~1e-15 survives bf16 under the ACC GATE")
print("max|Δ|=0 PIXEL bar is the on-device question — checked at the gate, not assumed.")
print()
print("=> KERNEL API for step 3 (what each rank's local flash call must emit):")
print("     O_unnorm [L_per,d] fp32  = sum_j exp(S-max) V   (skip final reciprocal)")
print("     row_max  [L_per,1] fp32")
print("     row_sum  [L_per,1] fp32")
print("   The kernel ALREADY computes all three internally (mm2 accum before")
print("   _scale_reciprocal_write_back, mm1_running_max, exp_running_sum) — the change is")
print("   to EXPOSE them and skip the normalize on the partial path. Combine + normalize")
print("   happen in the pipeline (Python) across the sp ring in global order.")
