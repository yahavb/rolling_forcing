"""
PIPELINE COMBINE proof — models the EXACT (O_unnorm, row_max, row_sum) the modified
wan_flash_self_attn(return_partials=True) emits per KV shard, and proves the pipeline's
cross-rank combine reconstructs full attention.

This is the CPU-verifiable half of ring attention. The kernel-internal write path is
device-only (NKI); THIS proof covers the combine logic that consumes the kernel outputs —
which is the higher-risk correctness surface (the earlier merged-CP garbage bug was a
combine/layout bug, not kernel-internal).

Kernel emits, for query-shard r attending KV-shard s (both length L_per):
    O_unnorm[r,s] = sum_j exp(S_sj - rowmax_s) V_sj      (fp32, [L_per, d])   -- src_buf
    row_max[r,s]  = max_j S_sj                            (fp32, [L_per])      -- un-negated
    row_sum[r,s]  = sum_j exp(S_sj - rowmax_s)            (fp32, [L_per])      -- exp_running_sum
where S_sj = scale * (Q_r . K_sj), with causal masking by GLOBAL positions.

Pipeline combine for query-shard r (global order s=0..sp-1):
    m=-inf, l=0, acc=0
    for s in 0..sp-1:
        m_new = max(m, row_max[r,s])
        corr_prev = exp(m - m_new); corr_cur = exp(row_max[r,s] - m_new)
        l   = l*corr_prev   + row_sum[r,s]*corr_cur
        acc = acc*corr_prev + O_unnorm[r,s]*corr_cur
        m   = m_new
    out_r = acc / l
Must equal softmax(Q_r @ K_full) @ V_full  (fp64 tolerance; bit-exact in global order per
verify_ring_attention_exact.py).
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


def kernel_partial(q_rank, k_rank, causal):
    """EXACT (O_unnorm, row_max, row_sum) the modified kernel emits for one (Q_r, K_s, V_s)."""
    S = torch.einsum('hqd,hkd->hqk', Qs[q_rank], Ks[k_rank]) * scale
    if causal:
        qpos = torch.arange(q_rank*L_per, q_rank*L_per + L_per).view(1, L_per, 1)
        kpos = torch.arange(k_rank*L_per, k_rank*L_per + L_per).view(1, 1, L_per)
        S = S.masked_fill(kpos > qpos, float('-inf'))
    row_max = S.max(dim=-1, keepdim=True).values                 # [H, L_per, 1]
    m_safe = torch.where(torch.isinf(row_max), torch.zeros_like(row_max), row_max)
    p = torch.exp(S - m_safe)                                    # fully-masked row -> 0
    O_unnorm = torch.einsum('hqk,hkd->hqd', p, Vs[k_rank])       # [H, L_per, d]
    row_sum = p.sum(dim=-1, keepdim=True)                        # [H, L_per, 1]
    return O_unnorm, row_max, row_sum


def pipeline_combine(q_rank, causal):
    m = torch.full((H, L_per, 1), float('-inf'))
    l = torch.zeros((H, L_per, 1))
    acc = torch.zeros((H, L_per, d))
    for s in range(sp):                                          # GLOBAL order
        O_s, max_s, sum_s = kernel_partial(q_rank, s, causal)
        m_new = torch.maximum(m, max_s)
        m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
        corr_prev = torch.nan_to_num(
            torch.exp(torch.where(torch.isinf(m), torch.full_like(m, float('-inf')), m) - m_safe), nan=0.0)
        corr_cur = torch.nan_to_num(
            torch.exp(torch.where(torch.isinf(max_s), torch.full_like(max_s, float('-inf')), max_s) - m_safe), nan=0.0)
        l = l * corr_prev + sum_s * corr_cur
        acc = acc * corr_prev + O_s * corr_cur
        m = m_new
    return acc / l


def reference(q_rank, causal):
    S = torch.einsum('hqd,hkd->hqk', Qs[q_rank], K) * scale
    if causal:
        qpos = torch.arange(q_rank*L_per, q_rank*L_per + L_per).view(1, L_per, 1)
        kpos = torch.arange(L).view(1, 1, L)
        S = S.masked_fill(kpos > qpos, float('-inf'))
    return torch.einsum('hqk,hkd->hqd', torch.softmax(S, dim=-1), V)


TOL = 1e-9
ok = True
for causal in (False, True):
    tag = "CAUSAL" if causal else "FULL  "
    maxd = max((reference(r, causal) - pipeline_combine(r, causal)).abs().max().item() for r in range(sp))
    verdict = "CORRECT (fp64)" if maxd < TOL else "DIVERGES"
    print(f"[{tag}] pipeline combine of kernel partials vs full attention: max|Δ|={maxd:.2e}  {verdict}")
    if maxd >= TOL:
        ok = False

# also assert masked-row safety: a query attending a fully-masked shard must contribute 0
Oz, mz, sz = kernel_partial(0, sp-1, causal=True)   # q shard 0 vs last k shard, fully future -> masked
assert sz.abs().max().item() == 0.0, "fully-masked shard must yield row_sum=0 (no contribution)"
print("masked-shard safety: fully-future KV shard contributes row_sum=0  OK")

print()
assert ok, "pipeline combine of kernel partials is WRONG"
print("PROOF: the pipeline combine of the kernel's (O_unnorm,row_max,row_sum) partials,")
print("merged in GLOBAL shard order, reconstructs full attention to fp64 tolerance, full")
print("and causal, with correct fully-masked-shard handling. This is the CPU-verifiable")
print("contract between the modified kernel and the pipeline. The kernel-internal write")
print("path (un-negate max, row-stat DMA, fp32 result) is DEVICE-ONLY and is verified at")
print("the on-device ACC GATE (max|Δ|=0 pixels).")
