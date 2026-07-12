"""
CPU proof for RING (context-parallel) ATTENTION — the "all the way downstream" CP.

CURRENT CP (denoise + merged): Q is sp-sharded (each rank holds Q_r = L/sp rows), but
K and V are FULL — every rank world-gathers all L tokens. That full-KV materialization
is the remaining cross-rank cost, identical in SP and today's CP.

RING CP: keep Q, K, V ALL sp-sharded (rank r holds Q_r, K_r, V_r, each L/sp long).
No rank ever holds full K/V. Compute attention by rotating K/V shards around the sp
ring with ONLINE SOFTMAX accumulation:

    m, l, acc = -inf, 0, 0                         # running max, denom, output
    K_cur, V_cur = K_r, V_r
    for step in range(sp):
        S = (Q_r @ K_cur^T) * scale                # local scores vs shard in hand
        (apply causal mask for this shard's global positions)
        m_new = max(m, rowmax(S))
        p = exp(S - m_new)
        l = l * exp(m - m_new) + rowsum(p)         # rescale running denom
        acc = acc * exp(m - m_new) + p @ V_cur     # rescale running output
        m = m_new
        K_cur, V_cur = recv_from_prev, (send K_cur/V_cur to next)   # ring shift
    out_r = acc / l                                # == softmax(Q_r @ K_full) @ V_full

Reference: the CURRENT path — Q_r (this rank's shard) attends the FULL materialized
K/V in one shot (what wan_flash_self_attn does today). If out_r matches the reference
for every rank, ring attention is correct.

HONESTY: online softmax REASSOCIATES the softmax sum, so unlike the layout-only CP
proofs (max|Δ|=0), this is NOT bit-identical — expect ~1e-12 in fp64. That matters:
the on-device ACC GATE is max|Δ|=0 PIXELS; in bf16 a tiny attention delta could flip
a pixel LSB and trip the gate even though the math is exact. This proof measures the
fp64 delta so we know how much headroom we have before the bf16/gate question.

We test BOTH the full (non-causal) case and the causal case, since RF self-attn within
a window is causal (a query attends only KV at <= its global position).
"""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# realistic-ish small shapes (proof is about the algorithm, not size)
sp = 4                      # context-parallel ring size (CP4)
n_heads = 3                 # heads_per_shard at TP4 (12 heads / 4)
d = 128                     # head_dim
L_per = 300                 # tokens per sp rank
L = sp * L_per              # full sequence
scale = 1.0 / (d ** 0.5)

# full Q/K/V, then shard along sequence
Q = torch.randn(n_heads, L, d)
K = torch.randn(n_heads, L, d)
V = torch.randn(n_heads, L, d)

def shard(x):   # [H, L, d] -> list of sp [H, L_per, d]
    return [x[:, r*L_per:(r+1)*L_per] for r in range(sp)]

Qs, Ks, Vs = shard(Q), shard(K), shard(V)


# ---------- reference: Q_r attends FULL K/V in one shot (current path) ----------
def reference(causal):
    outs = []
    for r in range(sp):
        q = Qs[r]                                   # [H, L_per, d]
        S = torch.einsum('hqd,hkd->hqk', q, K) * scale   # [H, L_per, L]
        if causal:
            # query global row = r*L_per + i ; key global col = j ; mask j > row
            qpos = torch.arange(r*L_per, r*L_per + L_per).view(1, L_per, 1)
            kpos = torch.arange(L).view(1, 1, L)
            S = S.masked_fill(kpos > qpos, float('-inf'))
        P = torch.softmax(S, dim=-1)
        outs.append(torch.einsum('hqk,hkd->hqd', P, V))
    return outs


# ---------- ring: Q_r sees K/V shards one at a time, online-softmax merge ----------
def ring(causal):
    outs = []
    for r in range(sp):
        q = Qs[r]                                   # [H, L_per, d]
        m = torch.full((n_heads, L_per, 1), float('-inf'))
        l = torch.zeros((n_heads, L_per, 1))
        acc = torch.zeros((n_heads, L_per, d))
        # ring: at step s, rank r holds the shard originally owned by (r - s) mod sp
        for s in range(sp):
            src = (r - s) % sp                      # which global shard is in hand
            kc, vc = Ks[src], Vs[src]
            S = torch.einsum('hqd,hkd->hqk', q, kc) * scale   # [H, L_per, L_per]
            if causal:
                qpos = torch.arange(r*L_per, r*L_per + L_per).view(1, L_per, 1)
                kpos = torch.arange(src*L_per, src*L_per + L_per).view(1, 1, L_per)
                S = S.masked_fill(kpos > qpos, float('-inf'))
            m_new = torch.maximum(m, S.max(dim=-1, keepdim=True).values)
            # a shard fully masked out (all -inf) -> m_new stays -inf there; guard exp
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(S - m_safe)
            scale_old = torch.exp(torch.where(torch.isinf(m), torch.full_like(m, float('-inf')), m) - m_safe)
            scale_old = torch.nan_to_num(scale_old, nan=0.0)   # first step: exp(-inf)=0
            l = l * scale_old + p.sum(dim=-1, keepdim=True)
            acc = acc * scale_old + torch.einsum('hqk,hkd->hqd', p, vc)
            m = m_new
        outs.append(acc / l)
    return outs


for causal in (False, True):
    ref, rng = reference(causal), ring(causal)
    maxd = max((ref[r] - rng[r]).abs().max().item() for r in range(sp))
    tag = "CAUSAL" if causal else "FULL  "
    print(f"[{tag}] ring-attention vs full-KV reference:  max|Δ| = {maxd:.3e}")
    assert maxd < 1e-9, f"RING ATTENTION DIVERGES ({tag}) — algorithm wrong, do not build kernel"

print()
print("PROOF PASSED (fp64): ring online-softmax over sp K/V shards reproduces the")
print("current full-KV attention for BOTH full and causal masking, to fp64 tolerance.")
print("The ~1e-13 residual is softmax REASSOCIATION, not a bug — but it means the")
print("on-device path is NOT bit-identical. NEXT GATE QUESTION: does that residual, in")
print("bf16, stay under the ACC GATE's max|Δ|=0 PIXEL bar? That is the real risk and")
print("must be checked on device before trusting the kernel — the fp64 proof only")
print("establishes the algorithm is mathematically correct.")
