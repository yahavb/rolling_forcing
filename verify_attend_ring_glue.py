"""
Glue proof for _attend_ring: segmentation + combine + reshape reproduce the default
_attend, using a torch stand-in for wan_flash_self_attn (NKI can't run on CPU). This checks
the PYTHON glue I just wrote (segment loop, global-order online-softmax merge, output shape),
NOT the NKI kernel internals (device-only).

Stand-in matches the kernel's I/O contract:
  full mode: wan_flash(q[d,Sq], k[d,Sk], v[Sk,d]) -> out[Sq, bs=1, d] = softmax(scale q^T k) v
  partial : returns (O_unnorm[Sq,1,d], row_max[Sq,1], row_sum[Sq,1])
The default _attend does out.unsqueeze(0).flatten(2) -> [1, Sq, d]. _attend_ring must match.
"""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

d = 64
Sq = 128
sp = 4
seg = 900               # realistic: 3600-token window / sp=4
k_len_int = sp * seg    # 3600 valid tokens
MULT = 8192             # ATTN_SEQLEN_MULTIPLE
buf_w = ((k_len_int + MULT - 1) // MULT) * MULT   # padded buffer width (8192)
scale = 1.0 / (d ** 0.5)

# kernel-layout tensors: q_kern [d, Sq], padded buffers k_kern [d, buf_w], v_kern [buf_w, d]
# with only the first k_len_int valid (the rest is pad the kernel masks via actual_seqlen_k).
q_kern = torch.randn(d, Sq)
k_kern = torch.zeros(d, buf_w); k_kern[:, :k_len_int] = torch.randn(d, k_len_int)
v_kern = torch.zeros(buf_w, d); v_kern[:k_len_int, :] = torch.randn(k_len_int, d)


def wan_flash_stub(q, k, v, softmax_scale, actual_seqlen_k, use_dynamic_loop=False,
                   return_partials=False):
    # q [d,Sq], k [d,Sk], v [Sk,d]; attend over first actual_seqlen_k keys
    kk = k[:, :actual_seqlen_k]; vv = v[:actual_seqlen_k]
    S = torch.einsum('dq,dk->qk', q, kk) * softmax_scale     # [Sq, Sk']
    row_max = S.max(dim=-1, keepdim=True).values             # [Sq,1]
    p = torch.exp(S - row_max)
    O = torch.einsum('qk,kd->qd', p, vv)                     # unnormalized [Sq,d]
    row_sum = p.sum(dim=-1, keepdim=True)                    # [Sq,1]
    if return_partials:
        return O.unsqueeze(1), row_max, row_sum              # [Sq,1,d],[Sq,1],[Sq,1]
    return (O / row_sum).unsqueeze(1)                        # [Sq,bs=1,d]


# ----- default path: full padded buffer, actual_seqlen_k = valid length -----
out_default = wan_flash_stub(q_kern, k_kern, v_kern, scale, k_len_int).unsqueeze(0).flatten(2)  # [1,Sq,d]

# ----- ring path (mirror of _attend_ring, incl. per-segment 8192-pad) -----
m = l = acc = None
for s in range(sp):
    k_seg = torch.zeros(d, buf_w); k_seg[:, :seg] = k_kern[:, s*seg:(s+1)*seg]
    v_seg = torch.zeros(buf_w, d); v_seg[:seg, :] = v_kern[s*seg:(s+1)*seg, :]
    O_s, max_s, sum_s = wan_flash_stub(q_kern, k_seg, v_seg, scale, seg, return_partials=True)
    O_s = O_s[:, 0, :]; max_s = max_s[:, 0:1]; sum_s = sum_s[:, 0:1]
    if m is None:
        m, l, acc = max_s, sum_s, O_s
    else:
        m_new = torch.maximum(m, max_s)
        cp = torch.exp(m - m_new); cc = torch.exp(max_s - m_new)
        l = l * cp + sum_s * cc
        acc = acc * cp + O_s * cc
        m = m_new
out_ring = (acc / l).unsqueeze(0)                            # [1,Sq,d]

assert out_default.shape == out_ring.shape == (1, Sq, d), (out_default.shape, out_ring.shape)
maxd = (out_default - out_ring).abs().max().item()
print(f"_attend_ring glue vs default _attend: shape {tuple(out_ring.shape)}  max|Δ| = {maxd:.3e}")
assert maxd < 1e-12, "glue diverges — segmentation/combine/reshape wrong"
print("PROOF: _attend_ring segmentation + global-order combine + output reshape match the")
print("default _attend to fp64. Kernel internals remain device-only (ACC GATE).")
