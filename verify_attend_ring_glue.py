"""
Glue proof for _attend_ring: segmentation + per-segment 8192-pad + global-order combine +
output reshape reproduce the default _attend, using a torch stand-in for wan_flash_self_attn
(NKI can't run on CPU). Checks the PYTHON glue (3-D [bs,d,Sk] layout, segment loop, combine,
reshape) — NOT the NKI kernel internals (device-only).

REAL layout (from _attend on device): q_kern [bs,d,Sq], k_kern [bs,d,Sk], v_kern [bs,Sk,d],
where bs = heads_per_shard (=3 at TP4), and Sk is the 8192-multiple padded buffer width.
Kernel returns result [Sq,bs,d]; partials (O[Sq,bs,d], row_max[Sq,bs], row_sum[Sq,bs]).
Default _attend does out.unsqueeze(0).flatten(2) -> [1, Sq, bs*d]. _attend_ring must match.
"""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

bs = 3                  # heads_per_shard
d = 128
Sq = 128
sp = 4
seg = 900               # realistic: 3600-token window / sp=4
k_len_int = sp * seg    # 3600 valid tokens
MULT = 8192
buf_w = ((seg + MULT - 1) // MULT) * MULT     # per-segment padded width (8192)
full_buf = ((k_len_int + MULT - 1) // MULT) * MULT  # default path buffer width
scale = 1.0 / (d ** 0.5)

q_kern = torch.randn(bs, d, Sq)
# default padded buffer: first k_len_int valid, rest pad
k_kern = torch.zeros(bs, d, full_buf); k_kern[:, :, :k_len_int] = torch.randn(bs, d, k_len_int)
v_kern = torch.zeros(bs, full_buf, d); v_kern[:, :k_len_int, :] = torch.randn(bs, k_len_int, d)


def wan_flash_stub(q, k, v, softmax_scale, actual_seqlen_k, use_dynamic_loop=False,
                   return_partials=False):
    # q [bs,d,Sq], k [bs,d,Sk], v [bs,Sk,d]; attend first actual_seqlen_k keys
    kk = k[:, :, :actual_seqlen_k]; vv = v[:, :actual_seqlen_k, :]
    S = torch.einsum('bdq,bdk->bqk', q, kk) * softmax_scale     # [bs,Sq,Sk']
    row_max = S.max(dim=-1, keepdim=True).values               # [bs,Sq,1]
    p = torch.exp(S - row_max)
    O = torch.einsum('bqk,bkd->bqd', p, vv)                    # [bs,Sq,d] unnormalized
    row_sum = p.sum(dim=-1, keepdim=True)                      # [bs,Sq,1]
    # kernel returns O [Sq,bs,d] and row_max/row_sum [Sq,bs,1] (trailing 1 from HBM layout)
    if return_partials:
        return (O.permute(1, 0, 2),
                row_max.permute(1, 0, 2),      # [Sq,bs,1]
                row_sum.permute(1, 0, 2))      # [Sq,bs,1]
    return (O / row_sum).permute(1, 0, 2)                      # [Sq,bs,d]


# ----- default path -----
out_default = wan_flash_stub(q_kern, k_kern, v_kern, scale, k_len_int).unsqueeze(0).flatten(2)  # [1,Sq,bs*d]

# ----- ring path (mirror of _attend_ring) -----
m = l = acc = None
for s in range(sp):
    k_seg = torch.zeros(bs, d, buf_w); k_seg[:, :, :seg] = k_kern[:, :, s*seg:(s+1)*seg]
    v_seg = torch.zeros(bs, buf_w, d); v_seg[:, :seg, :] = v_kern[:, s*seg:(s+1)*seg, :]
    O_s, max_s, sum_s = wan_flash_stub(q_kern, k_seg, v_seg, scale, seg, return_partials=True)
    # max_s, sum_s already [Sq,bs,1] from kernel — no unsqueeze (matches _attend_ring)
    if m is None:
        m, l, acc = max_s, sum_s, O_s
    else:
        m_new = torch.maximum(m, max_s)
        cp = torch.exp(m - m_new); cc = torch.exp(max_s - m_new)
        l = l * cp + sum_s * cc
        acc = acc * cp + O_s * cc
        m = m_new
out_ring = (acc / l).unsqueeze(0).flatten(2)                   # [1,Sq,bs*d]

assert out_default.shape == out_ring.shape == (1, Sq, bs*d), (out_default.shape, out_ring.shape)
maxd = (out_default - out_ring).abs().max().item()
print(f"_attend_ring glue vs default _attend: shape {tuple(out_ring.shape)}  max|Δ| = {maxd:.3e}")
assert maxd < 1e-12, "glue diverges — segmentation/combine/reshape wrong"
print("PROOF: _attend_ring (3-D [bs,d,Sk] layout, per-seg 8192-pad, global-order combine,")
print("reshape) matches default _attend to fp64. Kernel internals remain device-only (gate).")
