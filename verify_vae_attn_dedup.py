"""CPU proof: VAE AttentionBlock q-local + kv-full == old gather-full-then-slice (max|Δ|=0).

Old AttentionBlock.forward: all_gather full x -> norm+to_qkv+attention+proj on the FULL (h,W)
map on EVERY rank -> keep only [w_start:w_end]. = world-x duplicated compute + full-tensor
all_gather + full output, 15/16 discarded. New: each rank computes Q on its OWN width-slice;
only K,V from the gathered full map (attention reads all keys). q_local @ k_full,v_full yields
exactly this rank's output tokens. norm (per-position RMS over channels) + to_qkv/proj
(conv1x1, per-position) are slice-local. Uses the REAL _split_qkv + sdpa reshape layout."""
import torch
torch.manual_seed(0); torch.set_default_dtype(torch.float64)

def split_qkv(qkv):
    cc = qkv.shape[1] // 3; bt = qkv.shape[0]
    qkv = qkv.reshape(bt, 1, cc*3, -1).permute(0, 1, 3, 2).contiguous()
    return qkv[:, :, :, 0:cc], qkv[:, :, :, cc:2*cc], qkv[:, :, :, 2*cc:3*cc]
def sdpa(q, k, v):
    D = q.shape[-1]; s = (q @ k.transpose(-2, -1)) * (D**-0.5)
    s = s - s.amax(3, keepdim=True); e = torch.exp(s); a = e / e.sum(3, keepdim=True)
    return a @ v

worst = 0.0
for world, b, c, t, h, w_local in [(16,1,16,3,8,5),(4,1,8,2,6,5),(8,1,32,1,4,3)]:
    W = world * w_local
    Wqkv = torch.randn(3*c, c, 1, 1)
    to_qkv = lambda x: torch.nn.functional.conv2d(x, Wqkv)
    x_full = torch.randn(b, c, t, h, W)
    xw = x_full.transpose(1, 2).reshape(b*t, c, h, W)
    q, k, v = split_qkv(to_qkv(xw))
    o = sdpa(q, k, v).squeeze(1).permute(0, 2, 1).reshape(b*t, c, h, W)
    for r in range(world):
        xl = x_full[..., r*w_local:(r+1)*w_local].transpose(1, 2).reshape(b*t, c, h, w_local)
        ql, _, _ = split_qkv(to_qkv(xl))
        on = sdpa(ql, k, v).squeeze(1).permute(0, 2, 1).reshape(b*t, c, h, w_local)
        worst = max(worst, (on - o[..., r*w_local:(r+1)*w_local]).abs().max().item())
    print(f"world={world} c={c} h={h} w_local={w_local}: max|Δ|={worst:.2e}")

assert worst < 1e-10, "LAYOUT MISALIGN"
print("\nPROOF PASSED: q-local + kv-full == full-then-slice (real layout). Removes 15/16")
print("duplicated norm/qkv/proj/attention + the full-output; gathers only x for K,V.")
