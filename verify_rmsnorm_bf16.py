"""CPU proof: bf16-elementwise RMSNorm rel-error is within bf16 rounding (NOT bit-identical).

WanRMSNorm old: x.float() -> _norm entirely in fp32 -> downcast -> * weight (bf16). That is a
full [L,dim] fp32 elementwise pass = the gpsimd cost. New: fp32 ONLY for the reduction+rsqrt
(a per-row scalar), then x*scale*weight in bf16. Cuts the fp32 full-tensor pass.

This is NOT max|Δ|=0 (all prior wins were). It trades a bf16-rounding-level error for less
gpsimd work. This file bounds that error so we know what the on-device ACC gate is judging."""
import torch
torch.manual_seed(0)
eps = 1e-6
worst_rel = 0.0
for L, dim in [(4500, 128), (4500, 1536), (3600, 1536), (18000, 1536)]:
    x = torch.randn(L, dim, dtype=torch.bfloat16)
    w = (1 + 0.1 * torch.randn(dim)).to(torch.bfloat16)

    def old(x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * w

    def new(x, w):
        inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        return (x * inv.type_as(x)) * w

    a = old(x, w).float(); b = new(x, w).float()
    rel = (a - b).abs().max().item() / (a.abs().max().item() + 1e-9)
    worst_rel = max(worst_rel, rel)
    print(f"L={L} dim={dim}: max|Δ|={(a-b).abs().max():.3e}  rel={rel:.3e}")

print(f"\nworst rel error = {worst_rel:.3e}  (bf16 machine eps ~ 7.8e-3)")
assert worst_rel < 8e-3, "exceeds bf16 rounding — would risk quality"
print("WITHIN bf16 rounding. NOT bit-identical; on-device ACC/RING gate + frame eyeball decides.")
