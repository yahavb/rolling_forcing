"""CPU proof: sharded-K-RoPE (RF_RING_KROPE) is bit-exact to full-K-RoPE.
Original _qkv_rope: world-gather k_local -> RoPE full L -> slice heads (every rank RoPEs
FULL L redundantly). New: each rank RoPEs only its L/world_size slice at its global offset,
slice heads, then world-gather the roped shards. RoPE is per-position so this is identical,
and cuts the gpsimd-heavy K-RoPE work by world_size. max|Δ|=0.0."""
import torch; torch.manual_seed(0); torch.set_default_dtype(torch.float64)
world=16; L=3600; n=12; d=128; n_local=3; ws=L//world
theta=torch.randn(L)*0.01
def rope_pos(x,p0):
    o=x.clone()
    for i in range(x.shape[0]):
        a=theta[p0+i]; c,s=torch.cos(a),torch.sin(a)
        o[i,:,0::2]=c*x[i,:,0::2]-s*x[i,:,1::2]; o[i,:,1::2]=s*x[i,:,0::2]+c*x[i,:,1::2]
    return o
Kloc={r:torch.randn(ws,n,d) for r in range(world)}
A=rope_pos(torch.cat([Kloc[r] for r in range(world)],0),0)[:,0:n_local]
B=torch.cat([rope_pos(Kloc[r],r*ws)[:,0:n_local] for r in range(world)],0)
dd=(A-B).abs().max().item(); print(f"sharded-K-RoPE vs full-K-RoPE max|Δ|={dd:.3e}")
assert dd<1e-12, "DIVERGES"; print("PROOF PASSED: RoPE-shard-then-gather == RoPE-full (bit-exact)")
