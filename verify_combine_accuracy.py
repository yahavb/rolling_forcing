# NUMERIC ACCURACY TEST of the EXACT stripped combine now in _attend_ring (commit 961457c).
# Mirrors the committed loop line-for-line, feeds kernel-accurate partials, and reports the
# measured max|Δ| vs true full attention. This is a RUN, not a prediction.
import torch
torch.manual_seed(0); torch.set_default_dtype(torch.float64)

def full_attn(Q,K,V,scale):
    S=torch.einsum('bdq,bdk->bqk',Q,K)*scale
    return torch.einsum('bqk,bkd->bqd',torch.softmax(S,-1),V).permute(1,0,2)  # [Sq,bs,d]

def kernel_partial(Q,Kseg,Vseg,scale):  # what wan_flash return_partials emits (proven bit-exact on device, run rx7dt)
    S=torch.einsum('bdq,bdk->bqk',Q,Kseg)*scale
    rmax=S.max(-1,keepdim=True).values
    p=torch.exp(S-rmax)
    O=torch.einsum('bqk,bkd->bqd',p,Vseg)
    return O.permute(1,0,2), rmax.permute(1,0,2), p.sum(-1,keepdim=True).permute(1,0,2)  # [Sq,bs,d],[Sq,bs,1],[Sq,bs,1]

def committed_combine(Q,K,V,scale,nseg):
    bs,d,Sk=K.shape; seg=Sk//nseg
    m=l=acc=None
    for s in range(nseg):
        O_s,max_s,sum_s=kernel_partial(Q,K[:,:,s*seg:(s+1)*seg],V[:,s*seg:(s+1)*seg,:],scale)
        # ↓↓↓ EXACT lines from committed _attend_ring (961457c) ↓↓↓
        if m is None:
            m,l,acc = max_s,sum_s,O_s
        else:
            m_new=torch.maximum(m,max_s)
            cp=torch.exp(m-m_new); cc=torch.exp(max_s-m_new)
            l=l*cp+sum_s*cc; acc=acc*cp+O_s*cc; m=m_new
    return acc/l

# realistic RF shapes: bs=3 heads/shard, d=128, cu window 3600 / sp=4 = 4 segs of 900
bs,d,Sq=3,128,128; scale=1/d**0.5
results={}
for nseg,Sk in [(1,3600),(2,3600),(4,3600),(4,14400)]:  # cu(3600) at NSEG 1/2/4, and dn(14400) NSEG4
    Q=torch.randn(bs,d,Sq); K=torch.randn(bs,d,Sk); V=torch.randn(bs,Sk,d)
    ref=full_attn(Q,K,V,scale); out=committed_combine(Q,K,V,scale,nseg)
    md=(out-ref).abs().max().item()
    results[(nseg,Sk)]=md
    print(f"nseg={nseg} window={Sk:5d}: max|Δ| = {md:.3e}  {'PASS' if md<1e-9 else 'FAIL'}")
worst=max(results.values())
print(f"\nWORST max|Δ| = {worst:.3e}  =>  {'ALL PASS (<1e-9)' if worst<1e-9 else 'FAIL'}")
