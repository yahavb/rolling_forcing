"""CPU (gloo/fp32) numerical-parity harness for the 1.3B causal STUDENT.

WHY: the distilled checkpoints don't converge and render blurry. Before blaming the
DMD recipe we must rule out a HARDWARE-accuracy bug — i.e. that the on-device Neuron
compute (NKI flash attn, RoPE, bf16 rounding, FSDP gather) diverges from correct math.

This runs ONE student rollout (noise -> x0) on a chosen device, with a FIXED seed and
FIXED inputs, and writes x0 + per-block diagnostics to a .pt. Run it twice:

    # on the trn3 pod (on-device, bf16):
    DEVICE=neuron DTYPE=bf16 python3 validate_student_cpu.py \
        --config ../configs/rolling_forcing_dmd_t4.yaml \
        --ckpt /var/mdl/rolling_forcing/distill/<TS>/model.iter200.pt \
        --embeds /tmp/embeds.pt --out /tmp/parity.neuron.pt

    # on ANY box with the repo + weights (CPU reference, fp32 — SLOW but exact):
    DEVICE=cpu DTYPE=fp32 python3 validate_student_cpu.py \
        --config ../configs/rolling_forcing_dmd_t4.yaml \
        --ckpt /var/mdl/rolling_forcing/distill/<TS>/model.iter200.pt \
        --embeds /tmp/embeds.pt --out /tmp/parity.cpu.pt

    # then diff the two:
    python3 validate_student_cpu.py --diff /tmp/parity.cpu.pt /tmp/parity.neuron.pt

DESIGN
- Single process, NO torch.distributed, NO FSDP. Loads the FULL (already-gathered)
  `generator` state dict from the checkpoint straight into one WanDiffusionWrapper.
- The compute path is IDENTICAL to training: wan/modules/attention.py:attention()
  branches on device.type=="neuron" (NKI) vs else (F.scaled_dot_product_attention),
  so `cpu` is a faithful reference for the on-device kernels.
- Scope = the 1.3B student ONLY. The 14B teacher needs its TP group and does not fit a
  single core; the student's own noise->x0 forward is what produces the blurry frames,
  so it is the right thing to validate first.
- Use --ema to load `generator_ema` (the weights you actually SHIP) instead of the raw
  `generator` snapshot.
"""
import argparse
import os
import sys

import torch


def _resolve_device(name):
    if name == "neuron":
        # importing torch_neuronx registers the "neuron" device / eager backend
        import torch_neuronx  # noqa: F401
        return torch.device("neuron")
    return torch.device(name)


def _load_generator_sd(ckpt_path, use_ema):
    payload = torch.load(ckpt_path, map_location="cpu")
    key = "generator_ema" if use_ema else "generator"
    if key not in payload:
        avail = [k for k in payload.keys() if isinstance(payload[k], dict)]
        raise SystemExit(
            f"'{key}' not in checkpoint {ckpt_path}. dict keys present: {avail}. "
            f"(use/omit --ema to pick generator_ema vs generator)")
    sd = payload[key]
    print(f"[load] {key}: {len(sd)} tensors, distill_iter={payload.get('distill_iter')}")
    return sd


def run(args):
    # run from training/ so `configs/`, `utils`, `wan`, `pipeline` resolve like the trainer
    from omegaconf import OmegaConf
    from utils.misc import set_seed
    from utils.wan_wrapper import WanDiffusionWrapper
    from pipeline import RollingForcingTrainingPipeline

    device = _resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
             "fp16": torch.float16}[args.dtype]
    print(f"[cfg] device={device} dtype={dtype} seed={args.seed} ema={args.ema}")

    base = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(base, OmegaConf.load(args.config))

    # deterministic inputs
    set_seed(args.seed)

    # ── build the student EXACTLY as the trainer does (is_causal=True) ──
    gen = WanDiffusionWrapper(**OmegaConf.to_container(
        getattr(config, "model_kwargs", {}), resolve=True), is_causal=True)
    sd = _load_generator_sd(args.ckpt, args.ema)
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load] WARN missing={len(missing)} unexpected={len(unexpected)} "
              f"(first missing: {missing[:3]}, first unexpected: {unexpected[:3]})")
    gen = gen.to(device=device, dtype=dtype)
    gen.eval()

    scheduler = gen.get_scheduler()
    scheduler.timesteps = scheduler.timesteps.to(device)
    if getattr(scheduler, "alphas_cumprod", None) is not None:
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # ── prompt embeds (same precomputed file the trainer uses) ──
    payload = torch.load(args.embeds, map_location="cpu")
    prompts = list(payload["prompt_embeds"].keys())
    prompt = args.prompt or prompts[0]
    emb = payload["prompt_embeds"][prompt].to(device=device, dtype=dtype)
    print(f"[emb] using prompt: {prompt[:60]}...  shape={tuple(emb.shape)}")
    cond = {"prompt_embeds": emb}

    # ── denoising_step_list -> warped, same as trainer ──
    dsl = torch.tensor(config.denoising_step_list, dtype=torch.long)
    if getattr(config, "warp_denoising_step", False):
        ts = torch.cat((scheduler.timesteps.cpu(),
                        torch.tensor([0], dtype=torch.float32)))
        dsl = ts[1000 - dsl]

    pipeline = RollingForcingTrainingPipeline(
        denoising_step_list=dsl, scheduler=scheduler, generator=gen,
        num_frame_per_block=getattr(config, "num_frame_per_block", 3),
        same_step_across_blocks=getattr(config, "same_step_across_blocks", True),
        last_step_only=getattr(config, "last_step_only", False),
    )

    # ── fixed noise, shape from config (deterministic per seed) ──
    _b, _f, _c, _h, _w = config.image_or_video_shape
    n_frames = int(config.num_training_frames)
    g = torch.Generator().manual_seed(args.seed)
    noise = torch.randn((1, n_frames, _c, _h, _w), generator=g,
                        dtype=torch.float32).to(device=device, dtype=dtype)
    print(f"[roll] noise {tuple(noise.shape)}  frames={n_frames}  steps={list(dsl)}")

    with torch.no_grad():
        x0 = pipeline.inference_with_rolling_forcing(noise=noise, **cond)

    x0c = x0.detach().to("cpu", torch.float32)
    stats = {
        "x0": x0c,
        "x0_mean": x0c.mean().item(),
        "x0_std": x0c.std().item(),
        "x0_min": x0c.min().item(),
        "x0_max": x0c.max().item(),
        "shape": tuple(x0c.shape),
        "device": str(device), "dtype": str(dtype),
        "seed": args.seed, "prompt": prompt, "ema": args.ema,
        "ckpt": args.ckpt, "steps": [int(x) for x in dsl.tolist()],
    }
    torch.save(stats, args.out)
    print(f"[done] wrote {args.out}")
    print(f"       x0 mean={stats['x0_mean']:.5f} std={stats['x0_std']:.5f} "
          f"min={stats['x0_min']:.4f} max={stats['x0_max']:.4f}")
    if not torch.isfinite(x0c).all():
        n_nan = torch.isnan(x0c).sum().item()
        n_inf = torch.isinf(x0c).sum().item()
        print(f"       !!! NON-FINITE x0: nan={n_nan} inf={n_inf} "
              f"(a HARDWARE/compute bug produces this; recipe divergence usually doesn't)")


def diff(a_path, b_path):
    a = torch.load(a_path, map_location="cpu")
    b = torch.load(b_path, map_location="cpu")
    xa, xb = a["x0"].float(), b["x0"].float()
    if xa.shape != xb.shape:
        raise SystemExit(f"shape mismatch {xa.shape} vs {xb.shape}")
    d = (xa - xb).abs()
    denom = xa.abs().clamp_min(1e-6)
    rel = (d / denom)
    print(f"A = {a_path}  ({a['device']}/{a['dtype']})")
    print(f"B = {b_path}  ({b['device']}/{b['dtype']})")
    print(f"  shape           : {tuple(xa.shape)}")
    print(f"  A mean/std      : {xa.mean():.5f} / {xa.std():.5f}")
    print(f"  B mean/std      : {xb.mean():.5f} / {xb.std():.5f}")
    print(f"  max |A-B|       : {d.max().item():.6f}")
    print(f"  mean |A-B|      : {d.mean().item():.6f}")
    print(f"  max rel err     : {rel.max().item():.4f}")
    print(f"  mean rel err    : {rel.mean().item():.6f}")
    cos = torch.nn.functional.cosine_similarity(
        xa.flatten().unsqueeze(0), xb.flatten().unsqueeze(0)).item()
    print(f"  cosine sim      : {cos:.6f}")
    # bf16 has ~3 decimal digits; a correct kernel typically lands mean-rel < ~2e-2 and
    # cosine > ~0.999. Much worse => the on-device compute is the accuracy culprit.
    verdict = ("LIKELY OK (bf16-rounding-level diff)" if cos > 0.999 and rel.mean() < 2e-2
               else "SUSPECT — on-device compute diverges beyond bf16 rounding")
    print(f"  VERDICT         : {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../configs/rolling_forcing_dmd_t4.yaml")
    ap.add_argument("--ckpt")
    ap.add_argument("--embeds", default=os.environ.get("PRECOMPUTED_EMBEDS", "/tmp/embeds.pt"))
    ap.add_argument("--out", default="/tmp/parity.pt")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cpu"),
                    choices=["cpu", "neuron"])
    ap.add_argument("--dtype", default=os.environ.get("DTYPE", "fp32"),
                    choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--prompt", default=None, help="prompt text; default = first in embeds")
    ap.add_argument("--ema", action="store_true", help="load generator_ema not generator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diff", nargs=2, metavar=("A.pt", "B.pt"),
                    help="compare two output .pt files and exit")
    args = ap.parse_args()

    if args.diff:
        diff(args.diff[0], args.diff[1])
        return
    if not args.ckpt:
        raise SystemExit("--ckpt is required (unless --diff)")
    run(args)


if __name__ == "__main__":
    main()
