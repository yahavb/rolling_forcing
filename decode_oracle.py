"""ORACLE DECODER — decode StreamDiffusionV2's dumped DiT latents with RF's PROVEN VAE.

Purpose: split DiT-correctness from VAE-correctness after 15 noise runs in streamv2v.
- If the resulting mp4 is SHARP -> streamv2v's DiT latents are GOOD; the bug is streamv2v's VAE.
- If the mp4 is NOISE -> streamv2v's DiT itself is producing garbage latents; VAE is a red herring.

Input: dit_latents.pt (torch.save from streamv2v inference_neuron.py oracle dump) holding
  {"latents": [1, num_frames, 16, H//8, W//8] bf16/float, "num_frames", "height", "width"}.

Run ON RF hardware (Neuron) from the rolling_forcing repo:
  python decode_oracle.py /path/to/dit_latents.pt /path/to/out.mp4

The latent layout MUST match RF's expectation: [1, num_frames, 16, LH, LW]. streamv2v's
generate_rolling_window returns exactly that, so no permute is needed here (RF's
decode_to_pixel permutes internally).
"""
import sys
import torch
import numpy as np
import imageio

from models.vae import build_vae


def main():
    lat_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "oracle_out.mp4"
    nfpb = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    blob = torch.load(lat_path, map_location="cpu")
    latents = blob["latents"] if isinstance(blob, dict) else blob
    num_frames = blob.get("num_frames", latents.shape[1]) if isinstance(blob, dict) else latents.shape[1]
    print(f"[oracle] loaded latents {tuple(latents.shape)}  num_frames={num_frames}")
    print(f"[oracle] latent stats mean={latents.float().mean():.4f} std={latents.float().std():.4f} "
          f"min={latents.float().min():.3f} max={latents.float().max():.3f}")

    vae = build_vae(dtype=torch.bfloat16)
    latents = latents.to(dtype=torch.bfloat16, device="neuron")

    # decode block-by-block with streaming cache, exactly like RF's stream_generate
    vae.model.clear_cache()
    frames = []
    chunk_idx = 0
    for b in range(0, num_frames, nfpb):
        blk = latents[:, b:b + nfpb].contiguous()
        px = vae.decode_to_pixel(blk, use_cache=True, chunk_idx=chunk_idx)  # [1, T, 3, H, W]
        frames.append(px.to("cpu").float())
        chunk_idx += 1
    video = torch.cat(frames, dim=1)[:, :num_frames]        # [1, num_frames, 3, H, W]
    video = (video * 0.5 + 0.5).clamp(0, 1)                 # to [0,1]

    arr = (video[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)  # [T,H,W,3]
    imageio.mimwrite(out_path, list(arr), fps=16, quality=8)
    print(f"[oracle] wrote {out_path}  ({arr.shape[0]} frames {arr.shape[1]}x{arr.shape[2]})")


if __name__ == "__main__":
    main()
