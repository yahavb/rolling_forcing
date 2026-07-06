"""Precompute UMT5 prompt embeddings for DMD distillation — ONCE, on CPU, 1 process.

Removes T5 from the training job entirely: with these embeds cached, the trainer
never instantiates/moves the ~9.6GB UMT5 text encoder onto the Neuron device and
never runs the T5 forward in the training loop (freeing that HBM and dropping the
T5-broadcast collective that would otherwise interleave with the FSDP student's
process groups).

Reads the SAME config the trainer uses (OmegaConf.load(--config_path) merged over
configs/default_config.yaml, exactly like train.py), so data_path (the prompt
file) and negative_prompt come straight from the config — no separate captions
arg. Run it from cwd=training/ so that `configs/default_config.yaml` and a
`../configs/...` --config_path resolve.

Usage (from training/):
  python3 precompute_embeds.py \
    --config_path ../configs/rolling_forcing_dmd_t4.yaml \
    --out /tmp/embeds.pt

Output (torch.save):
  {
    "prompt_embeds": {prompt_str: tensor[1, 512, 4096]},   # one per data_path line
    "negative_prompt_embeds": tensor[1, 512, 4096],        # config.negative_prompt
  }
The trainer (PRECOMPUTED_EMBEDS=<out>) looks embeds up by prompt string; a prompt
missing from "prompt_embeds" fails loudly there.
"""
import argparse
import os

from omegaconf import OmegaConf
import torch

# WanTextEncoder loads UMT5 on CPU (device=torch.device('cpu')) and its forward
# returns {"prompt_embeds": context[B, 512, 4096]} — the exact dict the trainer
# consumes. Importing works because we run from cwd=training/.
from utils.wan_wrapper import WanTextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Same YAML passed to train.py; data_path + "
                             "negative_prompt are read from it.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output .pt path for the precomputed embeds dict.")
    args = parser.parse_args()

    # Merge exactly like train.py so data_path / negative_prompt match training.
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # Read the prompt file the trainer's TextDataset would read (one prompt/line).
    with open(config.data_path, encoding="utf-8") as f:
        prompts = [line.rstrip("\n") for line in f]
    prompts = [p for p in prompts if p.strip() != ""]
    print(f"precomputing embeds for {len(prompts)} prompts (CPU, 1 process)")

    # Bare CPU text encoder — no Neuron, no compile, no distributed.
    encoder = WanTextEncoder().eval().requires_grad_(False)

    @torch.no_grad()
    def encode(text):
        # WanTextEncoder.forward zeroes padding past seq_len and returns
        # {"prompt_embeds": [B, 512, 4096]}. B=1 here → [1, 512, 4096].
        out = encoder(text_prompts=[text])["prompt_embeds"]
        return out.detach().to(torch.bfloat16).cpu().contiguous()

    prompt_embeds = {}
    for i, pr in enumerate(prompts):
        prompt_embeds[pr] = encode(pr)
        print(f"  [{i + 1}/{len(prompts)}] {pr[:60]}")

    # CFG needs the negative-prompt (unconditional) embedding too.
    negative_prompt_embeds = encode(config.negative_prompt)
    print("  [uncond] encoded config.negative_prompt for CFG")

    payload = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, args.out)
    print(f"wrote {len(prompt_embeds)} prompt embeds (+1 negative) -> {args.out}")


if __name__ == "__main__":
    main()
