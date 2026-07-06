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

# Import the T5 submodules DIRECTLY — NOT utils.wan_wrapper, NOT the `wan` top-level
# package. `wan/__init__.py` eagerly imports image2video/text2video, which pull in
# torchvision (absent from the training image, and unneeded for text encoding).
# These two submodules are torchvision-free. We replicate WanTextEncoder's exact
# build + forward below so the embeds are byte-identical to in-loop T5.
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.t5 import umt5_xxl

_T5_CKPT = "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
_T5_TOKENIZER_DIR = "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/"


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

    # Bare CPU UMT5 encoder — no Neuron, no compile, no distributed. This mirrors
    # WanTextEncoder.__init__ EXACTLY (umt5_xxl encoder_only, fp32, CPU; load the
    # bf16-enc checkpoint; seq_len=512 whitespace tokenizer) so the embeds match.
    text_encoder = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).eval().requires_grad_(False)
    text_encoder.load_state_dict(
        torch.load(_T5_CKPT, map_location="cpu", weights_only=False))
    tokenizer = HuggingfaceTokenizer(
        name=_T5_TOKENIZER_DIR, seq_len=512, clean="whitespace")

    @torch.no_grad()
    def encode(text):
        # Replicates WanTextEncoder.forward: tokenize -> encode -> zero padding
        # past each sequence's real length. Returns [1, 512, 4096] bf16 on CPU.
        ids, mask = tokenizer([text], return_mask=True, add_special_tokens=True)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        return context.detach().to(torch.bfloat16).cpu().contiguous()

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
