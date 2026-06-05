# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI serving endpoint for Rolling Forcing on Neuron.

Uses the same fast e2e_pipeline code path (proper collectives, no file-based
shard exchange). Rank 0 runs the HTTP server; all ranks participate
symmetrically in T5/DiT/VAE via broadcast coordination.

Usage:
    torchrun --nproc_per_node=8 serve.py

Endpoints:
    POST /generate/stream  — SSE streaming (frames delivered as blocks complete)
    POST /generate         — full video (returns base64 mp4 + frames)
    GET  /health
    GET  /readiness
"""

import asyncio
import base64
import json
import os
import sys
import time
from io import BytesIO
from typing import Optional

import torch
import torch.distributed as dist
from PIL import Image

from models.dit_pipeline import (
    build_dit_pipeline,
    destroy_parallel_groups,
    init_parallel_groups,
)
from models.t5 import (
    build_text_encoder,
    destroy_t5_parallel_group,
    encode_one_prompt,
    init_t5_parallel_group,
)
from models.vae import (
    build_vae,
    destroy_vae_parallel_group,
    init_vae_parallel_group,
)
from utils import w_shard
from utils.logging_utils import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

TP_DEGREE = int(os.environ.get("TP_DEGREE", "4"))
DEFAULT_NUM_FRAMES = int(os.environ.get("DEFAULT_NUM_FRAMES", "126"))
DEFAULT_FPS = int(os.environ.get("DEFAULT_FPS", "16"))
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/rolling_forcing_dmd.yaml")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/rolling_forcing_dmd.pt")
WARMUP_FRAMES = int(os.environ.get("WARMUP_FRAMES", "21"))

# Commands for rank coordination
CMD_STREAM = 2
CMD_GENERATE = 1
CMD_SHUTDOWN = 99

NEURON_DEVICE = torch.device("neuron")


# ─── Model loading ───────────────────────────────────────────────────────────

def load_models(rank, world):
    """Load T5, DiT, VAE on all ranks."""
    sp_degree = world // TP_DEGREE

    init_t5_parallel_group()
    init_parallel_groups(sp_degree, TP_DEGREE)
    init_vae_parallel_group()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    logger.info("Building T5 text encoder...")
    text_encoder = build_text_encoder(device="neuron")

    logger.info("Building DiT pipeline (TP=%d, SP=%d)...", TP_DEGREE, sp_degree)
    pipe = build_dit_pipeline(CONFIG_PATH, CHECKPOINT_PATH, TP_DEGREE, use_ema=True)

    logger.info("Building VAE decoder (width-sharded)...")
    vae = build_vae(dtype=torch.bfloat16)

    dist.barrier()
    if rank == 0:
        logger.info("All models loaded. Pipeline ready.")

    return text_encoder, pipe, vae


# ─── Streaming decode (same fast path as e2e_pipeline.py) ────────────────────

def stream_generate(pipe, vae, prompt_embeds, noise, rank, world):
    """Yield (pixel_frames_list, chunk_idx) per DiT block."""
    gen = pipe.inference_rolling_forcing_stream(noise, {"prompt_embeds": prompt_embeds})

    for chunk_idx, chunk in enumerate(gen):
        chunk_latent = w_shard(chunk, rank, world)
        chunk_device = vae.decode_to_pixel_device(
            chunk_latent, use_cache=True, chunk_idx=chunk_idx)
        chunk_video = vae.postprocess_pixels(chunk_device)
        yield chunk_video, chunk_idx


def full_generate(pipe, vae, prompt_embeds, noise, rank, world):
    """Full video generation, returns concatenated pixel tensor."""
    video_chunks = []
    vae.model.clear_cache()
    for chunk_video, _ in stream_generate(pipe, vae, prompt_embeds, noise, rank, world):
        video_chunks.append(chunk_video)
    return torch.cat(video_chunks, dim=1)


# ─── Pixel tensor to frames/video ───────────────────────────────────────────

def pixels_to_frames(video_tensor):
    """Convert [1, T, C, H, W_local] float tensor to list of PIL Images.

    For width-sharded VAE, we need to gather across ranks first.
    But in the e2e_pipeline path, each rank has its local shard.
    On rank 0, we gather all shards.
    """
    video = (video_tensor * 0.5 + 0.5).clamp(0, 1)
    video = video[0]  # [T, C, H, W]
    video = video.permute(0, 2, 3, 1)  # [T, H, W, C]
    video_np = (255.0 * video).to(torch.uint8).numpy()
    return [Image.fromarray(video_np[i]) for i in range(video_np.shape[0])]


def gather_width_shards(local_tensor, rank, world):
    """All-gather width-sharded pixel tensors to get full frame on rank 0."""
    if world == 1:
        return local_tensor

    # local_tensor: [1, T, C, H, W_local]
    gathered = [torch.empty_like(local_tensor) for _ in range(world)]
    dist.all_gather(gathered, local_tensor.contiguous())

    if rank == 0:
        return torch.cat(gathered, dim=-1)  # cat along W
    return None


# ─── Worker loop (ranks 1+) ─────────────────────────────────────────────────

def worker_loop(text_encoder, pipe, vae, rank, world):
    """Non-rank-0 workers: wait for commands, participate in collective ops."""
    logger.info("[Rank %d] Entering worker loop", rank)

    while True:
        cmd = torch.zeros(1, dtype=torch.long, device=NEURON_DEVICE)
        dist.broadcast(cmd, src=0)
        cmd_val = cmd.item()

        if cmd_val == CMD_SHUTDOWN:
            logger.info("[Rank %d] Shutdown.", rank)
            break

        if cmd_val in (CMD_STREAM, CMD_GENERATE):
            # Receive metadata: [num_frames, seed]
            meta = torch.zeros(2, dtype=torch.long, device=NEURON_DEVICE)
            dist.broadcast(meta, src=0)
            num_frames = meta[0].item()
            seed = meta[1].item()

            # Receive prompt
            prompt_len = torch.zeros(1, dtype=torch.long, device=NEURON_DEVICE)
            dist.broadcast(prompt_len, src=0)
            prompt_ids = torch.zeros(prompt_len.item(), dtype=torch.long, device=NEURON_DEVICE)
            dist.broadcast(prompt_ids, src=0)
            prompt_text = bytes(prompt_ids.cpu().tolist()).decode("utf-8")

            # T5 encode (all ranks participate)
            prompt_embeds = encode_one_prompt(text_encoder, prompt_text)

            # Generate noise (deterministic)
            torch.manual_seed(seed)
            noise = torch.randn(
                1, num_frames, 16, 60, 104, dtype=torch.bfloat16,
            ).to(NEURON_DEVICE)

            # Run generation (all ranks participate in DiT + VAE collectives)
            vae.model.clear_cache()
            if cmd_val == CMD_STREAM:
                for chunk_video, chunk_idx in stream_generate(
                    pipe, vae, prompt_embeds, noise, rank, world
                ):
                    # Gather width shards so rank 0 has full frames
                    gather_width_shards(chunk_video, rank, world)
            else:
                video = full_generate(pipe, vae, prompt_embeds, noise, rank, world)
                gather_width_shards(video, rank, world)


# ─── Server (rank 0) ────────────────────────────────────────────────────────

def run_server(text_encoder, pipe, vae, rank, world):
    """Rank 0: FastAPI server coordinating all ranks."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn

    app = FastAPI(title="Rolling Forcing Video Generation (e2e_pipeline)")

    class GenerateRequest(BaseModel):
        prompt: str
        num_frames: Optional[int] = Field(default=None, ge=3, le=481)
        seed: Optional[int] = Field(default=None)
        fps: Optional[int] = Field(default=None, ge=1, le=60)

    def broadcast_command(cmd_val, num_frames, seed, prompt):
        """Send command + metadata + prompt to all worker ranks."""
        cmd = torch.tensor([cmd_val], dtype=torch.long, device=NEURON_DEVICE)
        dist.broadcast(cmd, src=0)

        meta = torch.tensor([num_frames, seed], dtype=torch.long, device=NEURON_DEVICE)
        dist.broadcast(meta, src=0)

        prompt_bytes = prompt.encode("utf-8")
        prompt_len = torch.tensor([len(prompt_bytes)], dtype=torch.long, device=NEURON_DEVICE)
        dist.broadcast(prompt_len, src=0)
        prompt_ids = torch.tensor(list(prompt_bytes), dtype=torch.long, device=NEURON_DEVICE)
        dist.broadcast(prompt_ids, src=0)

    @app.post("/generate/stream")
    async def generate_stream(request: GenerateRequest):
        """Streaming generation: delivers frames via SSE as blocks complete."""
        num_frames = request.num_frames or DEFAULT_NUM_FRAMES
        seed = request.seed or 0

        async def event_stream():
            try:
                broadcast_command(CMD_STREAM, num_frames, seed, request.prompt)

                # T5 encode (rank 0 participates)
                prompt_embeds = encode_one_prompt(text_encoder, request.prompt)

                # Noise (same seed as workers)
                torch.manual_seed(seed)
                noise = torch.randn(
                    1, num_frames, 16, 60, 104, dtype=torch.bfloat16,
                ).to(NEURON_DEVICE)

                vae.model.clear_cache()
                frame_count = 0
                num_frame_per_block = pipe.num_frame_per_block
                # Estimate total pixel frames (4x temporal upsampling)
                total_pixel_frames = (num_frames - 1) * 4 + 1

                for chunk_video, chunk_idx in stream_generate(
                    pipe, vae, prompt_embeds, noise, rank, world
                ):
                    # Gather full-width frames on rank 0
                    full_video = gather_width_shards(chunk_video, rank, world)
                    frames = pixels_to_frames(full_video)

                    for frame in frames:
                        buf = BytesIO()
                        frame.save(buf, format="PNG")
                        frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                        data = {
                            "frame_index": frame_count,
                            "frame": frame_b64,
                            "total_frames": total_pixel_frames,
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        frame_count += 1
                        await asyncio.sleep(0)

                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                logger.error("Stream error: %s", e, exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/generate")
    async def generate_full(request: GenerateRequest):
        """Full generation: returns complete video as base64."""
        num_frames = request.num_frames or DEFAULT_NUM_FRAMES
        fps = request.fps or DEFAULT_FPS
        seed = request.seed or 0

        start_time = time.time()

        try:
            broadcast_command(CMD_GENERATE, num_frames, seed, request.prompt)

            prompt_embeds = encode_one_prompt(text_encoder, request.prompt)

            torch.manual_seed(seed)
            noise = torch.randn(
                1, num_frames, 16, 60, 104, dtype=torch.bfloat16,
            ).to(NEURON_DEVICE)

            vae.model.clear_cache()
            video = full_generate(pipe, vae, prompt_embeds, noise, rank, world)
            full_video = gather_width_shards(video, rank, world)
            frames = pixels_to_frames(full_video)

            # Encode frames
            frames_b64 = []
            for frame in frames:
                buf = BytesIO()
                frame.save(buf, format="PNG")
                frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

            # Encode video
            import tempfile
            import numpy as np
            from torchvision.io import write_video

            video_np = [np.array(f) for f in frames]
            video_tensor = torch.from_numpy(np.stack(video_np))
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                tmp_path = f.name
            write_video(tmp_path, video_tensor, fps=fps)
            with open(tmp_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")
            os.remove(tmp_path)

            return {
                "video": video_b64,
                "frames": frames_b64,
                "execution_time": time.time() - start_time,
                "num_frames": len(frames),
            }

        except Exception as e:
            logger.error("Generate error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/readiness")
    async def readiness():
        return {"status": "ready", "model_loaded": True, "tp_degree": TP_DEGREE}

    @app.get("/")
    async def root():
        return {
            "service": "Rolling Forcing Video Generation (e2e_pipeline)",
            "model": "Wan2.1-T2V-1.3B",
            "parallelism": f"TP={TP_DEGREE}, SP={world // TP_DEGREE}, VAE=width-shard-{world}",
            "endpoints": ["/generate", "/generate/stream", "/health", "/readiness"],
            "default_num_frames": DEFAULT_NUM_FRAMES,
            "default_fps": DEFAULT_FPS,
        }

    logger.info("Starting uvicorn on rank 0 (port 8000)...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault("NEURON_FALLBACK_ENABLED", "0")
    os.environ.setdefault("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "0")

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world % TP_DEGREE == 0

    text_encoder, pipe, vae = load_models(rank, world)

    # Warmup: trigger compilation with a short generation
    if WARMUP_FRAMES > 0:
        if rank == 0:
            logger.info("Warmup: generating %d frames to trigger compilation...", WARMUP_FRAMES)

        warmup_prompt = "A cat walking on the beach at sunset"

        if rank == 0:
            t0 = time.perf_counter()
        prompt_embeds = encode_one_prompt(text_encoder, warmup_prompt)
        if rank == 0:
            torch.neuron.synchronize()
            logger.info("  T5: %.1f ms", (time.perf_counter() - t0) * 1000)

        torch.manual_seed(42)
        noise = torch.randn(
            1, WARMUP_FRAMES, 16, 60, 104, dtype=torch.bfloat16,
        ).to(NEURON_DEVICE)

        vae.model.clear_cache()
        for chunk_video, chunk_idx in stream_generate(
            pipe, vae, prompt_embeds, noise, rank, world
        ):
            if rank == 0:
                torch.neuron.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000
                frames = chunk_video.shape[1] if chunk_video.dim() >= 2 else 0
                logger.info("  warmup block %2d: %d frames (%.1f s elapsed)",
                            chunk_idx, frames, elapsed / 1000)
                t0 = time.perf_counter()

        dist.barrier()
        if rank == 0:
            logger.info("Warmup complete — all kernels compiled.")

    # Rank 0 runs HTTP server; other ranks enter worker loop
    if rank == 0:
        run_server(text_encoder, pipe, vae, rank, world)
        # After server exits, signal workers to shutdown
        cmd = torch.tensor([CMD_SHUTDOWN], dtype=torch.long, device=NEURON_DEVICE)
        dist.broadcast(cmd, src=0)
    else:
        worker_loop(text_encoder, pipe, vae, rank, world)

    destroy_t5_parallel_group()
    destroy_parallel_groups()
    destroy_vae_parallel_group()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
