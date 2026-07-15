# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Neuron Science Team, Amazon Annapurna Labs
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import nki
import nki.language as nl

from torch_neuronx import wrap_nki


@nki.jit
def _causal_rope_rotation_nki(x, cos_sin, num_heads=12, head_dim=128):
    seq_len = x.shape[0]
    N = num_heads
    D = head_dim
    P = nl.tile_size.pmax

    assert seq_len % P == 0
    num_tiles = seq_len // P
    out = nl.ndarray((seq_len, N, D), dtype=x.dtype, buffer=nl.shared_hbm)

    for tile_i in nl.sequential_range(num_tiles):
        ts = tile_i * P
        cs_sb = nl.load(cos_sin[nl.ds(ts, P), :])
        cos_tile = cs_sb[:, nl.ds(0, D)]
        sin_tile = cs_sb[:, nl.ds(D, D)]
        x_sb = nl.load(x[nl.ds(ts, P), :, :])

        # Batch the even/odd swap over ALL heads in ONE strided op pair (was 12 per-head
        # strided writes/tile — the gpsimd hotspot). x_swap_all[:, :, 0::2] = x[:, :, 1::2]
        # etc. is bit-identical to the per-head swap (verified max|Δ|=0), but 12x fewer
        # strided ops. Keep multiply/add per-head where the [P,D] cos/sin broadcast is
        # known-good.
        x_swap_all = nl.ndarray((P, N, D), dtype=x_sb.dtype, buffer=nl.sbuf)
        x_swap_all[:, :, 0::2] = x_sb[:, :, 1::2]
        x_swap_all[:, :, 1::2] = x_sb[:, :, 0::2]

        # Win5: batch the multiply/add over ALL heads too (was 3 ops x N heads/tile = the
        # remaining 94%-gpsimd cost). cos/sin are [P,D]; N is a FREE axis, so reshape to
        # [P,1,D] and broadcast_to([P,N,D]) = a STRIDE-0 view that folds into the multiply at
        # no cost (no per-head loop, no data copy — per NKI free-dim broadcast rules). Then
        # ONE multiply/multiply/add over the full [P,N,D] tile. out = x*cos + swap(x)*sin.
        # Bit-identical to the per-head loop (verify max|Δ|=0): 3 ops/tile vs 3*N.
        # Copy cos/sin into OWN contiguous [P,1,D] tiles (cos_tile/sin_tile are slices of the
        # [P,2D] cs_sb; broadcasting a slice-of-a-larger-tile failed to allocate). Then
        # broadcast the size-1 middle axis to [P,N,D] (stride-0 view) and do ONE multiply/add.
        cos_1 = nl.ndarray((P, 1, D), dtype=x_sb.dtype, buffer=nl.sbuf)
        sin_1 = nl.ndarray((P, 1, D), dtype=x_sb.dtype, buffer=nl.sbuf)
        cos_1[:, 0, :] = cos_tile
        sin_1[:, 0, :] = sin_tile
        cos_b = nl.broadcast_to(cos_1, (P, N, D))
        sin_b = nl.broadcast_to(sin_1, (P, N, D))

        out_sb = nl.ndarray((P, N, D), dtype=x.dtype, buffer=nl.sbuf)
        x_cos = nl.multiply(x_sb, cos_b)
        x_sin = nl.multiply(x_swap_all, sin_b)
        out_sb[:, :, :] = nl.add(x_cos, x_sin)

        nl.store(out[nl.ds(ts, P), :, :], out_sb)

    return out


causal_rope_rotation = wrap_nki(_causal_rope_rotation_nki)


@nki.jit
def _causal_rope_rotation_qk_nki(q, k, cos_sin, num_heads=12, head_dim=128):
    """Fused q/k RoPE: rotate BOTH q and k against the SAME cos/sin grid in ONE launch.

    q and k in the merged path share the exact same position grid and head count, so we
    load the cos/sin tile + build the broadcast ONCE per tile and apply it to both — two
    separate HBM outputs, NO torch.cat on the inputs and NO strided split on the outputs
    (which is what made the Python-level q/k cat a net regression: it added copies/launches
    to save launches). This halves the RoPE launch count in forward_merged with zero extra
    copies, and additionally shares the per-tile cos/sin load+broadcast between q and k.
    Bit-identical to two separate _causal_rope_rotation_nki calls (same math per tensor).
    """
    seq_len = q.shape[0]
    N = num_heads
    D = head_dim
    P = nl.tile_size.pmax

    assert seq_len % P == 0
    assert k.shape[0] == seq_len
    num_tiles = seq_len // P
    out_q = nl.ndarray((seq_len, N, D), dtype=q.dtype, buffer=nl.shared_hbm)
    out_k = nl.ndarray((seq_len, N, D), dtype=k.dtype, buffer=nl.shared_hbm)

    for tile_i in nl.sequential_range(num_tiles):
        ts = tile_i * P
        cs_sb = nl.load(cos_sin[nl.ds(ts, P), :])
        cos_tile = cs_sb[:, nl.ds(0, D)]
        sin_tile = cs_sb[:, nl.ds(D, D)]

        # Shared cos/sin broadcast (built once, used for both q and k) — same [P,1,D] ->
        # [P,N,D] stride-0 view as the single-tensor kernel.
        cos_1 = nl.ndarray((P, 1, D), dtype=q.dtype, buffer=nl.sbuf)
        sin_1 = nl.ndarray((P, 1, D), dtype=q.dtype, buffer=nl.sbuf)
        cos_1[:, 0, :] = cos_tile
        sin_1[:, 0, :] = sin_tile
        cos_b = nl.broadcast_to(cos_1, (P, N, D))
        sin_b = nl.broadcast_to(sin_1, (P, N, D))

        # q and k unrolled explicitly (NKI cannot iterate a Python tuple of device
        # tensors — "expecting simple variable"). Both reuse cos_b/sin_b built above.
        q_sb = nl.load(q[nl.ds(ts, P), :, :])
        q_swap = nl.ndarray((P, N, D), dtype=q_sb.dtype, buffer=nl.sbuf)
        q_swap[:, :, 0::2] = q_sb[:, :, 1::2]
        q_swap[:, :, 1::2] = q_sb[:, :, 0::2]
        q_out_sb = nl.ndarray((P, N, D), dtype=q.dtype, buffer=nl.sbuf)
        q_out_sb[:, :, :] = nl.add(nl.multiply(q_sb, cos_b),
                                   nl.multiply(q_swap, sin_b))
        nl.store(out_q[nl.ds(ts, P), :, :], q_out_sb)

        k_sb = nl.load(k[nl.ds(ts, P), :, :])
        k_swap = nl.ndarray((P, N, D), dtype=k_sb.dtype, buffer=nl.sbuf)
        k_swap[:, :, 0::2] = k_sb[:, :, 1::2]
        k_swap[:, :, 1::2] = k_sb[:, :, 0::2]
        k_out_sb = nl.ndarray((P, N, D), dtype=k.dtype, buffer=nl.sbuf)
        k_out_sb[:, :, :] = nl.add(nl.multiply(k_sb, cos_b),
                                   nl.multiply(k_swap, sin_b))
        nl.store(out_k[nl.ds(ts, P), :, :], k_out_sb)

    return out_q, out_k


causal_rope_rotation_qk = wrap_nki(_causal_rope_rotation_qk_nki)


def build_rope_grids(freqs_cos, freqs_sin, sign_pattern, start_frame,
                     F=15, H=30, W=52, head_dim=128):
    """Build 3D RoPE cos/sin grids in PyTorch (no NKI compilation needed)."""
    d = head_dim
    c = d // 2
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    seq_len = F * H * W
    device = freqs_cos.device

    frame_idx = start_frame.flatten() + torch.arange(F, device=device)

    cos_half = torch.cat([
        torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(F, 1, 1, -1).expand(F, H, W, -1),
        freqs_cos[:H, s0:s0 + s1].view(1, H, 1, -1).expand(F, H, W, -1),
        freqs_cos[:W, s0 + s1:].view(1, 1, W, -1).expand(F, H, W, -1),
    ], dim=-1).reshape(seq_len, c)

    sin_half = torch.cat([
        torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(F, 1, 1, -1).expand(F, H, W, -1),
        freqs_sin[:H, s0:s0 + s1].view(1, H, 1, -1).expand(F, H, W, -1),
        freqs_sin[:W, s0 + s1:].view(1, 1, W, -1).expand(F, H, W, -1),
    ], dim=-1).reshape(seq_len, c)

    cos_expanded = cos_half.repeat_interleave(2, dim=-1)
    sin_expanded = sin_half.repeat_interleave(2, dim=-1)
    sign = torch.ones(d, device=device, dtype=sin_expanded.dtype)
    sign[0::2] = -1.0
    sin_signed = sin_expanded * sign.unsqueeze(0)
    cos_sin = torch.cat([cos_expanded, sin_signed], dim=-1).contiguous()

    P = 128
    pad = (P - seq_len % P) % P
    if pad > 0:
        cos_sin = torch.nn.functional.pad(cos_sin, (0, 0, 0, pad))
    return cos_sin
