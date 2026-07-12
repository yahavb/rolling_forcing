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

"""In-SBUF all-gather for the TRUE CACHE-SHARD KV-window exchange.

Replaces the two torch.distributed ``all_gather_into_tensor`` barrier collectives
in ``CausalWanSelfAttention._attend_cache_shard`` (RF_RING_CACHESHARD path only)
with the reference SBUF-to-SBUF all-gather kernel
(``nkilib.experimental.collectives.sb2sb_allgather.allgather_sb2sb``): the gather
happens on-chip in SBUF instead of round-tripping through the runtime, so it is a
candidate for compute overlap and avoids the runtime barrier.

The kernel body is copied verbatim from the reference (same
``ncc.all_gather(..., collective_dim=1)`` contract, ``H <= 128`` partition limit,
gather along the last/free dim in rank order). ``gather_kv_world`` is a thin
torch-side adapter that reshapes RF's 3D K/V window-shards to the 2D ``[H, W]``
the kernel expects and reshapes the gathered ``[H, W * world]`` back into the
exact ``[N, bs, ...]`` tensors the torch path produced, so the proven
per-block-interleave reshape in ``_attend_cache_shard`` is untouched.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
import torch
from nki.collectives import ReplicaGroup
from torch_neuronx import wrap_nki

from kernels.nkilib_compat import kernel_assert


@wrap_nki
@nki.jit
def allgather_sb2sb(
    inp: nl.ndarray,
    replica_groups: ReplicaGroup,
    tp_degree: int,
) -> nl.ndarray:
    """SBUF-to-SBUF all-gather. Verbatim from
    nkilib.experimental.collectives.sb2sb_allgather.allgather_sb2sb.

    inp is [H, W] on HBM (H <= 128 partition, W local width per rank); returns
    [H, K = W * tp_degree] on shared_hbm, concatenated along the last dim in rank
    order: out = [data_0 | data_1 | ... | data_{tp_degree-1}].
    """
    H, W = inp.shape
    K = W * tp_degree
    dtype = inp.dtype

    kernel_assert(H <= 128, "H must be <= 128 to fit in SBUF partition")

    in_buf = nl.ndarray((H, W), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=in_buf, src=inp[0:H, 0:W])

    out_buf = nl.ndarray((H, K), dtype=dtype, buffer=nl.sbuf)
    out = nl.ndarray((H, K), dtype=dtype, buffer=nl.shared_hbm)

    ncc.all_gather(dsts=[out_buf], srcs=[in_buf], replica_group=replica_groups, collective_dim=1)

    nisa.dma_copy(dst=out[0:H, 0:K], src=out_buf)
    return out


# Cache ReplicaGroup objects by world size — they are compile-time-constant
# (static) args to the nki.jit kernel; reusing the same object keeps the traced
# graph stable across calls.
_REPLICA_GROUPS = {}


def _world_replica_group(world_size: int) -> ReplicaGroup:
    rg = _REPLICA_GROUPS.get(world_size)
    if rg is None:
        # The RF "world" group is all ranks -> global ranks 0..world_size-1.
        rg = ReplicaGroup([list(range(world_size))])
        _REPLICA_GROUPS[world_size] = rg
    return rg


def gather_kv_world(
    k_own: torch.Tensor,
    v_own: torch.Tensor,
    world_size: int,
):
    """SB2SB all-gather of this rank's K/V window-shard over the 'world' group.

    Layout / axis mapping (why d -> H and (bs, seqlen) -> W):
      - allgather_sb2sb gathers along the LAST dim of a 2D [H, W] tensor with
        H <= 128 (partition). RF's head_dim d == 128 exactly, so mapping d -> H
        sits precisely at the partition limit and needs NO tiling. Mapping the
        sequence length (k_len ~900-3600) into H instead would exceed 128 and
        force allgather_sb2sb_tiled.
      - Inputs (bs = heads/shard, d = 128, s = k_len_int):
            k_own [bs, d, s] ; v_own [bs, s, d]
      - K path: permute to [d, bs, s] then reshape to 2D [H=d, W=bs*s].
        gather -> [d, world*bs*s] laid out rank-major on the free dim:
            gathered[dd, r*(bs*s) + b*s + t] == rank_r.k_own[b, dd, t]
        view [d, N, bs, s] then permute(1,2,0,3) -> [N, bs, d, s], i.e.
            k_g[r, b, dd, t] == rank_r.k_own[b, dd, t]
        which is EXACTLY what ps.all_gather_into_tensor(k_g.view(N*bs,d,s), ...)
        produced (k_g[r] == rank r's k_own).
      - V path is symmetric: permute to [d, bs, s], gather, view [d, N, bs, s]
        then permute(1,2,3,0) -> [N, bs, s, d], matching v_g[r] == rank r's v_own.

    Returns (k_g, v_g) contiguous, in the identical shapes the torch
    all_gather_into_tensor path yielded:
        k_g [world, bs, d, s]   v_g [world, bs, s, d]
    so the downstream per-block-interleave reshape in _attend_cache_shard is
    unchanged.
    """
    N = world_size
    rg = _world_replica_group(N)

    bs, d, s = k_own.shape
    # [bs, d, s] -> [d, bs, s] -> 2D [H=d, W=bs*s]
    k2d = k_own.permute(1, 0, 2).reshape(d, bs * s).contiguous()
    # [bs, s, d] -> [d, bs, s] -> 2D [H=d, W=bs*s]
    v2d = v_own.permute(2, 0, 1).reshape(d, bs * s).contiguous()

    k_gathered = allgather_sb2sb(k2d, rg, N)   # [d, N*bs*s]
    v_gathered = allgather_sb2sb(v2d, rg, N)   # [d, N*bs*s]

    k_g = k_gathered.view(d, N, bs, s).permute(1, 2, 0, 3).contiguous()   # [N, bs, d, s]
    v_g = v_gathered.view(d, N, bs, s).permute(1, 2, 3, 0).contiguous()   # [N, bs, s, d]
    return k_g, v_g
