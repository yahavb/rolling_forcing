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

import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

import nki
from torch_neuronx import wrap_nki


@wrap_nki
@nki.jit
def restore_layout_rank_slice(gathered: nl.ndarray, rank: int, N: int = 2,
                              nfpb: int = 3, max_frames: int = 15, frame_seqlen: int = 1560):
    """Produce ONLY this rank's output slice full[rank*L_full_N:(rank+1)*L_full_N] directly
    from `gathered`, instead of building the whole deinterleaved `full` (N x bigger) and
    slicing. The old restore_layout DMA-copied all N ranks' data then the caller kept 1/N and
    discarded the rest — ~N x wasted DMA on a 93%-dma NEFF. This emits just the 2-6 contiguous
    dma sub-ranges that make up this rank's slice (verify: bit-identical, max|Δ|=0)."""
    L_full, dim = gathered.shape
    L_cu = nfpb * frame_seqlen
    L_dn = max_frames * frame_seqlen
    kernel_assert(L_full == L_cu + L_dn,
                  f"restore_layout_rank_slice expects L_full = {L_cu + L_dn}, got {L_full}")
    L_full_N = L_full // N
    L_cu_N = L_cu // N
    L_dn_N = L_dn // N

    out = nl.ndarray(shape=(L_full_N, dim), dtype=gathered.dtype, buffer=nl.shared_hbm)

    s = rank * L_full_N
    e = s + L_full_N
    r = s
    while r < e:
        if r < L_cu:                          # cu region: block w = r // L_cu_N
            w = r // L_cu_N
            off = r % L_cu_N
            n = min(L_cu_N - off, e - r, L_cu - r)
            src0 = w * L_full_N + off
        else:                                 # dn region
            rp = r - L_cu
            w = rp // L_dn_N
            off = rp % L_dn_N
            n = min(L_dn_N - off, e - r)
            src0 = w * L_full_N + L_cu_N + off
        nisa.dma_copy(dst=out[nl.ds(r - s, n), :], src=gathered[nl.ds(src0, n), :])
        r += n

    return out


@wrap_nki
@nki.jit
def restore_layout(gathered: nl.ndarray, N: int = 2, nfpb: int = 3, max_frames: int = 15, frame_seqlen: int = 1560):
    L_full, dim = gathered.shape
    L_cu = nfpb * frame_seqlen
    L_dn = max_frames * frame_seqlen

    kernel_assert(
        L_full == L_cu + L_dn,
        f"restore_layout expects L_full = {L_cu + L_dn}, got {L_full}",
    )

    L_full_N = L_full // N
    L_cu_N = L_cu // N
    L_dn_N = L_dn // N

    out = nl.ndarray(shape=(L_full, dim), dtype=gathered.dtype, buffer=nl.shared_hbm)

    for w in range(N):
        nisa.dma_copy(
            dst=out[nl.ds(w * L_cu_N, L_cu_N), :],
            src=gathered[nl.ds(w * L_full_N, L_cu_N), :],
        )
        nisa.dma_copy(
            dst=out[nl.ds(L_cu + w * L_dn_N, L_dn_N), :],
            src=gathered[nl.ds(w * L_full_N + L_cu_N, L_dn_N), :],
        )

    return out
