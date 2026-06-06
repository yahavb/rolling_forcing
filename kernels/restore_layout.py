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
