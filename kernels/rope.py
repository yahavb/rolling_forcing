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

from utils import _compile


@_compile
def causal_rope_rotation(
    x,
    cos_sin,
    head_start: int = 0,
    head_end: int = 12,
    head_dim: int = 128,
):
    seq_len = x.shape[0]
    N = head_end - head_start
    D = head_dim

    x_heads = x[:seq_len, head_start:head_end, :].float()

    cos_table = cos_sin[:seq_len, :D].unsqueeze(1).expand(seq_len, N, D)
    sin_table = cos_sin[:seq_len, D:].unsqueeze(1).expand(seq_len, N, D)

    x_cos = x_heads * cos_table

    x_paired = x_heads.view(seq_len, N, D // 2, 2)
    x_even = x_paired[:, :, :, 0]
    x_odd = x_paired[:, :, :, 1]

    sin_paired = sin_table.view(seq_len, N, D // 2, 2)
    sin_even = sin_paired[:, :, :, 0]
    sin_odd = sin_paired[:, :, :, 1]

    x_sin = torch.stack([x_odd * sin_even, x_even * sin_odd], dim=-1)
    x_sin = x_sin.view(seq_len, N, D)

    out = (x_cos + x_sin).to(x.dtype)
    return out


def build_rope_grids(
    freqs_cos,
    freqs_sin,
    sign_pattern,
    start_frame,
    F: int = 15,
    H: int = 30,
    W: int = 52,
    head_dim: int = 128,
):
    c = head_dim // 2
    D = head_dim
    s0 = c - 2 * (c // 3)
    s1 = c // 3

    sf = start_frame.view(-1)[0].item()

    frame_cos = freqs_cos[sf:sf + F, :s0]
    frame_sin = freqs_sin[sf:sf + F, :s0]

    h_cos = freqs_cos[:H, s0:s0 + s1]
    h_sin = freqs_sin[:H, s0:s0 + s1]

    w_cos = freqs_cos[:W, s0 + s1:s0 + 2 * s1]
    w_sin = freqs_sin[:W, s0 + s1:s0 + 2 * s1]

    cos_grid = torch.cat([
        frame_cos.view(F, 1, 1, s0).expand(F, H, W, s0),
        h_cos.view(1, H, 1, s1).expand(F, H, W, s1),
        w_cos.view(1, 1, W, s1).expand(F, H, W, s1),
    ], dim=-1)

    sin_grid = torch.cat([
        frame_sin.view(F, 1, 1, s0).expand(F, H, W, s0),
        h_sin.view(1, H, 1, s1).expand(F, H, W, s1),
        w_sin.view(1, 1, W, s1).expand(F, H, W, s1),
    ], dim=-1)

    sign = sign_pattern[:H, :D]

    cos_flat = cos_grid.unsqueeze(-1).expand(F, H, W, c, 2).reshape(F, H, W, D)

    sin_flat = sin_grid.unsqueeze(-1).expand(F, H, W, c, 2).reshape(F, H, W, D)
    sin_signed = sin_flat * sign.view(1, H, 1, D)

    combined = torch.cat([cos_flat, sin_signed], dim=-1)
    return combined.reshape(F * H, W * 2 * D)
