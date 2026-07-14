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

import math

import torch
import torch.nn as nn

from kernels.kv_cache_copy import cache_copy, kv_cache_copy
from kernels.restore_layout import restore_layout
from kernels.rope import causal_rope_rotation, build_rope_grids
from kernels.self_attention_nst import wan_flash_self_attn, wan_flash_self_attn_gather_kv
from utils import _compile
from utils import parallel_state as ps

from models.dit_layers import (
    ATTN_SEQLEN_MULTIPLE,
    WanLayerNorm,
    WanRMSNorm,
    WanT2VCrossAttention,
)


def expand_e_shard(e, start_frame, end_frame, start_off, shard_len, frame_seqlen):
    B, _, I, C = e.shape
    F_sub = end_frame - start_frame
    e_sub = e[:, start_frame:end_frame]
    e_t = e_sub.transpose(1, 2)
    e_exp = e_t.unsqueeze(3).expand(B, I, F_sub, frame_seqlen, C).reshape(
        B, I, F_sub * frame_seqlen, C)
    return e_exp[:, :, start_off:start_off + shard_len]


def modulated_norm_scale_shard(norm_x, mod_slice, e_slice, ones):
    return norm_x * (ones + (mod_slice + e_slice))


def modulated_norm_shift_shard(y, mod_slice, e_slice):
    return y + (mod_slice + e_slice)


def modulated_residual_shard(x, y, mod_slice, e_slice):
    return x + y * (mod_slice + e_slice)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=1,
                 qk_norm=True,
                 eps=1e-6,
                 layer_idx=0,
                 frame_length=1560):
        assert dim % num_heads == 0
        assert qk_norm, "qk_norm must be True"
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.eps = eps
        self.frame_length = frame_length
        self.max_attention_size = 21 * self.frame_length
        self.block_length = 3 * self.frame_length
        self.kv_cache_logical_size = 24 * self.frame_length
        self.layer_idx = layer_idx

        # TRUE CACHE-SHARDING (RF_RING_CACHESHARD): each rank stores ONLY its 1/world token
        # slice of the persistent cache (16x memory). The bookkeeping divides cleanly by
        # world_size — each rank's stream is structurally identical to the full stream at
        # 1/world scale (proven: verify_cache_shard.py max|Δ|=0). Sharded sizes used by
        # _cache_write / _assemble_kv when the flag is on.
        import os as _os_cs
        self._cache_shard = (_os_cs.environ.get("RF_RING_CACHESHARD", "0") == "1")
        # sharded (per-rank, 1/world) position sizes — set after world_size is known below.
        self._cs_block_length = None
        self._cs_max_attention_size = None
        self._cs_kv_cache_logical_size = None

        tp_degree = ps.get_world_size("attn-tp")
        sp_degree = ps.get_world_size("attn-sp")
        assert num_heads % tp_degree == 0, (
            f"num_heads ({num_heads}) must be divisible by tp_degree ({tp_degree})")
        self.sp_degree = sp_degree
        self.tp_degree = tp_degree
        self.world_size = sp_degree * tp_degree
        self.heads_per_shard = num_heads // tp_degree
        self.sp_rank = ps.get_rank("attn-sp")
        self.tp_rank = ps.get_rank("attn-tp")

        if self._cache_shard and self.world_size > 1:
            assert self.block_length % self.world_size == 0, f"block_length {self.block_length} % world {self.world_size}"
            assert self.max_attention_size % self.world_size == 0, f"max_attn {self.max_attention_size} % world {self.world_size}"
            assert self.kv_cache_logical_size % self.world_size == 0, f"kv_logical {self.kv_cache_logical_size} % world {self.world_size}"
            self._cs_block_length = self.block_length // self.world_size
            self._cs_max_attention_size = self.max_attention_size // self.world_size
            self._cs_kv_cache_logical_size = self.kv_cache_logical_size // self.world_size

        self.q = _compile(nn.Linear(dim, dim))
        self.k = _compile(nn.Linear(dim, dim))
        self.v = _compile(nn.Linear(dim, dim))
        if tp_degree > 1:
            self.o = _compile(nn.Linear(dim // tp_degree, dim))
        else:
            self.o = _compile(nn.Linear(dim, dim))
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

        sign_pattern = torch.ones(self.head_dim, dtype=torch.float32)
        sign_pattern[0::2] = -1.0
        self.register_buffer(
            'sign_pattern',
            sign_pattern.unsqueeze(0).expand(128, -1).contiguous(),
            persistent=False)

        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    @staticmethod
    def shard_state_dict(full_sd, dim, num_heads):
        tp_degree = ps.get_world_size("attn-tp")
        tp_rank = ps.get_rank("attn-tp")
        sd = {}
        head_dim = dim // num_heads
        heads_per_shard = num_heads // tp_degree
        shard_dim = heads_per_shard * head_dim

        for key, val in full_sd.items():
            if key == "o.weight":
                sd[key] = val[:, tp_rank * shard_dim:(tp_rank + 1) * shard_dim].clone()
            elif key == "o.bias":
                sd[key] = val.clone() if tp_rank == 0 else torch.zeros_like(val)
            else:
                sd[key] = val.clone()
        return sd

    def _local_qkv_norm(self, x):
        q = self.norm_q(self.q(x))[0]
        k = self.norm_k(self.k(x))[0]
        v = self.v(x)[0]
        return q, k, v

    def _gather_qkv(self, q_local, k_local, v_local, L, gather_q=True):
        def _gather(t):
            if self.world_size == 1:
                return t
            out = torch.empty(L, self.dim, dtype=t.dtype, device=t.device)
            ps.all_gather_into_tensor(out, t, "world")
            return out

        # CP: q may be gathered separately over attn-tp (caller does it); skip here.
        q_out = _gather(q_local) if gather_q else None
        return q_out, _gather(k_local), _gather(v_local)

    def _slice_heads(self, t):
        return self._slice_heads_2d(t).unsqueeze(0)

    def _slice_heads_2d(self, t):
        d = self.head_dim
        n = self.num_heads
        n_local = self.heads_per_shard
        h_start = self.tp_rank * n_local
        h_end = h_start + n_local
        L = t.shape[0]
        return t.view(L, n, d)[:, h_start:h_end]

    def _will_anchor_write(self, kv_cache, cache_start):
        # TRUE CACHE-SHARD: every position quantity scales by 1/world uniformly, so the
        # bookkeeping is structurally identical at 1/world scale (verify_cache_shard.py).
        block_length = self._cs_block_length if self._cache_shard else self.block_length
        kv_cache_size = (self._cs_kv_cache_logical_size if self._cache_shard
                         else self.kv_cache_logical_size)
        # cache_start arrives in FULL sequence coords (advances by full block/phase); convert
        # to sharded coords so cache_end / global_end_index / eviction all live at 1/world
        # scale consistently (verify_cache_start_shardcoords.py: sharded == full // world).
        if self._cache_shard and self.world_size > 1:
            cache_start = cache_start // self.world_size
        cache_end = cache_start + block_length
        global_end_index = kv_cache["global_end_index"]
        local_end_index_current = kv_cache["local_end_index"]
        num_new_tokens = max(cache_end - global_end_index, 0)
        if num_new_tokens > 0 and num_new_tokens + local_end_index_current > kv_cache_size:
            num_evicted = num_new_tokens + local_end_index_current - kv_cache_size
        else:
            num_evicted = 0
        local_end_index = local_end_index_current + num_new_tokens - num_evicted
        return local_end_index == block_length

    def _cache_copy_inplace(self, k_dst, k_src, v_dst=None, v_src=None):
        assert k_src.shape == k_dst.shape and k_src.numel() > 0, (
            f"_cache_copy_inplace K shape mismatch: src{tuple(k_src.shape)} != dst{tuple(k_dst.shape)}")
        assert v_dst is None or v_src.shape == v_dst.shape and v_src.numel() > 0, (
            f"_cache_copy_inplace V shape mismatch: src{tuple(v_src.shape) if v_src is not None else None} "
            f"!= dst{tuple(v_dst.shape) if v_dst is not None else None}")
        """Device-dispatched cache copy: copy_ on CPU, NKI kernel on Neuron."""
        if v_dst is not None:
            kv_cache_copy(k_dst, k_src, v_dst, v_src)
        else:
            cache_copy(k_dst, k_src)

    def _nki_rope_apply(self, x, grid_sizes, freqs_cos, freqs_sin, start_frame,
                        rope_grid_cache=None, start_frame_int=None,
                        head_start=None, head_end=None,
                        tok_offset=0, tok_len=None):
        # CP: tok_offset/tok_len RoPE only a contiguous token sub-range using the
        # position-matched grid slice combined[tok_offset:tok_offset+tok_len].
        # RoPE is per-position, so RoPE(q)[slice] == RoPE(grid[slice], q[slice]) —
        # bit-identical to rope-full-then-slice, but on 1/sp the tokens.
        assert (head_start is None) == (head_end is None), (
            "head_start and head_end must be provided together")

        b, s, n, d = x.shape
        f, h, w = grid_sizes
        seq_len = f * h * w
        if tok_len is None:
            tok_len = seq_len
            assert seq_len == s
        else:
            assert s == tok_len, f"x len {s} != tok_len {tok_len}"
        if head_start is None:
            head_start = 0
            head_end = n

        cache_key = None
        combined = None
        if rope_grid_cache is not None:
            assert start_frame_int is not None, (
                "start_frame_int must be provided with rope_grid_cache")
            cache_key = (grid_sizes, int(start_frame_int))
            combined = rope_grid_cache.get(cache_key)

        if combined is None:
            sf = start_frame.to(torch.int32).reshape(1, 1)
            combined = build_rope_grids(
                freqs_cos, freqs_sin, self.sign_pattern, sf,
                F=f, H=h, W=w, head_dim=d)[:seq_len]
            if cache_key is not None:
                rope_grid_cache[cache_key] = combined

        # slice the grid to the requested token sub-range (CP query shard)
        combined = combined[tok_offset:tok_offset + tok_len]

        n_local = head_end - head_start
        P = 128
        pad = (P - tok_len % P) % P
        x_local = x[0, :tok_len, head_start:head_end, :]
        x_padded = torch.nn.functional.pad(x_local, (0, 0, 0, 0, 0, pad))
        combined_padded = torch.nn.functional.pad(combined, (0, 0, 0, pad))

        out = causal_rope_rotation(
            x_padded, combined_padded,
            num_heads=n_local, head_dim=d)

        return out[:tok_len].unsqueeze(0)

    def _qkv_rope(self, x, grid_sizes, freqs_cos, freqs_sin, current_start,
                  rope_grid_cache=None, kv_cache=None):
        f, h, w = grid_sizes
        frame_seqlen = h * w
        L = f * h * w

        assert L % self.world_size == 0, (
            f"L ({L}) must be divisible by world_size ({self.world_size})")

        q_local, k_local, v_local = self._local_qkv_norm(x)

        n = self.num_heads
        d = self.head_dim
        n_local = self.heads_per_shard
        h_start = self.tp_rank * n_local
        h_end = h_start + n_local

        sp_shard_len = L // self.sp_degree
        sp_start = self.sp_rank * sp_shard_len

        start_frame_int = current_start // frame_seqlen
        start_frame_t = torch.tensor(start_frame_int, device=x.device)

        # CP query path: gather q ONLY over the attn-tp group -> this rank's sp-shard
        # [sp_start : sp_start+sp_shard_len] directly (proven == world-gather-then-
        # slice, bit-identical), instead of all-gathering the full L over world and
        # discarding (sp-1)/sp of it. RoPE only this shard's positions. k/v still
        # gather full L over world (attention reads the whole KV window).
        # Validated at CP2/CP4/CP8 (max|Δ|=0.0 vs the old full-gather path).
        if self.world_size > 1 and self._cache_shard:
            # TRUE CACHE-SHARD (RF_RING_CACHESHARD): produce k/v/roped_key as THIS RANK's
            # 1/world token slice ONLY — do NOT world-gather them (that is the whole point:
            # the persistent cache stores 1/world of the sequence per rank; the full window
            # is reassembled at attention time in _attend). x is already world-sharded, so
            # k_local/v_local ARE this rank's slice. RoPE is per-position, so RoPE the local
            # K slice at its GLOBAL offset world_rank*world_shard (proven bit-exact vs
            # rope-full-then-slice — verify_krope_shard.py). roped_query stays the sp-shard
            # (same as the default world>1 path). Heads are TP-sliced exactly as elsewhere.
            world_rank = ps.get_rank("world")
            world_shard = L // self.world_size
            q_tp = torch.empty(sp_shard_len, self.dim, dtype=q_local.dtype, device=q_local.device)
            ps.all_gather_into_tensor(q_tp, q_local, "attn-tp")
            q_tp_4d = q_tp.view(sp_shard_len, n, d).unsqueeze(0)
            roped_query = self._nki_rope_apply(
                q_tp_4d, grid_sizes, freqs_cos, freqs_sin,
                start_frame=start_frame_t, rope_grid_cache=rope_grid_cache,
                start_frame_int=start_frame_int,
                head_start=h_start, head_end=h_end,
                tok_offset=sp_start, tok_len=sp_shard_len)
            v = self._slice_heads(v_local)  # [1, world_shard, n_local, d]
            k_shard_4d = k_local.view(world_shard, n, d).unsqueeze(0)
            roped_key = self._nki_rope_apply(
                k_shard_4d, grid_sizes, freqs_cos, freqs_sin,
                start_frame=start_frame_t, rope_grid_cache=rope_grid_cache,
                start_frame_int=start_frame_int, head_start=h_start, head_end=h_end,
                tok_offset=world_rank * world_shard, tok_len=world_shard)  # [1, world_shard, n_local, d]
            if kv_cache is None or self._will_anchor_write(kv_cache, current_start):
                k = self._slice_heads(k_local)  # unroped, this rank's world-shard, n_local heads
            else:
                k = None
            return roped_query, k, v, roped_key
        import os as _os_kr
        _krope_shard = (_os_kr.environ.get("RF_RING_KROPE", "0") == "1")
        if self.world_size > 1 and _krope_shard:
            # SHARDED-K-RoPE (sharding-enabled win): RoPE is per-position, so RoPE only THIS
            # rank's L/world_size K slice at its global offset, THEN world-gather the roped
            # shards (all_gather concatenates in rank order == the full roped K). Proven
            # bit-exact (RoPE-shard-then-gather == RoPE-full). Cuts the gpsimd-heavy K-RoPE
            # work by world_size (16x): today every rank RoPEs the FULL L redundantly.
            q_tp = torch.empty(sp_shard_len, self.dim, dtype=q_local.dtype, device=q_local.device)
            ps.all_gather_into_tensor(q_tp, q_local, "attn-tp")
            q_tp_4d = q_tp.view(sp_shard_len, n, d).unsqueeze(0)
            roped_query = self._nki_rope_apply(
                q_tp_4d, grid_sizes, freqs_cos, freqs_sin,
                start_frame=start_frame_t, rope_grid_cache=rope_grid_cache,
                start_frame_int=start_frame_int,
                head_start=h_start, head_end=h_end,
                tok_offset=sp_start, tok_len=sp_shard_len)
            # SINGLE k/v world-gather (was 3 redundant gathers — ntrace: sharding cost is
            # LAUNCHES not compute, so collapse them). _gather_qkv(gather_q=False) gathers
            # BOTH k and v in one call; reuse for v AND the anchor-k path.
            _, k_full, v_full = self._gather_qkv(q_local, k_local, v_local, L, gather_q=False)
            v = self._slice_heads(v_full)
            # k: RoPE this rank's world-shard at its global offset, gather the ROPED shards.
            world_rank = ps.get_rank("world")
            world_shard = L // self.world_size
            k_shard_4d = k_local.view(world_shard, n, d).unsqueeze(0)
            rk_shard = self._nki_rope_apply(
                k_shard_4d, grid_sizes, freqs_cos, freqs_sin,
                start_frame=start_frame_t, rope_grid_cache=rope_grid_cache,
                start_frame_int=start_frame_int, head_start=h_start, head_end=h_end,
                tok_offset=world_rank * world_shard, tok_len=world_shard)  # [1, world_shard, n_local, d]
            # gather the per-rank roped K shards (heads already TP-sliced) over world.
            n_local = self.heads_per_shard
            rk_flat = rk_shard[0].reshape(world_shard, n_local * d)
            rk_gathered = torch.empty(L, n_local * d, dtype=rk_flat.dtype, device=rk_flat.device)
            ps.all_gather_into_tensor(rk_gathered, rk_flat, "world")
            roped_key = rk_gathered.view(L, n_local, d).unsqueeze(0)
            # k (unroped, head-sliced) for anchor write — reuse the k_full already gathered.
            if kv_cache is None or self._will_anchor_write(kv_cache, current_start):
                k = self._slice_heads(k_full)
            else:
                k = None
            return roped_query, k, v, roped_key
        if self.world_size > 1:
            q_tp = torch.empty(sp_shard_len, self.dim, dtype=q_local.dtype, device=q_local.device)
            ps.all_gather_into_tensor(q_tp, q_local, "attn-tp")
            _, k_full, v_full = self._gather_qkv(q_local, k_local, v_local, L, gather_q=False)
            k_full_4d = k_full.view(L, n, d).unsqueeze(0)
            v = self._slice_heads(v_full)
            q_tp_4d = q_tp.view(sp_shard_len, n, d).unsqueeze(0)
            roped_query = self._nki_rope_apply(
                q_tp_4d, grid_sizes, freqs_cos, freqs_sin,
                start_frame=start_frame_t, rope_grid_cache=rope_grid_cache,
                start_frame_int=start_frame_int,
                head_start=h_start, head_end=h_end,
                tok_offset=sp_start, tok_len=sp_shard_len)
        else:
            q_full, k_full, v_full = self._gather_qkv(q_local, k_local, v_local, L)
            k_full_4d = k_full.view(L, n, d).unsqueeze(0)
            v = self._slice_heads(v_full)
            q_full_4d = q_full.view(L, n, d).unsqueeze(0)
            roped_q_full = self._nki_rope_apply(
                q_full_4d, grid_sizes, freqs_cos, freqs_sin,
                start_frame=start_frame_t, rope_grid_cache=rope_grid_cache,
                start_frame_int=start_frame_int,
                head_start=h_start, head_end=h_end)
            roped_query = roped_q_full[:, sp_start:sp_start + sp_shard_len]

        roped_key = self._nki_rope_apply(
            k_full_4d, grid_sizes, freqs_cos, freqs_sin,
            start_frame=start_frame_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=start_frame_int,
            head_start=h_start, head_end=h_end)

        if kv_cache is None or self._will_anchor_write(kv_cache, current_start):
            k = self._slice_heads(k_full)
        else:
            k = None

        return roped_query, k, v, roped_key

    def _cache_write(self, k, v, roped_key, kv_cache, cache_start, shared_buffers):
        # TRUE CACHE-SHARD: k/v/roped_key are this rank's 1/world slice and kv_cache["k"]/["v"]
        # are allocated at 1/world size, so every position quantity uses the sharded size.
        # Structure is unchanged — each rank's stream is the full stream at 1/world scale
        # (verify_cache_shard.py max|Δ|=0).
        block_length = self._cs_block_length if self._cache_shard else self.block_length
        kv_cache_size = (self._cs_kv_cache_logical_size if self._cache_shard
                         else self.kv_cache_logical_size)
        # cache_start arrives in FULL sequence coords; convert to sharded coords so cache_end /
        # global_end_index / num_evicted / evict slice all live at 1/world scale consistently
        # (verify_cache_start_shardcoords.py: sharded bookkeeping == full // world every phase).
        if self._cache_shard and self.world_size > 1:
            cache_start = cache_start // self.world_size
        cache_end = cache_start + block_length
        global_end_index = kv_cache["global_end_index"]
        local_end_index_current = kv_cache["local_end_index"]
        num_new_tokens = cache_end - global_end_index
        sink_tokens = block_length

        buffer_k, buffer_v = shared_buffers

        num_evicted = 0
        if (num_new_tokens > 0) and (
                num_new_tokens + local_end_index_current > kv_cache_size):
            num_evicted = num_new_tokens + local_end_index_current - kv_cache_size
            evict_rolled = kv_cache_size - 2 * sink_tokens
            src_start = sink_tokens + num_evicted
            self._cache_copy_inplace(
                buffer_k[0, :evict_rolled], kv_cache["k"][0, src_start:src_start + evict_rolled],
                buffer_v[0, :evict_rolled], kv_cache["v"][0, src_start:src_start + evict_rolled])
            self._cache_copy_inplace(
                kv_cache["k"][0, sink_tokens:sink_tokens + evict_rolled], buffer_k[0, :evict_rolled],
                kv_cache["v"][0, sink_tokens:sink_tokens + evict_rolled], buffer_v[0, :evict_rolled])

        local_end_index = local_end_index_current + num_new_tokens - num_evicted
        local_start_index = local_end_index - block_length

        if local_start_index == 0:
            self._cache_copy_inplace(
                kv_cache["k"][0, :block_length], k[0, :block_length],
                kv_cache["v"][0, :block_length], v[0, :block_length])
        else:
            self._cache_copy_inplace(
                kv_cache["k"][0, local_start_index:local_end_index], roped_key[0, :block_length],
                kv_cache["v"][0, local_start_index:local_end_index], v[0, :block_length])

        if num_new_tokens > 0:
            kv_cache["global_end_index"] = cache_end
            kv_cache["local_end_index"] = local_end_index

        return local_end_index, local_start_index

    def _assemble_kv(self, roped_key, v, kv_cache, grid_sizes, freqs_cos, freqs_sin,
                     shared_buffers, updating_cache, valid_tokens,
                     current_start_frame, local_end_index, local_start_index,
                     rope_grid_cache=None):
        _, h, w = grid_sizes
        grid_sizes_one_block = (3, h, w)
        buffer_k, buffer_v = shared_buffers
        device = roped_key.device

        # TRUE CACHE-SHARD: every position quantity scales by 1/world (verify_cache_shard.py).
        # In the denoise path only the final valid_tokens copy is live (updating_cache=False,
        # local_start_index==0 at phase 0), but substitute the named sizes structurally so any
        # future denoise phase that rolls stays consistent. valid_tokens is already sharded by
        # the caller (forward). frame_length in the wc branch has no sharded companion; that
        # branch is dead in the sharded denoise path (see report note).
        block_length = self._cs_block_length if self._cache_shard else self.block_length
        max_attention_size = (self._cs_max_attention_size if self._cache_shard
                              else self.max_attention_size)

        # CACHE-SHARD anchor RoPE: the cached sink block below holds only THIS rank's
        # per-block-0 TOKEN shard (ws_block tokens at in-block global offset
        # world_rank*ws_block). Pass tok_offset/tok_len so the single-block grid (3,h,w)
        # is sliced to this rank's tokens instead of asserting seq_len(block)==s(ws_block).
        # Do NOT pass head_start/head_end: the cache already stores head-sliced K (n_local
        # heads), so _nki_rope_apply defaults head_end=x.shape[2]=n_local (all cached heads).
        # RoPE is per-position AND per-head-independent -> bit-exact (verify_anchor_shard.py
        # max|Δ|=0). Non-shard path: empty kwargs -> full-block RoPE, unchanged.
        anchor_shard_kw = {}
        if self._cache_shard and self.world_size > 1:
            ws_block = self._cs_block_length
            world_rank = ps.get_rank("world")
            anchor_shard_kw = dict(
                tok_offset=world_rank * ws_block, tok_len=ws_block)

        if updating_cache:
            cache_len = min(local_end_index, max_attention_size)
            cache_start_pos = max(0, local_end_index - max_attention_size)

            self._cache_copy_inplace(
                buffer_k[0, :cache_len],
                kv_cache["k"][0, cache_start_pos:cache_start_pos + cache_len],
                buffer_v[0, :cache_len],
                kv_cache["v"][0, cache_start_pos:cache_start_pos + cache_len])

            if cache_start_pos == 0:
                anchor_roped = self._nki_rope_apply(
                    kv_cache["k"][0, :block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin,
                    start_frame=torch.tensor(0, device=device),
                    rope_grid_cache=rope_grid_cache, start_frame_int=0,
                    **anchor_shard_kw)
                self._cache_copy_inplace(
                    buffer_k[0, :block_length], anchor_roped[0])

            return cache_len

        offset = 0
        if local_start_index > 0:
            wc_max = max_attention_size - valid_tokens - block_length
            wc_end = local_start_index
            wc_start = max(block_length, wc_end - wc_max)
            wc_len = wc_end - wc_start

            wc_frame_length = wc_len // self.frame_length
            rope_start_frame = current_start_frame - wc_frame_length - 3
            anchor_roped = self._nki_rope_apply(
                kv_cache["k"][0, :block_length].unsqueeze(0),
                grid_sizes_one_block, freqs_cos, freqs_sin,
                start_frame=torch.tensor(rope_start_frame, device=device),
                rope_grid_cache=rope_grid_cache,
                start_frame_int=rope_start_frame,
                **anchor_shard_kw)
            self._cache_copy_inplace(
                buffer_k[0, :block_length], anchor_roped[0],
                buffer_v[0, :block_length], kv_cache["v"][0, :block_length])
            offset = block_length

            if wc_len > 0:
                self._cache_copy_inplace(
                    buffer_k[0, offset:offset + wc_len], kv_cache["k"][0, wc_start:wc_start + wc_len],
                    buffer_v[0, offset:offset + wc_len], kv_cache["v"][0, wc_start:wc_start + wc_len])
            offset += wc_len

        self._cache_copy_inplace(
            buffer_k[0, offset:offset + valid_tokens], roped_key[0, :valid_tokens],
            buffer_v[0, offset:offset + valid_tokens], v[0, :valid_tokens])
        return offset + valid_tokens

    def _attend(self, roped_query, shared_buffers, k_len_int):
        buffer_k, buffer_v = shared_buffers
        q_kern = roped_query[0].permute(1, 2, 0).contiguous()
        k_kern = buffer_k[0].permute(1, 2, 0).contiguous()
        v_kern = buffer_v[0].permute(1, 0, 2).contiguous()

        import os as _os
        if self._cache_shard and self.world_size > 1:
            return self._attend_cache_shard(q_kern, k_kern, v_kern, k_len_int)

        assert k_kern.shape[2] % ATTN_SEQLEN_MULTIPLE == 0
        assert v_kern.shape[1] % ATTN_SEQLEN_MULTIPLE == 0

        if _os.environ.get("RF_RING_SHARD", "0") == "1" and self.sp_degree > 1:
            return self._attend_ring_shard(q_kern, k_kern, v_kern, k_len_int)
        if _os.environ.get("RF_RING", "0") == "1" and self.sp_degree > 1:
            return self._attend_ring(q_kern, k_kern, v_kern, k_len_int)

        out = wan_flash_self_attn(
            q_kern, k_kern, v_kern,
            softmax_scale=self.softmax_scale,
            actual_seqlen_k=k_len_int,
            use_dynamic_loop=False,
        )
        return out.unsqueeze(0).flatten(2)

    def _attend_ring(self, q_kern, k_kern, v_kern, k_len_int):
        """Ring/context-parallel attention over the assembled KV window (STAGE 3a).

        Shards the window's valid k_len_int tokens into sp contiguous global-position
        segments, runs the flash kernel in partial mode (return_partials=True) on each
        segment, and combines the per-segment (O_unnorm, row_max, row_sum) in GLOBAL order
        with the online-softmax merge — bit-exact to full-window attention
        (verify_ring_kvcache_multiphase.py max|Δ|<1e-12; global-order combine is exact per
        verify_ring_attention_exact.py). k_kern is [d, seqlen_k]; v_kern is [seqlen_k, d].

        NOTE (stage 3a): this validates the ring kernel+combine on device (ACC GATE
        max|Δ|=0). It does NOT yet remove the K/V world-gather, so it is a CORRECTNESS
        step, not an fps win — the fps lever is stage 3b (shard the persistent cache so the
        world-gather is eliminated). Kept behind RF_RING so the 14.13 path is default.
        """
        sp = self.sp_degree
        # split the VALID window (k_len_int, e.g. 3600) into sp contiguous global-position
        # segments. seg need NOT be a multiple of ATTN_SEQLEN_MULTIPLE — that constraint is
        # on the padded BUFFER width, not the valid length (the default path passes a
        # 8192-padded buffer with actual_seqlen_k=k_len_int). Each segment gets its own
        # 8192-padded buffer with actual_seqlen_k=seg; the kernel masks past seg.
        # kernel layout: q_kern [bs,d,Sq], k_kern [bs,d,Sk], v_kern [bs,Sk,d]  (bs=heads/shard).
        # DIAGNOSTIC (RF_RING_NSEG): number of segments to split the window into. Default sp.
        # RF_RING_NSEG=1 => one segment = whole window in ONE return_partials call + normalize
        # (no cross-shard combine). Isolates kernel-partial-write correctness from combine:
        #   1-seg garbage  => kernel partial-write bug; 1-seg OK => combine/segmentation bug.
        import os as _os2
        nseg = int(_os2.environ.get("RF_RING_NSEG", str(sp)))
        assert k_len_int % nseg == 0, (
            f"ring: valid window k_len_int={k_len_int} must be divisible by nseg={nseg}")
        bs, d_dim, Sk = k_kern.shape
        # FAST PATH (nseg==1): the whole window is on this rank in one piece. combine-of-one
        # is identity, so skip the return_partials/combine machinery entirely and make a
        # single plain kernel call — byte-identical to the baseline _attend, baseline speed.
        # RF_RING=1 RF_RING_NSEG=1 => ring plumbing at 14fps (proves the ring path costs
        # nothing when K/V is not sharded). Sharded K/V (nseg=sp) is the real win.
        if nseg == 1:
            out = wan_flash_self_attn(
                q_kern, k_kern, v_kern,
                softmax_scale=self.softmax_scale,
                actual_seqlen_k=k_len_int,
                use_dynamic_loop=False,
            )
            return out.unsqueeze(0).flatten(2)
        seg = k_len_int // nseg
        buf_w = ((seg + ATTN_SEQLEN_MULTIPLE - 1) // ATTN_SEQLEN_MULTIPLE) * ATTN_SEQLEN_MULTIPLE
        buf_full = ((k_len_int + ATTN_SEQLEN_MULTIPLE - 1) // ATTN_SEQLEN_MULTIPLE) * ATTN_SEQLEN_MULTIPLE

        m = None  # running max [Sq, bs]
        l = None  # running sum  [Sq, bs]
        acc = None  # running unnormalized output [Sq, bs, d]
        for s in range(nseg):  # GLOBAL segment order — required for bit-exactness
            # pad each segment's k/v to a multiple of ATTN_SEQLEN_MULTIPLE; kernel ignores
            # the pad via actual_seqlen_k=seg (same masking the default path uses).
            k_seg = k_kern.new_zeros((bs, d_dim, buf_w))
            v_seg = v_kern.new_zeros((bs, buf_w, d_dim))
            k_seg[:, :, :seg] = k_kern[:, :, s * seg:(s + 1) * seg]
            v_seg[:, :seg, :] = v_kern[:, s * seg:(s + 1) * seg, :]
            O_s, max_s, sum_s = wan_flash_self_attn(
                q_kern, k_seg.contiguous(), v_seg.contiguous(),
                softmax_scale=self.softmax_scale,
                actual_seqlen_k=seg,
                use_dynamic_loop=False,
                return_partials=True,
            )
            # DIAGNOSTIC RF_RING_DEBUG: on rank 0, verify this segment's partials against a
            # normalized single-seg call (return_partials=False) on the SAME seg. The
            # normalized default is known-correct (NSEG=1 passed). If O_s/sum_s reconstruct
            # softmax(seg) then the kernel partials are right and the bug is combine-only.
            if _os2.environ.get("RF_RING_DEBUG", "0") == "1" and ps.get_rank("world") == 0:
                ref_seg = wan_flash_self_attn(
                    q_kern, k_seg.contiguous(), v_seg.contiguous(),
                    softmax_scale=self.softmax_scale, actual_seqlen_k=seg,
                    use_dynamic_loop=False, return_partials=False,
                )  # [Sq, bs, d] normalized = O_s / sum_s
                recon = (O_s / sum_s).to(ref_seg.dtype)
                dO = (recon - ref_seg).abs().max().item()
                # HYPOTHESIS: exported max_s != the internal max used for O_s/sum_s.
                # Recompute the TRUE per-row max on-device from the same inputs and compare.
                # q_kern [bs,d,Sq], k_seg [bs,d,seg] -> scores [bs,Sq,seg]; max over keys.
                S_true = torch.einsum('bdq,bdk->bqk', q_kern.float(),
                                      k_seg[:, :, :seg].float()) * self.softmax_scale
                true_max = S_true.max(dim=-1).values.permute(1, 0).unsqueeze(-1)  # [Sq,bs,1]
                dmax = (max_s.float() - true_max).abs().max().item()
                print(f"[RING_DEBUG seg={s}] |O_s/sum_s-default|={dO:.3e} "
                      f"max_s[{max_s.min().item():.3e},{max_s.max().item():.3e}] "
                      f"true_max[{true_max.min().item():.3e},{true_max.max().item():.3e}] "
                      f"|max_s-true_max|={dmax:.3e}", flush=True)
            # kernel partial outputs: O_s [Sq, bs, d], max_s/sum_s [Sq, bs, 1]
            # (trailing 1 kept from the kernel's HBM layout; broadcasts over d).
            # DIAGNOSTIC RF_RING_SAFECOMBINE=1: recombine WITHOUT row_max. Each segment's
            # O_s = sum_j exp(S-max_s)V, sum_s = sum_j exp(S-max_s). Multiply both by
            # exp(max_s) to put every segment on the SAME (unshifted) scale, then just add:
            #   O_true = sum_s exp(max_s) O_s / sum_s exp(max_s) sum_s.
            # Safe at seqlen 900 (no overflow). If NSEG=4 PASSES here but FAILS with the
            # max-merge, row_max from the kernel is the bug. If it still FAILS, O_s/sum_s
            # are wrong per-segment.
            if _os2.environ.get("RF_RING_SAFECOMBINE", "0") == "1":
                w = torch.exp(torch.where(max_s > 1e30, torch.full_like(max_s, float('-inf')), max_s))
                if m is None:
                    acc = O_s * w
                    l = sum_s * w
                    m = torch.zeros_like(max_s)
                else:
                    acc = acc + O_s * w
                    l = l + sum_s * w
                continue
            # Minimal compile-safe online-softmax merge. RING_DEBUG (run rx7dt) proved every
            # segment's max_s is finite and small (−9..+68) — segments are NEVER fully masked
            # (each query row has `seg` valid keys), so no sentinel/inf ever occurs. The
            # earlier isinf/nan_to_num/-inf guards were dead code AND are exactly the
            # data-dependent ops that miscompile under torch.compile(fullgraph=True) on Neuron
            # -> the NSEG=4 garbage. Strip them; plain max/exp/add only (matches the CPU proof).
            if m is None:
                m, l, acc = max_s, sum_s, O_s
            else:
                m_new = torch.maximum(m, max_s)
                cp = torch.exp(m - m_new)
                cc = torch.exp(max_s - m_new)
                l = l * cp + sum_s * cc
                acc = acc * cp + O_s * cc
                m = m_new
        out = acc / l                              # [Sq, bs, d] fp32
        # FINAL self-check: inputs (O_s/sum_s/max_s) all proven bit-exact on device, yet
        # frames=255. The ONE thing not yet measured is the combine OUTPUT. Compare `out`
        # against a single full-window default call (proven-correct path) over the WHOLE
        # window. ~0 => combine is correct, bug is downstream (reshape/dtype/o-proj/caller);
        # large => the combine LOOP miscompiles on device despite bit-exact inputs.
        if _os2.environ.get("RF_RING_DEBUG", "0") == "1" and ps.get_rank("world") == 0:
            kfull = k_kern.new_zeros((bs, d_dim, buf_full))
            vfull = v_kern.new_zeros((bs, buf_full, d_dim))
            kfull[:, :, :k_len_int] = k_kern[:, :, :k_len_int]
            vfull[:, :k_len_int, :] = v_kern[:, :k_len_int, :]
            ref_full = wan_flash_self_attn(
                q_kern, kfull.contiguous(), vfull.contiguous(),
                softmax_scale=self.softmax_scale, actual_seqlen_k=k_len_int,
                use_dynamic_loop=False, return_partials=False)  # [Sq,bs,d] normalized
            dcomb = (out.to(ref_full.dtype) - ref_full).abs().max().item()
            print(f"[RING_DEBUG COMBINE] |combined_out - full_default|max={dcomb:.3e} "
                  f"out[{out.min().item():.3e},{out.max().item():.3e}]", flush=True)
        # combine runs in fp32 (kernel partials are fp32 for cross-rank precision), but the
        # default kernel returns q.dtype (bf16) and the downstream Linear (self.o) expects
        # that — cast back or the o-proj matmul hits 'input datatypes mismatched'.
        out = out.to(q_kern.dtype)
        # match default _attend contract exactly: out.unsqueeze(0).flatten(2)
        # -> [1, Sq, bs*d]  (same as wan_flash_self_attn result path).
        return out.unsqueeze(0).flatten(2)

    def _attend_ring_shard(self, q_kern, k_kern, v_kern, k_len_int):
        """STAGE 3b: sharded-window attention. Each rank contributes ONLY its RAW L/sp slice;
        all_gather over attn-sp reassembles the contiguous window; ONE flash call (like the
        baseline). Fixes the 3 wastes of the first attempt (run w4rmj, ~5.9fps): (1) gather
        RAW seg-length slices, NOT 8192-padded (was 9.1x byte waste); (2) reassemble into ONE
        contiguous [bs,d,k_len] window and make a SINGLE kernel call (was 4 padded calls +
        online-softmax combine); (3) all_gather over attn-sp is the ONLY exchange primitive
        (Neuron has no P2P — run sdjkl). Result is byte-identical to the baseline full-window
        attention (concatenation of raw slices in rank order == the window). q_kern [bs,d,Sq];
        k_kern [bs,d,Sk]; v_kern [bs,Sk,d]."""
        sp = self.sp_degree
        r = self.sp_rank
        assert k_len_int % sp == 0, f"ring-shard: k_len={k_len_int} not divisible by sp={sp}"
        seg = k_len_int // sp
        bs, d_dim, Sk = k_kern.shape

        # this rank's RAW slice (global segment r), no padding.
        k_own = k_kern[:, :, r * seg:(r + 1) * seg].contiguous()          # [bs,d,seg]
        v_own = v_kern[:, r * seg:(r + 1) * seg, :].contiguous()          # [bs,seg,d]
        # all_gather RAW slices over attn-sp -> contiguous window in rank order.
        # k gathered on the LAST (seqlen) axis, v on the MIDDLE (seqlen) axis.
        k_cat = torch.empty((bs, d_dim, sp * seg), dtype=k_own.dtype, device=k_own.device)
        v_cat = torch.empty((bs, sp * seg, d_dim), dtype=v_own.dtype, device=v_own.device)
        # all_gather_into_tensor concatenates on dim 0; gather into a [sp,...] view then
        # move the sp axis into the seqlen position to get the contiguous window.
        k_g = torch.empty((sp, bs, d_dim, seg), dtype=k_own.dtype, device=k_own.device)
        v_g = torch.empty((sp, bs, seg, d_dim), dtype=v_own.dtype, device=v_own.device)
        ps.all_gather_into_tensor(k_g.view(sp * bs, d_dim, seg), k_own, "attn-sp")
        ps.all_gather_into_tensor(v_g.view(sp * bs, seg, d_dim), v_own, "attn-sp")
        # k_g[s] is rank s's [bs,d,seg]; window = concat over s on seqlen -> [bs,d,sp*seg].
        k_cat = k_g.permute(1, 2, 0, 3).reshape(bs, d_dim, sp * seg).contiguous()
        v_cat = v_g.permute(1, 0, 2, 3).reshape(bs, sp * seg, d_dim).contiguous()
        # ONE flash call over the reassembled contiguous window (baseline-identical path).
        out = wan_flash_self_attn(
            q_kern, k_cat, v_cat,
            softmax_scale=self.softmax_scale,
            actual_seqlen_k=k_len_int,
            use_dynamic_loop=False,
        )
        return out.unsqueeze(0).flatten(2)

    def _attend_cache_shard(self, q_kern, k_kern, v_kern, k_len_int):
        """TRUE CACHE-SHARD attention (RF_RING_CACHESHARD). Each rank holds only its 1/world
        token-shard of the KV window (k_len_int is the SHARDED window length). all_gather the
        per-rank window-shards over the 'world' group and reassemble the FULL window via the
        PER-BLOCK INTERLEAVED reshape proven bit-exact in verify_cache_shard.py:
            full window = for each block in the window, concat rank0..rank(world-1) slices.
        Then ONE wan_flash_self_attn call over the reassembled window (identical structure to
        _attend_ring_shard's nseg==1 fast path). q_kern [bs,d,Sq]; k_kern [bs,d,buf_w];
        v_kern [bs,buf_w,d] (bs = heads/shard). all_gather is the only exchange primitive
        (Neuron has no P2P)."""
        N = self.world_size
        bs, d_dim, _ = k_kern.shape
        # this rank's RAW valid window shard (no padding) — the leading k_len_int columns.
        k_own = k_kern[:, :, :k_len_int].contiguous()          # [bs, d, k_len_int]
        v_own = v_kern[:, :k_len_int, :].contiguous()          # [bs, k_len_int, d]
        ws_block = self._cs_block_length
        assert k_len_int % ws_block == 0, (
            f"cache-shard: window shard {k_len_int} not a multiple of ws_block {ws_block}")
        nblocks = k_len_int // ws_block

        # IN-KERNEL GATHER (RF_CACHESHARD_INKERNEL): the fps lever. Replace the two torch
        # all_gather BARRIERS below with one NKI kernel that DMAs each rank's shard to
        # shared_hbm then ncc.all_gather (DMA engine, overlaps Q compute) + reassembles the
        # full window in-kernel. Torch all_gather = memory win only (~12fps); in-kernel
        # overlapping exchange is the throughput play. ReplicaGroup = the 'world' ranks.
        import os as _os_ik
        if _os_ik.environ.get("RF_CACHESHARD_INKERNEL", "0") == "1":
            from nki.collectives import ReplicaGroup
            rg = ReplicaGroup([list(range(N))])
            out = wan_flash_self_attn_gather_kv(
                q_kern, k_own, v_own, rg, N, nblocks,
                softmax_scale=self.softmax_scale)
            return out.unsqueeze(0).flatten(2)

        # STAGE 1 (RF_CACHESHARD_STAGE1): KV-parallel, the design floor. Each rank does ONE
        # flash over its OWN 1/world KV shard (k_own, all nblocks*ws tokens) -> partial; then
        # all_gather the PARTIALS (not KV) over world and online-softmax-merge in rank order
        # (proven ORDER-INVARIANT + bit-exact, verify_kvparallel_oneflash.py). This is 1 flash
        # per rank (NOT N -> 0.32fps, NOT full reassembly -> 5.29fps) with KV 1/world. Torch
        # all_gather = barrier (no overlap yet) -> this measures the NON-OVERLAPPED FLOOR before
        # the fused kernel moves the collective inside to overlap the matmul (stages 2-3).
        if _os_ik.environ.get("RF_CACHESHARD_STAGE1", "0") == "1":
            buf_w = ((k_len_int + ATTN_SEQLEN_MULTIPLE - 1) // ATTN_SEQLEN_MULTIPLE) * ATTN_SEQLEN_MULTIPLE
            k_pad = k_own.new_zeros((bs, d_dim, buf_w)); k_pad[:, :, :k_len_int] = k_own
            v_pad = v_own.new_zeros((bs, buf_w, d_dim)); v_pad[:, :k_len_int, :] = v_own
            O_own, max_own, sum_own = wan_flash_self_attn(
                q_kern, k_pad.contiguous(), v_pad.contiguous(),
                softmax_scale=self.softmax_scale, actual_seqlen_k=k_len_int,
                use_dynamic_loop=False, return_partials=True)
            # gather partials over world. O_own [Sq,bs,d], max/sum [Sq,bs,1].
            Sq = O_own.shape[0]
            O_g = torch.empty((N, Sq, bs, d_dim), dtype=O_own.dtype, device=O_own.device)
            mx_g = torch.empty((N, Sq, bs, 1), dtype=max_own.dtype, device=max_own.device)
            sm_g = torch.empty((N, Sq, bs, 1), dtype=sum_own.dtype, device=sum_own.device)
            ps.all_gather_into_tensor(O_g.view(N * Sq, bs, d_dim), O_own.contiguous(), "world")
            ps.all_gather_into_tensor(mx_g.view(N * Sq, bs, 1), max_own.contiguous(), "world")
            ps.all_gather_into_tensor(sm_g.view(N * Sq, bs, 1), sum_own.contiguous(), "world")
            m = l = acc = None
            for r in range(N):                       # rank order (order-invariant per proof)
                O_s, max_s, sum_s = O_g[r], mx_g[r], sm_g[r]
                if m is None:
                    m, l, acc = max_s, sum_s, O_s
                else:
                    m_new = torch.maximum(m, max_s)
                    cp, cc = torch.exp(m - m_new), torch.exp(max_s - m_new)
                    l = l * cp + sum_s * cc
                    acc = acc * cp + O_s * cc
                    m = m_new
            out = (acc / l).to(q_kern.dtype)          # [Sq,bs,d]
            return out.unsqueeze(0).flatten(2)

        # all_gather over world: dim-0 concat into a [N, bs, ...] view (k on last/seqlen axis,
        # v on middle/seqlen axis).
        k_g = torch.empty((N, bs, d_dim, k_len_int), dtype=k_own.dtype, device=k_own.device)
        v_g = torch.empty((N, bs, k_len_int, d_dim), dtype=v_own.dtype, device=v_own.device)
        ps.all_gather_into_tensor(k_g.view(N * bs, d_dim, k_len_int), k_own, "world")
        ps.all_gather_into_tensor(v_g.view(N * bs, k_len_int, d_dim), v_own, "world")

        # NO-REASSEMBLY COMBINE (RF_CACHESHARD_COMBINE): isolate whether the strided
        # per-block reassembly copy is the fps killer (torch path = 12.25 WITH it; in-kernel
        # strided dma_copy = 5.29). Instead of rebuilding the full window, flash each shard
        # segment in GLOBAL ORDER (per block b, per rank r: the contiguous [b*ws:+ws] slice)
        # with return_partials and online-softmax-merge (bit-exact, verify_ring_attention_exact
        # requires global order). Each gathered slice k_g[r,:, :, b*ws:+ws] is CONTIGUOUS — no
        # strided copy. Torch can't overlap (all_gather is a barrier) but this measures the
        # reassembly cost cleanly; if it recovers toward 14, move this combine in-kernel.
        if _os_ik.environ.get("RF_CACHESHARD_COMBINE", "0") == "1":
            ws = ws_block
            buf_w = ((ws + ATTN_SEQLEN_MULTIPLE - 1) // ATTN_SEQLEN_MULTIPLE) * ATTN_SEQLEN_MULTIPLE
            m = l = acc = None
            for b in range(nblocks):
                for r in range(N):                       # GLOBAL order: block b, rank 0..N-1
                    k_seg = k_own.new_zeros((bs, d_dim, buf_w))
                    v_seg = v_own.new_zeros((bs, buf_w, d_dim))
                    k_seg[:, :, :ws] = k_g[r, :, :, b * ws:(b + 1) * ws]
                    v_seg[:, :ws, :] = v_g[r, :, b * ws:(b + 1) * ws, :]
                    O_s, max_s, sum_s = wan_flash_self_attn(
                        q_kern, k_seg.contiguous(), v_seg.contiguous(),
                        softmax_scale=self.softmax_scale, actual_seqlen_k=ws,
                        use_dynamic_loop=False, return_partials=True)
                    # kernel partials: O_s [Sq,bs,d], max_s/sum_s [Sq,bs,1]. Merge exactly as
                    # _attend_ring (no extra unsqueeze — the trailing 1 broadcasts over d).
                    if m is None:
                        m, l, acc = max_s, sum_s, O_s
                    else:
                        m_new = torch.maximum(m, max_s)
                        cp, cc = torch.exp(m - m_new), torch.exp(max_s - m_new)
                        l = l * cp + sum_s * cc
                        acc = acc * cp + O_s * cc
                        m = m_new
            out = (acc / l).to(q_kern.dtype)          # [Sq,bs,d]
            return out.unsqueeze(0).flatten(2)
        # PER-BLOCK INTERLEAVE: each rank's window-shard holds nblocks blocks of ws_block
        # tokens each; the full window = for each block, concat rank0..N-1 (verify_cache_shard).
        full_len = N * k_len_int
        # k_g [N,bs,d,nblocks,ws_block] -> [bs,d,nblocks,N,ws_block] -> [bs,d,nblocks*N*ws_block]
        k_cat = (k_g.view(N, bs, d_dim, nblocks, ws_block)
                     .permute(1, 2, 3, 0, 4)
                     .reshape(bs, d_dim, full_len).contiguous())
        v_cat = (v_g.view(N, bs, nblocks, ws_block, d_dim)
                     .permute(1, 2, 0, 3, 4)
                     .reshape(bs, full_len, d_dim).contiguous())
        out = wan_flash_self_attn(
            q_kern, k_cat, v_cat,
            softmax_scale=self.softmax_scale,
            actual_seqlen_k=full_len,
            use_dynamic_loop=False,
        )
        return out.unsqueeze(0).flatten(2)

    def _output_proj(self, out):
        out = self.o(out)
        if self.tp_degree > 1:
            seq_len = out.shape[1]
            out_flat = out.reshape(-1, self.dim)
            rs_out = torch.empty(
                seq_len // self.tp_degree, self.dim,
                dtype=out.dtype, device=out.device)
            ps.reduce_scatter_tensor(rs_out, out_flat, "attn-tp")
            out = rs_out.unsqueeze(0)
        return out

    def _forward_merged_cache_shard(
        self,
        x,
        grid_sizes,
        freqs_cos, freqs_sin,
        kv_cache,
        cache_update_start, current_start,
        cu_shared_buffers, dn_shared_buffers,
        num_valid_frames_dn,
        nfpb_cu,
        rope_grid_cache=None,
    ):
        """TRUE CACHE-SHARD (RF_RING_CACHESHARD) merged cu/dn path. Each rank holds ONLY its
        UNIFORM per-block 1/world slice of cu and dn (verify_dn_multiblock_shard.py PROVEN
        layout: rank r holds, for every block b, that block's r-th ws_block(=block//world)
        slice). NO world-gather of k/v — the persistent cache stores 1/world per rank and the
        full window is reassembled at attention time by _attend_cache_shard's per-block
        interleave. Mirrors the denoise cache-shard path (_qkv_rope cache-shard branch),
        applied independently to the cu stream and the dn stream.

        The INPUT x-shard (dit_model._forward_inference merged branch) already produced this
        per-block layout; k_local/v_local from _local_qkv_norm ARE this rank's per-block slices,
        laid out [cu_shard ; dn_shard]."""
        f_full, h, w = grid_sizes
        frame_seqlen = h * w
        grid_cu = (nfpb_cu, h, w)
        grid_dn = (f_full - nfpb_cu, h, w)

        N = self.world_size
        tp = self.tp_degree
        sp = self.sp_degree
        world_rank = ps.get_rank("world")

        n = self.num_heads
        d = self.head_dim
        n_local = self.heads_per_shard
        h_start = self.tp_rank * n_local
        h_end = h_start + n_local

        block = self.block_length              # full block token count (3 * frame_seqlen)
        ws_block = self._cs_block_length        # per-rank per-block slice = block // N
        assert ws_block is not None

        L_cu = nfpb_cu * frame_seqlen
        L_dn = (f_full - nfpb_cu) * frame_seqlen
        assert L_cu % block == 0 and L_dn % block == 0, (
            f"cache-shard merged: L_cu {L_cu} / L_dn {L_dn} not block({block})-aligned")
        nb_cu = L_cu // block                   # blocks in cu stream (==1 for nfpb_cu==nfpb)
        nb_dn = L_dn // block                   # blocks in dn stream (rolling window)
        cu_shard_len = nb_cu * ws_block         # this rank's cu tokens
        dn_shard_len = nb_dn * ws_block         # this rank's dn tokens

        q_local, k_local, v_local = self._local_qkv_norm(x)   # this rank's [cu;dn] slices, all heads
        q_cu_l = q_local[:cu_shard_len]
        q_dn_l = q_local[cu_shard_len:]
        k_cu_l = k_local[:cu_shard_len]
        k_dn_l = k_local[cu_shard_len:]
        v_cu_l = v_local[:cu_shard_len]
        v_dn_l = v_local[cu_shard_len:]

        cu_sf_int = cache_update_start // frame_seqlen
        dn_sf_int = current_start // frame_seqlen
        cu_sf_t = torch.tensor(cu_sf_int, device=x.device)
        dn_sf_t = torch.tensor(dn_sf_int, device=x.device)
        current_start_frame_dn = current_start // frame_seqlen

        def rope_per_block(t_local, grid, sf_t, sf_int, nblocks):
            # RoPE this rank's per-block shard at each block's GLOBAL token offset. RoPE is
            # per-position, so RoPE-shard-at-global-offset == RoPE-full-then-slice
            # (verify_krope_shard.py). rank world_rank holds, per block b, the tokens at global
            # offset b*block + world_rank*ws_block (verify_dn_multiblock_shard.py).
            pieces = []
            for b in range(nblocks):
                seg = t_local[b * ws_block:(b + 1) * ws_block].view(ws_block, n, d).unsqueeze(0)
                roped = self._nki_rope_apply(
                    seg, grid, freqs_cos, freqs_sin,
                    start_frame=sf_t, rope_grid_cache=rope_grid_cache,
                    start_frame_int=sf_int, head_start=h_start, head_end=h_end,
                    tok_offset=b * block + world_rank * ws_block, tok_len=ws_block)
                pieces.append(roped)
            return torch.cat(pieces, dim=1)     # [1, nblocks*ws_block, n_local, d]

        # --- roped keys / values (this rank's shard only; NO world-gather) ---
        rk_cu = rope_per_block(k_cu_l, grid_cu, cu_sf_t, cu_sf_int, nb_cu)
        rk_dn = rope_per_block(k_dn_l, grid_dn, dn_sf_t, dn_sf_int, nb_dn)
        v_cu = self._slice_heads(v_cu_l)        # [1, cu_shard_len, n_local, d]
        v_dn = self._slice_heads(v_dn_l)

        # --- roped queries: RoPE this rank's shard, gather ROPED q over attn-tp -> sp-shard ---
        # (roped-shard-then-gather == gather-then-rope, RoPE is per-position). Gather over
        # attn-tp reconstructs the sp-group's query tokens; permute the dn gather to PER-BLOCK
        # grouped order so the sp query matches the per-block cache layout at sp granularity.
        rq_cu_local = rope_per_block(q_cu_l, grid_cu, cu_sf_t, cu_sf_int, nb_cu)  # [1, cu_shard_len, n_local, d]
        rq_dn_local = rope_per_block(q_dn_l, grid_dn, dn_sf_t, dn_sf_int, nb_dn)  # [1, dn_shard_len, n_local, d]

        def gather_q_tp(rq_local, nblocks):
            # rq_local [1, nblocks*ws_block, n_local, d] -> flatten heads, all_gather over
            # attn-tp (tp consecutive world ranks), reshape [tp, nblocks, ws_block, n_local*d],
            # permute to PER-BLOCK grouped [nblocks, tp, ws_block, ...] -> contiguous sp query.
            flat = rq_local[0].reshape(nblocks * ws_block, n_local * d).contiguous()
            g = torch.empty(tp * nblocks * ws_block, n_local * d,
                            dtype=flat.dtype, device=flat.device)
            ps.all_gather_into_tensor(g, flat, "attn-tp")
            g = (g.view(tp, nblocks, ws_block, n_local * d)
                  .permute(1, 0, 2, 3)
                  .reshape(nblocks * tp * ws_block, n_local, d))
            return g.unsqueeze(0)                # [1, nblocks*tp*ws_block, n_local, d]

        q_cu_sp = gather_q_tp(rq_cu_local, nb_cu)   # [1, L_cu_sp, n_local, d]
        q_dn_sp = gather_q_tp(rq_dn_local, nb_dn)   # [1, L_dn_sp, n_local, d]

        # --- cache write (sharded sizes substituted inside _cache_write via self._cache_shard) ---
        if self._will_anchor_write(kv_cache, cache_update_start):
            k_cu = self._slice_heads_2d(k_cu_l).contiguous().unsqueeze(0)
        else:
            k_cu = None
        le_cu, ls_cu = self._cache_write(
            k_cu, v_cu, rk_cu, kv_cache, cache_update_start, cu_shared_buffers)

        if self._will_anchor_write(kv_cache, current_start):
            k_dn = self._slice_heads_2d(k_dn_l).contiguous().unsqueeze(0)
        else:
            k_dn = None
        le_dn, ls_dn = self._cache_write(
            k_dn, v_dn, rk_dn, kv_cache, current_start, dn_shared_buffers)

        # valid_tokens are SHARDED (1/world) — the sharded cache/buffers hold 1/world.
        vt_cu = (nfpb_cu * frame_seqlen) // N
        vt_dn = (num_valid_frames_dn * frame_seqlen) // N
        klen_cu = self._assemble_kv(
            rk_cu, v_cu, kv_cache, grid_cu, freqs_cos, freqs_sin,
            cu_shared_buffers, True, vt_cu,
            cache_update_start // frame_seqlen, le_cu, ls_cu,
            rope_grid_cache=rope_grid_cache)
        klen_dn = self._assemble_kv(
            rk_dn, v_dn, kv_cache, grid_dn, freqs_cos, freqs_sin,
            dn_shared_buffers, False, vt_dn,
            current_start_frame_dn, le_dn, ls_dn,
            rope_grid_cache=rope_grid_cache)

        # --- attention: _attend_cache_shard reassembles the FULL window per stream ---
        y_cu = self._attend(q_cu_sp, cu_shared_buffers, klen_cu)
        y_dn = self._attend(q_dn_sp, dn_shared_buffers, klen_dn)

        # --- OUTPUT reassembly ---
        # UNVERIFIED FOR PER-BLOCK CACHE-SHARD LAYOUT: the query sp-shard here is PER-BLOCK
        # grouped (cu contiguous; dn per-block at sp granularity), NOT the contiguous
        # dn[sp*L_dn_sp:] layout that restore_layout was written for. The output token ORDER
        # therefore differs from the non-shard path, so the reduce_scatter + restore_layout +
        # world-gather deinterleave below may place tokens incorrectly. This does NOT corrupt
        # the persistent cache (write/assemble use the proven per-block layout); it only risks
        # a scrambled OUTPUT, which the on-device ACC gate (gate_frames) will catch. Mirrors
        # the non-shard output tail for shape-correctness; reassembly correctness is deferred
        # to the device ACC gate rather than guessed here.
        sp_ = self.sp_degree
        L_cu_sp = q_cu_sp.shape[1]
        L_dn_sp = q_dn_sp.shape[1]
        L_full = L_cu + L_dn
        L_cu_N = L_cu // N
        L_dn_N = L_dn // N
        L_full_N = L_full // N

        y = self.o(torch.cat([y_cu, y_dn], dim=1))

        if tp > 1:
            y = y.reshape((L_cu_sp + L_dn_sp), self.dim)
            y_cu_part = y[:L_cu_sp].reshape(tp, L_cu_N, self.dim)
            y_dn_part = y[L_cu_sp:].reshape(tp, L_dn_N, self.dim)
            rearranged = torch.cat([y_cu_part, y_dn_part], dim=1).reshape(-1, self.dim)
            rs_out = torch.empty(L_full_N, self.dim, dtype=y.dtype, device=y.device)
            ps.reduce_scatter_tensor(rs_out, rearranged, "attn-tp")
            cu_dn_sep = rs_out
        else:
            cu_dn_sep = y.reshape(L_full_N, self.dim)

        if N == 1:
            return cu_dn_sep.unsqueeze(0)

        gathered = torch.empty(N * L_full_N, self.dim, dtype=cu_dn_sep.dtype, device=cu_dn_sep.device)
        ps.all_gather_into_tensor(gathered, cu_dn_sep, "world")
        full = restore_layout(gathered, N=N, nfpb=3, max_frames=15,
                              frame_seqlen=self.frame_length)
        out = full[world_rank * L_full_N:(world_rank + 1) * L_full_N]
        return out.unsqueeze(0)

    def forward_merged(
        self,
        x,
        grid_sizes,
        freqs_cos, freqs_sin,
        kv_cache,
        cache_update_start, current_start,
        cu_shared_buffers, dn_shared_buffers,
        num_valid_frames_dn,
        nfpb_cu,
        rope_grid_cache=None,
    ):
        assert x.shape[0] == 1
        if self._cache_shard and self.world_size > 1:
            return self._forward_merged_cache_shard(
                x, grid_sizes, freqs_cos, freqs_sin, kv_cache,
                cache_update_start, current_start,
                cu_shared_buffers, dn_shared_buffers,
                num_valid_frames_dn, nfpb_cu, rope_grid_cache)

        f_full, h, w = grid_sizes
        frame_seqlen = h * w
        grid_cu = (nfpb_cu, h, w)
        grid_dn = (f_full - nfpb_cu, h, w)
        L_cu = nfpb_cu * frame_seqlen
        L_dn = (f_full - nfpb_cu) * frame_seqlen
        L_full = L_cu + L_dn
        current_start_frame_dn = current_start // frame_seqlen

        sp = self.sp_degree
        tp = self.tp_degree
        N = self.world_size
        assert L_cu % N == 0, f"L_cu ({L_cu}) must be divisible by world_size ({N})"
        assert L_dn % N == 0, f"L_dn ({L_dn}) must be divisible by world_size ({N})"
        L_cu_sp = L_cu // sp
        L_dn_sp = L_dn // sp
        L_cu_N = L_cu // N
        L_dn_N = L_dn // N
        L_full_N = L_full // N

        q_local, k_local, v_local = self._local_qkv_norm(x)
        q_full, k_full, v_full = self._gather_qkv(
            q_local, k_local, v_local, L_full)

        n = self.num_heads
        d = self.head_dim
        n_local = self.heads_per_shard
        h_start = self.tp_rank * n_local
        h_end = h_start + n_local

        v_full_h = self._slice_heads_2d(v_full).contiguous()

        q_full_4d = q_full.view(L_full, n, d)
        k_full_4d = k_full.view(L_full, n, d)
        q_cu = q_full_4d[:L_cu].unsqueeze(0)
        q_dn = q_full_4d[L_cu:].unsqueeze(0)
        k_cu_full = k_full_4d[:L_cu].unsqueeze(0)
        k_dn_full = k_full_4d[L_cu:].unsqueeze(0)
        v_cu = v_full_h[:L_cu].unsqueeze(0)
        v_dn = v_full_h[L_cu:].unsqueeze(0)

        cu_sf_int = cache_update_start // frame_seqlen
        dn_sf_int = current_start // frame_seqlen
        cu_sf_t = torch.tensor(cu_sf_int, device=q_cu.device)
        dn_sf_t = torch.tensor(dn_sf_int, device=q_dn.device)

        rq_cu_full = self._nki_rope_apply(
            q_cu, grid_cu, freqs_cos, freqs_sin,
            start_frame=cu_sf_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=cu_sf_int,
            head_start=h_start, head_end=h_end)
        rk_cu = self._nki_rope_apply(
            k_cu_full, grid_cu, freqs_cos, freqs_sin,
            start_frame=cu_sf_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=cu_sf_int,
            head_start=h_start, head_end=h_end)
        rq_dn_full = self._nki_rope_apply(
            q_dn, grid_dn, freqs_cos, freqs_sin,
            start_frame=dn_sf_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=dn_sf_int,
            head_start=h_start, head_end=h_end)
        rk_dn = self._nki_rope_apply(
            k_dn_full, grid_dn, freqs_cos, freqs_sin,
            start_frame=dn_sf_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=dn_sf_int,
            head_start=h_start, head_end=h_end)

        if self._will_anchor_write(kv_cache, cache_update_start):
            k_cu = self._slice_heads_2d(k_full[:L_cu]).contiguous().unsqueeze(0)
        else:
            k_cu = None
        le_cu, ls_cu = self._cache_write(
            k_cu, v_cu, rk_cu, kv_cache, cache_update_start, cu_shared_buffers)

        if self._will_anchor_write(kv_cache, current_start):
            k_dn = self._slice_heads_2d(k_full[L_cu:]).contiguous().unsqueeze(0)
        else:
            k_dn = None
        le_dn, ls_dn = self._cache_write(
            k_dn, v_dn, rk_dn, kv_cache, current_start, dn_shared_buffers)

        klen_cu = self._assemble_kv(
            rk_cu, v_cu, kv_cache, grid_cu, freqs_cos, freqs_sin,
            cu_shared_buffers, True, nfpb_cu * frame_seqlen,
            cache_update_start // frame_seqlen, le_cu, ls_cu,
            rope_grid_cache=rope_grid_cache)
        klen_dn = self._assemble_kv(
            rk_dn, v_dn, kv_cache, grid_dn, freqs_cos, freqs_sin,
            dn_shared_buffers, False, num_valid_frames_dn * frame_seqlen,
            current_start_frame_dn, le_dn, ls_dn,
            rope_grid_cache=rope_grid_cache)

        q_cu_sp = rq_cu_full[:, self.sp_rank * L_cu_sp:(self.sp_rank + 1) * L_cu_sp]
        q_dn_sp = rq_dn_full[:, self.sp_rank * L_dn_sp:(self.sp_rank + 1) * L_dn_sp]
        y_cu = self._attend(q_cu_sp, cu_shared_buffers, klen_cu)
        y_dn = self._attend(q_dn_sp, dn_shared_buffers, klen_dn)

        y = self.o(torch.cat([y_cu, y_dn], dim=1))

        if tp > 1:
            y = y.reshape(L_full // sp, self.dim)
            y_cu_part = y[:L_cu_sp].reshape(tp, L_cu_N, self.dim)
            y_dn_part = y[L_cu_sp:].reshape(tp, L_dn_N, self.dim)
            rearranged = torch.cat([y_cu_part, y_dn_part], dim=1).reshape(-1, self.dim)
            rs_out = torch.empty(L_full_N, self.dim, dtype=y.dtype, device=y.device)
            ps.reduce_scatter_tensor(rs_out, rearranged, "attn-tp")
            cu_dn_sep = rs_out
        else:
            cu_dn_sep = y.reshape(L_full_N, self.dim)

        if N == 1:
            return cu_dn_sep.unsqueeze(0)

        gathered = torch.empty(N * L_full_N, self.dim, dtype=cu_dn_sep.dtype, device=cu_dn_sep.device)
        ps.all_gather_into_tensor(gathered, cu_dn_sep, "world")
        # Derive nfpb/max_frames from this forward's actual geometry instead of
        # hardcoding the T=5 shape (nfpb=3, max_frames=15). f_full and nfpb_cu are
        # the same values used to build L_cu/L_dn above, so restore_layout's internal
        # L_full == L_cu + L_dn assertion holds for any denoising-step count T
        # (T=4 -> max_frames=12, T=5 -> max_frames=15).
        full = restore_layout(gathered, N=N, nfpb=nfpb_cu,
                              max_frames=(f_full - nfpb_cu),
                              frame_seqlen=self.frame_length)
        rank_world = ps.get_rank("world")
        out = full[rank_world * L_full_N:(rank_world + 1) * L_full_N]
        return out.unsqueeze(0)

    def forward(
        self,
        x,
        grid_sizes,
        freqs_cos,
        freqs_sin,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        updating_cache=False,
        num_valid_frames=None,
        shared_buffers=None,
        rope_grid_cache=None,
    ):
        assert kv_cache is not None
        assert x.shape[0] == 1, f"Batch size must be 1, got {x.shape[0]}"
        if cache_start is None:
            cache_start = current_start

        f, h, w = grid_sizes
        frame_seqlen = h * w
        current_start_frame = current_start // frame_seqlen

        roped_query, k, v, roped_key = self._qkv_rope(
            x, grid_sizes, freqs_cos, freqs_sin, current_start,
            rope_grid_cache=rope_grid_cache,
            kv_cache=kv_cache,
        )

        if num_valid_frames is not None:
            valid_tokens = num_valid_frames * frame_seqlen
        else:
            valid_tokens = f * h * w

        # TRUE CACHE-SHARD: valid_tokens (the tokens written+attended this call) also scales
        # by 1/world — roped_key/v from _qkv_rope are this rank's 1/world slice, and the
        # sharded cache/buffers hold 1/world. frame_seqlen % world_size == 0 (asserted in the
        # pipeline), so this divides cleanly.
        if self._cache_shard and self.world_size > 1:
            assert valid_tokens % self.world_size == 0, (
                f"valid_tokens ({valid_tokens}) not divisible by world_size "
                f"({self.world_size})")
            valid_tokens = valid_tokens // self.world_size

        local_end_index, local_start_index = self._cache_write(
            k, v, roped_key, kv_cache, cache_start, shared_buffers)

        k_len_int = self._assemble_kv(
            roped_key, v, kv_cache, grid_sizes, freqs_cos, freqs_sin,
            shared_buffers, updating_cache, valid_tokens,
            current_start_frame, local_end_index, local_start_index,
            rope_grid_cache=rope_grid_cache)

        x = self._attend(roped_query, shared_buffers, k_len_int)

        x = self._output_proj(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 layer_idx=0,
                 frame_length=1560):
        super().__init__()
        assert cross_attn_type == 't2v_cross_attn'
        assert cross_attn_norm
        self.layer_idx = layer_idx
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        self.norm1 = WanLayerNorm(dim, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.norm2 = WanLayerNorm(dim, eps)

        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, layer_idx,
            frame_length=frame_length)
        self.cross_attn = WanT2VCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps, layer_idx=layer_idx)

        self.ffn = _compile(nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim)))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.world_size = ps.get_world_size("world")
        self.rank = ps.get_rank("world")

        self._modulated_norm_scale_shard = _compile(modulated_norm_scale_shard)
        self._modulated_norm_shift_shard = _compile(modulated_norm_shift_shard)
        self._modulated_residual_shard = _compile(modulated_residual_shard)

    @staticmethod
    def shard_state_dict(full_sd, dim, num_heads):
        sd = {}
        self_attn_full = {
            key[len("self_attn."):]: val
            for key, val in full_sd.items()
            if key.startswith("self_attn.")
        }
        self_attn_sharded = CausalWanSelfAttention.shard_state_dict(
            self_attn_full, dim, num_heads)
        for key, val in self_attn_sharded.items():
            sd[f"self_attn.{key}"] = val
        for key, val in full_sd.items():
            if not key.startswith("self_attn."):
                sd[key] = val.clone()
        return sd

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs_cos,
        freqs_sin,
        context,
        context_lens,
        updating_cache=False,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        num_valid_frames=None,
        shared_buffers=None,
        mode="denoise",
        cache_update_start=None,
        cu_shared_buffers=None,
        nfpb_cu=None,
        rope_grid_cache=None,
    ):
        e0s, e1s, e2s, e3s, e4s, e5s = (
            e[:, 0], e[:, 1], e[:, 2], e[:, 3], e[:, 4], e[:, 5])
        m0, m1, m2, m3, m4, m5 = (
            self.modulation[:, 0], self.modulation[:, 1], self.modulation[:, 2],
            self.modulation[:, 3], self.modulation[:, 4], self.modulation[:, 5])
        ones_shard = torch.ones_like(e0s)

        attn_in = self._modulated_norm_shift_shard(
            self._modulated_norm_scale_shard(self.norm1(x), m1, e1s, ones_shard),
            m0, e0s,
        )

        if mode == "merged":
            assert cache_update_start is not None and nfpb_cu is not None
            y = self.self_attn.forward_merged(
                attn_in,
                grid_sizes,
                freqs_cos, freqs_sin,
                kv_cache,
                cache_update_start, current_start,
                cu_shared_buffers, shared_buffers,
                num_valid_frames_dn=num_valid_frames,
                nfpb_cu=nfpb_cu,
                rope_grid_cache=rope_grid_cache,
            )
        else:
            y = self.self_attn(
                attn_in,
                grid_sizes,
                freqs_cos,
                freqs_sin,
                kv_cache,
                current_start,
                cache_start,
                updating_cache=updating_cache,
                num_valid_frames=num_valid_frames,
                shared_buffers=shared_buffers,
                rope_grid_cache=rope_grid_cache,
            )
        x = self._modulated_residual_shard(x, y, m2, e2s)

        x = x + self.cross_attn(
            self.norm3(x), context, context_lens,
            crossattn_cache=crossattn_cache)

        y = self.ffn(
            self._modulated_norm_shift_shard(
                self._modulated_norm_scale_shard(self.norm2(x), m4, e4s, ones_shard),
                m3, e3s,
            )
        )
        x = self._modulated_residual_shard(x, y, m5, e5s)

        return x
