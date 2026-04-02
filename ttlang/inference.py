"""
Nanochat d32 full inference on Tenstorrent hardware via TT-Lang.

Loads a prepared weight bundle (from prepare_weights.py), runs autoregressive
generation with KV cache. All compute kernels are TT-Lang, host handles
embedding lookups, head splitting, KV cache management, and sampling.

Usage:
    python inference.py --weights /path/to/d32_weights.pt --prompt "Hello world"
"""

import os
import sys
import math
import time
import pickle
import argparse
import torch
import ttnn
import ttl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE = 32
# These are derived from model config at runtime, but we need HEAD_DIM/HALF_TILES
# as compile-time constants for the kernel definitions. d32 values:
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
HEAD_TILES = HEAD_DIM // TILE       # 4
HALF_TILES = HALF_DIM // TILE       # 2
NUM_HEADS = 16
SDPA_GRID_X = 8
SDPA_GRID_Y = 2  # 8 * 2 = 16 cores = NUM_HEADS
SOFTCAP = 15.0
QK_SCALE_SQ = 1.15 * 1.15
SDPA_SCALE_VAL = QK_SCALE_SQ / math.sqrt(HEAD_DIM)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def to_ttnn_l1(tensor, device):
    return ttnn.from_torch(
        tensor.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


def pad_to_tile(n):
    return ((n + TILE - 1) // TILE) * TILE


def pad_vocab_to_tile(t):
    """Pad last dim of 2D tensor to tile multiple."""
    cols = t.shape[1]
    padded = pad_to_tile(cols)
    if padded == cols:
        return t
    return torch.nn.functional.pad(t, (0, padded - cols))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

# -- RMSNorm factory (parameterized by embedding width) --
def make_rmsnorm_kernel(n_dim):
    dim_tiles = n_dim // TILE

    @ttl.operation(grid="auto")
    def rmsnorm_kernel(x, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        rsq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        # Pass 1: sum of squares across dim tiles
                        with x_dfb.wait() as x0:
                            with sq_dfb.reserve() as sq:
                                sq.store(x0 * x0)
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)
                        for j in range(dim_tiles - 1):
                            with x_dfb.wait() as xj:
                                with sq_dfb.reserve() as sq:
                                    sq.store(xj * xj)
                            with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as new_acc:
                                new_acc.store(av + rv)

                        # broadcast, scale by 1/N, add eps, rsqrt
                        with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                            bc.store(ttl.math.broadcast(total, bc, dims=[1]))
                        with bcast_dfb.wait() as bv, red_dfb.reserve() as scaled:
                            scaled.store(bv * ms + ttl.math.fill(bv, 1e-5))
                        with red_dfb.wait() as msq, rsq_dfb.reserve() as rsq:
                            rsq.store(ttl.math.rsqrt(msq))

                        # Pass 2: x * rsqrt(mean(x^2) + eps)
                        with rsq_dfb.wait() as rsqv:
                            for j in range(dim_tiles):
                                with x_dfb.wait() as xj, out_dfb.reserve() as o:
                                    o.store(xj * rsqv)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            with sc_dfb.reserve() as blk1, ms_dfb.reserve() as blk2:
                tx1 = ttl.copy(scaler[0, 0], blk1)
                tx2 = ttl.copy(mean_scale[0, 0], blk2)
                tx1.wait(); tx2.wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[tile_idx, j], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()

    return rmsnorm_kernel


rmsnorm_head = make_rmsnorm_kernel(HEAD_DIM)
# rmsnorm_embd created dynamically in NanochatModel.__init__ based on config


# -- Linear projection: out = x @ w --
# Full-K-in-DFB pattern: compiler handles K accumulation in f32 DST.
# NCOLS output columns per work unit for A-tile reuse.
K_CHUNK_MAX = 64  # used for MLP proj weight chunking
NCOLS = 4  # output columns per work unit (fits in L1 with K=64)

def make_linear_kernel(k_tiles):
    @ttl.operation(grid="auto")
    def linear_kernel(x, w, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_tiles = w.shape[1] // TILE
        col_groups = n_tiles // NCOLS
        total = m_tiles * col_groups
        units_per_core = -(-total // grid_cols)

        a_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_tiles), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_tiles, NCOLS), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, NCOLS), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    a_blk = a_dfb.wait()
                    w_blk = w_dfb.wait()
                    with out_dfb.reserve() as o:
                        o.store(a_blk @ w_blk)
                    a_blk.pop()
                    w_blk.pop()

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // col_groups
                    gi = t % col_groups
                    sc = gi * NCOLS
                    with a_dfb.reserve() as blk1, w_dfb.reserve() as blk2:
                        tx1 = ttl.copy(x[mi, 0:k_tiles], blk1)
                        tx2 = ttl.copy(w[0:k_tiles, sc:sc + NCOLS], blk2)
                        tx1.wait(); tx2.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // col_groups
                    gi = t % col_groups
                    sc = gi * NCOLS
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[mi, sc:sc + NCOLS]); tx.wait()

    return linear_kernel


# -- Fused triple linear: out1=x@w1, out2=x@w2, out3=x@w3 in one kernel --
# Reads x once, streams through all 3 weight matrices per work unit.
def make_triple_linear_kernel(k_tiles):
    @ttl.operation(grid="auto")
    def triple_linear_kernel(x, w1, w2, w3, out1, out2, out3):
        FUSED_NCOLS = 2
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_tiles = w1.shape[1] // TILE
        col_groups = n_tiles // FUSED_NCOLS
        total = m_tiles * col_groups
        units_per_core = -(-total // grid_cols)

        a_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_tiles), buffer_factor=1)
        w1_dfb = ttl.make_dataflow_buffer_like(w1, shape=(k_tiles, FUSED_NCOLS), buffer_factor=1)
        w2_dfb = ttl.make_dataflow_buffer_like(w2, shape=(k_tiles, FUSED_NCOLS), buffer_factor=1)
        w3_dfb = ttl.make_dataflow_buffer_like(w3, shape=(k_tiles, FUSED_NCOLS), buffer_factor=1)
        o1_dfb = ttl.make_dataflow_buffer_like(out1, shape=(1, FUSED_NCOLS), buffer_factor=1)
        o2_dfb = ttl.make_dataflow_buffer_like(out2, shape=(1, FUSED_NCOLS), buffer_factor=1)
        o3_dfb = ttl.make_dataflow_buffer_like(out3, shape=(1, FUSED_NCOLS), buffer_factor=1)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    with a_dfb.wait() as a_blk:
                        with w1_dfb.wait() as w1_blk, o1_dfb.reserve() as o:
                            o.store(a_blk @ w1_blk)
                        with w2_dfb.wait() as w2_blk, o2_dfb.reserve() as o:
                            o.store(a_blk @ w2_blk)
                        with w3_dfb.wait() as w3_blk, o3_dfb.reserve() as o:
                            o.store(a_blk @ w3_blk)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // col_groups
                    gi = t % col_groups
                    sc = gi * FUSED_NCOLS
                    with a_dfb.reserve() as b1, w1_dfb.reserve() as b2, w2_dfb.reserve() as b3, w3_dfb.reserve() as b4:
                        tx1 = ttl.copy(x[mi, 0:k_tiles], b1)
                        tx2 = ttl.copy(w1[0:k_tiles, sc:sc + FUSED_NCOLS], b2)
                        tx3 = ttl.copy(w2[0:k_tiles, sc:sc + FUSED_NCOLS], b3)
                        tx4 = ttl.copy(w3[0:k_tiles, sc:sc + FUSED_NCOLS], b4)
                        tx1.wait(); tx2.wait(); tx3.wait(); tx4.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // col_groups
                    gi = t % col_groups
                    sc = gi * FUSED_NCOLS
                    with o1_dfb.wait() as b1, o2_dfb.wait() as b2, o3_dfb.wait() as b3:
                        tx1 = ttl.copy(b1, out1[mi, sc:sc + FUSED_NCOLS])
                        tx2 = ttl.copy(b2, out2[mi, sc:sc + FUSED_NCOLS])
                        tx3 = ttl.copy(b3, out3[mi, sc:sc + FUSED_NCOLS])
                        tx1.wait(); tx2.wait(); tx3.wait()

    return triple_linear_kernel

# Standard linear for nanochat projections (K=64 for 2048-wide inputs)
linear_kernel = make_linear_kernel(64)
# Fused QKV: reads input once, computes 3 matmuls
triple_linear_kernel = make_triple_linear_kernel(64)
# Gate matmul for value embeddings: k=1 tile (reads first 32 cols of input)
linear_kernel_k1 = make_linear_kernel(1)


# -- Linear with column offset: out = x[:, col_off:col_off+k] @ w --
# Used for MLP proj where we read slices of a wide tensor without readback.
def make_linear_slice_kernel(k_tiles, col_offset_tiles):
    @ttl.operation(grid="auto")
    def linear_slice_kernel(x, w, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_tiles = w.shape[1] // TILE
        col_groups = n_tiles // NCOLS
        total = m_tiles * col_groups
        units_per_core = -(-total // grid_cols)

        a_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_tiles), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_tiles, NCOLS), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, NCOLS), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    a_blk = a_dfb.wait()
                    w_blk = w_dfb.wait()
                    with out_dfb.reserve() as o:
                        o.store(a_blk @ w_blk)
                    a_blk.pop()
                    w_blk.pop()

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // col_groups
                    gi = t % col_groups
                    sc = gi * NCOLS
                    with a_dfb.reserve() as blk1, w_dfb.reserve() as blk2:
                        tx1 = ttl.copy(x[mi, col_offset_tiles:col_offset_tiles + k_tiles], blk1)
                        tx2 = ttl.copy(w[0:k_tiles, sc:sc + NCOLS], blk2)
                        tx1.wait(); tx2.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // col_groups
                    gi = t % col_groups
                    sc = gi * NCOLS
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[mi, sc:sc + NCOLS]); tx.wait()

    return linear_slice_kernel

# MLP proj slice kernels: 4 chunks of K=64 tiles with different column offsets
N_MLP_CHUNKS = 4
mlp_proj_slice_kernels = [
    make_linear_slice_kernel(K_CHUNK_MAX, c * K_CHUNK_MAX) for c in range(N_MLP_CHUNKS)
]


# -- Elementwise ReLU squared --
@ttl.operation(grid="auto")
def relu_sq_kernel(x, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with x_dfb.wait() as xv, out_dfb.reserve() as o:
                    r = ttl.math.relu(xv)
                    o.store(r * r)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


# -- Rotary embeddings (broadcast cos/sin from single tile-row to all heads) --
# x: (n_head * TILE, HEAD_DIM), cos/sin: (TILE, HALF_DIM), out: same as x
@ttl.operation(grid="auto")
def rotary_kernel(x, cos, sin, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    seq_tiles = x.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // grid_cols)

    x1_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    x2_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    cos_dfb = ttl.make_dataflow_buffer_like(cos, shape=(1, 1), buffer_factor=2)
    sin_dfb = ttl.make_dataflow_buffer_like(sin, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < seq_tiles:
                for j in range(HALF_TILES):
                    with x1_dfb.wait() as x1, x2_dfb.wait() as x2, cos_dfb.wait() as c, sin_dfb.wait() as s:
                        with out_dfb.reserve() as o:
                            o.store(x1 * c + x2 * s)
                        with out_dfb.reserve() as o:
                            o.store(ttl.math.neg(x1) * s + x2 * c)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < seq_tiles:
                for j in range(HALF_TILES):
                    with x1_dfb.reserve() as b1, x2_dfb.reserve() as b2, cos_dfb.reserve() as b3, sin_dfb.reserve() as b4:
                        tx1 = ttl.copy(x[t, j], b1)
                        tx2 = ttl.copy(x[t, j + HALF_TILES], b2)
                        tx3 = ttl.copy(cos[0, j], b3)
                        tx4 = ttl.copy(sin[0, j], b4)
                        tx1.wait(); tx2.wait(); tx3.wait(); tx4.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < seq_tiles:
                for j in range(HALF_TILES):
                    with out_dfb.wait() as b1, out_dfb.wait() as b2:
                        tx1 = ttl.copy(b1, out[t, j])
                        tx2 = ttl.copy(b2, out[t, j + HALF_TILES])
                        tx1.wait(); tx2.wait()


# -- Residual add: out = x + y --
@ttl.operation(grid="auto")
def residual_add_kernel(x, y, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with x_dfb.wait() as xv, y_dfb.wait() as yv, out_dfb.reserve() as o:
                    o.store(xv + yv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as b1, y_dfb.reserve() as b2:
                    tx1 = ttl.copy(x[row, col], b1)
                    tx2 = ttl.copy(y[row, col], b2)
                    tx1.wait(); tx2.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


# -- Reshape: interleaved heads -> batched heads --
# Input: (TILE, n_head * HEAD_DIM) with heads interleaved in columns
# Output: (n_head * TILE, HEAD_DIM) with heads stacked in rows
@ttl.operation(grid=(SDPA_GRID_X, SDPA_GRID_Y))
def reshape_to_heads(inp, out):
    @ttl.compute()
    def compute():
        nx, ny = ttl.node(dims=2)
        h = ny * SDPA_GRID_X + nx
        for j in range(HEAD_TILES):
            with inp_dfb.wait() as blk, out_dfb.reserve() as o:
                o.store(blk)

    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def dm_read():
        nx, ny = ttl.node(dims=2)
        h = ny * SDPA_GRID_X + nx
        for j in range(HEAD_TILES):
            with inp_dfb.reserve() as blk:
                tx = ttl.copy(inp[0, h * HEAD_TILES + j], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        nx, ny = ttl.node(dims=2)
        h = ny * SDPA_GRID_X + nx
        for j in range(HEAD_TILES):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[h, j]); tx.wait()


# -- Reshape: batched heads -> interleaved heads (reverse) --
# Input: (n_head * TILE, HEAD_DIM) with heads stacked in rows
# Output: (TILE, n_head * HEAD_DIM) with heads interleaved in columns
@ttl.operation(grid=(SDPA_GRID_X, SDPA_GRID_Y))
def reshape_from_heads(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        nx, ny = ttl.node(dims=2)
        h = ny * SDPA_GRID_X + nx
        for j in range(HEAD_TILES):
            with inp_dfb.wait() as blk, out_dfb.reserve() as o:
                o.store(blk)

    @ttl.datamovement()
    def dm_read():
        nx, ny = ttl.node(dims=2)
        h = ny * SDPA_GRID_X + nx
        for j in range(HEAD_TILES):
            with inp_dfb.reserve() as blk:
                tx = ttl.copy(inp[h, j], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        nx, ny = ttl.node(dims=2)
        h = ny * SDPA_GRID_X + nx
        for j in range(HEAD_TILES):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, h * HEAD_TILES + j]); tx.wait()


# -- Value embedding gated add: out = v + 3 * sigmoid(gate_logits) * ve_val --
# gate_logits: (TILE, n_embd) from expanded gate matmul
# ve_val: (TILE, n_embd) from embedding lookup (padded)
# v, out: (TILE, n_embd)
@ttl.operation(grid="auto")
def ve_gated_add_kernel(v, gate_logits, ve_val, three_tile, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    col_tiles = v.shape[1] // TILE
    tiles_per_core = -(-col_tiles // grid_cols)

    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, 1), buffer_factor=2)
    g_dfb = ttl.make_dataflow_buffer_like(gate_logits, shape=(1, 1), buffer_factor=2)
    ve_dfb = ttl.make_dataflow_buffer_like(ve_val, shape=(1, 1), buffer_factor=2)
    thr_dfb = ttl.make_dataflow_buffer_like(three_tile, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        with thr_dfb.wait() as thr:
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < col_tiles:
                    with v_dfb.wait() as vb, g_dfb.wait() as gb, ve_dfb.wait() as veb, out_dfb.reserve() as o:
                        o.store(vb + thr * ttl.math.sigmoid(gb) * veb)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        with thr_dfb.reserve() as blk:
            tx = ttl.copy(three_tile[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < col_tiles:
                with v_dfb.reserve() as b1, g_dfb.reserve() as b2, ve_dfb.reserve() as b3:
                    tx1 = ttl.copy(v[0, t], b1)
                    tx2 = ttl.copy(gate_logits[0, t], b2)
                    tx3 = ttl.copy(ve_val[0, t], b3)
                    tx1.wait(); tx2.wait(); tx3.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < col_tiles:
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[0, t]); tx.wait()


# -- Device-side tensor copy --
@ttl.operation(grid="auto")
def copy_kernel(inp, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = inp.shape[0] // TILE
    col_tiles = inp.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with inp_dfb.wait() as blk, out_dfb.reserve() as o:
                    o.store(blk)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with inp_dfb.reserve() as blk:
                    tx = ttl.copy(inp[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


# -- Scaled residual: out = lr * x + l0 * x0 --
@ttl.operation(grid="auto")
def scaled_residual_kernel(x, x0, lambda_r_tile, lambda_0_tile, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    x0_dfb = ttl.make_dataflow_buffer_like(x0, shape=(1, 1), buffer_factor=2)
    lr_dfb = ttl.make_dataflow_buffer_like(lambda_r_tile, shape=(1, 1), buffer_factor=1)
    l0_dfb = ttl.make_dataflow_buffer_like(lambda_0_tile, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        with lr_dfb.wait() as lr, l0_dfb.wait() as l0:
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    with x_dfb.wait() as xv, x0_dfb.wait() as x0v, out_dfb.reserve() as o:
                        o.store(lr * xv + l0 * x0v)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        with lr_dfb.reserve() as b1, l0_dfb.reserve() as b2:
            tx1 = ttl.copy(lambda_r_tile[0, 0], b1)
            tx2 = ttl.copy(lambda_0_tile[0, 0], b2)
            tx1.wait(); tx2.wait()
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as b1, x0_dfb.reserve() as b2:
                    tx1 = ttl.copy(x[row, col], b1)
                    tx2 = ttl.copy(x0[row, col], b2)
                    tx1.wait(); tx2.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


# -- Softcap: out = cap * tanh(x / cap) --
@ttl.operation(grid="auto")
def softcap_kernel(x, inv_cap_tile, cap_tile, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    ic_dfb = ttl.make_dataflow_buffer_like(inv_cap_tile, shape=(1, 1), buffer_factor=1)
    c_dfb = ttl.make_dataflow_buffer_like(cap_tile, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        with ic_dfb.wait() as ic, c_dfb.wait() as cap:
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    with x_dfb.wait() as xv, out_dfb.reserve() as o:
                        o.store(cap * ttl.math.tanh(xv * ic))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        with ic_dfb.reserve() as b1, c_dfb.reserve() as b2:
            tx1 = ttl.copy(inv_cap_tile[0, 0], b1)
            tx2 = ttl.copy(cap_tile[0, 0], b2)
            tx1.wait(); tx2.wait()
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


# -- Flash Attention: online softmax, streams KV chunks through L1 --
KV_CHUNK = 1  # tiles per KV chunk


@ttl.operation(grid=(SDPA_GRID_X, SDPA_GRID_Y))
def flash_attention(Q_all, K_all, V_all, scale_tile, scaler, neg_inf_tile,
                    zero_tile, zero_head, mask, out):
    n_heads = Q_all.shape[0] // TILE
    skv = K_all.shape[0] // TILE // n_heads
    n_chunks = skv // KV_CHUNK

    q_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, HEAD_TILES), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    ninf_dfb = ttl.make_dataflow_buffer_like(neg_inf_tile, shape=(1, 1), buffer_factor=2)
    zero_dfb = ttl.make_dataflow_buffer_like(zero_tile, shape=(1, 1), buffer_factor=2)
    zero_head_dfb = ttl.make_dataflow_buffer_like(zero_head, shape=(1, HEAD_TILES), buffer_factor=2)

    k_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(KV_CHUNK, HEAD_TILES), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(V_all, shape=(KV_CHUNK, HEAD_TILES), buffer_factor=2)
    mask_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, KV_CHUNK), buffer_factor=2)

    kt_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(HEAD_TILES, KV_CHUNK), buffer_factor=2)
    qk_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, KV_CHUNK), buffer_factor=2)
    scaled_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, KV_CHUNK), buffer_factor=2)
    cm_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    m_new_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    m_new_bc_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, KV_CHUNK), buffer_factor=2)
    alpha_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    alpha_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)
    exp_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, KV_CHUNK), buffer_factor=2)
    cs_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    corrected_o_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)

    m_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    l_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    o_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)

    l_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        nx, ny = ttl.node(dims=2)
        h = ny * SDPA_GRID_X + nx
        q_blk = q_dfb.wait()
        sc_blk = sc_dfb.wait()
        scaler_blk = scaler_dfb.wait()

        with ninf_dfb.wait() as ni, m_dfb.reserve() as m_init:
            m_init.store(ni)
        with zero_dfb.wait() as z, l_dfb.reserve() as l_init:
            l_init.store(z)
        with zero_head_dfb.wait() as zh, o_dfb.reserve() as o_init:
            o_init.store(zh)

        for c in range(n_chunks):
            with k_dfb.wait() as kc, kt_dfb.reserve() as kt:
                kt.store(ttl.transpose(kc))
            with kt_dfb.wait() as ktv, qk_dfb.reserve() as qk:
                qk.store(q_blk @ ktv)
            with qk_dfb.wait() as qkv, mask_dfb.wait() as mv:
                with scaled_dfb.reserve() as scd:
                    scd.store(sc_blk * qkv + mv)

            with scaled_dfb.wait() as sd:
                with cm_dfb.reserve() as cm:
                    cm.store(ttl.math.reduce_max(sd, scaler_blk, dims=[1]))
                with m_dfb.wait() as m_old:
                    with cm_dfb.wait() as cm:
                        with m_new_dfb.reserve() as mn:
                            mn.store(ttl.math.max(m_old, cm))
                    with m_new_dfb.wait() as mn:
                        with alpha_dfb.reserve() as alpha:
                            alpha.store(ttl.math.exp(m_old - mn))
                        with m_new_bc_dfb.reserve() as mnb:
                            mnb.store(ttl.math.broadcast(mn, mnb, dims=[1]))
                        with m_dfb.reserve() as m_next:
                            m_next.store(mn)
                with m_new_bc_dfb.wait() as mnb:
                    with exp_dfb.reserve() as ex:
                        ex.store(ttl.math.exp(sd - mnb))

            with exp_dfb.wait() as exp_blk:
                with cs_dfb.reserve() as cs:
                    cs.store(ttl.math.reduce_sum(exp_blk, scaler_blk, dims=[1]))
                with alpha_dfb.wait() as alpha_blk:
                    with l_dfb.wait() as l_old, cs_dfb.wait() as cs:
                        with l_dfb.reserve() as l_new:
                            l_new.store(alpha_blk * l_old + cs)
                    with alpha_bc_dfb.reserve() as abc:
                        abc.store(ttl.math.broadcast(alpha_blk, abc, dims=[1]))
                with alpha_bc_dfb.wait() as abc, o_dfb.wait() as o_old:
                    with corrected_o_dfb.reserve() as co:
                        co.store(abc * o_old)
                with corrected_o_dfb.wait() as co, v_dfb.wait() as vc:
                    with o_dfb.reserve() as o_new:
                        o_new.store(co + exp_blk @ vc)

        with l_dfb.wait() as l_final, l_bc_dfb.reserve() as lbc:
            lbc.store(ttl.math.broadcast(l_final, lbc, dims=[1]))
        with o_dfb.wait() as o_final, l_bc_dfb.wait() as lbc:
            with out_dfb.reserve() as o:
                o.store(o_final / lbc)

        q_blk.pop()
        sc_blk.pop()
        scaler_blk.pop()

    @ttl.datamovement()
    def dm_read():
        nx, ny = ttl.node(dims=2)
        h = ny * SDPA_GRID_X + nx
        kv_off = h * skv
        with q_dfb.reserve() as b1, sc_dfb.reserve() as b2, scaler_dfb.reserve() as b3, ninf_dfb.reserve() as b4, zero_dfb.reserve() as b5, zero_head_dfb.reserve() as b6:
            tx1 = ttl.copy(Q_all[h:h + 1, 0:HEAD_TILES], b1)
            tx2 = ttl.copy(scale_tile[0, 0], b2)
            tx3 = ttl.copy(scaler[0, 0], b3)
            tx4 = ttl.copy(neg_inf_tile[0, 0], b4)
            tx5 = ttl.copy(zero_tile[0, 0], b5)
            tx6 = ttl.copy(zero_head[0, 0:HEAD_TILES], b6)
            tx1.wait(); tx2.wait(); tx3.wait(); tx4.wait(); tx5.wait(); tx6.wait()
        for c in range(n_chunks):
            with k_dfb.reserve() as b1, v_dfb.reserve() as b2, mask_dfb.reserve() as b3:
                tx1 = ttl.copy(K_all[kv_off + c * KV_CHUNK:kv_off + (c + 1) * KV_CHUNK, 0:HEAD_TILES], b1)
                tx2 = ttl.copy(V_all[kv_off + c * KV_CHUNK:kv_off + (c + 1) * KV_CHUNK, 0:HEAD_TILES], b2)
                tx3 = ttl.copy(mask[0, c * KV_CHUNK:(c + 1) * KV_CHUNK], b3)
                tx1.wait(); tx2.wait(); tx3.wait()

    @ttl.datamovement()
    def dm_write():
        nx, ny = ttl.node(dims=2)
        h = ny * SDPA_GRID_X + nx
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[h:h + 1, 0:HEAD_TILES]); tx.wait()


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------
def load_checkpoint(checkpoint_dir, step):
    """Load model state dict and metadata from nanochat checkpoint."""
    import json
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    model_config = meta["model_config"]
    if "window_pattern" not in model_config:
        model_config["window_pattern"] = "L"

    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    print(f"Loading {model_path}...")
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    n_layer = model_config["n_layer"]
    if "resid_lambdas" not in state:
        state["resid_lambdas"] = torch.ones(n_layer)
    if "x0_lambdas" not in state:
        state["x0_lambdas"] = torch.zeros(n_layer)

    return state, model_config


def load_tokenizer(tokenizer_dir):
    """Load tiktoken tokenizer from pickle."""
    import tiktoken
    pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pkl_path, "rb") as f:
        enc = pickle.load(f)
    return enc


def precompute_rotary(seq_len, head_dim, base=100000):
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    return cos, sin


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class NanochatModel:
    def __init__(self, state, model_config, tokenizer, device):
        self.device = device
        self.n_layer = model_config["n_layer"]
        self.n_head = model_config["n_head"]
        self.n_kv_head = model_config["n_kv_head"]
        self.n_embd = model_config["n_embd"]
        self.vocab_size = model_config["vocab_size"]
        self.mlp_dim = 4 * self.n_embd
        self.tokenizer = tokenizer
        seq_len = model_config.get("sequence_len", 2048)
        self.max_seq = seq_len

        # RMSNorm kernel for this embedding width
        self.rmsnorm_embd = make_rmsnorm_kernel(self.n_embd)

        # Embeddings on device for ttnn.embedding lookup
        self.wte_tt = to_ttnn(state["transformer.wte.weight"].to(torch.bfloat16), device)
        print(f"  wte: {state['transformer.wte.weight'].shape} on device")

        # Rotary embeddings: precompute expanded table on device
        # Each position gets TILE identical rows so ttnn.slice gives a ready tile
        cos, sin = precompute_rotary(seq_len * 10, HEAD_DIM)
        cos_expanded = cos[:seq_len].unsqueeze(1).expand(-1, TILE, -1).reshape(seq_len * TILE, HALF_DIM)
        sin_expanded = sin[:seq_len].unsqueeze(1).expand(-1, TILE, -1).reshape(seq_len * TILE, HALF_DIM)
        self.cos_table_tt = to_ttnn(cos_expanded.contiguous(), device)
        self.sin_table_tt = to_ttnn(sin_expanded.contiguous(), device)
        print(f"  Rotary tables: ({seq_len * TILE}, {HALF_DIM}) on device")

        # Per-layer scalars -> constant tiles on device
        resid_lambdas = state["resid_lambdas"].float()
        x0_lambdas = state["x0_lambdas"].float()
        self.lr_tiles = []
        self.l0_tiles = []
        for i in range(self.n_layer):
            self.lr_tiles.append(to_ttnn(torch.full((TILE, TILE), resid_lambdas[i].item(), dtype=torch.bfloat16), device))
            self.l0_tiles.append(to_ttnn(torch.full((TILE, TILE), x0_lambdas[i].item(), dtype=torch.bfloat16), device))

        # Per-layer weights on device (transpose for x @ w layout)
        self.w_q = []
        self.w_k = []
        self.w_v = []
        self.w_proj = []
        self.w_fc = []
        self.w_mlp_proj = []
        self.ve_gate = []       # CPU, tiny
        self.value_embed = []   # CPU, large
        for i in range(self.n_layer):
            prefix = f"transformer.h.{i}"
            self.w_q.append(to_ttnn(state[f"{prefix}.attn.c_q.weight"].to(torch.bfloat16).t().contiguous(), device))
            self.w_k.append(to_ttnn(state[f"{prefix}.attn.c_k.weight"].to(torch.bfloat16).t().contiguous(), device))
            self.w_v.append(to_ttnn(state[f"{prefix}.attn.c_v.weight"].to(torch.bfloat16).t().contiguous(), device))
            self.w_proj.append(to_ttnn(state[f"{prefix}.attn.c_proj.weight"].to(torch.bfloat16).t().contiguous(), device))
            self.w_fc.append(to_ttnn(state[f"{prefix}.mlp.c_fc.weight"].to(torch.bfloat16).t().contiguous(), device))
            # MLP proj: (8192, 2048) has k_tiles=256 which exceeds L1.
            # Split into chunks of (2048, 2048) along K dimension.
            mlp_proj_full = state[f"{prefix}.mlp.c_proj.weight"].to(torch.bfloat16).t().contiguous()  # (8192, 2048)
            k_total = mlp_proj_full.shape[0]
            chunk_size = K_CHUNK_MAX * TILE  # 2048
            n_chunks = k_total // chunk_size
            mlp_proj_chunks = []
            for c in range(n_chunks):
                chunk = mlp_proj_full[c * chunk_size:(c + 1) * chunk_size, :]
                mlp_proj_chunks.append(to_ttnn(chunk, device))
            self.w_mlp_proj.append(mlp_proj_chunks)

            # Value embedding: expanded gate weights and embedding table on device
            ve_gate_key = f"{prefix}.attn.ve_gate.weight"
            ve_key = f"value_embeds.{i}.weight"
            if has_ve(i, self.n_layer) and ve_gate_key in state:
                # Expand gate weight: (n_kv_head, 12) -> (TILE, n_embd)
                # Each head's 128-column region gets the same gate weight
                gate_w = state[ve_gate_key].to(torch.bfloat16)  # (n_kv_head, 12)
                gate_t = gate_w.t()  # (12, n_kv_head)
                expanded = gate_t.repeat_interleave(HEAD_DIM, dim=1)  # (12, n_embd)
                padded_gate = torch.zeros(TILE, self.n_embd, dtype=torch.bfloat16)
                padded_gate[:gate_t.shape[0], :] = expanded
                self.ve_gate.append(to_ttnn(padded_gate, device))
                # VE table on device for ttnn.embedding lookup
                self.value_embed.append(to_ttnn(state[ve_key].to(torch.bfloat16), device))
            else:
                self.ve_gate.append(None)
                self.value_embed.append(None)

            if i % 8 == 0:
                print(f"  Loaded layer {i} weights to device")

        # LM head: (out_features, in_features) -> transpose to (in, out), crop to vocab_size
        lm_head_raw = state["lm_head.weight"].to(torch.bfloat16)
        lm_head = lm_head_raw[:self.vocab_size, :].t().contiguous()
        lm_head = pad_vocab_to_tile(lm_head)
        self.lm_head = to_ttnn(lm_head, device)
        self.lm_head_cols = lm_head.shape[1]
        print(f"  Loaded lm_head: {lm_head.shape}")

        # Constant tiles in L1 for fast access
        self.scaler_tt = to_ttnn_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
        self.ms_embd_tt = to_ttnn_l1(torch.full((TILE, TILE), 1.0 / self.n_embd, dtype=torch.bfloat16), device)
        self.ms_head_tt = to_ttnn_l1(torch.full((TILE, TILE), 1.0 / HEAD_DIM, dtype=torch.bfloat16), device)
        self.scale_tt = to_ttnn_l1(torch.full((TILE, TILE), SDPA_SCALE_VAL, dtype=torch.bfloat16), device)
        self.inv_cap_tt = to_ttnn_l1(torch.full((TILE, TILE), 1.0 / SOFTCAP, dtype=torch.bfloat16), device)
        self.cap_tt = to_ttnn_l1(torch.full((TILE, TILE), SOFTCAP, dtype=torch.bfloat16), device)
        self.ninf_tt = to_ttnn_l1(torch.full((TILE, TILE), -10000.0, dtype=torch.bfloat16), device)
        self.zero_tt = to_ttnn_l1(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        self.zero_head_tt = to_ttnn_l1(torch.zeros(TILE, HEAD_DIM, dtype=torch.bfloat16), device)
        self.three_tt = to_ttnn_l1(torch.full((TILE, TILE), 3.0, dtype=torch.bfloat16), device)

        # Precomputed SDPA masks: (max_seq, TILE, padded_seq) on device
        padded_seq = pad_to_tile(self.max_seq)
        self.padded_seq = padded_seq
        all_masks = torch.full((self.max_seq, TILE, padded_seq), -10000.0, dtype=torch.bfloat16)
        for p in range(self.max_seq):
            all_masks[p, :, :p + 1] = 0.0
        self.all_masks_tt = to_ttnn(all_masks.contiguous(), device)
        mask_mb = self.max_seq * TILE * padded_seq * 2 / 1e6
        print(f"  SDPA masks: ({self.max_seq}, {TILE}, {padded_seq}) = {mask_mb:.0f}MB on device")

        # KV cache on device using ttnn.update_cache
        # Shape: (1, n_head, max_seq, HEAD_DIM) per layer -- standard TTNN KV cache layout
        padded_seq = pad_to_tile(self.max_seq)
        self.padded_seq = padded_seq
        self.k_cache = []
        self.v_cache = []
        for i in range(self.n_layer):
            self.k_cache.append(to_ttnn(
                torch.zeros(1, self.n_head, padded_seq, HEAD_DIM, dtype=torch.bfloat16), device))
            self.v_cache.append(to_ttnn(
                torch.zeros(1, self.n_head, padded_seq, HEAD_DIM, dtype=torch.bfloat16), device))
        cache_mb = self.n_layer * 2 * self.n_head * padded_seq * HEAD_DIM * 2 / 1e6
        print(f"  KV cache: {self.n_layer} layers x 2 x (1, {self.n_head}, {padded_seq}, {HEAD_DIM}) = {cache_mb:.0f}MB")

        # Pre-allocated scratch tensors in L1 (reused every decode step)
        # (32, 2048) = 128KB each, (512, 128) = 128KB each, (32, 8192) = 512KB each
        n_embd = self.n_embd
        n_head = self.n_head
        mlp_dim = self.mlp_dim
        self.x_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.normed_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.x0_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.temp_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.normed2_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.q_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.k_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.v_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.y_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.hidden_tt = to_ttnn_l1(torch.zeros(TILE, mlp_dim, dtype=torch.bfloat16), device)
        self.hidden_act_tt = to_ttnn_l1(torch.zeros(TILE, mlp_dim, dtype=torch.bfloat16), device)
        self.mlp_out_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.buf_a_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.partial_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.gate_logits_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.ve_val_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)

        # Batched head tensors in L1 (n_head * TILE, HEAD_DIM) = 128KB each
        self.q_batch_tt = to_ttnn_l1(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        self.k_batch_tt = to_ttnn_l1(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        self.v_batch_tt = to_ttnn_l1(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        self.q_rot_tt = to_ttnn_l1(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        self.k_rot_tt = to_ttnn_l1(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        self.q_norm_tt = to_ttnn_l1(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        self.k_norm_tt = to_ttnn_l1(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        self.sdpa_out_tt = to_ttnn_l1(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        self.attn_concat_tt = to_ttnn_l1(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)

        # LM head scratch (logits are large: 32*65536=4MB, keep on DRAM)
        lm_cols = self.lm_head_cols
        self.logits_tt = to_ttnn(torch.zeros(TILE, lm_cols, dtype=torch.bfloat16), device)
        self.logits_cap_tt = to_ttnn(torch.zeros(TILE, lm_cols, dtype=torch.bfloat16), device)

        print("  Model ready")

    def reset_cache(self):
        padded_seq = self.padded_seq
        for layer in range(self.n_layer):
            self.k_cache[layer] = to_ttnn(
                torch.zeros(1, self.n_head, padded_seq, HEAD_DIM, dtype=torch.bfloat16), self.device)
            self.v_cache[layer] = to_ttnn(
                torch.zeros(1, self.n_head, padded_seq, HEAD_DIM, dtype=torch.bfloat16), self.device)

    def encode(self, text):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available")
        bos = self.tokenizer.encode_single_token("<|bos|>")
        user_start = self.tokenizer.encode_single_token("<|user_start|>")
        user_end = self.tokenizer.encode_single_token("<|user_end|>")
        assistant_start = self.tokenizer.encode_single_token("<|assistant_start|>")
        ids = [bos, user_start] + self.tokenizer.encode_ordinary(text) + [user_end, assistant_start]
        return ids

    def decode(self, ids):
        if self.tokenizer is None:
            return str(ids)
        return self.tokenizer.decode(ids)

    def decode_step(self, token_id, pos):
        """Run one decode step: single token -> logits over vocab.

        token_id: int, the current token
        pos: int, position in the sequence (0-indexed)
        Returns: logits as (vocab_size,) CPU float tensor
        """
        device = self.device
        n_embd = self.n_embd
        n_head = self.n_head
        mlp_dim = self.mlp_dim

        # Use pre-allocated scratch tensors
        x_tt = self.x_tt
        normed_tt = self.normed_tt
        x0_tt = self.x0_tt
        temp_tt = self.temp_tt
        normed2_tt = self.normed2_tt
        q_tt = self.q_tt
        k_tt = self.k_tt
        v_tt = self.v_tt
        y_tt = self.y_tt
        hidden_tt = self.hidden_tt
        hidden_act_tt = self.hidden_act_tt
        buf_a_tt = self.buf_a_tt
        q_batch_tt = self.q_batch_tt
        k_batch_tt = self.k_batch_tt
        v_batch_tt = self.v_batch_tt
        q_rot_tt = self.q_rot_tt
        k_rot_tt = self.k_rot_tt
        q_norm_tt = self.q_norm_tt
        k_norm_tt = self.k_norm_tt
        sdpa_out_tt = self.sdpa_out_tt
        attn_concat_tt = self.attn_concat_tt

        # 1. Token ID -> device, embedding lookup on device
        token_id_tt = ttnn.from_torch(
            torch.tensor([[token_id]], dtype=torch.uint32),
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wte_raw = ttnn.embedding(token_id_tt, self.wte_tt)  # (1, 1, n_embd) TILE
        # Pad to (1, TILE, n_embd), then reshape to (TILE, n_embd)
        wte_padded = ttnn.pad(wte_raw, padding=((0, 0), (0, TILE - 1), (0, 0)), value=0.0)
        x_tt = ttnn.reshape(wte_padded, (TILE, n_embd))
        self.rmsnorm_embd(x_tt, self.scaler_tt, self.ms_embd_tt, normed_tt)
        # x0 = normed (copy on device), x = normed (copy on device)
        copy_kernel(normed_tt, x0_tt)
        copy_kernel(normed_tt, x_tt)

        # Rotary: slice from precomputed table (rotary kernel broadcasts internally)
        rot_start = pos * TILE
        cos_tt = ttnn.slice(self.cos_table_tt, [rot_start, 0], [rot_start + TILE, HALF_DIM])
        sin_tt = ttnn.slice(self.sin_table_tt, [rot_start, 0], [rot_start + TILE, HALF_DIM])

        # SDPA mask: slice from precomputed masks on device
        padded_seq = self.padded_seq
        mask_3d = ttnn.slice(self.all_masks_tt, [pos, 0, 0], [pos + 1, TILE, padded_seq])
        mask_tt = ttnn.reshape(mask_3d, (TILE, padded_seq))

        for layer_idx in range(self.n_layer):
            # -- Scaled residual: temp = lr*x + l0*x0 --
            scaled_residual_kernel(x_tt, x0_tt,
                                   self.lr_tiles[layer_idx], self.l0_tiles[layer_idx],
                                   temp_tt)

            # -- Pre-attention norm --
            self.rmsnorm_embd(temp_tt, self.scaler_tt, self.ms_embd_tt, normed_tt)

            # -- QKV projections (fused: reads normed once) --
            triple_linear_kernel(normed_tt,
                                 self.w_q[layer_idx], self.w_k[layer_idx], self.w_v[layer_idx],
                                 q_tt, k_tt, v_tt)

            # -- Reshape Q, K, V to head-batched on device --
            reshape_to_heads(q_tt, q_batch_tt)
            reshape_to_heads(k_tt, k_batch_tt)

            # -- Value embedding (all on device) --
            if self.value_embed[layer_idx] is not None:
                # Gate: normed_tt[:, :32] @ expanded_gate_w -> (TILE, n_embd) gate logits
                linear_kernel_k1(normed_tt, self.ve_gate[layer_idx], self.gate_logits_tt)
                # VE lookup on device
                ve_raw = ttnn.embedding(token_id_tt, self.value_embed[layer_idx])
                ve_tiled = ttnn.to_layout(ve_raw, ttnn.TILE_LAYOUT)
                ve_2d = ttnn.reshape(ve_tiled, (TILE, n_embd))
                # v_out = v + 3 * sigmoid(gate_logits) * ve_val, then copy back
                ve_gated_add_kernel(v_tt, self.gate_logits_tt, ve_2d, self.three_tt, self.ve_val_tt)
                copy_kernel(self.ve_val_tt, v_tt)

            reshape_to_heads(v_tt, v_batch_tt)

            # -- Rotary (one call for all heads) --
            rotary_kernel(q_batch_tt, cos_tt, sin_tt, q_rot_tt)
            rotary_kernel(k_batch_tt, cos_tt, sin_tt, k_rot_tt)

            # -- QK norm (one call for all heads) --
            rmsnorm_head(q_rot_tt, self.scaler_tt, self.ms_head_tt, q_norm_tt)
            rmsnorm_head(k_rot_tt, self.scaler_tt, self.ms_head_tt, k_norm_tt)

            # -- KV cache update on device via ttnn.update_cache --
            k_norm_4d = ttnn.reshape(k_norm_tt, (1, n_head, TILE, HEAD_DIM))
            v_batch_4d = ttnn.reshape(v_batch_tt, (1, n_head, TILE, HEAD_DIM))
            ttnn.kv_cache.update_cache_for_token_(self.k_cache[layer_idx], k_norm_4d, pos, 0)
            ttnn.kv_cache.update_cache_for_token_(self.v_cache[layer_idx], v_batch_4d, pos, 0)

            # -- Flash attention (reshape cache to 2D for flash kernel) --
            k_cache_2d = ttnn.reshape(self.k_cache[layer_idx], (n_head * padded_seq, HEAD_DIM))
            v_cache_2d = ttnn.reshape(self.v_cache[layer_idx], (n_head * padded_seq, HEAD_DIM))
            flash_attention(q_norm_tt, k_cache_2d, v_cache_2d,
                           self.scale_tt, self.scaler_tt,
                           self.ninf_tt, self.zero_tt, self.zero_head_tt,
                           mask_tt, sdpa_out_tt)

            # -- Reshape SDPA output back to interleaved --
            reshape_from_heads(sdpa_out_tt, attn_concat_tt)

            # -- Output projection --
            linear_kernel(attn_concat_tt, self.w_proj[layer_idx], y_tt)

            # -- Residual: buf_a = temp + y --
            residual_add_kernel(temp_tt, y_tt, buf_a_tt)

            # -- Pre-MLP norm --
            self.rmsnorm_embd(buf_a_tt, self.scaler_tt, self.ms_embd_tt, normed2_tt)

            # -- MLP: fc -> relu^2 -> proj (chunked along K for proj) --
            linear_kernel(normed2_tt, self.w_fc[layer_idx], hidden_tt)
            relu_sq_kernel(hidden_tt, hidden_act_tt)
            # MLP proj: hidden_act(32, 8192) @ w_proj chunked along K dim
            # Each slice kernel reads a different K-range of hidden_act_tt
            # Accumulate using ping-pong between self.mlp_out_tt and self.partial_tt
            chunks = self.w_mlp_proj[layer_idx]
            acc_a, acc_b = self.mlp_out_tt, self.partial_tt
            mlp_proj_slice_kernels[0](hidden_act_tt, chunks[0], acc_a)
            for c_idx in range(1, len(chunks)):
                # Use y_tt as temp for slice output (free here, after output proj consumed it)
                mlp_proj_slice_kernels[c_idx](hidden_act_tt, chunks[c_idx], normed2_tt)
                residual_add_kernel(acc_a, normed2_tt, acc_b)
                acc_a, acc_b = acc_b, acc_a

            # -- Residual: x = buf_a + mlp_out --
            residual_add_kernel(buf_a_tt, acc_a, x_tt)

        # -- Final norm --
        self.rmsnorm_embd(x_tt, self.scaler_tt, self.ms_embd_tt, normed_tt)

        # -- LM head --
        linear_kernel(normed_tt, self.lm_head, self.logits_tt)

        # -- Softcap --
        softcap_kernel(self.logits_tt, self.inv_cap_tt, self.cap_tt, self.logits_cap_tt)

        # Return logits on device -- sampling handles readback
        return self.logits_cap_tt


def generate(model, prompt_tokens, max_tokens=64, temperature=0.8, top_k=50, seed=42):
    """Autoregressive generation with KV cache."""
    model.reset_cache()
    rng = torch.Generator()
    rng.manual_seed(seed)

    all_tokens = list(prompt_tokens)
    generated = []
    step_times = []

    for step in range(len(prompt_tokens) + max_tokens):
        if step < len(prompt_tokens):
            token_id = prompt_tokens[step]
        else:
            token_id = next_token

        pos = step
        t0 = time.time()
        logits_tt = model.decode_step(token_id, pos)
        dt = time.time() - t0
        step_times.append(dt)

        if step >= len(prompt_tokens) - 1:
            # Sample next token
            if temperature == 0:
                # Greedy: argmax on device, read back one int
                argmax_tt = ttnn.argmax(logits_tt, dim=-1)
                argmax_cpu = ttnn.to_torch(argmax_tt)
                next_token = argmax_cpu.flatten()[0].item()
            else:
                # Sampling: read back row 0 for CPU-side top-k + multinomial
                logits_row0 = ttnn.slice(logits_tt, [0, 0], [1, model.vocab_size])
                logits_cpu = ttnn.to_torch(logits_row0).float().squeeze(0)
                logits_scaled = logits_cpu / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits_scaled, min(top_k, logits_scaled.shape[0]))
                    logits_scaled[logits_scaled < v[-1]] = -float("inf")
                probs = torch.softmax(logits_scaled, dim=0)
                next_token = torch.multinomial(probs, num_samples=1, generator=rng).item()

            if step >= len(prompt_tokens) - 1:
                generated.append(next_token)

                # Stop on BOS or assistant_end
                try:
                    bos = model.tokenizer.encode_single_token("<|bos|>")
                    aend = model.tokenizer.encode_single_token("<|assistant_end|>")
                    if next_token in (bos, aend):
                        break
                except:
                    pass

            if len(generated) >= max_tokens:
                break

    return generated, step_times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanochat d32 TT-Lang inference")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.path.expanduser("~/.cache/nanochat/chatsft_checkpoints/d32"),
                        help="Checkpoint directory")
    parser.add_argument("--step", type=int, default=650, help="Checkpoint step")
    parser.add_argument("--tokenizer-dir", type=str,
                        default=None,
                        help="Tokenizer directory (default: same as checkpoint dir)")
    parser.add_argument("--prompt", type=str, default="Write a fibonacci function in Python",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Suppress noisy warnings from TT runtime
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    print("Opening TT device...")
    device = ttnn.open_device(device_id=0)

    print(f"Loading checkpoint from {args.checkpoint_dir} step {args.step}...")
    state, model_config = load_checkpoint(args.checkpoint_dir, args.step)
    print(f"  Config: {model_config}")

    print("Loading tokenizer...")
    tok_dir = args.tokenizer_dir if args.tokenizer_dir else args.checkpoint_dir
    tokenizer = load_tokenizer(tok_dir)

    print("Building model (loading weights to device)...")
    model = NanochatModel(state, model_config, tokenizer, device)
    del state  # free CPU memory

    print(f"\nPrompt: {args.prompt}")
    prompt_tokens = model.encode(args.prompt)
    print(f"Tokens: {prompt_tokens} ({len(prompt_tokens)} tokens)")

    print("Generating...", flush=True)
    generated, step_times = generate(model, prompt_tokens,
                                      max_tokens=args.max_tokens,
                                      temperature=args.temperature,
                                      top_k=args.top_k,
                                      seed=args.seed)

    output_text = model.decode(generated)
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {output_text}")
    print(f"{'='*60}")

    # Timing stats from per-step measurements
    n_prompt = len(prompt_tokens)
    prompt_times = step_times[:n_prompt]
    gen_times = step_times[n_prompt:]
    total_decode_time = sum(step_times)
    if gen_times:
        avg_gen_ms = sum(gen_times) / len(gen_times) * 1000
        gen_tok_s = len(gen_times) / sum(gen_times)
        # Skip first 2 steps (warmup/compilation) for steady-state measurement
        n_warmup = min(2, len(gen_times))
        steady_times = gen_times[n_warmup:]
        if steady_times:
            steady_tok_s = len(steady_times) / sum(steady_times)
            steady_ms = sum(steady_times) / len(steady_times) * 1000
        else:
            steady_tok_s = gen_tok_s
            steady_ms = avg_gen_ms
        print(f"Prompt: {n_prompt} tokens in {sum(prompt_times):.1f}s")
        print(f"Generation: {len(gen_times)} tokens, avg {avg_gen_ms:.1f}ms/tok ({gen_tok_s:.2f} tok/s)")
        print(f"Steady-state (excl {n_warmup} warmup): {steady_ms:.1f}ms/tok ({steady_tok_s:.2f} tok/s)")
        print(f"Total decode time: {total_decode_time:.1f}s")

    ttnn.close_device(device)
