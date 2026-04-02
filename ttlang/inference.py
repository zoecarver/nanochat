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
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
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
                    with a_dfb.reserve() as blk:
                        tx = ttl.copy(x[mi, 0:k_tiles], blk); tx.wait()
                    with w_dfb.reserve() as blk:
                        tx = ttl.copy(w[0:k_tiles, sc:sc + NCOLS], blk); tx.wait()

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

# Standard linear for nanochat projections (K=64 for 2048-wide inputs)
linear_kernel = make_linear_kernel(64)


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


# -- Rotary embeddings --
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
                    with x1_dfb.reserve() as blk:
                        tx = ttl.copy(x[t, j], blk); tx.wait()
                    with x2_dfb.reserve() as blk:
                        tx = ttl.copy(x[t, j + HALF_TILES], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos[t, j], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin[t, j], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < seq_tiles:
                for j in range(HALF_TILES):
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[t, j]); tx.wait()
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[t, j + HALF_TILES]); tx.wait()


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
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()
                with y_dfb.reserve() as blk:
                    tx = ttl.copy(y[row, col], blk); tx.wait()

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
        with lr_dfb.reserve() as blk:
            tx = ttl.copy(lambda_r_tile[0, 0], blk); tx.wait()
        with l0_dfb.reserve() as blk:
            tx = ttl.copy(lambda_0_tile[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()
                with x0_dfb.reserve() as blk:
                    tx = ttl.copy(x0[row, col], blk); tx.wait()

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
        with ic_dfb.reserve() as blk:
            tx = ttl.copy(inv_cap_tile[0, 0], blk); tx.wait()
        with c_dfb.reserve() as blk:
            tx = ttl.copy(cap_tile[0, 0], blk); tx.wait()
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
        with q_dfb.reserve() as blk:
            tx = ttl.copy(Q_all[h:h + 1, 0:HEAD_TILES], blk); tx.wait()
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scale_tile[0, 0], blk); tx.wait()
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ninf_dfb.reserve() as blk:
            tx = ttl.copy(neg_inf_tile[0, 0], blk); tx.wait()
        with zero_dfb.reserve() as blk:
            tx = ttl.copy(zero_tile[0, 0], blk); tx.wait()
        with zero_head_dfb.reserve() as blk:
            tx = ttl.copy(zero_head[0, 0:HEAD_TILES], blk); tx.wait()
        for c in range(n_chunks):
            with k_dfb.reserve() as blk:
                tx = ttl.copy(K_all[kv_off + c * KV_CHUNK:kv_off + (c + 1) * KV_CHUNK, 0:HEAD_TILES], blk); tx.wait()
            with v_dfb.reserve() as blk:
                tx = ttl.copy(V_all[kv_off + c * KV_CHUNK:kv_off + (c + 1) * KV_CHUNK, 0:HEAD_TILES], blk); tx.wait()
            with mask_dfb.reserve() as blk:
                tx = ttl.copy(mask[0, c * KV_CHUNK:(c + 1) * KV_CHUNK], blk); tx.wait()

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

        # Embeddings (stay on CPU for lookup)
        self.wte = state["transformer.wte.weight"].to(torch.bfloat16)
        print(f"  wte: {self.wte.shape}")

        # Rotary embeddings
        cos, sin = precompute_rotary(seq_len * 10, HEAD_DIM)
        self.rotary_cos = cos
        self.rotary_sin = sin

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

            # Value embedding gate and embeddings (CPU)
            ve_gate_key = f"{prefix}.attn.ve_gate.weight"
            ve_key = f"value_embeds.{i}.weight"
            if has_ve(i, self.n_layer) and ve_gate_key in state:
                self.ve_gate.append(state[ve_gate_key].to(torch.bfloat16))
                self.value_embed.append(state[ve_key].to(torch.bfloat16))
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

        # Constant tiles on device
        self.scaler_tt = to_ttnn(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
        self.ms_embd_tt = to_ttnn(torch.full((TILE, TILE), 1.0 / self.n_embd, dtype=torch.bfloat16), device)
        self.ms_head_tt = to_ttnn(torch.full((TILE, TILE), 1.0 / HEAD_DIM, dtype=torch.bfloat16), device)
        self.scale_tt = to_ttnn(torch.full((TILE, TILE), SDPA_SCALE_VAL, dtype=torch.bfloat16), device)
        self.inv_cap_tt = to_ttnn(torch.full((TILE, TILE), 1.0 / SOFTCAP, dtype=torch.bfloat16), device)
        self.cap_tt = to_ttnn(torch.full((TILE, TILE), SOFTCAP, dtype=torch.bfloat16), device)
        self.ninf_tt = to_ttnn_l1(torch.full((TILE, TILE), -10000.0, dtype=torch.bfloat16), device)
        self.zero_tt = to_ttnn_l1(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        self.zero_head_tt = to_ttnn_l1(torch.zeros(TILE, HEAD_DIM, dtype=torch.bfloat16), device)

        # KV cache: per layer, per head, on CPU
        self.k_cache = [[torch.zeros(self.max_seq, HEAD_DIM, dtype=torch.bfloat16)
                         for _ in range(self.n_head)] for _ in range(self.n_layer)]
        self.v_cache = [[torch.zeros(self.max_seq, HEAD_DIM, dtype=torch.bfloat16)
                         for _ in range(self.n_head)] for _ in range(self.n_layer)]
        print("  Model ready")

    def reset_cache(self):
        for layer in range(self.n_layer):
            for head in range(self.n_head):
                self.k_cache[layer][head].zero_()
                self.v_cache[layer][head].zero_()

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

        # 1. Embedding lookup (CPU) -> pad to (32, n_embd)
        x_cpu = torch.zeros(TILE, n_embd, dtype=torch.bfloat16)
        x_cpu[0] = self.wte[token_id]

        # Upload and norm
        x_tt = to_ttnn(x_cpu, device)
        normed_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        self.rmsnorm_embd(x_tt, self.scaler_tt, self.ms_embd_tt, normed_tt)
        # x0 = normed embedding (saved for residual blending)
        x0_tt = normed_tt
        # x starts as normed embedding
        x_tt = to_ttnn(ttnn.to_torch(normed_tt), device)  # copy for separate buffer

        # Pre-allocate working tensors
        temp_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        normed2_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        q_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        k_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        v_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        y_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        hidden_tt = to_ttnn(torch.zeros(TILE, mlp_dim, dtype=torch.bfloat16), device)
        hidden_act_tt = to_ttnn(torch.zeros(TILE, mlp_dim, dtype=torch.bfloat16), device)
        mlp_out_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
        buf_a_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)

        # Rotary: cos/sin for this position, replicated for all heads
        cos_row = self.rotary_cos[pos]  # (half_dim,) = (64,)
        sin_row = self.rotary_sin[pos]
        cos_batched = cos_row.unsqueeze(0).expand(TILE * n_head, -1).contiguous()
        sin_batched = sin_row.unsqueeze(0).expand(TILE * n_head, -1).contiguous()
        cos_tt = to_ttnn(cos_batched, device)
        sin_tt = to_ttnn(sin_batched, device)

        # Batched head working tensors
        q_rot_tt = to_ttnn(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        k_rot_tt = to_ttnn(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        q_norm_tt = to_ttnn(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
        k_norm_tt = to_ttnn(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)

        # SDPA mask: (32, padded_cache_len)
        cache_len = pos + 1
        padded_cache = pad_to_tile(cache_len)

        mask_cpu = torch.zeros(TILE, padded_cache, dtype=torch.bfloat16)
        if padded_cache > cache_len:
            mask_cpu[:, cache_len:] = -10000.0
        mask_tt = to_ttnn(mask_cpu, device)

        # Multi-head SDPA output (all heads batched)
        sdpa_out_tt = to_ttnn(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)

        for layer_idx in range(self.n_layer):
            # -- Scaled residual: temp = lr*x + l0*x0 --
            scaled_residual_kernel(x_tt, x0_tt,
                                   self.lr_tiles[layer_idx], self.l0_tiles[layer_idx],
                                   temp_tt)

            # -- Pre-attention norm --
            self.rmsnorm_embd(temp_tt, self.scaler_tt, self.ms_embd_tt, normed_tt)

            # -- QKV projections --
            linear_kernel(normed_tt, self.w_q[layer_idx], q_tt)
            linear_kernel(normed_tt, self.w_k[layer_idx], k_tt)
            linear_kernel(normed_tt, self.w_v[layer_idx], v_tt)

            # -- Read back to CPU for head processing --
            q_cpu = ttnn.to_torch(q_tt)    # (32, 2048)
            k_cpu = ttnn.to_torch(k_tt)
            v_cpu = ttnn.to_torch(v_tt)

            # -- Value embedding (CPU) --
            if self.value_embed[layer_idx] is not None:
                normed_cpu = ttnn.to_torch(normed_tt)
                gate_in = normed_cpu[0:1, :12].float()  # (1, 12)
                ve_gate_w = self.ve_gate[layer_idx].float()  # (n_kv_head, 12)
                gate = 3.0 * torch.sigmoid(gate_in @ ve_gate_w.t())  # (1, n_kv_head)
                ve_val = self.value_embed[layer_idx][token_id]  # (kv_dim,) bf16
                for h in range(n_head):
                    s = h * HEAD_DIM
                    e = s + HEAD_DIM
                    v_cpu[0, s:e] += gate[0, h].to(torch.bfloat16) * ve_val[s:e]

            # -- Reshape to head-batched --
            q_batch = q_cpu.view(TILE, n_head, HEAD_DIM).permute(1, 0, 2).contiguous().view(TILE * n_head, HEAD_DIM)
            k_batch = k_cpu.view(TILE, n_head, HEAD_DIM).permute(1, 0, 2).contiguous().view(TILE * n_head, HEAD_DIM)

            # Upload batched tensors
            q_batch_tt = to_ttnn(q_batch, device)
            k_batch_tt = to_ttnn(k_batch, device)

            # -- Rotary (one call for all heads) --
            rotary_kernel(q_batch_tt, cos_tt, sin_tt, q_rot_tt)
            rotary_kernel(k_batch_tt, cos_tt, sin_tt, k_rot_tt)

            # -- QK norm (one call for all heads) --
            rmsnorm_head(q_rot_tt, self.scaler_tt, self.ms_head_tt, q_norm_tt)
            rmsnorm_head(k_rot_tt, self.scaler_tt, self.ms_head_tt, k_norm_tt)

            # -- Read back normed Q, K and split into heads --
            q_normed_cpu = ttnn.to_torch(q_norm_tt)  # (n_head*32, 128)
            k_normed_cpu = ttnn.to_torch(k_norm_tt)
            q_heads = q_normed_cpu.view(n_head, TILE, HEAD_DIM)
            k_heads = k_normed_cpu.view(n_head, TILE, HEAD_DIM)

            # V heads (already on CPU, not batched through rotary/norm)
            v_heads = v_cpu.view(TILE, n_head, HEAD_DIM).permute(1, 0, 2).contiguous()

            # -- Update KV cache and build batched K/V for all heads --
            k_all_list = []
            v_all_list = []
            for h in range(n_head):
                self.k_cache[layer_idx][h][pos] = k_heads[h, 0]
                self.v_cache[layer_idx][h][pos] = v_heads[h, 0]

                k_cached = torch.zeros(padded_cache, HEAD_DIM, dtype=torch.bfloat16)
                k_cached[:cache_len] = self.k_cache[layer_idx][h][:cache_len]
                v_cached = torch.zeros(padded_cache, HEAD_DIM, dtype=torch.bfloat16)
                v_cached[:cache_len] = self.v_cache[layer_idx][h][:cache_len]
                k_all_list.append(k_cached)
                v_all_list.append(v_cached)

            # Stack heads vertically: (n_head*padded_cache, 128), (n_head*32, 128)
            q_all = q_normed_cpu  # already (n_head*32, 128)
            k_all = torch.cat(k_all_list, dim=0)
            v_all = torch.cat(v_all_list, dim=0)

            # Single multi-head SDPA call (16 cores in parallel)
            q_all_tt = to_ttnn(q_all, device)
            k_all_tt = to_ttnn(k_all, device)
            v_all_tt = to_ttnn(v_all, device)
            sdpa_out_tt = to_ttnn(torch.zeros(TILE * n_head, HEAD_DIM, dtype=torch.bfloat16), device)
            flash_attention(q_all_tt, k_all_tt, v_all_tt,
                           self.scale_tt, self.scaler_tt,
                           self.ninf_tt, self.zero_tt, self.zero_head_tt,
                           mask_tt, sdpa_out_tt)

            # Reshape: (n_head*32, 128) -> (32, n_head, 128) -> (32, n_embd)
            sdpa_result = ttnn.to_torch(sdpa_out_tt).view(n_head, TILE, HEAD_DIM)
            attn_concat = sdpa_result.permute(1, 0, 2).contiguous().view(TILE, n_embd)

            # -- Output projection --
            attn_tt = to_ttnn(attn_concat, device)
            linear_kernel(attn_tt, self.w_proj[layer_idx], y_tt)

            # -- Residual: buf_a = temp + y --
            residual_add_kernel(temp_tt, y_tt, buf_a_tt)

            # -- Pre-MLP norm --
            self.rmsnorm_embd(buf_a_tt, self.scaler_tt, self.ms_embd_tt, normed2_tt)

            # -- MLP: fc -> relu^2 -> proj (chunked along K for proj) --
            linear_kernel(normed2_tt, self.w_fc[layer_idx], hidden_tt)
            relu_sq_kernel(hidden_tt, hidden_act_tt)
            # MLP proj is chunked: hidden_act(32, 8192) split into 4x(32, 2048) chunks
            hidden_act_cpu = ttnn.to_torch(hidden_act_tt)  # (32, 8192)
            chunk_size = K_CHUNK_MAX * TILE  # 2048
            chunks = self.w_mlp_proj[layer_idx]
            partial_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
            mlp_out_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
            for c_idx, w_chunk in enumerate(chunks):
                x_chunk = hidden_act_cpu[:, c_idx * chunk_size:(c_idx + 1) * chunk_size].contiguous()
                x_chunk_tt = to_ttnn(x_chunk, device)
                linear_kernel(x_chunk_tt, w_chunk, partial_tt)
                if c_idx == 0:
                    mlp_out_tt = to_ttnn(ttnn.to_torch(partial_tt), device)
                else:
                    new_mlp_tt = to_ttnn(torch.zeros(TILE, n_embd, dtype=torch.bfloat16), device)
                    residual_add_kernel(mlp_out_tt, partial_tt, new_mlp_tt)
                    mlp_out_tt = new_mlp_tt

            # -- Residual: x = buf_a + mlp_out --
            residual_add_kernel(buf_a_tt, mlp_out_tt, x_tt)

        # -- Final norm --
        self.rmsnorm_embd(x_tt, self.scaler_tt, self.ms_embd_tt, normed_tt)

        # -- LM head --
        lm_cols = self.lm_head_cols
        logits_tt = to_ttnn(torch.zeros(TILE, lm_cols, dtype=torch.bfloat16), device)
        linear_kernel(normed_tt, self.lm_head, logits_tt)

        # -- Softcap --
        logits_cap_tt = to_ttnn(torch.zeros(TILE, lm_cols, dtype=torch.bfloat16), device)
        softcap_kernel(logits_tt, self.inv_cap_tt, self.cap_tt, logits_cap_tt)

        # -- Read back logits (only row 0, crop to vocab_size) --
        logits_cpu = ttnn.to_torch(logits_cap_tt)  # (32, padded_vocab)
        logits = logits_cpu[0, :self.vocab_size].float()  # (vocab_size,) float32

        return logits


def generate(model, prompt_tokens, max_tokens=64, temperature=0.8, top_k=50, seed=42):
    """Autoregressive generation with KV cache."""
    model.reset_cache()
    rng = torch.Generator()
    rng.manual_seed(seed)

    all_tokens = list(prompt_tokens)
    generated = []

    for step in range(len(prompt_tokens) + max_tokens):
        if step < len(prompt_tokens):
            token_id = prompt_tokens[step]
        else:
            token_id = next_token

        pos = step
        t0 = time.time()
        logits = model.decode_step(token_id, pos)
        dt = time.time() - t0

        if step >= len(prompt_tokens) - 1:
            # Sample next token
            if temperature == 0:
                next_token = torch.argmax(logits).item()
            else:
                logits_scaled = logits / temperature
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

    return generated


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
    parser.add_argument("--prompt", type=str, default="The chemical formula of water is",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
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
    t0 = time.time()
    generated = generate(model, prompt_tokens,
                         max_tokens=args.max_tokens,
                         temperature=args.temperature,
                         top_k=args.top_k,
                         seed=args.seed)
    total_time = time.time() - t0

    output_text = model.decode(generated)
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {output_text}")
    print(f"{'='*60}")
    print(f"Generated {len(generated)} tokens in {total_time:.1f}s "
          f"({len(generated)/total_time:.2f} tok/s)")

    ttnn.close_device(device)
