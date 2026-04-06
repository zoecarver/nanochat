"""
Nanochat d12 training on Tenstorrent hardware via TT-Lang.

Hand-written forward and backward passes with AdamW optimizer,
analogous to llm.c but in TT-Lang. All compute streams from DRAM
through L1 dataflow buffers.

Usage:
    # Run PyTorch reference validation (no device needed):
    python ttlang/train.py --reference-only

    # Train on device:
    python ttlang/train.py --weights /path/to/d12_weights.pt
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
class ModelConfig:
    def __init__(self, n_layer=12, n_head=6, n_embd=768, vocab_size=32768,
                 sequence_len=2048, softcap=15.0, qk_scale=1.15):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.mlp_hidden = 4 * n_embd
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.softcap = softcap
        self.qk_scale = qk_scale


# Presets
D1_CONFIG = ModelConfig(n_layer=1)
D4_CONFIG = ModelConfig(n_layer=4)
D12_CONFIG = ModelConfig(n_layer=12)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE = 32


def pad_to_tile(n):
    return ((n + TILE - 1) // TILE) * TILE


# ---------------------------------------------------------------------------
# PyTorch reference: RMSNorm
# ---------------------------------------------------------------------------
def pt_rmsnorm(x):
    return F.rms_norm(x, (x.size(-1),))


# ---------------------------------------------------------------------------
# PyTorch reference: rotary embeddings
# ---------------------------------------------------------------------------
def pt_precompute_rotary(seq_len, head_dim, base=100000):
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos[None, :, None, :], sin[None, :, None, :]  # (1, T, 1, half_dim)


def pt_apply_rotary(x, cos, sin):
    """x: (B, T, n_head, head_dim), cos/sin: (1, T, 1, half_dim)"""
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# ---------------------------------------------------------------------------
# PyTorch reference: full forward pass (single layer for validation)
# ---------------------------------------------------------------------------
class PytorchRefModel:
    """Minimal PyTorch reference for gradient validation.

    Stores weights as plain tensors (not nn.Parameters) so we can compute
    gradients manually and compare against TT-Lang kernels.
    """

    def __init__(self, config, device='cpu', dtype=torch.float32):
        self.config = config
        self.device = device
        self.dtype = dtype
        C = config.n_embd
        V = config.vocab_size
        H = config.mlp_hidden
        n_head = config.n_head
        head_dim = config.head_dim

        # Initialize weights matching gpt.py init_weights
        s = 3**0.5 * C**-0.5

        self.wte = torch.randn(V, C, device=device, dtype=dtype) * 0.8
        self.lm_head = torch.randn(V, C, device=device, dtype=dtype) * 0.001

        self.resid_lambdas = torch.ones(config.n_layer, device=device, dtype=dtype)
        self.x0_lambdas = torch.full((config.n_layer,), 0.1, device=device, dtype=dtype)

        self.layers = []
        for i in range(config.n_layer):
            layer = {
                'w_q': torch.empty(C, C, device=device, dtype=dtype).uniform_(-s, s),
                'w_k': torch.empty(C, C, device=device, dtype=dtype).uniform_(-s, s),
                'w_v': torch.empty(C, C, device=device, dtype=dtype).uniform_(-s, s),
                'w_proj': torch.zeros(C, C, device=device, dtype=dtype),
                'w_fc': torch.empty(C, H, device=device, dtype=dtype).uniform_(-s * 0.5, s * 0.5),
                'w_mlp_proj': torch.zeros(H, C, device=device, dtype=dtype),
            }
            self.layers.append(layer)

        # Precompute rotary
        cos, sin = pt_precompute_rotary(config.sequence_len, head_dim)
        self.cos = cos.to(device=device, dtype=dtype)
        self.sin = sin.to(device=device, dtype=dtype)

    def get_all_params(self):
        """Return dict of name -> tensor for all trainable parameters."""
        params = {
            'wte': self.wte,
            'lm_head': self.lm_head,
            'resid_lambdas': self.resid_lambdas,
            'x0_lambdas': self.x0_lambdas,
        }
        for i, layer in enumerate(self.layers):
            for k, v in layer.items():
                params[f'layer.{i}.{k}'] = v
        return params

    def forward(self, input_ids, targets=None):
        """Full forward pass. Returns (loss, cache) where cache has saved
        activations needed for backward.

        input_ids: (B, T) long tensor
        targets: (B, T) long tensor or None
        """
        B, T = input_ids.shape
        config = self.config
        n_head = config.n_head
        head_dim = config.head_dim
        softcap = config.softcap

        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        # Embedding + initial norm
        x = self.wte[input_ids]  # (B, T, C)
        x = pt_rmsnorm(x)
        x0 = x.clone()

        # Save activations for backward
        saved = {
            'input_ids': input_ids,
            'x_layers': [x.clone()],  # x before each layer
            'x0': x0,
        }

        for i in range(config.n_layer):
            lr = self.resid_lambdas[i]
            l0 = self.x0_lambdas[i]
            layer = self.layers[i]

            # Scaled residual
            x = lr * x + l0 * x0

            # Pre-attention norm
            normed = pt_rmsnorm(x)

            # QKV projections
            q = (normed @ layer['w_q']).view(B, T, n_head, head_dim)
            k = (normed @ layer['w_k']).view(B, T, n_head, head_dim)
            v = (normed @ layer['w_v']).view(B, T, n_head, head_dim)

            # Rotary embeddings
            q = pt_apply_rotary(q, cos, sin)
            k = pt_apply_rotary(k, cos, sin)

            # QK norm + scale
            q = pt_rmsnorm(q) * config.qk_scale
            k = pt_rmsnorm(k) * config.qk_scale

            # Causal attention (naive, for reference correctness)
            q_t = q.transpose(1, 2)  # (B, n_head, T, head_dim)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            scale = 1.0 / math.sqrt(head_dim)
            attn_scores = q_t @ k_t.transpose(-2, -1) * scale  # (B, n_head, T, T)
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_head, T, T)
            attn_out = attn_weights @ v_t  # (B, n_head, T, head_dim)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)

            # Output projection + residual
            y = attn_out @ layer['w_proj']
            x = x + y

            # Pre-MLP norm
            normed2 = pt_rmsnorm(x)

            # MLP: fc -> relu^2 -> proj
            hidden = normed2 @ layer['w_fc']
            hidden_act = F.relu(hidden).square()
            mlp_out = hidden_act @ layer['w_mlp_proj']
            x = x + mlp_out

            saved['x_layers'].append(x.clone())

        # Final norm + lm_head
        x = pt_rmsnorm(x)
        logits = x @ self.lm_head.t()  # (B, T, V)
        logits = logits[..., :config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction='mean')
            return loss, saved
        return logits, saved

    def forward_backward(self, input_ids, targets):
        """Compute loss and all gradients via autograd for validation."""
        params = self.get_all_params()
        for p in params.values():
            p.requires_grad_(True)

        loss, _ = self.forward(input_ids, targets)
        loss.backward()

        grads = {name: p.grad.clone() for name, p in params.items()}

        for p in params.values():
            p.requires_grad_(False)
            p.grad = None

        return loss.item(), grads


# ---------------------------------------------------------------------------
# Validation: test PyTorch reference produces reasonable loss
# ---------------------------------------------------------------------------
def test_pytorch_reference():
    print("=" * 60)
    print("Testing PyTorch reference model (d1)")
    print("=" * 60)

    config = D1_CONFIG
    model = PytorchRefModel(config, dtype=torch.float32)

    B, T = 1, 64  # small for fast testing
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    # Forward
    loss, saved = model.forward(input_ids, targets)
    print(f"Forward loss: {loss.item():.4f}")
    print(f"Expected ~ln({config.vocab_size}) = {math.log(config.vocab_size):.4f} for random init")
    assert abs(loss.item() - math.log(config.vocab_size)) < 2.0, \
        f"Loss {loss.item()} too far from expected {math.log(config.vocab_size)}"

    # Forward + backward (autograd)
    loss_val, grads = model.forward_backward(input_ids, targets)
    print(f"Backward loss: {loss_val:.4f}")
    print(f"Number of gradient tensors: {len(grads)}")

    # Check key gradients are nonzero
    for name, grad in grads.items():
        norm = grad.float().norm().item()
        print(f"  {name:30s}  grad_norm={norm:.6f}  shape={list(grad.shape)}")
    # lm_head and wte must always have gradients
    assert grads['lm_head'].float().norm().item() > 0, "lm_head gradient is zero!"
    assert grads['wte'].float().norm().item() > 0, "wte gradient is zero!"

    # Test with non-zero projections to verify full gradient flow
    print("\nTesting with non-zero projections...")
    model2 = PytorchRefModel(config, dtype=torch.float32)
    for layer in model2.layers:
        layer['w_proj'] = torch.randn_like(layer['w_proj']) * 0.01
        layer['w_mlp_proj'] = torch.randn_like(layer['w_mlp_proj']) * 0.01
    loss2, grads2 = model2.forward_backward(input_ids, targets)
    all_nonzero = True
    for name, grad in grads2.items():
        norm = grad.float().norm().item()
        if norm == 0:
            print(f"  WARNING: {name} still has zero gradient")
            all_nonzero = False
    if all_nonzero:
        print("  All gradients nonzero with non-zero projections: PASS")

    print("\nPyTorch reference: PASS")
    return True


# ===========================================================================
# TT-Lang Kernels
# ===========================================================================
import ttnn
import ttl

# ---------------------------------------------------------------------------
# TT-Lang helpers
# ---------------------------------------------------------------------------
def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def to_ttnn_l1(tensor, device):
    return ttnn.from_torch(
        tensor.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


def to_ttnn_f32(tensor, device):
    return ttnn.from_torch(
        tensor.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def to_ttnn_l1_f32(tensor, device):
    return ttnn.from_torch(
        tensor.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


# ---------------------------------------------------------------------------
# TT-Lang: Linear kernel (reused from inference.py)
# out = x @ w, where x is (M, K) and w is (K, N)
# ---------------------------------------------------------------------------
NCOLS = 4  # output columns per work unit

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


# ---------------------------------------------------------------------------
# TT-Lang: RMSNorm with rstd output (for backward pass)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# TT-Lang: Elementwise ReLU squared
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# TT-Lang: Residual add: out = x + y
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# TT-Lang: Softcap: out = cap * tanh(x / cap)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# TT-Lang: Scaled residual: out = lr * x + l0 * x0
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# TT-Lang: Rotary embeddings for training (per-position cos/sin)
# x: (n_head * seq_tiles * TILE, HEAD_DIM)
# cos/sin: (seq_tiles * TILE, HALF_DIM)
# ---------------------------------------------------------------------------
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
HEAD_TILES = HEAD_DIM // TILE   # 4
HALF_TILES = HALF_DIM // TILE   # 2


def make_rotary_training_kernel(n_head, seq_tiles):
    @ttl.operation(grid="auto")
    def rotary_training_kernel(x, cos, sin, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        total_seq_tiles = n_head * seq_tiles
        tiles_per_core = -(-total_seq_tiles // grid_cols)

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
                if t < total_seq_tiles:
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
                if t < total_seq_tiles:
                    pos_tile = t % seq_tiles
                    for j in range(HALF_TILES):
                        with x1_dfb.reserve() as b1, x2_dfb.reserve() as b2, cos_dfb.reserve() as b3, sin_dfb.reserve() as b4:
                            tx1 = ttl.copy(x[t, j], b1)
                            tx2 = ttl.copy(x[t, j + HALF_TILES], b2)
                            tx3 = ttl.copy(cos[pos_tile, j], b3)
                            tx4 = ttl.copy(sin[pos_tile, j], b4)
                            tx1.wait(); tx2.wait(); tx3.wait(); tx4.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_seq_tiles:
                    for j in range(HALF_TILES):
                        with out_dfb.wait() as b1:
                            tx1 = ttl.copy(b1, out[t, j]); tx1.wait()
                        with out_dfb.wait() as b2:
                            tx2 = ttl.copy(b2, out[t, j + HALF_TILES]); tx2.wait()

    return rotary_training_kernel


# ---------------------------------------------------------------------------
# TT-Lang: Reshape interleaved heads <-> batched heads (training shapes)
# to_heads:   (T, n_head * HEAD_DIM) -> (n_head * T, HEAD_DIM)
# from_heads: (n_head * T, HEAD_DIM) -> (T, n_head * HEAD_DIM)
# ---------------------------------------------------------------------------
def make_reshape_to_heads_kernel(n_head, seq_tiles):
    @ttl.operation(grid="auto")
    def reshape_to_heads(inp, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        total = n_head * seq_tiles * HEAD_TILES
        tiles_per_core = -(-total // grid_cols)

        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    with inp_dfb.wait() as blk, out_dfb.reserve() as o:
                        o.store(blk)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    # Decode: which head, which seq tile, which head tile
                    work = t
                    h = work // (seq_tiles * HEAD_TILES)
                    rem = work % (seq_tiles * HEAD_TILES)
                    s = rem // HEAD_TILES
                    j = rem % HEAD_TILES
                    # Input layout: (seq_tiles, n_head * HEAD_TILES)
                    inp_col = h * HEAD_TILES + j
                    with inp_dfb.reserve() as blk:
                        tx = ttl.copy(inp[s, inp_col], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    work = t
                    h = work // (seq_tiles * HEAD_TILES)
                    rem = work % (seq_tiles * HEAD_TILES)
                    s = rem // HEAD_TILES
                    j = rem % HEAD_TILES
                    # Output layout: (n_head * seq_tiles, HEAD_TILES)
                    out_row = h * seq_tiles + s
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[out_row, j]); tx.wait()

    return reshape_to_heads


def make_reshape_from_heads_kernel(n_head, seq_tiles):
    @ttl.operation(grid="auto")
    def reshape_from_heads(inp, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        total = n_head * seq_tiles * HEAD_TILES
        tiles_per_core = -(-total // grid_cols)

        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    with inp_dfb.wait() as blk, out_dfb.reserve() as o:
                        o.store(blk)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    work = t
                    h = work // (seq_tiles * HEAD_TILES)
                    rem = work % (seq_tiles * HEAD_TILES)
                    s = rem // HEAD_TILES
                    j = rem % HEAD_TILES
                    # Input layout: (n_head * seq_tiles, HEAD_TILES)
                    inp_row = h * seq_tiles + s
                    with inp_dfb.reserve() as blk:
                        tx = ttl.copy(inp[inp_row, j], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total:
                    work = t
                    h = work // (seq_tiles * HEAD_TILES)
                    rem = work % (seq_tiles * HEAD_TILES)
                    s = rem // HEAD_TILES
                    j = rem % HEAD_TILES
                    # Output layout: (seq_tiles, n_head * HEAD_TILES)
                    out_col = h * HEAD_TILES + j
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[s, out_col]); tx.wait()

    return reshape_from_heads


# ===========================================================================
# Backward Kernels
# ===========================================================================

# ---------------------------------------------------------------------------
# TT-Lang: Transpose 2D: (M, N) -> (N, M) in tiles
# ---------------------------------------------------------------------------
@ttl.operation(grid="auto")
def transpose_2d_kernel(x, out):
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
                    o.store(ttl.transpose(xv))

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
                    tx = ttl.copy(blk, out[col, row]); tx.wait()


# ---------------------------------------------------------------------------
# TT-Lang: Softcap backward: dx = dout * (1 - tanh(x/cap)^2)
# ---------------------------------------------------------------------------
@ttl.operation(grid="auto")
def softcap_backward_kernel(x, dout, inv_cap_tile, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    dout_dfb = ttl.make_dataflow_buffer_like(dout, shape=(1, 1), buffer_factor=2)
    ic_dfb = ttl.make_dataflow_buffer_like(inv_cap_tile, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        with ic_dfb.wait() as ic:
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    with x_dfb.wait() as xv, dout_dfb.wait() as dv, out_dfb.reserve() as o:
                        th = ttl.math.tanh(xv * ic)
                        o.store(dv * (ttl.math.fill(xv, 1.0) - th * th))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        with ic_dfb.reserve() as blk:
            tx = ttl.copy(inv_cap_tile[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as b1, dout_dfb.reserve() as b2:
                    tx1 = ttl.copy(x[row, col], b1)
                    tx2 = ttl.copy(dout[row, col], b2)
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


# ---------------------------------------------------------------------------
# TT-Lang: ReLU^2 backward: dx = 2 * relu(x) * dout
# ---------------------------------------------------------------------------
@ttl.operation(grid="auto")
def relu_sq_backward_kernel(x, dout, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    dout_dfb = ttl.make_dataflow_buffer_like(dout, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with x_dfb.wait() as xv, dout_dfb.wait() as dv, out_dfb.reserve() as o:
                    r = ttl.math.relu(xv)
                    o.store((r + r) * dv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as b1, dout_dfb.reserve() as b2:
                    tx1 = ttl.copy(x[row, col], b1)
                    tx2 = ttl.copy(dout[row, col], b2)
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


# ---------------------------------------------------------------------------
# TT-Lang: Rotary backward: dx1 = dy1*cos - dy2*sin, dx2 = dy1*sin + dy2*cos
# ---------------------------------------------------------------------------
def make_rotary_backward_kernel(n_head, seq_tiles):
    @ttl.operation(grid="auto")
    def rotary_backward_kernel(dout, cos, sin, dx):
        grid_cols, _ = ttl.grid_size(dims=2)
        total_seq_tiles = n_head * seq_tiles
        tiles_per_core = -(-total_seq_tiles // grid_cols)

        dy1_dfb = ttl.make_dataflow_buffer_like(dout, shape=(1, 1), buffer_factor=2)
        dy2_dfb = ttl.make_dataflow_buffer_like(dout, shape=(1, 1), buffer_factor=2)
        cos_dfb = ttl.make_dataflow_buffer_like(cos, shape=(1, 1), buffer_factor=2)
        sin_dfb = ttl.make_dataflow_buffer_like(sin, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(dx, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_seq_tiles:
                    for j in range(HALF_TILES):
                        with dy1_dfb.wait() as d1, dy2_dfb.wait() as d2, cos_dfb.wait() as c, sin_dfb.wait() as s:
                            with out_dfb.reserve() as o:
                                o.store(d1 * c - d2 * s)
                            with out_dfb.reserve() as o:
                                o.store(d1 * s + d2 * c)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_seq_tiles:
                    pos_tile = t % seq_tiles
                    for j in range(HALF_TILES):
                        with dy1_dfb.reserve() as b1, dy2_dfb.reserve() as b2, cos_dfb.reserve() as b3, sin_dfb.reserve() as b4:
                            tx1 = ttl.copy(dout[t, j], b1)
                            tx2 = ttl.copy(dout[t, j + HALF_TILES], b2)
                            tx3 = ttl.copy(cos[pos_tile, j], b3)
                            tx4 = ttl.copy(sin[pos_tile, j], b4)
                            tx1.wait(); tx2.wait(); tx3.wait(); tx4.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_seq_tiles:
                    for j in range(HALF_TILES):
                        with out_dfb.wait() as b1:
                            tx1 = ttl.copy(b1, dx[t, j]); tx1.wait()
                        with out_dfb.wait() as b2:
                            tx2 = ttl.copy(b2, dx[t, j + HALF_TILES]); tx2.wait()

    return rotary_backward_kernel


# ---------------------------------------------------------------------------
# TT-Lang: Linear backward dW = X^T @ dY with K-chunked accumulation
# X^T: (M, K) = (C, T), dY: (K, N) = (T, N), dW: (M, N) = (C, N)
# K is large (T=2048=64 tiles), so we chunk along K.
# ---------------------------------------------------------------------------
K_CHUNK = 16  # tiles per K chunk

def make_linear_backward_dw_kernel(k_tiles):
    n_chunks = k_tiles // K_CHUNK

    @ttl.operation(grid="auto")
    def linear_backward_dw_kernel(xt, dy, dw):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = xt.shape[0] // TILE
        n_tiles = dy.shape[1] // TILE
        col_groups = n_tiles // NCOLS
        total = m_tiles * col_groups
        units_per_core = -(-total // grid_cols)

        a_dfb = ttl.make_dataflow_buffer_like(xt, shape=(1, K_CHUNK), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(dy, shape=(K_CHUNK, NCOLS), buffer_factor=2)
        partial_dfb = ttl.make_dataflow_buffer_like(dw, shape=(1, NCOLS), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(dw, shape=(1, NCOLS), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(dw, shape=(1, NCOLS), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    # First chunk -> accumulator
                    with a_dfb.wait() as a, b_dfb.wait() as b, acc_dfb.reserve() as acc:
                        acc.store(a @ b)
                    # Remaining chunks: partial matmul + accumulate
                    for chunk in range(n_chunks - 1):
                        with a_dfb.wait() as a, b_dfb.wait() as b, partial_dfb.reserve() as p:
                            p.store(a @ b)
                        with partial_dfb.wait() as pv, acc_dfb.wait() as old, acc_dfb.reserve() as new:
                            new.store(pv + old)
                    # Write final accumulated result
                    with acc_dfb.wait() as final, out_dfb.reserve() as o:
                        o.store(final)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // col_groups
                    gi = t % col_groups
                    sc = gi * NCOLS
                    for chunk in range(n_chunks):
                        k_start = chunk * K_CHUNK
                        with a_dfb.reserve() as b1, b_dfb.reserve() as b2:
                            tx1 = ttl.copy(xt[mi, k_start:k_start + K_CHUNK], b1)
                            tx2 = ttl.copy(dy[k_start:k_start + K_CHUNK, sc:sc + NCOLS], b2)
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
                        tx = ttl.copy(blk, dw[mi, sc:sc + NCOLS]); tx.wait()

    return linear_backward_dw_kernel


# ---------------------------------------------------------------------------
# TT-Lang: RMSNorm backward
# Given x, dout, rstd (per-row), computes:
#   c = sum(dout * x) per row
#   dx = rstd * dout - (rstd^3 * c / N) * x
# Two-pass: (1) compute dot(dout, x) per row, (2) compute dx
# ---------------------------------------------------------------------------
def make_rmsnorm_backward_kernel(n_dim):
    dim_tiles = n_dim // TILE

    @ttl.operation(grid="auto")
    def rmsnorm_backward_kernel(x, dout, rstd, scaler, mean_scale, dx):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        dout_dfb = ttl.make_dataflow_buffer_like(dout, shape=(1, 1), buffer_factor=2)
        rstd_dfb = ttl.make_dataflow_buffer_like(rstd, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        prod_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(dx, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        # Pass 1: c = sum(dout * x) across dim tiles
                        with x_dfb.wait() as x0, dout_dfb.wait() as d0:
                            with prod_dfb.reserve() as p:
                                p.store(x0 * d0)
                        with prod_dfb.wait() as pv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(pv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)
                        for j in range(dim_tiles - 1):
                            with x_dfb.wait() as xj, dout_dfb.wait() as dj:
                                with prod_dfb.reserve() as p:
                                    p.store(xj * dj)
                            with prod_dfb.wait() as pv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(pv, sc, dims=[1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as new_acc:
                                new_acc.store(av + rv)

                        # scale = rstd^3 * c / N
                        # broadcast c, multiply by rstd^3 * (1/N)
                        with rstd_dfb.wait() as rv:
                            with acc_dfb.wait() as c_val, bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(c_val, bc, dims=[1]))
                            # scale_tile = c * rstd^3 / N = c * rstd * rstd^2 * mean_scale
                            with bcast_dfb.wait() as c_bc, red_dfb.reserve() as scale:
                                scale.store(c_bc * rv * rv * rv * ms)

                            # Pass 2: dx = rstd * dout - scale * x
                            with red_dfb.wait() as scale_v:
                                for j in range(dim_tiles):
                                    with x_dfb.wait() as xj, dout_dfb.wait() as dj, out_dfb.reserve() as o:
                                        o.store(rv * dj - scale_v * xj)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            with sc_dfb.reserve() as b1, ms_dfb.reserve() as b2:
                tx1 = ttl.copy(scaler[0, 0], b1)
                tx2 = ttl.copy(mean_scale[0, 0], b2)
                tx1.wait(); tx2.wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Pass 1: x and dout for dot product
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as b1, dout_dfb.reserve() as b2:
                            tx1 = ttl.copy(x[tile_idx, j], b1)
                            tx2 = ttl.copy(dout[tile_idx, j], b2)
                            tx1.wait(); tx2.wait()
                    # rstd for this row
                    with rstd_dfb.reserve() as blk:
                        tx = ttl.copy(rstd[tile_idx, 0], blk); tx.wait()
                    # Pass 2: x and dout again for final computation
                    for j in range(dim_tiles):
                        with x_dfb.reserve() as b1, dout_dfb.reserve() as b2:
                            tx1 = ttl.copy(x[tile_idx, j], b1)
                            tx2 = ttl.copy(dout[tile_idx, j], b2)
                            tx1.wait(); tx2.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, dx[tile_idx, j]); tx.wait()

    return rmsnorm_backward_kernel


# ---------------------------------------------------------------------------
# TT-Lang: Training flash attention (causal, one core per head)
# ---------------------------------------------------------------------------
HEAD_TILES = 4  # head_dim // TILE = 128 // 32

def make_training_attention_kernel(n_head, seq_tiles):
    """Flash attention for training: full causal attention.
    Q, K: (n_head * seq_tiles, HEAD_TILES) tiles -- after QK norm + scale
    V: (n_head * seq_tiles, HEAD_TILES) tiles
    out: same shape as V
    """
    @ttl.operation(grid=(n_head, 1))
    def training_attention(Q, K, V, scale_tile, scaler, neg_inf_tile,
                           zero_tile, zero_head, causal_mask, out):
        q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(1, HEAD_TILES), buffer_factor=2)
        k_dfb = ttl.make_dataflow_buffer_like(K, shape=(1, HEAD_TILES), buffer_factor=2)
        v_dfb = ttl.make_dataflow_buffer_like(V, shape=(1, HEAD_TILES), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), buffer_factor=1)
        scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ninf_dfb = ttl.make_dataflow_buffer_like(neg_inf_tile, shape=(1, 1), buffer_factor=1)
        zero_dfb = ttl.make_dataflow_buffer_like(zero_tile, shape=(1, 1), buffer_factor=1)
        zh_dfb = ttl.make_dataflow_buffer_like(zero_head, shape=(1, HEAD_TILES), buffer_factor=1)
        mask_dfb = ttl.make_dataflow_buffer_like(causal_mask, shape=(1, 1), buffer_factor=2)

        kt_dfb = ttl.make_dataflow_buffer_like(K, shape=(HEAD_TILES, 1), buffer_factor=2)
        qk_dfb = ttl.make_dataflow_buffer_like(Q, shape=(1, 1), buffer_factor=2)
        scaled_dfb = ttl.make_dataflow_buffer_like(Q, shape=(1, 1), buffer_factor=2)
        cm_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        m_new_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        alpha_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        alpha_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)
        exp_dfb = ttl.make_dataflow_buffer_like(Q, shape=(1, 1), buffer_factor=2)
        cs_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        co_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)

        m_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        l_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        o_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)
        l_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)

        # Temp DFB for draining m after inner loop
        m_drain_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            h, _ = ttl.node(dims=2)
            with sc_dfb.wait() as sc_blk, scaler_dfb.wait() as scaler_blk, \
                 ninf_dfb.wait() as ninf_blk, zero_dfb.wait() as zero_blk, \
                 zh_dfb.wait() as zh_blk:

                for q_row in range(seq_tiles):
                    with q_dfb.wait() as q_blk:
                        # Init running state: m=-inf, l=0, o=0
                        with m_dfb.reserve() as mi:
                            mi.store(ninf_blk)
                        with l_dfb.reserve() as li:
                            li.store(zero_blk)
                        with o_dfb.reserve() as oi:
                            oi.store(zh_blk)

                        for kv_col in range(q_row + 1):
                            with k_dfb.wait() as kc, kt_dfb.reserve() as kt:
                                kt.store(ttl.transpose(kc))
                            with kt_dfb.wait() as ktv, qk_dfb.reserve() as qk:
                                qk.store(q_blk @ ktv)
                            with qk_dfb.wait() as qkv, mask_dfb.wait() as mv:
                                with scaled_dfb.reserve() as scd:
                                    scd.store(sc_blk * qkv + mv)

                            # Online softmax
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
                                        with exp_dfb.reserve() as ex:
                                            ex.store(ttl.math.exp(sd - mn))
                                        with m_dfb.reserve() as m_next:
                                            m_next.store(mn)

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
                                    with co_dfb.reserve() as co:
                                        co.store(abc * o_old)
                                with co_dfb.wait() as co, v_dfb.wait() as vc:
                                    with o_dfb.reserve() as o_new:
                                        o_new.store(co + exp_blk @ vc)

                        # Drain m (l and o consumed in normalize below)
                        with m_dfb.wait() as m_final, m_drain_dfb.reserve() as md:
                            md.store(m_final)

                    # Normalize: o = o / l
                    with l_dfb.wait() as l_final, l_bc_dfb.reserve() as lbc:
                        lbc.store(ttl.math.broadcast(l_final, lbc, dims=[1]))
                    with o_dfb.wait() as o_final, l_bc_dfb.wait() as lbc:
                        with out_dfb.reserve() as o:
                            o.store(o_final / lbc)

        @ttl.datamovement()
        def dm_read():
            h, _ = ttl.node(dims=2)
            kv_base = h * seq_tiles
            # Load constants
            with sc_dfb.reserve() as b:
                tx = ttl.copy(scale_tile[0, 0], b); tx.wait()
            with scaler_dfb.reserve() as b:
                tx = ttl.copy(scaler[0, 0], b); tx.wait()
            with ninf_dfb.reserve() as b:
                tx = ttl.copy(neg_inf_tile[0, 0], b); tx.wait()
            with zero_dfb.reserve() as b:
                tx = ttl.copy(zero_tile[0, 0], b); tx.wait()
            with zh_dfb.reserve() as b:
                tx = ttl.copy(zero_head[0, 0:HEAD_TILES], b); tx.wait()

            for q_row in range(seq_tiles):
                # Load Q for this row
                with q_dfb.reserve() as b:
                    tx = ttl.copy(Q[kv_base + q_row:kv_base + q_row + 1, 0:HEAD_TILES], b)
                    tx.wait()
                for kv_col in range(q_row + 1):
                    # Load K, V, mask
                    with k_dfb.reserve() as b:
                        tx = ttl.copy(K[kv_base + kv_col:kv_base + kv_col + 1, 0:HEAD_TILES], b)
                        tx.wait()
                    with v_dfb.reserve() as b:
                        tx = ttl.copy(V[kv_base + kv_col:kv_base + kv_col + 1, 0:HEAD_TILES], b)
                        tx.wait()
                    if kv_col == q_row:
                        with mask_dfb.reserve() as b:
                            tx = ttl.copy(causal_mask[0, 0], b); tx.wait()
                    else:
                        with mask_dfb.reserve() as b:
                            tx = ttl.copy(zero_tile[0, 0], b); tx.wait()

        @ttl.datamovement()
        def dm_write():
            h, _ = ttl.node(dims=2)
            out_base = h * seq_tiles
            for q_row in range(seq_tiles):
                # Drain the m accumulator (written by compute, discard)
                with m_drain_dfb.wait() as _discard:
                    pass
                with out_dfb.wait() as b:
                    tx = ttl.copy(b, out[out_base + q_row:out_base + q_row + 1, 0:HEAD_TILES])
                    tx.wait()

    return training_attention


# ---------------------------------------------------------------------------
# TT-Lang kernel tests at training shapes
# ---------------------------------------------------------------------------
def test_linear_kernel(device):
    """Test linear kernel at training shape: (2048, 768) @ (768, 768)"""
    print("\n--- test_linear_kernel ---")
    config = D12_CONFIG
    T = 2048
    C = config.n_embd
    k_tiles = C // TILE  # 24

    torch.manual_seed(42)
    x_pt = torch.randn(T, C, dtype=torch.bfloat16)
    w_pt = torch.randn(C, C, dtype=torch.bfloat16)
    expected = (x_pt.float() @ w_pt.float()).to(torch.bfloat16)

    x_tt = to_ttnn(x_pt.view(T, C), device)
    w_tt = to_ttnn(w_pt.view(C, C), device)
    out_tt = to_ttnn(torch.zeros(T, C, dtype=torch.bfloat16), device)

    kernel = make_linear_kernel(k_tiles)
    kernel(x_tt, w_tt, out_tt)

    result = ttnn.to_torch(out_tt).view(T, C)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: ({T}, {C}) @ ({C}, {C})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 5.0, f"Linear kernel error too large: {max_err}"
    print("  PASS")


def test_linear_kernel_wide(device):
    """Test linear at MLP shape: (2048, 768) @ (768, 3072)"""
    print("\n--- test_linear_kernel_wide ---")
    T, C, H = 2048, 768, 3072
    k_tiles = C // TILE  # 24

    torch.manual_seed(42)
    x_pt = torch.randn(T, C, dtype=torch.bfloat16)
    w_pt = torch.randn(C, H, dtype=torch.bfloat16)
    expected = (x_pt.float() @ w_pt.float()).to(torch.bfloat16)

    x_tt = to_ttnn(x_pt, device)
    w_tt = to_ttnn(w_pt, device)
    out_tt = to_ttnn(torch.zeros(T, H, dtype=torch.bfloat16), device)

    kernel = make_linear_kernel(k_tiles)
    kernel(x_tt, w_tt, out_tt)

    result = ttnn.to_torch(out_tt).view(T, H)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: ({T}, {C}) @ ({C}, {H})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 10.0, f"Linear kernel error too large: {max_err}"
    print("  PASS")


def test_rmsnorm_kernel(device):
    """Test RMSNorm at training shape: (2048, 768)"""
    print("\n--- test_rmsnorm_kernel ---")
    T, C = 2048, 768

    torch.manual_seed(42)
    x_pt = torch.randn(T, C, dtype=torch.bfloat16)
    expected = pt_rmsnorm(x_pt.float()).to(torch.bfloat16)

    x_tt = to_ttnn(x_pt, device)
    out_tt = to_ttnn(torch.zeros(T, C, dtype=torch.bfloat16), device)
    scaler_tt = to_ttnn_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    ms_val = 1.0 / C
    ms_tt = to_ttnn_l1(torch.full((TILE, TILE), ms_val, dtype=torch.bfloat16), device)

    kernel = make_rmsnorm_kernel(C)
    kernel(x_tt, scaler_tt, ms_tt, out_tt)

    result = ttnn.to_torch(out_tt).view(T, C)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: ({T}, {C})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 0.1, f"RMSNorm kernel error too large: {max_err}"
    print("  PASS")


def test_relu_sq_kernel(device):
    """Test ReLU^2 at training shape: (2048, 3072)"""
    print("\n--- test_relu_sq_kernel ---")
    T, H = 2048, 3072

    torch.manual_seed(42)
    x_pt = torch.randn(T, H, dtype=torch.bfloat16)
    expected = F.relu(x_pt.float()).square().to(torch.bfloat16)

    x_tt = to_ttnn(x_pt, device)
    out_tt = to_ttnn(torch.zeros(T, H, dtype=torch.bfloat16), device)

    relu_sq_kernel(x_tt, out_tt)

    result = ttnn.to_torch(out_tt).view(T, H)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: ({T}, {H})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 0.5, f"ReLU^2 kernel error too large: {max_err}"
    print("  PASS")


def test_softcap_kernel(device):
    """Test softcap at training shape: (2048, 32768)"""
    print("\n--- test_softcap_kernel ---")
    T, V = 2048, pad_to_tile(32768)
    cap = 15.0

    torch.manual_seed(42)
    x_pt = torch.randn(T, V, dtype=torch.bfloat16) * 20  # some values outside cap
    expected = (cap * torch.tanh(x_pt.float() / cap)).to(torch.bfloat16)

    x_tt = to_ttnn(x_pt, device)
    out_tt = to_ttnn(torch.zeros(T, V, dtype=torch.bfloat16), device)
    inv_cap_tt = to_ttnn_l1(torch.full((TILE, TILE), 1.0 / cap, dtype=torch.bfloat16), device)
    cap_tt = to_ttnn_l1(torch.full((TILE, TILE), cap, dtype=torch.bfloat16), device)

    softcap_kernel(x_tt, inv_cap_tt, cap_tt, out_tt)

    result = ttnn.to_torch(out_tt).view(T, V)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: ({T}, {V})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 1.0, f"Softcap kernel error too large: {max_err}"
    print("  PASS")


def test_reshape_round_trip(device):
    """Test reshape_to_heads -> reshape_from_heads round trip."""
    print("\n--- test_reshape_round_trip ---")
    T, C = 2048, 768
    n_head = 6
    seq_tiles = T // TILE  # 64

    torch.manual_seed(42)
    x_pt = torch.randn(T, C, dtype=torch.bfloat16)

    x_tt = to_ttnn(x_pt, device)
    # to_heads: (64, 24) tiles -> (384, 4) tiles
    mid_tt = to_ttnn(torch.zeros(n_head * T, HEAD_DIM, dtype=torch.bfloat16), device)
    # from_heads: (384, 4) tiles -> (64, 24) tiles
    out_tt = to_ttnn(torch.zeros(T, C, dtype=torch.bfloat16), device)

    to_heads = make_reshape_to_heads_kernel(n_head, seq_tiles)
    from_heads = make_reshape_from_heads_kernel(n_head, seq_tiles)

    to_heads(x_tt, mid_tt)
    from_heads(mid_tt, out_tt)

    result = ttnn.to_torch(out_tt).view(T, C)
    max_err = (result.float() - x_pt.float()).abs().max().item()
    print(f"  round-trip max_err={max_err:.6f}")
    assert max_err == 0.0, f"Reshape round trip error: {max_err}"

    # Also verify the intermediate layout is correct
    mid = ttnn.to_torch(mid_tt).view(n_head * T, HEAD_DIM)
    # Head 0 should be x[:, 0:128], Head 1 should be x[:, 128:256], etc.
    for h in range(n_head):
        head_data = mid[h * T:(h + 1) * T]
        expected_head = x_pt[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
        head_err = (head_data.float() - expected_head.float()).abs().max().item()
        assert head_err == 0.0, f"Head {h} mismatch: {head_err}"
    print("  intermediate layout correct")
    print("  PASS")


def test_rotary_training_kernel(device):
    """Test rotary embeddings at training shapes."""
    print("\n--- test_rotary_training_kernel ---")
    # Start small: 1 head, 64 positions (2 tiles)
    T, n_head = 64, 1
    seq_tiles = T // TILE  # 2

    T, n_head = 2048, 6
    seq_tiles = T // TILE

    torch.manual_seed(42)
    x_pt = torch.randn(n_head * T, HEAD_DIM, dtype=torch.bfloat16)
    cos_pt, sin_pt = pt_precompute_rotary(T, HEAD_DIM)
    cos_2d = cos_pt.squeeze(0).squeeze(1).to(torch.bfloat16)
    sin_2d = sin_pt.squeeze(0).squeeze(1).to(torch.bfloat16)

    expected = torch.zeros_like(x_pt)
    for h in range(n_head):
        hx = x_pt[h * T:(h + 1) * T]
        x1 = hx[:, :HALF_DIM]
        x2 = hx[:, HALF_DIM:]
        y1 = x1.float() * cos_2d.float() + x2.float() * sin_2d.float()
        y2 = x1.float() * (-sin_2d.float()) + x2.float() * cos_2d.float()
        expected[h * T:(h + 1) * T] = torch.cat([y1, y2], dim=1).to(torch.bfloat16)

    x_tt = to_ttnn(x_pt, device)
    cos_tt = to_ttnn(cos_2d, device)
    sin_tt = to_ttnn(sin_2d, device)
    out_tt = to_ttnn(torch.zeros_like(x_pt), device)

    kernel = make_rotary_training_kernel(n_head, seq_tiles)
    kernel(x_tt, cos_tt, sin_tt, out_tt)

    result = ttnn.to_torch(out_tt).view(n_head * T, HEAD_DIM)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: ({n_head * T}, {HEAD_DIM})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 0.5, f"Rotary kernel error too large: {max_err}"
    print("  PASS")


def test_forward_kernels(device):
    """Run all forward kernel tests."""
    print("=" * 60)
    print("Testing forward kernels at training shapes")
    print("=" * 60)
    test_linear_kernel(device)
    test_linear_kernel_wide(device)
    test_rmsnorm_kernel(device)
    test_relu_sq_kernel(device)
    test_softcap_kernel(device)
    test_reshape_round_trip(device)
    test_rotary_training_kernel(device)
    print("\nAll forward kernel tests: PASS")


# ---------------------------------------------------------------------------
# Backward kernel tests
# ---------------------------------------------------------------------------
def test_transpose_2d(device):
    """Test transpose: (768, 2048) -> (2048, 768)"""
    print("\n--- test_transpose_2d ---")
    M, N = 768, 2048

    torch.manual_seed(42)
    x_pt = torch.randn(M, N, dtype=torch.bfloat16)
    expected = x_pt.t().contiguous()

    x_tt = to_ttnn(x_pt, device)
    out_tt = to_ttnn(torch.zeros(N, M, dtype=torch.bfloat16), device)

    transpose_2d_kernel(x_tt, out_tt)

    result = ttnn.to_torch(out_tt).view(N, M)
    max_err = (result.float() - expected.float()).abs().max().item()
    print(f"  shape: ({M}, {N}) -> ({N}, {M})")
    print(f"  max_err={max_err:.6f}")
    assert max_err == 0.0, f"Transpose error: {max_err}"
    print("  PASS")


def test_softcap_backward(device):
    """Test softcap backward against PyTorch autograd."""
    print("\n--- test_softcap_backward ---")
    T, V = 2048, pad_to_tile(32768)
    cap = 15.0

    torch.manual_seed(42)
    x_pt = torch.randn(T, V, dtype=torch.float32) * 20
    x_pt.requires_grad_(True)
    y = cap * torch.tanh(x_pt / cap)
    dout_pt = torch.randn_like(y)
    y.backward(dout_pt)
    expected = x_pt.grad.to(torch.bfloat16)

    x_tt = to_ttnn(x_pt.detach().to(torch.bfloat16), device)
    dout_tt = to_ttnn(dout_pt.to(torch.bfloat16), device)
    inv_cap_tt = to_ttnn_l1(torch.full((TILE, TILE), 1.0 / cap, dtype=torch.bfloat16), device)
    out_tt = to_ttnn(torch.zeros(T, V, dtype=torch.bfloat16), device)

    softcap_backward_kernel(x_tt, dout_tt, inv_cap_tt, out_tt)

    result = ttnn.to_torch(out_tt).view(T, V)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: ({T}, {V})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 1.0, f"Softcap backward error too large: {max_err}"
    print("  PASS")


def test_relu_sq_backward(device):
    """Test relu^2 backward against PyTorch autograd."""
    print("\n--- test_relu_sq_backward ---")
    T, H = 2048, 3072

    torch.manual_seed(42)
    x_pt = torch.randn(T, H, dtype=torch.float32)
    x_pt.requires_grad_(True)
    y = F.relu(x_pt).square()
    dout_pt = torch.randn_like(y)
    y.backward(dout_pt)
    expected = x_pt.grad.to(torch.bfloat16)

    x_tt = to_ttnn(x_pt.detach().to(torch.bfloat16), device)
    dout_tt = to_ttnn(dout_pt.to(torch.bfloat16), device)
    out_tt = to_ttnn(torch.zeros(T, H, dtype=torch.bfloat16), device)

    relu_sq_backward_kernel(x_tt, dout_tt, out_tt)

    result = ttnn.to_torch(out_tt).view(T, H)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: ({T}, {H})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 1.0, f"ReLU^2 backward error too large: {max_err}"
    print("  PASS")


def test_rotary_backward(device):
    """Test rotary backward against PyTorch autograd."""
    print("\n--- test_rotary_backward ---")
    T, n_head = 2048, 6
    seq_tiles = T // TILE

    torch.manual_seed(42)
    # Forward in float32 to get autograd reference
    x_pt = torch.randn(n_head * T, HEAD_DIM, dtype=torch.float32)
    x_pt.requires_grad_(True)
    cos_pt, sin_pt = pt_precompute_rotary(T, HEAD_DIM)
    cos_2d = cos_pt.squeeze(0).squeeze(1)
    sin_2d = sin_pt.squeeze(0).squeeze(1)

    # Apply rotary per head
    ys = []
    for h in range(n_head):
        hx = x_pt[h * T:(h + 1) * T]
        x1, x2 = hx[:, :HALF_DIM], hx[:, HALF_DIM:]
        y1 = x1 * cos_2d + x2 * sin_2d
        y2 = x1 * (-sin_2d) + x2 * cos_2d
        ys.append(torch.cat([y1, y2], dim=1))
    y = torch.cat(ys, dim=0)
    dout_pt = torch.randn_like(y)
    y.backward(dout_pt)
    expected = x_pt.grad.to(torch.bfloat16)

    dout_tt = to_ttnn(dout_pt.to(torch.bfloat16), device)
    cos_tt = to_ttnn(cos_2d.to(torch.bfloat16), device)
    sin_tt = to_ttnn(sin_2d.to(torch.bfloat16), device)
    dx_tt = to_ttnn(torch.zeros(n_head * T, HEAD_DIM, dtype=torch.bfloat16), device)

    kernel = make_rotary_backward_kernel(n_head, seq_tiles)
    kernel(dout_tt, cos_tt, sin_tt, dx_tt)

    result = ttnn.to_torch(dx_tt).view(n_head * T, HEAD_DIM)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: ({n_head * T}, {HEAD_DIM})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 0.5, f"Rotary backward error too large: {max_err}"
    print("  PASS")


def test_linear_backward_dx(device):
    """Test linear backward dx = dY @ W^T (reuses forward linear kernel)."""
    print("\n--- test_linear_backward_dx ---")
    T, C = 2048, 768
    k_tiles = C // TILE

    torch.manual_seed(42)
    x_pt = torch.randn(T, C, dtype=torch.float32)
    w_pt = torch.randn(C, C, dtype=torch.float32)
    x_pt.requires_grad_(True)
    y = x_pt @ w_pt
    dout_pt = torch.randn_like(y)
    y.backward(dout_pt)
    expected = x_pt.grad.to(torch.bfloat16)

    # dx = dout @ W^T
    dout_tt = to_ttnn(dout_pt.to(torch.bfloat16), device)
    wt_pt = w_pt.t().contiguous().to(torch.bfloat16)
    wt_tt = to_ttnn(wt_pt, device)
    dx_tt = to_ttnn(torch.zeros(T, C, dtype=torch.bfloat16), device)

    kernel = make_linear_kernel(k_tiles)
    kernel(dout_tt, wt_tt, dx_tt)

    result = ttnn.to_torch(dx_tt).view(T, C)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: dout({T}, {C}) @ W^T({C}, {C})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 5.0, f"Linear backward dx error too large: {max_err}"
    print("  PASS")


def test_linear_backward_dw(device):
    """Test linear backward dW = X^T @ dY with K-chunked accumulation."""
    print("\n--- test_linear_backward_dw ---")
    T, C = 2048, 768
    k_tiles = T // TILE  # 64

    torch.manual_seed(42)
    x_pt = torch.randn(T, C, dtype=torch.float32)
    w_pt = torch.randn(C, C, dtype=torch.float32)
    w_pt.requires_grad_(True)
    y = x_pt @ w_pt
    dout_pt = torch.randn_like(y)
    y.backward(dout_pt)
    expected = w_pt.grad.to(torch.bfloat16)

    # dW = X^T @ dout
    xt_pt = x_pt.t().contiguous().to(torch.bfloat16)
    xt_tt = to_ttnn(xt_pt, device)
    dout_tt = to_ttnn(dout_pt.to(torch.bfloat16), device)
    dw_tt = to_ttnn(torch.zeros(C, C, dtype=torch.bfloat16), device)

    kernel = make_linear_backward_dw_kernel(k_tiles)
    kernel(xt_tt, dout_tt, dw_tt)

    result = ttnn.to_torch(dw_tt).view(C, C)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: X^T({C}, {T}) @ dout({T}, {C})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 20.0, f"Linear backward dW error too large: {max_err}"
    print("  PASS")


def test_rmsnorm_backward(device):
    """Test rmsnorm backward against PyTorch autograd."""
    print("\n--- test_rmsnorm_backward ---")
    T, C = 2048, 768

    torch.manual_seed(42)
    x_pt = torch.randn(T, C, dtype=torch.float32)
    x_pt.requires_grad_(True)
    y = F.rms_norm(x_pt, (C,))
    dout_pt = torch.randn_like(y)
    y.backward(dout_pt)
    expected = x_pt.grad.to(torch.bfloat16)

    # Compute rstd for the kernel: rstd = 1/sqrt(mean(x^2) + eps)
    x_f32 = x_pt.detach()
    rstd_pt = (x_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
    # Broadcast rstd to tile shape: (T, 1) -> (T, TILE) with repeated values
    rstd_tiled = rstd_pt.expand(T, TILE).contiguous().to(torch.bfloat16)

    x_tt = to_ttnn(x_f32.to(torch.bfloat16), device)
    dout_tt = to_ttnn(dout_pt.to(torch.bfloat16), device)
    rstd_tt = to_ttnn(rstd_tiled, device)
    scaler_tt = to_ttnn_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    ms_val = 1.0 / C
    ms_tt = to_ttnn_l1(torch.full((TILE, TILE), ms_val, dtype=torch.bfloat16), device)
    dx_tt = to_ttnn(torch.zeros(T, C, dtype=torch.bfloat16), device)

    kernel = make_rmsnorm_backward_kernel(C)
    kernel(x_tt, dout_tt, rstd_tt, scaler_tt, ms_tt, dx_tt)

    result = ttnn.to_torch(dx_tt).view(T, C)
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    print(f"  shape: ({T}, {C})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert max_err < 0.5, f"RMSNorm backward error too large: {max_err}"
    print("  PASS")


def test_backward_kernels(device):
    """Run all backward kernel tests."""
    print("=" * 60)
    print("Testing backward kernels")
    print("=" * 60)
    test_transpose_2d(device)
    test_softcap_backward(device)
    test_relu_sq_backward(device)
    test_rotary_backward(device)
    test_linear_backward_dx(device)
    test_linear_backward_dw(device)
    test_rmsnorm_backward(device)
    print("\nAll backward kernel tests: PASS")


# ===========================================================================
# AdamW Kernel
# ===========================================================================
@ttl.operation(grid="auto")
def adamw_kernel(param, grad, m, v, param_out, m_out, v_out,
                 b1_t, omb1_t, b2_t, omb2_t,
                 nlbc_t, b2c_t, decay_t, eps_t):
    """AdamW update. Constants are precomputed scalar tiles:
    b1, 1-b1, b2, 1-b2, -lr/(1-b1^t), 1/(1-b2^t), 1-lr*wd, eps
    """
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = param.shape[0] // TILE
    col_tiles = param.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    p_dfb = ttl.make_dataflow_buffer_like(param, shape=(1, 1), buffer_factor=2)
    g_dfb = ttl.make_dataflow_buffer_like(grad, shape=(1, 1), buffer_factor=2)
    m_dfb = ttl.make_dataflow_buffer_like(m, shape=(1, 1), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(1, 1), buffer_factor=2)
    po_dfb = ttl.make_dataflow_buffer_like(param_out, shape=(1, 1), buffer_factor=2)
    mo_dfb = ttl.make_dataflow_buffer_like(m_out, shape=(1, 1), buffer_factor=2)
    vo_dfb = ttl.make_dataflow_buffer_like(v_out, shape=(1, 1), buffer_factor=2)
    # Intermediate DFBs to break register pressure
    dp_dfb = ttl.make_dataflow_buffer_like(param, shape=(1, 1), buffer_factor=2)
    step_dfb = ttl.make_dataflow_buffer_like(param, shape=(1, 1), buffer_factor=2)
    vh_dfb = ttl.make_dataflow_buffer_like(param, shape=(1, 1), buffer_factor=2)
    b1_dfb = ttl.make_dataflow_buffer_like(b1_t, shape=(1, 1), buffer_factor=1)
    omb1_dfb = ttl.make_dataflow_buffer_like(omb1_t, shape=(1, 1), buffer_factor=1)
    b2_dfb = ttl.make_dataflow_buffer_like(b2_t, shape=(1, 1), buffer_factor=1)
    omb2_dfb = ttl.make_dataflow_buffer_like(omb2_t, shape=(1, 1), buffer_factor=1)
    nlbc_dfb = ttl.make_dataflow_buffer_like(nlbc_t, shape=(1, 1), buffer_factor=1)
    b2c_dfb = ttl.make_dataflow_buffer_like(b2c_t, shape=(1, 1), buffer_factor=1)
    dec_dfb = ttl.make_dataflow_buffer_like(decay_t, shape=(1, 1), buffer_factor=1)
    eps_dfb = ttl.make_dataflow_buffer_like(eps_t, shape=(1, 1), buffer_factor=1)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        with b1_dfb.wait() as b1v, omb1_dfb.wait() as omb1v, \
             b2_dfb.wait() as b2v, omb2_dfb.wait() as omb2v, \
             nlbc_dfb.wait() as nlbcv, b2c_dfb.wait() as b2cv, \
             dec_dfb.wait() as decv, eps_dfb.wait() as epsv:
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    with p_dfb.wait() as pv, g_dfb.wait() as gv, \
                         m_dfb.wait() as mv, v_dfb.wait() as vv:
                        with mo_dfb.reserve() as mo:
                            mo.store(b1v * mv + omb1v * gv)
                        with vo_dfb.reserve() as vo:
                            vo.store(b2v * vv + omb2v * gv * gv)
                        # Save decay*p and v_hat separately
                        with dp_dfb.reserve() as dp:
                            dp.store(decv * pv)
                        with vh_dfb.reserve() as vh:
                            vh.store(b2cv * (b2v * vv + omb2v * gv * gv))
                        with step_dfb.reserve() as sv:
                            sv.store(nlbcv * (b1v * mv + omb1v * gv))
                    # Combine: step / (sqrt(v_hat) + eps) + decay*p
                    with vh_dfb.wait() as vh, step_dfb.wait() as sv, \
                         dp_dfb.wait() as dp, po_dfb.reserve() as po:
                        po.store(dp + sv * ttl.math.recip(
                            ttl.math.sqrt(vh) + epsv))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        with b1_dfb.reserve() as blk:
            tx = ttl.copy(b1_t[0, 0], blk); tx.wait()
        with omb1_dfb.reserve() as blk:
            tx = ttl.copy(omb1_t[0, 0], blk); tx.wait()
        with b2_dfb.reserve() as blk:
            tx = ttl.copy(b2_t[0, 0], blk); tx.wait()
        with omb2_dfb.reserve() as blk:
            tx = ttl.copy(omb2_t[0, 0], blk); tx.wait()
        with nlbc_dfb.reserve() as blk:
            tx = ttl.copy(nlbc_t[0, 0], blk); tx.wait()
        with b2c_dfb.reserve() as blk:
            tx = ttl.copy(b2c_t[0, 0], blk); tx.wait()
        with dec_dfb.reserve() as blk:
            tx = ttl.copy(decay_t[0, 0], blk); tx.wait()
        with eps_dfb.reserve() as blk:
            tx = ttl.copy(eps_t[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with p_dfb.reserve() as b1, g_dfb.reserve() as b2, \
                     m_dfb.reserve() as b3, v_dfb.reserve() as b4:
                    tx1 = ttl.copy(param[row, col], b1)
                    tx2 = ttl.copy(grad[row, col], b2)
                    tx3 = ttl.copy(m[row, col], b3)
                    tx4 = ttl.copy(v[row, col], b4)
                    tx1.wait(); tx2.wait(); tx3.wait(); tx4.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with mo_dfb.wait() as blk:
                    tx = ttl.copy(blk, m_out[row, col]); tx.wait()
                with vo_dfb.wait() as blk:
                    tx = ttl.copy(blk, v_out[row, col]); tx.wait()
                with po_dfb.wait() as blk:
                    tx = ttl.copy(blk, param_out[row, col]); tx.wait()


def make_adamw_constants(lr, beta1, beta2, wd, step, device):
    """Create constant tiles for AdamW kernel (float32 for precision)."""
    def tile(val):
        return to_ttnn_l1_f32(torch.full((TILE, TILE), val), device)
    b1_corr = 1.0 / (1.0 - beta1 ** step)
    b2_corr = 1.0 / (1.0 - beta2 ** step)
    return {
        'b1': tile(beta1), 'omb1': tile(1.0 - beta1),
        'b2': tile(beta2), 'omb2': tile(1.0 - beta2),
        'nlbc': tile(-lr * b1_corr), 'b2c': tile(b2_corr),
        'decay': tile(1.0 - lr * wd), 'eps': tile(1e-8),
    }


# ===========================================================================
# Training Pipeline
# ===========================================================================
class TrainingState:
    """Holds all device tensors and kernels for training."""

    def __init__(self, config, model, device, T=2048):
        self.config = config
        self.device = device
        self.T = T
        C = config.n_embd
        H = config.mlp_hidden
        V = config.vocab_size
        n_head = config.n_head
        hd = config.head_dim
        seq_tiles = T // TILE

        # --- Kernel instances ---
        self.linear_cc = make_linear_kernel(C // TILE)
        self.linear_ch = make_linear_kernel(C // TILE)
        self.linear_hc = make_linear_backward_dw_kernel(H // TILE)
        self.linear_cv = make_linear_kernel(C // TILE)
        self.rmsnorm_c = make_rmsnorm_kernel(C)
        self.rmsnorm_hd = make_rmsnorm_kernel(hd)
        self.rotary_fwd = make_rotary_training_kernel(n_head, seq_tiles)
        self.rotary_bwd = make_rotary_backward_kernel(n_head, seq_tiles)
        self.to_heads = make_reshape_to_heads_kernel(n_head, seq_tiles)
        self.from_heads = make_reshape_from_heads_kernel(n_head, seq_tiles)
        self.linear_bwd_dw = make_linear_backward_dw_kernel(T // TILE)
        self.rmsnorm_bwd_c = make_rmsnorm_backward_kernel(C)
        self.rmsnorm_bwd_hd = make_rmsnorm_backward_kernel(hd)
        self.attn_fwd = make_training_attention_kernel(n_head, seq_tiles)

        # --- Upload weights ---
        def up(t):
            return to_ttnn(t.to(torch.bfloat16), device)

        self.wte_cpu = model.wte.to(torch.bfloat16)
        self.lm_head_t = up(model.lm_head.t().contiguous())
        self.w_q = [up(l['w_q']) for l in model.layers]
        self.w_k = [up(l['w_k']) for l in model.layers]
        self.w_v = [up(l['w_v']) for l in model.layers]
        self.w_proj = [up(l['w_proj']) for l in model.layers]
        self.w_fc = [up(l['w_fc']) for l in model.layers]
        self.w_mlp_proj = [up(l['w_mlp_proj']) for l in model.layers]
        # Transposed weights for backward dx
        self.w_q_t = [up(l['w_q'].t().contiguous()) for l in model.layers]
        self.w_k_t = [up(l['w_k'].t().contiguous()) for l in model.layers]
        self.w_v_t = [up(l['w_v'].t().contiguous()) for l in model.layers]
        self.w_proj_t = [up(l['w_proj'].t().contiguous()) for l in model.layers]
        self.w_fc_t = [up(l['w_fc'].t().contiguous()) for l in model.layers]
        self.w_mlp_proj_t = [up(l['w_mlp_proj'].t().contiguous()) for l in model.layers]
        self.lm_head_tt = up(model.lm_head)  # for backward dw

        # Lambda tiles
        self.lr_tiles = [to_ttnn_l1(torch.full((TILE, TILE),
                         model.resid_lambdas[i].item(), dtype=torch.bfloat16), device)
                         for i in range(config.n_layer)]
        self.l0_tiles = [to_ttnn_l1(torch.full((TILE, TILE),
                         model.x0_lambdas[i].item(), dtype=torch.bfloat16), device)
                         for i in range(config.n_layer)]

        # --- Constants ---
        self.scaler = to_ttnn_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
        self.ms_c = to_ttnn_l1(torch.full((TILE, TILE), 1.0 / C, dtype=torch.bfloat16), device)
        self.ms_hd = to_ttnn_l1(torch.full((TILE, TILE), 1.0 / hd, dtype=torch.bfloat16), device)
        self.inv_cap = to_ttnn_l1(torch.full((TILE, TILE), 1.0 / config.softcap,
                                  dtype=torch.bfloat16), device)
        self.cap = to_ttnn_l1(torch.full((TILE, TILE), config.softcap,
                              dtype=torch.bfloat16), device)
        cos, sin = pt_precompute_rotary(T, hd)
        self.cos_tt = up(cos.squeeze(0).squeeze(1))
        self.sin_tt = up(sin.squeeze(0).squeeze(1))

        # Attention constants
        sdpa_scale = config.qk_scale ** 2 / math.sqrt(hd)
        self.sdpa_scale = to_ttnn_l1(torch.full((TILE, TILE), sdpa_scale,
                                     dtype=torch.bfloat16), device)
        self.neg_inf = to_ttnn_l1(torch.full((TILE, TILE), float('-inf'),
                                  dtype=torch.bfloat16), device)
        self.zero_tile = to_ttnn_l1(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
        self.zero_head = to_ttnn_l1(torch.zeros(TILE, hd, dtype=torch.bfloat16), device)
        causal = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
        causal[torch.triu(torch.ones(TILE, TILE, dtype=torch.bool), diagonal=1)] = float('-inf')
        self.causal_mask = to_ttnn_l1(causal, device)

        # --- Scratch tensors ---
        def alloc(r, c):
            return to_ttnn(torch.zeros(r, c, dtype=torch.bfloat16), device)
        nT = n_head * T
        self.s1 = alloc(T, C)
        self.s2 = alloc(T, C)
        self.s3 = alloc(T, C)
        self.q_flat = alloc(T, C)
        self.k_flat = alloc(T, C)
        self.v_flat = alloc(T, C)
        self.q_heads = alloc(nT, hd)
        self.k_heads = alloc(nT, hd)
        self.v_heads = alloc(nT, hd)
        self.q_rot = alloc(nT, hd)
        self.k_rot = alloc(nT, hd)
        self.q_norm = alloc(nT, hd)
        self.k_norm = alloc(nT, hd)
        self.attn_out = alloc(nT, hd)
        self.attn_concat = alloc(T, C)
        self.proj_out = alloc(T, C)
        self.hidden = alloc(T, H)
        self.hidden_act = alloc(T, H)
        self.mlp_out = alloc(T, C)
        Vp = pad_to_tile(V)
        self.logits = alloc(T, Vp)
        self.logits_capped = alloc(T, Vp)
        # Backward scratch
        self.dx = alloc(T, C)
        self.d_normed = alloc(T, C)
        self.d_resid = alloc(T, C)  # persists across attention backward
        self.d_attn_heads = alloc(nT, hd)
        self.d_q_heads = alloc(nT, hd)
        self.d_k_heads = alloc(nT, hd)
        self.d_v_heads = alloc(nT, hd)
        self.d_q_rot = alloc(nT, hd)
        self.d_k_rot = alloc(nT, hd)
        self.d_hidden = alloc(T, H)
        self.d_hidden_act = alloc(T, H)

        # Optimizer state and master weights (float32 for precision)
        self.m_state = {}
        self.v_state = {}
        self.master_weights = {}
        weight_shapes = {}
        for i in range(config.n_layer):
            weight_shapes[f'layer.{i}.w_q'] = (C, C)
            weight_shapes[f'layer.{i}.w_k'] = (C, C)
            weight_shapes[f'layer.{i}.w_v'] = (C, C)
            weight_shapes[f'layer.{i}.w_proj'] = (C, C)
            weight_shapes[f'layer.{i}.w_fc'] = (C, H)
            weight_shapes[f'layer.{i}.w_mlp_proj'] = (H, C)
        weight_shapes['lm_head'] = (V, C)
        def alloc_f32(r, c):
            return to_ttnn_f32(torch.zeros(r, c), device)
        for k, (r, c) in weight_shapes.items():
            self.m_state[k] = alloc_f32(r, c)
            self.v_state[k] = alloc_f32(r, c)
        # Float32 master weights (copy of initial bf16 weights)
        for i in range(config.n_layer):
            for wname in ['w_q', 'w_k', 'w_v', 'w_proj', 'w_fc', 'w_mlp_proj']:
                gname = f'layer.{i}.{wname}'
                w_list = getattr(self, wname)
                w_cpu = ttnn.to_torch(w_list[i]).reshape(weight_shapes[gname])
                self.master_weights[gname] = to_ttnn_f32(w_cpu, device)
        lm_cpu = ttnn.to_torch(self.lm_head_tt).reshape(V, C)
        self.master_weights['lm_head'] = to_ttnn_f32(lm_cpu, device)
        # wte optimizer state on host
        self.m_state['wte'] = torch.zeros(V, C, dtype=torch.float32)
        self.v_state['wte'] = torch.zeros(V, C, dtype=torch.float32)

    def forward(self, input_ids, targets):
        """Full forward pass. Returns (loss, saved_activations)."""
        T = self.T
        C = self.config.n_embd
        V = self.config.vocab_size
        n_head = self.config.n_head
        hd = self.config.head_dim
        device = self.device

        # Embedding on host
        x_cpu = self.wte_cpu[input_ids.squeeze(0)].contiguous()
        x_tt = to_ttnn(x_cpu, device)

        # Initial RMSNorm
        self.rmsnorm_c(x_tt, self.scaler, self.ms_c, self.s1)
        x_tt = self.s1

        # Clone x0
        x0_cpu = ttnn.to_torch(x_tt).reshape(T, C).to(torch.bfloat16)
        x0_tt = to_ttnn(x0_cpu, device)

        # Save inter-layer x for gradient checkpointing
        saved_x = [x0_cpu.clone()]

        for i in range(self.config.n_layer):
            # Scaled residual
            scaled_residual_kernel(x_tt, x0_tt, self.lr_tiles[i], self.l0_tiles[i], self.s2)
            x_tt = self.s2

            # Pre-attention RMSNorm
            self.rmsnorm_c(x_tt, self.scaler, self.ms_c, self.s1)

            # QKV
            self.linear_cc(self.s1, self.w_q[i], self.q_flat)
            self.linear_cc(self.s1, self.w_k[i], self.k_flat)
            self.linear_cc(self.s1, self.w_v[i], self.v_flat)

            # Reshape to heads
            self.to_heads(self.q_flat, self.q_heads)
            self.to_heads(self.k_flat, self.k_heads)
            self.to_heads(self.v_flat, self.v_heads)

            # Rotary
            self.rotary_fwd(self.q_heads, self.cos_tt, self.sin_tt, self.q_rot)
            self.rotary_fwd(self.k_heads, self.cos_tt, self.sin_tt, self.k_rot)

            # QK norm
            self.rmsnorm_hd(self.q_rot, self.scaler, self.ms_hd, self.q_norm)
            self.rmsnorm_hd(self.k_rot, self.scaler, self.ms_hd, self.k_norm)

            # Flash attention on device (causal, scale includes qk_scale^2/sqrt(hd))
            self.attn_fwd(self.q_norm, self.k_norm, self.v_heads,
                          self.sdpa_scale, self.scaler, self.neg_inf,
                          self.zero_tile, self.zero_head, self.causal_mask,
                          self.attn_out)

            # Reshape from heads + proj + residual
            self.from_heads(self.attn_out, self.attn_concat)
            self.linear_cc(self.attn_concat, self.w_proj[i], self.proj_out)
            residual_add_kernel(x_tt, self.proj_out, self.s3)
            x_tt = self.s3

            # Pre-MLP RMSNorm
            self.rmsnorm_c(x_tt, self.scaler, self.ms_c, self.s1)

            # MLP: fc -> relu^2 -> proj + residual
            self.linear_ch(self.s1, self.w_fc[i], self.hidden)
            relu_sq_kernel(self.hidden, self.hidden_act)
            self.linear_hc(self.hidden_act, self.w_mlp_proj[i], self.mlp_out)
            residual_add_kernel(x_tt, self.mlp_out, self.s2)
            x_tt = self.s2

            # Save x for gradient checkpointing
            x_cpu = ttnn.to_torch(x_tt).reshape(T, C).to(torch.bfloat16)
            saved_x.append(x_cpu.clone())

        # Final RMSNorm
        self.rmsnorm_c(x_tt, self.scaler, self.ms_c, self.s1)

        # LM head + softcap
        self.linear_cv(self.s1, self.lm_head_t, self.logits)
        softcap_kernel(self.logits, self.inv_cap, self.cap, self.logits_capped)

        # Loss on host
        logits_cpu = ttnn.to_torch(self.logits_capped).reshape(T, -1)[:, :V].float()
        loss = F.cross_entropy(logits_cpu, targets.squeeze(0), reduction='mean')

        # dlogits = softmax(logits) - one_hot(targets)
        probs = F.softmax(logits_cpu, dim=-1)
        dlogits = probs.clone()
        dlogits.scatter_(1, targets.squeeze(0).unsqueeze(1), dlogits.gather(1, targets.squeeze(0).unsqueeze(1)) - 1.0)
        dlogits /= T  # mean reduction

        return loss.item(), saved_x, x0_cpu, dlogits

    def _make_rstd(self, x_cpu, dim):
        """Compute rstd on host, return device tensor (rows, TILE)."""
        rstd = (x_cpu.pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
        return to_ttnn(rstd.expand(-1, TILE).contiguous().to(torch.bfloat16), self.device)

    def backward(self, saved_x, x0_cpu, dlogits, input_ids):
        """Full backward pass. Uses device kernels for dx flow, host for dW
        and attention backward."""
        T = self.T
        C = self.config.n_embd
        H = self.config.mlp_hidden
        V = self.config.vocab_size
        n_head = self.config.n_head
        hd = self.config.head_dim
        device = self.device
        qk_scale = self.config.qk_scale

        grads = {}

        # --- Softcap + lm_head backward (host, operates on logits) ---
        logits_pre_cpu = ttnn.to_torch(self.logits).reshape(T, -1)[:, :V].float()
        cap = self.config.softcap
        th = torch.tanh(logits_pre_cpu / cap)
        d_logits_pre = dlogits * (1.0 - th * th)

        lm_head_cpu = ttnn.to_torch(self.lm_head_tt).reshape(V, C).float()
        final_normed_cpu = ttnn.to_torch(self.s1).reshape(T, C).float()
        d_final_normed = d_logits_pre @ lm_head_cpu
        grads['lm_head'] = d_logits_pre.t() @ final_normed_cpu

        # --- Final rmsnorm backward (device) ---
        x_final_cpu = saved_x[-1].float()
        x_final_tt = to_ttnn(saved_x[-1], device)
        rstd_tt = self._make_rstd(x_final_cpu, C)
        d_fn_tt = to_ttnn(d_final_normed.to(torch.bfloat16), device)
        self.rmsnorm_bwd_c(x_final_tt, d_fn_tt, rstd_tt, self.scaler, self.ms_c, self.dx)
        # De-alias: dout_tt must not share storage with scratch tensors
        dout_tt = to_ttnn(ttnn.to_torch(self.dx).reshape(T, C).to(torch.bfloat16), device)

        d_x0 = torch.zeros(T, C, dtype=torch.float32)

        # --- Per-layer backward ---
        for i in reversed(range(self.config.n_layer)):
            lr_val = float(ttnn.to_torch(self.lr_tiles[i]).flatten()[0])
            l0_val = float(ttnn.to_torch(self.l0_tiles[i]).flatten()[0])

            # == Recompute forward on device ==
            x_i_tt = to_ttnn(saved_x[i], device)
            x0_tt = to_ttnn(x0_cpu, device)
            scaled_residual_kernel(x_i_tt, x0_tt, self.lr_tiles[i],
                                   self.l0_tiles[i], self.s2)  # s2 = x_scaled
            self.rmsnorm_c(self.s2, self.scaler, self.ms_c, self.s1)  # s1 = normed
            normed_cpu = ttnn.to_torch(self.s1).reshape(T, C).float()

            self.linear_cc(self.s1, self.w_q[i], self.q_flat)
            self.linear_cc(self.s1, self.w_k[i], self.k_flat)
            self.linear_cc(self.s1, self.w_v[i], self.v_flat)
            self.to_heads(self.q_flat, self.q_heads)
            self.to_heads(self.k_flat, self.k_heads)
            self.to_heads(self.v_flat, self.v_heads)
            self.rotary_fwd(self.q_heads, self.cos_tt, self.sin_tt, self.q_rot)
            self.rotary_fwd(self.k_heads, self.cos_tt, self.sin_tt, self.k_rot)
            self.rmsnorm_hd(self.q_rot, self.scaler, self.ms_hd, self.q_norm)
            self.rmsnorm_hd(self.k_rot, self.scaler, self.ms_hd, self.k_norm)

            # Attention forward on host (need P for backward)
            q_cpu = ttnn.to_torch(self.q_norm).reshape(n_head, T, hd).float() * qk_scale
            k_cpu = ttnn.to_torch(self.k_norm).reshape(n_head, T, hd).float() * qk_scale
            v_cpu = ttnn.to_torch(self.v_heads).reshape(n_head, T, hd).float()
            scale = 1.0 / math.sqrt(hd)
            scores = q_cpu @ k_cpu.transpose(-2, -1) * scale
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))
            P = F.softmax(scores, dim=-1)
            attn_out = P @ v_cpu
            attn_out_tt = to_ttnn(attn_out.to(torch.bfloat16).reshape(n_head * T, hd), device)

            self.from_heads(attn_out_tt, self.attn_concat)
            attn_concat_cpu = ttnn.to_torch(self.attn_concat).reshape(T, C).float()
            self.linear_cc(self.attn_concat, self.w_proj[i], self.proj_out)
            residual_add_kernel(self.s2, self.proj_out, self.s3)  # s3 = x_post_attn
            x_post_attn_cpu = ttnn.to_torch(self.s3).reshape(T, C).float()

            self.rmsnorm_c(self.s3, self.scaler, self.ms_c, self.s1)  # s1 = normed2
            normed2_cpu = ttnn.to_torch(self.s1).reshape(T, C).float()
            self.linear_ch(self.s1, self.w_fc[i], self.hidden)
            relu_sq_kernel(self.hidden, self.hidden_act)
            hidden_pre_cpu = ttnn.to_torch(self.hidden).reshape(T, H).float()
            hidden_act_cpu = ttnn.to_torch(self.hidden_act).reshape(T, H).float()
            self.linear_hc(self.hidden_act, self.w_mlp_proj[i], self.mlp_out)

            # == Backward MLP (device kernels for dx, host for dW) ==

            # d_hidden_act = dout @ w_mlp_proj^T
            self.linear_cc(dout_tt, self.w_mlp_proj_t[i], self.d_hidden_act)
            dout_cpu = ttnn.to_torch(dout_tt).reshape(T, C).float()
            grads[f'layer.{i}.w_mlp_proj'] = hidden_act_cpu.t() @ dout_cpu

            # relu_sq backward on device
            relu_sq_backward_kernel(self.hidden, self.d_hidden_act, self.d_hidden)

            # d_normed2 = d_hidden @ w_fc^T (k=96, use chunked kernel)
            self.linear_hc(self.d_hidden, self.w_fc_t[i], self.d_normed)
            d_hidden_cpu = ttnn.to_torch(self.d_hidden).reshape(T, H).float()
            grads[f'layer.{i}.w_fc'] = normed2_cpu.t() @ d_hidden_cpu

            # rmsnorm backward on device
            rstd2_tt = self._make_rstd(x_post_attn_cpu, C)
            self.rmsnorm_bwd_c(self.s3, self.d_normed, rstd2_tt,
                               self.scaler, self.ms_c, self.dx)

            # d_x_post_attn = rmsnorm_bwd + dout (residual)
            residual_add_kernel(self.dx, dout_tt, self.d_resid)  # d_resid persists

            # == Backward attention ==

            # d_attn_concat = d_x_post_attn @ w_proj^T
            self.linear_cc(self.d_resid, self.w_proj_t[i], self.s1)
            d_x_post_attn_cpu = ttnn.to_torch(self.d_resid).reshape(T, C).float()
            grads[f'layer.{i}.w_proj'] = attn_concat_cpu.t() @ d_x_post_attn_cpu

            # reshape to heads
            self.to_heads(self.s1, self.d_attn_heads)

            # Attention backward on host
            d_attn_cpu = ttnn.to_torch(self.d_attn_heads).reshape(n_head, T, hd).float()
            dV = P.transpose(-2, -1) @ d_attn_cpu
            dP = d_attn_cpu @ v_cpu.transpose(-2, -1)
            dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
            dQ_input = dS @ k_cpu * scale
            dK_input = dS.transpose(-2, -1) @ q_cpu * scale

            # QK norm backward on host (includes 1.15 scale + rmsnorm)
            dQ_norm = dQ_input * qk_scale
            dK_norm = dK_input * qk_scale
            q_rot_cpu = ttnn.to_torch(self.q_rot).reshape(n_head, T, hd).float()
            k_rot_cpu = ttnn.to_torch(self.k_rot).reshape(n_head, T, hd).float()
            rstd_q = (q_rot_cpu.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()
            rstd_k = (k_rot_cpu.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()
            cq = (dQ_norm * q_rot_cpu).sum(-1, keepdim=True)
            ck = (dK_norm * k_rot_cpu).sum(-1, keepdim=True)
            dQ_rot = rstd_q * dQ_norm - rstd_q.pow(3) * cq / hd * q_rot_cpu
            dK_rot = rstd_k * dK_norm - rstd_k.pow(3) * ck / hd * k_rot_cpu

            # Rotary backward on device
            dQ_rot_tt = to_ttnn(dQ_rot.to(torch.bfloat16).reshape(n_head * T, hd), device)
            dK_rot_tt = to_ttnn(dK_rot.to(torch.bfloat16).reshape(n_head * T, hd), device)
            self.rotary_bwd(dQ_rot_tt, self.cos_tt, self.sin_tt, self.d_q_heads)
            self.rotary_bwd(dK_rot_tt, self.cos_tt, self.sin_tt, self.d_k_heads)

            # Reshape from heads and sum QKV backward dx on device
            self.from_heads(self.d_q_heads, self.q_flat)
            self.from_heads(self.d_k_heads, self.k_flat)
            dV_tt = to_ttnn(dV.to(torch.bfloat16).reshape(n_head * T, hd), device)
            self.from_heads(dV_tt, self.v_flat)

            # dW for QKV on host
            dq_cpu = ttnn.to_torch(self.q_flat).reshape(T, C).float()
            dk_cpu = ttnn.to_torch(self.k_flat).reshape(T, C).float()
            dv_cpu = ttnn.to_torch(self.v_flat).reshape(T, C).float()
            grads[f'layer.{i}.w_q'] = normed_cpu.t() @ dq_cpu
            grads[f'layer.{i}.w_k'] = normed_cpu.t() @ dk_cpu
            grads[f'layer.{i}.w_v'] = normed_cpu.t() @ dv_cpu

            # d_normed = dQ@wq^T + dK@wk^T + dV@wv^T (device)
            self.linear_cc(self.q_flat, self.w_q_t[i], self.dx)
            self.linear_cc(self.k_flat, self.w_k_t[i], self.d_normed)
            residual_add_kernel(self.dx, self.d_normed, self.s1)
            self.linear_cc(self.v_flat, self.w_v_t[i], self.dx)
            residual_add_kernel(self.s1, self.dx, self.d_normed)

            # Pre-attention rmsnorm backward (device)
            x_scaled_cpu = ttnn.to_torch(self.s2).reshape(T, C).float()
            rstd1_tt = self._make_rstd(x_scaled_cpu, C)
            self.rmsnorm_bwd_c(self.s2, self.d_normed, rstd1_tt,
                               self.scaler, self.ms_c, self.dx)

            # d_x_scaled = rmsnorm_bwd + d_x_post_attn (residual)
            residual_add_kernel(self.dx, self.d_resid, self.s1)

            # Scaled residual backward (host, scalar multiply)
            d_x_scaled_cpu = ttnn.to_torch(self.s1).reshape(T, C).float()
            dout_cpu = lr_val * d_x_scaled_cpu
            d_x0 += l0_val * d_x_scaled_cpu
            dout_tt = to_ttnn(dout_cpu.to(torch.bfloat16), device)

        # Embedding backward (host)
        grads['wte'] = torch.zeros_like(self.wte_cpu.float())
        x_embed_cpu = self.wte_cpu[input_ids.squeeze(0)].float()
        d_total = dout_cpu + d_x0
        rstd_e = (x_embed_cpu.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()
        ce = (d_total * x_embed_cpu).sum(-1, keepdim=True)
        d_embed = rstd_e * d_total - rstd_e.pow(3) * ce / C * x_embed_cpu
        for t_idx in range(T):
            tok = input_ids[0, t_idx].item()
            grads['wte'][tok] += d_embed[t_idx]

        return grads

    def adamw_step(self, grads, step, lr=1e-3, beta1=0.9, beta2=0.999, wd=0.01,
                   use_host=False):
        """Update weights. use_host=True for host-side AdamW (debugging)."""
        device = self.device
        C, H, V = self.config.n_embd, self.config.mlp_hidden, self.config.vocab_size

        attr_map = {
            'w_q': ('w_q', 'w_q_t'), 'w_k': ('w_k', 'w_k_t'),
            'w_v': ('w_v', 'w_v_t'), 'w_proj': ('w_proj', 'w_proj_t'),
            'w_fc': ('w_fc', 'w_fc_t'), 'w_mlp_proj': ('w_mlp_proj', 'w_mlp_proj_t'),
        }

        if use_host:
            self._adamw_step_host(grads, step, lr, beta1, beta2, wd, attr_map)
        else:
            self._adamw_step_device(grads, step, lr, beta1, beta2, wd, attr_map)

    def _adamw_step_host(self, grads, step, lr, beta1, beta2, wd, attr_map):
        """Host-side AdamW for all weights (debugging reference)."""
        device = self.device
        C, H, V = self.config.n_embd, self.config.mlp_hidden, self.config.vocab_size

        def host_update(name, w_cpu, g):
            g = g.float()
            if name not in self._host_m:
                self._host_m[name] = torch.zeros_like(g)
                self._host_v[name] = torch.zeros_like(g)
            m, v = self._host_m[name], self._host_v[name]
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** (step + 1))
            v_hat = v / (1 - beta2 ** (step + 1))
            return (w_cpu.float() * (1 - lr * wd) -
                    lr * m_hat / (v_hat.sqrt() + 1e-8)).to(torch.bfloat16)

        if not hasattr(self, '_host_m'):
            self._host_m = {}
            self._host_v = {}

        for i in range(self.config.n_layer):
            for wname in ['w_q', 'w_k', 'w_v', 'w_proj', 'w_fc', 'w_mlp_proj']:
                gname = f'layer.{i}.{wname}'
                w_list = getattr(self, attr_map[wname][0])
                wt_list = getattr(self, attr_map[wname][1])
                w_cpu = ttnn.to_torch(w_list[i]).reshape(grads[gname].shape)
                w_new = host_update(gname, w_cpu, grads[gname])
                w_list[i] = to_ttnn(w_new, device)
                wt_list[i] = to_ttnn(w_new.t().contiguous(), device)

        # lm_head
        lm_cpu = ttnn.to_torch(self.lm_head_tt).reshape(V, C)
        lm_new = host_update('lm_head', lm_cpu, grads['lm_head'])
        self.lm_head_tt = to_ttnn(lm_new, device)
        self.lm_head_t = to_ttnn(lm_new.t().contiguous(), device)

        # wte
        self.wte_cpu = host_update('wte', self.wte_cpu, grads['wte'])

    def _adamw_step_device(self, grads, step, lr, beta1, beta2, wd, attr_map):
        """Device-side AdamW using float32 master weights."""
        device = self.device
        C, H, V = self.config.n_embd, self.config.mlp_hidden, self.config.vocab_size
        constants = make_adamw_constants(lr, beta1, beta2, wd, step + 1, device)

        for i in range(self.config.n_layer):
            for wname in ['w_q', 'w_k', 'w_v', 'w_proj', 'w_fc', 'w_mlp_proj']:
                gname = f'layer.{i}.{wname}'
                g_tt = to_ttnn_f32(grads[gname], device)
                w_list = getattr(self, attr_map[wname][0])
                wt_list = getattr(self, attr_map[wname][1])
                mw = self.master_weights[gname]
                adamw_kernel(mw, g_tt, self.m_state[gname],
                             self.v_state[gname], mw,
                             self.m_state[gname], self.v_state[gname],
                             constants['b1'], constants['omb1'],
                             constants['b2'], constants['omb2'],
                             constants['nlbc'], constants['b2c'],
                             constants['decay'], constants['eps'])
                # Cast back to bf16 for forward/backward
                w_cpu = ttnn.to_torch(mw).reshape(grads[gname].shape)
                w_list[i] = to_ttnn(w_cpu.to(torch.bfloat16), device)
                wt_list[i] = to_ttnn(w_cpu.t().contiguous().to(torch.bfloat16), device)

        # lm_head on device
        g_tt = to_ttnn_f32(grads['lm_head'], device)
        mw = self.master_weights['lm_head']
        adamw_kernel(mw, g_tt, self.m_state['lm_head'],
                     self.v_state['lm_head'], mw,
                     self.m_state['lm_head'], self.v_state['lm_head'],
                     constants['b1'], constants['omb1'],
                     constants['b2'], constants['omb2'],
                     constants['nlbc'], constants['b2c'],
                     constants['decay'], constants['eps'])
        lm_cpu = ttnn.to_torch(mw).reshape(V, C)
        self.lm_head_tt = to_ttnn(lm_cpu.to(torch.bfloat16), device)
        self.lm_head_t = to_ttnn(lm_cpu.t().contiguous().to(torch.bfloat16), device)

        # wte on host (float32)
        g = grads['wte'].float()
        m, v = self.m_state['wte'], self.v_state['wte']
        m[:] = beta1 * m + (1 - beta1) * g
        v[:] = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))
        self.wte_cpu = (self.wte_cpu.float() * (1 - lr * wd) -
                        lr * m_hat / (v_hat.sqrt() + 1e-8)).to(torch.bfloat16)


def test_training_attention(device):
    """Test training attention kernel against PyTorch reference."""
    print("\n" + "=" * 60)
    print("Testing training attention kernel")
    print("=" * 60)

    n_head = 6
    T = 64  # small for quick testing (2 tiles)
    hd = 128
    seq_tiles = T // TILE
    qk_scale = 1.15
    scale = qk_scale ** 2 / math.sqrt(hd)

    torch.manual_seed(42)
    Q = torch.randn(n_head, T, hd, dtype=torch.float32)
    K = torch.randn(n_head, T, hd, dtype=torch.float32)
    V = torch.randn(n_head, T, hd, dtype=torch.float32)

    # PyTorch reference
    scores = Q @ K.transpose(-2, -1) * scale
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask, float('-inf'))
    P = F.softmax(scores, dim=-1)
    ref = (P @ V).to(torch.bfloat16)

    # Device kernel
    Q_tt = to_ttnn(Q.to(torch.bfloat16).reshape(n_head * T, hd), device)
    K_tt = to_ttnn(K.to(torch.bfloat16).reshape(n_head * T, hd), device)
    V_tt = to_ttnn(V.to(torch.bfloat16).reshape(n_head * T, hd), device)
    out_tt = to_ttnn(torch.zeros(n_head * T, hd, dtype=torch.bfloat16), device)

    scale_tile = to_ttnn_l1(torch.full((TILE, TILE), scale, dtype=torch.bfloat16), device)
    scaler = to_ttnn_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    neg_inf = to_ttnn_l1(torch.full((TILE, TILE), float('-inf'), dtype=torch.bfloat16), device)
    zero_tile = to_ttnn_l1(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
    zero_head = to_ttnn_l1(torch.zeros(TILE, hd, dtype=torch.bfloat16), device)
    causal = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    causal[torch.triu(torch.ones(TILE, TILE, dtype=torch.bool), diagonal=1)] = float('-inf')
    causal_mask = to_ttnn_l1(causal, device)

    kernel = make_training_attention_kernel(n_head, seq_tiles)
    print("  Kernel compiled, running...")
    kernel(Q_tt, K_tt, V_tt, scale_tile, scaler, neg_inf,
           zero_tile, zero_head, causal_mask, out_tt)

    result = ttnn.to_torch(out_tt).reshape(n_head, T, hd)
    max_err = (result.float() - ref.float()).abs().max().item()
    mean_err = (result.float() - ref.float()).abs().mean().item()
    print(f"  shape: ({n_head}, {T}, {hd})")
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    assert mean_err < 0.5, f"Attention mean error too large: {mean_err}"
    print("  PASS")


def test_full_forward(device):
    """Test full forward pass on d1: TT loss matches PyTorch."""
    print("\n" + "=" * 60)
    print("Testing full forward pass (d1, T=128)")
    print("=" * 60)

    config = D1_CONFIG
    T = 128
    torch.manual_seed(42)

    model = PytorchRefModel(config, dtype=torch.float32)
    for layer in model.layers:
        layer['w_proj'] = torch.randn_like(layer['w_proj']) * 0.01
        layer['w_mlp_proj'] = torch.randn_like(layer['w_mlp_proj']) * 0.01

    input_ids = torch.randint(0, config.vocab_size, (1, T))
    targets = torch.randint(0, config.vocab_size, (1, T))

    ref_loss, _ = model.forward(input_ids, targets)
    print(f"  PyTorch loss: {ref_loss.item():.4f}")

    state = TrainingState(config, model, device, T=T)
    tt_loss, _, _, _ = state.forward(input_ids, targets)
    print(f"  TT loss:      {tt_loss:.4f}")
    print(f"  diff:         {abs(tt_loss - ref_loss.item()):.4f}")
    assert abs(tt_loss - ref_loss.item()) < 2.0, \
        f"TT loss {tt_loss} too far from PyTorch {ref_loss.item()}"
    print("  PASS")


def test_training(device, config=D1_CONFIG, T=128, n_steps=5, label="d1"):
    """Test training: loss decreases over steps."""
    print("\n" + "=" * 60)
    print(f"Testing training loop ({label}, T={T}, {n_steps} steps)")
    print("=" * 60)
    lr = 1e-3
    beta1, beta2, wd = 0.9, 0.999, 0.01
    torch.manual_seed(42)

    model = PytorchRefModel(config, dtype=torch.float32)
    for layer in model.layers:
        layer['w_proj'] = torch.randn_like(layer['w_proj']) * 0.01
        layer['w_mlp_proj'] = torch.randn_like(layer['w_mlp_proj']) * 0.01

    input_ids = torch.randint(0, config.vocab_size, (1, T))
    targets = torch.randint(0, config.vocab_size, (1, T))

    state = TrainingState(config, model, device, T=T)

    losses = []
    for step in range(n_steps):
        loss, saved_x, x0_cpu, dlogits = state.forward(input_ids, targets)
        losses.append(loss)
        print(f"  step {step}: loss={loss:.4f}")

        grads = state.backward(saved_x, x0_cpu, dlogits, input_ids)
        state.adamw_step(grads, step, lr=lr)

    print(f"\n  losses: {[f'{l:.4f}' for l in losses]}")
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("  Loss decreased: PASS")


def test_adamw_kernel(device):
    """Test AdamW kernel against PyTorch reference."""
    print("\n" + "=" * 60)
    print("Testing AdamW kernel")
    print("=" * 60)

    torch.manual_seed(42)
    rows, cols = 64, 64  # 2x2 tiles
    param = torch.randn(rows, cols, dtype=torch.bfloat16)
    grad = torch.randn(rows, cols, dtype=torch.bfloat16)
    m = torch.zeros(rows, cols, dtype=torch.float32)
    v = torch.zeros(rows, cols, dtype=torch.float32)

    lr, beta1, beta2, wd = 1e-3, 0.9, 0.999, 0.01
    step = 1

    # PyTorch reference (float32)
    p_ref = param.float().clone()
    g_ref = grad.float()
    m_ref = m.clone()
    v_ref = v.clone()
    m_ref = beta1 * m_ref + (1 - beta1) * g_ref
    v_ref = beta2 * v_ref + (1 - beta2) * g_ref * g_ref
    m_hat = m_ref / (1 - beta1 ** step)
    v_hat = v_ref / (1 - beta2 ** step)
    p_ref = p_ref * (1 - lr * wd) - lr * m_hat / (v_hat.sqrt() + 1e-8)

    # Device kernel
    p_tt = to_ttnn(param, device)
    g_tt = to_ttnn(grad, device)
    m_tt = to_ttnn(m.to(torch.bfloat16), device)
    v_tt = to_ttnn(v.to(torch.bfloat16), device)
    po_tt = to_ttnn(torch.zeros(rows, cols, dtype=torch.bfloat16), device)
    mo_tt = to_ttnn(torch.zeros(rows, cols, dtype=torch.bfloat16), device)
    vo_tt = to_ttnn(torch.zeros(rows, cols, dtype=torch.bfloat16), device)
    constants = make_adamw_constants(lr, beta1, beta2, wd, step, device)
    adamw_kernel(p_tt, g_tt, m_tt, v_tt, po_tt, mo_tt, vo_tt,
                 constants['b1'], constants['omb1'],
                 constants['b2'], constants['omb2'],
                 constants['nlbc'], constants['b2c'],
                 constants['decay'], constants['eps'])

    p_out = ttnn.to_torch(po_tt).reshape(rows, cols).float()
    m_out = ttnn.to_torch(mo_tt).reshape(rows, cols).float()
    v_out = ttnn.to_torch(vo_tt).reshape(rows, cols).float()

    p_err = (p_out - p_ref).abs().max().item()
    m_err = (m_out - m_ref.to(torch.bfloat16).float()).abs().max().item()
    v_err = (v_out - v_ref.to(torch.bfloat16).float()).abs().max().item()

    # Check weight update magnitude
    p_delta_ref = (p_ref - param.float()).abs().mean().item()
    p_delta_dev = (p_out - param.float()).abs().mean().item()

    print(f"  param max_err: {p_err:.6f}")
    print(f"  m max_err: {m_err:.6f}")
    print(f"  v max_err: {v_err:.6f}")
    print(f"  ref weight update magnitude: {p_delta_ref:.6f}")
    print(f"  dev weight update magnitude: {p_delta_dev:.6f}")
    print(f"  ref param sample: {p_ref.flatten()[:5].tolist()}")
    print(f"  dev param sample: {p_out.flatten()[:5].tolist()}")

    assert p_err < 0.01, f"AdamW param error too large: {p_err}"
    print("  PASS")


def test_backward_triage(device, config=D1_CONFIG, T=128):
    """Compare device backward gradients against PyTorch autograd."""
    print("\n" + "=" * 60)
    print(f"Backward triage: device vs autograd (d{config.n_layer}, T={T})")
    print("=" * 60)

    torch.manual_seed(42)
    model = PytorchRefModel(config, dtype=torch.float32)
    for layer in model.layers:
        layer['w_proj'] = torch.randn_like(layer['w_proj']) * 0.01
        layer['w_mlp_proj'] = torch.randn_like(layer['w_mlp_proj']) * 0.01

    input_ids = torch.randint(0, config.vocab_size, (1, T))
    targets = torch.randint(0, config.vocab_size, (1, T))

    # --- PyTorch autograd reference ---
    ref_loss, ref_grads = model.forward_backward(input_ids, targets)
    print(f"  PyTorch ref loss: {ref_loss:.6f}")

    # --- Device backward ---
    state = TrainingState(config, model, device, T=T)
    loss, saved_x, x0_cpu, dlogits = state.forward(input_ids, targets)
    print(f"  Device fwd loss:  {loss:.6f}")
    dev_grads = state.backward(saved_x, x0_cpu, dlogits, input_ids)

    # --- Compare ---
    print(f"\n  {'gradient':30s} {'ref_norm':>10s} {'dev_norm':>10s} {'diff_norm':>10s} {'rel_err':>10s}")
    print("  " + "-" * 75)
    all_ok = True
    for name in sorted(ref_grads.keys()):
        if name in ('resid_lambdas', 'x0_lambdas'):
            continue
        ref_g = ref_grads[name].float()
        dev_g = dev_grads.get(name)
        if dev_g is None:
            print(f"  {name:30s} MISSING from device backward")
            all_ok = False
            continue
        dev_g = dev_g.float()
        if ref_g.shape != dev_g.shape:
            print(f"  {name:30s} shape mismatch: ref={list(ref_g.shape)} dev={list(dev_g.shape)}")
            all_ok = False
            continue
        rn = ref_g.norm().item()
        dn = dev_g.norm().item()
        diff = (ref_g - dev_g).norm().item()
        rel = diff / max(rn, 1e-8)
        status = "OK" if rel < 0.1 else "BAD" if rel < 1.0 else "WRONG"
        if status != "OK":
            all_ok = False
        print(f"  {name:30s} {rn:10.4f} {dn:10.4f} {diff:10.4f} {rel:10.4f}  {status}")

    if all_ok:
        print("\n  All gradients match (rel_err < 0.1): PASS")
    else:
        print("\n  Some gradients diverge: FAIL")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_training_attention(device)
        test_training(device, D1_CONFIG, T=128, n_steps=5, label="d1")
        test_training(device, D12_CONFIG, T=128, n_steps=5, label="d12")
        test_training(device, D12_CONFIG, T=2048, n_steps=3, label="d12-T2048")
    finally:
        ttnn.close_device(device)
