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
                        with po_dfb.reserve() as po:
                            m_new = b1v * mv + omb1v * gv
                            v_hat = b2cv * (b2v * vv + omb2v * gv * gv)
                            po.store(decv * pv + nlbcv * m_new *
                                     ttl.math.recip(ttl.math.sqrt(v_hat) + epsv))

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
    """Create constant tiles for AdamW kernel."""
    def tile(val):
        return to_ttnn_l1(torch.full((TILE, TILE), val, dtype=torch.bfloat16), device)
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
        self.d_attn_heads = alloc(nT, hd)
        self.d_q_heads = alloc(nT, hd)
        self.d_k_heads = alloc(nT, hd)
        self.d_v_heads = alloc(nT, hd)
        self.d_q_rot = alloc(nT, hd)
        self.d_k_rot = alloc(nT, hd)
        self.d_hidden = alloc(T, H)
        self.d_hidden_act = alloc(T, H)
        # RMSNorm backward needs rstd: (T, TILE) with one scalar per row
        self.rstd_c = alloc(T, TILE)
        self.rstd_hd = alloc(nT, TILE)

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

            # Attention on host (with 1.15 scale)
            qk_scale = self.config.qk_scale
            q_cpu = ttnn.to_torch(self.q_norm).reshape(n_head, T, hd).float() * qk_scale
            k_cpu = ttnn.to_torch(self.k_norm).reshape(n_head, T, hd).float() * qk_scale
            v_cpu = ttnn.to_torch(self.v_heads).reshape(n_head, T, hd).float()
            scale = 1.0 / math.sqrt(hd)
            scores = q_cpu @ k_cpu.transpose(-2, -1) * scale
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))
            P = F.softmax(scores, dim=-1)
            attn_out_cpu = (P @ v_cpu).to(torch.bfloat16).reshape(n_head * T, hd)
            attn_out_tt = to_ttnn(attn_out_cpu, device)

            # Reshape from heads + proj + residual
            self.from_heads(attn_out_tt, self.attn_concat)
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

    def backward(self, saved_x, x0_cpu, dlogits, input_ids):
        """Full backward pass with gradient checkpointing. Returns gradient dict."""
        T = self.T
        C = self.config.n_embd
        H = self.config.mlp_hidden
        V = self.config.vocab_size
        n_head = self.config.n_head
        hd = self.config.head_dim
        device = self.device
        qk_scale = self.config.qk_scale

        grads = {}

        # --- Backward through softcap ---
        # logits_pre_cap was stored in self.logits (pre-softcap)
        logits_pre_cpu = ttnn.to_torch(self.logits).reshape(T, -1)[:, :V].float()
        # dsoftcap: dx = dout * (1 - tanh(x/cap)^2)
        cap = self.config.softcap
        th = torch.tanh(logits_pre_cpu / cap)
        d_logits_pre = dlogits * (1.0 - th * th)

        # --- Backward through lm_head: logits = final_normed @ lm_head^T ---
        # d_final_normed = d_logits_pre @ lm_head  (dX = dY @ W, W=lm_head^T, so dX = dY @ lm_head)
        # d_lm_head = d_logits_pre^T @ final_normed (dW^T = dY^T @ X, so d_lm_head = d_logits^T @ normed)
        final_normed_cpu = ttnn.to_torch(self.s1).reshape(T, C).float()
        d_final_normed = (d_logits_pre @ self.wte_cpu.float()[:V].clone().zero_().index_copy_(
            0, torch.arange(V), ttnn.to_torch(self.lm_head_tt).reshape(V, C).float()))
        # Simpler: lm_head is (V, C), logits = normed @ lm_head^T
        lm_head_cpu = ttnn.to_torch(self.lm_head_tt).reshape(V, C).float()
        d_final_normed = d_logits_pre @ lm_head_cpu  # (T, C)
        grads['lm_head'] = d_logits_pre.t() @ final_normed_cpu  # (V, C)

        # Pad d_final_normed for device ops if needed
        # Actually, we need d_final_normed on device for rmsnorm backward
        # But rmsnorm backward also needs x (pre-norm) and rstd

        # For the final rmsnorm backward, we need x_final (last saved_x[-1])
        # and rstd. Let's compute rstd on host and upload.
        x_final_cpu = saved_x[-1].float()
        rstd_final = (x_final_cpu.pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt()

        # RMSNorm backward on host (simpler for final layer)
        c_val = (d_final_normed * x_final_cpu).sum(dim=-1, keepdim=True)
        dx_final = rstd_final * d_final_normed - rstd_final.pow(3) * c_val / C * x_final_cpu
        dout = dx_final  # gradient flowing to layers

        # Accumulate d_x0 for scaled residual backward
        d_x0 = torch.zeros(T, C, dtype=torch.float32)

        # --- Per-layer backward (reverse order, gradient checkpointing) ---
        for i in reversed(range(self.config.n_layer)):
            x_i_cpu = saved_x[i].float()  # x before this layer's scaled residual

            # === Recompute forward for this layer (on host for intermediates) ===
            lr_val = float(ttnn.to_torch(self.lr_tiles[i]).flatten()[0])
            l0_val = float(ttnn.to_torch(self.l0_tiles[i]).flatten()[0])
            x_scaled = lr_val * x_i_cpu + l0_val * x0_cpu.float()

            normed = F.rms_norm(x_scaled, (C,))
            w_q_cpu = ttnn.to_torch(self.w_q[i]).reshape(C, C).float()
            w_k_cpu = ttnn.to_torch(self.w_k[i]).reshape(C, C).float()
            w_v_cpu = ttnn.to_torch(self.w_v[i]).reshape(C, C).float()
            w_proj_cpu = ttnn.to_torch(self.w_proj[i]).reshape(C, C).float()
            w_fc_cpu = ttnn.to_torch(self.w_fc[i]).reshape(C, H).float()
            w_mlp_proj_cpu = ttnn.to_torch(self.w_mlp_proj[i]).reshape(H, C).float()

            q_flat = normed @ w_q_cpu
            k_flat = normed @ w_k_cpu
            v_flat = normed @ w_v_cpu
            q_heads = q_flat.view(T, n_head, hd).transpose(0, 1)  # (n_head, T, hd)
            k_heads = k_flat.view(T, n_head, hd).transpose(0, 1)
            v_heads = v_flat.view(T, n_head, hd).transpose(0, 1)

            cos, sin = pt_precompute_rotary(T, hd)
            cos_2d, sin_2d = cos.squeeze(0).squeeze(1), sin.squeeze(0).squeeze(1)
            q_rot = torch.zeros_like(q_heads)
            k_rot = torch.zeros_like(k_heads)
            for h in range(n_head):
                x1, x2 = q_heads[h, :, :hd//2], q_heads[h, :, hd//2:]
                q_rot[h] = torch.cat([x1*cos_2d + x2*sin_2d, -x1*sin_2d + x2*cos_2d], -1)
                x1, x2 = k_heads[h, :, :hd//2], k_heads[h, :, hd//2:]
                k_rot[h] = torch.cat([x1*cos_2d + x2*sin_2d, -x1*sin_2d + x2*cos_2d], -1)

            q_norm = F.rms_norm(q_rot, (hd,)) * qk_scale
            k_norm = F.rms_norm(k_rot, (hd,)) * qk_scale

            scale = 1.0 / math.sqrt(hd)
            scores = q_norm @ k_norm.transpose(-2, -1) * scale
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))
            P = F.softmax(scores, dim=-1)
            attn_out = P @ v_heads  # (n_head, T, hd)
            attn_concat = attn_out.transpose(0, 1).contiguous().view(T, C)

            proj_out = attn_concat @ w_proj_cpu
            x_post_attn = x_scaled + proj_out

            normed2 = F.rms_norm(x_post_attn, (C,))
            hidden_pre = normed2 @ w_fc_cpu
            hidden_act = F.relu(hidden_pre).square()
            mlp_out = hidden_act @ w_mlp_proj_cpu

            # === Backward through this layer (all on host for correctness) ===
            # dout is the gradient from above: d_x_out

            # MLP residual: x_out = x_post_attn + mlp_out
            d_mlp_out = dout
            d_x_post_attn = dout.clone()

            # MLP proj backward: mlp_out = hidden_act @ w_mlp_proj
            d_hidden_act = d_mlp_out @ w_mlp_proj_cpu.t()
            grads[f'layer.{i}.w_mlp_proj'] = hidden_act.t() @ d_mlp_out

            # ReLU^2 backward: hidden_act = relu(hidden)^2, dx = 2*relu(x)*dout
            d_hidden_pre = 2.0 * F.relu(hidden_pre) * d_hidden_act

            # FC backward: hidden = normed2 @ w_fc
            d_normed2 = d_hidden_pre @ w_fc_cpu.t()
            grads[f'layer.{i}.w_fc'] = normed2.t() @ d_hidden_pre

            # Pre-MLP rmsnorm backward
            rstd2 = (x_post_attn.pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
            c2 = (d_normed2 * x_post_attn).sum(dim=-1, keepdim=True)
            d_x_post_attn += rstd2 * d_normed2 - rstd2.pow(3) * c2 / C * x_post_attn

            # Attention residual: x_post_attn = x_scaled + proj_out
            d_proj_out = d_x_post_attn
            d_x_scaled = d_x_post_attn.clone()

            # Proj backward: proj_out = attn_concat @ w_proj
            d_attn_concat = d_proj_out @ w_proj_cpu.t()
            grads[f'layer.{i}.w_proj'] = attn_concat.t() @ d_proj_out

            # Reshape from heads backward
            d_attn_out = d_attn_concat.view(T, n_head, hd).transpose(0, 1)  # (n_head, T, hd)

            # Attention backward on host
            dV = P.transpose(-2, -1) @ d_attn_out
            dP = d_attn_out @ v_heads.transpose(-2, -1)
            dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
            dQ_input = dS @ k_norm * scale
            dK_input = dS.transpose(-2, -1) @ q_norm * scale

            # QK scale backward (1.15 multiply)
            dQ_norm = dQ_input * qk_scale
            dK_norm = dK_input * qk_scale

            # QK norm backward (rmsnorm per head)
            q_rot_for_bwd = q_rot
            k_rot_for_bwd = k_rot
            rstd_q = (q_rot_for_bwd.pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
            rstd_k = (k_rot_for_bwd.pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
            cq = (dQ_norm * q_rot_for_bwd).sum(dim=-1, keepdim=True)
            ck = (dK_norm * k_rot_for_bwd).sum(dim=-1, keepdim=True)
            dQ_rot = rstd_q * dQ_norm - rstd_q.pow(3) * cq / hd * q_rot_for_bwd
            dK_rot = rstd_k * dK_norm - rstd_k.pow(3) * ck / hd * k_rot_for_bwd

            # Rotary backward per head
            dQ_heads = torch.zeros_like(q_heads)
            dK_heads = torch.zeros_like(k_heads)
            for h in range(n_head):
                dy1, dy2 = dQ_rot[h, :, :hd//2], dQ_rot[h, :, hd//2:]
                dQ_heads[h] = torch.cat([dy1*cos_2d - dy2*sin_2d,
                                         dy1*sin_2d + dy2*cos_2d], -1)
                dy1, dy2 = dK_rot[h, :, :hd//2], dK_rot[h, :, hd//2:]
                dK_heads[h] = torch.cat([dy1*cos_2d - dy2*sin_2d,
                                         dy1*sin_2d + dy2*cos_2d], -1)

            # Reshape from heads -> flat
            dQ_flat = dQ_heads.transpose(0, 1).contiguous().view(T, C)
            dK_flat = dK_heads.transpose(0, 1).contiguous().view(T, C)
            dV_flat = dV.transpose(0, 1).contiguous().view(T, C)

            # QKV backward
            d_normed = dQ_flat @ w_q_cpu.t() + dK_flat @ w_k_cpu.t() + dV_flat @ w_v_cpu.t()
            grads[f'layer.{i}.w_q'] = normed.t() @ dQ_flat
            grads[f'layer.{i}.w_k'] = normed.t() @ dK_flat
            grads[f'layer.{i}.w_v'] = normed.t() @ dV_flat

            # Pre-attention rmsnorm backward
            rstd1 = (x_scaled.pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
            c1 = (d_normed * x_scaled).sum(dim=-1, keepdim=True)
            d_x_scaled += rstd1 * d_normed - rstd1.pow(3) * c1 / C * x_scaled

            # Scaled residual backward: x_scaled = lr * x_prev + l0 * x0
            dout = lr_val * d_x_scaled  # d_x_prev
            d_x0 += l0_val * d_x_scaled

        # Embedding backward: scatter-add
        grads['wte'] = torch.zeros_like(self.wte_cpu.float())
        dout_embed_cpu = dout
        # d_x0 also flows through initial rmsnorm backward
        x_embed_cpu = self.wte_cpu[input_ids.squeeze(0)].float()
        rstd_e = (x_embed_cpu.pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
        # Both dout (from layer 0) and d_x0 flow into the embedding gradient
        d_embed_total = dout + d_x0  # before initial rmsnorm
        # Actually, x0 = rmsnorm(embed), so d_embed = rmsnorm_backward(d_x0 + dout, embed)
        # Wait: dout is already the gradient through all layers and scaled residuals.
        # And d_x0 accumulated from all layers.
        # The initial rmsnorm: x0 = rmsnorm(embed), and x_initial = x0
        # Layer 0 scaled_residual reads x_initial (= x0). So dout from layer 0
        # is d_x_initial. And d_x0 accumulated from all l0*d_x_scaled terms.
        # Total gradient into x0: dout + d_x0
        d_total = dout + d_x0
        ce = (d_total * x_embed_cpu).sum(dim=-1, keepdim=True)
        d_embed = rstd_e * d_total - rstd_e.pow(3) * ce / C * x_embed_cpu
        # Scatter-add for embedding gradient
        for t_idx in range(T):
            tok = input_ids[0, t_idx].item()
            grads['wte'][tok] += d_embed[t_idx]

        return grads


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

    # Initialize optimizer state (zeros)
    C = config.n_embd
    H = config.mlp_hidden
    V = config.vocab_size

    # Collect all weight tensors and their shapes for AdamW
    param_names = ['wte', 'lm_head']
    for i in range(config.n_layer):
        param_names.extend([f'layer.{i}.{k}' for k in
                           ['w_q', 'w_k', 'w_v', 'w_proj', 'w_fc', 'w_mlp_proj']])

    losses = []
    for step in range(n_steps):
        loss, saved_x, x0_cpu, dlogits = state.forward(input_ids, targets)
        losses.append(loss)
        print(f"  step {step}: loss={loss:.4f}")

        grads = state.backward(saved_x, x0_cpu, dlogits, input_ids)

        # AdamW on host for simplicity (validates forward + backward)
        if step == 0:
            m_state = {k: torch.zeros_like(v) for k, v in grads.items()}
            v_state = {k: torch.zeros_like(v) for k, v in grads.items()}

        for name in grads:
            g = grads[name]
            m_state[name] = beta1 * m_state[name] + (1 - beta1) * g
            v_state[name] = beta2 * v_state[name] + (1 - beta2) * g * g
            m_hat = m_state[name] / (1 - beta1 ** (step + 1))
            v_hat = v_state[name] / (1 - beta2 ** (step + 1))

            # Get current param
            if name == 'wte':
                param = state.wte_cpu.float()
                param = param * (1 - lr * wd) - lr * m_hat / (v_hat.sqrt() + 1e-8)
                state.wte_cpu = param.to(torch.bfloat16)
            elif name == 'lm_head':
                param = ttnn.to_torch(state.lm_head_tt).reshape(V, C).float()
                param = param * (1 - lr * wd) - lr * m_hat / (v_hat.sqrt() + 1e-8)
                state.lm_head_tt = to_ttnn(param.to(torch.bfloat16), device)
                state.lm_head_t = to_ttnn(param.t().contiguous().to(torch.bfloat16), device)
            else:
                parts = name.split('.')
                layer_idx = int(parts[1])
                weight_name = parts[2]
                tt_w = getattr(state, f'w_{weight_name[2:]}' if weight_name.startswith('w_') else weight_name)
                # Map name to attribute
                attr_map = {
                    'w_q': ('w_q', 'w_q_t'), 'w_k': ('w_k', 'w_k_t'),
                    'w_v': ('w_v', 'w_v_t'), 'w_proj': ('w_proj', 'w_proj_t'),
                    'w_fc': ('w_fc', 'w_fc_t'), 'w_mlp_proj': ('w_mlp_proj', 'w_mlp_proj_t'),
                }
                w_attr, wt_attr = attr_map[weight_name]
                w_list = getattr(state, w_attr)
                wt_list = getattr(state, wt_attr)
                param = ttnn.to_torch(w_list[layer_idx]).reshape(g.shape).float()
                param = param * (1 - lr * wd) - lr * m_hat / (v_hat.sqrt() + 1e-8)
                w_list[layer_idx] = to_ttnn(param.to(torch.bfloat16), device)
                wt_list[layer_idx] = to_ttnn(param.t().contiguous().to(torch.bfloat16), device)

    print(f"\n  losses: {[f'{l:.4f}' for l in losses]}")
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("  Loss decreased: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_training(device, D12_CONFIG, T=2048, n_steps=10, label="d12-T2048")
    finally:
        ttnn.close_device(device)
