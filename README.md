# nanochat TT-Lang

Inference and training for the nanochat LLM on Tenstorrent hardware, written in [TT-Lang](https://docs.tenstorrent.com/tt-lang/index.html).

## Inference

Autoregressive decode for d32 (~1.1B parameters). All compute kernels are in [`ttlang/inference.py`](ttlang/inference.py). The only TTNN ops used are `ttnn.embedding` and `ttnn.kv_cache.update_cache_for_token_`.

### TT-Lang Kernels

- `rmsnorm_kernel` -- two-pass RMSNorm (sum-of-squares reduction, then normalize)
- `linear_kernel` -- matrix multiply with streaming K-accumulation
- `triple_linear_kernel` -- fused QKV projection (reads input once, computes 3 matmuls)
- `fused_mlp_proj_kernel` -- fused MLP output projection with L1 ping-pong accumulation across 4 K-chunks
- `relu_sq_kernel` -- elementwise ReLU squared activation
- `rotary_kernel` -- rotary position embeddings with cos/sin broadcast
- `reshape_to_heads` / `reshape_from_heads` -- head-batched layout transforms
- `flash_attention` -- single-kernel online flash attention (16 cores, one per head)
- `scaled_residual_kernel` -- weighted residual connection (lambda_r * x + lambda_0 * x0)
- `residual_add_kernel` -- elementwise addition
- `ve_gated_add_kernel` -- gated value embedding addition
- `copy_kernel` -- device-side tensor copy
- `softcap_kernel` -- logit soft-capping (tanh-based)

### Kernel Fusion Examples

Two fusions that gave significant wall-time speedups:

**Fused QKV projections** ([`9034746`](https://github.com/zoecarver/nanochat/commit/90347464dc8c743ff2a32590319b4fedc3f55a33), +6.7%): The three QKV matmuls share the same input. The fused kernel reads the input once into a DFB, then sequentially multiplies by each weight matrix, writing three outputs. Uses NCOLS=2 and buffer_factor=1 to fit 7 DFBs in L1.

**Fused MLP projection** ([`f849d3f`](https://github.com/zoecarver/nanochat/commit/f849d3ff20d29b79c4aba79440e865e846219795), +21%): Replaced 4 separate slice matmul kernels + 3 residual add kernels (7 dispatches per layer) with a single kernel. Loops over 4 K-chunks of the (8192, 2048) weight matrix, accumulating partial matmul results in L1 via ping-pong DFBs. The intermediate partial sums never touch DRAM.

Together these fusions improved decode throughput from 12.3 tok/s to 15.9 tok/s (+29%).

## Training

Single-file training for d12 (768-dim, 6 heads, 12 layers) at T=2048. All forward and backward kernels are in [`ttlang/train.py`](ttlang/train.py). TTNN is used only for `ttnn.embedding` (forward) and host-side loss/dlogits computation. Weight gradients (dW) are computed on host; all dx-path gradients run on device.

### Architecture

```
Forward:  embedding [host] -> per-layer device kernels -> lm_head + loss [host]
Backward: dlogits [host] -> per-layer device kernels -> embedding backward [host]
Optimizer: AdamW with float32 master weights [device]
```

### TT-Lang Kernels (Forward)

- `linear_kernel` -- streaming matmul (reused for all projections)
- `rmsnorm_kernel` -- two-pass RMSNorm (pre-attention + pre-MLP + QK norm)
- `relu_sq_kernel` -- elementwise ReLU squared
- `rotary_training_kernel` -- rotary embeddings for full sequence (per-position cos/sin)
- `reshape_to_heads` / `reshape_from_heads` -- (T, C) <-> (n_head*T, head_dim)
- `training_attention` -- flash attention with online softmax (1 core per head, saves m/l for backward)
- `scaled_residual_kernel` / `residual_add_kernel` -- residual connections
- `softcap_kernel` -- logit soft-capping

### TT-Lang Kernels (Backward)

- `training_attention_backward` -- FA2-style backward (recomputes P block-by-block from saved m/l, accumulates dK/dV in L1, dQ via DRAM read-modify-write)
- `rmsnorm_backward_kernel` -- RMSNorm backward (pre-attention, pre-MLP, QK norm)
- `relu_sq_backward_kernel` -- ReLU squared backward
- `rotary_backward_kernel` -- reverse rotation
- `linear_backward_dw_kernel` -- dW = X^T @ dY with K-chunked accumulation

### TT-Lang Kernels (Optimizer)

- `adamw_kernel` -- AdamW with float32 master weights and optimizer state, bf16 gradient input
