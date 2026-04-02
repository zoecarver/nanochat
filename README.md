# nanochat TT-Lang Inference

Autoregressive inference for the nanochat d32 (~1.1B parameter) LLM on Tenstorrent hardware, written almost entirely in [TT-Lang](https://docs.tenstorrent.com/tt-lang/index.html).

All compute kernels are in [`ttlang/inference.py`](ttlang/inference.py). The only TTNN ops used are `ttnn.embedding` and `ttnn.kv_cache.update_cache_for_token_`.

## TT-Lang Kernels

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

## Kernel Fusion Examples

Two fusions that gave significant wall-time speedups:

**Fused QKV projections** ([`9034746`](https://github.com/zoecarver/nanochat/commit/90347464dc8c743ff2a32590319b4fedc3f55a33), +6.7%): The three QKV matmuls share the same input. The fused kernel reads the input once into a DFB, then sequentially multiplies by each weight matrix, writing three outputs. Uses NCOLS=2 and buffer_factor=1 to fit 7 DFBs in L1.

**Fused MLP projection** ([`f849d3f`](https://github.com/zoecarver/nanochat/commit/f849d3ff20d29b79c4aba79440e865e846219795), +21%): Replaced 4 separate slice matmul kernels + 3 residual add kernels (7 dispatches per layer) with a single kernel. Loops over 4 K-chunks of the (8192, 2048) weight matrix, accumulating partial matmul results in L1 via ping-pong DFBs. The intermediate partial sums never touch DRAM.

Together these fusions improved decode throughput from 12.3 tok/s to 15.9 tok/s (+29%).
