# Scaled Dot-Product Attention for nanochat inference.
#
# Adapted from reference SDPA kernel to current TT-Lang API.
# Supports nanochat's head_dim=128 (4 tiles).
#
# Algorithm (Flash Attention simplified, single K chunk):
#   1. QK = Q @ K^T
#   2. QK_scaled = QK * (1/sqrt(d_k))
#   3. max = reduce_max(QK_scaled, row-wise)
#   4. exp_scores = exp(QK_scaled - max)
#   5. sum = reduce_sum(exp_scores, row-wise)
#   6. attn = exp(QK_scaled - max) / sum
#   7. out = attn @ V
#
# No causal mask (for decode with KV cache, all cached tokens are valid).
# Single core, fits small-to-medium sequences in L1.

import math
import torch
import ttnn
import ttl

TILE = 32
HEAD_DIM = 128
HEAD_TILES = HEAD_DIM // TILE  # 4


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@ttl.operation(grid=(1, 1))
def sdpa_kernel(Q, K, V, scale_tile, scaler, out):
    """SDPA: out = softmax(Q @ K^T / sqrt(d)) @ V

    Q: (seq_q, head_dim) in elements
    K: (seq_kv, head_dim)
    V: (seq_kv, head_dim)
    scale_tile: (1, 1) tile filled with 1/sqrt(head_dim)
    scaler: (1, 1) tile of all 1.0s
    out: (seq_q, head_dim)
    """
    sq_tiles = Q.shape[0] // TILE
    skv_tiles = K.shape[0] // TILE

    # Input DFBs
    q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(sq_tiles, HEAD_TILES), buffer_factor=1)
    k_dfb = ttl.make_dataflow_buffer_like(K, shape=(skv_tiles, HEAD_TILES), buffer_factor=1)
    v_dfb = ttl.make_dataflow_buffer_like(V, shape=(skv_tiles, HEAD_TILES), buffer_factor=1)
    scale_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), buffer_factor=1)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)

    # Intermediates
    kt_dfb = ttl.make_dataflow_buffer_like(K, shape=(HEAD_TILES, skv_tiles), buffer_factor=2)
    qk_dfb = ttl.make_dataflow_buffer_like(Q, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    scale_row_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(sq_tiles, 1), buffer_factor=2)
    scale_bcast_dfb = ttl.make_dataflow_buffer_like(Q, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    scaled_dfb = ttl.make_dataflow_buffer_like(Q, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    # Reduce outputs are (sq_tiles, 1) - one column after row-wise reduce
    max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(sq_tiles, 1), buffer_factor=2)
    max_bcast_dfb = ttl.make_dataflow_buffer_like(Q, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    exp_dfb = ttl.make_dataflow_buffer_like(Q, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(sq_tiles, 1), buffer_factor=2)
    sum_bcast_dfb = ttl.make_dataflow_buffer_like(Q, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(sq_tiles, HEAD_TILES), buffer_factor=1)

    @ttl.compute()
    def compute():
        # Step 1: K^T = transpose(K)
        with k_dfb.wait() as kv, kt_dfb.reserve() as kt:
            kt.store(ttl.transpose(kv))

        # Step 2: QK = Q @ K^T
        with q_dfb.wait() as qv, kt_dfb.wait() as ktv, qk_dfb.reserve() as qk:
            qk.store(qv @ ktv)

        # Step 3: QK_scaled = QK * scale (two-step broadcast: rows then cols)
        with scale_dfb.wait() as s, scale_row_dfb.reserve() as sr:
            sr.store(ttl.math.broadcast(s, sr, dims=[0]))
        with scale_row_dfb.wait() as sr, scale_bcast_dfb.reserve() as sb:
            sb.store(ttl.math.broadcast(sr, sb, dims=[1]))
        with scale_bcast_dfb.wait() as sb, qk_dfb.wait() as qkv, scaled_dfb.reserve() as scd:
            scd.store(sb * qkv)

        # Steps 4-7: Row-wise softmax + attn @ V
        with scaled_dfb.wait() as sdv, sc_dfb.wait() as sc:
            # 4. max = reduce_max(scaled, row-wise)
            with max_dfb.reserve() as mx:
                mx.store(ttl.math.reduce_max(sdv, sc, dims=[1]))

            # 5. Broadcast max for subtraction
            with max_dfb.wait() as mxv, max_bcast_dfb.reserve() as mxb:
                mxb.store(ttl.math.broadcast(mxv, mxb, dims=[1]))

            with max_bcast_dfb.wait() as mxbv:
                # 6. exp(scaled - max)
                with exp_dfb.reserve() as ex:
                    ex.store(ttl.math.exp(sdv - mxbv))

                # 7. sum = reduce_sum(exp, row-wise)
                with exp_dfb.wait() as exv, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(exv, sc, dims=[1]))

                # 8. Broadcast sum
                with sum_dfb.wait() as smv, sum_bcast_dfb.reserve() as smb:
                    smb.store(ttl.math.broadcast(smv, smb, dims=[1]))

                # 9. attn_weights = exp(scaled - max) / sum
                with sum_bcast_dfb.wait() as smbv, qk_dfb.reserve() as attn:
                    attn.store(ttl.math.exp(sdv - mxbv) / smbv)

        # Step 10: out = attn @ V
        with qk_dfb.wait() as av, v_dfb.wait() as vv, out_dfb.reserve() as o:
            o.store(av @ vv)

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(Q[0:sq_tiles, 0:HEAD_TILES], blk); tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(K[0:skv_tiles, 0:HEAD_TILES], blk); tx.wait()
        with v_dfb.reserve() as blk:
            tx = ttl.copy(V[0:skv_tiles, 0:HEAD_TILES], blk); tx.wait()
        with scale_dfb.reserve() as blk:
            tx = ttl.copy(scale_tile[0, 0], blk); tx.wait()
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:sq_tiles, 0:HEAD_TILES]); tx.wait()


def torch_sdpa(q, k, v, is_causal=False):
    """PyTorch reference SDPA."""
    q4 = q.float().unsqueeze(0).unsqueeze(0)
    k4 = k.float().unsqueeze(0).unsqueeze(0)
    v4 = v.float().unsqueeze(0).unsqueeze(0)
    out = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, is_causal=is_causal)
    return out.squeeze(0).squeeze(0)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test 1: Small - 32 tokens, head_dim=128
    seq_len = 32
    print(f"=== Test 1: SDPA (seq={seq_len}, head_dim={HEAD_DIM}) ===")
    q = torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16)
    expected = torch_sdpa(q, k, v).to(torch.bfloat16)

    scale_val = 1.0 / math.sqrt(HEAD_DIM)
    scale_tt = to_ttnn(torch.full((TILE, TILE), scale_val, dtype=torch.bfloat16), device)
    scaler_tt = to_ttnn(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    out_tt = to_ttnn(torch.zeros(seq_len, HEAD_DIM, dtype=torch.bfloat16), device)

    sdpa_kernel(to_ttnn(q, device), to_ttnn(k, device), to_ttnn(v, device),
                scale_tt, scaler_tt, out_tt)
    result = ttnn.to_torch(out_tt)

    err = (expected.float() - result.float()).abs().max().item()
    mean_err = (expected.float() - result.float()).abs().mean().item()
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    print(f"  Expected[0,:4]: {expected[0,:4]}")
    print(f"  Got[0,:4]:      {result[0,:4]}")
    print(f"  Max error: {err:.4f}, Mean: {mean_err:.4f}, PCC: {pcc:.6f}")
    print(f"  {'PASS' if pcc > 0.99 else 'FAIL'}")

    # Test 2: Decode-like - Q=32 tokens, K/V=64 tokens (non-square attention)
    sq2, skv2 = 32, 64
    print(f"\n=== Test 2: SDPA decode (Q={sq2}, KV={skv2}, head_dim={HEAD_DIM}) ===")
    q2 = torch.randn(sq2, HEAD_DIM, dtype=torch.bfloat16)
    k2 = torch.randn(skv2, HEAD_DIM, dtype=torch.bfloat16)
    v2 = torch.randn(skv2, HEAD_DIM, dtype=torch.bfloat16)
    expected2 = torch_sdpa(q2, k2, v2).to(torch.bfloat16)
    out2 = to_ttnn(torch.zeros(sq2, HEAD_DIM, dtype=torch.bfloat16), device)

    sdpa_kernel(to_ttnn(q2, device), to_ttnn(k2, device), to_ttnn(v2, device),
                scale_tt, scaler_tt, out2)
    result2 = ttnn.to_torch(out2)
    err2 = (expected2.float() - result2.float()).abs().max().item()
    pcc2 = torch.corrcoef(torch.stack([result2.float().flatten(), expected2.float().flatten()]))[0, 1].item()
    print(f"  Expected[0,:4]: {expected2[0,:4]}")
    print(f"  Got[0,:4]:      {result2[0,:4]}")
    print(f"  Max error: {err2:.4f}, PCC: {pcc2:.6f}")
    print(f"  {'PASS' if pcc2 > 0.99 else 'FAIL'}")

    # Test 3: Larger decode - Q=32, K/V=128 (4 tile rows of cache)
    sq3, skv3 = 32, 128
    print(f"\n=== Test 3: SDPA decode (Q={sq3}, KV={skv3}, head_dim={HEAD_DIM}) ===")
    q3 = torch.randn(sq3, HEAD_DIM, dtype=torch.bfloat16)
    k3 = torch.randn(skv3, HEAD_DIM, dtype=torch.bfloat16)
    v3 = torch.randn(skv3, HEAD_DIM, dtype=torch.bfloat16)
    expected3 = torch_sdpa(q3, k3, v3).to(torch.bfloat16)
    out3 = to_ttnn(torch.zeros(sq3, HEAD_DIM, dtype=torch.bfloat16), device)

    sdpa_kernel(to_ttnn(q3, device), to_ttnn(k3, device), to_ttnn(v3, device),
                scale_tt, scaler_tt, out3)
    result3 = ttnn.to_torch(out3)
    err3 = (expected3.float() - result3.float()).abs().max().item()
    pcc3 = torch.corrcoef(torch.stack([result3.float().flatten(), expected3.float().flatten()]))[0, 1].item()
    print(f"  Max error: {err3:.4f}, PCC: {pcc3:.6f}")
    print(f"  {'PASS' if pcc3 > 0.99 else 'FAIL'}")

    ttnn.close_device(device)
