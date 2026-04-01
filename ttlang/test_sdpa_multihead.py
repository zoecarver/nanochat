# Multi-head Scaled Dot-Product Attention kernel.
#
# grid=(16, 1): one core per attention head, all 16 heads in parallel.
# Eliminates per-head host round-trips (upload Q/K/V, download result).
#
# Input layout: heads stacked vertically.
#   Q_all: (NUM_HEADS * sq, head_dim)  -- each head's Q is (sq, head_dim)
#   K_all: (NUM_HEADS * skv, head_dim) -- each head's K cache
#   V_all: (NUM_HEADS * skv, head_dim)
#   out:   (NUM_HEADS * sq, head_dim)
#
# Each core h reads Q_all[h*sq:(h+1)*sq], K_all[h*skv:(h+1)*skv], etc.

import math
import torch
import ttnn
import ttl

TILE = 32
HEAD_DIM = 128
HEAD_TILES = HEAD_DIM // TILE  # 4
NUM_HEADS = 16


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


GRID_X = 8
GRID_Y = 2  # 8 * 2 = 16 cores = NUM_HEADS


@ttl.operation(grid=(GRID_X, GRID_Y))
def sdpa_multihead(Q_all, K_all, V_all, scale_tile, scaler, mask, out):
    """Multi-head SDPA: each core handles one head.

    Q_all: (NUM_HEADS * sq_tiles * TILE, head_dim)
    K_all: (NUM_HEADS * skv_tiles * TILE, head_dim)
    V_all: (NUM_HEADS * skv_tiles * TILE, head_dim)
    mask:  (sq_tiles * TILE, skv_tiles * TILE) -- shared across heads
    """
    total_q = Q_all.shape[0] // TILE
    sq_tiles = total_q // NUM_HEADS
    total_kv = K_all.shape[0] // TILE
    skv_tiles = total_kv // NUM_HEADS

    # Per-head DFBs (same sizes as single-head kernel)
    q_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(sq_tiles, HEAD_TILES), buffer_factor=1)
    k_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(skv_tiles, HEAD_TILES), buffer_factor=1)
    v_dfb = ttl.make_dataflow_buffer_like(V_all, shape=(skv_tiles, HEAD_TILES), buffer_factor=1)
    scale_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), buffer_factor=1)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    mask_dfb = ttl.make_dataflow_buffer_like(mask, shape=(sq_tiles, skv_tiles), buffer_factor=1)

    # Intermediates
    kt_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(HEAD_TILES, skv_tiles), buffer_factor=2)
    qk_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    scale_row_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(sq_tiles, 1), buffer_factor=2)
    scale_bcast_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    scaled_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(sq_tiles, 1), buffer_factor=2)
    max_bcast_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    exp_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(sq_tiles, 1), buffer_factor=2)
    sum_bcast_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(sq_tiles, skv_tiles), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(sq_tiles, HEAD_TILES), buffer_factor=1)

    @ttl.compute()
    def compute():
        # Compute is identical to single-head -- each core runs one head
        # K^T
        with k_dfb.wait() as kv, kt_dfb.reserve() as kt:
            kt.store(ttl.transpose(kv))
        # QK = Q @ K^T
        with q_dfb.wait() as qv, kt_dfb.wait() as ktv, qk_dfb.reserve() as qk:
            qk.store(qv @ ktv)
        # scaled = scale * QK + mask
        with scale_dfb.wait() as s, scale_row_dfb.reserve() as sr:
            sr.store(ttl.math.broadcast(s, sr, dims=[0]))
        with scale_row_dfb.wait() as sr, scale_bcast_dfb.reserve() as sb:
            sb.store(ttl.math.broadcast(sr, sb, dims=[1]))
        with scale_bcast_dfb.wait() as sb, qk_dfb.wait() as qkv, mask_dfb.wait() as m, scaled_dfb.reserve() as scd:
            scd.store(sb * qkv + m)
        # Softmax
        with scaled_dfb.wait() as sdv, sc_dfb.wait() as sc:
            with max_dfb.reserve() as mx:
                mx.store(ttl.math.reduce_max(sdv, sc, dims=[1]))
            with max_dfb.wait() as mxv, max_bcast_dfb.reserve() as mxb:
                mxb.store(ttl.math.broadcast(mxv, mxb, dims=[1]))
            with max_bcast_dfb.wait() as mxbv:
                with exp_dfb.reserve() as ex:
                    ex.store(ttl.math.exp(sdv - mxbv))
                with exp_dfb.wait() as exv, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(exv, sc, dims=[1]))
                with sum_dfb.wait() as smv, sum_bcast_dfb.reserve() as smb:
                    smb.store(ttl.math.broadcast(smv, smb, dims=[1]))
                with sum_bcast_dfb.wait() as smbv, qk_dfb.reserve() as attn:
                    attn.store(ttl.math.exp(sdv - mxbv) / smbv)
        # out = attn @ V
        with qk_dfb.wait() as av, v_dfb.wait() as vv, out_dfb.reserve() as o:
            o.store(av @ vv)

    @ttl.datamovement()
    def dm_read():
        nx, ny = ttl.node(dims=2)
        h = ny * GRID_X + nx
        q_off = h * sq_tiles
        kv_off = h * skv_tiles
        with q_dfb.reserve() as blk:
            tx = ttl.copy(Q_all[q_off:q_off + sq_tiles, 0:HEAD_TILES], blk); tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(K_all[kv_off:kv_off + skv_tiles, 0:HEAD_TILES], blk); tx.wait()
        with v_dfb.reserve() as blk:
            tx = ttl.copy(V_all[kv_off:kv_off + skv_tiles, 0:HEAD_TILES], blk); tx.wait()
        with scale_dfb.reserve() as blk:
            tx = ttl.copy(scale_tile[0, 0], blk); tx.wait()
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with mask_dfb.reserve() as blk:
            tx = ttl.copy(mask[0:sq_tiles, 0:skv_tiles], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        nx, ny = ttl.node(dims=2)
        h = ny * GRID_X + nx
        q_off = h * sq_tiles
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[q_off:q_off + sq_tiles, 0:HEAD_TILES]); tx.wait()


def torch_sdpa(q, k, v, mask=None):
    """PyTorch reference SDPA."""
    q4 = q.float().unsqueeze(0).unsqueeze(0)
    k4 = k.float().unsqueeze(0).unsqueeze(0)
    v4 = v.float().unsqueeze(0).unsqueeze(0)
    if mask is not None:
        mask4 = mask.float().unsqueeze(0).unsqueeze(0)
        out = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask4)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4)
    return out.squeeze(0).squeeze(0)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test 1: 16 heads, Q=32 tokens, KV=32 tokens (square attention)
    sq, skv = 32, 32
    print(f"=== Test 1: Multi-head SDPA ({NUM_HEADS} heads, Q={sq}, KV={skv}) ===")

    # Generate per-head Q, K, V and compute references
    qs = [torch.randn(sq, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
    ks = [torch.randn(skv, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
    vs = [torch.randn(skv, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
    expecteds = [torch_sdpa(q, k, v).to(torch.bfloat16) for q, k, v in zip(qs, ks, vs)]

    # Stack into batched tensors (heads along dim 0)
    Q_all = torch.cat(qs, dim=0)       # (512, 128)
    K_all = torch.cat(ks, dim=0)       # (512, 128)
    V_all = torch.cat(vs, dim=0)       # (512, 128)
    out_all = torch.zeros(NUM_HEADS * sq, HEAD_DIM, dtype=torch.bfloat16)

    scale_val = 1.0 / math.sqrt(HEAD_DIM)
    scale_tt = to_ttnn(torch.full((TILE, TILE), scale_val, dtype=torch.bfloat16), device)
    scaler_tt = to_ttnn(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    # No mask (zeros = no masking effect after addition)
    mask_tt = to_ttnn(torch.zeros(sq, skv, dtype=torch.bfloat16), device)

    out_tt = to_ttnn(out_all, device)
    sdpa_multihead(to_ttnn(Q_all, device), to_ttnn(K_all, device), to_ttnn(V_all, device),
                   scale_tt, scaler_tt, mask_tt, out_tt)
    result_all = ttnn.to_torch(out_tt)  # (512, 128)

    # Check each head
    all_pass = True
    for h in range(NUM_HEADS):
        result_h = result_all[h * sq:(h + 1) * sq]
        expected_h = expecteds[h]
        err = (expected_h.float() - result_h.float()).abs().max().item()
        pcc = torch.corrcoef(torch.stack([result_h.float().flatten(), expected_h.float().flatten()]))[0, 1].item()
        status = "PASS" if pcc > 0.99 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Head {h:2d}: max_err={err:.4f}, PCC={pcc:.6f} {status}")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")

    # Test 2: Decode-like - Q=32, KV=64 (non-square, simulates cache)
    sq2, skv2 = 32, 64
    print(f"\n=== Test 2: Multi-head SDPA decode ({NUM_HEADS} heads, Q={sq2}, KV={skv2}) ===")

    qs2 = [torch.randn(sq2, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
    ks2 = [torch.randn(skv2, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
    vs2 = [torch.randn(skv2, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
    # Build mask: -inf for positions beyond cache_len
    mask_cpu2 = torch.zeros(sq2, skv2, dtype=torch.bfloat16)
    # Simulate: first 48 tokens valid, rest masked
    cache_len2 = 48
    mask_cpu2[:, cache_len2:] = -float('inf')
    expecteds2 = [torch_sdpa(q, k, v, mask=mask_cpu2).to(torch.bfloat16)
                  for q, k, v in zip(qs2, ks2, vs2)]

    Q_all2 = torch.cat(qs2, dim=0)
    K_all2 = torch.cat(ks2, dim=0)
    V_all2 = torch.cat(vs2, dim=0)
    out_tt2 = to_ttnn(torch.zeros(NUM_HEADS * sq2, HEAD_DIM, dtype=torch.bfloat16), device)
    mask_tt2 = to_ttnn(mask_cpu2, device)

    sdpa_multihead(to_ttnn(Q_all2, device), to_ttnn(K_all2, device), to_ttnn(V_all2, device),
                   scale_tt, scaler_tt, mask_tt2, out_tt2)
    result_all2 = ttnn.to_torch(out_tt2)

    all_pass2 = True
    for h in range(NUM_HEADS):
        result_h = result_all2[h * sq2:(h + 1) * sq2]
        expected_h = expecteds2[h]
        err = (expected_h.float() - result_h.float()).abs().max().item()
        pcc = torch.corrcoef(torch.stack([result_h.float().flatten(), expected_h.float().flatten()]))[0, 1].item()
        status = "PASS" if pcc > 0.99 else "FAIL"
        if status == "FAIL":
            all_pass2 = False
        print(f"  Head {h:2d}: max_err={err:.4f}, PCC={pcc:.6f} {status}")
    print(f"  Overall: {'PASS' if all_pass2 else 'FAIL'}")

    ttnn.close_device(device)
