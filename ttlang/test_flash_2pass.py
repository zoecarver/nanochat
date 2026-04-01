# Two-pass Flash Attention with grid="auto" and full core utilization.
#
# Pass 1: Compute per-chunk row maxes (grid="auto")
#   Work units = NUM_HEADS * n_kv_chunks
#   Each unit: QK = Q_h @ K_c^T, chunk_max = rowmax(scale * QK + mask_c)
#   Host reduces chunk maxes -> global max per head
#
# Pass 2: Compute partial outputs using global max (grid="auto")
#   Each unit: exp_scores = exp(scale * QK + mask - global_max_h)
#              partial_o = exp_scores @ V_c, partial_l = rowsum(exp_scores)
#   Host reduces: O = sum(partial_o), L = sum(partial_l), result = O / L

import math
import torch
import ttnn
import ttl

TILE = 32
HEAD_DIM = 128
HEAD_TILES = HEAD_DIM // TILE  # 4
NUM_HEADS = 16
KV_CHUNK = 1  # tiles per KV chunk


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# ---------------------------------------------------------------------------
# Pass 1: chunk maxes
# ---------------------------------------------------------------------------
@ttl.operation(grid="auto")
def flash_pass1(Q_all, K_all, scale_tile, scaler, mask, chunk_maxes):
    grid_cols, _ = ttl.grid_size(dims=2)
    n_heads = Q_all.shape[0] // TILE
    skv = K_all.shape[0] // TILE // n_heads
    n_work = n_heads * skv
    upc = -(-n_work // grid_cols)

    q_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, HEAD_TILES), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(KV_CHUNK, HEAD_TILES), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    mask_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, KV_CHUNK), buffer_factor=2)
    kt_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(HEAD_TILES, KV_CHUNK), buffer_factor=2)
    qk_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, KV_CHUNK), buffer_factor=2)
    scaled_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, KV_CHUNK), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(chunk_maxes, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        cx, _ = ttl.node(dims=2)
        for lt in range(upc):
            t = cx * upc + lt
            if t < n_work:
                with k_dfb.wait() as kc, kt_dfb.reserve() as kt:
                    kt.store(ttl.transpose(kc))
                with q_dfb.wait() as qv, kt_dfb.wait() as ktv, qk_dfb.reserve() as qk:
                    qk.store(qv @ ktv)
                with sc_dfb.wait() as s, qk_dfb.wait() as qkv, mask_dfb.wait() as m:
                    with scaled_dfb.reserve() as scd:
                        scd.store(s * qkv + m)
                with scaled_dfb.wait() as sdv, scaler_dfb.wait() as sc:
                    with out_dfb.reserve() as o:
                        o.store(ttl.math.reduce_max(sdv, sc, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        cx, _ = ttl.node(dims=2)
        for lt in range(upc):
            t = cx * upc + lt
            if t < n_work:
                h = t // skv
                c = t % skv
                kv_off = h * skv + c
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(Q_all[h:h + 1, 0:HEAD_TILES], blk); tx.wait()
                with k_dfb.reserve() as blk:
                    tx = ttl.copy(K_all[kv_off:kv_off + KV_CHUNK, 0:HEAD_TILES], blk); tx.wait()
                with sc_dfb.reserve() as blk:
                    tx = ttl.copy(scale_tile[0, 0], blk); tx.wait()
                with scaler_dfb.reserve() as blk:
                    tx = ttl.copy(scaler[0, 0], blk); tx.wait()
                with mask_dfb.reserve() as blk:
                    tx = ttl.copy(mask[0, c:c + KV_CHUNK], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        cx, _ = ttl.node(dims=2)
        for lt in range(upc):
            t = cx * upc + lt
            if t < n_work:
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, chunk_maxes[t, 0]); tx.wait()


# ---------------------------------------------------------------------------
# Pass 2: partial outputs
# ---------------------------------------------------------------------------
@ttl.operation(grid="auto")
def flash_pass2(Q_all, K_all, V_all, global_max, scale_tile, scaler, mask, partial_o, partial_l):
    grid_cols, _ = ttl.grid_size(dims=2)
    n_heads = Q_all.shape[0] // TILE
    skv = K_all.shape[0] // TILE // n_heads
    n_work = n_heads * skv
    upc = -(-n_work // grid_cols)

    q_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, HEAD_TILES), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(KV_CHUNK, HEAD_TILES), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(V_all, shape=(KV_CHUNK, HEAD_TILES), buffer_factor=2)
    gm_dfb = ttl.make_dataflow_buffer_like(global_max, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    mask_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, KV_CHUNK), buffer_factor=2)

    kt_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(HEAD_TILES, KV_CHUNK), buffer_factor=2)
    qk_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, KV_CHUNK), buffer_factor=2)
    scaled_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, KV_CHUNK), buffer_factor=2)
    gm_bc_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, KV_CHUNK), buffer_factor=2)
    exp_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, KV_CHUNK), buffer_factor=2)
    po_dfb = ttl.make_dataflow_buffer_like(partial_o, shape=(1, HEAD_TILES), buffer_factor=2)
    pl_dfb = ttl.make_dataflow_buffer_like(partial_l, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        cx, _ = ttl.node(dims=2)
        for lt in range(upc):
            t = cx * upc + lt
            if t < n_work:
                # K^T
                with k_dfb.wait() as kc, kt_dfb.reserve() as kt:
                    kt.store(ttl.transpose(kc))
                # QK = Q @ K^T
                with q_dfb.wait() as qv, kt_dfb.wait() as ktv, qk_dfb.reserve() as qk:
                    qk.store(qv @ ktv)
                # scaled = scale * QK + mask
                with sc_dfb.wait() as s, qk_dfb.wait() as qkv, mask_dfb.wait() as m:
                    with scaled_dfb.reserve() as scd:
                        scd.store(s * qkv + m)
                # Broadcast global max within tile (replicates col 0 across columns)
                with gm_dfb.wait() as gm, gm_bc_dfb.reserve() as gmb:
                    gmb.store(ttl.math.broadcast(gm, gmb, dims=[1]))
                # exp(scaled - global_max)
                with scaled_dfb.wait() as sd, gm_bc_dfb.wait() as gmb:
                    with exp_dfb.reserve() as ex:
                        ex.store(ttl.math.exp(sd - gmb))
                # partial_o = exp @ V
                exp_blk = exp_dfb.wait()
                with v_dfb.wait() as vc, po_dfb.reserve() as po:
                    po.store(exp_blk @ vc)
                # partial_l = rowsum(exp)
                with scaler_dfb.wait() as sc, pl_dfb.reserve() as pl:
                    pl.store(ttl.math.reduce_sum(exp_blk, sc, dims=[1]))
                exp_blk.pop()

    @ttl.datamovement()
    def dm_read():
        cx, _ = ttl.node(dims=2)
        for lt in range(upc):
            t = cx * upc + lt
            if t < n_work:
                h = t // skv
                c = t % skv
                kv_off = h * skv + c
                with q_dfb.reserve() as blk:
                    tx = ttl.copy(Q_all[h:h + 1, 0:HEAD_TILES], blk); tx.wait()
                with k_dfb.reserve() as blk:
                    tx = ttl.copy(K_all[kv_off:kv_off + KV_CHUNK, 0:HEAD_TILES], blk); tx.wait()
                with v_dfb.reserve() as blk:
                    tx = ttl.copy(V_all[kv_off:kv_off + KV_CHUNK, 0:HEAD_TILES], blk); tx.wait()
                with gm_dfb.reserve() as blk:
                    tx = ttl.copy(global_max[h:h + 1, 0:1], blk); tx.wait()
                with sc_dfb.reserve() as blk:
                    tx = ttl.copy(scale_tile[0, 0], blk); tx.wait()
                with scaler_dfb.reserve() as blk:
                    tx = ttl.copy(scaler[0, 0], blk); tx.wait()
                with mask_dfb.reserve() as blk:
                    tx = ttl.copy(mask[0, c:c + KV_CHUNK], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        cx, _ = ttl.node(dims=2)
        for lt in range(upc):
            t = cx * upc + lt
            if t < n_work:
                with po_dfb.wait() as blk:
                    tx = ttl.copy(blk, partial_o[t:t + 1, 0:HEAD_TILES]); tx.wait()
                with pl_dfb.wait() as blk:
                    tx = ttl.copy(blk, partial_l[t, 0]); tx.wait()


# ---------------------------------------------------------------------------
# Host reduction helpers
# ---------------------------------------------------------------------------
def host_reduce_max(chunk_maxes_torch, n_heads, n_chunks):
    """Reduce chunk maxes to global max per head."""
    # chunk_maxes: (n_heads * n_chunks * 32, 32) - tiles stacked vertically
    cm = chunk_maxes_torch.view(n_heads, n_chunks, TILE, TILE)
    # Element-wise max across chunks (broadcast in kernel 2 uses col 0)
    return cm.max(dim=1).values.contiguous()  # (n_heads, 32, 32)


def host_reduce_output(partial_o_torch, partial_l_torch, n_heads, n_chunks):
    """Reduce partial outputs to final attention output per head."""
    # partial_o: (n_heads * n_chunks * 32, 128)
    po = partial_o_torch.float().view(n_heads, n_chunks, TILE, HEAD_DIM)
    O = po.sum(dim=1)  # (n_heads, 32, 128)

    # partial_l: (n_heads * n_chunks * 32, 32) - row sums in col 0
    pl = partial_l_torch.float().view(n_heads, n_chunks, TILE, TILE)
    L = pl[:, :, :, 0].sum(dim=1)  # (n_heads, 32)

    result = O / L.unsqueeze(-1)  # (n_heads, 32, 128)
    return result.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def torch_sdpa(q, k, v, mask=None):
    q4 = q.float().unsqueeze(0).unsqueeze(0)
    k4 = k.float().unsqueeze(0).unsqueeze(0)
    v4 = v.float().unsqueeze(0).unsqueeze(0)
    if mask is not None:
        out = torch.nn.functional.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=mask.float().unsqueeze(0).unsqueeze(0))
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4)
    return out.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    scale_val = 1.0 / math.sqrt(HEAD_DIM)
    scale_tt = to_ttnn(torch.full((TILE, TILE), scale_val, dtype=torch.bfloat16), device)
    scaler_tt = to_ttnn(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)

    def run_test(name, skv_tokens):
        n_chunks = skv_tokens // TILE
        n_work = NUM_HEADS * n_chunks
        print(f"\n=== {name}: {NUM_HEADS} heads, KV={skv_tokens}, {n_chunks} chunks, {n_work} work units ===")

        # Generate per-head data
        qs = [torch.randn(TILE, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
        ks = [torch.randn(skv_tokens, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
        vs = [torch.randn(skv_tokens, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]

        # PyTorch reference
        refs = [torch_sdpa(q, k, v).to(torch.bfloat16) for q, k, v in zip(qs, ks, vs)]

        # Batch tensors (heads stacked vertically)
        Q_all = torch.cat(qs, dim=0)                    # (NUM_HEADS*32, 128)
        K_all = torch.cat(ks, dim=0)                    # (NUM_HEADS*skv, 128)
        V_all = torch.cat(vs, dim=0)
        mask_cpu = torch.zeros(TILE, skv_tokens, dtype=torch.bfloat16)

        Q_tt = to_ttnn(Q_all, device)
        K_tt = to_ttnn(K_all, device)
        V_tt = to_ttnn(V_all, device)
        mask_tt = to_ttnn(mask_cpu, device)

        # Pass 1: chunk maxes
        cm_tt = to_ttnn(torch.zeros(n_work * TILE, TILE, dtype=torch.bfloat16), device)
        flash_pass1(Q_tt, K_tt, scale_tt, scaler_tt, mask_tt, cm_tt)
        cm_torch = ttnn.to_torch(cm_tt)

        # Host reduce: global max per head
        gm_torch = host_reduce_max(cm_torch, NUM_HEADS, n_chunks)
        gm_tt = to_ttnn(gm_torch.reshape(NUM_HEADS * TILE, TILE), device)

        # Pass 2: partial outputs
        po_tt = to_ttnn(torch.zeros(n_work * TILE, HEAD_DIM, dtype=torch.bfloat16), device)
        pl_tt = to_ttnn(torch.zeros(n_work * TILE, TILE, dtype=torch.bfloat16), device)
        flash_pass2(Q_tt, K_tt, V_tt, gm_tt, scale_tt, scaler_tt, mask_tt, po_tt, pl_tt)
        po_torch = ttnn.to_torch(po_tt)
        pl_torch = ttnn.to_torch(pl_tt)

        # Host reduce: final output
        result_all = host_reduce_output(po_torch, pl_torch, NUM_HEADS, n_chunks)

        # Compare per head
        all_pass = True
        for h in range(NUM_HEADS):
            r = result_all[h]
            e = refs[h]
            err = (e.float() - r.float()).abs().max().item()
            pcc = torch.corrcoef(torch.stack([r.float().flatten(), e.float().flatten()]))[0, 1].item()
            status = "PASS" if pcc > 0.99 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  Head {h:2d}: max_err={err:.4f}, PCC={pcc:.6f} {status}")
        print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")

    run_test("1 chunk", 32)
    run_test("2 chunks", 64)
    run_test("4 chunks", 128)

    ttnn.close_device(device)
