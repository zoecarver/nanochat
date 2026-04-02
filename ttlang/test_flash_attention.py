# Single-kernel online Flash Attention.
# One work unit per head, streams KV chunks with online softmax.
# No host reductions, no multi-kernel pipeline.

import math
import torch
import ttnn
import ttl

TILE = 32
HEAD_DIM = 128
HEAD_TILES = HEAD_DIM // TILE  # 4
NUM_HEADS = 16
KV_CHUNK = 1


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


GRID_X = 8
GRID_Y = 2


@ttl.operation(grid=(GRID_X, GRID_Y))
def flash_attention(Q_all, K_all, V_all, scale_tile, scaler, neg_inf_tile,
                    zero_tile, zero_head, mask, out):
    n_heads = Q_all.shape[0] // TILE
    skv = K_all.shape[0] // TILE // n_heads
    n_chunks = skv // KV_CHUNK

    # Inputs loaded once per head
    q_dfb = ttl.make_dataflow_buffer_like(Q_all, shape=(1, HEAD_TILES), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    ninf_dfb = ttl.make_dataflow_buffer_like(neg_inf_tile, shape=(1, 1), buffer_factor=2)
    zero_dfb = ttl.make_dataflow_buffer_like(zero_tile, shape=(1, 1), buffer_factor=2)
    zero_head_dfb = ttl.make_dataflow_buffer_like(zero_head, shape=(1, HEAD_TILES), buffer_factor=2)

    # Per-chunk inputs (streamed)
    k_dfb = ttl.make_dataflow_buffer_like(K_all, shape=(KV_CHUNK, HEAD_TILES), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(V_all, shape=(KV_CHUNK, HEAD_TILES), buffer_factor=2)
    mask_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, KV_CHUNK), buffer_factor=2)

    # Compute intermediates
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

    # Running state (ping-pong with buffer_factor=2)
    m_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    l_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    o_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)

    # Final output
    l_bc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HEAD_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        nx, ny = ttl.node(dims=2)
        h = ny * GRID_X + nx
        q_blk = q_dfb.wait()
        sc_blk = sc_dfb.wait()
        scaler_blk = scaler_dfb.wait()

        # Initialize: m = -inf, l = 0, O = 0
        with ninf_dfb.wait() as ni, m_dfb.reserve() as m_init:
            m_init.store(ni)
        with zero_dfb.wait() as z, l_dfb.reserve() as l_init:
            l_init.store(z)
        with zero_head_dfb.wait() as zh, o_dfb.reserve() as o_init:
            o_init.store(zh)

        for c in range(n_chunks):
            # K^T
            with k_dfb.wait() as kc, kt_dfb.reserve() as kt:
                kt.store(ttl.transpose(kc))
            # QK = Q @ K_c^T
            with kt_dfb.wait() as ktv, qk_dfb.reserve() as qk:
                qk.store(q_blk @ ktv)
            # scaled = scale * QK + mask
            with qk_dfb.wait() as qkv, mask_dfb.wait() as mv:
                with scaled_dfb.reserve() as scd:
                    scd.store(sc_blk * qkv + mv)

            with scaled_dfb.wait() as sd:
                # chunk_max = rowmax(scaled)
                with cm_dfb.reserve() as cm:
                    cm.store(ttl.math.reduce_max(sd, scaler_blk, dims=[1]))
                # m_new = max(m_old, chunk_max), alpha = exp(m_old - m_new)
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
                # exp_scores = exp(scaled - broadcast(m_new))
                with m_new_bc_dfb.wait() as mnb:
                    with exp_dfb.reserve() as ex:
                        ex.store(ttl.math.exp(sd - mnb))

            with exp_dfb.wait() as exp_blk:
                # chunk_sum = rowsum(exp_scores)
                with cs_dfb.reserve() as cs:
                    cs.store(ttl.math.reduce_sum(exp_blk, scaler_blk, dims=[1]))
                with alpha_dfb.wait() as alpha_blk:
                    # l_new = alpha * l_old + chunk_sum
                    with l_dfb.wait() as l_old, cs_dfb.wait() as cs:
                        with l_dfb.reserve() as l_new:
                            l_new.store(alpha_blk * l_old + cs)
                    # O_new = alpha_bc * O_old + exp_scores @ V
                    with alpha_bc_dfb.reserve() as abc:
                        abc.store(ttl.math.broadcast(alpha_blk, abc, dims=[1]))
                with alpha_bc_dfb.wait() as abc, o_dfb.wait() as o_old:
                    with corrected_o_dfb.reserve() as co:
                        co.store(abc * o_old)
                with corrected_o_dfb.wait() as co, v_dfb.wait() as vc:
                    with o_dfb.reserve() as o_new:
                        o_new.store(co + exp_blk @ vc)

        # Final: O = O / broadcast(l)
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
        h = ny * GRID_X + nx
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
            kv_off = h * skv + c * KV_CHUNK
            with k_dfb.reserve() as blk:
                tx = ttl.copy(K_all[kv_off:kv_off + KV_CHUNK, 0:HEAD_TILES], blk); tx.wait()
            with v_dfb.reserve() as blk:
                tx = ttl.copy(V_all[kv_off:kv_off + KV_CHUNK, 0:HEAD_TILES], blk); tx.wait()
            with mask_dfb.reserve() as blk:
                tx = ttl.copy(mask[0, c * KV_CHUNK:(c + 1) * KV_CHUNK], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        nx, ny = ttl.node(dims=2)
        h = ny * GRID_X + nx
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[h:h + 1, 0:HEAD_TILES]); tx.wait()


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


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    scale_val = 1.0 / math.sqrt(HEAD_DIM)
    scale_tt = to_ttnn(torch.full((TILE, TILE), scale_val, dtype=torch.bfloat16), device)
    scaler_tt = to_ttnn(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    ninf_tt = to_ttnn(torch.full((TILE, TILE), -10000.0, dtype=torch.bfloat16), device)
    zero_tt = to_ttnn(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
    zero_head_tt = to_ttnn(torch.zeros(TILE, HEAD_DIM, dtype=torch.bfloat16), device)

    def run_test(name, skv_tokens):
        n_chunks = skv_tokens // TILE
        print(f"\n=== {name}: {NUM_HEADS} heads, KV={skv_tokens}, {n_chunks} chunks ===")

        qs = [torch.randn(TILE, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
        ks = [torch.randn(skv_tokens, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
        vs = [torch.randn(skv_tokens, HEAD_DIM, dtype=torch.bfloat16) for _ in range(NUM_HEADS)]
        refs = [torch_sdpa(q, k, v).to(torch.bfloat16) for q, k, v in zip(qs, ks, vs)]

        Q_all = torch.cat(qs, dim=0)
        K_all = torch.cat(ks, dim=0)
        V_all = torch.cat(vs, dim=0)
        mask_cpu = torch.zeros(TILE, skv_tokens, dtype=torch.bfloat16)

        out_tt = to_ttnn(torch.zeros(NUM_HEADS * TILE, HEAD_DIM, dtype=torch.bfloat16), device)
        flash_attention(
            to_ttnn(Q_all, device), to_ttnn(K_all, device), to_ttnn(V_all, device),
            scale_tt, scaler_tt, ninf_tt, zero_tt, zero_head_tt,
            to_ttnn(mask_cpu, device), out_tt)

        result_all = ttnn.to_torch(out_tt).view(NUM_HEADS, TILE, HEAD_DIM)
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
