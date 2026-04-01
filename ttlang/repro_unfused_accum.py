# Reproducer: unfused add accumulation vs fused, with multi-tile output.
#
# Compares two patterns for streaming K accumulation:
#   FUSED:   acc.store(prev + a @ b)      -- single store, stays in f32 DST
#   UNFUSED: mm.store(a @ b) then store(old + mv) -- separate blocks
#
# With BLOCK_N=8 output columns, the unfused version shows much larger
# error, suggesting the intermediate bf16 truncation between the matmul
# and the add loses the f32 precision benefit.

import torch
import ttnn
import ttl

TILE = 32
BLOCK_N = 8


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# FUSED: acc.store(prev + a @ b)
@ttl.operation(grid=(1, 1))
def matmul_fused(a, w, out):
    Kt = a.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(w, shape=(1, BLOCK_N), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BLOCK_N), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BLOCK_N))

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, w_dfb.wait() as wv:
            with acc_dfb.reserve() as acc:
                acc.store(av @ wv)
        for k in range(1, Kt):
            with a_dfb.wait() as av, w_dfb.wait() as wv, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + av @ wv)
        with acc_dfb.wait() as final:
            with out_dfb.reserve() as o:
                o.store(final)

    @ttl.datamovement()
    def dm_read():
        for k in range(Kt):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0, k], blk); tx.wait()
            with w_dfb.reserve() as blk:
                tx = ttl.copy(w[k, 0:BLOCK_N], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0:BLOCK_N]); tx.wait()


# UNFUSED: matmul to temp DFB, then add to accumulator separately
@ttl.operation(grid=(1, 1))
def matmul_unfused(a, w, out):
    Kt = a.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(w, shape=(1, BLOCK_N), buffer_factor=2)
    mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BLOCK_N), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BLOCK_N), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BLOCK_N))

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, w_dfb.wait() as wv, mm_dfb.reserve() as mm:
            mm.store(av @ wv)
        with mm_dfb.wait() as mv, acc_dfb.reserve() as acc:
            acc.store(mv)
        for k in range(1, Kt):
            with a_dfb.wait() as av, w_dfb.wait() as wv, mm_dfb.reserve() as mm:
                mm.store(av @ wv)
            with mm_dfb.wait() as mv, acc_dfb.wait() as old, acc_dfb.reserve() as new:
                new.store(old + mv)
        with acc_dfb.wait() as final:
            with out_dfb.reserve() as o:
                o.store(final)

    @ttl.datamovement()
    def dm_read():
        for k in range(Kt):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0, k], blk); tx.wait()
            with w_dfb.reserve() as blk:
                tx = ttl.copy(w[k, 0:BLOCK_N], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0:BLOCK_N]); tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    K = 2048  # K=64 tiles
    N = BLOCK_N * TILE  # 256 (single column group, no outer N loop)

    a = torch.randn(32, K, dtype=torch.bfloat16)
    w = torch.randn(K, N, dtype=torch.bfloat16)
    expected = (a.float() @ w.float()).to(torch.bfloat16)

    out_f = to_ttnn(torch.zeros(32, N, dtype=torch.bfloat16), device)
    matmul_fused(to_ttnn(a, device), to_ttnn(w, device), out_f)
    rf = ttnn.to_torch(out_f)
    err_f = (expected.float() - rf.float()).abs().max().item()
    mean_f = (expected.float() - rf.float()).abs().mean().item()

    out_u = to_ttnn(torch.zeros(32, N, dtype=torch.bfloat16), device)
    matmul_unfused(to_ttnn(a, device), to_ttnn(w, device), out_u)
    ru = ttnn.to_torch(out_u)
    err_u = (expected.float() - ru.float()).abs().max().item()
    mean_u = (expected.float() - ru.float()).abs().mean().item()

    print(f"(32,{K}) @ ({K},{N}), K={K//TILE} tiles, BLOCK_N={BLOCK_N}")
    print(f"  Fused   (prev + a @ b):  max={err_f:.2f}, mean={mean_f:.4f}")
    print(f"  Unfused (separate add):  max={err_u:.2f}, mean={mean_u:.4f}")
    print(f"  Ratio: {err_u / max(err_f, 0.001):.1f}x worse")

    ttnn.close_device(device)
