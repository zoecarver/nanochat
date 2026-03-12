# Linear projection kernel: out = x @ w
#
# Handles arbitrary (seq, K) @ (K, N) -> (seq, N).
# grid="auto" parallelizes over output columns.
# Uses multi-tile DFBs for K accumulation in DST (f32 precision).
# Streams weight columns to handle large N.
#
# For nanochat d32:
#   Q/K/V/proj: (seq, 2048) @ (2048, 2048) -> K_TILES=64
#   MLP fc:     (seq, 2048) @ (2048, 8192) -> K_TILES=64
#   MLP proj:   (seq, 8192) @ (8192, 2048) -> K_TILES=256 (chunked)
#   LM head:    (seq, 2048) @ (2048, 32768) -> K_TILES=64

import torch
import ttnn
import ttl

TILE = 32
K_CHUNK = 64  # Max tiles in K dimension per DFB (128KB per chunk)


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@ttl.kernel(grid="auto")
def linear_kernel(x, w, out):
    """General linear projection: out = x @ w.

    x: (M, K) where M = seq_tiles (padded to tile), K = input dim tiles
    w: (K, N) where N = output dim tiles
    out: (M, N)

    Grid parallelizes over N (output columns).
    For each output column, loads full x row and weight column,
    multi-tile DFB handles K accumulation in DST.
    x is re-read per output column (streaming from DRAM).
    """
    grid_cols, _ = ttl.grid_size(dims=2)
    k_tiles = x.shape[1] // TILE
    n_tiles = w.shape[1] // TILE
    cols_per_core = -(-n_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, k_tiles), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(w, shape=(k_tiles, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_j in range(cols_per_core):
            j = core_x * cols_per_core + local_j
            if j < n_tiles:
                with x_dfb.wait() as xv, w_dfb.wait() as wv, out_dfb.reserve() as o:
                    o.store(xv @ wv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_j in range(cols_per_core):
            j = core_x * cols_per_core + local_j
            if j < n_tiles:
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[0, 0:k_tiles], blk); tx.wait()
                with w_dfb.reserve() as blk:
                    tx = ttl.copy(w[0:k_tiles, j], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_j in range(cols_per_core):
            j = core_x * cols_per_core + local_j
            if j < n_tiles:
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[0, j]); tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test 1: Small square - (32, 64) @ (64, 32) = K=2 tiles
    print("=== Test 1: (32,64) @ (64,32) ===")
    a1 = torch.randn(32, 64, dtype=torch.bfloat16)
    w1 = torch.randn(64, 32, dtype=torch.bfloat16)
    expected1 = (a1.float() @ w1.float()).to(torch.bfloat16)
    out1 = to_ttnn(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    linear_kernel(to_ttnn(a1, device), to_ttnn(w1, device), out1)
    result1 = ttnn.to_torch(out1)
    err1 = (expected1.float() - result1.float()).abs().max().item()
    print(f"  Max error: {err1:.4f} {'PASS' if err1 < 2.0 else 'FAIL'}")

    # Test 2: (32, 2048) @ (2048, 2048) - Q/K/V projection size
    print("\n=== Test 2: (32,2048) @ (2048,2048) K=64 ===")
    a2 = torch.randn(32, 2048, dtype=torch.bfloat16)
    w2 = torch.randn(2048, 2048, dtype=torch.bfloat16)
    expected2 = (a2.float() @ w2.float()).to(torch.bfloat16)
    out2 = to_ttnn(torch.zeros(32, 2048, dtype=torch.bfloat16), device)
    linear_kernel(to_ttnn(a2, device), to_ttnn(w2, device), out2)
    result2 = ttnn.to_torch(out2)
    err2 = (expected2.float() - result2.float()).abs().max().item()
    mean2 = (expected2.float() - result2.float()).abs().mean().item()
    print(f"  Expected[0,:4]: {expected2[0,:4]}")
    print(f"  Got[0,:4]:      {result2[0,:4]}")
    print(f"  Max error: {err2:.4f}, Mean: {mean2:.4f}")
    print(f"  {'PASS' if err2 < 5.0 else 'FAIL'}")

    # Test 3: (32, 2048) @ (2048, 8192) - MLP fc size
    print("\n=== Test 3: (32,2048) @ (2048,8192) MLP fc ===")
    a3 = torch.randn(32, 2048, dtype=torch.bfloat16)
    w3 = torch.randn(2048, 8192, dtype=torch.bfloat16)
    expected3 = (a3.float() @ w3.float()).to(torch.bfloat16)
    out3 = to_ttnn(torch.zeros(32, 8192, dtype=torch.bfloat16), device)
    linear_kernel(to_ttnn(a3, device), to_ttnn(w3, device), out3)
    result3 = ttnn.to_torch(out3)
    err3 = (expected3.float() - result3.float()).abs().max().item()
    mean3 = (expected3.float() - result3.float()).abs().mean().item()
    print(f"  Expected[0,:4]: {expected3[0,:4]}")
    print(f"  Got[0,:4]:      {result3[0,:4]}")
    print(f"  Max error: {err3:.4f}, Mean: {mean3:.4f}")
    print(f"  {'PASS' if err3 < 5.0 else 'FAIL'}")

    ttnn.close_device(device)
