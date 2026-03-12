# Test matmul shapes to determine what works on HW.
# We need: (seq, 2048) @ (2048, 8192) for MLP.
# Strategy: stream weight columns in chunks, accumulate partial results.
# But first let's test basic matmul shapes to see what compiles.

import torch
import ttnn
import ttl

TILE = 32


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# Test 1: Square matmul (1,1) @ (1,1) = (1,1) - should always work
@ttl.kernel(grid=(1, 1))
def matmul_1x1(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
            o.store(av @ bv)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk); tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0]); tx.wait()


# Test 2: Non-square (1,2) @ (2,1) = (1,1) - accumulating K dim
@ttl.kernel(grid=(1, 1))
def matmul_accum(a, b, out):
    """Accumulate matmul over K dimension using multi-tile DFB for K accumulation in DST."""
    k_tiles = a.shape[1] // TILE

    # Use multi-tile DFBs: a is (1, K), b is (K, 1), matmul produces (1, 1) with K accumulated in DST
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, k_tiles), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(k_tiles, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
            o.store(av @ bv)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0:k_tiles], blk); tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:k_tiles, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0]); tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test 1: 32x32 @ 32x32
    print("=== Test 1: (32,32) @ (32,32) ===")
    a1 = torch.randn(32, 32, dtype=torch.bfloat16)
    b1 = torch.randn(32, 32, dtype=torch.bfloat16)
    expected1 = (a1.float() @ b1.float()).to(torch.bfloat16)
    out1 = to_ttnn(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    matmul_1x1(to_ttnn(a1, device), to_ttnn(b1, device), out1)
    result1 = ttnn.to_torch(out1)
    err1 = (expected1.float() - result1.float()).abs().max().item()
    print(f"  Expected[0,:4]: {expected1[0,:4]}")
    print(f"  Got[0,:4]:      {result1[0,:4]}")
    print(f"  Max error: {err1:.4f}")
    print(f"  {'PASS' if err1 < 2.0 else 'FAIL'}")

    # Test 2: (32, 64) @ (64, 32) via accumulating K=2
    print("\n=== Test 2: (32,64) @ (64,32) accum K=2 ===")
    a2 = torch.randn(32, 64, dtype=torch.bfloat16)
    b2 = torch.randn(64, 32, dtype=torch.bfloat16)
    expected2 = (a2.float() @ b2.float()).to(torch.bfloat16)
    out2 = to_ttnn(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    matmul_accum(to_ttnn(a2, device), to_ttnn(b2, device), out2)
    result2 = ttnn.to_torch(out2)
    err2 = (expected2.float() - result2.float()).abs().max().item()
    print(f"  Expected[0,:4]: {expected2[0,:4]}")
    print(f"  Got[0,:4]:      {result2[0,:4]}")
    print(f"  Max error: {err2:.4f}")
    print(f"  {'PASS' if err2 < 2.0 else 'FAIL'}")

    # Test 3: Larger K - (32, 2048) @ (2048, 32) via accum K=64
    print("\n=== Test 3: (32,2048) @ (2048,32) accum K=64 ===")
    a3 = torch.randn(32, 2048, dtype=torch.bfloat16)
    b3 = torch.randn(2048, 32, dtype=torch.bfloat16)
    expected3 = (a3.float() @ b3.float()).to(torch.bfloat16)
    out3 = to_ttnn(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    matmul_accum(to_ttnn(a3, device), to_ttnn(b3, device), out3)
    result3 = ttnn.to_torch(out3)
    err3 = (expected3.float() - result3.float()).abs().max().item()
    print(f"  Expected[0,:4]: {expected3[0,:4]}")
    print(f"  Got[0,:4]:      {result3[0,:4]}")
    print(f"  Max error: {err3:.4f}")
    print(f"  {'PASS' if err3 < 5.0 else 'FAIL'}")

    ttnn.close_device(device)
