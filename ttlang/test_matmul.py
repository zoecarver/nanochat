# Test matmul with fused K accumulation (Strategy 1 from tt-lang reference).
# Pattern: acc.store(prev + a @ b) fuses the accumulation with the matmul,
# keeping the K accumulation in f32 DST.

import torch
import ttnn
import ttl

TILE = 32
NCOLS = 8


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_matmul(block_n):
    """Fused K accumulation matmul: out = a @ w.

    Streams K=1 tile at a time. First step seeds accumulator with a@w,
    remaining steps fuse: acc.store(prev + a @ b).
    """
    @ttl.operation(grid="auto")
    def matmul(a, w, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        Mt = a.shape[0] // TILE
        Kt = a.shape[1] // TILE
        Nt = w.shape[1] // TILE
        num_n = Nt // block_n
        total = Mt * num_n
        units_per_core = -(-total // grid_cols)

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(1, block_n), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, block_n), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, block_n))

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    # First K step: seed accumulator
                    with a_dfb.wait() as av, w_dfb.wait() as wv:
                        with acc_dfb.reserve() as acc:
                            acc.store(av @ wv)
                    # Remaining K steps: fused accumulate
                    for k in range(1, Kt):
                        with a_dfb.wait() as av, w_dfb.wait() as wv, acc_dfb.wait() as prev:
                            with acc_dfb.reserve() as acc:
                                acc.store(prev + av @ wv)
                    # Write result
                    with acc_dfb.wait() as final:
                        with out_dfb.reserve() as o:
                            o.store(final)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // num_n
                    ni = t % num_n
                    n_off = ni * block_n
                    for k in range(Kt):
                        with a_dfb.reserve() as blk:
                            tx = ttl.copy(a[mi, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(w[k, n_off:n_off + block_n], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // num_n
                    ni = t % num_n
                    n_off = ni * block_n
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[mi, n_off:n_off + block_n]); tx.wait()

    return matmul


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test 1: K=2 (64 elem)
    print("=== Test 1: (32,64) @ (64,256) K=2 ===")
    a1 = torch.randn(32, 64, dtype=torch.bfloat16)
    w1 = torch.randn(64, 256, dtype=torch.bfloat16)
    expected1 = (a1.float() @ w1.float()).to(torch.bfloat16)
    out1 = to_ttnn(torch.zeros(32, 256, dtype=torch.bfloat16), device)
    make_matmul(NCOLS)(to_ttnn(a1, device), to_ttnn(w1, device), out1)
    result1 = ttnn.to_torch(out1)
    err1 = (expected1.float() - result1.float()).abs().max().item()
    print(f"  Max error: {err1:.4f} {'PASS' if err1 < 2.0 else 'FAIL'}")

    # Test 2: K=64 (2048 elem) - Q/K/V projection size
    print("\n=== Test 2: (32,2048) @ (2048,2048) K=64 ===")
    a2 = torch.randn(32, 2048, dtype=torch.bfloat16)
    w2 = torch.randn(2048, 2048, dtype=torch.bfloat16)
    expected2 = (a2.float() @ w2.float()).to(torch.bfloat16)
    out2 = to_ttnn(torch.zeros(32, 2048, dtype=torch.bfloat16), device)
    make_matmul(NCOLS)(to_ttnn(a2, device), to_ttnn(w2, device), out2)
    result2 = ttnn.to_torch(out2)
    err2 = (expected2.float() - result2.float()).abs().max().item()
    mean2 = (expected2.float() - result2.float()).abs().mean().item()
    print(f"  Expected[0,:4]: {expected2[0,:4]}")
    print(f"  Got[0,:4]:      {result2[0,:4]}")
    print(f"  Max error: {err2:.4f}, Mean: {mean2:.4f}")
    print(f"  {'PASS' if err2 < 5.0 else 'FAIL'}")

    # Test 3: K=64, N=8192 - MLP fc size
    print("\n=== Test 3: (32,2048) @ (2048,8192) K=64 MLP fc ===")
    a3 = torch.randn(32, 2048, dtype=torch.bfloat16)
    w3 = torch.randn(2048, 8192, dtype=torch.bfloat16)
    expected3 = (a3.float() @ w3.float()).to(torch.bfloat16)
    out3 = to_ttnn(torch.zeros(32, 8192, dtype=torch.bfloat16), device)
    make_matmul(NCOLS)(to_ttnn(a3, device), to_ttnn(w3, device), out3)
    result3 = ttnn.to_torch(out3)
    err3 = (expected3.float() - result3.float()).abs().max().item()
    mean3 = (expected3.float() - result3.float()).abs().mean().item()
    print(f"  Max error: {err3:.4f}, Mean: {mean3:.4f}")
    print(f"  {'PASS' if err3 < 5.0 else 'FAIL'}")

    ttnn.close_device(device)
