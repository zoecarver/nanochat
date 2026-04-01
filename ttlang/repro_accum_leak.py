# Bug: fused matmul accumulation (prev + a @ b) produces wrong results
# when the outer N loop runs more than once with multi-tile output blocks.
#
# BLOCK_N=1: works for any num_n
# BLOCK_N=8: num_n=1 correct, num_n>=2 wrong (error scales with num_n)
#
# The accumulator appears to retain state from the previous N iteration.
#
# Workaround: distribute column groups across cores (grid="auto") so
# each core only runs the inner K loop once per column group.

import torch
import ttnn
import ttl

TILE = 32
BLOCK_N = 8


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@ttl.operation(grid=(1, 1))
def matmul_multi_n(a, w, out):
    Kt = a.shape[1] // TILE
    Nt = w.shape[1] // TILE
    num_n = Nt // BLOCK_N

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(w, shape=(1, BLOCK_N), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BLOCK_N), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, BLOCK_N))

    @ttl.compute()
    def compute():
        for ni in range(num_n):
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
        for ni in range(num_n):
            n_off = ni * BLOCK_N
            for k in range(Kt):
                with a_dfb.reserve() as blk:
                    tx = ttl.copy(a[0, k], blk); tx.wait()
                with w_dfb.reserve() as blk:
                    tx = ttl.copy(w[k, n_off:n_off + BLOCK_N], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        for ni in range(num_n):
            n_off = ni * BLOCK_N
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, n_off:n_off + BLOCK_N]); tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    K = 512  # K=16 tiles (smaller K still shows the bug)

    # num_n=1: correct (baseline)
    N1 = BLOCK_N * TILE  # 256
    a = torch.randn(32, K, dtype=torch.bfloat16)
    w1 = torch.randn(K, N1, dtype=torch.bfloat16)
    expected1 = (a.float() @ w1.float()).to(torch.bfloat16)
    out1 = to_ttnn(torch.zeros(32, N1, dtype=torch.bfloat16), device)
    matmul_multi_n(to_ttnn(a, device), to_ttnn(w1, device), out1)
    r1 = ttnn.to_torch(out1)
    err1 = (expected1.float() - r1.float()).abs().max().item()

    # num_n=2: broken
    N2 = 2 * BLOCK_N * TILE  # 512
    w2 = torch.randn(K, N2, dtype=torch.bfloat16)
    expected2 = (a.float() @ w2.float()).to(torch.bfloat16)
    out2 = to_ttnn(torch.zeros(32, N2, dtype=torch.bfloat16), device)
    matmul_multi_n(to_ttnn(a, device), to_ttnn(w2, device), out2)
    r2 = ttnn.to_torch(out2)
    err2 = (expected2.float() - r2.float()).abs().max().item()

    # num_n=4: worse
    N4 = 4 * BLOCK_N * TILE  # 1024
    w4 = torch.randn(K, N4, dtype=torch.bfloat16)
    expected4 = (a.float() @ w4.float()).to(torch.bfloat16)
    out4 = to_ttnn(torch.zeros(32, N4, dtype=torch.bfloat16), device)
    matmul_multi_n(to_ttnn(a, device), to_ttnn(w4, device), out4)
    r4 = ttnn.to_torch(out4)
    err4 = (expected4.float() - r4.float()).abs().max().item()

    print(f"Fused matmul (prev + a @ b), K={K}, BLOCK_N={BLOCK_N}")
    print(f"  num_n=1 (N={N1}): max_err={err1:.1f}  {'PASS' if err1 < 5 else 'FAIL'}")
    print(f"  num_n=2 (N={N2}): max_err={err2:.1f}  {'PASS' if err2 < 5 else 'FAIL'}")
    print(f"  num_n=4 (N={N4}): max_err={err4:.1f}  {'PASS' if err4 < 5 else 'FAIL'}")
    print()
    print(f"Note: BLOCK_N=1 works for any num_n. Bug is multi-tile output + outer N loop.")

    ttnn.close_device(device)
