# RMSNorm kernel for nanochat inference on Tenstorrent hardware.
#
# Implements: norm(x) = x / sqrt(mean(x^2) + eps)
# No learnable weights (matches nanochat's norm() function).
#
# Pattern: streaming (1,1) tile DFBs, grid="auto", two-pass read.
# Pass 1: sum of squares across embd dimension
# Pass 2: normalize each tile by rsqrt(mean)

import torch
import ttnn
import ttl

TILE = 32
N_EMBD = 2048
EMBD_TILES = N_EMBD // TILE  # 64
C_MEAN_SCALE = 1.0 / N_EMBD
C_EPS = 1e-5


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@ttl.kernel(grid="auto")
def rmsnorm_kernel(x, scaler, mean_scale, out):
    """RMSNorm: out = x * rsqrt(mean(x^2) + eps)

    x: (seq_tiles, EMBD_TILES) in tile coords
    scaler: (1, 1) tile of all 1.0s (for reduce)
    mean_scale: (1, 1) tile of all 1/N_EMBD
    out: same shape as x
    """
    grid_cols, _ = ttl.grid_size(dims=2)
    seq_tiles = x.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=2)

    sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    rsq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Pass 1: accumulate sum of squares across embd tiles
                    with x_dfb.wait() as x0:
                        with sq_dfb.reserve() as sq:
                            sq.store(x0 * x0)
                    with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                    with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                        acc.store(rv)
                    for j in range(EMBD_TILES - 1):
                        with x_dfb.wait() as xj:
                            with sq_dfb.reserve() as sq:
                                sq.store(xj * xj)
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as new_acc:
                            new_acc.store(av + rv)

                    # Broadcast col 0 to all cols, scale by 1/N, rsqrt
                    with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                        bc.store(ttl.math.broadcast(total, dims=[1]))
                    with bcast_dfb.wait() as bv, red_dfb.reserve() as scaled:
                        scaled.store(bv * ms)
                    with red_dfb.wait() as msq, rsq_dfb.reserve() as rsq:
                        rsq.store(ttl.math.rsqrt(msq))

                    # Pass 2: normalize x * rsqrt(mean + eps)
                    with rsq_dfb.wait() as rsqv:
                        for j in range(EMBD_TILES):
                            with x_dfb.wait() as xj, out_dfb.reserve() as o:
                                o.store(xj * rsqv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                # Pass 1: stream all embd tiles
                for j in range(EMBD_TILES):
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x[tile_idx, j], blk); tx.wait()
                # Pass 2: stream all embd tiles again
                for j in range(EMBD_TILES):
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x[tile_idx, j], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                for j in range(EMBD_TILES):
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()


def torch_rmsnorm(x, eps=1e-5):
    return torch.nn.functional.rms_norm(x, (x.size(-1),), eps=eps)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test with a small sequence first (1 token = 1 tile row, padded to 32)
    seq_len = 32  # 1 tile row
    x_torch = torch.randn(seq_len, N_EMBD, dtype=torch.bfloat16)

    expected = torch_rmsnorm(x_torch.float()).to(torch.bfloat16)

    x_tt = to_ttnn(x_torch, device)
    scaler_tt = to_ttnn(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
    ms_tt = to_ttnn(torch.full((TILE, TILE), C_MEAN_SCALE, dtype=torch.bfloat16), device)
    out_tt = to_ttnn(torch.zeros(seq_len, N_EMBD, dtype=torch.bfloat16), device)

    rmsnorm_kernel(x_tt, scaler_tt, ms_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    print(f"Input shape: {x_torch.shape}")
    print(f"Expected[0, :8]: {expected[0, :8]}")
    print(f"Got[0, :8]:      {result[0, :8]}")

    max_err = (expected.float() - result.float()).abs().max().item()
    mean_err = (expected.float() - result.float()).abs().mean().item()
    print(f"Max error: {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")

    if max_err < 1.0:
        print("PASS")
    else:
        print("FAIL")

    # Test with multiple tile rows (e.g. 4 tokens = 128 rows, but we pad)
    seq_len2 = 128
    x_torch2 = torch.randn(seq_len2, N_EMBD, dtype=torch.bfloat16)
    expected2 = torch_rmsnorm(x_torch2.float()).to(torch.bfloat16)

    x_tt2 = to_ttnn(x_torch2, device)
    out_tt2 = to_ttnn(torch.zeros(seq_len2, N_EMBD, dtype=torch.bfloat16), device)

    rmsnorm_kernel(x_tt2, scaler_tt, ms_tt, out_tt2)

    result2 = ttnn.to_torch(out_tt2)
    max_err2 = (expected2.float() - result2.float()).abs().max().item()
    print(f"\nMulti-row test (seq={seq_len2}):")
    print(f"Expected[0, :8]: {expected2[0, :8]}")
    print(f"Got[0, :8]:      {result2[0, :8]}")
    print(f"Max error: {max_err2:.6f}")
    if max_err2 < 1.0:
        print("PASS")
    else:
        print("FAIL")

    ttnn.close_device(device)
