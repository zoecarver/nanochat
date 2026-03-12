# Residual connection kernels for nanochat inference.
#
# 1. residual_add_kernel: out = x + y (simple residual)
# 2. scaled_residual_kernel: out = lambda_r * x + lambda_0 * x0 (per-layer scaling)
# 3. softcap_kernel: out = softcap * tanh(x / softcap) (logit capping)

import torch
import ttnn
import ttl

TILE = 32
N_EMBD = 2048
EMBD_TILES = N_EMBD // TILE


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@ttl.kernel(grid="auto")
def residual_add_kernel(x, y, out):
    """out = x + y, elementwise streaming."""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with x_dfb.wait() as xv, y_dfb.wait() as yv, out_dfb.reserve() as o:
                    o.store(xv + yv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()
                with y_dfb.reserve() as blk:
                    tx = ttl.copy(y[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


@ttl.kernel(grid="auto")
def scaled_residual_kernel(x, x0, lambda_r_tile, lambda_0_tile, out):
    """out = lambda_r * x + lambda_0 * x0 (per-layer residual scaling).

    lambda_r_tile, lambda_0_tile: (1, 1) tiles filled with scalar values.
    """
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    x0_dfb = ttl.make_dataflow_buffer_like(x0, shape=(1, 1), buffer_factor=2)
    lr_dfb = ttl.make_dataflow_buffer_like(lambda_r_tile, shape=(1, 1), buffer_factor=1)
    l0_dfb = ttl.make_dataflow_buffer_like(lambda_0_tile, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with lr_dfb.wait() as lr, l0_dfb.wait() as l0:
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    with x_dfb.wait() as xv, x0_dfb.wait() as x0v, out_dfb.reserve() as o:
                        o.store(lr * xv + l0 * x0v)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        with lr_dfb.reserve() as blk:
            tx = ttl.copy(lambda_r_tile[0, 0], blk); tx.wait()
        with l0_dfb.reserve() as blk:
            tx = ttl.copy(lambda_0_tile[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()
                with x0_dfb.reserve() as blk:
                    tx = ttl.copy(x0[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


@ttl.kernel(grid="auto")
def softcap_kernel(x, inv_cap_tile, cap_tile, out):
    """out = softcap * tanh(x / softcap) = softcap * tanh(x * inv_softcap)"""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    ic_dfb = ttl.make_dataflow_buffer_like(inv_cap_tile, shape=(1, 1), buffer_factor=1)
    c_dfb = ttl.make_dataflow_buffer_like(cap_tile, shape=(1, 1), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with ic_dfb.wait() as ic, c_dfb.wait() as cap:
            for local_t in range(tiles_per_core):
                t = core_x * tiles_per_core + local_t
                if t < total_tiles:
                    with x_dfb.wait() as xv, out_dfb.reserve() as o:
                        o.store(cap * ttl.math.tanh(xv * ic))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        with ic_dfb.reserve() as blk:
            tx = ttl.copy(inv_cap_tile[0, 0], blk); tx.wait()
        with c_dfb.reserve() as blk:
            tx = ttl.copy(cap_tile[0, 0], blk); tx.wait()
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test 1: Simple residual add
    print("=== Test 1: Residual add ===")
    x1 = torch.randn(32, N_EMBD, dtype=torch.bfloat16)
    y1 = torch.randn(32, N_EMBD, dtype=torch.bfloat16)
    expected1 = x1 + y1
    out1 = to_ttnn(torch.zeros_like(x1), device)
    residual_add_kernel(to_ttnn(x1, device), to_ttnn(y1, device), out1)
    result1 = ttnn.to_torch(out1)
    err1 = (expected1.float() - result1.float()).abs().max().item()
    print(f"  Max error: {err1:.4f} {'PASS' if err1 < 0.01 else 'FAIL'}")

    # Test 2: Scaled residual
    print("\n=== Test 2: Scaled residual ===")
    x2 = torch.randn(32, N_EMBD, dtype=torch.bfloat16)
    x0_2 = torch.randn(32, N_EMBD, dtype=torch.bfloat16)
    lr_val, l0_val = 0.95, 0.1
    expected2 = (lr_val * x2.float() + l0_val * x0_2.float()).to(torch.bfloat16)
    lr_tile = torch.full((TILE, TILE), lr_val, dtype=torch.bfloat16)
    l0_tile = torch.full((TILE, TILE), l0_val, dtype=torch.bfloat16)
    out2 = to_ttnn(torch.zeros_like(x2), device)
    scaled_residual_kernel(
        to_ttnn(x2, device), to_ttnn(x0_2, device),
        to_ttnn(lr_tile, device), to_ttnn(l0_tile, device), out2)
    result2 = ttnn.to_torch(out2)
    err2 = (expected2.float() - result2.float()).abs().max().item()
    print(f"  Max error: {err2:.4f} {'PASS' if err2 < 0.1 else 'FAIL'}")

    # Test 3: Softcap
    print("\n=== Test 3: Softcap (cap=15) ===")
    x3 = torch.randn(32, N_EMBD, dtype=torch.bfloat16) * 20  # some values > softcap
    softcap = 15.0
    expected3 = (softcap * torch.tanh(x3.float() / softcap)).to(torch.bfloat16)
    ic_tile = torch.full((TILE, TILE), 1.0 / softcap, dtype=torch.bfloat16)
    c_tile = torch.full((TILE, TILE), softcap, dtype=torch.bfloat16)
    out3 = to_ttnn(torch.zeros_like(x3), device)
    softcap_kernel(to_ttnn(x3, device), to_ttnn(ic_tile, device), to_ttnn(c_tile, device), out3)
    result3 = ttnn.to_torch(out3)
    err3 = (expected3.float() - result3.float()).abs().max().item()
    print(f"  Expected[0,:4]: {expected3[0,:4]}")
    print(f"  Got[0,:4]:      {result3[0,:4]}")
    print(f"  Max error: {err3:.4f} {'PASS' if err3 < 1.0 else 'FAIL'}")

    ttnn.close_device(device)
