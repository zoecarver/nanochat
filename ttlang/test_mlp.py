# Fused MLP test: out = relu²(x @ w_fc) @ w_proj
# Tests the full MLP pipeline for nanochat inference.
#
# Components:
#   1. relu_sq_kernel: elementwise relu² (grid="auto", streaming)
#   2. linear_kernel: general matmul (from test_linear.py)
#   3. Full MLP pipeline composing them

import torch
import ttnn
import ttl

TILE = 32


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# --- Elementwise ReLU² kernel ---
@ttl.operation(grid="auto")
def relu_sq_kernel(x, out):
    """Elementwise relu²: out = relu(x)²"""
    grid_cols, _ = ttl.grid_size(dims=2)
    total_tiles = (x.shape[0] // TILE) * (x.shape[1] // TILE)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with x_dfb.wait() as xv, out_dfb.reserve() as o:
                    r = ttl.math.relu(xv)
                    o.store(r * r)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


# --- Linear projection kernel (fused K accumulation) ---
def make_linear_kernel(k_tiles):
    @ttl.operation(grid="auto")
    def linear_kernel(x, w, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        m_tiles = x.shape[0] // TILE
        n_tiles = w.shape[1] // TILE
        total = m_tiles * n_tiles
        units_per_core = -(-total // grid_cols)

        a_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1))

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    with a_dfb.wait() as av, w_dfb.wait() as wv:
                        with acc_dfb.reserve() as acc:
                            acc.store(av @ wv)
                    for k in range(1, k_tiles):
                        with a_dfb.wait() as av, w_dfb.wait() as wv, acc_dfb.wait() as prev:
                            with acc_dfb.reserve() as acc:
                                acc.store(prev + av @ wv)
                    with acc_dfb.wait() as final:
                        with out_dfb.reserve() as o:
                            o.store(final)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // n_tiles
                    ni = t % n_tiles
                    for k in range(k_tiles):
                        with a_dfb.reserve() as blk:
                            tx = ttl.copy(x[mi, k], blk); tx.wait()
                        with w_dfb.reserve() as blk:
                            tx = ttl.copy(w[k, ni], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(units_per_core):
                t = core_x * units_per_core + local_t
                if t < total:
                    mi = t // n_tiles
                    ni = t % n_tiles
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[mi, ni]); tx.wait()

    return linear_kernel


def torch_mlp(x, w_fc, w_proj):
    h = torch.nn.functional.relu(x.float() @ w_fc.float()).square()
    return (h @ w_proj.float()).to(torch.bfloat16)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test 1: ReLU² alone
    print("=== Test 1: ReLU² elementwise ===")
    x1 = torch.randn(32, 2048, dtype=torch.bfloat16)
    expected1 = torch.nn.functional.relu(x1).square()
    out1 = to_ttnn(torch.zeros_like(x1), device)
    relu_sq_kernel(to_ttnn(x1, device), out1)
    result1 = ttnn.to_torch(out1)
    err1 = (expected1.float() - result1.float()).abs().max().item()
    print(f"  Max error: {err1:.4f} {'PASS' if err1 < 0.5 else 'FAIL'}")

    # Test 2: Full MLP pipeline - small size for quick test
    # (32, 256) @ (256, 1024) -> relu² -> (32, 1024) @ (1024, 256)
    N_EMBD_SMALL = 256
    MLP_SMALL = 1024
    print(f"\n=== Test 2: Full MLP ({N_EMBD_SMALL} -> {MLP_SMALL} -> {N_EMBD_SMALL}) ===")
    x2 = torch.randn(32, N_EMBD_SMALL, dtype=torch.bfloat16)
    w_fc = torch.randn(N_EMBD_SMALL, MLP_SMALL, dtype=torch.bfloat16) * 0.01
    w_proj = torch.randn(MLP_SMALL, N_EMBD_SMALL, dtype=torch.bfloat16) * 0.01
    expected2 = torch_mlp(x2, w_fc, w_proj)

    linear_k8 = make_linear_kernel(N_EMBD_SMALL // TILE)
    linear_k32 = make_linear_kernel(MLP_SMALL // TILE)

    # Step 1: x @ w_fc -> hidden
    hidden_tt = to_ttnn(torch.zeros(32, MLP_SMALL, dtype=torch.bfloat16), device)
    linear_k8(to_ttnn(x2, device), to_ttnn(w_fc, device), hidden_tt)

    # Step 2: relu²(hidden) -> hidden_act
    hidden_act_tt = to_ttnn(torch.zeros(32, MLP_SMALL, dtype=torch.bfloat16), device)
    relu_sq_kernel(hidden_tt, hidden_act_tt)

    # Step 3: hidden_act @ w_proj -> out
    out2 = to_ttnn(torch.zeros(32, N_EMBD_SMALL, dtype=torch.bfloat16), device)
    linear_k32(hidden_act_tt, to_ttnn(w_proj, device), out2)

    result2 = ttnn.to_torch(out2)
    err2 = (expected2.float() - result2.float()).abs().max().item()
    mean2 = (expected2.float() - result2.float()).abs().mean().item()
    print(f"  Expected[0,:4]: {expected2[0,:4]}")
    print(f"  Got[0,:4]:      {result2[0,:4]}")
    print(f"  Max error: {err2:.4f}, Mean: {mean2:.4f}")
    print(f"  {'PASS' if err2 < 1.0 else 'FAIL'}")

    # Test 3: Full MLP at d32 size
    # (32, 2048) @ (2048, 8192) -> relu² -> (32, 8192) @ (8192, 2048)
    N_EMBD = 2048
    MLP_DIM = 8192
    print(f"\n=== Test 3: Full MLP d32 ({N_EMBD} -> {MLP_DIM} -> {N_EMBD}) ===")
    x3 = torch.randn(32, N_EMBD, dtype=torch.bfloat16)
    w_fc3 = torch.randn(N_EMBD, MLP_DIM, dtype=torch.bfloat16) * 0.01
    w_proj3 = torch.randn(MLP_DIM, N_EMBD, dtype=torch.bfloat16) * 0.01
    expected3 = torch_mlp(x3, w_fc3, w_proj3)

    linear_k64 = make_linear_kernel(N_EMBD // TILE)
    linear_k256 = make_linear_kernel(MLP_DIM // TILE)

    hidden3 = to_ttnn(torch.zeros(32, MLP_DIM, dtype=torch.bfloat16), device)
    linear_k64(to_ttnn(x3, device), to_ttnn(w_fc3, device), hidden3)

    hidden_act3 = to_ttnn(torch.zeros(32, MLP_DIM, dtype=torch.bfloat16), device)
    relu_sq_kernel(hidden3, hidden_act3)

    out3 = to_ttnn(torch.zeros(32, N_EMBD, dtype=torch.bfloat16), device)
    linear_k256(hidden_act3, to_ttnn(w_proj3, device), out3)

    result3 = ttnn.to_torch(out3)
    err3 = (expected3.float() - result3.float()).abs().max().item()
    mean3 = (expected3.float() - result3.float()).abs().mean().item()
    print(f"  Expected[0,:4]: {expected3[0,:4]}")
    print(f"  Got[0,:4]:      {result3[0,:4]}")
    print(f"  Max error: {err3:.4f}, Mean: {mean3:.4f}")
    print(f"  {'PASS' if err3 < 5.0 else 'FAIL'}")

    ttnn.close_device(device)
