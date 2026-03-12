# Rotary embedding kernel for nanochat inference.
#
# Applies rotary position embeddings to queries and keys:
#   y1 = x1 * cos + x2 * sin
#   y2 = x1 * (-sin) + x2 * cos
# where x1 = x[..., :d], x2 = x[..., d:], d = head_dim/2
#
# For d32: head_dim=128, so d=64 (2 tiles).
# Input layout for single head: (seq, head_dim) = (seq_tiles, 4 tiles)
# cos/sin: (seq, head_dim/2) = (seq_tiles, 2 tiles)
#
# We process per-head, so input is (seq_tiles, head_dim_tiles).
# Grid="auto" parallelizes over sequence positions.

import torch
import ttnn
import ttl

TILE = 32
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
HEAD_DIM_TILES = HEAD_DIM // TILE  # 4
HALF_DIM_TILES = HALF_DIM // TILE  # 2


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@ttl.kernel(grid="auto")
def rotary_kernel(x, cos, sin, out):
    """Apply rotary embeddings to a single head.

    x: (seq, head_dim) = (seq_tiles, 4) in tiles
    cos: (seq, half_dim) = (seq_tiles, 2) in tiles
    sin: (seq, half_dim) = (seq_tiles, 2) in tiles
    out: same shape as x

    For each seq position:
      x1, x2 = x[:, :half], x[:, half:]
      out[:, :half] = x1 * cos + x2 * sin
      out[:, half:] = x1 * (-sin) + x2 * cos
    """
    grid_cols, _ = ttl.grid_size(dims=2)
    seq_tiles = x.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // grid_cols)

    # Separate DFBs for x1 (first half) and x2 (second half)
    x1_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    x2_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    cos_dfb = ttl.make_dataflow_buffer_like(cos, shape=(1, 1), buffer_factor=2)
    sin_dfb = ttl.make_dataflow_buffer_like(sin, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < seq_tiles:
                for j in range(HALF_DIM_TILES):
                    with x1_dfb.wait() as x1, x2_dfb.wait() as x2, cos_dfb.wait() as c, sin_dfb.wait() as s:
                        # y1 = x1*cos + x2*sin
                        with out_dfb.reserve() as o:
                            o.store(x1 * c + x2 * s)
                        # y2 = -x1*sin + x2*cos
                        with out_dfb.reserve() as o:
                            o.store(ttl.math.neg(x1) * s + x2 * c)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < seq_tiles:
                for j in range(HALF_DIM_TILES):
                    with x1_dfb.reserve() as blk:
                        tx = ttl.copy(x[t, j], blk); tx.wait()
                    with x2_dfb.reserve() as blk:
                        tx = ttl.copy(x[t, j + HALF_DIM_TILES], blk); tx.wait()
                    with cos_dfb.reserve() as blk:
                        tx = ttl.copy(cos[t, j], blk); tx.wait()
                    with sin_dfb.reserve() as blk:
                        tx = ttl.copy(sin[t, j], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < seq_tiles:
                for j in range(HALF_DIM_TILES):
                    # y1 tile (first half)
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[t, j]); tx.wait()
                    # y2 tile (second half)
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[t, j + HALF_DIM_TILES]); tx.wait()


def torch_rotary(x, cos, sin):
    """PyTorch reference rotary embedding (matches nanochat)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test: single token, single head
    seq_len = 32  # 1 tile row
    print(f"=== Test: Rotary emb (seq={seq_len}, head_dim={HEAD_DIM}) ===")

    x = torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16)
    cos = torch.randn(seq_len, HALF_DIM, dtype=torch.bfloat16)
    sin = torch.randn(seq_len, HALF_DIM, dtype=torch.bfloat16)

    expected = torch_rotary(x.float(), cos.float(), sin.float()).to(torch.bfloat16)

    out_tt = to_ttnn(torch.zeros(seq_len, HEAD_DIM, dtype=torch.bfloat16), device)
    rotary_kernel(to_ttnn(x, device), to_ttnn(cos, device), to_ttnn(sin, device), out_tt)

    result = ttnn.to_torch(out_tt)
    err = (expected.float() - result.float()).abs().max().item()
    mean_err = (expected.float() - result.float()).abs().mean().item()
    print(f"  Expected[0,:4]: {expected[0,:4]}")
    print(f"  Got[0,:4]:      {result[0,:4]}")
    print(f"  Max error: {err:.4f}, Mean: {mean_err:.4f}")
    print(f"  {'PASS' if err < 0.5 else 'FAIL'}")

    # Test with multiple tile rows (e.g. prefill)
    seq_len2 = 128
    print(f"\n=== Test: Rotary emb (seq={seq_len2}, head_dim={HEAD_DIM}) ===")
    x2 = torch.randn(seq_len2, HEAD_DIM, dtype=torch.bfloat16)
    cos2 = torch.randn(seq_len2, HALF_DIM, dtype=torch.bfloat16)
    sin2 = torch.randn(seq_len2, HALF_DIM, dtype=torch.bfloat16)
    expected2 = torch_rotary(x2.float(), cos2.float(), sin2.float()).to(torch.bfloat16)
    out2 = to_ttnn(torch.zeros(seq_len2, HEAD_DIM, dtype=torch.bfloat16), device)
    rotary_kernel(to_ttnn(x2, device), to_ttnn(cos2, device), to_ttnn(sin2, device), out2)
    result2 = ttnn.to_torch(out2)
    err2 = (expected2.float() - result2.float()).abs().max().item()
    print(f"  Max error: {err2:.4f} {'PASS' if err2 < 0.5 else 'FAIL'}")

    ttnn.close_device(device)
