"""
Prepare nanochat d32 weights for TT-Lang inference.

Loads the SFT checkpoint, transposes weight matrices for x @ w layout,
precomputes rotary embeddings, and saves a flat .pt bundle.

Usage:
    python prepare_weights.py [--checkpoint-dir DIR] [--step STEP] [--output PATH]

Default: loads from ~/.cache/nanochat/chatsft_checkpoints/d32/
"""

import os
import sys
import json
import math
import argparse
import torch
import pickle

TILE = 32


def pad_to_tile(t, dim=-1):
    """Pad tensor along dim to next multiple of TILE."""
    size = t.shape[dim]
    if size % TILE == 0:
        return t
    pad_size = TILE - (size % TILE)
    pad_spec = [0] * (2 * t.ndim)
    # pad_spec is in reverse dim order for F.pad
    pad_spec[2 * (t.ndim - 1 - dim) + 1] = pad_size
    return torch.nn.functional.pad(t, pad_spec)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def precompute_rotary(seq_len, head_dim, base=100000):
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    return cos, sin


def main():
    parser = argparse.ArgumentParser(description="Prepare nanochat weights for TT-Lang")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Path to checkpoint dir (default: ~/.cache/nanochat/chatsft_checkpoints/d32/)")
    parser.add_argument("--step", type=int, default=650, help="Checkpoint step (default: 650)")
    parser.add_argument("--output", type=str, default="ttlang/d32_weights.pt",
                        help="Output bundle path (default: ttlang/d32_weights.pt)")
    parser.add_argument("--tokenizer-dir", type=str, default=None,
                        help="Tokenizer directory (default: ~/.cache/nanochat/tokenizer/)")
    args = parser.parse_args()

    # Resolve paths
    base_dir = os.environ.get("NANOCHAT_BASE_DIR",
                               os.path.join(os.path.expanduser("~"), ".cache", "nanochat"))
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", "d32")
    if args.tokenizer_dir is None:
        args.tokenizer_dir = os.path.join(base_dir, "tokenizer")

    step = args.step
    print(f"Loading checkpoint from {args.checkpoint_dir} step {step}")

    # Load metadata
    meta_path = os.path.join(args.checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    model_config = meta["model_config"]
    # Patch missing keys
    if "window_pattern" not in model_config:
        model_config["window_pattern"] = "L"
    print(f"Model config: {model_config}")

    n_layer = model_config["n_layer"]
    n_head = model_config["n_head"]
    n_kv_head = model_config["n_kv_head"]
    n_embd = model_config["n_embd"]
    head_dim = n_embd // n_head
    vocab_size = model_config["vocab_size"]
    seq_len = model_config["sequence_len"]

    # Load model state dict
    model_path = os.path.join(args.checkpoint_dir, f"model_{step:06d}.pt")
    print(f"Loading {model_path}...")
    state = torch.load(model_path, map_location="cpu")
    # Strip torch.compile prefix
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    # Patch missing keys
    if "resid_lambdas" not in state:
        state["resid_lambdas"] = torch.ones(n_layer)
    if "x0_lambdas" not in state:
        state["x0_lambdas"] = torch.zeros(n_layer)

    print(f"State dict keys: {len(state)} keys")

    # Build weight bundle
    bundle = {}
    bundle["config"] = model_config

    # Token embeddings: (padded_vocab, n_embd) -> keep as-is for CPU lookup
    wte = state["transformer.wte.weight"].to(torch.bfloat16)
    bundle["wte"] = wte
    print(f"  wte: {wte.shape}")

    # LM head: nn.Linear stores (out_features, in_features) = (padded_vocab, n_embd)
    # We need (n_embd, vocab_size) for x @ w, so transpose and crop to vocab_size
    lm_head_raw = state["lm_head.weight"].to(torch.bfloat16)  # (padded_vocab, n_embd)
    lm_head = lm_head_raw[:vocab_size, :].t().contiguous()  # (n_embd, vocab_size)
    # Pad vocab dim to tile multiple
    lm_head = pad_to_tile(lm_head, dim=1)
    bundle["lm_head"] = lm_head
    print(f"  lm_head: {lm_head.shape} (transposed, cropped to vocab_size, padded)")

    # Per-layer scalars
    bundle["resid_lambdas"] = state["resid_lambdas"].float()
    bundle["x0_lambdas"] = state["x0_lambdas"].float()
    print(f"  resid_lambdas: {bundle['resid_lambdas'][:4]}...")
    print(f"  x0_lambdas: {bundle['x0_lambdas'][:4]}...")

    # Rotary embeddings
    cos, sin = precompute_rotary(seq_len * 10, head_dim)
    bundle["rotary_cos"] = cos  # (seq, head_dim/2)
    bundle["rotary_sin"] = sin
    print(f"  rotary: cos {cos.shape}, sin {sin.shape}")

    # Per-layer weights
    layers = []
    for i in range(n_layer):
        layer = {}
        prefix = f"transformer.h.{i}"

        # Attention weights: nn.Linear (out, in) -> transpose to (in, out) for x @ w
        for name, key in [("w_q", "attn.c_q.weight"),
                          ("w_k", "attn.c_k.weight"),
                          ("w_v", "attn.c_v.weight"),
                          ("w_proj", "attn.c_proj.weight")]:
            w = state[f"{prefix}.{key}"].to(torch.bfloat16).t().contiguous()
            layer[name] = w

        # MLP weights: same transpose
        w_fc = state[f"{prefix}.mlp.c_fc.weight"].to(torch.bfloat16).t().contiguous()
        w_mlp_proj = state[f"{prefix}.mlp.c_proj.weight"].to(torch.bfloat16).t().contiguous()
        layer["w_fc"] = w_fc
        layer["w_mlp_proj"] = w_mlp_proj

        # Value embedding gate (if present)
        if has_ve(i, n_layer):
            ve_gate_key = f"{prefix}.attn.ve_gate.weight"
            if ve_gate_key in state:
                # (n_kv_head, 12) -> keep as-is for CPU matmul
                layer["ve_gate"] = state[ve_gate_key].to(torch.bfloat16)
            else:
                layer["ve_gate"] = None
            # Value embedding: (padded_vocab, kv_dim) -> keep for CPU lookup
            ve_key = f"value_embeds.{i}.weight"
            if ve_key in state:
                layer["value_embed"] = state[ve_key].to(torch.bfloat16)
            else:
                layer["value_embed"] = None
        else:
            layer["ve_gate"] = None
            layer["value_embed"] = None

        layers.append(layer)
        if i == 0:
            print(f"  layer 0: w_q={layer['w_q'].shape} w_fc={layer['w_fc'].shape} "
                  f"ve={'yes' if layer['value_embed'] is not None else 'no'}")

    bundle["layers"] = layers

    # Save tokenizer data for decoding on remote
    tokenizer_pkl = os.path.join(args.tokenizer_dir, "tokenizer.pkl")
    if os.path.exists(tokenizer_pkl):
        with open(tokenizer_pkl, "rb") as f:
            bundle["tokenizer_pkl"] = f.read()  # raw bytes
        print(f"  tokenizer: included from {tokenizer_pkl}")
    else:
        print(f"  WARNING: tokenizer not found at {tokenizer_pkl}")

    # Save bundle
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(bundle, args.output)
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved bundle to {args.output} ({file_size:.1f} MB)")
    print("Copy this file to the remote and run inference.py")


if __name__ == "__main__":
    main()
