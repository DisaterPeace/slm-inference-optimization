"""Phase 2 - Quantization & dequantization, implemented from scratch.

We DON'T use a quantization library here on purpose. Implementing quantize +
dequantize by hand is the best way to actually understand it (and to answer
interview questions about it). We run it on a REAL weight matrix from the model
so the error and memory numbers are meaningful.

Key ideas demonstrated:
  - symmetric INT8 quantization (one scale, no zero-point)
  - asymmetric / "affine" INT4 GROUP quantization (scale + zero-point per group)
      this is how GPTQ/AWQ actually store 4-bit weights
  - DEQUANTIZATION: turning the packed ints back into floats for the matmul
  - the accuracy/memory tradeoff, measured numerically
"""

from __future__ import annotations

import torch


def quantize_int8_symmetric(w: torch.Tensor):
    """Per-tensor symmetric INT8.

    "Symmetric" = the range is centered on 0, so we need no zero-point: the
    integer 0 already maps to the float 0.0. We map [-absmax, +absmax] onto the
    integer range [-127, 127].
    """
    absmax = w.abs().max()
    scale = absmax / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_int8_symmetric(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Undo it: float ≈ integer * scale."""
    return q.to(torch.float32) * scale


def quantize_int4_group(w: torch.Tensor, group_size: int = 128):
    """Asymmetric INT4 with a separate scale+zero-point per GROUP of weights.

    Why groups? A single scale for a whole big matrix wastes precision: one
    outlier weight blows up `absmax` and makes every step huge. Splitting each
    row into small groups (e.g. 128 weights) lets each group fit its own tight
    min/max, dramatically lowering error for almost no extra storage. This is
    exactly what GPTQ/AWQ do (typical group_size = 128).

    Asymmetric ("affine") = we track BOTH min and max, so we need a zero-point.
    4 bits -> integers 0..15 (16 levels).
    """
    out_features, in_features = w.shape
    assert in_features % group_size == 0, "in_features must divide by group_size"

    w_groups = w.reshape(out_features, in_features // group_size, group_size)

    wmin = w_groups.min(dim=-1, keepdim=True).values
    wmax = w_groups.max(dim=-1, keepdim=True).values

    scale = (wmax - wmin) / 15.0
    scale = scale.clamp(min=1e-8)                       # avoid divide-by-zero
    zero_point = torch.round(-wmin / scale)             # which int maps to 0.0

    q = torch.round(w_groups / scale + zero_point).clamp(0, 15).to(torch.uint8)
    return q, scale, zero_point


def dequantize_int4_group(q, scale, zero_point, original_shape) -> torch.Tensor:
    """Undo it: float ≈ (integer - zero_point) * scale, then reshape back."""
    w = (q.to(torch.float32) - zero_point) * scale
    return w.reshape(original_shape)


def quantize_intn_group(w: torch.Tensor, nbits: int, group_size: int = 128):
    """Asymmetric n-bit group quantization — generalizes the INT4 scheme above.

    Same affine min/max-per-group idea, but to `nbits` bits → integers 0..(2**nbits-1).
    Used to model the storage/error of INT2 (GGUF Q2-style) and INT4 (AWQ/GPTQ-style)
    on a real weight matrix. Returns (q, scale, zero_point).
    """
    levels = (1 << nbits) - 1
    out_features, in_features = w.shape
    assert in_features % group_size == 0, "in_features must divide by group_size"
    w_groups = w.reshape(out_features, in_features // group_size, group_size)
    wmin = w_groups.min(dim=-1, keepdim=True).values
    wmax = w_groups.max(dim=-1, keepdim=True).values
    scale = ((wmax - wmin) / levels).clamp(min=1e-8)
    zero_point = torch.round(-wmin / scale)
    q = torch.round(w_groups / scale + zero_point).clamp(0, levels).to(torch.uint8)
    return q, scale, zero_point


def dequantize_intn_group(q, scale, zero_point, original_shape) -> torch.Tensor:
    """Undo n-bit group quantization."""
    w = (q.to(torch.float32) - zero_point) * scale
    return w.reshape(original_shape)


def report(name: str, w: torch.Tensor, w_dq: torch.Tensor, stored_bytes: float) -> None:
    """Print accuracy + memory for one quantization scheme."""
    # Relative error: how far off, on average, is the dequantized weight?
    err = (w - w_dq).abs()
    rel_err = (err.sum() / w.abs().sum()).item()
    fp16_bytes = w.numel() * 2
    print(
        f"{name:<22} | mean abs err {err.mean().item():.5f} | "
        f"rel err {rel_err*100:5.2f}% | "
        f"{stored_bytes/1024:7.1f} KB ({fp16_bytes/stored_bytes:.1f}x smaller than fp16)"
    )


def main() -> None:
    # Load a real model so we quantize an actual weight matrix (not random data).
    from transformers import AutoModelForCausalLM
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)

    # Grab a real weight matrix: the first layer's attention query projection.
    # (This is one of the W_Q matrices that produce the Query vectors from
    #  Phase 1 -- so we're literally quantizing the attention weights.)
    name, w = next(
        (n, p) for n, p in model.named_parameters() if "q_proj.weight" in n
    )
    w = w.detach().float().cpu()
    print(f"Quantizing real weight: {name}")
    print(f"shape = {tuple(w.shape)}  ({w.numel():,} values)\n")

    # --- INT8 symmetric ---
    q8, s8 = quantize_int8_symmetric(w)
    w8 = dequantize_int8_symmetric(q8, s8)
    int8_bytes = q8.numel() * 1 + s8.numel() * 4          # int8 weights + 1 scale
    report("INT8 symmetric", w, w8, int8_bytes)

    # --- INT4 group (GPTQ/AWQ style) ---
    q4, s4, z4 = quantize_int4_group(w, group_size=128)
    w4 = dequantize_int4_group(q4, s4, z4, w.shape)
    # 4-bit weights (so half a byte each) + fp16 scales + fp16 zero-points
    int4_bytes = q4.numel() * 0.5 + s4.numel() * 2 + z4.numel() * 2
    report("INT4 group=128", w, w4, int4_bytes)

    print(
        "\nTakeaway: INT4 is ~4x smaller than FP16 but loses some accuracy. "
        "Grouping keeps that loss small. AWQ/GPTQ are smarter ways to pick what\n"
        "to round so the loss matters even less (explained separately)."
    )


if __name__ == "__main__":
    main()
