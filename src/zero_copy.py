"""Phase 4 - Eliminating extra CPU<->GPU copies ("zero-copy" optimizations).

Two measured demos:

  1) PINNED vs PAGEABLE host memory for host->device transfers.
     Pinned (page-locked) memory lets the GPU's copy engine read it directly
     and asynchronously, skipping a hidden staging-buffer hop.

  2) PER-TOKEN SYNC cost. A decode loop that calls `.item()` every step forces
     a GPU->CPU sync each token, stalling CPU/GPU overlap. Keeping results on
     the GPU and copying once at the end removes those stalls. This is exactly
     the `int(argmax.item())` pattern in our phase0/phase1 code.
"""

from __future__ import annotations

import torch


def cuda_time(fn, iters: int = 1) -> float:
    """Time a GPU operation accurately using CUDA events (ms). Includes a sync."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def demo_pinned_vs_pageable() -> None:
    print("=== Demo 1: pinned vs pageable host->device transfer ===")
    n = 256 * 1024 * 1024 // 4  # 256 MB of float32
    nbytes = n * 4

    pageable = torch.empty(n, dtype=torch.float32)              # normal CPU memory
    pinned = torch.empty(n, dtype=torch.float32).pin_memory()   # page-locked

    # warmup
    _ = pageable.to("cuda"); torch.cuda.synchronize()

    t_pageable = cuda_time(lambda: pageable.to("cuda", non_blocking=False), iters=5)
    t_pinned = cuda_time(lambda: pinned.to("cuda", non_blocking=True), iters=5)

    gb = nbytes / 1024**3
    print(f"transfer size: {gb*1024:.0f} MB")
    print(f"pageable: {t_pageable:6.2f} ms  ({gb/(t_pageable/1000):5.1f} GB/s)")
    print(f"pinned:   {t_pinned:6.2f} ms  ({gb/(t_pinned/1000):5.1f} GB/s)")
    print(f"-> pinned is {t_pageable/t_pinned:.2f}x faster\n")


def demo_per_token_sync() -> None:
    print("=== Demo 2: per-token .item() sync vs on-GPU accumulation ===")
    steps = 512
    # A small bit of GPU work each step, standing in for one decode step.
    w = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    def step_compute():
        x = torch.randn(1, 1024, device="cuda", dtype=torch.float16)
        return (x @ w).argmax(dim=-1)  # returns a GPU tensor (the "next token")

    # warmup
    for _ in range(10):
        step_compute()
    torch.cuda.synchronize()

    # --- BAD: copy each token to CPU every step (forces a sync per token) ---
    def bad():
        out = []
        for _ in range(steps):
            tok = step_compute()
            out.append(int(tok.item()))   # <-- GPU->CPU sync EVERY step
        return out

    # --- GOOD: keep tokens on GPU, single copy at the very end ---
    def good():
        toks = []
        for _ in range(steps):
            toks.append(step_compute())   # stays on GPU
        all_tokens = torch.cat(toks)
        return all_tokens.cpu().tolist()  # <-- ONE GPU->CPU copy total

    t_bad = cuda_time(bad)
    t_good = cuda_time(good)
    print(f"per-token .item() sync:  {t_bad:7.2f} ms  ({t_bad/steps*1000:.1f} us/token)")
    print(f"on-GPU, copy once:       {t_good:7.2f} ms  ({t_good/steps*1000:.1f} us/token)")
    print(f"-> removing per-token syncs: {t_bad/t_good:.2f}x faster\n")


def main() -> None:
    assert torch.cuda.is_available(), "needs a GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    demo_pinned_vs_pageable()
    demo_per_token_sync()


if __name__ == "__main__":
    main()
