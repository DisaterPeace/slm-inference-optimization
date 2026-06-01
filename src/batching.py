"""Phase 8 - Continuous batching & concurrency.

Two ideas that together explain how a real server gets throughput:

  1) BATCHING THROUGHPUT (measured live):
     A single decode step underutilizes the GPU (we proved this in Phase 1 -
     decode is overhead/bandwidth bound on a small model). If we decode B
     sequences *at once*, we reuse the same weight read for B tokens, so total
     tokens/sec climbs with batch size until we hit a memory/compute wall.
     This is the engine behind "continuous batching" in vLLM/TGI.

  2) CONCURRENCY CAPACITY (memory math, reuses Phase 3 numbers):
     How many requests fit in a fixed KV-memory budget? Naive max-length
     reservation fits few; PagedAttention fits many more. Throughput in a real
     server is gated by how many sequences you can hold at once - so this is
     PagedAttention's benefit expressed as "concurrent users per GPU".
"""

from __future__ import annotations

import time

import torch

from paged_attention import KVConfig, BLOCK_SIZE


@torch.no_grad()
def measure_batched_throughput(
    model, tokenizer, device: str,
    batch_sizes=(1, 2, 4, 8, 16),
    decode_steps: int = 32,
    prompt: str = "The history of computing began",
):
    """For each batch size, run `decode_steps` cached decode steps over B
    sequences in parallel and report aggregate tokens/sec + per-step latency.
    """
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    base_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    results = []

    for B in batch_sizes:
        input_ids = base_ids.repeat(B, 1)              # [B, prompt_len]
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(-1)    # [B, 1]

        # warm the decode kernels for this batch shape before timing
        for _ in range(3):
            out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(-1)
        sync()

        t0 = time.perf_counter()
        for _ in range(decode_steps):
            out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(-1)
        sync()
        dt = time.perf_counter() - t0

        total_tokens = B * decode_steps
        results.append({
            "batch": B,
            "tokens_per_sec": total_tokens / dt,
            "ms_per_step": dt / decode_steps * 1000,
        })

    return results


def concurrency_capacity(kv: KVConfig, budget_gb: float, avg_len: int, max_len: int):
    """How many concurrent requests fit in `budget_gb` of KV-cache memory,
    naive (reserve max_len each) vs paged (blocks for actual avg length)."""
    budget = budget_gb * 1024**3
    naive_fit = int(budget // (max_len * kv.bytes_per_token))
    paged_blocks_per_seq = (avg_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    paged_fit = int(budget // (paged_blocks_per_seq * kv.bytes_per_block))
    return {
        "budget_gb": budget_gb,
        "avg_len": avg_len,
        "max_len": max_len,
        "naive_fit": naive_fit,
        "paged_fit": paged_fit,
        "ratio": (paged_fit / naive_fit) if naive_fit else 0,
    }


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device).eval()

    print(f"Device: {device}\n")
    print(f"{'batch':>6} | {'tokens/sec':>11} | {'ms/step':>8}")
    print("-" * 32)
    for r in measure_batched_throughput(model, tok, device):
        print(f"{r['batch']:>6} | {r['tokens_per_sec']:>11.1f} | {r['ms_per_step']:>8.2f}")

    kv = KVConfig()
    cap = concurrency_capacity(kv, budget_gb=4, avg_len=180, max_len=2048)
    print(f"\nIn a {cap['budget_gb']:.0f} GB KV budget (avg len {cap['avg_len']}): "
          f"naive fits ~{cap['naive_fit']} requests, paged fits ~{cap['paged_fit']} "
          f"({cap['ratio']:.1f}x more concurrent users).")


if __name__ == "__main__":
    main()
