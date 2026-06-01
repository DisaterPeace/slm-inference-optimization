# NanoServe — SLM Inference Optimization Test Bench

A local, interactive test bench for exploring the optimizations that make small
language model (SLM) inference fast and memory-efficient — KV caching,
weight quantization, continuous batching, PagedAttention memory layout, and
host↔device transfer paths — with **live, measured** telemetry on every run.

Built around `Qwen2.5-0.5B-Instruct` and `Phi-4-mini-3.8B`, served through a
FastAPI + WebSocket backend with a dependency-free vanilla-JS dashboard.

> **Design principle: measure honestly, label clearly.**
> Some optimizations (KV cache on/off, FP16/INT8/4-bit quantization, the
> transfer path) genuinely change the live forward pass and every number is
> measured on the GPU. Others — notably PagedAttention — are vLLM kernel-level
> features that HuggingFace does not expose, so they are reproduced as a
> **faithful memory-model simulation** (real KV byte math, clearly labelled as
> such in the UI) rather than faked into the live path. Knowing which is which
> is the point.

---

## Features

- **Live generation** with per-token latency streamed over WebSocket
  (TTFT, TPOT, p50/p90/p99, tokens/sec).
- **A/B benchmarking**: run the same prompt with the KV cache on vs. off and
  see the O(N) vs. O(N²) divergence as the smoothed latency curves separate.
- **Quantization**, switchable per run:
  - **FP16 / INT8 / 4-bit (NF4)** load and run *live* via bitsandbytes.
  - **INT8 / INT4 (AWQ/GPTQ-style group) / INT2 (GGUF-style)** size-and-error
    are computed from scratch on a real weight matrix.
- **Continuous batching** throughput sweep (batch 1→16) showing why shared
  weight reads give near-free throughput scaling.
- **PagedAttention memory simulation**: a block allocator + page grid showing
  reserved-but-wasted vs. used KV memory, and the resulting concurrency
  capacity, for a chosen block size.
- **Zero-copy transfers**: pinned vs. pageable host→device bandwidth benchmark,
  plus a real pinned-buffer decode path you can toggle.
- **Multi-turn chat** with full conversation context, a **Stop** button that
  actually cancels generation server-side, and a live **VRAM gauge** and
  **event console** fed by real `torch.cuda` readings.

## Tech stack

`Python 3.12` · `PyTorch 2.x (CUDA 12.8 / Blackwell sm_120)` · `transformers` ·
`bitsandbytes` · `FastAPI` · `uvicorn` · vanilla JS + Canvas (no build step).

## Architecture

```
src/
  server.py            FastAPI app: WebSocket streaming generation + REST endpoints.
                       Generation runs in a worker thread with cooperative
                       cancellation so Stop frees the GPU and the event loop
                       never blocks.
  quantization.py      From-scratch symmetric INT8 and affine n-bit group
                       quantization / dequantization (the AWQ/GPTQ scheme).
  paged_attention.py   KV-cache byte math + PagedAttention block-allocator model.
  zero_copy.py         Pinned vs. pageable host↔device transfer timing.
  batching.py          Batched-decode throughput + concurrency-capacity model.
web/
  index.html, app.js, style.css   Single-page dashboard (no framework).
```

## Running it

```bash
# 1. create a venv (Python 3.11/3.12) and install torch for your CUDA build
python -m venv .venv && . .venv/Scripts/activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. launch the server
python -m uvicorn server:app --app-dir src --port 8000

# 3. open http://localhost:8000
```

The default model (`Qwen2.5-0.5B`) loads at startup. `Phi-4-mini-3.8B` loads on
first use and should be run in **4-bit** to fit an 8 GB GPU.

## Key finding

On small models served single-stream, decode is **overhead/bandwidth-bound**,
not compute-bound: each step is dominated by streaming the model weights out of
VRAM, so the KV cache shows a modest (~1.2×) speedup and pinned-memory transfers
are negligible (you move ~8 bytes/token across PCIe). The same techniques scale
dramatically with model size and concurrency — which is exactly what the
batching and PagedAttention panels make visible. The test bench is built to show
*where* each optimization does and does not pay off, with the numbers to back it.

## Hardware

Developed and tested on an NVIDIA RTX 5060 Laptop GPU (8 GB, Blackwell sm_120),
Windows 11, Python 3.12, PyTorch cu128.
