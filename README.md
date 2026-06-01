# NanoServe — SLM Inference Optimization Test Bench

An interactive playground for tuning small-language-model (SLM) inference and
**seeing exactly what each optimization changes** — quantization (FP16 / INT8 /
4-bit) and KV caching applied live, with every latency and memory number measured
on the GPU, and any two runs diffable side by side.

Built around `Qwen2.5-0.5B-Instruct` and `Phi-4-mini-3.8B`, served through a
FastAPI + WebSocket backend with a dependency-free vanilla-JS dashboard.

> **Design principle: measure honestly, label clearly.** Every knob in the UI
> changes the *live* forward pass and every metric is measured on the GPU.
> LLM decode is **memory-bandwidth-bound**, so the playground is built to reveal
> *where each optimization does and doesn't pay off* — including honest results
> like "4-bit buys memory, not speed, on a small model." Engine-level techniques
> that HuggingFace doesn't expose (PagedAttention, fused kernels) are named as
> such rather than faked.

---

## Features

- **Chat playground** — multi-turn conversation with full context, greedy
  (deterministic) decoding so comparisons isolate the *optimization*, not
  sampling noise. A **Stop** button cancels generation server-side (worker
  thread + cooperative cancellation, so the GPU is freed immediately).
- **Live, switchable optimizations** — **model** (0.5B / 3.8B), **quantization**
  (FP16 / INT8 / 4-bit NF4 via bitsandbytes), and **KV cache** on/off, all
  applied to the live run.
- **Two A/B modes**:
  - *cache on vs. off* — the O(N) vs. O(N²) decode divergence.
  - *optimized vs. baseline* — 4-bit + KV cache stacked against FP16 + no cache,
    reporting the combined speed **and** VRAM delta (honestly, even when the
    "optimized" path is slower).
- **Live telemetry** — TTFT, TPOT, p50/p90/p99, tokens/sec, and a VRAM gauge fed
  by real `torch.cuda.memory_allocated`, plus a smoothed per-token latency chart.
- **Run history + compare** — every run is logged; pick any two to see a
  metric-delta table and a side-by-side output diff.
- **Event console** — raw server events (model load, prefill/TTFT, decode mode)
  streamed live.

## Tech stack

`Python 3.12` · `PyTorch 2.x (CUDA 12.8 / Blackwell sm_120)` · `transformers` ·
`bitsandbytes` · `FastAPI` · `uvicorn` · vanilla JS + Canvas (no build step).

## Architecture

```
src/
  server.py            FastAPI app: WebSocket streaming generation with per-token
                       timing, greedy decoding, multi-turn chat, and live VRAM /
                       event telemetry. Generation runs in a worker thread with
                       cooperative cancellation so Stop frees the GPU and the
                       event loop never blocks.
web/
  index.html, app.js, style.css   Single-page dashboard (no framework).

src/  (reference implementations — standalone studies of each technique,
       runnable directly with `python <file>`; not imported by the server)
  quantization.py      From-scratch symmetric INT8 and affine n-bit group
                       quantization / dequantization (the AWQ/GPTQ scheme),
                       measured on a real weight matrix.
  paged_attention.py   KV-cache byte math + a PagedAttention block-allocator model
                       (used-vs-wasted memory, concurrency capacity).
  zero_copy.py         Pinned vs. pageable host↔device transfer timing.
  batching.py          Batched-decode throughput + concurrency-capacity model.
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
VRAM. So the KV cache shows only a modest (~1.2×) speedup, and **4-bit
quantization buys ~2.2× less VRAM but no speed** (bitsandbytes adds dequant
overhead) — both visible directly in the *optimized vs. baseline* A/B. The same
techniques scale dramatically with model size and concurrency. The whole point of
the bench is to show *where* each optimization does and doesn't pay off, with the
measured numbers to back it — which is a more useful answer than assuming every
"optimization" is always a win.

## Hardware

Developed and tested on an NVIDIA RTX 5060 Laptop GPU (8 GB, Blackwell sm_120),
Windows 11, Python 3.12, PyTorch cu128.
