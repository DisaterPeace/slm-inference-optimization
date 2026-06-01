r"""Phase 7 - FastAPI backend for the interactive inference playground.

Serves a single-page frontend and exposes:
  WS  /ws/generate     stream tokens with per-token timing; toggles for
                       KV cache on/off and FP16 vs 4-bit (NF4) quantization.
  GET /api/model_info  VRAM footprint of the loaded model(s).
  GET /api/paged       PagedAttention memory math + block allocation for a batch.
  GET /api/quant       From-scratch INT8/INT4 quantization error on a real weight.
  GET /api/zerocopy    Pinned-vs-pageable and per-token-sync micro-benchmarks.

Run:  ..\.venv\Scripts\python.exe -m uvicorn server:app --port 8000
(from the src/ directory)
"""

from __future__ import annotations

import asyncio
import json
import gc
import os
import threading
import time

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from quantization import (
    quantize_int4_group, dequantize_int4_group,
    quantize_int8_symmetric, dequantize_int8_symmetric,
    quantize_intn_group, dequantize_intn_group,
)
from paged_attention import KVConfig, BLOCK_SIZE
from zero_copy import cuda_time
from batching import measure_batched_throughput, concurrency_capacity

DEFAULT_MODEL_KEY = "qwen25_05b"
MODEL_SPECS = {
    "qwen25_05b": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B",
        "short": "Qwen 0.5B",
        "params": "0.5B",
        "trust_remote_code": False,
    },
    "phi4_mini": {
        "id": "microsoft/Phi-4-mini-instruct",
        "label": "Phi-4 Mini 3.8B",
        "short": "Phi-4 Mini",
        "params": "3.8B",
        # Phi-4-mini is a `phi3`-architecture model that transformers 5.x supports
        # natively. Using the hub's bundled modeling code (trust_remote_code=True)
        # breaks here because it imports the old `LossKwargs` symbol, which 5.9
        # renamed to `TransformersKwargs`. The native impl avoids that entirely.
        "trust_remote_code": False,
    },
}
WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")

app = FastAPI()

# ---------------------------------------------------------------------------
# Model loading: FP16 eagerly, 4-bit lazily (and cached). Both fit in 8 GB.
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizers: dict[str, object] = {}
_models: dict[tuple[str, str], object] = {}
_footprint_mb: dict[tuple[str, str], float] = {}
_active_model_key: str | None = None
_batching_cache: dict[tuple[str, str], list[dict]] = {}


def _model_spec(model_key: str) -> dict:
    if model_key not in MODEL_SPECS:
        raise HTTPException(status_code=400, detail=f"unknown model: {model_key}")
    return MODEL_SPECS[model_key]


def get_tokenizer(model_key: str):
    spec = _model_spec(model_key)
    if model_key not in _tokenizers:
        _tokenizers[model_key] = AutoTokenizer.from_pretrained(
            spec["id"], trust_remote_code=spec["trust_remote_code"]
        )
    return _tokenizers[model_key]


def unload_models_except(model_key: str, precision: str) -> None:
    """Keep GPU memory focused on the selected model/precision."""
    global _active_model_key
    keep = (model_key, precision)
    for key in list(_models):
        if key != keep:
            del _models[key]
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _active_model_key = model_key


def get_model(model_key: str = DEFAULT_MODEL_KEY, precision: str = "fp16"):
    """precision is 'fp16', 'int8' (LLM.int8), or 'nf4' (4-bit). Cached on first use."""
    spec = _model_spec(model_key)
    if precision not in {"fp16", "int8", "nf4"}:
        raise HTTPException(status_code=400, detail=f"unknown precision: {precision}")
    cache_key = (model_key, precision)
    unload_models_except(model_key, precision)
    if cache_key in _models:
        return _models[cache_key]

    if precision in {"nf4", "int8"}:
        if DEVICE != "cuda":
            raise HTTPException(status_code=400, detail=f"{precision} loading needs CUDA")
        if precision == "nf4":
            cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
        else:  # LLM.int8() weight-only 8-bit
            cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec["id"],
            quantization_config=cfg,
            device_map=DEVICE,
            trust_remote_code=spec["trust_remote_code"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            spec["id"],
            dtype=torch.float16,
            trust_remote_code=spec["trust_remote_code"],
        )
        model.to(DEVICE)
    model.eval()
    # Warm up: the first CUDA generation compiles kernels (~0.5s). Burn that off
    # now so the UI shows honest TTFT from the very first user request.
    tokenizer = get_tokenizer(model_key)
    with torch.no_grad():
        warm = tokenizer("warmup", return_tensors="pt").input_ids.to(DEVICE)
        out = model(warm, use_cache=True)
        model(torch.tensor([[0]], device=DEVICE), past_key_values=out.past_key_values, use_cache=True)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    _models[cache_key] = model
    _footprint_mb[cache_key] = model.get_memory_footprint() / 1024**2
    return model


# load default FP16 at startup so the first request is fast
get_model(DEFAULT_MODEL_KEY, "fp16")


def vram_allocated_mb() -> float:
    """Live VRAM the process holds right now (real torch allocator reading)."""
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def vram_total_mb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.get_device_properties(0).total_memory / 1024**2
    return 0.0


def build_prompt(tokenizer, user_msg: str) -> str:
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def stream_generate(model, tokenizer, prompt: str, max_new_tokens: int,
                    use_cache: bool, pinned: bool = False):
    """Yield (delta_text, dt_ms, index). index 0's dt is the prefill = TTFT.

    `pinned` toggles a REAL zero-copy transfer path for the per-step token input:
    a page-locked (pinned) host buffer reused across steps + non_blocking H2D copy,
    versus an ordinary pageable allocation each step. At single-token decode this
    moves ~8 bytes/step, so the effect is in the noise — pinned memory matters for
    large/batched transfers (see the zero-copy benchmark). The path is genuine, though.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = input_ids.shape[1]
    eos = tokenizer.eos_token_id

    def sync():
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    # pre-allocate one pinned host buffer and reuse it (the correct zero-copy pattern;
    # pinning per-step would add allocation overhead that defeats the purpose)
    use_pinned = pinned and DEVICE == "cuda"
    pin_buf = torch.empty((1, 1), dtype=torch.long).pin_memory() if use_pinned else None

    def step_input(token_id: int):
        if use_pinned:
            pin_buf[0, 0] = token_id
            return pin_buf.to(DEVICE, non_blocking=True)
        return torch.tensor([[token_id]], device=DEVICE)

    generated: list[int] = []
    prev_text = ""
    past = None
    seq = input_ids

    # --- prefill (first forward over the whole prompt) ---
    t0 = time.perf_counter()
    out = model(input_ids, use_cache=use_cache)
    past = out.past_key_values if use_cache else None
    logits = out.logits[:, -1, :]
    next_id = int(torch.argmax(logits, dim=-1))
    sync()
    yield _emit(tokenizer, generated, next_id, prev_text, (time.perf_counter() - t0) * 1000, 0)
    prev_text = tokenizer.decode(generated, skip_special_tokens=True)

    # --- decode loop ---
    for i in range(1, max_new_tokens):
        if next_id == eos:
            break
        t0 = time.perf_counter()
        if use_cache:
            inp = step_input(next_id)
            out = model(inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:
            seq = torch.cat([seq, step_input(next_id)], dim=1)
            out = model(seq, use_cache=False)
        logits = out.logits[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1))
        sync()
        event = _emit(tokenizer, generated, next_id, prev_text, (time.perf_counter() - t0) * 1000, i)
        prev_text = tokenizer.decode(generated, skip_special_tokens=True)
        yield event

    yield {
        "type": "done", "prompt_tokens": prompt_len, "generated": len(generated),
        "vram_mb": vram_allocated_mb(),
    }


def _emit(tokenizer, generated, next_id, prev_text, dt_ms, index):
    generated.append(next_id)
    full = tokenizer.decode(generated, skip_special_tokens=True)
    return {"type": "token", "text": full[len(prev_text):], "dt_ms": dt_ms, "index": index}


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()

    async def log(msg: str):
        await ws.send_json({"type": "log", "msg": msg, "vram_mb": vram_allocated_mb()})

    loop = asyncio.get_running_loop()
    stop_event = threading.Event()  # set when the client disconnects / hits Stop

    try:
        req = await ws.receive_json()
        model_key = req.get("model", DEFAULT_MODEL_KEY)
        max_new = int(req.get("max_new_tokens", 96))
        use_cache = bool(req.get("use_cache", True))
        precision = req.get("precision", "fp16")
        pinned = bool(req.get("pinned", False))

        spec = _model_spec(model_key)
        engine = "KV-cache (O(N))" if use_cache else "naive recompute (O(N²))"
        mem = "pinned/zero-copy H2D" if pinned else "pageable H2D"
        await log(f"pipeline start · {spec['short']} · {precision.upper()} · {engine} · {mem} · {max_new} tok")
        await ws.send_json({"type": "meta", "vram_total_mb": vram_total_mb()})

        cached = (model_key, precision) in _models
        if not cached:
            await log(f"loading {spec['id']} [{precision}] → VRAM (first use)…")
        tokenizer = get_tokenizer(model_key)
        # multi-turn chat: if the client sends a full message history, template the
        # whole conversation so the model has context; else wrap a single prompt.
        messages = req.get("messages")
        if messages:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            await log(f"context · {len(messages)} messages in conversation")
        else:
            prompt = build_prompt(tokenizer, req.get("prompt", "Hello!"))
        model = get_model(model_key, precision)
        footprint = _footprint_mb.get((model_key, precision), 0.0)
        await log(f"model ready · weights {footprint:.0f} MB · VRAM allocated {vram_allocated_mb():.0f} MB")

        # --- run the blocking generation in a worker THREAD ---------------------
        # Generation is GPU/CPU-bound; running it directly in the async handler
        # would block the event loop (so Stop couldn't be detected and other
        # requests would queue behind it). The thread streams events back through
        # a queue; a watcher task flips stop_event the instant the client leaves,
        # and the generator checks it between tokens — so Stop frees the GPU fast.
        queue: asyncio.Queue = asyncio.Queue()
        END = {"type": "__end__"}

        def worker():
            try:
                for event in stream_generate(model, tokenizer, prompt, max_new, use_cache, pinned):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
                    if stop_event.is_set():
                        break
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, END)

        async def watch_disconnect():
            try:
                while True:
                    await ws.receive()  # raises WebSocketDisconnect when client closes
            except Exception:
                stop_event.set()

        watcher = asyncio.create_task(watch_disconnect())
        threading.Thread(target=worker, daemon=True).start()

        try:
            while True:
                event = await queue.get()
                if event is END:
                    break
                if event.get("type") == "token" and event.get("index") == 0:
                    await log(f"prefill done · TTFT {event['dt_ms']:.0f} ms · decoding…")
                await ws.send_json(event)
        finally:
            watcher.cancel()

        if not stop_event.is_set():
            await log("pipeline complete")  # client's gone if stopped — don't send
    except WebSocketDisconnect:
        stop_event.set()  # let the worker thread wind down on its next token
    except Exception as e:  # surface errors to the UI instead of silently dying
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass  # socket already gone (client stopped) — swallow


@app.get("/api/model_info")
def model_info():
    def footprint(model_key: str, precision: str) -> float | None:
        return _footprint_mb.get((model_key, precision))

    active = _active_model_key or DEFAULT_MODEL_KEY
    return {
        "active_model": active,
        "models": [
            {
                "key": key,
                "id": spec["id"],
                "label": spec["label"],
                "short": spec["short"],
                "params": spec["params"],
                "fp16_mb": footprint(key, "fp16"),
                "int8_mb": footprint(key, "int8"),
                "nf4_mb": footprint(key, "nf4"),
            }
            for key, spec in MODEL_SPECS.items()
        ],
        "fp16_mb": footprint(active, "fp16"),
        "int8_mb": footprint(active, "int8"),
        "nf4_mb": footprint(active, "nf4"),
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
    }


@app.get("/api/paged")
def paged(
    lengths: str = "40,55,60,80,120,150,200,300,512,900",
    max_len: int = 2048,
    model_key: str = DEFAULT_MODEL_KEY,
    block_size: int = BLOCK_SIZE,
):
    spec = _model_spec(model_key)
    kv = KVConfig(spec["id"], trust_remote_code=spec["trust_remote_code"])
    actual = [int(x) for x in lengths.split(",") if x.strip()]
    bytes_per_block = kv.bytes_per_token * block_size
    naive_blocks_each = (max_len + block_size - 1) // block_size
    paged_each = [(L + block_size - 1) // block_size for L in actual]
    naive_mb = len(actual) * naive_blocks_each * bytes_per_block / 1024**2
    paged_mb = sum(paged_each) * bytes_per_block / 1024**2
    useful_mb = sum(actual) * kv.bytes_per_token / 1024**2
    return {
        "lengths": actual,
        "block_size": block_size,
        "bytes_per_token": kv.bytes_per_token,
        "blocks_per_seq": paged_each,
        "naive_blocks_per_seq": naive_blocks_each,
        "naive_mb": naive_mb, "paged_mb": paged_mb, "useful_mb": useful_mb,
        "reduction": naive_mb / paged_mb if paged_mb else 0,
    }


@app.get("/api/quant")
def quant(group_size: int = 128, model_key: str = DEFAULT_MODEL_KEY):
    model = get_model(model_key, "fp16")
    weights = [(n, p) for n, p in model.named_parameters() if p.ndim == 2]
    preferred = [
        (n, p)
        for n, p in weights
        if any(part in n for part in ("q_proj.weight", "qkv_proj.weight", "query_key_value.weight"))
    ]
    if not weights:
        raise HTTPException(status_code=500, detail="no 2D model weights found")
    name, w = (preferred or weights)[0]
    w = w.detach().float().cpu()

    q8, s8 = quantize_int8_symmetric(w)
    w8 = dequantize_int8_symmetric(q8, s8)
    q4, s4, z4 = quantize_int4_group(w, group_size=group_size)
    w4 = dequantize_int4_group(q4, s4, z4, w.shape)
    # INT2 modeled with the same affine group scheme (≈ GGUF Q2-style storage/error)
    q2, s2, z2 = quantize_intn_group(w, nbits=2, group_size=group_size)
    w2 = dequantize_intn_group(q2, s2, z2, w.shape)

    def rel_err(a, b):
        return ((a - b).abs().sum() / a.abs().sum()).item() * 100

    return {
        "weight": name, "shape": list(w.shape), "group_size": group_size,
        "int8_rel_err": rel_err(w, w8),
        "int4_rel_err": rel_err(w, w4),
        "int2_rel_err": rel_err(w, w2),
        "fp16_kb": w.numel() * 2 / 1024,
        "int8_kb": (q8.numel() + s8.numel() * 4) / 1024,
        "int4_kb": (q4.numel() * 0.5 + s4.numel() * 2 + z4.numel() * 2) / 1024,
        "int2_kb": (q2.numel() * 0.25 + s2.numel() * 2 + z2.numel() * 2) / 1024,
    }


@app.get("/api/batching")
def batching(
    avg_len: int = 180,
    max_len: int = 2048,
    budget_gb: float = 4.0,
    model_key: str = DEFAULT_MODEL_KEY,
    precision: str = "fp16",
):
    """Batched-decode throughput (measured once, cached) + concurrency capacity."""
    cache_key = (model_key, precision)
    if cache_key not in _batching_cache:
        tokenizer = get_tokenizer(model_key)
        model = get_model(model_key, precision)
        _batching_cache[cache_key] = measure_batched_throughput(model, tokenizer, DEVICE)
    spec = _model_spec(model_key)
    kv = KVConfig(spec["id"], trust_remote_code=spec["trust_remote_code"])
    cap = concurrency_capacity(kv, budget_gb=budget_gb, avg_len=avg_len, max_len=max_len)
    throughput = _batching_cache[cache_key]
    base = throughput[0]["tokens_per_sec"]
    return {
        "throughput": throughput,
        "batch1_tps": base,
        "best_tps": throughput[-1]["tokens_per_sec"],
        "batch_speedup": throughput[-1]["tokens_per_sec"] / base if base else 0,
        "capacity": cap,
    }


@app.get("/api/zerocopy")
def zerocopy():
    if DEVICE != "cuda":
        return {"error": "needs a GPU"}
    n = 64 * 1024 * 1024 // 4  # 64 MB
    pageable = torch.empty(n, dtype=torch.float32)
    pinned = torch.empty(n, dtype=torch.float32).pin_memory()
    # warm up both paths before timing so we measure steady-state, not first-run
    _ = pageable.to("cuda"); _ = pinned.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    # 20 iters gives a stable median; PCIe bandwidth fluctuates ±5% thermally
    t_pageable = cuda_time(lambda: pageable.to("cuda", non_blocking=False), iters=20)
    t_pinned = cuda_time(lambda: pinned.to("cuda", non_blocking=True), iters=20)
    gb = n * 4 / 1024**3
    return {
        "transfer_mb": n * 4 / 1024**2,
        "pageable_gbps": gb / (t_pageable / 1000),
        "pinned_gbps": gb / (t_pinned / 1000),
        "speedup": t_pageable / t_pinned,
    }


# static frontend (mounted last so it doesn't shadow /api routes)
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/", StaticFiles(directory=WEB_DIR), name="static")
