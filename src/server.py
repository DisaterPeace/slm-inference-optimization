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


def build_prompt(tokenizer, user_msg: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_msg})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _select_token(logits, context_ids, params, generator):
    """Pick the next token id from raw logits.

    Greedy (argmax) when temperature <= 0. Otherwise applies, in order:
    repetition penalty over the tokens seen so far, temperature scaling,
    top-k filtering, and top-p (nucleus) filtering, then samples. Returns an int.
    """
    logits = logits.clone()
    penalty = params.get("repetition_penalty", 1.0) or 1.0
    if penalty != 1.0 and context_ids:
        ids = torch.tensor(sorted(set(context_ids)), device=logits.device)
        s = logits[0, ids]
        logits[0, ids] = torch.where(s > 0, s / penalty, s * penalty)

    temp = params.get("temperature", 0.0) or 0.0
    if temp <= 0:
        return int(torch.argmax(logits, dim=-1))

    logits = logits / temp
    top_k = params.get("top_k", 0) or 0
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    top_p = params.get("top_p", 1.0) or 1.0
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()   # keep the first token over the threshold
        remove[..., 0] = False
        logits = logits.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


@torch.no_grad()
def stream_generate(model, tokenizer, prompt: str, max_new_tokens: int,
                    use_cache: bool, pinned: bool = False, params: dict | None = None):
    """Yield (delta_text, dt_ms, index). index 0's dt is the prefill = TTFT.

    `params` carries the real sampling knobs (temperature, top_p, top_k,
    repetition_penalty, seed). With temperature <= 0 decoding is greedy/deterministic;
    otherwise tokens are sampled, optionally reproducibly via the seed.

    `pinned` toggles a REAL zero-copy transfer path for the per-step token input:
    a page-locked (pinned) host buffer reused across steps + non_blocking H2D copy.
    """
    params = params or {}
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = input_ids.shape[1]
    eos = tokenizer.eos_token_id

    # seeded generator only matters when sampling (temperature > 0)
    generator = None
    if (params.get("temperature", 0) or 0) > 0:
        generator = torch.Generator(device=DEVICE)
        if params.get("seed") is not None:
            generator.manual_seed(int(params["seed"]))
    context_ids = input_ids[0].tolist()  # prompt + generated, for repetition penalty

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
    next_id = _select_token(logits, context_ids, params, generator)
    context_ids.append(next_id)
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
        next_id = _select_token(logits, context_ids, params, generator)
        context_ids.append(next_id)
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
        system = (req.get("system") or "").strip()
        seed = req.get("seed")
        params = {
            "temperature": float(req.get("temperature", 0.0) or 0.0),
            "top_p": float(req.get("top_p", 1.0) or 1.0),
            "top_k": int(req.get("top_k", 0) or 0),
            "repetition_penalty": float(req.get("repetition_penalty", 1.0) or 1.0),
            "seed": int(seed) if seed not in (None, "", "null") else None,
        }

        spec = _model_spec(model_key)
        engine = "KV-cache (O(N))" if use_cache else "naive recompute (O(N²))"
        decode = "greedy" if params["temperature"] <= 0 else (
            f"sample T={params['temperature']:g} top_p={params['top_p']:g} top_k={params['top_k']}")
        await log(f"pipeline start · {spec['short']} · {precision.upper()} · {engine} · {decode} · {max_new} tok")
        await ws.send_json({"type": "meta", "vram_total_mb": vram_total_mb()})

        cached = (model_key, precision) in _models
        if not cached:
            await log(f"loading {spec['id']} [{precision}] → VRAM (first use)…")
        tokenizer = get_tokenizer(model_key)
        # multi-turn chat: if the client sends a full message history, template the
        # whole conversation so the model has context; else wrap a single prompt.
        # A system prompt (if any) is prepended either way.
        messages = req.get("messages")
        if messages:
            full = ([{"role": "system", "content": system}] if system else []) + messages
            prompt = tokenizer.apply_chat_template(
                full, tokenize=False, add_generation_prompt=True
            )
            await log(f"context · {len(full)} messages in conversation")
        else:
            prompt = build_prompt(tokenizer, req.get("prompt", "Hello!"), system)
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
                for event in stream_generate(model, tokenizer, prompt, max_new, use_cache, pinned, params):
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



# static frontend (mounted last so it doesn't shadow /api routes)
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/", StaticFiles(directory=WEB_DIR), name="static")
