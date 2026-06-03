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

from paged_attention import KVConfig  # per-token KV-cache byte math

DEFAULT_MODEL_KEY = "qwen25_05b"
MODEL_SPECS = {
    "qwen25_05b": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B",
        "short": "Qwen 0.5B",
        "params": "0.5B",
        "trust_remote_code": False,
    },
    "qwen25_15b": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5 1.5B",
        "short": "Qwen 1.5B",
        "params": "1.5B",
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


_kv_bpt: dict[str, int] = {}


def kv_bytes_per_token(model_key: str) -> int:
    """Per-token KV-cache size (bytes) for a model, from its real config. Cached."""
    if model_key not in _kv_bpt:
        try:
            spec = _model_spec(model_key)
            _kv_bpt[model_key] = KVConfig(spec["id"], spec["trust_remote_code"]).bytes_per_token
        except Exception:
            _kv_bpt[model_key] = 0
    return _kv_bpt[model_key]


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
    prob = float(torch.softmax(logits[0].float(), dim=-1)[next_id])
    context_ids.append(next_id)
    sync()
    yield _emit(tokenizer, generated, next_id, prev_text, (time.perf_counter() - t0) * 1000, 0, prob)
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
        prob = float(torch.softmax(logits[0].float(), dim=-1)[next_id])
        context_ids.append(next_id)
        sync()
        event = _emit(tokenizer, generated, next_id, prev_text, (time.perf_counter() - t0) * 1000, i, prob)
        prev_text = tokenizer.decode(generated, skip_special_tokens=True)
        yield event

    yield {
        "type": "done", "prompt_tokens": prompt_len, "generated": len(generated),
        "vram_mb": vram_allocated_mb(),
    }


def _emit(tokenizer, generated, next_id, prev_text, dt_ms, index, prob=None):
    generated.append(next_id)
    full = tokenizer.decode(generated, skip_special_tokens=True)
    ev = {"type": "token", "text": full[len(prev_text):], "dt_ms": dt_ms, "index": index}
    if prob is not None:
        ev["p"] = prob  # model's probability for the chosen token (confidence)
    return ev


# ---------------------------------------------------------------------------
# Second engine: llama.cpp (GGUF) — the edge / on-device path. Runs on CPU and
# stacks GGUF quantization + KV cache + mmap'd (zero-copy) weight loading. Built
# from source for this CPU's AVX2 (the prebuilt wheel assumed AVX-512 and crashed).
# ---------------------------------------------------------------------------
# Each model: repo + an ordered quant ladder (key -> (label, filename glob)).
# GGUF goes all the way down to 2-bit (Q2_K) — below what bitsandbytes/HF can do.
LLAMA_MODELS = {
    "qwen25_05b": {
        "label": "Qwen2.5 0.5B",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "quants": {
            "fp16": ("F16 · 16-bit (baseline)", "*fp16.gguf"),
            "q8_0": ("Q8_0 · 8-bit", "*q8_0.gguf"),
            "q6_k": ("Q6_K · 6-bit", "*q6_k.gguf"),
            "q5_k_m": ("Q5_K_M · 5-bit", "*q5_k_m.gguf"),
            "q4_k_m": ("Q4_K_M · 4-bit", "*q4_k_m.gguf"),
            "q3_k_m": ("Q3_K_M · 3-bit", "*q3_k_m.gguf"),
            "q2_k": ("Q2_K · 2-bit", "*q2_k.gguf"),
        },
    },
    "phi4_mini": {
        "label": "Phi-4 Mini 3.8B",
        "repo": "unsloth/Phi-4-mini-instruct-GGUF",
        "quants": {
            "bf16": ("BF16 · 16-bit (baseline ~7.6 GB)", "*BF16.gguf"),
            "q8_0": ("Q8_0 · 8-bit", "*Q8_0.gguf"),
            "q6_k": ("Q6_K · 6-bit", "*Q6_K.gguf"),
            "q5_k_m": ("Q5_K_M · 5-bit", "*Q5_K_M.gguf"),
            "q4_k_m": ("Q4_K_M · 4-bit", "*Q4_K_M.gguf"),
            "q3_k_m": ("Q3_K_M · 3-bit", "*Q3_K_M.gguf"),
            "q2_k": ("Q2_K · 2-bit", "*Q2_K.gguf"),
        },
    },
}
_llama_models: dict[tuple, object] = {}


def get_llama(model_key: str, quant_key: str, use_mmap: bool = True,
              logits_all: bool = False, n_ctx: int = 4096):
    """Load a GGUF model via llama.cpp (cached). use_mmap toggles zero-copy load
    (mmap the file) vs. reading the whole file into a buffer (a real copy) — the
    'copy elimination' lever. logits_all=True (with a small n_ctx) is used for the
    perplexity sweep. Only one GGUF model is kept resident at a time."""
    m = LLAMA_MODELS.get(model_key) or LLAMA_MODELS["qwen25_05b"]
    if quant_key not in m["quants"]:
        quant_key = "q4_k_m" if "q4_k_m" in m["quants"] else next(iter(m["quants"]))
    cache_key = (model_key, quant_key, use_mmap, logits_all, n_ctx)
    if cache_key not in _llama_models:
        from llama_cpp import Llama  # lazy import; CPU build
        # GGUF models are large; keep just the newest one in RAM
        for k in list(_llama_models):
            del _llama_models[k]
        gc.collect()
        _, glob = m["quants"][quant_key]
        _llama_models[cache_key] = Llama.from_pretrained(
            repo_id=m["repo"], filename=glob, n_ctx=n_ctx, verbose=False,
            use_mmap=use_mmap, logits_all=logits_all,
        )
    return _llama_models[cache_key]


def stream_generate_llama(llm, messages, max_new_tokens):
    """Stream tokens from a llama.cpp GGUF model with per-token timing.

    llama.cpp always uses its own KV cache and mmap'd weights; decoding is greedy
    here so runs are comparable to the HuggingFace engine. index 0's dt is the
    prompt-eval time (TTFT); later dt's are inter-token gaps.
    """
    t0 = time.perf_counter()
    n = 0
    for ch in llm.create_chat_completion(
        messages=messages, max_tokens=max_new_tokens, temperature=0.0, stream=True
    ):
        delta = ch["choices"][0]["delta"].get("content", "")
        if not delta:
            continue
        dt = (time.perf_counter() - t0) * 1000
        yield {"type": "token", "text": delta, "dt_ms": dt, "index": n}
        n += 1
        t0 = time.perf_counter()
    yield {"type": "done", "prompt_tokens": None, "generated": n, "vram_mb": vram_allocated_mb()}


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

        req_engine = req.get("engine", "hf")
        await ws.send_json({
            "type": "meta", "vram_total_mb": vram_total_mb(),
            "kv_bytes_per_token": kv_bytes_per_token(model_key),
        })

        if req_engine == "llamacpp":
            # --- llama.cpp (GGUF, CPU) — the edge / on-device engine -----------
            lm = LLAMA_MODELS.get(model_key) or LLAMA_MODELS["qwen25_05b"]
            qkey = precision if precision in lm["quants"] else (
                "q4_k_m" if "q4_k_m" in lm["quants"] else next(iter(lm["quants"])))
            qlabel = lm["quants"][qkey][0]
            use_mmap = bool(req.get("mmap", True))
            mmap_txt = "mmap (zero-copy)" if use_mmap else "full read (copy)"
            await log(f"pipeline start · llama.cpp (CPU) · {lm['label']} · {qlabel} · {mmap_txt} · greedy · {max_new} tok")
            msgs = req.get("messages") or [{"role": "user", "content": req.get("prompt", "Hello!")}]
            full_msgs = ([{"role": "system", "content": system}] if system else []) + msgs
            if (model_key, qkey, use_mmap, False, 4096) not in _llama_models:
                await log(f"loading {lm['repo']} [{qkey}] via {mmap_txt} (first use; may download)…")
            t_load = time.perf_counter()
            llm = await asyncio.to_thread(get_llama, model_key, qkey, use_mmap)
            load_ms = (time.perf_counter() - t_load) * 1000
            try:
                sz = os.path.getsize(llm.model_path) / 1024 ** 2
            except Exception:
                sz = 0.0
            await log(f"model ready · {mmap_txt} · loaded in {load_ms:.0f} ms · GGUF {sz:.0f} MB · CPU "
                      f"(GGUF quant + KV cache; mmap = copy elimination)")

            def make_gen():
                return stream_generate_llama(llm, full_msgs, max_new)
        else:
            # --- HuggingFace / transformers (GPU) ------------------------------
            spec = _model_spec(model_key)
            is_spec_target = model_key in SPEC_DRAFT
            use_spec = bool(req.get("speculative", False)) and is_spec_target
            cache_eng = "KV-cache (O(N))" if use_cache else "naive recompute (O(N²))"
            decode_lbl = "speculative (0.5B draft)" if use_spec else (
                "target alone" if is_spec_target else cache_eng)
            await log(f"pipeline start · {spec['short']} · {precision.upper()} · {decode_lbl} · greedy · {max_new} tok")
            if (model_key, precision) not in _models:
                await log(f"loading {spec['id']} [{precision}] → VRAM (first use)…")
            tokenizer = get_tokenizer(model_key)
            # multi-turn chat: full message history (with optional system prompt)
            messages = req.get("messages")
            if messages:
                full = ([{"role": "system", "content": system}] if system else []) + messages
                prompt = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=True)
                await log(f"context · {len(full)} messages in conversation")
            else:
                prompt = build_prompt(tokenizer, req.get("prompt", "Hello!"), system)

            if use_spec:
                # speculative ON: the 0.5B drafts, the 1.5B target verifies — streamed
                # per-token (with each token's source) so the panel below shows accept/
                # reject for THIS run. Off → falls through to the normal decode path.
                target = get_model(model_key, "fp16")
                draft = get_draft()
                await log(f"model ready · {spec['short']} target + 0.5B draft (speculative) · "
                          f"VRAM {vram_allocated_mb():.0f} MB")

                def make_gen():
                    return stream_speculative(target, draft, tokenizer, prompt, max_new)
            else:
                model = get_model(model_key, precision)
                footprint = _footprint_mb.get((model_key, precision), 0.0)
                await log(f"model ready · weights {footprint:.0f} MB · VRAM allocated {vram_allocated_mb():.0f} MB")

                def make_gen():
                    return stream_generate(model, tokenizer, prompt, max_new, use_cache, pinned, params)

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
                for event in make_gen():
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
        "spec_targets": list(SPEC_DRAFT.keys()),
        "llama_models": [
            {"key": mk, "label": m["label"],
             "quants": [{"key": qk, "label": lbl} for qk, (lbl, _) in m["quants"].items()]}
            for mk, m in LLAMA_MODELS.items()
        ],
    }


_QUANT_SWEEP_REF = (
    "The Earth orbits the Sun once each year, while the Moon orbits the Earth about once a "
    "month. Light from the Sun takes roughly eight minutes to reach us. A central processing "
    "unit executes instructions one step at a time, reading data and weights from memory."
)


def _perplexity(llm, text: str) -> float:
    """Perplexity of `text` under the model: exp(mean negative log-likelihood per token).
    Lower = the model finds the text more predictable = closer to full-precision behaviour."""
    import numpy as np
    toks = llm.tokenize(text.encode("utf-8"))
    llm.reset()
    llm.eval(toks)
    scores = llm.scores  # (n_ctx, vocab); row i predicts token i+1
    lps = []
    for i in range(len(toks) - 1):
        lg = np.asarray(scores[i], dtype=np.float64)
        mx = lg.max()
        lse = mx + np.log(np.exp(lg - mx).sum())
        lps.append(lg[toks[i + 1]] - lse)
    return float(np.exp(-sum(lps) / len(lps))) if lps else 0.0


def _quant_sweep_hf(model_key: str):
    """HuggingFace/GPU quant tradeoff: FP16 / INT8 / 4-bit NF4 (the 3 levels bitsandbytes
    offers). VRAM footprint (MB), perplexity from the logits, and decode tok/s. A coarse
    3-point curve — and near-flat on a small model — vs. GGUF's fine ladder."""
    tokenizer = get_tokenizer(model_key)
    ids = tokenizer(_QUANT_SWEEP_REF, return_tensors="pt").input_ids.to(DEVICE)
    labels = {"fp16": "FP16 · 16-bit (baseline)", "int8": "INT8 · 8-bit", "nf4": "NF4 · 4-bit"}
    points = []
    for prec in ("fp16", "int8", "nf4"):
        try:
            model = get_model(model_key, prec)
        except Exception:
            continue  # e.g. FP16 of a 3.8B model OOMs on 8 GB — just skip that point
        size_mb = _footprint_mb.get((model_key, prec), 0.0)
        with torch.no_grad():
            logits = model(ids).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1)
            tok_lp = lp[torch.arange(ids.shape[1] - 1), ids[0, 1:]]
            ppl = float(torch.exp(-tok_lp.mean()))
            t = time.perf_counter()
            out = model.generate(ids, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            dt = time.perf_counter() - t
        ntok = out.shape[1] - ids.shape[1]
        points.append({
            "quant": prec, "label": labels[prec],
            "size_mb": round(size_mb, 1), "ppl": round(ppl, 2),
            "tps": round(ntok / dt, 1) if dt else 0.0,
        })
    return {"model": model_key, "model_label": _model_spec(model_key)["label"],
            "engine": "hf", "points": points}


@app.get("/api/quant_sweep")
def quant_sweep(model_key: str = "qwen25_05b", engine: str = "llamacpp", include_fp: bool = False):
    """Sweep a model's quant ladder, measuring the size/quality/speed tradeoff.

    engine='hf'      → FP16/INT8/4-bit on the GPU (VRAM footprint + PPL + tok/s); 3 points.
    engine='llamacpp'→ the full GGUF ladder on the CPU (file size + PPL + tok/s); the rich curve.

    Perplexity is on a fixed reference text (lower = better), computed from the logits.
    SLOW: downloads/loads each level. FastAPI runs this sync endpoint in a threadpool.
    """
    if engine == "hf":
        return _quant_sweep_hf(model_key)
    m = LLAMA_MODELS.get(model_key) or LLAMA_MODELS["qwen25_05b"]
    points = []
    for qkey, (label, _glob) in m["quants"].items():
        if not include_fp and qkey in ("fp16", "bf16"):
            continue
        llm = get_llama(model_key, qkey, True, logits_all=True, n_ctx=512)
        try:
            size_mb = os.path.getsize(llm.model_path) / 1024 ** 2
        except Exception:
            size_mb = 0.0
        ppl = _perplexity(llm, _QUANT_SWEEP_REF)
        t = time.perf_counter()
        gen = llm.create_completion("Explain inference in one sentence.", max_tokens=24, temperature=0.0)
        dt = time.perf_counter() - t
        ntok = gen.get("usage", {}).get("completion_tokens") or 24
        points.append({
            "quant": qkey, "label": label,
            "size_mb": round(size_mb, 1), "ppl": round(ppl, 2),
            "tps": round(ntok / dt, 1) if dt else 0.0,
        })
    return {"model": model_key, "model_label": m["label"], "points": points}


# Speculative decoding: which chat models can act as a "target" and the draft to use.
# Target + draft must share a vocabulary, so only same-family Qwen pairs qualify.
SPEC_DRAFT = {"qwen25_15b": "qwen25_05b"}
_draft_model = None


def get_draft():
    """The 0.5B draft, kept resident separately so loading the target (which evicts
    the regular model cache) doesn't drop it. Both fit easily in 8 GB."""
    global _draft_model
    if _draft_model is None:
        did = MODEL_SPECS["qwen25_05b"]["id"]
        _draft_model = AutoModelForCausalLM.from_pretrained(did, dtype=torch.float16).to(DEVICE).eval()
    return _draft_model


@torch.no_grad()
def stream_speculative(target, draft, tokenizer, prompt, max_new_tokens, K=4):
    """Streaming greedy speculative decoding — the live Speculative toggle.

    The 0.5B *draft* proposes K tokens; the 1.5B *target* verifies them all in ONE
    forward pass and keeps the matching prefix (plus its own next token). Lossless vs.
    greedy target. Each token is emitted with its `source` — 'draft' (the draft guessed
    right, free) or 'target' (the target had to correct it) — so the bottom panel can
    paint accept/reject for *this* run. The done event carries the acceptance rate and
    tokens-per-target-pass (the real speculative win)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = ids.shape[1]
    eos = tokenizer.eos_token_id
    seq, generated, prev_text, n = ids, [], "", 0
    passes, drafted, accepted = 0, 0, 0
    while n < max_new_tokens:
        t0 = time.perf_counter()
        dseq, drafts = seq, []
        for _ in range(K):                                  # draft proposes K tokens
            dt = int(draft(dseq).logits[:, -1, :].argmax(-1))
            drafts.append(dt)
            dseq = torch.cat([dseq, torch.tensor([[dt]], device=DEVICE)], dim=1)
        drafted += K
        tl = target(torch.cat([seq, torch.tensor([drafts], device=DEVICE)], dim=1)).logits
        passes += 1                                         # ONE target pass verifies all K
        L, emit, nacc = seq.shape[1], [], 0
        for i in range(K):
            tgt = int(tl[:, L - 1 + i, :].argmax(-1))
            if tgt == drafts[i]:
                emit.append((drafts[i], "draft")); nacc += 1
            else:
                emit.append((tgt, "target")); break          # first mismatch → target corrects
        if nacc == K:
            emit.append((int(tl[:, L - 1 + K, :].argmax(-1)), "target"))  # bonus token
        accepted += nacc
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        per_ms = (time.perf_counter() - t0) * 1000 / max(len(emit), 1)
        for tid, src in emit:
            if n >= max_new_tokens:
                break
            generated.append(tid)
            seq = torch.cat([seq, torch.tensor([[tid]], device=DEVICE)], dim=1)
            full = tokenizer.decode(generated, skip_special_tokens=True)
            yield {"type": "token", "text": full[len(prev_text):], "dt_ms": per_ms,
                   "index": n, "source": src}
            prev_text = full
            n += 1
            if tid == eos:
                n = max_new_tokens
                break
    acc_rate = round(accepted / drafted * 100, 1) if drafted else 0
    yield {"type": "done", "prompt_tokens": prompt_len, "generated": len(generated),
           "vram_mb": vram_allocated_mb(),
           "spec": {"acceptance_rate": acc_rate, "passes": passes, "K": K,
                    "tokens_per_pass": round(len(generated) / passes, 2) if passes else 0}}


@app.get("/api/tokenize")
def tokenize(model_key: str = "qwen25_05b", text: str = ""):
    """Show how a prompt splits into tokens — the units the model actually sees."""
    tok = get_tokenizer(model_key)
    ids = tok.encode(text, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]
    words = len(text.split())
    return {
        "count": len(ids),
        "words": words,
        "tokens_per_word": round(len(ids) / words, 2) if words else 0,
        "tokens": pieces,
    }


# static frontend (mounted last so it doesn't shadow /api routes)
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/", StaticFiles(directory=WEB_DIR), name="static")
