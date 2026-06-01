// NanoServe test bench frontend. Plain JS, no build step.
const $ = (id) => document.getElementById(id);
let modelLabels = {
  qwen25_05b: "Qwen2.5 0.5B",
  phi4_mini: "Phi-4 Mini 3.8B",
};
let vramTotalMb = 0;

function modelLabel(key) { return modelLabels[key] || key; }
function currentModelLabel() {
  const key = $("modelSelect") ? $("modelSelect").value : "qwen25_05b";
  return modelLabel(key);
}

// ---------- device info ----------
function applyModelInfo(d) {
  $("device").textContent = d.device;
  if (d.models && $("modelSelect")) {
    modelLabels = Object.fromEntries(d.models.map(m => [m.key, m.label]));
    $("modelSelect").innerHTML = d.models.map(m =>
      `<option value="${m.key}" ${m.key === d.active_model ? "selected" : ""}>${m.label}</option>`
    ).join("");
  }
  const active = (d.models || []).find(m => m.key === d.active_model);
  if (active) {
    const parts = [active.label];
    if (active.fp16_mb) parts.push(`FP16 ${active.fp16_mb.toFixed(0)} MB`);
    if (active.int8_mb) parts.push(`INT8 ${active.int8_mb.toFixed(0)} MB`);
    if (active.nf4_mb) parts.push(`NF4 ${active.nf4_mb.toFixed(0)} MB`);
    $("deviceSub").textContent = parts.join(" · ");
  }
  if (d.llama_models) {
    llamaModels = Object.fromEntries(d.llama_models.map(m => [m.key, m.quants]));
    if ($("sweepModel") && !$("sweepModel").options.length) {
      $("sweepModel").innerHTML = d.llama_models.map(m => `<option value="${m.key}">${m.label}</option>`).join("");
    }
  }
  if ($("engine")) applyEngine();
}
function refreshModelInfo() {
  return fetch("/api/model_info")
    .then(r => r.json())
    .then(applyModelInfo)
    .catch(() => { $("device").textContent = "backend offline"; });
}
refreshModelInfo();

// ---------- utilities ----------
function escapeHtml(s) {
  return s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}
function bars(el, rows) {
  el.innerHTML = rows.map(r => {
    const pct = Math.max(2, (r.value / r.max) * 100);
    return `<div class="bar-row"><div class="bar-label">${r.label}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${r.color}">${r.text}</div></div></div>`;
  }).join("");
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
function pct(arr, p) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (idx - lo);
}
function mean(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }
function smooth(data, win) {
  if (data.length < win) return data.slice();
  const half = Math.floor(win / 2);
  return data.map((_, i) => {
    const a = Math.max(0, i - half), b = Math.min(data.length, i + half + 1);
    return mean(data.slice(a, b));
  });
}
function computeStats(ttft, decode) {
  const total = (ttft || 0) + decode.reduce((a, b) => a + b, 0);
  const tpot = mean(decode);
  return {
    ttft: ttft || 0, tpot,
    p50: pct(decode, 50), p90: pct(decode, 90), p99: pct(decode, 99),
    toks: tpot ? 1000 / tpot : 0, total, n: decode.length + 1,
  };
}

// ---------- terminal console + VRAM gauge ----------
function clk() {
  const d = new Date();
  return d.toTimeString().slice(0, 8) + "." + String(d.getMilliseconds()).padStart(3, "0");
}
function logLine(msg, cls) {
  const el = $("consoleLog");
  if (!el) return;
  const line = document.createElement("div");
  line.className = "cline" + (cls ? " " + cls : "");
  line.textContent = `[${clk()}] ${msg}`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
  while (el.childElementCount > 400) el.removeChild(el.firstChild);
}
function updateGauge(allocMb) {
  if (!allocMb && allocMb !== 0) return;
  const total = vramTotalMb || 8151;
  const pct = Math.min(100, (allocMb / total) * 100);
  $("vramGaugeFill").style.width = pct.toFixed(1) + "%";
  $("vramGaugeFill").style.background =
    pct > 90 ? "#f85149" : pct > 70 ? "#f0883e" : "#3fb950";
  $("vramGaugeLabel").textContent =
    `${allocMb.toFixed(0)} / ${total.toFixed(0)} MB  (${pct.toFixed(0)}%)`;
}

// ---------- telemetry readouts ----------
function setReadout(id, value, unit) {
  $(id).innerHTML = value + (unit ? `<span class="ro-unit">${unit}</span>` : "");
}
function updateTelemetry(ttft, decode, promptTokens, generated) {
  if (ttft != null) setReadout("tmTTFT", ttft.toFixed(0), "ms");
  if (decode.length) {
    const tpot = mean(decode);
    setReadout("tmTPOT", tpot.toFixed(1), "ms");
    setReadout("tmTPS", (1000 / tpot).toFixed(1), "tok/s");
  }
  const tot = (promptTokens != null ? promptTokens : "?");
  const gen = (generated != null ? generated : decode.length + (ttft != null ? 1 : 0));
  setReadout("tmTokens", `${tot}+${gen}`, "");
}

// ---------- run a single generation over the WS (abortable) ----------
let activeWs = null;       // the in-flight socket, so Stop can close it
let userStopped = false;   // set by the Stop button

function runGeneration(cfg, onToken) {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/generate`);
    activeWs = ws;
    let settled = false, ttft = null, text = "", decode = [];
    const t_open = performance.now();

    const finish = (extra) => {
      if (settled) return; settled = true; activeWs = null;
      resolve({
        ttft, decode, text,
        wall: performance.now() - t_open,
        stats: computeStats(ttft, decode),
        ...extra,
      });
    };

    ws.onopen = () => ws.send(JSON.stringify({
      prompt: cfg.prompt,
      messages: cfg.messages,            // multi-turn history (optional)
      system: cfg.system,
      max_new_tokens: cfg.maxTokens,
      use_cache: cfg.useCache,
      precision: cfg.precision,
      model: cfg.model,
      engine: cfg.engine,
      mmap: cfg.mmap,
    }));

    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "log") {
        logLine(m.msg);
        if (m.vram_mb != null) updateGauge(m.vram_mb);
      } else if (m.type === "meta") {
        if (m.vram_total_mb) vramTotalMb = m.vram_total_mb;
      } else if (m.type === "token") {
        text += m.text;
        if (m.index === 0) ttft = m.dt_ms; else decode.push(m.dt_ms);
        if (onToken) onToken(text, ttft, decode);
      } else if (m.type === "done") {
        if (m.vram_mb != null) updateGauge(m.vram_mb);
        finish({ promptTokens: m.prompt_tokens, generated: m.generated, aborted: false });
        ws.close();
      } else if (m.type === "error") {
        if (!settled) { settled = true; activeWs = null; reject(new Error(m.message)); }
        ws.close();
      }
    };
    // socket closed without a "done" → user hit Stop; resolve with partial output
    ws.onclose = () => finish({
      promptTokens: null,
      generated: decode.length + (ttft != null ? 1 : 0),
      aborted: true,
    });
    ws.onerror = () => { if (!settled) { settled = true; activeWs = null; reject(new Error("WebSocket error")); } };
  });
}

// ---------- chart ----------
let chartSeries = [];
function drawChart() {
  const c = $("latChart"), ctx = c.getContext("2d");
  const W = c.width, H = c.height, padL = 42, padB = 28, padT = 10, padR = 12;
  ctx.clearRect(0, 0, W, H);
  const all = chartSeries.flatMap(s => s.decode);
  if (!all.length) {
    ctx.fillStyle = "#6e7681"; ctx.font = "13px sans-serif";
    ctx.fillText("run the pipeline to see per-token latency", padL + 20, H / 2);
    return;
  }
  const ymax = Math.max(...all) * 1.15;
  const xmax = Math.max(...chartSeries.map(s => s.decode.length), 1);
  const px = (i) => padL + (i / Math.max(xmax - 1, 1)) * (W - padL - padR);
  const py = (v) => (H - padB) - (v / ymax) * (H - padB - padT);
  ctx.strokeStyle = "#1c2128"; ctx.lineWidth = 1;
  ctx.fillStyle = "#6e7681"; ctx.font = "10px sans-serif";
  for (let g = 0; g <= 4; g++) {
    const v = (ymax / 4) * g, y = py(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(v.toFixed(0), 4, y + 3);
  }
  ctx.strokeStyle = "#2a3038";
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB); ctx.stroke();
  ctx.fillText("ms / token", 4, padT + 2);
  ctx.fillText("token #", W / 2 - 18, H - 6);
  chartSeries.forEach(s => {
    if (!s.decode.length) return;
    ctx.fillStyle = s.color + "44";
    s.decode.forEach((v, i) => { ctx.beginPath(); ctx.arc(px(i), py(v), 1.6, 0, 7); ctx.fill(); });
    const sm = smooth(s.decode, 9);
    ctx.strokeStyle = s.color; ctx.lineWidth = 2.5;
    ctx.beginPath();
    sm.forEach((v, i) => i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)));
    ctx.stroke();
  });
  $("legend").innerHTML = chartSeries.map(s => `<span style="color:${s.color}">${s.label}</span>`).join("");
}

// ---------- stat table + run history ----------
function setStatTable(runs) {
  if (!runs.length) {
    $("statBody").innerHTML = `<tr class="empty"><td colspan="8">no runs yet — hit Run inference pipeline</td></tr>`;
    return;
  }
  $("statBody").innerHTML = runs.map(r => {
    const s = r.stats;
    return `<tr>
      <td><span class="swatch" style="background:${r.color}"></span>${r.label}</td>
      <td>${s.ttft.toFixed(0)}</td><td>${s.tpot.toFixed(1)}</td>
      <td>${s.p50.toFixed(1)}</td><td>${s.p90.toFixed(1)}</td><td>${s.p99.toFixed(1)}</td>
      <td>${s.toks.toFixed(1)}</td><td>${(s.total / 1000).toFixed(2)}s</td>
    </tr>`;
  }).join("");
}
let historyCount = 0;
let runLog = [];  // full record of each run, for the Compare panel
function configLabel(cfg, label) {
  if (cfg.engine === "llamacpp") {
    const qs = llamaModels[cfg.model] || [];
    const q = (qs.find(x => x.key === cfg.precision) || {}).label || cfg.precision;
    return `llama.cpp(CPU) · ${modelLabel(cfg.model)} · ${q} · ${label}`;
  }
  return `${modelLabel(cfg.model)} · ${cfg.precision.toUpperCase()} · ${label}`;
}
function logHistory(label, cfg, r) {
  historyCount++;
  const cl = configLabel(cfg, label) + (r.aborted ? " · stopped" : "");
  runLog.push({ n: historyCount, configLabel: cl, cfg, r });
  const body = $("historyBody");
  if (body.querySelector(".empty")) body.innerHTML = "";
  const s = r.stats;
  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${historyCount}</td>
    <td>${cl}</td>
    <td>${r.promptTokens != null ? r.promptTokens : "?"}+${r.generated}</td>
    <td>${s.ttft.toFixed(0)} ms</td><td>${s.tpot.toFixed(1)} ms</td>
    <td>${s.p99.toFixed(1)} ms</td><td>${s.toks.toFixed(1)}</td><td>${(s.total / 1000).toFixed(2)}s</td>`;
  body.insertBefore(row, body.firstChild);
  refreshCompareOptions();
}

// ---------- COMPARE RUNS ----------
function refreshCompareOptions() {
  if (runLog.length < 2) { $("cmpEmpty").style.display = ""; return; }
  $("cmpEmpty").style.display = "none";
  const opts = runLog.map(x => `<option value="${x.n}">#${x.n} · ${x.configLabel}</option>`).join("");
  $("cmpA").innerHTML = opts;
  $("cmpB").innerHTML = opts;
  $("cmpA").value = runLog[runLog.length - 2].n;  // previous run
  $("cmpB").value = runLog[runLog.length - 1].n;  // latest run
  renderCompare();
}
function cmpRow(name, a, b, lowerBetter, unit, digits) {
  const delta = b - a;
  const same = Math.abs(delta) < Math.pow(10, -digits) / 2;
  const better = lowerBetter ? delta < 0 : delta > 0;
  const color = same ? "var(--muted)" : (better ? "#3fb950" : "#f85149");
  const sign = delta > 0 ? "+" : "";
  const pctTxt = a ? ` (${sign}${(delta / a * 100).toFixed(0)}%)` : "";
  return `<tr><td>${name}</td><td>${a.toFixed(digits)}${unit}</td><td>${b.toFixed(digits)}${unit}</td>
    <td style="color:${color}">${same ? "—" : sign + delta.toFixed(digits) + unit + pctTxt}</td></tr>`;
}
function renderCompare() {
  if (runLog.length < 2) return;
  const A = runLog.find(x => x.n === +$("cmpA").value);
  const B = runLog.find(x => x.n === +$("cmpB").value);
  if (!A || !B) return;
  const sa = A.r.stats, sb = B.r.stats;
  $("cmpTable").innerHTML =
    `<table class="stat-table"><thead><tr>
       <th>metric</th><th>A · #${A.n}</th><th>B · #${B.n}</th><th>Δ (B vs A)</th>
     </tr></thead><tbody>
       ${cmpRow("TTFT", sa.ttft, sb.ttft, true, " ms", 0)}
       ${cmpRow("TPOT", sa.tpot, sb.tpot, true, " ms", 1)}
       ${cmpRow("p99 latency", sa.p99, sb.p99, true, " ms", 1)}
       ${cmpRow("Throughput", sa.toks, sb.toks, false, " tok/s", 1)}
       ${cmpRow("Total time", sa.total / 1000, sb.total / 1000, true, " s", 2)}
       ${cmpRow("Tokens out", A.r.generated, B.r.generated, false, "", 0)}
     </tbody></table>`;
  $("cmpOutputs").innerHTML =
    `<div class="cmp-col"><div class="cmp-head" style="color:#58a6ff">A · #${A.n} — ${A.configLabel}</div>
       <div class="cmp-text">${escapeHtml(A.r.text || "(no output)")}</div></div>
     <div class="cmp-col"><div class="cmp-head" style="color:#3fb950">B · #${B.n} — ${B.configLabel}</div>
       <div class="cmp-text">${escapeHtml(B.r.text || "(no output)")}</div></div>`;
}
$("cmpA").addEventListener("change", renderCompare);
$("cmpB").addEventListener("change", renderCompare);

// ---------- control panel ----------
function readCfg(useCacheOverride) {
  return {
    prompt: $("prompt").value,
    system: $("system").value.trim(),
    maxTokens: parseInt($("maxTokens").value),
    useCache: useCacheOverride !== undefined ? useCacheOverride : $("cacheEngine").value === "kv",
    precision: $("precision").value,
    model: $("modelSelect").value,
    engine: $("engine").value,
    mmap: $("mmapMode") ? $("mmapMode").value === "mmap" : true,
  };
}
function setBusy(b, msg) {
  const llama = $("engine").value === "llamacpp";
  $("runBtn").disabled = b;
  $("clearBtn").disabled = b;
  // the A/B buttons are HuggingFace concepts (naive cache; FP16-vs-4bit) — N/A on llama.cpp
  $("abBtn").disabled = b || llama;
  $("optBtn").disabled = b || llama;
  $("stopBtn").disabled = !b;   // Stop is only active while a run is in flight
  $("runStatus").textContent = msg || "";
}

// ---------- chat transcript (multi-turn conversation) ----------
let chatMessages = [];  // [{role:"user"|"assistant", content}]
function renderTranscript(streaming) {
  const parts = chatMessages.map(m =>
    `<div class="msg ${m.role}"><span class="who">${m.role === "user" ? "You" : "Model"}</span>${escapeHtml(m.content)}</div>`
  );
  // while a reply is streaming, the last stored message is the user's turn:
  // show a live assistant bubble for the in-progress text
  if (streaming != null) {
    parts.push(`<div class="msg assistant"><span class="who">Model</span>${escapeHtml(streaming)}<span class="cursor">▋</span></div>`);
  }
  $("output").innerHTML = parts.join("") || '<span class="hint">no messages yet — type something and hit Send</span>';
  $("output").scrollTop = $("output").scrollHeight;
}
const PREC_HINT = {
  fp16: "real · full 16-bit weights",
  int8: "real · bitsandbytes LLM.int8()",
  nf4: "real · bitsandbytes 4-bit (NF4 family, like AWQ/GPTQ)",
};
const HF_PRECISIONS = [
  ["fp16", "FP16 — baseline (full)"],
  ["int8", "INT8 — LLM.int8() (real)"],
  ["nf4", "4-bit NF4 — weight-only (real)"],
];
let llamaModels = {};  // { modelKey: [{key,label}] } filled from /api/model_info

// repopulate the quantization dropdown for the current engine + model
function refreshQuantOptions() {
  if ($("engine").value === "llamacpp") {
    const qs = llamaModels[$("modelSelect").value] || [{ key: "q4_k_m", label: "Q4_K_M · 4-bit" }];
    $("precision").innerHTML = qs.map(q => `<option value="${q.key}">${q.label}</option>`).join("");
    if (qs.find(q => q.key === "q4_k_m")) $("precision").value = "q4_k_m";
    $("precHint").textContent = "GGUF · down to 2-bit (CPU only)";
  } else {
    $("precision").innerHTML = HF_PRECISIONS.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
    $("precHint").textContent = PREC_HINT[$("precision").value] || "";
  }
}

// swap controls when the engine changes (llama.cpp = CPU/GGUF, always KV-cached)
function applyEngine() {
  const llama = $("engine").value === "llamacpp";
  $("engineHint").textContent = llama ? "llama.cpp · runs on the CPU (edge)" : "PyTorch on the GPU";
  $("cacheEngine").disabled = llama; if (llama) $("cacheEngine").value = "kv";
  $("mmapWrap").style.display = llama ? "" : "none";
  // A/B buttons are HuggingFace-only (llama.cpp always KV-caches; quant ladder differs)
  $("abBtn").disabled = llama;
  $("optBtn").disabled = llama;
  const why = llama ? "HuggingFace only — llama.cpp always uses its KV cache" : "";
  $("abBtn").title = why; $("optBtn").title = why;
  refreshQuantOptions();
}
$("engine").addEventListener("change", applyEngine);

$("maxTokens").addEventListener("input", e => $("maxTokensVal").textContent = e.target.value);
$("precision").addEventListener("change", e => {
  if ($("engine").value === "hf") $("precHint").textContent = PREC_HINT[e.target.value] || "";
});
$("precHint").textContent = PREC_HINT.fp16;
$("modelSelect").addEventListener("change", () => {
  $("deviceSub").textContent = `${currentModelLabel()} · loads on next run`;
  refreshQuantOptions();   // GGUF quant ladder differs per model
});

// send a chat turn — appends your message, generates with full conversation context
$("runBtn").addEventListener("click", async () => {
  const msg = $("prompt").value.trim();
  if (!msg) { $("runStatus").textContent = "type a message first"; return; }
  const cfg = readCfg();
  const label = cfg.useCache ? "KV on" : "naive";
  const color = cfg.useCache ? "#3fb950" : "#f85149";

  chatMessages.push({ role: "user", content: msg });
  $("prompt").value = "";
  renderTranscript("");                 // user bubble + empty streaming reply
  userStopped = false;
  setBusy(true, `generating (${label})…`);
  logLine(`▶ turn ${Math.ceil(chatMessages.length / 2)} · ${modelLabel(cfg.model)} · ${cfg.precision.toUpperCase()} · ${label} · ${cfg.maxTokens} tok`, "in");
  chartSeries = [{ decode: [], color, label }];
  try {
    const r = await runGeneration({ ...cfg, messages: chatMessages }, (text, ttft, decode) => {
      renderTranscript(text);
      chartSeries[0].decode = decode; drawChart();
      updateTelemetry(ttft, decode, null, null);
    });
    // store the reply (even a partial one from Stop) so the chat can continue
    chatMessages.push({ role: "assistant", content: r.text || "(stopped before any output)" });
    renderTranscript(null);
    chartSeries[0].decode = r.decode; chartSeries[0].stats = r.stats; drawChart();
    updateTelemetry(r.ttft, r.decode, r.promptTokens, r.generated);
    setStatTable(chartSeries);
    logHistory(label, cfg, r);
    if (r.aborted) logLine(`⏹ stopped by user after ${r.generated} tokens`, "err");
    refreshModelInfo();
  } catch (e) {
    chatMessages.push({ role: "assistant", content: `[error: ${e.message}]` });
    renderTranscript(null);
    logLine("ERROR: " + e.message, "err");
  }
  setBusy(false);
});

// Stop — closes the active socket; the server breaks out within ~1 token
$("stopBtn").addEventListener("click", () => {
  if (!activeWs) return;
  userStopped = true;
  $("runStatus").textContent = "stopping…";
  logLine("⏹ stop requested — closing stream", "err");
  try { activeWs.close(); } catch (e) { /* already closing */ }
});

// A/B: same prompt, cache on then off
$("abBtn").addEventListener("click", async () => {
  const base = readCfg();
  base.engine = "hf"; base.precision = "fp16";   // cache on/off is a HuggingFace comparison
  if (!base.prompt.trim()) base.prompt = "Explain how a CPU works, in detail.";  // A/B is a standalone benchmark
  userStopped = false;
  $("output").innerHTML = ""; chartSeries = [];
  const runs = [];
  try {
    setBusy(true, "warming up…");
    logLine("A/B warmup (8 tokens)…", "in");
    await runGeneration({ ...base, useCache: true, maxTokens: 8 });
    for (const useCache of [true, false]) {
      const cfg = { ...base, useCache };
      const label = useCache ? "KV on" : "naive";
      const color = useCache ? "#3fb950" : "#f85149";
      setBusy(true, `A/B: ${label} at ${cfg.maxTokens} tokens…`);
      const ser = { decode: [], color, label };
      chartSeries.push(ser);
      const r = await runGeneration(cfg, (text, ttft, decode) => {
        $("output").innerHTML = `<b style="color:${color}">${label}:</b>\n` + escapeHtml(text) + '<span class="cursor">▋</span>';
        $("output").scrollTop = $("output").scrollHeight;
        ser.decode = decode; drawChart();
        updateTelemetry(ttft, decode, null, null);
      });
      ser.decode = r.decode; ser.stats = r.stats; drawChart();
      updateTelemetry(r.ttft, r.decode, r.promptTokens, r.generated);
      runs.push({ ...ser, r });
      setStatTable(chartSeries);
      logHistory(label, cfg, r);
    }
    refreshModelInfo();
    const on = runs[0].stats, off = runs[1].stats;
    const tpotX = off.tpot / on.tpot, totalX = off.total / on.total;
    $("speedupNote").innerHTML =
      `KV cache is <b style="color:#3fb950">${tpotX.toFixed(2)}× faster per token</b> ` +
      `(${off.tpot.toFixed(1)} → ${on.tpot.toFixed(1)} ms) and ` +
      `<b style="color:#3fb950">${totalX.toFixed(2)}× faster overall</b> at ${base.maxTokens} tokens. ` +
      `Compare <b>TPOT and p99</b>, not TTFT — TTFT is the prefill, identical work either way. ` +
      `The gap widens with sequence length (naive is O(N²), cached is O(N)) and grows much larger on bigger models.`;
    $("output").innerHTML = `<b style="color:#3fb950">KV on:</b>\n${escapeHtml(runs[0].r.text)}\n\n` +
      `<b style="color:#f85149">naive:</b>\n${escapeHtml(runs[1].r.text)}`;
  } catch (e) {
    $("output").innerHTML = `<span style="color:#f85149">Error: ${escapeHtml(e.message)}</span>`;
    logLine("ERROR: " + e.message, "err");
  }
  setBusy(false);
});

// A/B: stack the optimizations — baseline (FP16 + no cache) vs optimized (4-bit + KV cache)
$("optBtn").addEventListener("click", async () => {
  const base = readCfg();
  base.engine = "hf";   // FP16 vs 4-bit is a HuggingFace comparison
  if (!base.prompt.trim()) base.prompt = "Explain how a CPU works, in detail.";
  userStopped = false;
  $("output").innerHTML = ""; chartSeries = [];
  const configs = [
    { label: "baseline · no cache", color: "#f85149", precision: "fp16", useCache: false },
    { label: "optimized · KV cache", color: "#3fb950", precision: "nf4", useCache: true },
  ];
  const runs = [];
  try {
    setBusy(true, "warming up…");
    logLine("optimized-vs-baseline warmup…", "in");
    await runGeneration({ ...base, precision: "fp16", useCache: true, maxTokens: 8 });
    for (const c of configs) {
      const cfg = { ...base, precision: c.precision, useCache: c.useCache };
      setBusy(true, `running ${c.label} (loads ${c.precision})…`);
      const ser = { decode: [], color: c.color, label: c.label };
      chartSeries.push(ser);
      const r = await runGeneration(cfg, (text, ttft, decode) => {
        $("output").innerHTML = `<b style="color:${c.color}">${c.label}:</b>\n` + escapeHtml(text) + '<span class="cursor">▋</span>';
        $("output").scrollTop = $("output").scrollHeight;
        ser.decode = decode; drawChart();
        updateTelemetry(ttft, decode, null, null);
      });
      ser.decode = r.decode; ser.stats = r.stats; drawChart();
      updateTelemetry(r.ttft, r.decode, r.promptTokens, r.generated);
      runs.push({ ...ser, r });
      setStatTable(chartSeries);
      logHistory(c.label, cfg, r);
    }
    const info = await fetch("/api/model_info").then(r => r.json()).catch(() => null);
    if (info) applyModelInfo(info);
    const b = runs[0].stats, o = runs[1].stats;
    const tpotX = b.tpot / o.tpot, totalX = b.total / o.total;   // >1 = optimized faster
    const col = x => x >= 1.0 ? "#3fb950" : "#f0883e";
    const ratio = `<b style="color:${col(tpotX)}">${tpotX.toFixed(2)}× per-token</b>, ` +
                  `<b style="color:${col(totalX)}">${totalX.toFixed(2)}× overall</b>`;
    const vramTxt = (info && info.fp16_mb && info.nf4_mb)
      ? `<b style="color:#3fb950">${(info.fp16_mb / info.nf4_mb).toFixed(1)}× less VRAM</b> (${info.fp16_mb.toFixed(0)} → ${info.nf4_mb.toFixed(0)} MB)`
      : "less VRAM";
    const speedStory = totalX >= 1.0
      ? `Net speed-up comes from the <b>KV cache</b> (O(N)→ vs O(N²) recompute).`
      : `It's actually <b>slower</b> here — at short outputs the KV cache barely helps, and 4-bit (bitsandbytes) ` +
        `adds <b>dequant overhead</b> on a model this small. That's the honest trade: <b>4-bit buys memory, not speed</b>, ` +
        `on small models. Raise max-tokens to watch the cache win grow.`;
    $("speedupNote").innerHTML =
      `Stacking 4-bit + KV cache vs FP16 + no cache: ${ratio}, and ${vramTxt}. ${speedStory} ` +
      `The other two optimizations — <b>PagedAttention</b> and <b>fused/zero-copy kernels</b> — live in the engine ` +
      `layer (vLLM / llama.cpp), not HuggingFace.`;
    $("output").innerHTML = `<b style="color:#f85149">baseline:</b>\n${escapeHtml(runs[0].r.text)}\n\n` +
      `<b style="color:#3fb950">optimized:</b>\n${escapeHtml(runs[1].r.text)}`;
  } catch (e) {
    $("output").innerHTML = `<span style="color:#f85149">Error: ${escapeHtml(e.message)}</span>`;
    logLine("ERROR: " + e.message, "err");
  }
  setBusy(false);
});

$("clearBtn").addEventListener("click", () => {
  chatMessages = [];
  renderTranscript(null);
  chartSeries = []; drawChart(); setStatTable([]); $("speedupNote").innerHTML = "";
  ["tmTTFT", "tmTPOT", "tmTPS", "tmTokens"].forEach(id => $(id).innerHTML = "–");
  runLog = []; historyCount = 0;
  $("historyBody").innerHTML = `<tr class="empty"><td colspan="8">runs you make will be logged here</td></tr>`;
  $("cmpTable").innerHTML = ""; $("cmpOutputs").innerHTML = "";
  $("cmpA").innerHTML = ""; $("cmpB").innerHTML = ""; $("cmpEmpty").style.display = "";
  logLine("conversation cleared — starting over", "in");
});
$("consoleClear").addEventListener("click", () => { $("consoleLog").innerHTML = ""; });

// ---------- QUANTIZATION TRADEOFF (size vs quality vs speed) ----------
let tradePoints = [];
function bestTradeoff(points) {
  // normalize size + ppl to [0,1]; the point closest to the ideal (small, low-ppl) corner wins
  const xs = points.map(p => p.size_mb), ys = points.map(p => p.ppl);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const d = points.map(p => Math.hypot(
    (p.size_mb - x0) / ((x1 - x0) || 1), (p.ppl - y0) / ((y1 - y0) || 1)));
  return d.indexOf(Math.min(...d));
}
function drawTradeoff() {
  const c = $("tradeChart"), ctx = c.getContext("2d");
  const W = c.width, H = c.height, padL = 48, padB = 34, padT = 14, padR = 16;
  ctx.clearRect(0, 0, W, H);
  if (!tradePoints.length) {
    ctx.fillStyle = "#6e7681"; ctx.font = "13px sans-serif";
    ctx.fillText("run the sweep to plot size vs quality", padL + 10, H / 2);
    return;
  }
  const xs = tradePoints.map(p => p.size_mb), ys = tradePoints.map(p => p.ppl);
  const xmin = Math.min(...xs) * 0.9, xmax = Math.max(...xs) * 1.08;
  const ymin = Math.min(...ys) * 0.97, ymax = Math.max(...ys) * 1.06;
  const px = v => padL + ((v - xmin) / (xmax - xmin)) * (W - padL - padR);
  const py = v => (H - padB) - ((v - ymin) / (ymax - ymin)) * (H - padB - padT);
  // grid + labels
  ctx.strokeStyle = "#1c2128"; ctx.fillStyle = "#6e7681"; ctx.font = "10px sans-serif"; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const yv = ymin + (ymax - ymin) * g / 4, y = py(yv);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(yv.toFixed(1), 6, y + 3);
  }
  ctx.strokeStyle = "#2a3038";
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB); ctx.stroke();
  ctx.fillText("perplexity (quality, lower=better)", 6, padT - 2);
  ctx.fillText("size (MB) →", W - 90, H - 6);
  // connecting line (sorted by size)
  const sorted = [...tradePoints].sort((a, b) => a.size_mb - b.size_mb);
  ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 2;
  ctx.beginPath(); sorted.forEach((p, i) => i ? ctx.lineTo(px(p.size_mb), py(p.ppl)) : ctx.moveTo(px(p.size_mb), py(p.ppl))); ctx.stroke();
  const best = bestTradeoff(tradePoints);
  tradePoints.forEach((p, i) => {
    const x = px(p.size_mb), y = py(p.ppl), isBest = i === best;
    ctx.fillStyle = isBest ? "#3fb950" : "#58a6ff";
    ctx.beginPath(); ctx.arc(x, y, isBest ? 6 : 4, 0, 7); ctx.fill();
    ctx.fillStyle = "#c9d3de"; ctx.font = "10px sans-serif";
    ctx.fillText(p.quant.toUpperCase().replace("_", ""), x + 8, y + 3);
  });
}
$("sweepBtn").addEventListener("click", () => {
  const model = $("sweepModel") ? $("sweepModel").value : "qwen25_05b";
  const engine = $("sweepEngine") ? $("sweepEngine").value : "llamacpp";
  $("sweepBtn").disabled = true;
  const where = engine === "hf" ? "GPU · loads FP16/INT8/4-bit" : "CPU · downloads + loads each GGUF quant";
  $("tradeNote").innerHTML = `running ${engine === "hf" ? "HuggingFace" : "llama.cpp"} sweep on <b>${modelLabel(model)}</b> (${where} — may take a while)…`;
  fetch(`/api/quant_sweep?model_key=${model}&engine=${engine}`).then(r => r.json()).then(d => {
    $("sweepBtn").disabled = false;
    tradePoints = d.points || [];
    drawTradeoff();
    $("tradeBody").innerHTML = tradePoints.length
      ? tradePoints.map(p => `<tr><td>${p.label}</td><td>${p.size_mb} MB</td><td>${p.ppl}</td><td>${p.tps}</td></tr>`).join("")
      : `<tr class="empty"><td colspan="4">no data</td></tr>`;
    if (tradePoints.length) {
      const b = tradePoints[bestTradeoff(tradePoints)];
      const big = tradePoints.reduce((a, p) => p.size_mb > a.size_mb ? p : a);
      const bestPpl = Math.min(...tradePoints.map(p => p.ppl));
      const tail = engine === "hf"
        ? `bitsandbytes only offers these 3 levels, and on a small model they're near-identical in quality — so <b>4-bit</b> is ~free memory savings. The fine knee needs GGUF (switch engine).`
        : `Below the knee, perplexity climbs fast for little extra size savings — which is why <b>Q4_K_M</b> is the common default.`;
      $("tradeNote").innerHTML =
        `Best size↔quality tradeoff: <b style="color:#3fb950">${b.label}</b> ` +
        `— ${b.size_mb} MB at perplexity ${b.ppl} (lowest is ${bestPpl}), ${b.tps} tok/s. ` +
        `That's <b>${(big.size_mb / b.size_mb).toFixed(1)}× smaller</b> than ${big.label}. ` + tail;
    }
  }).catch(e => { $("sweepBtn").disabled = false; $("tradeNote").textContent = "Error: " + e; });
});
drawTradeoff();

// ---------- STATIC BATCHING THROUGHPUT (HF / GPU) ----------
$("batchBtn").addEventListener("click", () => {
  $("batchBtn").disabled = true;
  $("batchNote").textContent = "running batched decode at batch 1→16 (Qwen 0.5B, FP16, GPU)…";
  fetch("/api/batching?model_key=qwen25_05b&precision=fp16").then(r => r.json()).then(d => {
    $("batchBtn").disabled = false;
    if (d.error) { $("batchNote").textContent = d.error; return; }
    $("batchHero").textContent = d.speedup + "×";
    const tps = d.throughput, max = Math.max(...tps.map(t => t.tokens_per_sec));
    bars($("batchBars"), tps.map(t => ({
      label: `batch ${t.batch}`, value: t.tokens_per_sec, max, color: "#58a6ff",
      text: `${t.tokens_per_sec.toFixed(0)} tok/s · ${t.ms_per_step.toFixed(0)} ms/step`,
    })));
    $("batchNote").innerHTML =
      `Batch 16 produces <b style="color:#58a6ff">${d.speedup}× more tokens/sec</b> than batch 1, ` +
      `while each step takes about the <b>same time</b> (${d.flat}× latency). The step is dominated by ` +
      `reading the weights once — extra sequences ride along nearly free. That's why servers batch ` +
      `aggressively; <b>continuous</b> batching just keeps the batch full at all times (scheduler-level).`;
  }).catch(e => { $("batchBtn").disabled = false; $("batchNote").textContent = "Error: " + e; });
});

drawChart();
renderTranscript(null);  // show empty-state hint in the transcript box
