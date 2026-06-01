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
  renderVram(d);
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
      max_new_tokens: cfg.maxTokens,
      use_cache: cfg.useCache,
      precision: cfg.precision,
      model: cfg.model,
      pinned: cfg.pinned,
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
function logHistory(label, cfg, r) {
  historyCount++;
  const body = $("historyBody");
  if (body.querySelector(".empty")) body.innerHTML = "";
  const s = r.stats;
  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${historyCount}</td>
    <td>${modelLabel(cfg.model)} · ${cfg.precision.toUpperCase()} · ${label}${r.aborted ? " · stopped" : ""}</td>
    <td>${r.promptTokens != null ? r.promptTokens : "?"}+${r.generated}</td>
    <td>${s.ttft.toFixed(0)} ms</td><td>${s.tpot.toFixed(1)} ms</td>
    <td>${s.p99.toFixed(1)} ms</td><td>${s.toks.toFixed(1)}</td><td>${(s.total / 1000).toFixed(2)}s</td>`;
  body.insertBefore(row, body.firstChild);
}

// ---------- control panel ----------
function readCfg(useCacheOverride) {
  return {
    prompt: $("prompt").value,
    maxTokens: parseInt($("maxTokens").value),
    useCache: useCacheOverride !== undefined ? useCacheOverride : $("cacheEngine").value === "kv",
    precision: $("precision").value,
    model: $("modelSelect").value,
    pinned: $("memLoad").value === "pinned",
  };
}
function setBusy(b, msg) {
  ["runBtn", "abBtn", "clearBtn"].forEach(id => $(id).disabled = b);
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

$("maxTokens").addEventListener("input", e => $("maxTokensVal").textContent = e.target.value);
$("precision").addEventListener("change", e => {
  $("precHint").textContent = PREC_HINT[e.target.value] || "";
  $("batchBars").innerHTML = ""; $("batchNote").textContent = ""; batchData = null;
});
$("precHint").textContent = PREC_HINT.fp16;
$("modelSelect").addEventListener("change", () => {
  $("deviceSub").textContent = `${currentModelLabel()} · loads on next run`;
  $("vramBars").innerHTML = "";
  $("batchBars").innerHTML = ""; $("batchHero").textContent = "–"; $("batchNote").textContent = "";
  $("quantHero").textContent = "–"; $("quantHeroErr").textContent = "–";
  $("quantBars").innerHTML = ""; $("quantNote").textContent = "press Measure to update this model";
  batchData = null;
  pagedCompute(`showing ${currentModelLabel()} KV dimensions`);
});
$("blockSize").addEventListener("change", () => pagedCompute(`block size = ${$("blockSize").value} tokens`));

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
    if (!r.aborted) autoUpdatePaged(r);
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
      if (useCache) autoUpdatePaged(r);
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

$("clearBtn").addEventListener("click", () => {
  chatMessages = [];
  renderTranscript(null);
  chartSeries = []; drawChart(); setStatTable([]); $("speedupNote").innerHTML = "";
  ["tmTTFT", "tmTPOT", "tmTPS", "tmTokens"].forEach(id => $(id).innerHTML = "–");
  logLine("conversation cleared — starting over", "in");
});
$("consoleClear").addEventListener("click", () => { $("consoleLog").innerHTML = ""; });
drawChart();
renderTranscript(null);  // show empty-state hint in the transcript box

// ---------- CONTINUOUS BATCHING ----------
function renderCapacity(d) {
  const cap = d.capacity, max = cap.paged_fit;
  bars($("capBars"), [
    { label: "naive reserve", value: cap.naive_fit, max, color: "#d62728", text: cap.naive_fit + " reqs" },
    { label: "paged", value: cap.paged_fit, max, color: "#2ca02c", text: cap.paged_fit + " reqs" },
  ]);
  $("capNote").innerHTML =
    `In a <b>${cap.budget_gb} GB</b> KV budget (avg ${cap.avg_len}, max ${cap.max_len}): ` +
    `naive fits ~<b>${cap.naive_fit}</b>, paging fits ~<b style="color:#2ca02c">${cap.paged_fit}</b> ` +
    `— <b style="color:#2ca02c">${cap.ratio.toFixed(1)}× more concurrent users</b> on the same GPU.`;
}
let batchData = null;
function fetchBatching(runBenchmark) {
  const avg = $("capAvg").value, max = $("capMax").value, budget = $("capBudget").value;
  const model = $("modelSelect").value, precision = $("precision").value;
  if (runBenchmark) { $("batchNote").textContent = "running batched-decode benchmark (loads once, ~10s)…"; $("batchBtn").disabled = true; }
  fetch(`/api/batching?avg_len=${avg}&max_len=${max}&budget_gb=${budget}&model_key=${model}&precision=${precision}`)
    .then(r => r.json()).then(d => {
      batchData = d; $("batchBtn").disabled = false;
      $("batchHero").textContent = d.batch_speedup.toFixed(1) + "×";
      const tps = d.throughput, maxTps = Math.max(...tps.map(t => t.tokens_per_sec));
      bars($("batchBars"), tps.map(t => ({
        label: `batch ${t.batch}`, value: t.tokens_per_sec, max: maxTps,
        color: "#58a6ff", text: `${t.tokens_per_sec.toFixed(0)} tok/s · ${t.ms_per_step.toFixed(0)} ms/step`,
      })));
      const flat = tps[tps.length - 1].ms_per_step / tps[0].ms_per_step;
      $("batchNote").innerHTML =
        `Batch 16 produces <b style="color:#58a6ff">${d.batch_speedup.toFixed(1)}× more tokens/sec</b> than batch 1, ` +
        `yet each step takes about the <b>same time</b> (${flat.toFixed(2)}× latency). ` +
        `The step is dominated by reading weights once — extra sequences ride along nearly free.`;
      renderCapacity(d); refreshModelInfo();
    })
    .catch(e => { $("batchBtn").disabled = false; $("batchNote").textContent = "Error: " + e; });
}
$("batchBtn").addEventListener("click", () => fetchBatching(true));
const debouncedCap = debounce(() => { if (batchData) fetchBatching(false); }, 500);
["capAvg", "capMax", "capBudget"].forEach(id => $(id).addEventListener("input", debouncedCap));

// ---------- PHYSICAL MEMORY PAGE GRID (simulation) ----------
function autoUpdatePaged(r) {
  const total = r.promptTokens + r.generated;
  $("lengths").value = $("lengths").value + "," + total;
  pagedCompute(`auto-added your last run (${total} tokens)`);
}
function pagedCompute(statusMsg) {
  const lengths = $("lengths").value, maxLen = $("maxLen").value;
  const model = $("modelSelect").value, blockSize = $("blockSize").value;
  fetch(`/api/paged?lengths=${encodeURIComponent(lengths)}&max_len=${maxLen}&model_key=${model}&block_size=${blockSize}`)
    .then(r => r.json()).then(d => {
      $("pagedHero").textContent = d.reduction.toFixed(1) + "×";
      const max = d.naive_mb;
      bars($("pagedBars"), [
        { label: "naive reserve", value: d.naive_mb, max, color: "#d62728", text: d.naive_mb.toFixed(0) + " MB" },
        { label: "paged", value: d.paged_mb, max, color: "#2ca02c", text: d.paged_mb.toFixed(1) + " MB" },
        { label: "actually needed", value: d.useful_mb, max, color: "#58a6ff", text: d.useful_mb.toFixed(1) + " MB" },
      ]);
      const naiveBlocks = d.naive_blocks_per_seq;
      $("blockGrid").innerHTML = d.lengths.map((L, i) => {
        const used = d.blocks_per_seq[i];
        const wasted = naiveBlocks - used;
        let html = `<div class="seq-row"><div class="seq-name">req ${i} (${L} tok)</div><div class="blocks">`;
        html += `<div class="blk used"></div>`.repeat(used);
        html += `<div class="blk wasted"></div>`.repeat(Math.min(wasted, 40));
        if (wasted > 40) html += `<span class="wasted" style="font-size:10px"> +${wasted - 40} more</span>`;
        return html + "</div></div>";
      }).join("");
      $("pagedNote").innerHTML =
        (statusMsg ? `<i>${statusMsg}</i><br>` : "") +
        `Naive reserves ${d.naive_blocks_per_seq} blocks/request regardless of length — grey cells are wasted. ` +
        `Block = ${d.block_size} tokens = ${(d.bytes_per_token * d.block_size / 1024).toFixed(0)} KB. ` +
        `<i>Simulation of vLLM's allocator — real KV byte math, not a live kernel.</i>`;
    })
    .catch(e => { $("pagedNote").textContent = "Error: " + e; });
}
$("pagedBtn").addEventListener("click", () => pagedCompute());
const debouncedPaged = debounce(() => pagedCompute(), 600);
$("lengths").addEventListener("input", debouncedPaged);
$("maxLen").addEventListener("input", debouncedPaged);
pagedCompute();

// ---------- QUANTIZATION (measured; AWQ/GPTQ/GGUF modeled) ----------
function quantFetch() {
  const gs = $("groupSize").value;
  $("quantNote").textContent = "measuring on a real weight…";
  fetch(`/api/quant?group_size=${gs}&model_key=${$("modelSelect").value}`).then(r => r.json()).then(d => {
    $("quantHero").textContent = (d.fp16_kb / d.int4_kb).toFixed(1) + "×";
    $("quantHeroErr").textContent = d.int4_rel_err.toFixed(1) + "%";
    const max = d.fp16_kb;
    bars($("quantBars"), [
      { label: "FP16 baseline", value: d.fp16_kb, max, color: "#1f77b4", text: d.fp16_kb.toFixed(0) + " KB" },
      { label: `INT8 (${d.int8_rel_err.toFixed(1)}% err)`, value: d.int8_kb, max, color: "#ff7f0e", text: d.int8_kb.toFixed(0) + " KB" },
      { label: `INT4 AWQ/GPTQ-style g=${gs} (${d.int4_rel_err.toFixed(1)}%)`, value: d.int4_kb, max, color: "#2ca02c", text: d.int4_kb.toFixed(0) + " KB" },
      { label: `INT2 GGUF-style (${d.int2_rel_err.toFixed(1)}%)`, value: d.int2_kb, max, color: "#9467bd", text: d.int2_kb.toFixed(0) + " KB" },
    ]);
    $("quantNote").innerHTML =
      `Weight: <code>${d.weight}</code> ${d.shape.join("×")}. ` +
      `INT4 group=${gs}: <b style="color:#2ca02c">${(d.fp16_kb / d.int4_kb).toFixed(1)}× smaller</b> at ${d.int4_rel_err.toFixed(1)}% error. ` +
      `INT2 is <b>${(d.fp16_kb / d.int2_kb).toFixed(1)}× smaller</b> but error jumps to ${d.int2_rel_err.toFixed(1)}% — why 2-bit needs clever schemes. ` +
      `<i>INT8 & 4-bit run live above; INT2/AWQ/GPTQ/GGUF are measured size+error here.</i>`;
    refreshModelInfo();
  }).catch(e => { $("quantNote").textContent = "Error: " + e; });
}
$("quantBtn").addEventListener("click", quantFetch);
$("groupSize").addEventListener("change", quantFetch);
quantFetch();

function renderVram(d) {
  if (!d.fp16_mb && !d.nf4_mb && !d.int8_mb) return;
  const max = Math.max(d.fp16_mb || 0, d.int8_mb || 0, d.nf4_mb || 0);
  const rows = [];
  if (d.fp16_mb) rows.push({ label: "model FP16", value: d.fp16_mb, max, color: "#1f77b4", text: d.fp16_mb.toFixed(0) + " MB" });
  if (d.int8_mb) rows.push({ label: "model INT8", value: d.int8_mb, max, color: "#ff7f0e", text: d.int8_mb.toFixed(0) + " MB" });
  if (d.nf4_mb) rows.push({ label: "model 4-bit", value: d.nf4_mb, max, color: "#2ca02c", text: d.nf4_mb.toFixed(0) + " MB" });
  bars($("vramBars"), rows);
}

// ---------- ZERO-COPY ----------
$("zcBtn").addEventListener("click", () => {
  $("zcNote").textContent = "running 20-iteration transfer benchmark…";
  $("zcBars").innerHTML = "";
  fetch("/api/zerocopy").then(r => r.json()).then(d => {
    if (d.error) { $("zcNote").textContent = d.error; return; }
    $("zcHero").textContent = d.speedup.toFixed(1) + "×";
    const max = Math.max(d.pinned_gbps, d.pageable_gbps);
    bars($("zcBars"), [
      { label: "pageable", value: d.pageable_gbps, max, color: "#d62728", text: d.pageable_gbps.toFixed(1) + " GB/s" },
      { label: "pinned", value: d.pinned_gbps, max, color: "#2ca02c", text: d.pinned_gbps.toFixed(1) + " GB/s" },
    ]);
    $("zcNote").innerHTML =
      `Pinned memory transfers <b style="color:#2ca02c">${d.speedup.toFixed(1)}× faster</b> ` +
      `(${d.transfer_mb.toFixed(0)} MB, averaged over 20 runs). ±5% variance is normal (PCIe thermal).`;
  });
});
