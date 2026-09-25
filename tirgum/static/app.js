// Tirgum web app: estimate → run with live progress → library of finished videos.
const $ = (id) => document.getElementById(id);
const STEPS = [["download", "Download"], ["detect", "Captions"], ["transcribe", "Transcribe"],
               ["scan", "Timing"], ["translate", "Translate"], ["render", "Render"]];
const MODELS = [["gemini-3.8-flash", "Gemini 3.8 Flash (cheapest, best in our tests)"],
                ["claude-sonnet-4-6", "Claude Sonnet 4.6"]];
let settings = {}, current = null, library = [], pollTimer = null, estimateTimer = null;

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const money = (v) => v == null ? "—" : `$${v < 0.1 ? Number(v).toFixed(3) : Number(v).toFixed(2)}`;
const tokens = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n ?? 0);
const clock = (sec) => { sec = Math.max(0, Math.round(sec || 0)); const m = Math.floor(sec / 60), s = sec % 60; return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`; };
const length = (sec) => sec ? clock(sec).replace(/ \d+s$/, "") : "";
const isYouTube = (u) => /^(https?:\/\/)?(www\.|m\.)?(youtube\.com|youtu\.be)\//i.test(u.trim());
const pref = (k, d) => { try { return localStorage.getItem("tirgum-" + k) ?? d; } catch { return d; } };
const setPref = (k, v) => { try { localStorage.setItem("tirgum-" + k, v); } catch {} };

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);
  return data;
}

// ---------------------------------------------------------------- estimate
function options() {
  return {
    translate_model: pref("model", settings.gemini?.set ? MODELS[0][0] : MODELS[1][0]),
    reading_speed: Number(pref("speed", 17)),
    captions: pref("captions", "auto"),
    box_color: pref("box", "white"),
    context: $("context")?.value || "",
  };
}

function scheduleEstimate() {
  clearTimeout(estimateTimer);
  const url = $("url").value.trim();
  if (!isYouTube(url)) { $("estimate").hidden = true; return; }
  estimateTimer = setTimeout(() => loadEstimate(url), 400);
}

async function loadEstimate(url) {
  const box = $("estimate");
  box.hidden = false;
  box.innerHTML = `<div class="loading"><span class="spinner"></span>Reading the video's details…</div>`;
  try {
    const est = await api("/api/estimate", { url, options: options() });
    if ($("url").value.trim() !== url) return;
    renderEstimate(est);
  } catch (e) {
    box.innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
}

function renderEstimate(est) {
  const o = options();
  const tokIn = Object.values(est.tokens).reduce((a, t) => a + t[0], 0);
  const tokOut = Object.values(est.tokens).reduce((a, t) => a + t[1], 0);
  const modelOpts = MODELS.map(([v, label]) => {
    const disabled = v.startsWith("gemini") ? !settings.gemini?.set : !settings.anthropic?.set;
    return `<option value="${v}" ${v === o.translate_model ? "selected" : ""} ${disabled ? "disabled" : ""}>${label}${disabled ? " — add key in Settings" : ""}</option>`;
  }).join("");
  $("estimate").innerHTML = `
    <div class="video-row">
      <img class="thumb" src="${esc(est.thumbnail)}" alt="" referrerpolicy="no-referrer">
      <div>
        <div class="video-title" dir="auto">${esc(est.title)}</div>
        <div class="video-meta" dir="auto">${esc(est.channel)}${est.channel ? " · " : ""}${length(est.duration)}</div>
      </div>
    </div>
    ${est.existing ? `<div class="notice">Already in your library. <a href="#" data-open="${esc(est.id)}">Open it</a>, or translate again below.</div>` : ""}
    <div class="stats">
      <div class="stat"><div class="k">Estimated cost</div><div class="v">${money(est.cost)}</div><div class="s">AI ${money(est.ai_cost)}${est.gpu ? ` · GPU ${money(est.gpu_cost)}` : ""}</div></div>
      <div class="stat"><div class="k">AI tokens</div><div class="v">${tokens(tokIn + tokOut)}</div><div class="s">${tokens(tokIn)} in · ${tokens(tokOut)} out</div></div>
      <div class="stat"><div class="k">Subtitles</div><div class="v">~${est.items}</div><div class="s">at ${o.reading_speed} chars/sec</div></div>
      <div class="stat"><div class="k">Time</div><div class="v">~${Math.max(1, Math.round(est.minutes))} min</div><div class="s">${est.gpu ? "GPU transcription" : "transcribing on this machine"}</div></div>
    </div>
    <div class="options">
      <label class="field">Translator<select id="model">${modelOpts}</select></label>
      <label class="field"><span>Reading speed: <b id="speedVal">${o.reading_speed}</b> chars/sec</span>
        <input id="speed" type="range" min="12" max="22" step="1" value="${o.reading_speed}"></label>
    </div>
    <details class="more"><summary>More options</summary>
      <div class="options">
        <label class="field">Hebrew captions in the video<select id="captions">
          ${[["auto", "Detect automatically"], ["on", "Yes, use them"], ["off", "No, translate speech"]].map(([v, l]) => `<option value="${v}" ${v === o.captions ? "selected" : ""}>${l}</option>`).join("")}
        </select></label>
        <label class="field">Subtitle box<select id="box">
          ${[["white", "White with dark text"], ["black", "Black with white text"]].map(([v, l]) => `<option value="${v}" ${v === o.box_color ? "selected" : ""}>${l}</option>`).join("")}
        </select></label>
        <label class="field">Context for the translator<input id="context" type="text" placeholder="e.g. comedy series, casual slang"></label>
      </div>
    </details>
    <div class="go">
      <button class="primary" id="start">Translate · ~${money(est.cost)}</button>
      <span class="hint">Estimate from the video's length; the real cost is shown when it finishes.</span>
    </div>`;
  $("model").onchange = (e) => { setPref("model", e.target.value); loadEstimate(est.url); };
  $("speed").oninput = (e) => { $("speedVal").textContent = e.target.value; };
  $("speed").onchange = (e) => { setPref("speed", e.target.value); loadEstimate(est.url); };
  $("captions").onchange = (e) => setPref("captions", e.target.value);
  $("box").onchange = (e) => setPref("box", e.target.value);
  $("start").onclick = () => startJob(est);
  $("estimate").querySelector("[data-open]")?.addEventListener("click", (e) => { e.preventDefault(); openViewer(est.id); });
}

// ---------------------------------------------------------------- jobs
async function startJob(est) {
  $("start").disabled = true;
  try {
    const job = await api("/api/jobs", { url: est.url, options: options(), estimate: est });
    current = { ...job, estimate: est };
    $("estimate").hidden = true;
    $("url").value = "";
    renderJob(current);
    poll();
  } catch (e) {
    $("start").disabled = false;
    alert(e.message);
  }
}

function poll() {
  clearTimeout(pollTimer);
  if (!current) return;
  pollTimer = setTimeout(async () => {
    try {
      current = { ...current, ...(await api(`/api/jobs/${current.id}`)) };
      renderJob(current);
      if (current.status === "done" || current.status === "failed") { loadLibrary(); return; }
    } catch {}
    poll();
  }, 1000);
}

function renderJob(job) {
  const box = $("job");
  box.hidden = false;
  const est = job.estimate || {};
  const elapsed = (job.finished || Date.now() / 1000) - (job.started || job.created);
  const pct = Math.round((job.progress || 0) * 100);
  const idx = STEPS.findIndex(([k]) => k === job.stage);
  const steps = STEPS.map(([k, label], i) => {
    const cls = job.status === "done" || i < idx ? "done" : i === idx && job.status === "running" ? "now" : "";
    return `<li class="${cls}">${label}</li>`;
  }).join("");
  const remaining = job.progress > 0.05 && job.status === "running" ? elapsed / job.progress - elapsed : null;
  let body = "";
  if (job.status === "done") {
    const r = job.record || {};
    body = `<div class="go">
        <button class="primary" data-watch="${esc(r.id)}">Watch</button>
        <a class="ghost" href="/api/library/${encodeURIComponent(r.id)}/video.en.mp4?download=true">Download video</a>
        <a class="ghost" href="/api/library/${encodeURIComponent(r.id)}/subtitles.en.srt?download=true">English .srt</a>
      </div>${costTable(r.usage, r.cost, est)}`;
  } else if (job.status === "failed") {
    body = `<div class="error">${esc(job.error || "Failed")}</div>`;
  }
  box.innerHTML = `
    <div class="video-row">
      ${est.thumbnail ? `<img class="thumb" src="${esc(est.thumbnail)}" alt="" referrerpolicy="no-referrer" style="width:140px">` : ""}
      <div>
        <div class="video-title" dir="auto">${esc(est.title || job.url)}</div>
        <div class="video-meta">${job.status === "done" ? `Finished in ${clock(elapsed)}` :
          job.status === "failed" ? "Failed" : job.status === "queued" ? "Waiting in queue" : esc(job.message || "")}</div>
      </div>
    </div>
    <div class="bar" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><i style="width:${pct}%"></i></div>
    <div class="bar-meta"><span>${pct}% · ${clock(elapsed)} elapsed${remaining ? ` · ~${clock(remaining)} left` : ""}</span>
      <span>AI cost so far ${money(job.cost ?? 0)}</span></div>
    <ol class="steps">${steps}</ol>
    ${body}`;
  box.querySelector("[data-watch]")?.addEventListener("click", (e) => openViewer(e.target.dataset.watch));
}

function costTable(usage, cost, est) {
  const models = [...new Set([...Object.keys(usage || {}), ...Object.keys(est?.tokens || {})])];
  if (!models.length) return "";
  const rows = models.map((m) => {
    const a = (usage || {})[m] || [0, 0], e = (est?.tokens || {})[m] || [0, 0];
    const price = settings.prices?.[m] || [0, 0];
    return `<tr><td>${esc(m)}</td><td class="num">${tokens(e[0])} / ${tokens(e[1])}</td>
      <td class="num">${tokens(a[0])} / ${tokens(a[1])}</td>
      <td class="num">${money(a[0] / 1e6 * price[0] + a[1] / 1e6 * price[1])}</td></tr>`;
  }).join("");
  return `<table class="compare"><thead><tr><th>Model</th><th class="num">Estimated tokens (in / out)</th>
    <th class="num">Actual tokens</th><th class="num">Actual cost</th></tr></thead><tbody>${rows}
    <tr><td><b>Total</b></td><td class="num">${money(est?.cost)} est.</td><td></td>
    <td class="num"><b>${money(cost?.total ?? cost)}</b>${cost?.gpu_estimated ? `<div class="hint">incl. ~${money(cost.gpu_estimated)} GPU</div>` : ""}</td></tr>
    </tbody></table>`;
}

// ---------------------------------------------------------------- library
async function loadLibrary() {
  try { library = await api("/api/library"); } catch { library = []; }
  renderLibrary();
}

function renderLibrary() {
  const q = $("libSearch").value.trim().toLowerCase();
  const items = library.filter((r) => !q || (r.title || "").toLowerCase().includes(q));
  $("library").innerHTML = items.map((r) => {
    const date = r.started ? new Date(r.started).toLocaleDateString(undefined, { day: "numeric", month: "short" }) : "";
    const cost = r.cost?.total ?? null;
    const badge = r.status && r.status !== "done" ? `<span class="badge ${esc(r.status)}">${esc(r.status)}</span>` : "";
    return `<button class="card" data-id="${esc(r.id)}">
      <img src="/api/library/${encodeURIComponent(r.id)}/thumb.jpg" alt="" loading="lazy">
      <div class="body">${badge}
        <div class="t" dir="auto">${esc(r.title)}</div>
        <div class="m">${[date, length(r.duration), cost != null ? money(cost) : ""].filter(Boolean).join(" · ")}</div>
      </div></button>`;
  }).join("") || `<div class="empty">${q ? "Nothing matches." : "Translated videos will appear here."}</div>`;
}

function openViewer(id) {
  const r = library.find((x) => x.id === id);
  if (!r) return;
  const has = (f) => (r.files || []).includes(f);
  const video = has("video.en.mp4") ? "video.en.mp4" : has("video.mp4") ? "video.mp4" : null;
  $("viewerBody").innerHTML = `
    <header style="display:flex;justify-content:space-between;gap:12px;align-items:start">
      <div><div class="video-title" dir="auto">${esc(r.title)}</div>
        <div class="video-meta">${[length(r.duration), r.translate_model, r.seconds ? `took ${clock(r.seconds)}` : ""].filter(Boolean).join(" · ")}</div></div>
      <button class="icon-btn" id="closeViewer" aria-label="Close">✕</button>
    </header>
    ${video ? `<video controls preload="metadata" src="/api/library/${encodeURIComponent(r.id)}/${video}"></video>` : `<div class="notice">No rendered video in this folder.</div>`}
    <div class="viewer-actions">
      ${["video.en.mp4", "subtitles.en.srt", "subtitles.he.srt"].filter(has).map((f) =>
        `<a class="ghost" href="/api/library/${encodeURIComponent(r.id)}/${f}?download=true">${{ "video.en.mp4": "Download video", "subtitles.en.srt": "English .srt", "subtitles.he.srt": "Hebrew .srt" }[f]}</a>`).join("")}
      ${r.url ? `<a class="ghost" href="${esc(r.url)}" target="_blank" rel="noopener">Open on YouTube</a>` : ""}
      ${has("video.en.mp4") ? `<button class="ghost" id="shareBtn">Share link</button>` : ""}
      <button class="danger" id="deleteBtn">Delete</button>
    </div>
    <div id="shareBox" class="share-box" hidden></div>
    ${r.error ? `<div class="error">${esc(r.error)}</div>` : ""}
    ${r.usage ? costTable(r.usage, r.cost, r.estimate) : r.legacy ? `<p class="hint">Translated before Tirgum tracked costs.</p>` : ""}`;
  $("closeViewer").onclick = () => $("viewer").close();
  $("deleteBtn").onclick = () => deleteVideo(r);
  if ($("shareBtn")) { $("shareBtn").onclick = () => createShare(r.id); loadShares(r.id); }
  $("viewer").showModal();
}

async function deleteVideo(r) {
  if (!confirm(`Delete "${r.title}"?\n\nThis permanently removes the video, its subtitles and its share links from disk.`)) return;
  try {
    await fetch(`/api/library/${encodeURIComponent(r.id)}`, { method: "DELETE" }).then(async (res) => {
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Delete failed");
    });
    $("viewer").close();
    loadLibrary();
  } catch (e) { alert(e.message); }
}

async function loadShares(id) {
  const shares = await api(`/api/library/${encodeURIComponent(id)}/shares`).catch(() => []);
  const box = $("shareBox");
  if (!shares.length) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = `<div>Anyone with one of these links can watch this video (no login):</div>` + shares.map((sh) => `
    <div class="share-row">
      <input readonly value="${esc(location.origin + sh.path)}" aria-label="Share link">
      <button class="linkbtn" data-copy="${esc(location.origin + sh.path)}">Copy</button>
      <span class="m">expires ${new Date(sh.expires * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" })}</span>
      <button class="linkbtn" data-revoke="${esc(sh.token)}">Revoke</button>
    </div>`).join("");
  box.querySelectorAll("[data-copy]").forEach((b) => b.onclick = async () => {
    try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = "Copied"; } catch { b.previousElementSibling.select(); }
  });
  box.querySelectorAll("[data-revoke]").forEach((b) => b.onclick = async () => {
    await fetch(`/api/shares/${encodeURIComponent(b.dataset.revoke)}`, { method: "DELETE" });
    loadShares(id);
  });
}

async function createShare(id) {
  try {
    const sh = await api(`/api/library/${encodeURIComponent(id)}/share`, { days: 7 });
    try { await navigator.clipboard.writeText(location.origin + sh.path); } catch {}
    await loadShares(id);
  } catch (e) { alert(e.message); }
}

// ---------------------------------------------------------------- browse
let browseKind = "videos", browsePlaylist = null;

async function loadBrowse() {
  const grid = $("browse");
  grid.innerHTML = `<div class="empty"><span class="loading" style="justify-content:center"><span class="spinner"></span>Loading from YouTube…</span></div>`;
  $("browseCrumb").hidden = !browsePlaylist;
  if (browsePlaylist) $("browseCrumb").innerHTML = `<button id="crumbBack">← Playlists</button> · <span dir="auto">${esc(browsePlaylist.title)}</span>`;
  $("crumbBack")?.addEventListener("click", () => { browsePlaylist = null; loadBrowse(); });
  const kind = browsePlaylist ? "videos" : browseKind;
  const qs = new URLSearchParams({ kind, ...(browsePlaylist ? { playlist: browsePlaylist.playlist } : {}) });
  try {
    const data = await api(`/api/browse?${qs}`);
    grid.innerHTML = data.items.map((it) => kind === "playlists"
      ? `<button class="card" data-playlist="${esc(it.playlist)}" data-title="${esc(it.title)}">
           <img src="${esc(it.thumbnail)}" alt="" loading="lazy" referrerpolicy="no-referrer">
           <div class="body"><div class="t" dir="auto">${esc(it.title)}</div>
           <div class="m">${it.count ? `${it.count} videos` : "Playlist"}</div></div></button>`
      : `<button class="card" data-url="${esc(it.url)}">
           <img src="${esc(it.thumbnail)}" alt="" loading="lazy" referrerpolicy="no-referrer">
           <div class="body">${it.in_library ? `<span class="badge done">In library</span>` : ""}
           <div class="t" dir="auto">${esc(it.title)}</div>
           <div class="m">${length(it.duration) || ""}</div></div></button>`).join("")
      || `<div class="empty">Nothing here.</div>`;
  } catch (e) {
    grid.innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
}

$("browse").addEventListener("click", (e) => {
  const card = e.target.closest(".card");
  if (!card) return;
  if (card.dataset.playlist) { browsePlaylist = { playlist: card.dataset.playlist, title: card.dataset.title }; loadBrowse(); return; }
  $("url").value = card.dataset.url;
  scheduleEstimate();
  window.scrollTo({ top: 0, behavior: "smooth" });
});
document.querySelectorAll(".tab").forEach((t) => t.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.setAttribute("aria-selected", x === t));
  browseKind = t.dataset.kind; browsePlaylist = null; loadBrowse();
});

$("viewer").addEventListener("close", () => $("viewer").querySelector("video")?.pause());
$("library").addEventListener("click", (e) => { const c = e.target.closest(".card"); if (c) openViewer(c.dataset.id); });
$("libSearch").addEventListener("input", renderLibrary);

// ---------------------------------------------------------------- settings
async function loadSettings() {
  try { settings = await api("/api/settings"); } catch { settings = {}; }
  document.querySelectorAll(".status[data-for]").forEach((el) => {
    const s = settings[el.dataset.for] || {};
    el.textContent = s.set ? `saved ${s.hint}` : "not set";
    el.classList.toggle("missing", !s.set);
  });
  if (!settings.anthropic?.set && !settings.gemini?.set) {
    $("urlHint").innerHTML = `Add an Anthropic or Gemini API key in <a href="#" id="hintSettings">Settings</a> to translate.`;
    $("hintSettings").onclick = (e) => { e.preventDefault(); $("settings").showModal(); };
  }
}

$("openSettings").onclick = () => { $("testResults").innerHTML = ""; $("settings").showModal(); };
$("settingsForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "save") return;
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target));
  await api("/api/settings", data);
  e.target.reset();
  await loadSettings();
  $("settings").close();
});
$("testKeys").onclick = async () => {
  $("testResults").innerHTML = `<span class="loading"><span class="spinner"></span>Testing…</span>`;
  const res = await api("/api/settings/test", {});
  $("testResults").innerHTML = Object.entries(res).map(([k, v]) =>
    `<span class="chip ${v === "ok" ? "ok" : "bad"}">${esc(k)}: ${esc(v)}</span>`).join("") || `<span class="chip">No keys saved yet</span>`;
};

// ---------------------------------------------------------------- start
$("url").addEventListener("input", scheduleEstimate);
$("urlForm").addEventListener("submit", (e) => { e.preventDefault(); scheduleEstimate(); });
$("pasteBtn").onclick = async () => {
  try { $("url").value = (await navigator.clipboard.readText()).trim(); scheduleEstimate(); }
  catch { $("url").focus(); }
};

(async () => {
  await loadSettings();
  loadLibrary();
  loadBrowse();
  const jobs = await api("/api/jobs").catch(() => []);
  const active = jobs.find((j) => j.status === "running" || j.status === "queued");
  if (active) { current = active; renderJob(current); poll(); }
})();
