// 車いすバスケ スタッツ・トラッキング フロントエンド(ビルドなしの ES module)

const $ = (sel) => document.querySelector(sel);
const SVG_NS = "http://www.w3.org/2000/svg";

const EVENT_LABELS = {
  made_shot: "シュート成功",
  missed_shot: "シュート失敗",
  free_throw: "フリースロー成功",
  rebound: "リバウンド",
  assist: "アシスト",
  turnover: "ターンオーバー",
};
const SOURCE_LABELS = {
  scoreboard: "得点表示", vision: "映像", "vision+scoreboard": "映像+得点表示",
  shot_clock: "ショットクロック", "vision+shot_clock": "映像+ショットクロック",
};
const STAT_COLS = [
  ["PTS", "得点"], ["FG", "FG"], ["3P", "3P"], ["FTM", "FT"], ["OREB", "OR"], ["DREB", "DR"], ["AST", "AST"], ["TOV", "TO"],
];

const state = {
  jobId: null,
  analysis: null,
  times: [],          // frames の時刻(二分探索用)
  tab: "live",
  eventFilter: new Set(Object.keys(EVENT_LABELS)),
  pollTimer: null,
};

// ---------------------------------------------------------------- API

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}

// ---------------------------------------------------------------- ジョブ選択・アップロード

async function loadJobs(selectId) {
  const { jobs } = await api("/api/jobs");
  const sel = $("#job-select");
  sel.innerHTML = "";
  for (const j of jobs) {
    const opt = document.createElement("option");
    opt.value = j.id;
    const tag = j.demo ? "デモ" : { queued: "待機中", running: "解析中", done: "完了", failed: "失敗" }[j.state];
    opt.textContent = `${j.name || j.id}(${tag})`;
    sel.append(opt);
  }
  // URL の #<job>@<秒> で、解析結果と再生位置を指定できる(場面の共有用)
  const [hashId, hashT] = location.hash.slice(1).split("@");
  const target = selectId || hashId || (jobs.find((j) => j.state === "done") || jobs[0] || {}).id;
  $("#empty").hidden = jobs.length > 0;
  if (target) {
    sel.value = target;
    await openJob(target);
    if (!selectId && hashT && state.analysis) seek(Number(hashT), false);
  }
}

async function openJob(id) {
  clearInterval(state.pollTimer);
  state.jobId = id;
  if (location.hash.slice(1).split("@")[0] !== id) history.replaceState(null, "", `#${id}`);
  const st = await api(`/api/jobs/${id}`);
  if (st.state === "done") {
    $("#job-progress").hidden = true;
    const analysis = await api(`/api/jobs/${id}/analysis`);
    showAnalysis(analysis);
  } else {
    $("#result").hidden = true;
    showProgress(st);
    if (st.state === "queued" || st.state === "running") {
      state.pollTimer = setInterval(async () => {
        const s = await api(`/api/jobs/${id}`);
        showProgress(s);
        if (s.state === "done" || s.state === "failed") {
          clearInterval(state.pollTimer);
          await loadJobs(id);
        }
      }, 2000);
    }
  }
}

function showProgress(st) {
  const box = $("#job-progress");
  box.hidden = false;
  const stages = { queued: "順番待ち", loading: "準備中", analyzing: "映像を解析中", events: "イベントを判定中", done: "完了" };
  const pct = Math.round((st.progress || 0) * 100);
  box.querySelector(".progress-label").textContent = st.state === "failed"
    ? `解析に失敗しました: ${st.error || ""}`
    : `${st.name}: ${stages[st.stage] || st.stage} ${pct}%`;
  box.querySelector(".fill").style.width = `${pct}%`;
}

async function setupUpload() {
  const dlg = $("#upload-dialog");
  const { profiles } = await api("/api/profiles");
  $("#profile-select").innerHTML = profiles.map((p) => `<option>${p}</option>`).join("");
  $("#upload-open").onclick = () => dlg.showModal();
  $("#upload-cancel").onclick = () => dlg.close();
  $("#upload-form").onsubmit = async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const btn = e.target.querySelector("button[type=submit]");
    btn.disabled = true;
    btn.textContent = "送信中…";
    try {
      const job = await api("/api/jobs", { method: "POST", body: fd });
      dlg.close();
      await loadJobs(job.id);
    } catch (err) {
      alert(`アップロードに失敗しました: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "アップロード";
    }
  };
  $("#job-select").onchange = (e) => openJob(e.target.value);
}

// ---------------------------------------------------------------- 結果の表示

function teamName(id) {
  const t = (state.analysis.teams || []).find((x) => x.id === id);
  return t ? t.name : id;
}

const teamColor = (id) => getComputedStyle(document.documentElement)
  .getPropertyValue(id === "a" ? "--team-a" : id === "b" ? "--team-b" : "--other").trim();

function fmtTime(t) {
  const m = Math.floor(t / 60);
  return `${m}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
}

function showAnalysis(a) {
  state.analysis = a;
  state.times = a.frames.map((f) => f.t);
  $("#result").hidden = false;
  document.querySelectorAll(".team-a-name").forEach((el) => { el.textContent = teamName("a"); });
  document.querySelectorAll(".team-b-name").forEach((el) => { el.textContent = teamName("b"); });

  const video = $("#video");
  video.src = `/api/jobs/${state.jobId}/video`;
  video.onerror = () => { $("#no-video").hidden = false; };
  video.onloadeddata = () => { $("#no-video").hidden = true; drawOverlay(); };

  renderStats();
  renderEventFilters();
  renderEvents();
  renderMeta();
  renderCourt();
}

function renderStats() {
  const { stats } = state.analysis;
  const head = `<thead><tr><th>チーム</th>${STAT_COLS.map(([, l]) => `<th>${l}</th>`).join("")}</tr></thead>`;
  const rows = Object.entries(stats).map(([id, s]) => {
    const cell = (k) => (k === "FG" ? `${s.FGM}/${s.FGA}` : k === "3P" ? `${s["3PM"]}/${s["3PA"]}` : s[k]);
    return `<tr><td><span class="team-dot" style="background:${teamColor(id)}"></span>${teamName(id)}</td>`
      + `${STAT_COLS.map(([k]) => `<td>${cell(k)}</td>`).join("")}</tr>`;
  }).join("");
  $("#stats-table").innerHTML = head + `<tbody>${rows}</tbody>`;
  $("#stats-note").textContent = "得点はスコアボードの読み取り、それ以外は映像からの推定。FT は成功のみ(失敗は検出対象外)。";
}

function renderEventFilters() {
  const box = $("#event-filters");
  box.innerHTML = "";
  for (const [k, label] of Object.entries(EVENT_LABELS)) {
    const b = document.createElement("button");
    b.textContent = label;
    b.className = state.eventFilter.has(k) ? "" : "off";
    b.onclick = () => {
      if (state.eventFilter.has(k)) state.eventFilter.delete(k); else state.eventFilter.add(k);
      renderEventFilters();
      renderEvents();
    };
    box.append(b);
  }
}

function renderEvents() {
  const list = $("#event-list");
  const evs = state.analysis.events.filter((e) => state.eventFilter.has(e.type));
  $("#event-count").textContent = `(${evs.length})`;
  list.innerHTML = "";
  for (const e of evs) {
    const li = document.createElement("li");
    li.dataset.t = e.t;
    let kind = EVENT_LABELS[e.type] || e.type;
    if (e.type === "made_shot") kind += ` ${e.points}P`;
    if (e.type === "rebound") kind += e.offensive ? "(OR)" : "(DR)";
    li.innerHTML = `<span class="t">${fmtTime(e.t)}</span>`
      + `<span><span class="team-dot" style="background:${teamColor(e.team)}"></span><span class="kind">${kind}</span> `
      + `${teamName(e.team)}${e.player != null ? ` <span class="src">#ID${e.player % 10000}</span>` : ""}</span>`
      + `<span class="src">${SOURCE_LABELS[e.source] || e.source}</span>`;
    li.onclick = () => seek(e.t - 2);
    list.append(li);
  }
}

function renderMeta() {
  const m = state.analysis.meta;
  const wide = state.analysis.segments.filter((s) => s.type === "wide").reduce((a, s) => a + (s.end - s.start), 0);
  const rows = [
    ["動画", m.video],
    ["長さ", `${m.duration.toFixed(1)} 秒(広角ショット ${wide.toFixed(1)} 秒)`],
    ["解析フレーム", `${state.analysis.frames.length}(${(m.fps / m.stride).toFixed(1)} fps)`],
    ["検出モデル", `${m.model}(${(m.detector_classes || []).join(", ")})`],
    ["会場プロファイル", m.profile],
    ["処理時間", `${m.processing_sec} 秒`],
  ];
  $("#meta").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("#dl-stats").href = `/api/jobs/${state.jobId}/stats.csv`;
  $("#dl-events").href = `/api/jobs/${state.jobId}/events.csv`;
  $("#dl-json").href = `/api/jobs/${state.jobId}/analysis`;
}

// ---------------------------------------------------------------- 動画とオーバーレイ

function seek(t, play = true) {
  const v = $("#video");
  const target = Math.max(0, t);
  const apply = () => {
    v.currentTime = target;
    if (play) v.play().catch(() => {});
  };
  if (v.readyState >= 1) apply(); else v.addEventListener("loadedmetadata", apply, { once: true });
  history.replaceState(null, "", `#${state.jobId}@${target.toFixed(1)}`);
  syncToTime(target);
}

function frameAt(t) {
  const ts = state.times;
  if (!ts.length) return null;
  let lo = 0;
  let hi = ts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] < t) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(ts[lo - 1] - t) < Math.abs(ts[lo] - t)) lo -= 1;
  return Math.abs(ts[lo] - t) < 0.5 ? state.analysis.frames[lo] : null;
}

function drawOverlay() {
  const video = $("#video");
  const canvas = $("#overlay");
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const fr = frameAt(video.currentTime);
  if (!$("#show-overlay").checked || !fr || !fr.wide || !video.videoWidth) return;

  // 解析座標(帯を除いてリサイズした画像)→ 元動画の座標 → 表示座標
  const m = state.analysis.meta;
  const [cx0, cy0, cx1, cy1] = m.crop;
  const [fw, fh] = m.frame_size || [cx1 - cx0, cy1 - cy0];
  const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
  const offX = (rect.width - video.videoWidth * scale) / 2;
  const offY = (rect.height - video.videoHeight * scale) / 2;
  const d = devicePixelRatio;
  const X = (x) => (offX + (cx0 + x * (cx1 - cx0) / fw) * scale) * d;
  const Y = (y) => (offY + (cy0 + y * (cy1 - cy0) / fh) * scale) * d;

  ctx.lineWidth = 2 * d;
  ctx.font = `${11 * d}px system-ui`;
  for (const p of fr.players) {
    const [x1, y1, x2, y2] = p.box;
    ctx.strokeStyle = teamColor(p.team);
    ctx.strokeRect(X(x1), Y(y1), X(x2) - X(x1), Y(y2) - Y(y1));
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(String(p.id % 10000), X(x1), Y(y1) - 3 * d);
  }
  if (fr.ball) {
    ctx.strokeStyle = "#ffd400";
    ctx.setLineDash(fr.ball.interpolated ? [4 * d, 3 * d] : []);
    ctx.beginPath();
    ctx.arc(X(fr.ball.xy[0]), Y(fr.ball.xy[1]), 9 * d, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const [rx, ry] of fr.rims || []) {
    ctx.strokeStyle = "rgba(255,120,0,0.8)";
    ctx.beginPath();
    ctx.arc(X(rx), Y(ry), 6 * d, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function syncToTime(t) {
  const sb = state.analysis.scoreboard;
  let cur = null;
  for (const r of sb) { if (r.t <= t) cur = r; else break; }
  $("#clock").textContent = cur
    ? `${teamName("a")} ${cur.score_a ?? "-"} - ${cur.score_b ?? "-"} ${teamName("b")}   試合時計 ${cur.clock ?? "--:--"}`
    : "";
  document.querySelectorAll("#event-list li").forEach((li) => {
    li.classList.toggle("current", Math.abs(Number(li.dataset.t) - t) < 1.5);
  });
  if (state.tab === "live") drawLive(t);
}

function startVideoLoop() {
  const video = $("#video");
  const tick = () => {
    if (state.analysis) {
      drawOverlay();
      syncToTime(video.currentTime);
    }
    if (!video.paused) requestAnimationFrame(tick);
  };
  video.addEventListener("play", () => requestAnimationFrame(tick));
  video.addEventListener("seeked", tick);
  $("#show-overlay").onchange = drawOverlay;
  window.addEventListener("resize", drawOverlay);
}

// ---------------------------------------------------------------- コート図

const COURT = { L: 28, W: 15, key: 4.9, ft: 5.8, basket: 1.575, three: 6.75, threeSide: 6.6, circle: 1.8 };

function el(tag, attrs, parent) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}

function arcPath(cx, cy, r, a0, a1) {
  const p = (a) => `${cx + r * Math.cos(a)} ${cy + r * Math.sin(a)}`;
  const large = Math.abs(a1 - a0) > Math.PI ? 1 : 0;
  return `M ${p(a0)} A ${r} ${r} 0 ${large} 1 ${p(a1)}`;
}

function drawCourtBase(svg) {
  svg.innerHTML = "";
  const hl = COURT.L / 2;
  const hw = COURT.W / 2;
  const line = { fill: "none", stroke: "var(--court-line)", "stroke-width": 0.08 };
  el("rect", { x: -hl, y: -hw, width: COURT.L, height: COURT.W, ...line, fill: "var(--court-floor)" }, svg);
  el("line", { x1: 0, y1: -hw, x2: 0, y2: hw, ...line }, svg);
  el("circle", { cx: 0, cy: 0, r: COURT.circle, ...line }, svg);
  for (const s of [-1, 1]) {
    const bx = s * hl;
    const ftx = s * (hl - COURT.ft);
    const k = COURT.key / 2;
    el("rect", { x: Math.min(bx, ftx), y: -k, width: COURT.ft, height: COURT.key, ...line, fill: "var(--court-paint)" }, svg);
    const a0 = s > 0 ? Math.PI / 2 : -Math.PI / 2;
    el("path", { d: arcPath(ftx, 0, COURT.circle, a0, a0 + Math.PI), ...line }, svg);
    const cx = s * (hl - COURT.basket);
    const th = Math.asin(COURT.threeSide / COURT.three);
    const xj = cx - s * COURT.three * Math.cos(th);
    el("line", { x1: bx, y1: -COURT.threeSide, x2: xj, y2: -COURT.threeSide, ...line }, svg);
    el("line", { x1: bx, y1: COURT.threeSide, x2: xj, y2: COURT.threeSide, ...line }, svg);
    // 円弧はリング中心からコート中央側に張り出す(右ゴールは角度 π 付近、左ゴールは 0 付近)
    const a = s > 0 ? Math.PI - th : -th;
    el("path", { d: arcPath(cx, 0, COURT.three, a, a + 2 * th), ...line }, svg);
    el("circle", { cx, cy: 0, r: 0.23, fill: "none", stroke: "#ff7a1a", "stroke-width": 0.06 }, svg);
  }
  return el("g", {}, svg);
}

function renderCourt() {
  $("#heat-opts").hidden = state.tab !== "heat";
  $("#spacing-table").hidden = state.tab !== "spacing";
  const layer = drawCourtBase($("#court"));
  const legend = $("#court-legend");
  if (state.tab === "shots") {
    drawShots(layer);
    legend.textContent = "● 成功 / × 失敗。位置はシュート直前にボールを持っていた選手の位置(追跡できた場合のみ)。";
  } else if (state.tab === "heat") {
    drawHeat(layer, document.querySelector("input[name=heat-team]:checked").value);
    legend.textContent = "広角ショット中に選手がいた位置の分布(0.5m 四方ごとの滞在フレーム数)。";
  } else if (state.tab === "spacing") {
    renderSpacing(layer);
    legend.textContent = "攻撃区間ごとの、攻撃側選手の凸包面積と選手間の平均距離。行をクリックするとその場面の配置を表示。";
  } else {
    drawLive($("#video").currentTime || 0);
    legend.textContent = "動画の再生位置に合わせた選手の位置(コート座標)。";
  }
}

function drawShots(layer) {
  for (const e of state.analysis.events) {
    if (!["made_shot", "missed_shot"].includes(e.type) || !e.court) continue;
    const [x, y] = e.court;
    const c = teamColor(e.team);
    if (e.type === "made_shot") {
      el("circle", { cx: x, cy: y, r: 0.35, fill: c, stroke: "#fff", "stroke-width": 0.05 }, layer);
    } else {
      const s = 0.3;
      el("path", { d: `M ${x - s} ${y - s} L ${x + s} ${y + s} M ${x - s} ${y + s} L ${x + s} ${y - s}`, stroke: c, "stroke-width": 0.12 }, layer);
    }
    const g = el("title", {}, layer);
    g.textContent = `${fmtTime(e.t)} ${teamName(e.team)} ${EVENT_LABELS[e.type]}`;
  }
}

function drawHeat(layer, team) {
  const cell = 0.5;
  const nx = COURT.L / cell;
  const ny = COURT.W / cell;
  const grid = new Float32Array(nx * ny);
  for (const f of state.analysis.frames) {
    for (const p of f.players) {
      if (p.team !== team || !p.court) continue;
      const i = Math.floor((p.court[0] + COURT.L / 2) / cell);
      const j = Math.floor((p.court[1] + COURT.W / 2) / cell);
      if (i >= 0 && i < nx && j >= 0 && j < ny) grid[j * nx + i] += 1;
    }
  }
  // 3x3 の平滑化で見やすくする
  const sm = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      let s = 0;
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) {
          const ii = i + di;
          const jj = j + dj;
          if (ii >= 0 && ii < nx && jj >= 0 && jj < ny) s += grid[jj * nx + ii] * (di === 0 && dj === 0 ? 2 : 1);
        }
      }
      sm[j * nx + i] = s;
    }
  }
  const max = Math.max(...sm, 1);
  const color = teamColor(team);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = sm[j * nx + i] / max;
      if (v < 0.03) continue;
      el("rect", {
        x: -COURT.L / 2 + i * cell, y: -COURT.W / 2 + j * cell, width: cell, height: cell,
        fill: color, "fill-opacity": (0.1 + 0.8 * v).toFixed(2),
      }, layer);
    }
  }
}

function drawPlayers(layer, fr, onlyTeam) {
  for (const p of fr.players) {
    if (!p.court || (onlyTeam && p.team !== onlyTeam)) continue;
    el("circle", { cx: p.court[0], cy: p.court[1], r: 0.32, fill: teamColor(p.team), stroke: "#fff", "stroke-width": 0.05 }, layer);
  }
}

function drawLive(t) {
  if (state.tab !== "live") return;
  const layer = drawCourtBase($("#court"));
  const fr = frameAt(t);
  if (fr && fr.wide) drawPlayers(layer, fr);
}

function convexHull(pts) {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), q) <= 0) lower.pop(); lower.push(q); }
  const upper = [];
  for (const q of p.reverse()) { while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function renderSpacing(layer) {
  const poss = state.analysis.possessions || [];
  const rows = poss.map((p, i) => {
    const sp = p.spacing;
    return `<tr class="clickable" data-i="${i}"><td><span class="team-dot" style="background:${teamColor(p.team)}"></span>${teamName(p.team)}</td>`
      + `<td>${fmtTime(p.start)}–${fmtTime(p.end)}</td><td>${p.half_court ? "ハーフコート" : "–"}</td>`
      + `<td>${sp ? sp.hull_area_m2.toFixed(1) : "–"}</td><td>${sp ? sp.mean_pair_dist_m.toFixed(1) : "–"}</td></tr>`;
  }).join("");
  $("#spacing-table").innerHTML = poss.length
    ? `<table><thead><tr><th>攻撃</th><th>時間</th><th>局面</th><th>凸包面積 m²</th><th>平均距離 m</th></tr></thead><tbody>${rows}</tbody></table>`
    : `<p class="note">攻撃区間が検出されていません(ボール保持の推定にはボールの検出が必要です)。</p>`;
  document.querySelectorAll("#spacing-table tr.clickable").forEach((tr) => {
    tr.onclick = () => {
      const p = poss[Number(tr.dataset.i)];
      const fr = frameAt((p.start + p.end) / 2);
      const g = drawCourtBase($("#court"));
      if (!fr) return;
      const pts = fr.players.filter((q) => q.team === p.team && q.court).map((q) => q.court);
      if (pts.length >= 3) {
        el("polygon", { points: convexHull(pts).map((q) => q.join(",")).join(" "), fill: teamColor(p.team),
          "fill-opacity": 0.2, stroke: teamColor(p.team), "stroke-width": 0.06 }, g);
      }
      drawPlayers(g, fr);
      seek(p.start);
    };
  });
}

function setupTabs() {
  document.querySelectorAll(".tabs button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
      state.tab = b.dataset.tab;
      renderCourt();
    };
  });
  document.querySelectorAll("input[name=heat-team]").forEach((r) => { r.onchange = renderCourt; });
}

// ---------------------------------------------------------------- 起動

setupTabs();
startVideoLoop();
setupUpload().then(() => loadJobs()).catch((e) => {
  $("#empty").hidden = false;
  $("#empty").textContent = `サーバーに接続できません: ${e.message}`;
});
