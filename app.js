import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---------- Config ----------
const STORE_KEY = "queercoded.templates.v1";
const FIXED_LEN = 20;             // frames every gesture is resampled to
const KEY_LMS = [11, 12, 23, 24]; // shoulders + hips: visibility gate
const NUM_LMS = 33;

// A gesture is delimited by the RESTING pose (standing, arms at sides).
const MOVE_TRIGGER_FRAMES = 2;    // frames of "not rest" that start a capture
const REST_SETTLE_FRAMES = 5;     // frames of "rest" that end a capture
const MAX_SEG_MS = 6000;          // abandon a capture that never returns to rest
const MIN_SEG_FRAMES = 4;         // ignore too-short blips
const COOLDOWN_MS = 1200;         // min gap before the same word fires again

// ---------- DOM ----------
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const bigWord = document.getElementById("bigWord");
const countdownEl = document.getElementById("countdown");

const threshInput = document.getElementById("thresh");
const threshVal = document.getElementById("threshVal");
const bestWordEl = document.getElementById("bestWord");
const bestDistEl = document.getElementById("bestDist");
const barEl = document.getElementById("bar");
const phraseEl = document.getElementById("phrase");
const clearPhraseBtn = document.getElementById("clearPhrase");

const wordInput = document.getElementById("wordInput");
const durInput = document.getElementById("dur");
const durVal = document.getElementById("durVal");
const recordBtn = document.getElementById("recordBtn");
const teachMsg = document.getElementById("teachMsg");

const codeList = document.getElementById("codeList");
const exportBtn = document.getElementById("exportBtn");
const importBtn = document.getElementById("importBtn");
const importFile = document.getElementById("importFile");
const clearAllBtn = document.getElementById("clearAll");

// ---------- State ----------
let landmarker = null;
let templates = loadTemplates();
let recording = null; // {frames, rest, endsAt, word, durMs}
let seg = newSeg();   // live segmentation state
let lastFireAt = 0;
let lastFiredWord = "";
let phrase = [];
let du = null;

function newSeg() {
  return { state: "rest", frames: [], rest: [], restCount: 0, moveCount: 0, startedAt: 0 };
}

// ---------- Geometry ----------
// Normalize landmarks to a flat [x,y,...] vector invariant to position and
// apparent size: center on hip midpoint, scale by torso length.
function normalizePose(lms) {
  const lHip = lms[23], rHip = lms[24], lSho = lms[11], rSho = lms[12];
  const cx = (lHip.x + rHip.x) / 2;
  const cy = (lHip.y + rHip.y) / 2;
  const shx = (lSho.x + rSho.x) / 2;
  const shy = (lSho.y + rSho.y) / 2;
  const torso = Math.hypot(shx - cx, shy - cy) || 1e-6;
  const out = new Array(NUM_LMS * 2);
  for (let i = 0; i < NUM_LMS; i++) {
    out[i * 2] = (lms[i].x - cx) / torso;
    out[i * 2 + 1] = (lms[i].y - cy) / torso;
  }
  return out;
}

function keyLandmarksVisible(lms) {
  return KEY_LMS.every((i) => (lms[i]?.visibility ?? 0) > 0.5);
}

// "Resting" = standing with hands by the sides: both wrists at/below hip
// height and horizontally close to the body (not extended outward).
function isResting(lms) {
  const lw = lms[15], rw = lms[16], lh = lms[23], rh = lms[24], ls = lms[11], rs = lms[12];
  if ((lw?.visibility ?? 0) < 0.5 || (rw?.visibility ?? 0) < 0.5) return false;
  const shoulderW = Math.hypot(ls.x - rs.x, ls.y - rs.y) || 1e-6;
  const wristsLow = lw.y > lh.y - 0.05 && rw.y > rh.y - 0.05; // y grows downward
  const wristsNarrow =
    Math.abs(lw.x - lh.x) < shoulderW * 0.8 && Math.abs(rw.x - rh.x) < shoulderW * 0.8;
  return wristsLow && wristsNarrow;
}

// Mean per-landmark euclidean distance between two normalized pose vectors.
function poseDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 2) s += Math.hypot(a[i] - b[i], a[i + 1] - b[i + 1]);
  return s / (a.length / 2);
}

// Resample a sequence of pose vectors to exactly L frames (linear interp).
function resampleSeq(frames, L) {
  if (frames.length === 0) return [];
  if (frames.length === 1) return Array.from({ length: L }, () => frames[0].slice());
  const out = [];
  for (let i = 0; i < L; i++) {
    const t = (i * (frames.length - 1)) / (L - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, frames.length - 1);
    const f = t - lo;
    const a = frames[lo], b = frames[hi];
    const v = new Array(a.length);
    for (let k = 0; k < a.length; k++) v[k] = a[k] * (1 - f) + b[k] * f;
    out.push(v);
  }
  return out;
}

// Dynamic Time Warping between two equal-length pose sequences, normalized by
// path length so scores are comparable. Robust to differences in tempo.
function dtw(A, B) {
  const m = A.length, n = B.length;
  const D = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(Infinity));
  D[0][0] = 0;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const c = poseDist(A[i - 1], B[j - 1]);
      D[i][j] = c + Math.min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]);
    }
  }
  return D[m][n] / (m + n);
}

// ---------- Persistence (localStorage) ----------
function loadTemplates() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; }
  catch { return []; }
}
function saveTemplates() { localStorage.setItem(STORE_KEY, JSON.stringify(templates)); }
function newId() { return "c_" + Math.random().toString(36).slice(2, 9) + performance.now().toString(36); }

// ---------- MediaPipe ----------
async function initPose() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

// ---------- Main loop ----------
function loop() {
  if (landmarker && video.readyState >= 2) {
    const res = landmarker.detectForVideo(video, performance.now());
    octx.clearRect(0, 0, overlay.width, overlay.height);

    if (res.landmarks && res.landmarks.length > 0) {
      const lms = res.landmarks[0];
      if (!du) du = new DrawingUtils(octx);
      du.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS, { color: "rgba(123,92,255,0.9)", lineWidth: 3 });
      du.drawLandmarks(lms, { color: "#ff4d9d", radius: 3, lineWidth: 1 });

      if (keyLandmarksVisible(lms)) {
        const vec = normalizePose(lms);
        const rest = isResting(lms);
        const now = performance.now();
        if (recording) {
          recording.frames.push(vec);
          recording.rest.push(rest);
          if (now >= recording.endsAt) finishRecording();
        } else {
          segmentStep(vec, rest, now);
        }
      }
    }
  }
  requestAnimationFrame(loop);
}

// ---------- Rest-delimited segmentation ----------
function segmentStep(vec, rest, now) {
  if (seg.state === "rest") {
    if (!rest) {
      seg.moveCount++;
      if (seg.moveCount >= MOVE_TRIGGER_FRAMES) {
        seg.state = "move";
        seg.frames = [vec];
        seg.rest = [rest];
        seg.startedAt = now;
        seg.restCount = 0;
      }
    } else {
      seg.moveCount = 0;
    }
    setPerformState("rest");
    return;
  }

  // state === "move"
  seg.frames.push(vec);
  seg.rest.push(rest);
  if (rest) {
    seg.restCount++;
    if (seg.restCount >= REST_SETTLE_FRAMES) { closeSegment(now); return; }
  } else {
    seg.restCount = 0;
  }
  if (now - seg.startedAt > MAX_SEG_MS) { seg = newSeg(); setPerformState("rest"); return; }
  setPerformState("move");
}

function closeSegment(now) {
  // Drop the trailing rest frames (the settle) so the gesture ends where the
  // movement ended, not after standing still.
  let end = seg.frames.length;
  while (end > 0 && seg.rest[end - 1]) end--;
  const frames = seg.frames.slice(0, end);
  seg = newSeg();
  setPerformState("rest");
  if (frames.length >= MIN_SEG_FRAMES) matchAndFire(frames, now);
}

function matchAndFire(frames, now) {
  if (templates.length === 0) { bestWordEl.textContent = "no codes yet"; return; }
  const live = resampleSeq(frames, FIXED_LEN);
  const thresh = parseFloat(threshInput.value);
  let best = { word: null, dist: Infinity };
  for (const t of templates) {
    const d = dtw(live, t.seq);
    if (d < best.dist) best = { word: t.word, dist: d };
  }
  bestWordEl.textContent = best.word ?? "—";
  bestDistEl.textContent = isFinite(best.dist) ? best.dist.toFixed(3) : "—";
  const pct = isFinite(best.dist) ? Math.max(0, Math.min(100, (1 - best.dist / thresh) * 100)) : 0;
  barEl.style.width = pct + "%";
  if (best.dist < thresh) {
    const ok = best.word !== lastFiredWord || now - lastFireAt > COOLDOWN_MS;
    if (ok) fireWord(best.word, now);
  }
}

function setPerformState(s) {
  if (!landmarker) return;
  statusEl.textContent = s === "move" ? "● reading movement…" : "● resting — strike a code";
}

function fireWord(word, now) {
  lastFireAt = now;
  lastFiredWord = word;
  showBigWord(word);
  phrase.push(word);
  renderPhrase();
}

let bigWordTimer = null;
function showBigWord(word) {
  bigWord.innerHTML = `<b>${escapeHtml(word)}</b>`;
  bigWord.classList.add("show");
  clearTimeout(bigWordTimer);
  bigWordTimer = setTimeout(() => bigWord.classList.remove("show"), 1200);
}

function renderPhrase() {
  phraseEl.innerHTML = phrase.length === 0
    ? '<span class="muted">Your matched words appear here…</span>'
    : phrase.map((w) => `<span class="chip">${escapeHtml(w)}</span>`).join("");
}

// ---------- Teaching ----------
async function startRecording() {
  const word = wordInput.value.trim();
  if (!word) { setTeachMsg("Type a word or phrase first.", "warn"); return; }
  if (recording) return;
  recordBtn.disabled = true;

  countdownEl.hidden = false;
  countdownEl.textContent = "rest";
  countdownEl.classList.remove("rec");
  await sleep(600);
  for (let i = 3; i >= 1; i--) { countdownEl.textContent = i; await sleep(650); }
  countdownEl.textContent = "GO";
  countdownEl.classList.add("rec");

  const durMs = parseFloat(durInput.value) * 1000;
  recording = { frames: [], rest: [], endsAt: performance.now() + durMs, word, durMs };
  setTeachMsg("Recording… perform the movement, then return to rest.", "");
}

function finishRecording() {
  const { frames, rest, word, durMs } = recording;
  recording = null;
  countdownEl.hidden = true;
  recordBtn.disabled = false;

  // Trim leading and trailing resting frames so every code starts and ends at rest.
  let a = 0, b = frames.length;
  while (a < b && rest[a]) a++;
  while (b > a && rest[b - 1]) b--;
  const core = frames.slice(a, b);

  if (core.length < 3) {
    setTeachMsg("No movement detected between rest poses. Start at rest (arms at sides), perform the movement, then return to rest.", "warn");
    return;
  }
  const seq = resampleSeq(core, FIXED_LEN);
  templates.push({ id: newId(), word, seq, durMs, createdAt: Date.now() });
  saveTemplates();
  renderCodeList();
  setTeachMsg(`Saved “${word}”. Switch to Perform to try it.`, "ok");
  wordInput.value = "";
}

// ---------- Codes list ----------
function renderCodeList() {
  if (templates.length === 0) {
    codeList.innerHTML = '<li class="empty">No codes saved yet. Go to Teach to make one.</li>';
    return;
  }
  codeList.innerHTML = "";
  for (const t of templates) {
    const li = document.createElement("li");
    li.className = "code-item";
    li.innerHTML = `
      <div>
        <div class="word">${escapeHtml(t.word)}</div>
        <div class="meta">${(t.durMs / 1000).toFixed(1)}s · ${new Date(t.createdAt).toLocaleDateString()}</div>
      </div>
      <div class="row-actions">
        <button class="btn small" data-act="rename" data-id="${t.id}">Rename</button>
        <button class="btn small danger" data-act="del" data-id="${t.id}">Delete</button>
      </div>`;
    codeList.appendChild(li);
  }
}

codeList.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  const { act, id } = btn.dataset;
  if (act === "del") {
    templates = templates.filter((t) => t.id !== id);
    saveTemplates();
    renderCodeList();
  } else if (act === "rename") {
    const t = templates.find((x) => x.id === id);
    const name = prompt("New word or phrase:", t.word);
    if (name && name.trim()) { t.word = name.trim(); saveTemplates(); renderCodeList(); }
  }
});

// ---------- Export / import ----------
exportBtn.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(templates, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "queercoded-codes.json"; a.click();
  URL.revokeObjectURL(url);
});
importBtn.addEventListener("click", () => importFile.click());
importFile.addEventListener("change", async () => {
  const file = importFile.files[0];
  if (!file) return;
  try {
    const incoming = JSON.parse(await file.text());
    if (!Array.isArray(incoming)) throw new Error("bad format");
    let added = 0;
    for (const t of incoming) {
      if (t && t.word && Array.isArray(t.seq)) { templates.push({ ...t, id: newId() }); added++; }
    }
    saveTemplates();
    renderCodeList();
    alert(`Imported ${added} code(s).`);
  } catch { alert("Could not read that file."); }
  importFile.value = "";
});
clearAllBtn.addEventListener("click", () => {
  if (confirm("Delete all saved codes? This cannot be undone.")) {
    templates = []; saveTemplates(); renderCodeList();
  }
});

// ---------- UI wiring ----------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
    tab.classList.add("is-active");
    const name = tab.dataset.tab;
    document.querySelectorAll(".tabpane").forEach((p) => { p.hidden = p.dataset.pane !== name; });
  });
});
threshInput.addEventListener("input", () => (threshVal.textContent = threshInput.value));
durInput.addEventListener("input", () => (durVal.textContent = parseFloat(durInput.value).toFixed(1) + "s"));
recordBtn.addEventListener("click", startRecording);
clearPhraseBtn.addEventListener("click", () => { phrase = []; renderPhrase(); });

// ---------- Helpers ----------
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
function setTeachMsg(msg, cls) { teachMsg.textContent = msg; teachMsg.className = "teach-msg " + (cls || "muted"); }
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- Boot ----------
(async function boot() {
  threshVal.textContent = threshInput.value;
  durVal.textContent = parseFloat(durInput.value).toFixed(1) + "s";
  renderCodeList();
  renderPhrase();
  try {
    statusEl.textContent = "Requesting camera…";
    await initCamera();
    statusEl.textContent = "Loading pose model…";
    await initPose();
    setPerformState("rest");
    loop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error: " + err.message + " (camera needs https or localhost)";
  }
})();
