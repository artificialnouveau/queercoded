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
const MAX_TEACH_MS = 8000;        // safety cap so a teaching capture can't hang
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
const triggerModeSel = document.getElementById("triggerMode");
const holdBtn = document.getElementById("holdBtn");
const soundToggle = document.getElementById("soundToggle");
const wordList = document.getElementById("wordList");

const wordInput = document.getElementById("wordInput");
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
let teach = null;     // active teaching capture (movement-delimited)
let seg = newSeg();   // live segmentation state
let lastFireAt = 0;
let lastFiredWord = "";
let phrase = [];
let du = null;

// Perform options / manual capture
let triggerMode = "auto";     // "auto" (rest pose) | "manual" (hold)
let soundOn = true;
let manualCapturing = false;
let manualFrames = [];
let audioCtx = null;

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
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

async function initPose() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  // GPU is faster but silently fails/hangs on some machines; fall back to CPU.
  try {
    landmarker = await PoseLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch (e) {
    console.warn("Pose GPU delegate failed, falling back to CPU:", e);
    landmarker = await PoseLandmarker.createFromOptions(fileset, opts("CPU"));
  }
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
    const now = performance.now();
    const res = landmarker.detectForVideo(video, now);
    octx.clearRect(0, 0, overlay.width, overlay.height);

    if (res.landmarks && res.landmarks.length > 0) {
      const lms = res.landmarks[0];
      if (!du) du = new DrawingUtils(octx);
      du.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS, { color: "rgba(123,92,255,0.9)", lineWidth: 3 });
      du.drawLandmarks(lms, { color: "#ff4d9d", radius: 3, lineWidth: 1 });

      if (keyLandmarksVisible(lms)) {
        const vec = normalizePose(lms);
        const rest = isResting(lms);
        if (teach) {
          teachStep(vec, rest, now);
        } else if (triggerMode === "manual") {
          if (manualCapturing) manualFrames.push(vec);
        } else {
          segmentStep(vec, rest, now);
        }
      }
    }

    // Safety cap so a teaching capture can never hang, even if the body leaves
    // the frame mid-movement.
    if (teach && teach.state === "capturing" && now - teach.startedAt > MAX_TEACH_MS) {
      finishTeach(now);
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
  if (s === "move") { statusEl.textContent = "● capturing movement…"; return; }
  statusEl.textContent = triggerMode === "manual" ? "● ready — hold to capture" : "● resting — perform a code";
}

function fireWord(word, now) {
  lastFireAt = now;
  lastFiredWord = word;
  showBigWord(word);
  ping();
  phrase.push(word);
  renderPhrase();
}

// Short confirmation tone on a successful match.
function ping() {
  if (!soundOn) return;
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
    const t = audioCtx.currentTime;
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.type = "sine";
    o.frequency.setValueAtTime(880, t);
    o.frequency.exponentialRampToValueAtTime(1320, t + 0.12);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.25);
    o.connect(g).connect(audioCtx.destination);
    o.start(t);
    o.stop(t + 0.26);
  } catch {}
}

// ---------- Manual capture (accessibility fallback) ----------
function startManual() {
  if (teach || triggerMode !== "manual" || manualCapturing) return;
  manualCapturing = true;
  manualFrames = [];
  setPerformState("move");
  holdBtn.classList.add("recording");
}
function stopManual() {
  if (!manualCapturing) return;
  manualCapturing = false;
  holdBtn.classList.remove("recording");
  setPerformState("rest");
  const frames = manualFrames;
  manualFrames = [];
  if (frames.length >= MIN_SEG_FRAMES) matchAndFire(frames, performance.now());
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

// ---------- Teaching (movement-delimited, like Perform) ----------
async function startTeach() {
  const word = wordInput.value.trim();
  if (!word) { setTeachMsg("Type a word or phrase first.", "warn"); return; }
  if (teach) return;

  recordBtn.disabled = true;
  countdownEl.hidden = false;
  countdownEl.classList.remove("rec");
  for (let i = 3; i >= 1; i--) { countdownEl.textContent = i; await sleep(600); }

  const manual = triggerMode === "manual";
  teach = {
    word, manual,
    state: "prime",          // prime -> ready -> capturing (manual jumps straight to capturing)
    frames: [], rest: [], restCount: 0, moveCount: 0,
    startedAt: performance.now(),
  };
  recordBtn.disabled = false;
  if (manual) {
    teach.state = "capturing";
    countdownEl.textContent = "● REC";
    countdownEl.classList.add("rec");
    recordBtn.textContent = "Stop & save";
    setTeachMsg("Recording… click Stop & save when your movement is done.", "");
  } else {
    recordBtn.textContent = "Cancel";
    setTeachMsg("Rest with your arms down, then perform your movement and return to rest.", "");
  }
}

// One frame of a teaching capture. Mirrors Perform: wait for rest, start on
// movement, end when the body returns to rest. No fixed time limit.
function teachStep(vec, rest, now) {
  const t = teach;
  if (t.manual) {
    t.frames.push(vec);
    t.rest.push(rest);
    return;
  }
  if (t.state === "prime") {
    countdownEl.textContent = "REST";
    countdownEl.classList.remove("rec");
    if (rest) t.state = "ready";
    return;
  }
  if (t.state === "ready") {
    countdownEl.textContent = "MOVE";
    if (!rest) {
      t.moveCount++;
      if (t.moveCount >= MOVE_TRIGGER_FRAMES) {
        t.state = "capturing";
        t.frames = [vec];
        t.rest = [rest];
        t.restCount = 0;
        t.startedAt = now;
      }
    } else {
      t.moveCount = 0;
    }
    return;
  }
  // capturing
  countdownEl.textContent = "● REC";
  countdownEl.classList.add("rec");
  t.frames.push(vec);
  t.rest.push(rest);
  if (rest) {
    t.restCount++;
    if (t.restCount >= REST_SETTLE_FRAMES) finishTeach(now);
  } else {
    t.restCount = 0;
  }
}

function finishTeach(now) {
  const t = teach;
  teach = null;
  countdownEl.hidden = true;
  countdownEl.classList.remove("rec");
  recordBtn.disabled = false;
  recordBtn.textContent = "Record movement";

  // Trim leading and trailing rest so every code starts and ends at rest.
  let a = 0, b = t.frames.length;
  while (a < b && t.rest[a]) a++;
  while (b > a && t.rest[b - 1]) b--;
  const core = t.frames.slice(a, b);

  if (core.length < 3) {
    setTeachMsg("No movement detected. Start at rest, perform your movement, then return to rest.", "warn");
    return;
  }
  const durMs = Math.max(300, Math.round(now - t.startedAt));
  const seq = resampleSeq(core, FIXED_LEN);
  templates.push({ id: newId(), word: t.word, seq, durMs, createdAt: Date.now() });
  saveTemplates();
  renderCodeList();
  const n = templates.filter((x) => x.word.toLowerCase() === t.word.toLowerCase()).length;
  setTeachMsg(
    n > 1 ? `Saved example ${n} for “${t.word}”. More examples improve recognition.`
          : `Saved “${t.word}”. Switch to Perform to try it.`,
    "ok"
  );
  wordInput.value = "";
}

function cancelTeach() {
  teach = null;
  countdownEl.hidden = true;
  countdownEl.classList.remove("rec");
  recordBtn.disabled = false;
  recordBtn.textContent = "Record movement";
  setTeachMsg("Cancelled.", "");
}

// ---------- Codes list ----------
function renderCodeList() {
  // Group templates by word (case-insensitive); each word can hold many examples.
  const groups = new Map();
  for (const t of templates) {
    const key = t.word.toLowerCase();
    if (!groups.has(key)) groups.set(key, { word: t.word, items: [] });
    groups.get(key).items.push(t);
  }

  // Refresh the autocomplete of existing words shown in the Teach input.
  if (wordList) {
    wordList.innerHTML = "";
    for (const g of groups.values()) {
      const o = document.createElement("option");
      o.value = g.word;
      wordList.appendChild(o);
    }
  }

  if (groups.size === 0) {
    codeList.innerHTML = '<li class="empty">No codes saved yet. Go to Teach to make one.</li>';
    return;
  }
  codeList.innerHTML = "";
  for (const g of groups.values()) {
    const count = g.items.length;
    const last = Math.max(...g.items.map((t) => t.createdAt));
    const li = document.createElement("li");
    li.className = "code-item";
    li.innerHTML = `
      <div>
        <div class="word">${escapeHtml(g.word)}<span class="count">${count} example${count > 1 ? "s" : ""}</span></div>
        <div class="meta">${new Date(last).toLocaleDateString()}</div>
      </div>
      <div class="row-actions">
        <button class="btn small" data-act="rename" data-word="${encodeURIComponent(g.word)}">Rename</button>
        <button class="btn small danger" data-act="del" data-word="${encodeURIComponent(g.word)}">Delete</button>
      </div>`;
    codeList.appendChild(li);
  }
}

codeList.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  const act = btn.dataset.act;
  const word = decodeURIComponent(btn.dataset.word || "");
  const matches = (t) => t.word.toLowerCase() === word.toLowerCase();
  if (act === "del") {
    if (!confirm(`Delete “${word}” and all its examples?`)) return;
    templates = templates.filter((t) => !matches(t));
    saveTemplates();
    renderCodeList();
  } else if (act === "rename") {
    const name = prompt("New word or phrase:", word);
    if (name && name.trim()) {
      const nn = name.trim();
      for (const t of templates) if (matches(t)) t.word = nn;
      saveTemplates();
      renderCodeList();
    }
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
recordBtn.addEventListener("click", () => {
  if (!teach) { startTeach(); return; }
  if (teach.manual) finishTeach(performance.now());
  else cancelTeach();
});
clearPhraseBtn.addEventListener("click", () => { phrase = []; renderPhrase(); });

triggerModeSel.addEventListener("change", () => {
  triggerMode = triggerModeSel.value;
  holdBtn.hidden = triggerMode !== "manual";
  manualCapturing = false;
  seg = newSeg();
  setPerformState("rest");
});
soundToggle.addEventListener("change", () => { soundOn = soundToggle.checked; });

// Hold-to-capture: pointer (mouse/touch)
holdBtn.addEventListener("pointerdown", (e) => { e.preventDefault(); startManual(); });
holdBtn.addEventListener("pointerup", stopManual);
holdBtn.addEventListener("pointerleave", stopManual);
holdBtn.addEventListener("pointercancel", stopManual);

// Hold-to-capture: Spacebar (manual mode, not while typing)
window.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || triggerMode !== "manual") return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;
  e.preventDefault();
  if (!e.repeat) startManual();
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space" && triggerMode === "manual") { e.preventDefault(); stopManual(); }
});

// ---------- Helpers ----------
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
function setTeachMsg(msg, cls) { teachMsg.textContent = msg; teachMsg.className = "teach-msg " + (cls || "muted"); }
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- Boot ----------
(async function boot() {
  threshVal.textContent = threshInput.value;
  renderCodeList();
  renderPhrase();

  // The pose engine is a ~15 MB first-time download (wasm + model). Kick it off
  // in parallel with the camera permission so the two overlap, and reassure the
  // user it is downloading, not frozen. The browser caches it after first load.
  let ready = false;
  statusEl.textContent = "Requesting camera + downloading pose engine (~15 MB first load)…";
  const slow = setTimeout(() => {
    if (!ready) statusEl.textContent = "Still downloading the pose engine (first load only). Hang tight…";
  }, 10000);

  try {
    const cam = initCamera().catch((e) => { throw new Error("camera: " + e.message); });
    const pose = initPose().catch((e) => { throw new Error("pose engine: " + e.message); });
    await Promise.all([cam, pose]);
    ready = true;
    clearTimeout(slow);
    statusEl.textContent = "Ready.";
    setPerformState("rest");
    loop();
  } catch (err) {
    clearTimeout(slow);
    console.error(err);
    statusEl.textContent = "Error loading " + err.message + " (camera needs https or localhost).";
  }
})();
