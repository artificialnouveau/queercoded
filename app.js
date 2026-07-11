import { createBackend, BLAZE_CONNECTIONS, COCO_CONNECTIONS } from "./backends.js";

// ---------- Config ----------
const STORE_KEY = "queercoded.templates.v1";
const FIXED_LEN = 20;             // frames every gesture is resampled to
const KEY_LMS = [11, 12];         // shoulders: visibility gate
const NUM_LMS = 33;

// Recording is bracketed by the hand-over-face rest pose: covering your face
// ARMS a capture, moving the hand away STARTS it, covering again STOPS it.
// Nothing else starts a recording, so idle movement can never fire a capture.
// The frames at both ends where the wrist is still near the face (the trip to
// and from the pose) are trimmed off, so a code spans only the gesture itself.
// After trimming, a capture must have moved some landmark at least MIN_TRAVEL
// (torso units) or it is dropped as a false start (face covered and uncovered
// with no gesture in between).
const MIN_TRAVEL = 0.3;
const MAX_SEG_MS = 10000;         // abandon a capture that never returns to the face
const MAX_TEACH_MS = 12000;       // safety cap so a teaching capture can't hang
const MIN_SEG_FRAMES = 4;         // ignore too-short blips
const COOLDOWN_MS = 1200;         // min gap before the same word fires again

// ---------- DOM ----------
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const bigWord = document.getElementById("bigWord");
const countdownEl = document.getElementById("countdown");
const bigStatus = document.getElementById("bigStatus");
const modelSel = document.getElementById("modelSel");

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
let backend = null;   // active pose backend (see backends.js)
let ready = false;    // backend loaded and detecting
let templates = loadTemplates();
let teach = null;     // active teaching capture (movement-delimited)
let seg = newSeg();   // live segmentation state
let lastFireAt = 0;
let lastFiredWord = "";
let phrase = [];

// Perform options / manual capture
let triggerMode = "auto";     // "auto" (rest pose) | "manual" (hold)
let soundOn = true;
let manualCapturing = false;
let manualFrames = [];
let audioCtx = null;
let playback = null;          // {word, items, idx, t0} ghost playback of a saved code

function newSeg() {
  return { state: "idle", frames: [], near: [], startedAt: 0 };
}

// Largest single-landmark displacement between two pose vectors.
function maxPoseDist(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i += 2) {
    const d = Math.hypot(a[i] - b[i], a[i + 1] - b[i + 1]);
    if (d > m) m = d;
  }
  return m;
}

// Drop the leading/trailing frames where a wrist was still near the face, so
// the capture spans the gesture itself rather than the trips to and from the
// rest pose. Mid-gesture passes near the face are kept (ends only).
function trimNearFace(frames, near) {
  let a = 0, b = frames.length;
  while (a < b - 1 && near[a]) a++;
  while (b > a + 1 && near[b - 1]) b--;
  return frames.slice(a, b);
}

// How far the pose ever strays from its first frame. Used to reject captures
// where the face was covered and uncovered with no gesture in between.
function travelOf(frames) {
  if (frames.length === 0) return 0;
  let m = 0;
  for (const f of frames) m = Math.max(m, maxPoseDist(f, frames[0]));
  return m;
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
  // Shoulders only. The hips used by normalizePose() are taken from the
  // model's estimate even when they are out of frame; requiring them to be
  // VISIBLE blocked all processing for close, face-and-torso framings, which
  // the hand-over-face rest pose is specifically meant to support.
  return KEY_LMS.every((i) => (lms[i]?.visibility ?? 0) > 0.35);
}

// "Resting" = a hand over the face: either wrist held close to the face.
// Chosen over hands-on-hips because a close camera framing often crops or
// barely sees the hips, while the face is always solidly tracked, and it is a
// distinct pose that a dance movement is unlikely to pass through slowly.
//
// The face anchor averages whichever of nose/ears are still visible, with a
// LOW visibility gate: the covering hand itself occludes the nose, and a
// strict gate here would make the rest pose undetectable exactly when held.
// Hysteresis (looser exit than enter) stops boundary flicker.
const REST_ENTER = 0.55;  // wrist-to-face distance (in shoulder widths) to enter rest
const REST_EXIT = 0.8;    // distance to leave rest once in it
const FACE_LMS = [0, 7, 8]; // nose + ears
// The trip to and from the face is not part of the gesture. Frames at either
// end of a capture where a wrist is within this radius of the face are
// trimmed, so a code spans the movement itself, not the rest-pose transitions.
const NEAR_FACE_R = 1.0;
// Rest only flips state after this many consecutive frames agree, on top of
// the enter/exit hysteresis. Kills single-frame flicker when the covering
// hand makes the face landmarks jump.
const REST_DEBOUNCE = 2;
let wasResting = false;
let restStreak = 0;
let restInfo = null;      // per-frame info for drawing the face target circle

function isResting(lms) {
  const lw = lms[15], rw = lms[16], ls = lms[11], rs = lms[12];
  restInfo = null;
  const face = FACE_LMS.map((i) => lms[i]).filter((p) => (p?.visibility ?? 0) > 0.2);
  if (face.length === 0 || (ls?.visibility ?? 0) < 0.35 || (rs?.visibility ?? 0) < 0.35) {
    wasResting = false;
    return false;
  }
  const anchor = {
    x: face.reduce((s, p) => s + p.x, 0) / face.length,
    y: face.reduce((s, p) => s + p.y, 0) / face.length,
  };
  const scale = Math.hypot(ls.x - rs.x, ls.y - rs.y) || 1e-6; // shoulder width
  const dL = (lw?.visibility ?? 0) > 0.2 ? Math.hypot(lw.x - anchor.x, lw.y - anchor.y) / scale : Infinity;
  const dR = (rw?.visibility ?? 0) > 0.2 ? Math.hypot(rw.x - anchor.x, rw.y - anchor.y) / scale : Infinity;
  const d = Math.min(dL, dR);
  const thr = wasResting ? REST_EXIT : REST_ENTER;
  const inRange = d < thr;
  if (inRange !== wasResting) {
    restStreak++;
    if (restStreak >= REST_DEBOUNCE) { wasResting = inRange; restStreak = 0; }
  } else {
    restStreak = 0;
  }
  restInfo = { anchor, scale, d };
  return wasResting;
}

// True while a wrist is anywhere near the face: wider than the rest radius so
// it also covers the approach/departure frames around the rest pose.
function isNearFace() {
  return restInfo ? restInfo.d < NEAR_FACE_R : false;
}

// Visual guide: a circle over the face that lights up when a wrist is close
// enough to count as "hand over face". Lets the performer see the target
// instead of guessing at an invisible threshold.
function drawRestTargets() {
  if (!restInfo) return;
  const { anchor, scale, d } = restInfo;
  const r = REST_ENTER * scale * overlay.width * 0.9;
  const inRange = d < (wasResting ? REST_EXIT : REST_ENTER);
  octx.beginPath();
  octx.arc(anchor.x * overlay.width, anchor.y * overlay.height, r, 0, Math.PI * 2);
  octx.strokeStyle = inRange ? "rgba(52,211,153,0.9)" : "rgba(255,255,255,0.5)";
  octx.lineWidth = inRange ? 4 : 2;
  octx.stroke();
  if (inRange) {
    octx.fillStyle = "rgba(52,211,153,0.15)";
    octx.fill();
  }
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

// ---------- Pose algorithms ----------
// Three genuinely different algorithms, all producing the same 33-slot pose
// (see backends.js). BlazePose keeps 33 native points; MoveNet and YOLO give
// 17, mapped into the same slots. Codes are tagged with the algorithm FAMILY
// and only matched within their family, since the three do not agree on scale.
const ALGOS = {
  "blaze-lite":       { family: "blaze",   label: "MediaPipe BlazePose Lite" },
  "blaze-full":       { family: "blaze",   label: "MediaPipe BlazePose Full" },
  "blaze-heavy":      { family: "blaze",   label: "MediaPipe BlazePose Heavy" },
  "movenet-lightning":{ family: "movenet", label: "MoveNet Lightning (TF.js)" },
  "movenet-thunder":  { family: "movenet", label: "MoveNet Thunder (TF.js)" },
  "yolo":             { family: "yolo",    label: "YOLO-Pose (ONNX Runtime Web)" },
};
const ALGO_STORE_KEY = "queercoded.algo.v1";
const YOLO_URL_KEY = "queercoded.yoloModelUrl";
let algoChoice = localStorage.getItem(ALGO_STORE_KEY);
if (!ALGOS[algoChoice]) algoChoice = "blaze-full";
let currentFamily = ALGOS[algoChoice].family;

function connectionsForFamily(fam) { return fam === "blaze" ? BLAZE_CONNECTIONS : COCO_CONNECTIONS; }

async function initPose() {
  backend = createBackend(algoChoice, { yoloModelUrl: localStorage.getItem(YOLO_URL_KEY) || "" });
  await backend.load();
  currentFamily = backend.family;
}

// Swap algorithm at runtime. Detection is paused (ready=false) during the
// load, so the video keeps playing. A failure (bad CDN, missing YOLO model)
// reverts to the previous algorithm instead of leaving the app dead.
async function switchAlgo(key) {
  if (!ALGOS[key] || key === algoChoice) return;
  // YOLO needs a model file; ask for a URL the first time if none is stored.
  if (key === "yolo" && !localStorage.getItem(YOLO_URL_KEY)) {
    const url = prompt(
      "YOLO-Pose needs a YOLOv8/YOLO11-pose model exported to ONNX (640x640).\n" +
      "Paste a URL to the .onnx file (CORS-enabled), or Cancel to keep the current algorithm.\n" +
      "See the README for how to export one.", "");
    if (!url || !url.trim()) { modelSel.value = algoChoice; return; }
    localStorage.setItem(YOLO_URL_KEY, url.trim());
  }
  const prev = algoChoice;
  algoChoice = key;
  localStorage.setItem(ALGO_STORE_KEY, key);
  if (teach) cancelTeach();
  seg = newSeg();
  ready = false;
  try { backend?.close(); } catch {}
  backend = null;
  statusEl.textContent = `Loading ${ALGOS[key].label}…`;
  try {
    await initPose();
    ready = true;
    setPerformState();
  } catch (e) {
    console.error(e);
    statusEl.textContent = `Could not load ${ALGOS[key].label}: ${e.message}. Reverting.`;
    // Roll back to the algorithm that was working.
    algoChoice = prev;
    localStorage.setItem(ALGO_STORE_KEY, prev);
    modelSel.value = prev;
    if (key === "yolo") localStorage.removeItem(YOLO_URL_KEY); // let them re-enter a URL
    try { await initPose(); ready = true; setPerformState(); } catch (e2) { console.error(e2); }
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

// Of the detected people, return the most prominent one: the nearest, which
// reads as the largest skeleton (shoulder width + torso length), lightly
// weighted by how confidently the shoulders are seen. This is what keeps a
// bystander in the background from stealing the tracking.
function pickMainPose(poses) {
  if (!poses || poses.length === 0) return null;
  if (poses.length === 1) return poses[0];
  let best = null, bestScore = -Infinity;
  for (const p of poses) {
    const ls = p[11], rs = p[12], lh = p[23], rh = p[24];
    if (!ls || !rs) continue;
    const shoulder = Math.hypot(ls.x - rs.x, ls.y - rs.y);
    const shx = (ls.x + rs.x) / 2, shy = (ls.y + rs.y) / 2;
    const hx = ((lh?.x ?? shx) + (rh?.x ?? shx)) / 2;
    const hy = ((lh?.y ?? shy) + (rh?.y ?? shy)) / 2;
    const torso = Math.hypot(shx - hx, shy - hy);
    const vis = ((ls.visibility ?? 0) + (rs.visibility ?? 0)) / 2;
    const score = (shoulder + torso) * (0.5 + vis);
    if (score > bestScore) { bestScore = score; best = p; }
  }
  return best || poses[0];
}

// Draw a skeleton (33-slot pose) with the given bone list. Replaces MediaPipe
// DrawingUtils so every algorithm draws through one path. Landmarks below a
// visibility floor are skipped, so 17-point models do not draw phantom bones.
function drawSkeleton(pose, connections, boneColor, dotColor) {
  const VIS = 0.3;
  octx.strokeStyle = boneColor;
  octx.lineWidth = 3;
  for (const [i, j] of connections) {
    const a = pose[i], b = pose[j];
    if (!a || !b || (a.visibility ?? 1) < VIS || (b.visibility ?? 1) < VIS) continue;
    octx.beginPath();
    octx.moveTo(a.x * overlay.width, a.y * overlay.height);
    octx.lineTo(b.x * overlay.width, b.y * overlay.height);
    octx.stroke();
  }
  octx.fillStyle = dotColor;
  for (const p of pose) {
    if (!p || (p.visibility ?? 1) < VIS) continue;
    octx.beginPath();
    octx.arc(p.x * overlay.width, p.y * overlay.height, 3, 0, Math.PI * 2);
    octx.fill();
  }
}

// ---------- Main loop ----------
// Async because MoveNet and YOLO detect asynchronously; `inflight` keeps frames
// from overlapping. BlazePose resolves synchronously, so this stays real-time.
let inflight = false;
async function loop() {
  if (ready && backend && !inflight && video.readyState >= 2) {
    inflight = true;
    try { await runFrame(); }
    finally { inflight = false; }
  }
  requestAnimationFrame(loop);
}

async function runFrame() {
  {
    const now = performance.now();
    let poses = [];
    try { poses = await backend.detect(video, now); }
    catch (e) { console.warn("detect failed:", e); }
    octx.clearRect(0, 0, overlay.width, overlay.height);

    let bodyVisible = false, restNow = false;
    const lms = pickMainPose(poses);
    if (lms) {
      drawSkeleton(lms, backend.connections, "rgba(123,92,255,0.9)", "#ff4d9d");

      if (keyLandmarksVisible(lms)) {
        bodyVisible = true;
        const vec = normalizePose(lms);
        restNow = isResting(lms);
        const nearNow = isNearFace();
        if (teach) {
          teachStep(vec, restNow, nearNow, now);
        } else if (triggerMode === "manual") {
          if (manualCapturing) manualFrames.push(vec);
        } else {
          segmentStep(vec, restNow, nearNow, now);
        }
        // Face target circle: the start/stop control, so always visible in
        // auto mode (except during ghost playback).
        if (triggerMode === "auto" && !playback) drawRestTargets();
      }

      // Keep the matched word floating just above the head. The video is
      // mirrored, so x flips; y is clamped so the word stays inside the frame.
      const nose = lms[0];
      if (nose) {
        const topY = Math.max(0.14, Math.min(...FACE_LMS.map((i) => lms[i]?.y ?? 1)) - 0.05);
        bigWord.style.left = ((1 - nose.x) * 100).toFixed(1) + "%";
        bigWord.style.top = (topY * 100).toFixed(1) + "%";
      }
    }

    // While recording, the screen belongs to the movement: hide any matched
    // word and show a big REC. The big SAVED/countdown flashes are left alone
    // (they only appear when not recording), so this never clobbers them.
    const recording = seg.state === "recording" || manualCapturing ||
      (teach && (teach.manual || teach.state === "capturing"));
    if (recording) {
      bigWord.classList.remove("show");
      if (!bigStatus.classList.contains("rec")) {
        bigStatus.textContent = "REC";
        bigStatus.className = "big-status rec show";
      }
    } else if (bigStatus.classList.contains("rec")) {
      bigStatus.className = "big-status";
    }

    // The teach overlay is driven here, EVERY frame, not inside the
    // detection-gated branch. Otherwise losing sight of the body freezes the
    // on-screen state ("stuck on 1" / stale REC) while the state machine waits.
    if (teach) updateTeachUI(bodyVisible, restNow);

    // Ghost playback of a saved code, drawn on top of the live view.
    if (playback) drawPlayback(now);

    // Safety cap so a teaching capture can never hang, even if the body leaves
    // the frame mid-movement.
    if (teach && teach.state === "capturing" && now - teach.startedAt > MAX_TEACH_MS) {
      finishTeach(now, true);
    }
  }
}

// ---------- Hand-over-face bracketed segmentation ----------
// idle: waiting for the face to be covered. armed: hand is on the face; the
// capture starts the moment it leaves. recording: collecting frames until the
// hand covers the face again.
function segmentStep(vec, rest, near, now) {
  if (seg.state === "idle") {
    if (rest) seg.state = "armed";
    setPerformState();
    return;
  }
  if (seg.state === "armed") {
    if (!rest) {
      seg.state = "recording";
      seg.frames = [vec];
      seg.near = [near];
      seg.startedAt = now;
    }
    setPerformState();
    return;
  }
  // recording
  seg.frames.push(vec);
  seg.near.push(near);
  if (rest) { closeSegment(now); return; }
  if (now - seg.startedAt > MAX_SEG_MS) { seg = newSeg(); setPerformState(); return; }
  setPerformState();
}

function closeSegment(now) {
  const frames = trimNearFace(seg.frames, seg.near);
  seg = newSeg();
  seg.state = "armed"; // the hand is on the face right now
  setPerformState();
  if (frames.length >= MIN_SEG_FRAMES && travelOf(frames) >= MIN_TRAVEL) matchAndFire(frames, now);
}

function matchAndFire(frames, now) {
  // Only compare against codes taught with the current algorithm family; the
  // three algorithms do not agree on scale, so cross-family matching is noise.
  const pool = templates.filter((t) => (t.family || "blaze") === currentFamily);
  if (pool.length === 0) {
    bestWordEl.textContent = templates.length ? "no codes for this algorithm" : "no codes yet";
    return;
  }
  const live = resampleSeq(frames, FIXED_LEN);
  const thresh = parseFloat(threshInput.value);
  let best = { word: null, dist: Infinity };
  for (const t of pool) {
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

// Status text is derived from the live segmentation and trigger state.
function setPerformState() {
  if (!ready) return;
  if (triggerMode === "manual") {
    statusEl.textContent = manualCapturing ? "● capturing movement…" : "● ready. Hold the button to capture.";
    return;
  }
  if (seg.state === "recording") statusEl.textContent = "● recording. Cover your face to finish.";
  else if (seg.state === "armed") statusEl.textContent = "Hand on face. Move it away and perform a code.";
  else statusEl.textContent = "● watching. Cover your face with one hand to start.";
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
  setPerformState();
  holdBtn.classList.add("recording");
}
function stopManual() {
  if (!manualCapturing) return;
  manualCapturing = false;
  holdBtn.classList.remove("recording");
  setPerformState();
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

// Big centered flash in the video for a transient status word (SAVED, and the
// 3-2-1 countdown). REC is driven continuously from the loop, not here.
let bigStatusTimer = null;
function flashBigStatus(text, cls, ms = 1400) {
  clearTimeout(bigStatusTimer);
  bigStatus.textContent = text;
  bigStatus.className = "big-status show" + (cls ? " " + cls : "");
  if (ms) bigStatusTimer = setTimeout(() => {
    if (!bigStatus.classList.contains("rec")) bigStatus.className = "big-status";
  }, ms);
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
  statusEl.textContent = "Get ready…";
  for (let i = 3; i >= 1; i--) { flashBigStatus(i, "count", 0); await sleep(600); }
  bigStatus.className = "big-status";
  countdownEl.hidden = false;
  countdownEl.classList.remove("rec");

  const manual = triggerMode === "manual";
  teach = {
    word, manual,
    state: "prime",          // prime -> armed -> capturing (manual jumps straight to capturing)
    frames: [], near: [],
    startedAt: performance.now(),
  };
  recordBtn.disabled = false;
  if (manual) {
    teach.state = "capturing";
    countdownEl.textContent = "REC";
    countdownEl.classList.add("rec");
    recordBtn.textContent = "Stop & save";
    setTeachMsg("Recording… click Stop & save when your movement is done.", "");
  } else {
    recordBtn.textContent = "Cancel";
    setTeachMsg("Cover your face with one hand. Recording starts when you move it away and stops when you cover your face again.", "");
  }
}

// One frame of a teaching capture. Mirrors Perform: cover the face to arm,
// move the hand away to start recording, cover again to finish and save.
function teachStep(vec, rest, near, now) {
  const t = teach;
  if (t.manual) {
    t.frames.push(vec);
    t.near.push(near);
    return;
  }
  if (t.state === "prime") {  // waiting for the hand to cover the face
    if (rest) t.state = "armed";
    return;
  }
  if (t.state === "armed") {  // hand on face; recording starts when it leaves
    if (!rest) {
      t.state = "capturing";
      t.frames = [vec];
      t.near = [near];
      t.startedAt = now;
    }
    return;
  }
  // capturing
  t.frames.push(vec);
  t.near.push(near);
  if (rest) finishTeach(now, false);
}

// On-screen overlay for the teach flow, driven every frame so it can also say
// when the body is NOT being detected instead of silently freezing.
function updateTeachUI(bodyVisible, rest) {
  const t = teach;
  if (!t) return;
  countdownEl.hidden = false;
  if (!bodyVisible) {
    countdownEl.textContent = "?";
    countdownEl.classList.remove("rec");
    statusEl.textContent = "Can't see your shoulders. Adjust your framing.";
    return;
  }
  if (t.manual || t.state === "capturing") {
    // The big REC overlay covers this state, so keep the pill out of the way.
    countdownEl.hidden = true;
    statusEl.textContent = t.manual
      ? "Recording. Click Stop & save when done."
      : "Recording. Cover your face with one hand to finish and save.";
    return;
  }
  countdownEl.classList.remove("rec");
  if (t.state === "prime") {
    countdownEl.textContent = "COVER";
    statusEl.textContent = "Step 1 of 2: cover your face with one hand (the circle).";
  } else { // armed
    countdownEl.textContent = "SET";
    statusEl.textContent = "Step 2 of 2: move your hand away and perform. Cover your face again to finish.";
  }
}

function finishTeach(now, timedOut = false) {
  const t = teach;
  teach = null;
  countdownEl.hidden = true;
  countdownEl.classList.remove("rec");
  recordBtn.disabled = false;
  recordBtn.textContent = "Record movement";
  setPerformState();

  // Trim leading rest and trailing rest/stillness so every code starts and
  // ends where the movement actually happened.
  const core = trimNearFace(t.frames, t.near);

  // Same travel gate as segmentation, applied to manual and timed-out
  // recordings too: a capture that never really went anywhere is not a code.
  if (core.length < 3 || travelOf(core) < MIN_TRAVEL) {
    setTeachMsg("No clear movement captured. Check the skeleton overlay is tracking you, then try a bigger movement between covering your face.", "warn");
    return;
  }
  const durMs = Math.max(300, Math.round(now - t.startedAt));
  const seq = resampleSeq(core, FIXED_LEN);
  templates.push({ id: newId(), word: t.word, seq, durMs, family: currentFamily, algo: algoChoice, createdAt: Date.now() });
  saveTemplates();
  renderCodeList();
  flashBigStatus("SAVED", "saved");
  const n = templates.filter((x) => x.word.toLowerCase() === t.word.toLowerCase()).length;
  let msg = n > 1
    ? `Saved example ${n} for “${t.word}”. More examples improve recognition.`
    : `Saved “${t.word}”. Switch to Perform to try it.`;
  if (timedOut) msg += " (Recording hit its time cap; covering your face to finish was never detected.)";
  setTeachMsg(msg, "ok");
  wordInput.value = "";
}

function cancelTeach() {
  teach = null;
  countdownEl.hidden = true;
  countdownEl.classList.remove("rec");
  recordBtn.disabled = false;
  recordBtn.textContent = "Record movement";
  setPerformState();
  setTeachMsg("Cancelled.", "");
}

// ---------- Ghost playback of saved codes ----------
// The stored seq is normalized (hip-midpoint origin, torso-length scale), so a
// synthetic skeleton is reprojected into image space at a fixed center/size and
// tweened between the stored frames. No video is stored or replayed, only the
// coordinate movement.
const GHOST_CX = 0.5;    // horizontal center (image-normalized)
const GHOST_CY = 0.5;    // vertical hip position
const GHOST_SCALE = 0.18; // torso length as a fraction of image space

// Play back one or more stored examples as a ghost skeleton. `key` identifies
// what is playing (a word group, or a single example id) so the matching row
// can show a Stop button.
function startPlaybackItems(items, label, key) {
  if (!items || items.length === 0) return;
  playback = { items, label, key, idx: 0, t0: performance.now() };
  renderCodeList();
}
function startPlayback(word) {
  const items = templates.filter((t) => t.word.toLowerCase() === word.toLowerCase());
  startPlaybackItems(items, word, "word:" + word.toLowerCase());
}
function startPlaybackExample(id) {
  const t = templates.find((x) => x.id === id);
  if (t) startPlaybackItems([t], t.word, "ex:" + id);
}

function stopPlayback() {
  playback = null;
  if (ready) setPerformState();
  renderCodeList();
}

function drawPlayback(now) {
  const pb = playback;
  const cur = pb.items[pb.idx];
  const dur = Math.min(5000, Math.max(800, cur.durMs || 2000));
  const p = (now - pb.t0) / dur;
  if (p >= 1) {
    pb.idx++;
    pb.t0 = now;
    if (pb.idx >= pb.items.length) { stopPlayback(); return; }
    return;
  }

  // Tween between the two nearest stored frames.
  const f = p * (FIXED_LEN - 1);
  const i = Math.floor(f);
  const frac = f - i;
  const a = cur.seq[i], b = cur.seq[Math.min(i + 1, FIXED_LEN - 1)];
  const ghost = [];
  for (let k = 0; k < NUM_LMS; k++) {
    const vx = a[k * 2] * (1 - frac) + b[k * 2] * frac;
    const vy = a[k * 2 + 1] * (1 - frac) + b[k * 2 + 1] * frac;
    // Slots a 17-point code never filled stay at the origin; mark them
    // invisible so drawSkeleton skips them instead of drawing to (0,0).
    const filled = a[k * 2] !== 0 || a[k * 2 + 1] !== 0 || b[k * 2] !== 0 || b[k * 2 + 1] !== 0;
    ghost.push({ x: GHOST_CX + vx * GHOST_SCALE, y: GHOST_CY + vy * GHOST_SCALE, visibility: filled ? 1 : 0 });
  }

  // Draw with the bone set of the algorithm the code was taught on.
  drawSkeleton(ghost, connectionsForFamily(cur.family || "blaze"), "rgba(255,195,113,0.95)", "#ffffff");

  statusEl.textContent =
    `Playing “${pb.label}”` + (pb.items.length > 1 ? ` (example ${pb.idx + 1}/${pb.items.length})` : "");
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
  const FAM_LABEL = { blaze: "BlazePose", movenet: "MoveNet", yolo: "YOLO" };
  codeList.innerHTML = "";
  for (const g of groups.values()) {
    const count = g.items.length;
    const last = Math.max(...g.items.map((t) => t.createdAt));
    const fams = [...new Set(g.items.map((t) => t.family || "blaze"))];
    const famBadge = fams.map((f) => `<span class="fam fam-${f}">${FAM_LABEL[f] || f}</span>`).join("");
    const wordKey = "word:" + g.word.toLowerCase();
    const wordPlaying = playback && playback.key === wordKey;
    const w = encodeURIComponent(g.word);
    const li = document.createElement("li");
    li.className = "code-item";

    // Header: the word, plus actions on the whole group.
    const playAllLabel = count > 1 ? "Play all" : "Play";
    let html = `
      <div class="code-head">
        <div class="code-info" data-act="play" data-word="${w}" title="Play this movement">
          <div class="word">${escapeHtml(g.word)}<span class="count">${count} example${count > 1 ? "s" : ""}</span></div>
          <div class="meta">${new Date(last).toLocaleDateString()} ${famBadge}</div>
        </div>
        <div class="row-actions">
          <button class="btn small ${wordPlaying ? "playing" : ""}" data-act="play" data-word="${w}">${wordPlaying ? "Stop" : playAllLabel}</button>
          <button class="btn small" data-act="rename" data-word="${w}">Rename</button>
          <button class="btn small danger" data-act="del" data-word="${w}">Delete</button>
        </div>
      </div>`;

    // One row per example: play it (dance along with just that take) or delete
    // only that take. Shown when there is more than one so single-example words
    // stay compact.
    if (count > 1) {
      html += '<ul class="example-list">';
      g.items.forEach((t, i) => {
        const exKey = "ex:" + t.id;
        const exPlaying = playback && playback.key === exKey;
        html += `
          <li class="example-row">
            <span class="ex-name">Example ${i + 1}<span class="ex-date">${new Date(t.createdAt).toLocaleDateString()}</span></span>
            <span class="row-actions">
              <button class="btn tiny ${exPlaying ? "playing" : ""}" data-act="play-ex" data-id="${t.id}">${exPlaying ? "Stop" : "Play"}</button>
              <button class="btn tiny danger" data-act="del-ex" data-id="${t.id}">Delete</button>
            </span>
          </li>`;
      });
      html += "</ul>";
    }

    li.innerHTML = html;
    codeList.appendChild(li);
  }
}

codeList.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const act = btn.dataset.act;
  const word = decodeURIComponent(btn.dataset.word || "");
  const id = btn.dataset.id;
  const matches = (t) => t.word.toLowerCase() === word.toLowerCase();
  if (act === "play") {
    if (playback && playback.key === "word:" + word.toLowerCase()) stopPlayback();
    else startPlayback(word);
  } else if (act === "play-ex") {
    if (playback && playback.key === "ex:" + id) stopPlayback();
    else startPlaybackExample(id);
  } else if (act === "del") {
    stopPlayback();
    if (!confirm(`Delete “${word}” and all its examples?`)) return;
    templates = templates.filter((t) => !matches(t));
    saveTemplates();
    renderCodeList();
  } else if (act === "del-ex") {
    const t = templates.find((x) => x.id === id);
    if (!t) return;
    const n = templates.filter((x) => x.word.toLowerCase() === t.word.toLowerCase()).length;
    if (!confirm(n > 1 ? `Delete this example of “${t.word}”?` : `Delete “${t.word}”? It has only this example.`)) return;
    stopPlayback();
    templates = templates.filter((x) => x.id !== id);
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
  setPerformState();
});
soundToggle.addEventListener("change", () => { soundOn = soundToggle.checked; });
modelSel.addEventListener("change", () => switchAlgo(modelSel.value));

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
  modelSel.value = algoChoice;
  renderCodeList();
  renderPhrase();

  // The pose engine is a ~15 MB first-time download (wasm + model). Kick it off
  // in parallel with the camera permission so the two overlap, and reassure the
  // user it is downloading, not frozen. The browser caches it after first load.
  statusEl.textContent = "Requesting camera + downloading the pose model (first load can take a minute)…";
  const slow = setTimeout(() => {
    if (!ready) statusEl.textContent = "Still downloading the pose engine (first load only). Hang tight…";
  }, 10000);

  // Start the render loop now; it idles until `ready` flips true.
  loop();

  try {
    const cam = initCamera().catch((e) => { throw new Error("camera: " + e.message); });
    const pose = initPose().catch((e) => { throw new Error("pose engine: " + e.message); });
    await Promise.all([cam, pose]);
    ready = true;
    clearTimeout(slow);
    statusEl.textContent = "Ready.";
    setPerformState();
  } catch (err) {
    clearTimeout(slow);
    console.error(err);
    statusEl.textContent = "Error loading " + err.message + " (camera needs https or localhost).";
  }
})();
