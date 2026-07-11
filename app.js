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

// A gesture starts when real MOTION is detected and ends on stillness or the
// hand-over-face rest pose. Motion-based starting (rather than "not in the
// rest pose") means standing anywhere doing nothing can never start a capture.
const MOVE_EPS = 0.12;            // max-landmark displacement that counts as movement...
const MOVE_WIN_MS = 250;          // ...measured over this look-back window
const PREROLL_MS = 300;           // include this much pre-movement history in the capture
const IDLE_BUF_MS = 800;          // rolling history kept while idle
const REST_SETTLE_FRAMES = 5;     // frames of "rest" that end a capture
// Holding still ALSO ends a capture (fallback for marginal rest-pose
// detection). "Still" means the FASTEST single landmark barely moved over a
// short time window. Max-over-landmarks matters: a one-arm gesture moves only
// a few of the 33 landmarks, so a mean would read as still mid-gesture.
// Time-window based so it is frame-rate independent.
const STILL_WIN_SHORT_MS = 220;   // compare against ~this long ago...
const STILL_WIN_LONG_MS = 450;    // ...and this long ago (catches oscillation)
const STILL_EPS = 0.06;           // max-landmark displacement counting as still
const STILL_CONSEC = 3;           // consecutive still frames required to end
// A capture only closes as a real gesture once some landmark has travelled
// this far (torso units) from the capture's starting pose. Below that the
// "movement" was a twitch or tracker jitter: keep recording a little longer in
// case the gesture is slow to build, then drop the capture quietly instead of
// erroring with "no movement". This is what made slow, deliberate movements
// read as nothing: they tripped the stillness check before covering any
// distance, and the trimmed capture came back empty.
const MIN_TRAVEL = 0.3;
const FALSE_START_FRAMES = 20;    // settled frames after which a low-travel capture is dropped
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
let playback = null;          // {word, items, idx, t0} ghost playback of a saved code

function newSeg() {
  return { state: "idle", buf: [], frames: [], rest: [], times: [], restCount: 0, stillCount: 0, startedAt: 0, startVec: null, travel: 0 };
}

// Rolling idle-history helpers. The buffer holds {vec, rest, t} entries so a
// capture can start on detected motion and still include a short preroll.
function pushIdle(buf, vec, rest, t) {
  buf.push({ vec, rest, t });
  while (buf.length && buf[0].t < t - IDLE_BUF_MS) buf.shift();
}

// True when some landmark has moved meaningfully within the look-back window.
function isMovingNow(buf, now) {
  const last = buf[buf.length - 1];
  for (let i = buf.length - 2; i >= 0; i--) {
    if (now - buf[i].t >= MOVE_WIN_MS) return maxPoseDist(last.vec, buf[i].vec) > MOVE_EPS;
  }
  return false; // not enough history yet
}

// Seed a capture from the idle buffer: everything from PREROLL_MS before now.
function prerollFrom(buf, now) {
  const start = now - PREROLL_MS;
  const picked = buf.filter((e) => e.t >= start);
  return {
    frames: picked.map((e) => e.vec),
    rest: picked.map((e) => e.rest),
    times: picked.map((e) => e.t),
  };
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

// True when no landmark has moved meaningfully over the recent time window.
// Checks two lags so a periodic movement (a wave returning to the same pose)
// is not mistaken for stillness.
function isStillNow(frames, times, now) {
  const last = frames[frames.length - 1];
  let iShort = -1, iLong = -1;
  for (let i = times.length - 1; i >= 0; i--) {
    if (iShort < 0 && times[i] <= now - STILL_WIN_SHORT_MS) iShort = i;
    if (times[i] <= now - STILL_WIN_LONG_MS) { iLong = i; break; }
  }
  if (iLong < 0) return false; // capture younger than the long window
  return (
    maxPoseDist(last, frames[iShort]) < STILL_EPS &&
    maxPoseDist(last, frames[iLong]) < STILL_EPS
  );
}

// Capture trim shared by teach and perform: drop leading preroll/rest frames
// and trailing rest/settle frames so a capture spans exactly the movement.
// Both ends are bounded so slow deliberate starts and endings are not eaten.
function trimCapture(frames, rest, times) {
  // Leading: frames still matching the starting pose, within the preroll span.
  let a = 0;
  const first = frames[0];
  while (a < frames.length - 1) {
    if (rest[a]) { a++; continue; }
    const withinWindow = times[a] - times[0] <= PREROLL_MS + STILL_WIN_SHORT_MS;
    if (withinWindow && maxPoseDist(frames[a], first) < STILL_EPS) { a++; continue; }
    break;
  }
  // Trailing: frames matching the final settle pose, within the settle span.
  let b = frames.length;
  const last = frames[b - 1];
  while (b > a + 1) {
    if (rest[b - 1]) { b--; continue; }
    const withinWindow = times[frames.length - 1] - times[b - 1] <= STILL_WIN_LONG_MS + STILL_WIN_SHORT_MS;
    if (withinWindow && maxPoseDist(frames[b - 1], last) < STILL_EPS) { b--; continue; }
    break;
  }
  return frames.slice(a, b);
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
  // 0.35 rather than 0.5: with a close upper-body framing the hips often sit
  // right at the frame edge and hover around 0.4 visibility. Gating at 0.5
  // made detection flicker, which froze the teach state machine.
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
let wasResting = false;
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
  wasResting = d < thr;
  restInfo = { anchor, scale, d };
  return wasResting;
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

// ---------- MediaPipe ----------
// All three variants output the same 33 landmarks, so saved codes stay
// compatible when the user switches. Lite is quick but loses track of fast
// limbs; Full is the best accuracy/speed balance for dance; Heavy is the most
// accurate but can drop the frame rate on slower machines.
const MODEL_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/";
const MODELS = {
  lite:  { label: "MediaPipe Pose Landmarker Lite",  url: MODEL_BASE + "pose_landmarker_lite/float16/1/pose_landmarker_lite.task" },
  full:  { label: "MediaPipe Pose Landmarker Full",  url: MODEL_BASE + "pose_landmarker_full/float16/1/pose_landmarker_full.task" },
  heavy: { label: "MediaPipe Pose Landmarker Heavy", url: MODEL_BASE + "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" },
};
const MODEL_STORE_KEY = "queercoded.model.v1";
let modelChoice = localStorage.getItem(MODEL_STORE_KEY);
if (!MODELS[modelChoice]) modelChoice = "full";

let fileset = null; // wasm runtime, fetched once and reused across model swaps

async function initPose() {
  fileset = fileset || await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODELS[modelChoice].url, delegate },
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

// Swap the pose model at runtime. The main loop skips detection while
// `landmarker` is null, so the video keeps playing during the download.
async function switchModel(key) {
  if (!MODELS[key] || key === modelChoice) return;
  modelChoice = key;
  localStorage.setItem(MODEL_STORE_KEY, key);
  if (teach) cancelTeach();
  seg = newSeg();
  const old = landmarker;
  landmarker = null;
  try { old?.close(); } catch {}
  statusEl.textContent = `Loading ${MODELS[key].label} model…`;
  try {
    await initPose();
    setPerformState("rest");
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Could not load the " + MODELS[key].label + " model: " + e.message;
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

    let bodyVisible = false, restNow = false;
    if (res.landmarks && res.landmarks.length > 0) {
      const lms = res.landmarks[0];
      if (!du) du = new DrawingUtils(octx);
      du.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS, { color: "rgba(123,92,255,0.9)", lineWidth: 3 });
      du.drawLandmarks(lms, { color: "#ff4d9d", radius: 3, lineWidth: 1 });

      if (keyLandmarksVisible(lms)) {
        bodyVisible = true;
        const vec = normalizePose(lms);
        restNow = isResting(lms);
        if (teach) {
          teachStep(vec, restNow, now);
        } else if (triggerMode === "manual") {
          if (manualCapturing) manualFrames.push(vec);
        } else {
          segmentStep(vec, restNow, now);
        }
        // Face target circle: visible whenever a movement could start or a
        // rest pose would end one, so the pose threshold is never a guess.
        const capturing = teach ? teach.state === "capturing" && teach.manual : false;
        if (triggerMode === "auto" && !playback && !capturing) drawRestTargets();
      }
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
  requestAnimationFrame(loop);
}

// ---------- Motion-delimited segmentation ----------
function segmentStep(vec, rest, now) {
  if (seg.state === "idle") {
    pushIdle(seg.buf, vec, rest, now);
    if (isMovingNow(seg.buf, now)) {
      const pre = prerollFrom(seg.buf, now);
      seg.state = "move";
      seg.frames = pre.frames;
      seg.rest = pre.rest;
      seg.times = pre.times;
      seg.startedAt = now;
      seg.restCount = 0;
      seg.stillCount = 0;
      seg.startVec = pre.frames[0] || vec;
      seg.travel = 0;
    }
    setPerformState("rest");
    return;
  }

  // state === "move"
  seg.frames.push(vec);
  seg.rest.push(rest);
  seg.times.push(now);
  seg.travel = Math.max(seg.travel, maxPoseDist(vec, seg.startVec));
  seg.stillCount = isStillNow(seg.frames, seg.times, now) ? seg.stillCount + 1 : 0;
  if (rest) seg.restCount++;
  else seg.restCount = 0;
  const settled = seg.restCount >= REST_SETTLE_FRAMES || seg.stillCount >= STILL_CONSEC;
  if (settled && seg.travel >= MIN_TRAVEL) {
    closeSegment(now);
    return;
  }
  // Settled but never travelled: a twitch armed the capture. Wait a little in
  // case the movement is slow to build, then drop it without matching.
  if (seg.restCount >= FALSE_START_FRAMES || seg.stillCount >= FALSE_START_FRAMES) {
    seg = newSeg();
    setPerformState("rest");
    return;
  }
  if (now - seg.startedAt > MAX_SEG_MS) { seg = newSeg(); setPerformState("rest"); return; }
  setPerformState("move");
}

function closeSegment(now) {
  const frames = trimCapture(seg.frames, seg.rest, seg.times);
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
  statusEl.textContent = triggerMode === "manual" ? "● ready — hold to capture" : "● watching — perform a code";
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
  statusEl.textContent = "Get ready…";
  for (let i = 3; i >= 1; i--) { countdownEl.textContent = i; await sleep(600); }

  const manual = triggerMode === "manual";
  teach = {
    word, manual,
    state: "prime",          // prime -> ready -> capturing (manual jumps straight to capturing)
    buf: [], frames: [], rest: [], times: [], restCount: 0, stillCount: 0,
    startVec: null, travel: 0,
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
    setTeachMsg("Hold still (or cover your face with one hand), perform your movement, then cover your face or hold still to finish.", "");
  }
}

// One frame of a teaching capture. Mirrors Perform: wait for rest, start on
// movement, end when the body returns to rest. No fixed time limit.
function teachStep(vec, rest, now) {
  const t = teach;
  if (t.manual) {
    t.frames.push(vec);
    t.rest.push(rest);
    t.times.push(now);
    return;
  }
  if (t.state === "prime") {
    // Settle first: a hand over the face, or simply holding still for a beat.
    pushIdle(t.buf, vec, rest, now);
    const spans = t.buf.length > 1 && now - t.buf[0].t >= STILL_WIN_LONG_MS;
    if (rest || (spans && !isMovingNow(t.buf, now))) t.state = "ready";
    return;
  }
  if (t.state === "ready") {
    pushIdle(t.buf, vec, rest, now);
    if (isMovingNow(t.buf, now)) {
      const pre = prerollFrom(t.buf, now);
      t.state = "capturing";
      t.frames = pre.frames;
      t.rest = pre.rest;
      t.times = pre.times;
      t.restCount = 0;
      t.stillCount = 0;
      t.startVec = pre.frames[0] || vec;
      t.travel = 0;
      t.startedAt = now;
    }
    return;
  }
  // capturing
  t.frames.push(vec);
  t.rest.push(rest);
  t.times.push(now);
  t.travel = Math.max(t.travel, maxPoseDist(vec, t.startVec));
  t.stillCount = isStillNow(t.frames, t.times, now) ? t.stillCount + 1 : 0;
  if (rest) t.restCount++;
  else t.restCount = 0;
  const settled = t.restCount >= REST_SETTLE_FRAMES || t.stillCount >= STILL_CONSEC;
  if (settled && t.travel >= MIN_TRAVEL) {
    finishTeach(now, false);
    return;
  }
  // Settled but never travelled: false start. Go back to waiting for the real
  // movement instead of failing the whole recording with an error.
  if (t.restCount >= FALSE_START_FRAMES || t.stillCount >= FALSE_START_FRAMES) {
    t.state = "ready";
    t.buf = [];
    t.frames = []; t.rest = []; t.times = [];
    t.restCount = 0; t.stillCount = 0; t.travel = 0;
  }
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
    statusEl.textContent = "Can't see your shoulders and hips. Adjust your framing.";
    return;
  }
  if (t.manual || t.state === "capturing") {
    countdownEl.textContent = "REC";
    countdownEl.classList.add("rec");
    statusEl.textContent = t.manual
      ? "Recording. Click Stop & save when done."
      : rest
        ? "Recording. Keep your hand over your face to finish…"
        : "Recording. When your movement is done, hold still or cover your face and it saves itself.";
    return;
  }
  countdownEl.classList.remove("rec");
  if (t.state === "prime") {
    countdownEl.textContent = "REST";
    statusEl.textContent = "Step 1 of 2: hold still for a moment (a hand over your face works too).";
  } else { // ready
    countdownEl.textContent = "MOVE";
    statusEl.textContent = "Step 2 of 2: perform your movement now, then hold still.";
  }
}

function finishTeach(now, timedOut = false) {
  const t = teach;
  teach = null;
  countdownEl.hidden = true;
  countdownEl.classList.remove("rec");
  recordBtn.disabled = false;
  recordBtn.textContent = "Record movement";
  setPerformState("rest");

  // Trim leading rest and trailing rest/stillness so every code starts and
  // ends where the movement actually happened.
  const core = t.frames.length ? trimCapture(t.frames, t.rest, t.times) : [];

  // Same travel gate as segmentation, applied to manual and timed-out
  // recordings too: a capture that never really went anywhere is not a code.
  let travel = 0;
  for (const f of core) travel = Math.max(travel, maxPoseDist(f, core[0]));

  if (core.length < 3 || travel < MIN_TRAVEL) {
    setTeachMsg("No clear movement captured. Check the skeleton overlay is tracking your whole upper body, then try a bigger movement.", "warn");
    return;
  }
  const durMs = Math.max(300, Math.round(now - t.startedAt));
  const seq = resampleSeq(core, FIXED_LEN);
  templates.push({ id: newId(), word: t.word, seq, durMs, createdAt: Date.now() });
  saveTemplates();
  renderCodeList();
  const n = templates.filter((x) => x.word.toLowerCase() === t.word.toLowerCase()).length;
  let msg = n > 1
    ? `Saved example ${n} for “${t.word}”. More examples improve recognition.`
    : `Saved “${t.word}”. Switch to Perform to try it.`;
  if (timedOut) msg += " (Recording hit its time cap; neither stillness nor a hand over your face was detected to end it.)";
  setTeachMsg(msg, "ok");
  wordInput.value = "";
}

function cancelTeach() {
  teach = null;
  countdownEl.hidden = true;
  countdownEl.classList.remove("rec");
  recordBtn.disabled = false;
  recordBtn.textContent = "Record movement";
  setPerformState("rest");
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

function startPlayback(word) {
  const items = templates.filter((t) => t.word.toLowerCase() === word.toLowerCase());
  if (items.length === 0) return;
  playback = { word, items, idx: 0, t0: performance.now() };
  renderCodeList();
}

function stopPlayback() {
  playback = null;
  if (landmarker) setPerformState("rest");
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
    ghost.push({ x: GHOST_CX + vx * GHOST_SCALE, y: GHOST_CY + vy * GHOST_SCALE, visibility: 1 });
  }

  if (!du) du = new DrawingUtils(octx);
  du.drawConnectors(ghost, PoseLandmarker.POSE_CONNECTIONS, {
    color: "rgba(255,195,113,0.95)",
    lineWidth: 5,
  });
  du.drawLandmarks(ghost, { color: "#ffffff", radius: 4, lineWidth: 1 });

  statusEl.textContent =
    `Playing “${pb.word}”` + (pb.items.length > 1 ? ` (example ${pb.idx + 1}/${pb.items.length})` : "");
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
    const isPlaying = playback && playback.word.toLowerCase() === g.word.toLowerCase();
    const w = encodeURIComponent(g.word);
    const li = document.createElement("li");
    li.className = "code-item";
    li.innerHTML = `
      <div class="code-info" data-act="play" data-word="${w}" title="Play this movement">
        <div class="word">${escapeHtml(g.word)}<span class="count">${count} example${count > 1 ? "s" : ""}</span></div>
        <div class="meta">${new Date(last).toLocaleDateString()} · click to play</div>
      </div>
      <div class="row-actions">
        <button class="btn small ${isPlaying ? "playing" : ""}" data-act="play" data-word="${w}">${isPlaying ? "Stop" : "Play"}</button>
        <button class="btn small" data-act="rename" data-word="${w}">Rename</button>
        <button class="btn small danger" data-act="del" data-word="${w}">Delete</button>
      </div>`;
    codeList.appendChild(li);
  }
}

codeList.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const act = btn.dataset.act;
  const word = decodeURIComponent(btn.dataset.word || "");
  const matches = (t) => t.word.toLowerCase() === word.toLowerCase();
  if (act === "play") {
    if (playback && playback.word.toLowerCase() === word.toLowerCase()) stopPlayback();
    else startPlayback(word);
  } else if (act === "del") {
    stopPlayback();
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
modelSel.addEventListener("change", () => switchModel(modelSel.value));

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
  modelSel.value = modelChoice;
  renderCodeList();
  renderPhrase();

  // The pose engine is a ~15 MB first-time download (wasm + model). Kick it off
  // in parallel with the camera permission so the two overlap, and reassure the
  // user it is downloading, not frozen. The browser caches it after first load.
  let ready = false;
  statusEl.textContent = "Requesting camera + downloading the pose model (first load can take a minute)…";
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
