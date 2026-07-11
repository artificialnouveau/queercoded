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
// with no gesture in between). Kept LOW: when hands swing out of frame the
// model's wrist estimates freeze or clamp, so a big real movement can measure
// as a small one; standing still measures ~0.05, so this still rejects it.
const MIN_TRAVEL = 0.17;
const MAX_TEACH_MS = 22000;       // safety cap so a teaching capture can't hang
const MIN_SEG_FRAMES = 4;         // ignore too-short blips
const COOLDOWN_MS = 1200;         // min gap before the same word fires again
// Teaching starts and stops on the hand-over-face pose, each held for a visible
// 3-2-1 countdown, so a brief brush of the face does nothing.
const START_HOLD_MS = 3000;
const STOP_HOLD_MS = 3000;

// ---------- DOM ----------
const video = document.getElementById("video");
const videoWrap = document.querySelector(".video-wrap");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const teachPreview = document.getElementById("teachPreview");
const statusEl = document.getElementById("status");
const bigWord = document.getElementById("bigWord");
const countdownEl = document.getElementById("countdown");
const bigStatus = document.getElementById("bigStatus");
const modelSel = document.getElementById("modelSel");

const threshInput = document.getElementById("thresh");
const threshVal = document.getElementById("threshVal");
const bestWordEl = document.getElementById("bestWord");
const matchPctEl = document.getElementById("matchPct");
const barEl = document.getElementById("bar");
const performListEl = document.getElementById("performList");
const closestHintEl = document.getElementById("closestHint");
const introHint = document.getElementById("introHint");
const phraseEl = document.getElementById("phrase");
const clearPhraseBtn = document.getElementById("clearPhrase");
const triggerModeSel = document.getElementById("triggerMode");
const holdBtn = document.getElementById("holdBtn");
const soundToggle = document.getElementById("soundToggle");
const speakToggle = document.getElementById("speakToggle");
const speakPhraseBtn = document.getElementById("speakPhrase");
const undoWordBtn = document.getElementById("undoWord");
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
let lastFireAt = 0;
let lastFiredWord = "";
let phrase = [];

// Perform options / manual capture
let triggerMode = "auto";     // "auto" (rest pose) | "manual" (hold)
let soundOn = true;
let speakOn = true;
// Per-word recognition thresholds, auto-calibrated from each word's examples.
// Rebuilt whenever the code set or algorithm family changes. See matchAndFire.
let wordStats = new Map();
let manualCapturing = false;
let manualFrames = [];
let audioCtx = null;
let playback = null;          // {word, items, idx, t0} ghost playback of a saved code
let ghostPreviewTimer = null; // delayed post-save ghost replay
// Latest live-body anchor {cx, cy, torso, at}, used to project the playback
// ghost onto the performer instead of a fixed spot on screen.
let liveFrame = null;

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

// "Hand over face" is the teach trigger, and WHICH hand matters: the RIGHT
// hand starts a recording, the LEFT hand stops and saves it. Splitting the two
// across hands removes every ambiguity of the old single-gesture design (which
// countdown am I looking at? did my stop just re-arm a start?).
//
// Chosen over hands-on-hips because a close camera framing often crops or
// barely sees the hips, while the face is always solidly tracked, and it is a
// distinct pose that a dance movement is unlikely to pass through slowly.
//
// The face anchor averages whichever of nose/ears are still visible, with a
// LOW visibility gate: the covering hand itself occludes the nose, and a
// strict gate here would make the pose undetectable exactly when held.
// Hysteresis (looser exit than enter) stops boundary flicker.
const REST_ENTER = 0.5;   // wrist-to-face distance (in shoulder widths) to enter
const REST_EXIT = 0.72;   // distance to leave once on the face
// Horizontal-centering gate. The pose tracks the WRIST, which sits below the
// face when the palm covers it, so the radius alone can't tell "palm over face"
// from "hand beside the face" (both put a wrist within range). A covering hand's
// wrist is roughly under the face centre; a hand at the side is offset sideways.
// So the wrist must also be within this horizontal distance of the face centre.
const CENTER_X_ENTER = 0.4; // |wrist.x - face.x| in shoulder widths to enter
const CENTER_X_EXIT = 0.55; // and to stay (hysteresis)
const FACE_LMS = [0, 7, 8]; // nose + ears
// The trip to and from the face is not part of the gesture. Frames at either
// end of a capture where a wrist is within this radius of the face are
// trimmed, so a code spans the movement itself, not the trigger transitions.
const NEAR_FACE_R = 1.0;
// A hand only flips on/off the face after this many consecutive frames agree,
// on top of the enter/exit hysteresis. Kills single-frame flicker when the
// covering hand makes the face landmarks jump.
const REST_DEBOUNCE = 2;
// Landmark indices are from the SUBJECT's perspective: 15 = their left wrist,
// 16 = their right wrist. The video is mirrored, so on screen the R label
// appears on the side where the viewer sees their right hand.
const handOnFace = {
  left: { on: false, streak: 0, unseenSince: 0 },
  right: { on: false, streak: 0, unseenSince: 0 },
};
// A covering palm often occludes its own wrist, so the wrist landmark drops
// out exactly while the hand IS on the face. A wrist that goes unseen while
// on the face is treated as still there for this long, so a hold does not
// break just because the hand did its job of covering.
const UNSEEN_STICKY_MS = 600;
let restInfo = null;      // per-frame info for drawing the face target circle

function updateHandsOnFace(lms, now) {
  const ls = lms[11], rs = lms[12];
  restInfo = null;
  const face = FACE_LMS.map((i) => lms[i]).filter((p) => (p?.visibility ?? 0) > 0.2);
  if (face.length === 0 || (ls?.visibility ?? 0) < 0.35 || (rs?.visibility ?? 0) < 0.35) {
    handOnFace.left.on = handOnFace.right.on = false;
    handOnFace.left.streak = handOnFace.right.streak = 0;
    return { left: false, right: false };
  }
  const anchor = {
    x: face.reduce((s, p) => s + p.x, 0) / face.length,
    y: face.reduce((s, p) => s + p.y, 0) / face.length,
  };
  const scale = Math.hypot(ls.x - rs.x, ls.y - rs.y) || 1e-6; // shoulder width
  // A wrist counts as "on the face" only when it is both close enough AND
  // roughly under the face centre horizontally. Debounced per hand, and an
  // unseen wrist stays briefly "on" (leaving the face requires being SEEN
  // away from it, not just disappearing behind it).
  const check = (w, st) => {
    const rThr = st.on ? REST_EXIT : REST_ENTER;
    const xThr = st.on ? CENTER_X_EXIT : CENTER_X_ENTER;
    let inRange = false, d = Infinity;
    if ((w?.visibility ?? 0) > 0.2) {
      st.unseenSince = 0;
      d = Math.hypot(w.x - anchor.x, w.y - anchor.y) / scale;
      const dx = Math.abs(w.x - anchor.x) / scale;
      inRange = d < rThr && dx < xThr;
    } else if (st.on) {
      if (!st.unseenSince) st.unseenSince = now;
      inRange = now - st.unseenSince < UNSEEN_STICKY_MS;
    }
    if (inRange !== st.on) {
      st.streak++;
      if (st.streak >= REST_DEBOUNCE) { st.on = inRange; st.streak = 0; }
    } else {
      st.streak = 0;
    }
    return d;
  };
  const dL = check(lms[15], handOnFace.left);
  const dR = check(lms[16], handOnFace.right);
  restInfo = { anchor, scale, d: Math.min(dL, dR), on: handOnFace.left.on || handOnFace.right.on };
  return { left: handOnFace.left.on, right: handOnFace.right.on };
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
  const { anchor, scale, on } = restInfo;
  const r = REST_ENTER * scale * overlay.width * 0.9;
  const inRange = on;
  octx.beginPath();
  octx.arc(anchor.x * overlay.width, anchor.y * overlay.height, r, 0, Math.PI * 2);
  octx.strokeStyle = inRange ? "rgba(236,255,0,0.95)" : "rgba(255,255,255,0.5)";
  octx.lineWidth = inRange ? 4 : 2;
  octx.stroke();
  if (inRange) {
    octx.fillStyle = "rgba(236,255,0,0.15)";
    octx.fill();
  }
}

// Big R / L letters on the wrists during a teach: RIGHT hand starts the
// recording, LEFT hand stops it. The hand whose turn it is glows yellow.
// The overlay canvas is CSS-mirrored, so text must be drawn locally flipped
// (scale(-1,1)) or the letters would read backwards.
function drawHandLabel(w, letter, active) {
  if (!w || (w.visibility ?? 0) < 0.3) return;
  let x = w.x * overlay.width;
  let y = w.y * overlay.height - 12;
  // Keep the letter clear of the face target circle: when the wrist is at or
  // near the face, push the label just outside the circle, away from its
  // centre, so it never covers the target the hand is aiming for.
  if (restInfo) {
    const ax = restInfo.anchor.x * overlay.width;
    const ay = restInfo.anchor.y * overlay.height;
    const rPx = REST_ENTER * restInfo.scale * overlay.width * 0.9;
    const dx = x - ax, dy = y - ay;
    const dist = Math.hypot(dx, dy);
    const clear = rPx + 30;
    if (dist < clear) {
      const ux = dist > 1 ? dx / dist : (letter === "R" ? 1 : -1);
      const uy = dist > 1 ? dy / dist : 0;
      x = ax + ux * clear;
      y = ay + uy * clear;
    }
  }
  octx.save();
  octx.translate(x, y);
  octx.scale(-1, 1); // cancel the CSS mirror so the letter reads correctly
  octx.font = `900 ${active ? 52 : 34}px system-ui, sans-serif`;
  octx.textAlign = "center";
  octx.textBaseline = "bottom";
  octx.lineWidth = active ? 8 : 5;
  octx.strokeStyle = "rgba(0,0,0,0.7)";
  octx.strokeText(letter, 0, 0);
  octx.fillStyle = active ? "#ECFF00" : "rgba(255,255,255,0.85)";
  octx.fillText(letter, 0, 0);
  octx.restore();
}
function drawHandLabels(lms) {
  const capturing = teach.state === "capturing";
  drawHandLabel(lms[16], "R", !capturing); // subject's right hand: start
  drawHandLabel(lms[15], "L", capturing);  // subject's left hand: stop and save
}

// Per-landmark weights for matching. All 33 points equally weighted let the
// face and torso (which barely move) dilute the wrists and ankles (which carry
// the dance), so expressive extremities count ~3x, mid-limbs 1.5x, torso 1x,
// and the near-rigid face is mostly ignored. Indices are BlazePose slots; the
// COCO-17 mapping fills 0, 2, 5, 7, 8, 11-16, 23-28, so its slots are covered.
const LM_WEIGHT = new Array(NUM_LMS).fill(1);
for (let i = 0; i <= 10; i++) LM_WEIGHT[i] = 0.3;             // face
for (const i of [13, 14, 25, 26]) LM_WEIGHT[i] = 1.5;         // elbows, knees
for (const i of [15, 16, 17, 18, 19, 20, 21, 22]) LM_WEIGHT[i] = 3; // wrists, hands
for (const i of [27, 28, 29, 30, 31, 32]) LM_WEIGHT[i] = 3;   // ankles, feet
const LM_WEIGHT_SUM = LM_WEIGHT.reduce((s, w) => s + w, 0);

// Weighted mean per-landmark euclidean distance between two normalized poses.
function poseDist(a, b) {
  let s = 0;
  for (let i = 0; i * 2 < a.length; i++) {
    s += LM_WEIGHT[i] * Math.hypot(a[i * 2] - b[i * 2], a[i * 2 + 1] - b[i * 2 + 1]);
  }
  return s / LM_WEIGHT_SUM;
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
function saveTemplates() {
  localStorage.setItem(STORE_KEY, JSON.stringify(templates));
  recomputeWordStats();
}
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
  recomputeWordStats(); // thresholds are per-family
  renderPerformable();  // performable list is per-family
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
  moving = false;
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
function drawSkeleton(pose, connections, boneColor, dotColor, boneWidth = 3, dotR = 3) {
  const VIS = 0.3;
  octx.strokeStyle = boneColor;
  octx.lineWidth = boneWidth;
  octx.lineCap = "round";
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
    octx.arc(p.x * overlay.width, p.y * overlay.height, dotR, 0, Math.PI * 2);
    octx.fill();
  }
}

// ---------- Riso live treatment ----------
// The vintage red/yellow print + glitch look on the performer. It is ON while a
// pose is being saved (Teach recording) and FLASHES for a moment when a move is
// matched (Perform), so the person on screen looks the way a saved/matched pose
// card looks. Applied to the whole frame (all algorithms), driven every frame
// from updateCaptureOverlay so it can't get stuck on.
let risoFlashUntil = 0;
let risoFlashTimer = null;
function flashRiso(now, ms = 800) {
  risoFlashUntil = now + ms;
  videoWrap.classList.add("flash");
  clearTimeout(risoFlashTimer);
  risoFlashTimer = setTimeout(() => videoWrap.classList.remove("flash"), ms);
}

// ---------- Saved-pose figures ----------
// Render the stored skeleton coordinates as riso silhouette figures (a red body
// with a yellow offset, like the printed pose cards). Nothing but coordinates is
// used, so no webcam image is ever stored or drawn. Slots shared by every
// algorithm family (BlazePose 33 and the COCO-17 mapping) so figures draw the
// same regardless of which model taught the code.
const FIG_BONES = [
  [11, 13], [13, 15], [12, 14], [14, 16], // arms
  [23, 25], [25, 27], [24, 26], [26, 28], // legs
  [11, 12], [23, 24], [11, 23], [12, 24], // shoulders, hips, sides
];
function figPoint(frame, i, cx, cy, sc) {
  const x = frame[i * 2], y = frame[i * 2 + 1];
  // A landmark a 17-point code never filled stays exactly at the origin.
  const filled = x !== 0 || y !== 0;
  // Mirror x so the figure faces the same way as the mirrored live video.
  return { x: cx - x * sc, y: cy + y * sc, v: filled };
}
function paintFigure(ctx, frame, cx, cy, sc, s, color) {
  const P = (i) => figPoint(frame, i, cx, cy, sc);
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = s * 0.15;
  // Torso as a filled quad so the body reads as a solid silhouette.
  const torso = [P(11), P(12), P(24), P(23)].filter((p) => p.v);
  if (torso.length >= 3) {
    ctx.beginPath();
    ctx.moveTo(torso[0].x, torso[0].y);
    for (const p of torso.slice(1)) ctx.lineTo(p.x, p.y);
    ctx.closePath();
    ctx.fill();
  }
  // Limbs as thick round-capped strokes.
  ctx.beginPath();
  for (const [a, b] of FIG_BONES) {
    const A = P(a), B = P(b);
    if (!A.v || !B.v) continue;
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
  }
  ctx.stroke();
  // Head: a disc at the nose, or just above the shoulders if the nose is unseen.
  const nose = P(0), ls = P(11), rs = P(12);
  const hx = nose.v ? nose.x : (ls.x + rs.x) / 2;
  const hy = nose.v ? nose.y : (ls.y + rs.y) / 2 - s * 0.14;
  ctx.beginPath();
  ctx.arc(hx, hy, s * 0.12, 0, Math.PI * 2);
  ctx.fill();
}
function drawFigureCell(ctx, frame, x0, s) {
  const cx = x0 + s * 0.5;
  const cy = s * 0.52;     // hips near centre so legs and raised arms both fit
  const sc = s * 0.18;     // torso length in px
  const off = s * 0.05;    // yellow print offset
  // Yellow layer first (offset), then red on top: the riso registration look.
  ctx.save();
  ctx.translate(off, off * 0.7);
  paintFigure(ctx, frame, cx, cy, sc, s, "#ECFF00");
  ctx.restore();
  paintFigure(ctx, frame, cx, cy, sc, s, "#FF002A");
}
// A row of `count` evenly-spaced frames from a stored seq, as one canvas.
function buildPoseStrip(seq, { count = 3, size = 74 } = {}) {
  const canvas = document.createElement("canvas");
  canvas.className = "pose-strip";
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = size * count * dpr;
  canvas.height = size * dpr;
  canvas.style.width = size * count + "px";
  canvas.style.height = size + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  for (let n = 0; n < count; n++) {
    const idx = count === 1 ? 0 : Math.round((n * (seq.length - 1)) / (count - 1));
    drawFigureCell(ctx, seq[idx], n * size, size);
  }
  return canvas;
}

// ---------- Main loop ----------
// Async because MoveNet and YOLO detect asynchronously; `inflight` keeps frames
// from overlapping. Detection is throttled to ~30fps, which is plenty for
// gesture capture and roughly halves GPU/battery load versus running every
// animation frame.
const MIN_FRAME_MS = 33;
let inflight = false;
let lastDetectAt = 0;
async function loop() {
  // Schedule the next frame FIRST, so a thrown error inside a frame can never
  // stop the loop (that is what froze detection). Errors are logged, not fatal.
  requestAnimationFrame(loop);
  const t = performance.now();
  if (ready && backend && !inflight && video.readyState >= 2 && t - lastDetectAt >= MIN_FRAME_MS) {
    lastDetectAt = t;
    inflight = true;
    try { await runFrame(); }
    catch (e) { console.error("frame error:", e); }
    finally { inflight = false; }
  }
}

async function runFrame() {
  {
    const now = performance.now();
    let poses = [];
    try { poses = await backend.detect(video, now); }
    catch (e) { console.warn("detect failed:", e); }
    octx.clearRect(0, 0, overlay.width, overlay.height);

    let bodyVisible = false;
    const lms = pickMainPose(poses);
    if (lms) {
      drawSkeleton(lms, backend.connections, "rgba(255,0,42,0.9)", "#ECFF00");

      if (keyLandmarksVisible(lms)) {
        bodyVisible = true;
        const vec = normalizePose(lms);
        // Live hip-centre and torso length, so playback can be drawn on the
        // performer's actual body (same normalization the stored seq uses).
        const lh = lms[23], rh = lms[24], ls = lms[11], rs = lms[12];
        if (lh && rh && ls && rs) {
          const cx = (lh.x + rh.x) / 2, cy = (lh.y + rh.y) / 2;
          const shx = (ls.x + rs.x) / 2, shy = (ls.y + rs.y) / 2;
          liveFrame = { cx, cy, torso: Math.hypot(shx - cx, shy - cy) || 1e-6, at: now };
        }
        const hands = updateHandsOnFace(lms, now);
        const nearNow = isNearFace();
        if (teach) {
          teachStep(vec, hands, nearNow, now);
        } else if (triggerMode === "manual") {
          if (manualCapturing) manualFrames.push(vec);
        } else {
          // Perform is continuous but fires only when a whole move completes.
          performStep(vec, now);
        }
        // Face target circle + R/L hand labels belong to Teach only.
        if (teach && !teach.manual && !playback) { drawRestTargets(); drawHandLabels(lms); }
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

    // The countdown timers only advance on frames where the body is tracked.
    // Without this, losing tracking mid-hold let the DISPLAYED countdown reach
    // zero while the state machine (which only runs on tracked frames) never
    // started the capture, so the first countdown appeared to do nothing.
    if (teach) {
      const dt = teach.lastTickAt ? now - teach.lastTickAt : 0;
      if (!bodyVisible && dt > 0) {
        if (teach.holdSince) teach.holdSince += dt;
        if (teach.stopSince) teach.stopSince += dt;
        if (teach.lastOn) teach.lastOn += dt;
        if (teach.stopLastOn) teach.stopLastOn += dt;
      }
      teach.lastTickAt = now;
    }

    // The teach overlay is driven here, EVERY frame, not inside the
    // detection-gated branch. Otherwise losing sight of the body freezes the
    // on-screen state while the state machine waits.
    if (teach) updateTeachUI(bodyVisible);

    // Big countdown / REC overlay, driven from the live capture phase. Runs
    // after updateTeachUI so it wins the status line during active phases.
    updateCaptureOverlay(now);

    // Ghost playback of a saved code, drawn on top of the live view.
    if (playback) drawPlayback(now);

    // Safety cap so a teaching capture can never hang, even if the body leaves
    // the frame mid-movement.
    if (teach && teach.state === "capturing" && now - teach.startedAt > MAX_TEACH_MS) {
      finishTeach(now, true);
    }
  }
}

// ---------- Continuous performance recognition ----------
// Perform watches continuously but recognizes a movement only once it COMPLETES:
// motion boundaries segment each move, and the whole finished segment is matched
// against saved codes. A partial move in progress updates the meter but never
// fires. When a completed move matches, its label is printed and spoken.
const PERF_BUF_MS = 8000;         // rolling history kept
// A completed move must have travelled this far. Matches teach's MIN_TRAVEL
// reasoning: hands that swing out of frame under-measure, so keep it low; the
// energy gate and per-word thresholds do the real still-body rejection.
const MIN_ACTIVE_TRAVEL = 0.18;
// Motion-boundary detection: a move begins when the fastest landmark's speed
// (displacement over SPEED_WIN_MS) crosses MOVE_START, and ends once speed has
// stayed below MOVE_END for SETTLE_MS. Matching (and firing) happens ONLY at
// the end of a move, on the whole segment, so a partial move can't fire early.
const SPEED_WIN_MS = 130;
const MOVE_START = 0.13;
const MOVE_END = 0.06;
const SETTLE_MS = 260;
const MIN_MOVE_MS = 350;
const MAX_MOVE_MS = 7000;
const perfBuf = [];               // {vec, t}
let moving = false, moveStart = 0, lastActive = 0;

function perfPush(vec, now) {
  perfBuf.push({ vec, t: now });
  while (perfBuf.length && perfBuf[0].t < now - PERF_BUF_MS) perfBuf.shift();
}

// Pose vectors buffered within [t0, t1].
function framesInRange(t0, t1) {
  const out = [];
  for (const e of perfBuf) if (e.t >= t0 && e.t <= t1) out.push(e.vec);
  return out;
}

// Fastest-landmark speed right now: displacement over the last ~SPEED_WIN_MS.
function speedNow(now) {
  const last = perfBuf[perfBuf.length - 1];
  if (!last) return 0;
  for (let i = perfBuf.length - 2; i >= 0; i--) {
    if (now - perfBuf[i].t >= SPEED_WIN_MS) return maxPoseDist(last.vec, perfBuf[i].vec);
  }
  return 0;
}

// Motion energy of a sequence: mean frame-to-frame pose change. A neutral
// stance has near-zero energy; a dance move has plenty. Codes usually START
// and END in a neutral pose, so DTW alone can warp a static resting body onto
// a code's neutral endpoints cheaply and false-fire while someone just stands
// there. Requiring the live segment's energy to be comparable to the code's
// (ENERGY_RATIO_MIN) makes stillness unmatchable against real movement.
function seqEnergy(seq) {
  if (seq.length < 2) return 0;
  let s = 0;
  for (let i = 1; i < seq.length; i++) s += poseDist(seq[i - 1], seq[i]);
  return s / (seq.length - 1);
}
const ENERGY_RATIO_MIN = 0.35; // live vs code energy: min(a,b)/max(a,b) must exceed this
const tmplEnergy = new Map();  // cached per template id (never persisted)
function energyOf(t) {
  let e = tmplEnergy.get(t.id);
  if (e == null) { e = seqEnergy(t.seq); tmplEnergy.set(t.id, e); }
  return e;
}
function energyCompatible(eLive, t) {
  const eT = energyOf(t);
  const hi = Math.max(eLive, eT), lo = Math.min(eLive, eT);
  return hi < 1e-6 || lo / hi >= ENERGY_RATIO_MIN;
}

// Best distance per word for a candidate segment (current algorithm family).
function scoreSegment(frames) {
  const pool = templates.filter((t) => (t.family || "blaze") === currentFamily);
  if (pool.length === 0) {
    bestWordEl.textContent = templates.length ? "none for this algorithm" : "none yet";
    return null;
  }
  if (frames.length < MIN_SEG_FRAMES) return null;
  const live = resampleSeq(frames, FIXED_LEN);
  const eLive = seqEnergy(live);
  const perWord = new Map();
  for (const t of pool) {
    if (!energyCompatible(eLive, t)) continue; // stillness can't match movement
    const d = dtw(live, t.seq);
    const k = t.word.toLowerCase();
    if (!perWord.has(k) || d < perWord.get(k).dist) perWord.set(k, { word: t.word, dist: d });
  }
  if (perWord.size === 0) return null;
  return [...perWord.values()].sort((a, b) => a.dist - b.dist);
}

// Update the meter/closest hint from a ranked list; returns the top comparison.
function showScore(ranked) {
  const best = ranked[0], second = ranked[1];
  const thresh = thresholdFor(best.word);
  const pct = Math.max(0, Math.min(100, (1 - best.dist / thresh) * 100));
  bestWordEl.textContent = best.word;
  matchPctEl.textContent = Math.round(pct) + "%";
  barEl.style.width = pct + "%";
  if (pct >= 25) showClosest(best.word, pct); else hideClosest();
  return { best, second, thresh };
}

// One Perform frame: track motion, show live feedback, and fire ONLY when a
// move completes (settles), matched over the whole segment.
function performStep(vec, now) {
  perfPush(vec, now);
  const sp = speedNow(now);

  if (!moving) {
    if (sp > MOVE_START) { moving = true; moveStart = now - SPEED_WIN_MS; lastActive = now; }
    else { barEl.style.width = "0%"; hideClosest(); }
    return;
  }

  if (sp > MOVE_END) lastActive = now;
  // Live feedback on the in-progress move, but never fires. Idle drift at a
  // resting stance easily crosses MOVE_START, so feedback (meter, closest
  // hint) waits until the segment has lasted and TRAVELLED like a real move;
  // otherwise standing still keeps flashing "closest" guesses.
  const soFar = framesInRange(moveStart, now);
  if (now - moveStart >= MIN_MOVE_MS && travelOf(soFar) >= MIN_ACTIVE_TRAVEL) {
    const inProgress = scoreSegment(soFar);
    if (inProgress) showScore(inProgress);
  } else {
    barEl.style.width = "0%";
    hideClosest();
  }

  const ended = now - lastActive >= SETTLE_MS;
  const tooLong = now - moveStart >= MAX_MOVE_MS;
  if (!ended && !tooLong) return;

  // Move complete: match the whole segment (start to when motion stopped).
  moving = false;
  const segFrames = framesInRange(moveStart, lastActive);
  if (now - moveStart < MIN_MOVE_MS || travelOf(segFrames) < MIN_ACTIVE_TRAVEL) {
    barEl.style.width = "0%"; hideClosest(); return;
  }
  const ranked = scoreSegment(segFrames);
  if (!ranked) { barEl.style.width = "0%"; hideClosest(); return; }
  const { best, second, thresh } = showScore(ranked);
  const ambiguous = second && (second.dist - best.dist) < AMBIG_GAP_FRAC * thresh;
  if (best.dist < thresh && !ambiguous) {
    const ok = best.word !== lastFiredWord || now - lastFireAt > COOLDOWN_MS;
    if (ok) fireWord(best.word, now);
  }
}

function showClosest(word, pct) {
  closestHintEl.hidden = false;
  closestHintEl.textContent = `closest: ${word} · ${Math.round(pct)}%`;
  closestHintEl.style.opacity = (0.35 + 0.6 * pct / 100).toFixed(2);
}
function hideClosest() { closestHintEl.hidden = true; }

const DEFAULT_SENS = 0.28;   // slider midpoint the auto thresholds scale against
const AMBIG_GAP_FRAC = 0.18; // second-best must be at least this much farther

// Recompute per-word thresholds from the spread of each word's own examples.
// A word whose examples are very consistent gets a tight threshold; a loose
// word gets a looser one. Words with a single example fall back to the slider.
function recomputeWordStats() {
  wordStats = new Map();
  const byWord = new Map();
  for (const t of templates) {
    if ((t.family || "blaze") !== currentFamily) continue;
    const k = t.word.toLowerCase();
    if (!byWord.has(k)) byWord.set(k, []);
    byWord.get(k).push(t);
  }
  for (const [k, items] of byWord) {
    if (items.length < 2) continue;
    const dists = [];
    for (let i = 0; i < items.length; i++)
      for (let j = i + 1; j < items.length; j++)
        dists.push(dtw(items[i].seq, items[j].seq));
    dists.sort((a, b) => a - b);
    const med = dists[Math.floor(dists.length / 2)];
    // A live performance sits farther from any example than examples sit from
    // each other, so allow roughly 2.4x the typical inter-example distance.
    wordStats.set(k, Math.max(0.1, Math.min(0.6, med * 2.4)));
  }
}

// Effective threshold for a word: its auto value (if any) scaled by where the
// user has the global sensitivity slider, else the slider value itself.
function thresholdFor(word) {
  const base = parseFloat(threshInput.value);
  const auto = wordStats.get(word.toLowerCase());
  if (auto == null) return base;
  return Math.max(0.06, Math.min(0.6, auto * (base / DEFAULT_SENS)));
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
  const eLive = seqEnergy(live);
  // Best (smallest) distance per distinct word.
  const perWord = new Map();
  for (const t of pool) {
    if (!energyCompatible(eLive, t)) continue; // stillness can't match movement
    const k = t.word.toLowerCase();
    const d = dtw(live, t.seq);
    if (!perWord.has(k) || d < perWord.get(k).dist) perWord.set(k, { word: t.word, dist: d });
  }
  if (perWord.size === 0) {
    bestWordEl.textContent = "—";
    matchPctEl.textContent = "—";
    barEl.style.width = "0%";
    return;
  }
  const ranked = [...perWord.values()].sort((a, b) => a.dist - b.dist);
  const best = ranked[0], second = ranked[1];
  const thresh = thresholdFor(best.word);

  const pct = isFinite(best.dist) ? Math.max(0, Math.min(100, (1 - best.dist / thresh) * 100)) : 0;
  bestWordEl.textContent = best.word ?? "—";
  matchPctEl.textContent = Math.round(pct) + "%";
  barEl.style.width = pct + "%";

  // Ambiguous when a different word is nearly as close: skip rather than guess.
  const ambiguous = second && (second.dist - best.dist) < AMBIG_GAP_FRAC * thresh;
  if (best.dist < thresh && !ambiguous) {
    const ok = best.word !== lastFiredWord || now - lastFireAt > COOLDOWN_MS;
    if (ok) fireWord(best.word, now);
  }
}

// Status text for the Perform tab.
function setPerformState() {
  if (!ready) return;
  if (triggerMode === "manual") {
    statusEl.textContent = manualCapturing ? "● capturing movement…" : "● ready. Hold the button to capture.";
    return;
  }
  statusEl.textContent = "● watching — perform a saved code and its label will appear.";
}

function fireWord(word, now) {
  lastFireAt = now;
  lastFiredWord = word;
  // Reveal word, ping and riso flash together, timed to when the speech
  // engine actually starts talking, so eye and ear get the word at once.
  const reveal = () => {
    showBigWord(word);
    flashRiso(performance.now());
    ping();
  };
  if (!speak(word, reveal)) reveal(); // speech off or unavailable: show now
  phrase.push(word);
  renderPhrase();
}

// Speak text aloud with the Web Speech API. Cancels any in-progress speech so
// rapid matches stay snappy rather than queueing up. `onStart` fires when the
// engine audibly begins (its startup lag is easily 100-500ms), with a fallback
// timer in case the engine never reports starting. Returns false when speech
// is off or unsupported so the caller can act immediately instead.
function speak(text, onStart) {
  if (!speakOn || !("speechSynthesis" in window) || !text) return false;
  try {
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1; u.pitch = 1;
    if (onStart) {
      let done = false;
      const go = () => { if (!done) { done = true; onStart(); } };
      u.onstart = go;
      setTimeout(go, 600); // engine hung or silent: don't hold the visual hostage
    }
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
    return true;
  } catch { return false; }
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

// Drive the big in-video overlay from the current capture phase: a start/stop
// 3-2-1 countdown, or REC while actually recording. Leaves SAVED flashes alone.
// Only Teach records, so the big REC / 3-2-1 countdown overlay is Teach-only.
// Perform is continuous and shows no recording overlay.
function updateCaptureOverlay(now) {
  const remain = (since, span) => Math.max(1, Math.ceil((span - (now - since)) / 1000));
  let kind = null, secs = 0;
  if (teach && teach.manual) kind = "rec";
  else if (teach && teach.state === "starting") { kind = "start"; secs = remain(teach.holdSince, START_HOLD_MS); }
  else if (teach && teach.state === "capturing") kind = teach.stopSince ? (secs = remain(teach.stopSince, STOP_HOLD_MS), "stop") : "rec";

  if (kind === "rec") {
    bigWord.classList.remove("show");
    if (!bigStatus.classList.contains("rec")) { bigStatus.textContent = "REC"; bigStatus.className = "big-status rec show"; }
  } else if (kind === "start" || kind === "stop") {
    bigWord.classList.remove("show");
    bigStatus.textContent = secs;
    bigStatus.className = "big-status show " + (kind === "stop" ? "stop" : "count");
  } else if (bigStatus.classList.contains("rec") || bigStatus.classList.contains("count") || bigStatus.classList.contains("stop")) {
    bigStatus.className = "big-status"; // leave SAVED flashes alone
  }

  // Riso treatment: on while actually recording a pose, plus a short window
  // after a match/save. Driven here every frame so it can never stick on.
  videoWrap.classList.toggle("riso", kind === "rec" || now < risoFlashUntil);

  if (!ready || !teach) return; // Perform manages its own status line
  if (kind === "start") statusEl.textContent = `Keep your face covered… recording in ${secs}`;
  else if (kind === "stop") statusEl.textContent = `Hold… saving in ${secs}`;
}

function renderPhrase() {
  phraseEl.innerHTML = phrase.length === 0
    ? '<span class="muted">Your matched words appear here…</span>'
    : phrase.map((w) => `<span class="chip">${escapeHtml(w)}</span>`).join("");
}

// ---------- Teaching (movement-delimited, like Perform) ----------
function startTeach() {
  const word = wordInput.value.trim();
  if (!word) { setTeachMsg("Type a word or phrase first.", "warn"); return; }
  if (teach) return;

  stopPlayback(); // clear any preview ghost from a previous save
  clearTimeout(ghostPreviewTimer); // and any replay still waiting to start
  clearTeachPreview();
  countdownEl.hidden = true;
  bigStatus.className = "big-status";

  const manual = triggerMode === "manual";
  teach = {
    word, manual,
    state: manual ? "capturing" : "prime", // prime -> starting -> capturing
    frames: [], near: [], onface: [], holdSince: 0, canStopL: false, canStopR: false, stopSince: 0,
    armed: false, // require the face to be uncovered once before starting
    startedAt: performance.now(),
  };
  if (manual) {
    recordBtn.textContent = "Stop & save";
    setTeachMsg("Recording… click Stop & save when your movement is done.", "");
  } else {
    recordBtn.textContent = "Cancel";
    setTeachMsg("Cover your face with your RIGHT hand (the big R) and hold for the 3-2-1 countdown to start. Perform, then cover your face with your LEFT hand (the big L) to stop and save.", "");
  }
}

// A hand may drop off the face for up to this long mid-hold (tracking flicker,
// slight slip) without resetting the countdown. Longer gaps cancel it.
const HOLD_GRACE_MS = 350;

// One frame of a teaching capture. RIGHT hand over the face held for the 3-2-1
// countdown STARTS recording; LEFT hand over the face held for the countdown
// STOPS and saves. Different hands for start and stop, so a stop can never
// re-arm a start and there is no ambiguity about which countdown is which.
// A hand can only stop the capture after IT has been off the face once, so the
// starting hand still resting there can't stop the capture instantly.
function teachStep(vec, hands, near, now) {
  const t = teach;
  if (t.manual) { t.frames.push(vec); t.near.push(near); t.onface.push(hands.left || hands.right); return; }
  if (t.state === "prime") {          // waiting for a fresh right-hand cover
    if (!hands.right) t.armed = true; // right hand must be off the face first...
    if (t.armed && hands.right) { t.state = "starting"; t.holdSince = now; t.lastOn = now; } // ...then cover
    return;
  }
  if (t.state === "starting") {       // start countdown; leaving the face cancels it
    if (hands.right) t.lastOn = now;
    else if (now - t.lastOn > HOLD_GRACE_MS) { t.state = "prime"; return; }
    if (now - t.holdSince >= START_HOLD_MS) {
      t.state = "capturing";
      t.frames = [vec]; t.near = [near]; t.onface = [hands.left || hands.right]; t.startedAt = now;
      t.canStopL = false; t.canStopR = false; t.stopSince = 0; t.stopLastOn = 0;
    }
    return;
  }
  // capturing: cover the face to stop. The LEFT hand is the cue (the big L),
  // but either hand completes the stop once IT has been off the face since the
  // capture began: with a palm on the face the model sometimes swaps which
  // wrist is which, and trusting only "left" made real stop holds break and
  // fall back to REC mid-countdown. The per-hand off-the-face requirement
  // still keeps the starting right hand from stopping the capture instantly.
  t.frames.push(vec);
  t.near.push(near);
  t.onface.push(hands.left || hands.right);
  if (!hands.left) t.canStopL = true;
  if (!hands.right) t.canStopR = true;
  const stopHeld = (t.canStopL && hands.left) || (t.canStopR && hands.right);
  if (stopHeld) {
    if (!t.stopSince) t.stopSince = now;
    t.stopLastOn = now;
    if (now - t.stopSince >= STOP_HOLD_MS) finishTeach(now, false);
  } else if (t.stopSince && now - t.stopLastOn > HOLD_GRACE_MS) {
    t.stopSince = 0;
  }
}

// On-screen overlay for the teach flow, driven every frame so it can also say
// when the body is NOT being detected instead of silently freezing.
function updateTeachUI(bodyVisible) {
  const t = teach;
  if (!t) return;
  if (!bodyVisible) {
    countdownEl.hidden = false;
    countdownEl.textContent = "?";
    statusEl.textContent = "Can't see your shoulders. Adjust your framing.";
    return;
  }
  // The big overlay owns the countdown / REC states; keep the pill hidden.
  if (t.manual || t.state === "starting" || t.state === "capturing") {
    countdownEl.hidden = true;
    if (t.manual) statusEl.textContent = "Recording. Click Stop & save when done.";
    else if (t.state === "capturing" && !t.stopSince) statusEl.textContent = "Recording. Cover your face with your LEFT hand (L) and hold to stop and save.";
    return;
  }
  // prime: waiting for a fresh right-hand cover
  countdownEl.hidden = false;
  if (!t.armed) {
    countdownEl.textContent = "…";
    statusEl.textContent = "Lower your right hand first, then cover your face with it to start the countdown.";
  } else {
    countdownEl.textContent = "COVER";
    statusEl.textContent = "Cover your face with your RIGHT hand (R) and hold to start the countdown.";
  }
}

function finishTeach(now, timedOut = false) {
  const t = teach;
  teach = null;
  countdownEl.hidden = true;
  recordBtn.disabled = false;
  recordBtn.textContent = "Record movement";
  setPerformState();

  // Trim the trigger holds off both ends so every code starts and ends where
  // the movement actually happened. The broad near-face radius catches the
  // approach and departure, but a dance may legitimately keep a hand near the
  // face the whole time, which would trim EVERYTHING; when that happens, retry
  // trimming only the strict hand-on-face frames before giving up.
  let core = trimNearFace(t.frames, t.near);
  if (core.length < 3 || travelOf(core) < MIN_TRAVEL) {
    core = trimNearFace(t.frames, t.onface || []);
  }

  // Same travel gate as segmentation, applied to manual and timed-out
  // recordings too: a capture that never really went anywhere is not a code.
  if (core.length < 3) {
    setTeachMsg("The recording ended up too short to save. Give it a beat between starting (R hand) and stopping (L hand), and check the skeleton overlay is tracking you.", "warn");
    return;
  }
  if (travelOf(core) < MIN_TRAVEL) {
    setTeachMsg("Too little movement was seen between start and stop. Tracking works best when your hands stay inside the frame; step back a little, or make the movement bigger.", "warn");
    return;
  }
  const durMs = Math.max(300, Math.round(now - t.startedAt));
  const seq = resampleSeq(core, FIXED_LEN);
  const tmpl = { id: newId(), word: t.word, seq, durMs, family: currentFamily, algo: algoChoice, createdAt: Date.now() };
  templates.push(tmpl);
  saveTemplates();
  renderCodeList();
  flashBigStatus("SAVED", "saved");
  flashRiso(now, 900);           // print/glitch the performer as the pose is saved
  showTeachPreview(seq);         // riso pose card of what was just captured
  // Ghost replay starts AFTER the SAVED/riso flash has cleared, so it isn't
  // lost in the noise of the save moment; the ghost is worth watching alone.
  clearTimeout(ghostPreviewTimer);
  ghostPreviewTimer = setTimeout(() => startPlaybackExample(tmpl.id), 1500);
  const n = templates.filter((x) => x.word.toLowerCase() === t.word.toLowerCase()).length;
  let msg = n > 1
    ? `Saved example ${n} for “${t.word}”. Click Record movement to add another, or type a new word.`
    : `Saved “${t.word}”. Click Record movement to add another example, or switch to Perform to try it.`;
  if (timedOut) msg += " (Recording hit its time cap; covering your face to finish was never detected.)";
  setTeachMsg(msg, "ok");
  // Keep the word so the next Record adds another example; select it for easy edit.
  wordInput.select();
}

// Show a riso pose card of the movement just saved, in the Teach panel.
function showTeachPreview(seq) {
  if (!teachPreview) return;
  teachPreview.innerHTML = '<div class="teach-preview-label">Saved as a pose card:</div>';
  try { teachPreview.appendChild(buildPoseStrip(seq, { count: 3, size: 84 })); } catch {}
  teachPreview.hidden = false;
}
function clearTeachPreview() {
  if (!teachPreview) return;
  teachPreview.hidden = true;
  teachPreview.innerHTML = "";
}

function cancelTeach() {
  teach = null;
  clearTimeout(ghostPreviewTimer);
  countdownEl.hidden = true;
  bigStatus.className = "big-status"; // clear any REC overlay
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
  clearTimeout(ghostPreviewTimer); // a pending post-save replay must not hijack this
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

  // Anchor the ghost on the live body when one is visible (so it dances on the
  // performer at their position and size), else fall back to a fixed spot.
  const live = liveFrame && (now - liveFrame.at < 300);
  const ax = live ? liveFrame.cx : GHOST_CX;
  const ay = live ? liveFrame.cy : GHOST_CY;
  const asc = live ? liveFrame.torso : GHOST_SCALE;

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
    ghost.push({ x: ax + vx * asc, y: ay + vy * asc, visibility: filled ? 1 : 0 });
  }

  // Draw with the bone set of the algorithm the code was taught on. A dark
  // underlay plus thick strokes keeps the ghost readable over a busy video.
  const conns = connectionsForFamily(cur.family || "blaze");
  drawSkeleton(ghost, conns, "rgba(0,0,0,0.55)", "rgba(0,0,0,0.55)", 11, 7);
  drawSkeleton(ghost, conns, "rgba(236,255,0,0.95)", "#FF002A", 5, 4);

  statusEl.textContent =
    `Playing “${pb.label}”` + (pb.items.length > 1 ? ` (example ${pb.idx + 1}/${pb.items.length})` : "");
}

// ---------- Codes list ----------
function renderCodeList() {
  renderPerformable(); // keep the Perform tab's code list in sync
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
    // A riso pose strip for the group, drawn from its most recent example.
    const rep = g.items.reduce((a, b) => (b.createdAt > a.createdAt ? b : a));
    try { li.insertBefore(buildPoseStrip(rep.seq), li.firstChild); } catch {}
    codeList.appendChild(li);
  }
}

const FAM_LABEL = { blaze: "BlazePose", movenet: "MoveNet", yolo: "YOLO" };

// Perform tab: the words you can currently perform (this algorithm family),
// as chips that preview the ghost. Empty state nudges to Teach, and notes when
// your codes live under a different algorithm.
function renderPerformable() {
  if (!performListEl) return;
  const words = [], seen = new Set(), otherFams = new Set();
  for (const t of templates) {
    const fam = t.family || "blaze";
    if (fam === currentFamily) {
      const k = t.word.toLowerCase();
      if (!seen.has(k)) { seen.add(k); words.push(t.word); }
    } else otherFams.add(fam);
  }
  if (words.length === 0) {
    const note = templates.length === 0
      ? "No codes yet."
      : `No codes for this algorithm. Yours were taught with ${[...otherFams].map((f) => FAM_LABEL[f] || f).join(", ")} — switch back in the algorithm picker, or teach new ones here.`;
    performListEl.innerHTML = `<span class="muted">${escapeHtml(note)}</span> <button class="btn tiny" data-goteach="1">Go to Teach</button>`;
    return;
  }
  performListEl.innerHTML = words
    .map((w) => `<button class="chip chip-btn" data-word="${encodeURIComponent(w)}" title="Preview “${escapeHtml(w)}”">${escapeHtml(w)}</button>`)
    .join("");
}

performListEl.addEventListener("click", (e) => {
  const goteach = e.target.closest("[data-goteach]");
  if (goteach) { activateTab(document.getElementById("tab-teach")); return; }
  const chip = e.target.closest("[data-word]");
  if (!chip) return;
  const word = decodeURIComponent(chip.dataset.word);
  if (playback && playback.key === "word:" + word.toLowerCase()) stopPlayback();
  else startPlayback(word);
});

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
// A code is valid only if its seq is FIXED_LEN frames of NUM_LMS*2 finite
// numbers. Guards against malformed or hostile files breaking the matcher.
function isValidTemplate(t) {
  if (!t || typeof t.word !== "string" || !t.word.trim()) return false;
  if (!Array.isArray(t.seq) || t.seq.length !== FIXED_LEN) return false;
  return t.seq.every((f) => Array.isArray(f) && f.length === NUM_LMS * 2 && f.every((n) => Number.isFinite(n)));
}
// Content signature for dedupe: word + family + coarsely-rounded sequence.
function codeSignature(t) {
  const fam = t.family || "blaze";
  return t.word.toLowerCase() + "|" + fam + "|" + t.seq.map((f) => f.map((n) => n.toFixed(2)).join(",")).join(";");
}

importFile.addEventListener("change", async () => {
  const file = importFile.files[0];
  if (!file) return;
  try {
    const incoming = JSON.parse(await file.text());
    if (!Array.isArray(incoming)) throw new Error("bad format");
    const seen = new Set(templates.map(codeSignature));
    let added = 0, skipped = 0, invalid = 0;
    for (const raw of incoming) {
      const t = { ...raw, family: raw.family || "blaze" };
      if (!isValidTemplate(t)) { invalid++; continue; }
      const sig = codeSignature(t);
      if (seen.has(sig)) { skipped++; continue; }
      seen.add(sig);
      templates.push({ ...t, id: newId(), createdAt: t.createdAt || Date.now() });
      added++;
    }
    saveTemplates();
    renderCodeList();
    let msg = `Imported ${added} code(s).`;
    if (skipped) msg += ` Skipped ${skipped} duplicate(s).`;
    if (invalid) msg += ` Ignored ${invalid} invalid entr${invalid === 1 ? "y" : "ies"}.`;
    alert(msg);
  } catch { alert("Could not read that file."); }
  importFile.value = "";
});
clearAllBtn.addEventListener("click", () => {
  if (confirm("Delete all saved codes? This cannot be undone.")) {
    templates = []; saveTemplates(); renderCodeList();
  }
});

// ---------- UI wiring ----------
const tabs = [...document.querySelectorAll(".tab")];
function activateTab(tab, focus = false) {
  const name = tab.dataset.tab;
  for (const t of tabs) {
    const on = t === tab;
    t.classList.toggle("is-active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
    t.tabIndex = on ? 0 : -1;
  }
  document.querySelectorAll(".tabpane").forEach((p) => { p.hidden = p.dataset.pane !== name; });
  // About / AlgoDance are reading pages: hide the video stage and let the panel
  // fill the width.
  const contentTab = name === "about" || name === "algodance";
  const layout = document.querySelector(".layout");
  layout.classList.toggle("content-full", contentTab);
  // AlgoDance is nothing but the PDF, filling the viewport edge to edge.
  layout.classList.toggle("pdf-full", name === "algodance");
  // Leaving the Teach tab mid-recording abandons it, so an active teach can
  // never keep "recording" (REC) into Perform. Also drop any manual capture.
  if (teach && name !== "teach") cancelTeach();
  manualCapturing = false;
  moving = false; // drop any half-finished Perform move
  hideClosest();
  if (ready) setPerformState();
  if (focus) tab.focus();
}
tabs.forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab));
  // Arrow / Home / End navigation, per the ARIA tabs pattern.
  tab.addEventListener("keydown", (e) => {
    const i = tabs.indexOf(tab);
    let j = -1;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % tabs.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") j = 0;
    else if (e.key === "End") j = tabs.length - 1;
    if (j >= 0) { e.preventDefault(); activateTab(tabs[j], true); }
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
  moving = false;
  setPerformState();
});
soundToggle.addEventListener("change", () => { soundOn = soundToggle.checked; });

// Landscape vs vertical presentation, remembered across visits. Vertical
// rotates the ENTIRE page 90 degrees clockwise (pure CSS on <body>), for a
// monitor physically turned on its side.
const orientSel = document.getElementById("orientSel");
const ORIENT_KEY = "queercoded.orientation.v1";
function applyOrientation(mode) {
  document.body.classList.toggle("rot90", mode === "rotated");
}
const savedOrient = localStorage.getItem(ORIENT_KEY) === "rotated" ? "rotated" : "landscape";
orientSel.value = savedOrient;
applyOrientation(savedOrient);
orientSel.addEventListener("change", () => {
  localStorage.setItem(ORIENT_KEY, orientSel.value);
  applyOrientation(orientSel.value);
});

speakToggle.addEventListener("change", () => {
  speakOn = speakToggle.checked;
  if (!speakOn && "speechSynthesis" in window) speechSynthesis.cancel();
});
speakPhraseBtn.addEventListener("click", () => {
  const wasOn = speakOn; speakOn = true; speak(phrase.join(", ")); speakOn = wasOn;
});
undoWordBtn.addEventListener("click", () => { phrase.pop(); renderPhrase(); });
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
function setTeachMsg(msg, cls) { teachMsg.textContent = msg; teachMsg.className = "teach-msg " + (cls || "muted"); }
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- Boot ----------
// First-run framing hint, shown once until dismissed.
const INTRO_KEY = "queercoded.seenIntro";
document.getElementById("introDismiss").addEventListener("click", () => {
  introHint.hidden = true;
  localStorage.setItem(INTRO_KEY, "1");
});

(async function boot() {
  // Pre-warm the speech engine: the voice list loads lazily, and asking for it
  // up front shaves the extra-long delay off the FIRST spoken match.
  if ("speechSynthesis" in window) speechSynthesis.getVoices();
  threshVal.textContent = threshInput.value;
  modelSel.value = algoChoice;
  renderCodeList();
  renderPhrase();
  if (!localStorage.getItem(INTRO_KEY)) introHint.hidden = false;

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
