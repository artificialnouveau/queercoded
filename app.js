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
const danceCount = document.getElementById("danceCount");
const countDots = document.getElementById("countDots");
const pbControls = document.getElementById("pbControls");
const bpmInput = document.getElementById("bpmInput");
const warmupOffer = document.getElementById("warmupOffer");
const routineListEl = document.getElementById("routineList");
const routineAddSel = document.getElementById("routineAdd");
const routineStartBtn = document.getElementById("routineStart");
const routineClearBtn = document.getElementById("routineClear");

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
let currentTab = "teach"; // recognition runs ONLY while the Perform tab is active
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
let wordBlends = new Map();   // word -> averaged multi-take pseudo-template
let manualCapturing = false;
let manualFrames = [];
let audioCtx = null;
let playback = null;          // ghost playback of a saved code (see startPlaybackItems)
let pbSpeed = 1;              // playback tempo: 0.5 / 0.75 / 1
let teachCountsOn = false;    // the teach metronome currently owns the count display
let rearmWord = null;         // hands-free next-take arming (same word, Teach tab)
let rearmSince = 0;
let warmupOffered = false;    // the Perform warm-up offer shows once per session
// Teach-on-a-beat tempo, persisted.
const BPM_KEY = "queercoded.bpm.v1";
function teachBpm() {
  const v = parseInt(bpmInput?.value, 10);
  return isFinite(v) ? Math.min(160, Math.max(60, v)) : 100;
}
if (bpmInput) {
  bpmInput.value = parseInt(localStorage.getItem(BPM_KEY), 10) || 100;
  bpmInput.addEventListener("change", () => {
    bpmInput.value = teachBpm();
    try { localStorage.setItem(BPM_KEY, String(teachBpm())); } catch {}
  });
}
let pbLoop = false;           // repeat playback until stopped
let pbSteps = false;          // pause on every count until the pose is hit
let liveLmsNow = null;        // {lms, at} latest smoothed live pose, for follow feedback
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

// Drop leading/trailing frames where the body is holding still. Teach
// captures include the stand-still moments after the move and before the
// stop cover; Perform segments are motion-bounded, so codes padded with
// stillness score worse against live segments AND skew the energy gate
// (a hold-diluted code looks "low energy" next to a motion-only segment).
function trimStillEnds(frames) {
  if (frames.length < 6) return frames;
  const STILL = 0.035; // per-step max-landmark motion below this = holding still
  let a = 0, b = frames.length;
  while (a < b - 4 && maxPoseDist(frames[a], frames[a + 1]) < STILL) a++;
  while (b > a + 4 && maxPoseDist(frames[b - 2], frames[b - 1]) < STILL) b--;
  return frames.slice(a, b);
}
// Stronger lead trim: frame-to-frame stillness misses slow idle jitter, so a
// long "getting ready" pause can survive into a code and stall its playback.
// Skip ahead until the pose has actually LEFT the opening pose, keeping a
// couple of frames of wind-up.
function trimLeadIn(frames) {
  if (frames.length < 8) return frames;
  const DEV = 0.12; // weighted max-landmark departure that counts as "moving"
  let a = 0;
  while (a < frames.length - 4 && maxPoseDist(frames[a], frames[0]) < DEV) a++;
  return frames.slice(Math.max(0, a - 2));
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
// Landmark x is normalized by image WIDTH and y by HEIGHT, so the same
// physical pose reads differently at different camera aspect ratios. Scale x
// into height units first ("square space") so codes match regardless of the
// camera's aspect; codes stored in this space carry sq:1 and older 4:3 codes
// are migrated once in loadTemplates.
function imgAspect() {
  return overlay.height > 0 ? overlay.width / overlay.height : 16 / 9;
}
function normalizePose(lms) {
  const A = imgAspect();
  const lHip = lms[23], rHip = lms[24], lSho = lms[11], rSho = lms[12];
  const cx = ((lHip.x + rHip.x) / 2) * A;
  const cy = (lHip.y + rHip.y) / 2;
  const shx = ((lSho.x + rSho.x) / 2) * A;
  const shy = (lSho.y + rSho.y) / 2;
  const torso = Math.hypot(shx - cx, shy - cy) || 1e-6;
  const out = new Array(NUM_LMS * 2);
  for (let i = 0; i < NUM_LMS; i++) {
    out[i * 2] = (lms[i].x * A - cx) / torso;
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
// The covering palm hides the very landmarks that detect it: nose/ears drop
// out as the hand arrives, which used to stall the countdown for seconds.
// The face anchor therefore survives brief occlusion (the head does not move
// while a palm covers it), remembered for up to this long.
const ANCHOR_STICKY_MS = 1500;
let lastFaceAnchor = null; // {x, y, at}
let restInfo = null;       // per-frame info for drawing the face target circle

function clearHandsOnFace() {
  handOnFace.left.on = handOnFace.right.on = false;
  handOnFace.left.streak = handOnFace.right.streak = 0;
  return { left: false, right: false };
}

function updateHandsOnFace(lms, now) {
  const ls = lms[11], rs = lms[12];
  restInfo = null;
  if ((ls?.visibility ?? 0) < 0.35 || (rs?.visibility ?? 0) < 0.35) return clearHandsOnFace();
  const face = FACE_LMS.map((i) => lms[i]).filter((p) => (p?.visibility ?? 0) > 0.15);
  let anchor;
  if (face.length > 0) {
    anchor = {
      x: face.reduce((s, p) => s + p.x, 0) / face.length,
      y: face.reduce((s, p) => s + p.y, 0) / face.length,
    };
    lastFaceAnchor = { x: anchor.x, y: anchor.y, at: now };
  } else if (lastFaceAnchor && now - lastFaceAnchor.at < ANCHOR_STICKY_MS) {
    anchor = lastFaceAnchor; // palm is hiding the face; keep the last known spot
  } else {
    return clearHandsOnFace();
  }
  const scale = Math.hypot(ls.x - rs.x, ls.y - rs.y) || 1e-6; // shoulder width
  // A wrist counts as "on the face" only when it is close enough, roughly
  // under the face centre horizontally, AND its forearm points UP (wrist
  // clearly above its own elbow). A covering palm has a vertical forearm;
  // crossed arms can put a wrist near the face but with a HORIZONTAL forearm,
  // which used to false-trigger the recording commands. Debounced per hand,
  // and an unseen wrist stays briefly "on" (leaving the face requires being
  // SEEN away from it, not just disappearing behind it).
  const check = (w, e, st) => {
    const rThr = st.on ? REST_EXIT : REST_ENTER;
    const xThr = st.on ? CENTER_X_EXIT : CENTER_X_ENTER;
    const upThr = st.on ? 0.08 : 0.2; // (elbow.y - wrist.y) in shoulder widths
    let inRange = false, d = Infinity;
    if ((w?.visibility ?? 0) > 0.15) {
      st.unseenSince = 0;
      d = Math.hypot(w.x - anchor.x, w.y - anchor.y) / scale;
      const dx = Math.abs(w.x - anchor.x) / scale;
      const forearmUp = (e?.visibility ?? 0) <= 0.2 || (e.y - w.y) / scale > upThr;
      inRange = d < rThr && dx < xThr && forearmUp;
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
  const dL = check(lms[15], lms[13], handOnFace.left);
  const dR = check(lms[16], lms[14], handOnFace.right);
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
  // Outside a teach (idle Teach tab) both letters show small and quiet, a
  // reminder of which hand does what; during a teach the hand whose turn it
  // is glows big and yellow.
  const capturing = teach?.state === "capturing";
  drawHandLabel(lms[16], "R", !!teach && !capturing); // subject's right hand: start
  drawHandLabel(lms[15], "L", !!teach && capturing);  // subject's left hand: stop and save
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
// Optional per-call weights (w, wSum) replace the global LM_WEIGHT: used for
// blend variance maps and live tracking confidence.
function poseDist(a, b, w, wSum) {
  const W = w || LM_WEIGHT;
  let s = 0;
  for (let i = 0; i * 2 < a.length; i++) {
    s += W[i] * Math.hypot(a[i * 2] - b[i * 2], a[i * 2 + 1] - b[i * 2 + 1]);
  }
  return s / (w ? wSum : LM_WEIGHT_SUM);
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
function dtw(A, B, w, wSum) {
  const m = A.length, n = B.length;
  const D = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(Infinity));
  D[0][0] = 0;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const c = poseDist(A[i - 1], B[j - 1], w, wSum);
      D[i][j] = c + Math.min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]);
    }
  }
  return D[m][n] / (m + n);
}
// Effective weights for matching against template `t`: the template's own
// variance map (blends) combined with the live tracking confidence, when
// either exists. Null means "use the plain global weights".
function effWeights(t, conf) {
  let w = t?.w || null;
  if (conf) {
    const base = w || LM_WEIGHT;
    w = base.map((x, i) => x * conf[i]);
  }
  if (!w) return null;
  let wSum = 0;
  for (const x of w) wSum += x;
  return wSum > 1e-6 ? { w, wSum } : null;
}

// ---------- Persistence (localStorage) ----------
function loadTemplates() {
  let list;
  try { list = JSON.parse(localStorage.getItem(STORE_KEY)) || []; }
  catch { list = []; }
  // Older codes were saved with the stand-still moments left in at both ends
  // (and a "getting ready" lead-in), which stalled playback openings, skewed
  // the energy gate, and scored worse against motion-bounded segments.
  // Normalize them once here, scaling the stored duration to match what was
  // cut so playback tempo stays true.
  for (const t of list) {
    if (!Array.isArray(t?.seq) || t.seq.length !== FIXED_LEN) continue;
    const trimmed = trimLeadIn(trimStillEnds(t.seq));
    if (trimmed.length >= 4 && trimmed.length < t.seq.length) {
      if (t.durMs) t.durMs = Math.max(300, Math.round(t.durMs * trimmed.length / t.seq.length));
      t.seq = resampleSeq(trimmed, FIXED_LEN);
    }
  }
  for (const t of list) migrateSq(t);
  return list;
}
// Codes taught before square-space normalization were captured at 640x480:
// stretch their x back to physical proportions (4:3) and renormalize by the
// frame's own torso length. One-time; marked with sq:1. Also applied to
// imported files from older exports.
function migrateSq(t) {
  if (!t || t.sq || !Array.isArray(t.seq)) return t;
  t.seq = t.seq.map((f) => {
    const v = f.slice();
    for (let i = 0; i < v.length; i += 2) v[i] *= 4 / 3;
    const sx = (v[22] + v[24]) / 2, sy = (v[23] + v[25]) / 2;   // shoulders 11,12
    const hx = (v[46] + v[48]) / 2, hy = (v[47] + v[49]) / 2;   // hips 23,24
    const torso = Math.hypot(sx - hx, sy - hy) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= torso;
    return v;
  });
  t.sq = 1;
  return t;
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
  updateBgBtn();        // background calibration is per-family too
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
  // Widescreen on landscape screens (more horizontal room for arms and
  // travel); tall on portrait phones so the fullscreen stream keeps the
  // whole body instead of cropping it. Cameras that cannot match return
  // their closest mode and the box crops to fit.
  const portrait = window.innerHeight > window.innerWidth;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: portrait ? 720 : 1280 },
      height: { ideal: portrait ? 1280 : 720 },
      facingMode: "user",
    },
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
function figPoint(frame, i, cx, cy, sc) {
  const x = frame[i * 2], y = frame[i * 2 + 1];
  // A landmark a 17-point code never filled stays exactly at the origin.
  const filled = x !== 0 || y !== 0;
  // Mirror x so the figure faces the same way as the mirrored live video.
  return { x: cx - x * sc, y: cy + y * sc, v: filled };
}
// One colour layer of the figure: a connected solid silhouette (torso fill
// plus continuous limb strokes), like the printed riso cards. No knockout
// seams: they chopped the body into disconnected lumps. When a recording has
// no usable leg data, a neutral standing pair of legs is drawn anyway so the
// figure always reads as a whole body.
// Generic riso body painter. `P(i)` returns a landmark in pixel space, `sc`
// is the torso length in pixels; all part widths derive from it. Used by the
// pose cards AND the live overlay so the whole app shares one body language.
// opts.defaultLegs draws a standing pair when leg data is missing (cards
// want a complete figure; the live overlay should not invent legs).
// opts.widthMul fattens every part without moving the joints, so the same
// pose can also be printed as a slightly larger halo layer behind the body.
function paintBody(ctx, P, sc, color, opts = {}) {
  const sw = sc * (opts.widthMul || 1); // widths only; lengths stay on sc
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  const disc = (p, r) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
  };
  // A limb segment as a TAPERED polygon (wide at the proximal joint, narrow
  // at the distal one) with round joints: fleshier than a uniform stroke, so
  // the figure reads as a body rather than a stick figure.
  const limb = (A, B, wA, wB) => {
    const dx = B.x - A.x, dy = B.y - A.y;
    const len = Math.hypot(dx, dy) || 1e-6;
    const nx = -dy / len, ny = dx / len;
    ctx.beginPath();
    ctx.moveTo(A.x + nx * wA / 2, A.y + ny * wA / 2);
    ctx.lineTo(B.x + nx * wB / 2, B.y + ny * wB / 2);
    ctx.lineTo(B.x - nx * wB / 2, B.y - ny * wB / 2);
    ctx.lineTo(A.x - nx * wA / 2, A.y - ny * wA / 2);
    ctx.closePath();
    ctx.fill();
    disc(A, wA / 2);
    disc(B, wB / 2);
  };
  const lsP = P(11), rsP = P(12), lhP = P(23), rhP = P(24);
  // Legs first (under the torso). A leg whose data is missing or collapsed
  // (taught with the lower body out of frame) is drawn standing straight
  // down, splayed slightly outward so default legs never fuse into a column.
  const legLen = sc * 0.95;
  const midHipX = lhP.v && rhP.v ? (lhP.x + rhP.x) / 2 : (lhP.v ? lhP.x : rhP.x);
  for (const [hip, knee, ank, toeI] of [[23, 25, 27, 31], [24, 26, 28, 32]]) {
    const H = P(hip);
    if (!H.v) continue;
    let K = P(knee), A = P(ank);
    const degenerate = !K.v || !A.v ||
      Math.hypot(A.x - H.x, A.y - H.y) < sc * 0.6;
    if (degenerate) {
      if (!opts.defaultLegs) continue; // the live overlay never invents legs
      const dir = H.x >= midHipX ? 1 : -1; // splay away from the midline
      K = { x: H.x + dir * sc * 0.1, y: H.y + legLen, v: true };
      A = { x: H.x + dir * sc * 0.18, y: H.y + legLen * 2, v: true };
    }
    limb(H, K, sw * 0.48, sw * 0.3);  // thigh
    limb(K, A, sw * 0.3, sw * 0.2);   // calf
    const T = P(toeI);
    const foot = !degenerate && T.v
      ? T
      : { x: A.x - sc * 0.3, y: A.y + sc * 0.1, v: true };
    limb(A, foot, sw * 0.2, sw * 0.16);
  }
  // Torso: filled quad softened with a thick round-jointed outline, so the
  // trunk has shoulders and hips instead of hard corners.
  const torso = [lsP, rsP, rhP, lhP].filter((p) => p.v);
  if (torso.length >= 3) {
    ctx.lineWidth = sw * 0.3;
    ctx.beginPath();
    ctx.moveTo(torso[0].x, torso[0].y);
    for (const p of torso.slice(1)) ctx.lineTo(p.x, p.y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  // Neck + head at human proportions.
  const nose = P(0);
  const hx = nose.v ? nose.x : (lsP.x + rsP.x) / 2;
  const hy = nose.v ? nose.y : (lsP.y + rsP.y) / 2 - sc * 0.55;
  if (lsP.v && rsP.v) {
    limb({ x: (lsP.x + rsP.x) / 2, y: (lsP.y + rsP.y) / 2 }, { x: hx, y: hy }, sw * 0.26, sw * 0.2);
  }
  disc({ x: hx, y: hy }, sw * 0.29);
  // Arms on top (they carry the pose): tapered shoulder-elbow-wrist ending
  // in a small hand.
  for (const [sho, elb, wri, k1, k2] of [[11, 13, 15, 17, 19], [12, 14, 16, 18, 20]]) {
    const S = P(sho), E = P(elb), W = P(wri);
    if (!S.v || !E.v || !W.v) continue;
    limb(S, E, sw * 0.36, sw * 0.26); // upper arm
    limb(E, W, sw * 0.26, sw * 0.18); // forearm
    const a = P(k1), b = P(k2);
    const n = (a.v ? 1 : 0) + (b.v ? 1 : 0);
    const hand = n
      ? { x: ((a.v ? a.x : 0) + (b.v ? b.x : 0)) / n, y: ((a.v ? a.y : 0) + (b.v ? b.y : 0)) / n }
      : { x: W.x + (W.x - E.x) * 0.22, y: W.y + (W.y - E.y) * 0.22 };
    limb(W, hand, sw * 0.18, sw * 0.22); // hand: a slight distal bulge
  }
}
// Card figure: paintBody over a stored normalized frame.
function paintFigure(ctx, frame, cx, cy, sc, color) {
  paintBody(ctx, (i) => figPoint(frame, i, cx, cy, sc), sc, color, { defaultLegs: true });
}
function drawFigureCell(ctx, frame, x0, s) {
  const cx = x0 + s * 0.5;
  const cy = s * 0.4;      // hips above centre so default legs fit below
  const sc = s * 0.19;     // torso length in px
  const off = s * 0.045;   // yellow print offset
  // Yellow layer first (offset), then red on top: the riso registration look.
  ctx.save();
  ctx.translate(off, off * 0.7);
  paintFigure(ctx, frame, cx, cy, sc, "#ECFF00");
  ctx.restore();
  paintFigure(ctx, frame, cx, cy, sc, "#FF002A");
}

// ---- Creative live presence: constellation + light trails ----
// No bones, no outline, nothing worn ON the performer. The tracked body is
// a constellation of soft glowing ink dots (left side yellow, right side
// red, head white) and the expressive points (head, hands, ankles) leave
// comet trails that fade in about a second: the feedback-trail language of
// TouchDesigner dance pieces, drawn additively so movement paints light.
const TRAIL_MS = 900;
const TRAIL_DEFS = [
  { i: 0,  rgb: "255,255,255" },
  { i: 15, rgb: "236,255,0" }, { i: 16, rgb: "255,0,42" },  // wrists
  { i: 27, rgb: "236,255,0" }, { i: 28, rgb: "255,0,42" },  // ankles
];
const ORB_DEFS = [
  { i: 0, rgb: "255,255,255", r: 0.16 },
  { i: 11, rgb: "236,255,0", r: 0.09 }, { i: 12, rgb: "255,0,42", r: 0.09 },
  { i: 13, rgb: "236,255,0", r: 0.08 }, { i: 14, rgb: "255,0,42", r: 0.08 },
  { i: 15, rgb: "236,255,0", r: 0.14 }, { i: 16, rgb: "255,0,42", r: 0.14 },
  { i: 23, rgb: "236,255,0", r: 0.09 }, { i: 24, rgb: "255,0,42", r: 0.09 },
  { i: 25, rgb: "236,255,0", r: 0.08 }, { i: 26, rgb: "255,0,42", r: 0.08 },
  { i: 27, rgb: "236,255,0", r: 0.11 }, { i: 28, rgb: "255,0,42", r: 0.11 },
];
const trails = new Map(); // landmark index -> [{x, y, t}]
function drawPresence(lms, now) {
  let sc = 0, P = null;
  if (lms) {
    const ls = lms[11], rs = lms[12], lh = lms[23], rh = lms[24];
    if (ls && rs) {
      const shx = ((ls.x + rs.x) / 2) * overlay.width;
      const shy = ((ls.y + rs.y) / 2) * overlay.height;
      const hx = (((lh?.x ?? ls.x) + (rh?.x ?? rs.x)) / 2) * overlay.width;
      const hy = (((lh?.y ?? ls.y) + (rh?.y ?? rs.y)) / 2) * overlay.height;
      sc = Math.hypot(shx - hx, shy - hy);
    }
    P = (i) => {
      const p = lms[i];
      return p && (p.visibility ?? 1) > 0.3
        ? { x: p.x * overlay.width, y: p.y * overlay.height }
        : null;
    };
  }
  // Record this frame's trail points; age out the old ones. Trails keep
  // fading even on untracked frames, so losing you never freezes a streak.
  for (const d of TRAIL_DEFS) {
    let arr = trails.get(d.i);
    if (!arr) trails.set(d.i, (arr = []));
    const p = P && sc > 8 ? P(d.i) : null;
    if (p) arr.push({ x: p.x, y: p.y, t: now });
    while (arr.length && now - arr[0].t > TRAIL_MS) arr.shift();
  }
  octx.save();
  octx.globalCompositeOperation = "lighter";
  octx.lineCap = "round";
  octx.lineJoin = "round";
  // Two passes per segment: a wide soft glow under a bright core.
  const wBase = Math.max(3, sc * 0.12);
  for (const d of TRAIL_DEFS) {
    const arr = trails.get(d.i);
    if (!arr || arr.length < 2) continue;
    for (let k = 1; k < arr.length; k++) {
      const a = arr[k - 1], b = arr[k];
      if (b.t - a.t > 250) continue; // tracking gap: do not bridge it
      const fade = Math.max(0, 1 - (now - b.t) / TRAIL_MS);
      octx.strokeStyle = `rgba(${d.rgb},${(0.1 * fade).toFixed(3)})`;
      octx.lineWidth = wBase * 2.4 * fade + 2;
      octx.beginPath(); octx.moveTo(a.x, a.y); octx.lineTo(b.x, b.y); octx.stroke();
      octx.strokeStyle = `rgba(${d.rgb},${(0.5 * fade).toFixed(3)})`;
      octx.lineWidth = wBase * fade + 1;
      octx.beginPath(); octx.moveTo(a.x, a.y); octx.lineTo(b.x, b.y); octx.stroke();
    }
  }
  // The constellation: a soft orb per joint, white-hot centre into ink.
  if (P && sc > 8) {
    for (const o of ORB_DEFS) {
      const p = P(o.i);
      if (!p) continue;
      const r = Math.max(4, sc * o.r);
      const g = octx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r);
      g.addColorStop(0, "rgba(255,255,255,0.9)");
      g.addColorStop(0.35, `rgba(${o.rgb},0.75)`);
      g.addColorStop(1, `rgba(${o.rgb},0)`);
      octx.fillStyle = g;
      octx.beginPath();
      octx.arc(p.x, p.y, r, 0, Math.PI * 2);
      octx.fill();
    }
  }
  octx.restore();
}

// ---- Riso print ghost ----
// The playback ghost is printed like the reference riso figures: a solid
// warm-red body with screenprint grain punched out of the ink, over a pale
// yellow halo that sits slightly off-register. The red body is built on an
// offscreen layer so the grain only erodes the ink, never the video below.
let inkCanvas = null, inkCtx = null, grainPattern = null;
function inkLayer() {
  if (!inkCanvas) {
    inkCanvas = document.createElement("canvas");
    inkCtx = inkCanvas.getContext("2d");
  }
  if (inkCanvas.width !== overlay.width || inkCanvas.height !== overlay.height) {
    inkCanvas.width = overlay.width;
    inkCanvas.height = overlay.height;
  }
  return inkCtx;
}
// A tiling noise texture, generated once. Used with destination-out so the
// ink gets the stippled, unevenly-soaked texture of a risograph pass.
function grainPat(ctx) {
  if (grainPattern) return grainPattern;
  const t = document.createElement("canvas");
  t.width = t.height = 160;
  const g = t.getContext("2d");
  const img = g.createImageData(160, 160);
  for (let i = 3; i < img.data.length; i += 4) {
    const r = Math.random();
    img.data[i] = r < 0.55 ? 0 : Math.floor((r - 0.55) / 0.45 * 235);
  }
  g.putImageData(img, 0, 0);
  grainPattern = ctx.createPattern(t, "repeat");
  return grainPattern;
}
// A 45-degree halftone dot screen, generated once and tiled. Punched out of
// the ink with destination-out it gives every layer the dotted tooth of a
// screened print.
let halftonePattern = null;
function halftonePat(ctx) {
  if (halftonePattern) return halftonePattern;
  const t = document.createElement("canvas");
  t.width = t.height = 96;
  const g = t.getContext("2d");
  g.fillStyle = "#000";
  const cell = 6;
  for (let y = 0, row = 0; y <= 96 + cell; y += cell, row++) {
    for (let x = (row % 2) * (cell / 2); x <= 96 + cell; x += cell) {
      g.beginPath();
      g.arc(x, y, 1.7, 0, Math.PI * 2);
      g.fill();
    }
  }
  halftonePattern = ctx.createPattern(t, "repeat");
  return halftonePattern;
}
// Screens whatever ink is on the layer: halftone dots plus two offset grain
// passes, so texture never tiles visibly and edges crumble a little.
function screenInk(g, dots, grain) {
  g.save();
  g.globalCompositeOperation = "destination-out";
  g.fillStyle = halftonePat(g);
  g.globalAlpha = dots;
  g.fillRect(0, 0, inkCanvas.width, inkCanvas.height);
  g.fillStyle = grainPat(g);
  g.globalAlpha = grain;
  g.fillRect(0, 0, inkCanvas.width, inkCanvas.height);
  g.translate(73, 41);
  g.globalAlpha = grain * 0.55;
  g.fillRect(-73, -41, inkCanvas.width, inkCanvas.height);
  g.restore();
}
// Paints one gouache/riso figure onto `octx`. `GP(i)` returns pixel-space
// points, `sc` is the torso length in pixels. Everything is FLAT ink: no
// gradients, no lighting. Depth comes from three layered shapes (yellow
// halo, red body, deep crimson overprint inside the form) and from the
// halftone/grain texture eaten out of each layer; jittered double-stamps
// make the silhouette edges wobble like a hand-pulled print instead of a
// clean vector line.
function paintRisoGhost(GP, sc) {
  const g = inkLayer();
  // Layer 1: pale yellow halo, off-register, edges roughened by the screen.
  g.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
  g.save();
  g.translate(sc * 0.11, sc * 0.07);
  paintBody(g, GP, sc, "#FFE95A", { widthMul: 1.14 });
  g.translate(sc * 0.025, -sc * 0.02); // jitter stamp: wobbly edge
  g.globalAlpha = 0.5;
  paintBody(g, GP, sc, "#FFE95A", { widthMul: 1.12 });
  g.restore();
  screenInk(g, 0.3, 0.4);
  octx.save();
  octx.globalAlpha = 0.85;
  octx.drawImage(inkCanvas, 0, 0);
  octx.restore();
  // Layer 2: the red body, double-stamped with a slight offset so the
  // contour is rough, then screened hard.
  g.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
  paintBody(g, GP, sc, "#E8452C");
  g.save();
  g.globalAlpha = 0.55;
  g.translate(sc * 0.02, -sc * 0.015);
  paintBody(g, GP, sc, "#E8452C", { widthMul: 1.04 });
  g.restore();
  // Layer 3: a deep crimson overprint INSIDE the form, a thinner flat shape
  // offset toward one side. This is all the shading a riso pass gets.
  g.save();
  g.globalAlpha = 0.55;
  g.translate(-sc * 0.035, sc * 0.03);
  paintBody(g, GP, sc, "#A8172E", { widthMul: 0.68 });
  g.restore();
  screenInk(g, 0.38, 0.5);
  octx.save();
  octx.globalAlpha = 0.94;
  octx.drawImage(inkCanvas, 0, 0);
  octx.restore();
}

// A row of `count` evenly-spaced frames from a stored seq, as one canvas.
function buildPoseStrip(seq, { count = 3, size = 96 } = {}) {
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

// ---------- One-Euro landmark smoothing ----------
// MediaPipe output jitters frame to frame, which shakes the drawn skeleton,
// pollutes captures, and trips the motion segmenter. A One-Euro filter
// smooths hard when the body is slow (kills jitter) and barely at all when
// it is fast (no lag on real dance moves). Applied to the tracked person's
// landmarks before anything else consumes them.
const EURO_MIN_CUTOFF = 1.5; // Hz: smoothing floor at rest
const EURO_BETA = 0.3;       // how quickly speed unlocks the filter
const EURO_D_CUTOFF = 1.0;   // Hz: derivative smoothing
const euro = { t: 0, x: null, dx: null, n: 0 };
function lowpassAlpha(dt, cutoff) {
  const r = 2 * Math.PI * cutoff * dt;
  return r / (r + 1);
}
function smoothPose(lms, now) {
  const t = now / 1000;
  const n = lms.length;
  // (Re)initialize after a tracking gap or a person switch (big shoulder jump).
  const jumped = euro.x && euro.n === n && lms[11] && euro.x[11]
    && Math.hypot(lms[11].x - euro.x[11].x, lms[11].y - euro.x[11].y) > 0.25;
  if (!euro.x || euro.n !== n || t - euro.t > 0.5 || jumped) {
    euro.x = lms.map((p) => ({ x: p?.x ?? 0, y: p?.y ?? 0 }));
    euro.dx = lms.map(() => ({ x: 0, y: 0 }));
    euro.n = n;
    euro.t = t;
    return lms;
  }
  const dt = Math.max(1e-3, t - euro.t);
  euro.t = t;
  const ad = lowpassAlpha(dt, EURO_D_CUTOFF);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const p = lms[i];
    if (!p) { out[i] = p; continue; }
    const prev = euro.x[i], d = euro.dx[i];
    d.x += ad * ((p.x - prev.x) / dt - d.x);
    d.y += ad * ((p.y - prev.y) / dt - d.y);
    prev.x += lowpassAlpha(dt, EURO_MIN_CUTOFF + EURO_BETA * Math.abs(d.x)) * (p.x - prev.x);
    prev.y += lowpassAlpha(dt, EURO_MIN_CUTOFF + EURO_BETA * Math.abs(d.y)) * (p.y - prev.y);
    out[i] = { x: prev.x, y: prev.y, visibility: p.visibility };
  }
  return out;
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
  // Reading tabs (About / AlgoDance) hide the video entirely: skip detection
  // there so the GPU and battery are not burned rendering nothing.
  const idleTab = currentTab === "about" || currentTab === "algodance";
  if (ready && backend && !inflight && !idleTab && video.readyState >= 2 && t - lastDetectAt >= MIN_FRAME_MS) {
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
    let lms = pickMainPose(poses);
    if (lms) lms = smoothPose(lms, now);
    // Constellation + light trails are the live look. Runs on untracked
    // frames too, so trails fade out instead of freezing when you leave.
    drawPresence(lms, now);
    if (lms) {
      liveLmsNow = { lms, at: now }; // follow-along feedback reads this
      if (keyLandmarksVisible(lms)) {
        bodyVisible = true;
        const vec = normalizePose(lms);
        if (playback?.capture) playback.capture.push(vec); // rehearse: your turn
        // Live hip-centre and torso length, so playback can be drawn on the
        // performer's actual body (same normalization the stored seq uses).
        const lh = lms[23], rh = lms[24], ls = lms[11], rs = lms[12];
        if (lh && rh && ls && rs) {
          const cx = (lh.x + rh.x) / 2, cy = (lh.y + rh.y) / 2;
          const shx = (ls.x + rs.x) / 2, shy = (ls.y + rs.y) / 2;
          // torso in SQUARE units (matching stored seqs); cx/cy stay in
          // image units for anchoring.
          const torso = Math.hypot((shx - cx) * imgAspect(), shy - cy) || 1e-6;
          liveFrame = { cx, cy, torso, at: now };
        }
        const hands = updateHandsOnFace(lms, now);
        const nearNow = isNearFace();
        // Hands carry most of a dance and most of the noise: an untracked
        // wrist gets a guessed position that pollutes captures and matches.
        // Teach only RECORDS frames, and Perform only matches, while at
        // least one wrist is USABLE: confidently seen, or clipped out the top
        // of the frame by a VERIFIABLY raised arm (the tracked elbow is at
        // shoulder height or higher and the wrist estimate points further
        // up). Arms-up moves (YMCA and friends) pass; a hallucinated
        // high-wrist estimate on a lowered arm cannot, because its elbow
        // hangs below the shoulders.
        const shoulderMidY = ((lms[11]?.y ?? 0) + (lms[12]?.y ?? 0)) / 2;
        const wristUsable = (w, e) =>
          (w?.visibility ?? 0) > 0.35 ||
          ((e?.visibility ?? 0) > 0.35 && !!w &&
            e.y < shoulderMidY + 0.05 && // the arm is genuinely raised
            w.y < e.y);                  // and the wrist continues upward
        const handVis = wristUsable(lms[15], lms[13]) || wristUsable(lms[16], lms[14]);
        if (teach) {
          teachStep(vec, hands, nearNow, now, handVis);
        } else if (currentTab === "perform" && !playback) {
          // Matching pauses while any ghost/routine/rehearse is on screen,
          // so following along never fires words by accident.
          // Recognition belongs to the Perform tab only: on Teach or Codes an
          // idle body must never fire (or even score) matches.
          if (bgCapture) {
            // Calibrating: collect idle movement, match nothing.
            bgCapture.frames.push(vec);
            const remain = Math.ceil((bgCapture.until - now) / 1000);
            statusEl.textContent = `Learning your idle movement… ${remain}s. Move casually; do NOT perform your codes.`;
            if (now >= bgCapture.until) finishBgCapture();
          } else if (triggerMode === "manual") {
            if (manualCapturing) manualFrames.push(vec); // deliberate capture: record everything
          } else if (handVis) {
            // Perform is continuous but fires only when a whole move completes.
            // Per-joint tracking confidence rides along so barely-seen joints
            // count less in matching.
            const conf = lms.map((p) => Math.max(0.2, Math.min(1, p?.visibility ?? 0.5)));
            performStep(vec, now, hands.left || hands.right, conf);
            if (perfHandsLost) { perfHandsLost = false; setPerformState(); }
          } else {
            moving = false;      // a half-tracked move is not evidence
            stillSince = 0;
            if (!perfHandsLost) {
              perfHandsLost = true;
              barEl.style.width = "0%";
              hideClosest();
            }
            statusEl.textContent = "Can't see your hands. Keep at least one hand in view; matching is paused.";
          }
        }
        // Face target circle + R/L hand labels belong to Teach only. The
        // R/L letters also show on the idle Teach tab (small, quiet) so you
        // see which hand starts and which stops BEFORE clicking Record.
        if (teach && !teach.manual && !playback) { drawRestTargets(); drawHandLabels(lms); }
        else if (!teach && currentTab === "teach" && !playback) {
          drawHandLabels(lms);
          // Between takes of the same word, covering the face with the RIGHT
          // hand (held ~half a second) starts the next take, no click needed.
          if (rearmWord && wordInput.value.trim().toLowerCase() === rearmWord) {
            if (hands.right) {
              if (!rearmSince) rearmSince = now;
              else if (now - rearmSince > 450) { rearmSince = 0; startTeach(); }
            } else {
              rearmSince = 0;
            }
          } else {
            rearmSince = 0;
          }
        }
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
        if (teach.rOnSince) teach.rOnSince += dt;
        if (teach.lOnSince) teach.lOnSince += dt;
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
const SETTLE_MS = 200;
const MIN_MOVE_MS = 350;
const MAX_MOVE_MS = 7000;
const perfBuf = [];               // {vec, t}
let moving = false, moveStart = 0, lastActive = 0;
let perfHandsLost = false;        // matching paused because no wrist is tracked

// ---------- Background class (idle-movement calibration) ----------
// The user records ~10s of their own ordinary movement (shifting, adjusting,
// crossing arms) as NEGATIVE examples, stored per algorithm family. A word
// then only fires when the live movement is closer to that word than to the
// user's own background, by a margin, which replaces guessing at absolute
// thresholds with a comparison against reality.
// v2: idle windows recorded in square-space normalization (v1 windows were
// aspect-distorted and are simply abandoned; recalibration takes 10s).
const BG_KEY = "queercoded.background.v2";
const BG_MARGIN = 0.85;        // best word must beat the background by this factor
const BG_CAPTURE_MS = 10000;
let bgStore = {};
try { bgStore = JSON.parse(localStorage.getItem(BG_KEY)) || {}; } catch { bgStore = {}; }
let bgCapture = null;          // {until, frames:[vec]} while calibrating
const bgBtn = document.getElementById("bgBtn");
function bgSeqs() { return bgStore[currentFamily] || []; }
function updateBgBtn() {
  if (bgBtn) bgBtn.textContent = bgSeqs().length
    ? "Recalibrate my idle movement (10s)"
    : "Calibrate: learn my idle movement (10s)";
}
function startBgCapture() {
  if (!ready || teach || bgCapture) return;
  bgCapture = { until: performance.now() + BG_CAPTURE_MS, frames: [] };
  bgBtn.disabled = true;
}
function finishBgCapture() {
  const frames = bgCapture.frames;
  bgCapture = null;
  bgBtn.disabled = false;
  // Overlapping ~2s windows become the background examples.
  const WIN = 60, STEP = 30;
  const seqs = [];
  for (let i = 0; i + WIN <= frames.length && seqs.length < 12; i += STEP) {
    seqs.push(resampleSeq(frames.slice(i, i + WIN), FIXED_LEN));
  }
  if (seqs.length < 2) {
    diag("calibration failed: too little tracked movement, try again");
    setPerformState();
    return;
  }
  bgStore[currentFamily] = seqs;
  try { localStorage.setItem(BG_KEY, JSON.stringify(bgStore)); } catch {}
  updateBgBtn();
  setPerformState();
  diag(`background calibrated: ${seqs.length} idle windows`);
}
// Smallest DTW distance from a live 20-frame sequence to any background window.
function bgDistSeq(live) {
  let m = Infinity;
  for (const s of bgSeqs()) {
    const d = dtw(live, s);
    if (d < m) m = d;
  }
  return m;
}
// Smallest pose distance from a single held pose to any background frame.
function bgDistPose(vec) {
  let m = Infinity;
  for (const s of bgSeqs()) {
    for (let i = 0; i < s.length; i += 2) {
      const d = poseDist(vec, s[i]);
      if (d < m) m = d;
    }
  }
  return m;
}

// ---------- Match diagnosis ----------
// A rolling, human-readable log of every decision the recognizer makes, shown
// in the Perform tab, so "it did not match" becomes "it lost to gate X by Y".
const diagBox = document.getElementById("diagBox");
const diagLines = [];
function diag(msg) {
  if (!diagBox) return;
  diagLines.push(`[${new Date().toLocaleTimeString()}] ${msg}`);
  while (diagLines.length > 12) diagLines.shift();
  diagBox.textContent = diagLines.join("\n");
}
const fmt = (n) => (isFinite(n) ? n.toFixed(3) : "none");

// The performer's neutral stance, learned as a slow average of the pose
// whenever the body is still. Codes start and end near neutral, so a segment
// that never DEVIATES far from the current neutral (a posture shift, a sway,
// settling after a move) is rejected outright: resting is never a dance move.
let restPose = null;
const REST_LEARN = 0.02;   // per-frame blend while still (~1s half-life at 30fps)
const REST_DEV_MIN = 0.2;  // a real move must stray at least this far from neutral
function learnRestPose(vec, sp) {
  if (moving || sp > MOVE_END) return;
  if (!restPose) { restPose = vec.slice(); return; }
  for (let i = 0; i < vec.length; i++) restPose[i] += (vec[i] - restPose[i]) * REST_LEARN;
}
function maxDevFrom(frames, ref) {
  let m = 0;
  for (const f of frames) { const d = poseDist(f, ref); if (d > m) m = d; }
  return m;
}
// After ANY word fires, nothing may fire for this long: the settle back to
// neutral is itself a motion and must not trigger a second (different) word.
// Short enough that a deliberate second pose right after the first still gets
// its own classification.
const FIRE_GAP_MS = 700;

// Every code of the current family PLUS each word's blend pseudo-template.
function matchPool() {
  const pool = templates.filter((t) => (t.family || "blaze") === currentFamily);
  for (const b of wordBlends.values()) pool.push(b);
  return pool;
}

function perfPush(vec, now, face, conf) {
  perfBuf.push({ vec, t: now, face: !!face, conf });
  while (perfBuf.length && perfBuf[0].t < now - PERF_BUF_MS) perfBuf.shift();
}
// Per-landmark tracking confidence averaged over a window: joints the model
// barely saw (hallucinated wrists, clipped legs) count less in matching.
function confInRange(t0, t1) {
  const acc = new Array(NUM_LMS).fill(0);
  let n = 0;
  for (const e of perfBuf) {
    if (e.t < t0 || e.t > t1 || !e.conf) continue;
    n++;
    for (let i = 0; i < NUM_LMS; i++) acc[i] += e.conf[i];
  }
  if (!n) return null;
  for (let i = 0; i < NUM_LMS; i++) acc[i] = Math.max(0.15, Math.min(1, acc[i] / n));
  return acc;
}

// Hand-on-face share of a window, and whether its tail ends on the face.
// Raising a hand to the face is the single most common idle gesture (and the
// app's own teach trigger), and it is skeletally an arm raise, so it kept
// "matching" arm codes. A segment that ends on the face, or mostly lives
// there, is a rest gesture, never a dance move.
function faceStats(t0, t1) {
  let n = 0, on = 0, tailOn = false;
  for (const e of perfBuf) {
    if (e.t < t0 || e.t > t1) continue;
    n++;
    if (e.face) { on++; if (e.t >= t1 - 300) tailOn = true; }
  }
  return { frac: n ? on / n : 0, tailOn };
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

// Peak excursion of a sequence from its own first frame (weighted pose
// distance). A code's excursion says how far that move actually leaves
// neutral; a live segment must reach a comparable share of it, or a
// near-neutral wobble is being pattern-matched onto a real move.
const tmplDev = new Map();
function devOfSeq(seq) {
  let m = 0;
  for (const f of seq) { const d = poseDist(f, seq[0]); if (d > m) m = d; }
  return m;
}
function devOf(t) {
  let d = tmplDev.get(t.id);
  if (d == null) { d = devOfSeq(t.seq); tmplDev.set(t.id, d); }
  return d;
}
const DEV_RATIO_MIN = 0.5; // live excursion must be at least half the code's

// Best distance per word for a candidate segment (current algorithm family,
// takes AND multi-take blends). `conf` (optional) down-weights joints the
// tracker barely saw during the segment.
function scoreSegment(frames, conf) {
  const pool = matchPool();
  if (pool.length === 0) {
    bestWordEl.textContent = templates.length ? "none for this algorithm" : "none yet";
    return null;
  }
  if (frames.length < MIN_SEG_FRAMES) return null;
  const live = resampleSeq(frames, FIXED_LEN);
  const eLive = seqEnergy(live);
  const devLive = devOfSeq(live);
  const perWord = new Map();
  for (const t of pool) {
    if (!energyCompatible(eLive, t)) continue;          // stillness can't match movement
    if (devLive < devOf(t) * DEV_RATIO_MIN) continue;   // near-neutral wobble can't match a real excursion
    const ew = effWeights(t, conf);
    const d = ew ? dtw(live, t.seq, ew.w, ew.wSum) : dtw(live, t.seq);
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

// ---------- Static-pose matching (the PRIMARY path) ----------
// Most codes are struck poses, not trajectories, so pose recognition leads:
// hold a clearly non-neutral pose for HOLD_MS and it is compared against
// EVERY frame of every code (the closest frame counts), which forgives how
// you got into the pose entirely. Runs continuously, even while the motion
// segmenter thinks the body is "moving", because stillness is tracked here
// by pose drift, not by the segmenter. One fire per hold; shifting to a new
// pose re-arms it. Neutral standing and hand-on-face can't fire.
const HOLD_MS = 450;
const HOLD_SHIFT = 0.09;     // pose change (weighted) that counts as a NEW hold
let stillSince = 0;
let holdFired = false;
let holdRefVec = null;
function holdDistToCode(vec, t, ew) {
  let m = Infinity;
  for (const f of t.seq) {
    const d = ew ? poseDist(vec, f, ew.w, ew.wSum) : poseDist(vec, f);
    if (d < m) m = d;
  }
  return m;
}
function holdStep(vec, now, face, conf) {
  if (face) { stillSince = now; holdFired = false; return; }
  if (!holdRefVec || poseDist(vec, holdRefVec) > HOLD_SHIFT) {
    holdRefVec = vec.slice(); // pose changed: a fresh hold begins
    stillSince = now;
    holdFired = false;
  }
  if (holdFired || !stillSince || now - stillSince < HOLD_MS) return false;
  // Neutral standing is not a pose.
  if (!restPose || poseDist(vec, restPose) < REST_DEV_MIN) return false;
  const pool = matchPool();
  if (pool.length === 0) return false;
  const perWord = new Map();
  for (const t of pool) {
    const d = holdDistToCode(vec, t, effWeights(t, conf));
    const k = t.word.toLowerCase();
    if (!perWord.has(k) || d < perWord.get(k).dist) perWord.set(k, { word: t.word, dist: d });
  }
  const ranked = [...perWord.values()].sort((a, b) => a.dist - b.dist);
  const best = ranked[0], second = ranked[1];
  const thresh = thresholdFor(best.word);
  const pct = Math.max(0, Math.min(100, (1 - best.dist / thresh) * 100));
  bestWordEl.textContent = best.word;
  matchPctEl.textContent = Math.round(pct) + "%";
  barEl.style.width = pct + "%";
  if (pct >= 25) showClosest(best.word, pct);
  const ambiguous = second && (second.dist - best.dist) < AMBIG_GAP_FRAC * thresh;
  // The held pose must resemble the word MORE than the user's own calibrated
  // idle movement (when a background exists).
  const bg = bgDistPose(vec);
  const beatsBg = !isFinite(bg) || best.dist < bg * BG_MARGIN;
  // Deliberately held poses fire at the full threshold (no extra margin):
  // holding still IS the confirmation.
  if (best.dist < thresh && !ambiguous && beatsBg) {
    const ok = now - lastFireAt > FIRE_GAP_MS
      && (best.word !== lastFiredWord || now - lastFireAt > COOLDOWN_MS);
    if (ok) {
      diag(`HOLD fired "${best.word}" d=${fmt(best.dist)} thr=${fmt(thresh)} bg=${fmt(bg)}`);
      holdFired = true; // once per hold; move to a new pose to fire again
      fireWord(best.word, now);
    }
  } else if (!holdDiagAt || now - holdDiagAt > 1500) {
    holdDiagAt = now;
    const why = ambiguous ? `ambiguous with "${second.word}" ${fmt(second.dist)}`
      : !beatsBg ? `background closer (bg=${fmt(bg)})`
      : `distance ${fmt(best.dist)} over threshold ${fmt(thresh)}`;
    diag(`hold "${best.word}" not fired: ${why}`);
    if (best.dist < thresh * 1.5) {
      showClosest(best.word, pct, coachTip(ambiguous ? "ambiguous" : !beatsBg ? "bg" : "far", second));
    }
  }
  return true;
}
let holdDiagAt = 0;

// One Perform frame: track motion, show live feedback, and fire ONLY when a
// move completes (settles), matched over the whole segment.
function performStep(vec, now, face, conf) {
  perfPush(vec, now, face, conf);
  const sp = speedNow(now);
  learnRestPose(vec, sp);

  // Pose-first: the hold matcher runs EVERY frame, tracking stillness by
  // pose drift rather than the motion segmenter, so a struck-and-held pose
  // is recognized even when limb jitter keeps the segmenter in "moving".
  const holding = holdStep(vec, now, face, conf);

  if (!moving) {
    if (sp > MOVE_START) {
      moving = true; moveStart = now - SPEED_WIN_MS; lastActive = now;
    } else if (!holding) {
      barEl.style.width = "0%";
      hideClosest();
    }
    return;
  }

  if (sp > MOVE_END) lastActive = now;
  // Live feedback on the in-progress move, but never fires. Idle drift at a
  // resting stance easily crosses MOVE_START, so feedback (meter, closest
  // hint) waits until the segment has lasted and TRAVELLED like a real move;
  // otherwise standing still keeps flashing "closest" guesses.
  const soFar = framesInRange(moveStart, now);
  const looksReal = now - moveStart >= MIN_MOVE_MS
    && !face
    && travelOf(soFar) >= MIN_ACTIVE_TRAVEL
    && (!restPose || maxDevFrom(soFar, restPose) >= REST_DEV_MIN);
  if (looksReal && !holding) {
    const inProgress = scoreSegment(soFar);
    if (inProgress) showScore(inProgress);
  } else if (!holding) {
    barEl.style.width = "0%";
    hideClosest();
  }

  const ended = now - lastActive >= SETTLE_MS;
  const tooLong = now - moveStart >= MAX_MOVE_MS;
  if (!ended && !tooLong) return;

  // Move complete: match the whole segment (start to when motion stopped).
  moving = false;
  const segFrames = framesInRange(moveStart, lastActive);
  const fs = faceStats(moveStart, lastActive);
  const gate =
    now - moveStart < MIN_MOVE_MS ? "too short" :
    travelOf(segFrames) < MIN_ACTIVE_TRAVEL ? "too little travel" :
    restPose && maxDevFrom(segFrames, restPose) < REST_DEV_MIN ? "stays near your resting stance" :
    fs.tailOn || fs.frac > 0.3 ? "reads as a hand-to-face rest gesture" : null;
  if (gate) {
    diag(`move rejected: ${gate}`);
    barEl.style.width = "0%"; hideClosest(); return;
  }
  let ranked = scoreSegment(segFrames, confInRange(moveStart, lastActive));
  let chosen = segFrames;
  // Also score recent WINDOWS at code-typical lengths (stillness included).
  // This covers two failure modes the bare motion segment cannot: a code
  // with an internal hold (raise, hold, lower) spans MORE than one motion
  // segment, and two poses chained without a full settle merge into one long
  // segment that scores closest to the FIRST pose. Beat-taught codes add
  // BEAT-QUANTIZED candidates: their length snapped to whole counts of
  // their own tempo, plus one count either side, so capture aligns to the
  // taught count structure instead of segmentation guesses. The best
  // interpretation wins.
  const fam = templates.filter((t) => (t.family || "blaze") === currentFamily);
  const cand = new Set();
  const durs = fam.map((t) => t.durMs || 2000).sort((x, y) => x - y);
  if (durs.length) cand.add(durs[Math.floor(durs.length / 2)]);
  for (const t of fam) {
    if (!t.bpm || !t.durMs) continue;
    const beatMs = 60000 / t.bpm;
    const q = Math.max(1, Math.round(t.durMs / beatMs)) * beatMs;
    for (const d of [q, q - beatMs, q + beatMs]) {
      if (d >= 500 && d <= 6000) cand.add(Math.round(d / 50) * 50);
    }
  }
  let tried = 0;
  for (const d of cand) {
    if (++tried > 8) break; // bound the per-settle matching cost
    const win = framesInRange(lastActive - d, lastActive);
    if (win.length < MIN_SEG_FRAMES) continue;
    const rankedWin = scoreSegment(win, confInRange(lastActive - d, lastActive));
    if (rankedWin && (!ranked || rankedWin[0].dist < ranked[0].dist)) { ranked = rankedWin; chosen = win; }
  }
  if (!ranked) {
    diag("move rejected: no code with comparable energy/excursion");
    barEl.style.width = "0%"; hideClosest(); return;
  }
  const { best, second, thresh } = showScore(ranked);
  const ambiguous = second && (second.dist - best.dist) < AMBIG_GAP_FRAC * thresh;
  // The movement must resemble the word MORE than the user's own calibrated
  // idle movement (when a background exists).
  const bg = bgDistSeq(resampleSeq(chosen, FIXED_LEN));
  const beatsBg = !isFinite(bg) || best.dist < bg * BG_MARGIN;
  if (best.dist < thresh * FIRE_MARGIN && !ambiguous && beatsBg) {
    const ok = now - lastFireAt > FIRE_GAP_MS
      && (best.word !== lastFiredWord || now - lastFireAt > COOLDOWN_MS);
    if (ok) {
      diag(`MOVE fired "${best.word}" d=${fmt(best.dist)} thr=${fmt(thresh)} bg=${fmt(bg)}`
        + (second ? ` 2nd "${second.word}" ${fmt(second.dist)}` : ""));
      fireWord(best.word, now);
    } else {
      diag(`move "${best.word}" matched but cooling down`);
    }
  } else {
    const why = ambiguous ? `ambiguous with "${second.word}" ${fmt(second.dist)}`
      : !beatsBg ? `background closer (bg=${fmt(bg)})`
      : `distance ${fmt(best.dist)} over ${fmt(thresh * FIRE_MARGIN)}`;
    diag(`move best "${best.word}": not fired, ${why}`);
    // Close but no fire: coach in plain words instead of leaving silence.
    if (best.dist < thresh * 1.5) {
      const pct2 = Math.max(0, Math.min(100, (1 - best.dist / thresh) * 100));
      showClosest(best.word, pct2, coachTip(ambiguous ? "ambiguous" : !beatsBg ? "bg" : "far", second));
    }
  }
}

// Translate a gate rejection into a human coaching hint.
function coachTip(gate, second) {
  if (gate === "ambiguous" && second) return `looks like “${second.word}” too, exaggerate the difference`;
  if (gate === "bg") return "too close to your idle movement, make it bigger and sharper";
  return "almost, finish the move cleanly and hold the last pose a beat";
}
function showClosest(word, pct, tip) {
  closestHintEl.hidden = false;
  closestHintEl.textContent = `closest: ${word} · ${Math.round(pct)}%` + (tip ? ` — ${tip}` : "");
  closestHintEl.style.opacity = (0.35 + 0.6 * pct / 100).toFixed(2);
}
function hideClosest() { closestHintEl.hidden = true; }

const DEFAULT_SENS = 0.28;    // slider midpoint the auto thresholds scale against
const AMBIG_GAP_FRAC = 0.25;  // second-best must be at least this much farther
const CROSS_WORD_FRAC = 0.55; // a match may use at most this share of the gap to the nearest OTHER word
const FIRE_MARGIN = 0.8;      // firing needs dist under this fraction of the threshold (display shows the full range)

// Recompute per-word calibration: how consistent a word's own examples are
// (auto threshold) AND how close it sits to every other word (cross cap).
// Consistent examples give a tight threshold; a word that lives near another
// word gets capped below the distance between them, so two different codes
// ("heart" vs "peace") can never blur into each other.
function recomputeWordStats() {
  wordStats = new Map();
  const byWord = new Map();
  for (const t of templates) {
    if ((t.family || "blaze") !== currentFamily) continue;
    const k = t.word.toLowerCase();
    if (!byWord.has(k)) byWord.set(k, []);
    byWord.get(k).push(t);
  }
  const entries = [...byWord.entries()];
  // Multi-take blending: for words with 2+ takes, build ONE averaged
  // template plus a per-landmark variance map. Joints the teacher performs
  // consistently across takes get full weight; wobbly joints count less. The
  // blend joins the match pool as a pseudo-template.
  wordBlends = new Map();
  tmplEnergy.clear(); // blend seqs change: drop cached energies/excursions
  tmplDev.clear();
  for (const [k, items] of entries) {
    if (items.length < 2) continue;
    const takes = items.map((t) => t.seq);
    const seq = [];
    const varSum = new Array(NUM_LMS).fill(0);
    for (let f = 0; f < FIXED_LEN; f++) {
      const mean = new Array(NUM_LMS * 2).fill(0);
      for (const s of takes) for (let i = 0; i < mean.length; i++) mean[i] += s[f][i];
      for (let i = 0; i < mean.length; i++) mean[i] /= takes.length;
      seq.push(mean);
      for (const s of takes) {
        for (let i = 0; i < NUM_LMS; i++) {
          varSum[i] += (s[f][i * 2] - mean[i * 2]) ** 2 + (s[f][i * 2 + 1] - mean[i * 2 + 1]) ** 2;
        }
      }
    }
    const n = FIXED_LEN * takes.length;
    const w = LM_WEIGHT.map((base, i) => base / (1 + 2.5 * Math.sqrt(varSum[i] / n)));
    let wSum = 0;
    for (const x of w) wSum += x;
    const durMs = Math.round(items.reduce((s, t) => s + (t.durMs || 2000), 0) / items.length);
    wordBlends.set(k, {
      id: "blend:" + k, word: items[0].word, seq, w, wSum, durMs,
      family: currentFamily, blend: true,
    });
  }
  for (const [k, items] of entries) {
    let auto = null;
    if (items.length >= 2) {
      const dists = [];
      for (let i = 0; i < items.length; i++)
        for (let j = i + 1; j < items.length; j++)
          dists.push(dtw(items[i].seq, items[j].seq));
      dists.sort((a, b) => a - b);
      const med = dists[Math.floor(dists.length / 2)];
      // A live performance sits farther from any example than examples sit
      // from each other, so allow roughly 2.4x the typical inter-example
      // distance. The ceiling stays firm: inconsistent examples must not
      // balloon a word's threshold until near-anything matches it.
      auto = Math.max(0.1, Math.min(0.42, med * 2.4));
    }
    // Smallest DTW distance from any example of this word to any example of
    // any OTHER word: the separation this word has to respect.
    let cross = Infinity;
    for (const [k2, items2] of entries) {
      if (k2 === k) continue;
      for (const a of items)
        for (const b of items2) {
          const d = dtw(a.seq, b.seq);
          if (d < cross) cross = d;
        }
    }
    wordStats.set(k, { auto, cross });
  }
}

// Effective threshold for a word: its auto value (if any) scaled by the
// sensitivity slider, hard-capped (DTW's path normalization understates
// still-vs-moving mismatch, so loose thresholds let standing "match" an arm
// move: 0.42, or 0.35 for single-example words, even with the slider maxed),
// and never beyond CROSS_WORD_FRAC of the gap to the nearest other word.
function thresholdFor(word) {
  const base = parseFloat(threshInput.value);
  const s = wordStats.get(word.toLowerCase());
  let t = s?.auto == null
    ? Math.min(base, 0.35)
    : Math.max(0.06, Math.min(0.42, s.auto * (base / DEFAULT_SENS)));
  if (s && isFinite(s.cross)) t = Math.min(t, Math.max(0.08, s.cross * CROSS_WORD_FRAC));
  return t;
}

function matchAndFire(frames, now) {
  // Only compare against codes taught with the current algorithm family; the
  // three algorithms do not agree on scale, so cross-family matching is noise.
  const pool = matchPool();
  if (pool.length === 0) {
    bestWordEl.textContent = templates.length ? "no codes for this algorithm" : "no codes yet";
    return;
  }
  const live = resampleSeq(frames, FIXED_LEN);
  const eLive = seqEnergy(live);
  const devLive = devOfSeq(live);
  // Best (smallest) distance per distinct word.
  const perWord = new Map();
  for (const t of pool) {
    if (!energyCompatible(eLive, t)) continue;        // stillness can't match movement
    if (devLive < devOf(t) * DEV_RATIO_MIN) continue; // near-neutral wobble can't match a real excursion
    const k = t.word.toLowerCase();
    const ew = effWeights(t, null); // manual capture has no per-frame confidence
    const d = ew ? dtw(live, t.seq, ew.w, ew.wSum) : dtw(live, t.seq);
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
  if (best.dist < thresh * FIRE_MARGIN && !ambiguous) {
    const ok = best.word !== lastFiredWord || now - lastFireAt > COOLDOWN_MS;
    if (ok) fireWord(best.word, now);
  }
}

// Status text under the video, per tab. Only Perform "watches".
function setPerformState() {
  if (!ready) return;
  if (currentTab === "teach") {
    statusEl.textContent = "● ready. Type a word and click Record movement to teach a code.";
    return;
  }
  if (currentTab !== "perform") {
    statusEl.textContent = "● ready.";
    return;
  }
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

  // Teach on a beat: while recording, a steady metronome ticks an 8-count
  // (reusing the tutorial numerals and dot strip) so movements are taught ON
  // counts and play back the same way. It goes quiet during the stop
  // hand-over-face countdown: that is bookkeeping, not dancing.
  if (teach && teach.state === "capturing" && teach.startedAt && !teach.stopSince && !playback) {
    const beatMs = 60000 / teachBpm();
    const count = 1 + Math.floor((now - teach.startedAt) / beatMs) % 8;
    if (teach.lastBeat !== count) {
      teach.lastBeat = count;
      danceCount.hidden = false;
      danceCount.textContent = String(count);
      danceCount.classList.remove("lead", "tick");
      void danceCount.offsetWidth;
      danceCount.classList.add("tick");
      countDots.hidden = false;
      const kids = countDots.children;
      for (let k = 0; k < kids.length; k++) {
        kids[k].className = k < count ? (k === count - 1 ? "cur" : "on") : "";
      }
      countTick(count === 1 || count === 5);
    }
    teachCountsOn = true;
  } else if (teachCountsOn) {
    teachCountsOn = false;
    if (!playback) { danceCount.hidden = true; countDots.hidden = true; }
  }

  if (!ready || !teach) return; // Perform manages its own status line
  if (kind === "start") statusEl.textContent = `Keep your face covered… recording in ${secs}`;
  else if (kind === "stop") statusEl.textContent = `Hold… saving in ${secs}`;
}

function renderPhrase() {
  const chips = phrase.map((w) => `<span class="chip">${escapeHtml(w)}</span>`).join("");
  phraseEl.innerHTML = phrase.length === 0
    ? '<span class="muted">Your matched words appear here…</span>'
    : chips;
  // Kiosk mode mirrors the phrase over the video so the audience reads it.
  const kp = document.getElementById("kioskPhrase");
  if (kp) kp.innerHTML = chips;
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
    frames: [], near: [], onface: [], holdSince: 0, canStopL: false, stopSince: 0,
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
// A hand must STAY on the face this long before any countdown appears, so a
// hand merely passing over the face triggers nothing at all instead of
// flashing a distracting countdown that instantly cancels.
const ARM_DELAY_MS = 450;

// One frame of a teaching capture. RIGHT hand over the face held for the 3-2-1
// countdown STARTS recording; LEFT hand over the face held for the countdown
// STOPS and saves. Different hands for start and stop, so a stop can never
// re-arm a start and there is no ambiguity about which countdown is which.
// A hand can only stop the capture after IT has been off the face once, so the
// starting hand still resting there can't stop the capture instantly.
// Teach records EVERY frame while capturing. Fast movement blurs the hands
// and drops wrist confidence, so gating recording on hand visibility threw
// away the most energetic frames and made big movements fail the travel
// check. Recording is deliberate, previewed as a ghost right after saving,
// and the One-Euro filter tames wild wrist guesses, so unlike Perform
// matching, teaching does not require visible hands.
function teachStep(vec, hands, near, now, handVis) {
  const t = teach;
  if (t.manual) {
    t.frames.push(vec); t.near.push(near); t.onface.push(hands.left || hands.right);
    (t.handv || (t.handv = [])).push(!!handVis);
    return;
  }
  if (t.state === "prime") {          // waiting for a fresh right-hand cover
    if (!hands.right) { t.armed = true; t.rOnSince = 0; } // right hand must be off the face first...
    else if (t.armed) {
      // ...then cover and STAY. Only after ARM_DELAY_MS does the countdown
      // begin, so a hand sweeping past the face never flashes one.
      if (!t.rOnSince) t.rOnSince = now;
      if (now - t.rOnSince >= ARM_DELAY_MS) { t.state = "starting"; t.holdSince = now; t.lastOn = now; }
    }
    return;
  }
  if (t.state === "starting") {       // start countdown; leaving the face cancels it
    if (hands.right) t.lastOn = now;
    else if (now - t.lastOn > HOLD_GRACE_MS) { t.state = "prime"; return; }
    if (now - t.holdSince >= START_HOLD_MS) {
      t.state = "capturing";
      t.frames = [vec]; t.near = [near]; t.onface = [hands.left || hands.right];
      t.handv = [!!handVis];
      t.startedAt = now;
      t.canStopL = false; t.stopSince = 0; t.stopLastOn = 0;
    }
    return;
  }
  // capturing: ONLY the LEFT hand (the big L) stops and saves. While a
  // recording is running, the right hand is completely inert; it can neither
  // stop this capture nor arm another one, so a habitual right-hand cover
  // mid-recording does nothing. canStopL requires the left hand to have been
  // off the face once since the capture began, so a both-hands start cover
  // can't stop the capture instantly.
  t.frames.push(vec);
  t.near.push(near);
  t.onface.push(hands.left || hands.right);
  t.handv.push(!!handVis);
  if (!hands.left) { t.canStopL = true; t.lOnSince = 0; }
  else if (t.canStopL && !t.lOnSince) t.lOnSince = now;
  // The stop countdown also waits out ARM_DELAY_MS of sustained cover, so a
  // left hand sweeping past the face mid-movement cannot flash it.
  const stopHeld = t.canStopL && hands.left && t.lOnSince && now - t.lOnSince >= ARM_DELAY_MS;
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

  // The dance is only the span where at least one hand was trackable: shave
  // hand-less frames off BOTH ends first (mid-move dropouts are motion blur
  // and stay), so a code never opens or closes on guessed wrists.
  const totalFrames = t.frames.length;
  if (t.handv && t.handv.length === totalFrames) {
    let lo = 0, hi = totalFrames;
    while (lo < hi - 3 && !t.handv[lo]) lo++;
    while (hi > lo + 3 && !t.handv[hi - 1]) hi--;
    if (hi - lo >= 3 && (lo > 0 || hi < totalFrames)) {
      t.frames = t.frames.slice(lo, hi);
      t.near = t.near.slice(lo, hi);
      t.onface = t.onface.slice(lo, hi);
    }
  }
  // Trim the trigger holds off both ends so every code starts and ends where
  // the movement actually happened. The broad near-face radius catches the
  // approach and departure, but a dance may legitimately keep a hand near the
  // face the whole time, which would trim EVERYTHING; when that happens, retry
  // trimming only the strict hand-on-face frames before giving up.
  let core = trimNearFace(t.frames, t.near);
  if (core.length < 3 || travelOf(core) < MIN_TRAVEL) {
    core = trimNearFace(t.frames, t.onface || []);
  }
  core = trimStillEnds(core); // codes span the movement, not the holds around it
  core = trimLeadIn(core);    // and never open on a "getting ready" pause

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
  // Duration must describe the TRIMMED movement, not the whole recording:
  // otherwise playback stretches the move over dead time and opens with a
  // long freeze.
  const durMs = Math.max(300, Math.round((now - t.startedAt) * (core.length / Math.max(1, totalFrames))));
  const seq = resampleSeq(core, FIXED_LEN);
  const tmpl = { id: newId(), word: t.word, seq, durMs, family: currentFamily, algo: algoChoice, createdAt: Date.now(), sq: 1, bpm: teachBpm() };
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
  // Guided takes: words with 3 examples calibrate FAR better thresholds, so
  // actively steer toward three, with the button itself naming the next take.
  const n = templates.filter((x) => x.word.toLowerCase() === t.word.toLowerCase()).length;
  let msg;
  if (n < 3) {
    recordBtn.textContent = `Record take ${n + 1} of 3`;
    msg = `Take ${n} of 3 saved for “${t.word}”. Do the SAME movement again — three takes make it much more reliable. Just cover your face with your RIGHT hand to record the next take, no click needed.`;
  } else {
    msg = `“${t.word}” now has ${n} takes — nicely calibrated. Switch to Perform to try it, or type a new word.`;
  }
  // Follow-up takes arm hands-free: covering the face with the right hand
  // starts the next take of the SAME word without touching the mouse.
  rearmWord = n < 3 ? t.word.toLowerCase() : null;
  rearmSince = 0;
  if (timedOut) msg += " (Recording hit its time cap; covering your face to finish was never detected.)";
  setTeachMsg(msg, "ok");
  // Keep the word so the next Record adds another example; select it for easy edit.
  wordInput.select();
}

// Show a riso pose card of the movement just saved, in the Teach panel.
function showTeachPreview(seq) {
  if (!teachPreview) return;
  teachPreview.innerHTML = '<div class="teach-preview-label">Saved as a pose card:</div>';
  try { teachPreview.appendChild(buildPoseStrip(seq, { count: 3, size: 100 })); } catch {}
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
  playback = { items, label, key, idx: 0, tRel: 0, lastNow: null, lastCount: 0, stepDone: 0, mode: "play" };
  resetPbPrefs();
  updatePbControls();
  renderCodeList();
}
// Rehearse: the ghost demonstrates once with counts, then YOUR TURN — a
// count-in with the start pose held faintly, you perform from memory, and
// your attempt is scored against the word with the same matcher Perform
// uses. Cycles demo -> your turn -> result until stopped.
function startRehearse(word) {
  const all = templates.filter((t) => t.word.toLowerCase() === word.toLowerCase());
  if (!all.length) return;
  clearTimeout(ghostPreviewTimer);
  const fam = all.filter((t) => (t.family || "blaze") === currentFamily);
  const items = fam.length ? fam : all;
  playback = {
    items: [playbackReps(items)[0]],
    allItems: items,
    label: word,
    key: "rh:" + word.toLowerCase(),
    idx: 0, tRel: 0, lastNow: null, lastCount: 0, stepDone: 0,
    mode: "rehearse", phase: "demo",
  };
  resetPbPrefs();
  updatePbControls();
  renderCodeList();
}
// Collapse near-duplicate examples for word playback: takes whose DTW distance
// is small are averaged frame-wise into ONE ghost, so "Play all" shows the
// distinct ways a word can be danced instead of replaying near-identical takes.
const PLAY_SIM_THRESH = 0.12;
function playbackReps(items) {
  const groups = [];
  for (const t of items) {
    const g = groups.find((g) => dtw(g.base.seq, t.seq) < PLAY_SIM_THRESH);
    if (g) g.members.push(t); else groups.push({ base: t, members: [t] });
  }
  return groups.map((g) => {
    if (g.members.length === 1) return g.base;
    const seq = [];
    for (let f = 0; f < FIXED_LEN; f++) {
      const v = new Array(NUM_LMS * 2).fill(0);
      for (const m of g.members) for (let k = 0; k < v.length; k++) v[k] += m.seq[f][k];
      for (let k = 0; k < v.length; k++) v[k] /= g.members.length;
      seq.push(v);
    }
    const durMs = Math.round(g.members.reduce((s, m) => s + (m.durMs || 2000), 0) / g.members.length);
    return { ...g.base, seq, durMs };
  });
}
function startPlayback(word) {
  const items = templates.filter((t) => t.word.toLowerCase() === word.toLowerCase());
  startPlaybackItems(playbackReps(items), word, "word:" + word.toLowerCase());
}
function startPlaybackExample(id) {
  const t = templates.find((x) => x.id === id);
  if (t) startPlaybackItems([t], t.word, "ex:" + id);
}

// ---- Routine: an ordered choreography of saved codes ----
const ROUTINE_KEY = "queercoded.routine.v1";
let routine = [];
try { routine = JSON.parse(localStorage.getItem(ROUTINE_KEY)) || []; } catch { routine = []; }
function saveRoutine() {
  try { localStorage.setItem(ROUTINE_KEY, JSON.stringify(routine)); } catch {}
}
function renderRoutine() {
  if (!routineListEl) return;
  routineListEl.innerHTML = routine.length
    ? routine.map((w, i) =>
        `<span class="chip" draggable="true" data-idx="${i}" title="Drag to reorder">${i + 1}. ${escapeHtml(w)}<button class="chip-x" data-ri="${i}" aria-label="Remove step">&times;</button></span>`).join("")
    : '<span class="muted">No moves yet: add your codes below, in order.</span>';
  const words = [...new Set(templates.filter((t) => (t.family || "blaze") === currentFamily).map((t) => t.word))];
  routineAddSel.innerHTML = '<option value="">Add a code…</option>' +
    words.map((w) => `<option value="${encodeURIComponent(w)}">${escapeHtml(w)}</option>`).join("");
  routineStartBtn.disabled = routine.length === 0;
}
routineAddSel?.addEventListener("change", () => {
  if (!routineAddSel.value) return;
  routine.push(decodeURIComponent(routineAddSel.value));
  saveRoutine();
  renderRoutine();
});
routineListEl?.addEventListener("click", (e) => {
  const b = e.target.closest("[data-ri]");
  if (!b) return;
  routine.splice(+b.dataset.ri, 1);
  saveRoutine();
  renderRoutine();
});
// Drag a routine chip onto another to reorder the choreography.
let routineDragIdx = null;
routineListEl?.addEventListener("dragstart", (e) => {
  const chip = e.target.closest(".chip[data-idx]");
  if (!chip) return;
  routineDragIdx = +chip.dataset.idx;
  e.dataTransfer.effectAllowed = "move";
});
routineListEl?.addEventListener("dragover", (e) => {
  if (routineDragIdx != null) e.preventDefault();
});
routineListEl?.addEventListener("drop", (e) => {
  const chip = e.target.closest(".chip[data-idx]");
  if (routineDragIdx == null || !chip) { routineDragIdx = null; return; }
  e.preventDefault();
  const to = +chip.dataset.idx;
  const [moved] = routine.splice(routineDragIdx, 1);
  routine.splice(to, 0, moved);
  routineDragIdx = null;
  saveRoutine();
  renderRoutine();
});
routineListEl?.addEventListener("dragend", () => { routineDragIdx = null; });
routineClearBtn?.addEventListener("click", () => {
  routine = [];
  saveRoutine();
  renderRoutine();
});
routineStartBtn?.addEventListener("click", () => {
  stopPlayback();
  startRoutine();
});
function startRoutine() {
  const steps = [];
  for (const w of routine) {
    const items = templates.filter((t) =>
      t.word.toLowerCase() === w.toLowerCase() && (t.family || "blaze") === currentFamily);
    if (items.length) steps.push({ word: w, items, rep: playbackReps(items)[0] });
  }
  if (!steps.length) return;
  clearTimeout(ghostPreviewTimer);
  playback = {
    mode: "routine", steps, stepIdx: 0, phase: "cue", scores: [],
    items: [steps[0].rep], allItems: steps[0].items,
    label: steps[0].word, key: "routine",
    idx: 0, tRel: 0, lastNow: null, lastCount: 0, stepDone: 0,
  };
  resetPbPrefs();
  updatePbControls();
  renderCodeList();
}
// Warm-up: play every word once with counts before matching starts.
function startWarmup() {
  const words = [...new Set(templates.filter((t) => (t.family || "blaze") === currentFamily).map((t) => t.word))];
  const items = [];
  for (const w of words) {
    const its = templates.filter((t) =>
      t.word.toLowerCase() === w.toLowerCase() && (t.family || "blaze") === currentFamily);
    const rep = playbackReps(its)[0];
    if (rep) items.push({ ...rep, _label: w });
  }
  if (items.length) startPlaybackItems(items, "warm-up", "warmup");
}
document.getElementById("warmupYes")?.addEventListener("click", () => {
  warmupOffer.hidden = true;
  startWarmup();
});
document.getElementById("warmupNo")?.addEventListener("click", () => { warmupOffer.hidden = true; });

// The practice strip (speed / Loop / Steps) belongs to the Perform screen.
function updatePbControls() {
  pbControls.hidden = !(playback && currentTab === "perform");
}
// Practice preferences are PER PLAYBACK: a Loop or half-speed left on from
// one practice must not silently apply to the next thing played.
function resetPbPrefs() {
  pbSpeed = 1;
  pbLoop = false;
  pbSteps = false;
  for (const x of pbControls.querySelectorAll("[data-speed]")) {
    x.classList.toggle("on", x.dataset.speed === "1");
  }
  document.getElementById("pbLoopBtn").classList.remove("on");
  document.getElementById("pbStepsBtn").classList.remove("on");
}
// Mobile bottom sheet: hide/show the panel so the stream stays unobstructed.
const sheetToggle = document.getElementById("sheetToggle");
sheetToggle?.addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("sheet-collapsed");
  sheetToggle.textContent = collapsed ? "Show panel" : "Hide panel";
});
function stopPlayback() {
  playback = null;
  danceCount.hidden = true;
  countDots.hidden = true;
  pbControls.hidden = true;
  if (ready) setPerformState();
  renderCodeList();
}
// Practice controls: tempo chips, loop, step-through.
pbControls.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.speed) {
    pbSpeed = parseFloat(b.dataset.speed);
    for (const x of pbControls.querySelectorAll("[data-speed]")) x.classList.toggle("on", x === b);
  } else if (b.id === "pbLoopBtn") {
    pbLoop = !pbLoop;
    b.classList.toggle("on", pbLoop);
  } else if (b.id === "pbStepsBtn") {
    pbSteps = !pbSteps;
    b.classList.toggle("on", pbSteps);
    // Off: release any freeze and stop re-freezing. On: arm from here.
    if (playback) { playback.waiting = false; playback.stepDone = pbSteps ? 0 : 8; }
  }
});
// Space skips a frozen step (unless typing, and without stealing the manual
// hold-to-capture Spacebar, which only listens in manual trigger mode).
window.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || !playback?.waiting || e.repeat) return;
  const el = document.activeElement;
  if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) return;
  e.preventDefault();
  playback.skipStep = true;
});

// ---- Dance-tutorial playback ----
// Every example plays like a dance class: a "5 6 7 8" count-in while the
// ghost holds its start pose, then the movement on counts 1-8 (big riso
// numeral + the dot strip), faint onion-skin echoes behind the figure so
// the direction of travel reads, and pulsing rings marking the joints that
// actually carry this move. A soft tick sounds on every count (respects the
// Ping toggle).
const FOCUS_CANDIDATES = [0, 13, 14, 15, 16, 25, 26, 27, 28];
const FOCUS_NAMES = { 0: "head", 13: "elbows", 14: "elbows", 15: "hands", 16: "hands", 25: "knees", 26: "knees", 27: "feet", 28: "feet" };
// Which joints move most across the stored frames: those are the move.
function focusJoints(seq) {
  const travel = new Map();
  let max = 0;
  for (const i of FOCUS_CANDIDATES) {
    let t = 0;
    for (let f = 1; f < seq.length; f++) {
      t += Math.hypot(seq[f][i * 2] - seq[f - 1][i * 2], seq[f][i * 2 + 1] - seq[f - 1][i * 2 + 1]);
    }
    travel.set(i, t);
    if (t > max) max = t;
  }
  if (!(max > 0)) return { joints: [], label: "" };
  const joints = FOCUS_CANDIDATES
    .filter((i) => travel.get(i) >= max * 0.55)
    .sort((a, b) => travel.get(b) - travel.get(a))
    .slice(0, 3);
  const label = [...new Set(joints.map((i) => FOCUS_NAMES[i]))].join(" and ");
  return { joints, label };
}
// A short metronome tick; the downbeats (1 and 5) land higher and louder.
function countTick(strong) {
  if (!soundOn) return;
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
    const t = audioCtx.currentTime;
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.type = "sine";
    o.frequency.setValueAtTime(strong ? 1320 : 880, t);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(strong ? 0.1 : 0.06, t + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.06);
    o.connect(g);
    g.connect(audioCtx.destination);
    o.start(t);
    o.stop(t + 0.08);
  } catch { /* audio is decoration; never let it break playback */ }
}
// The stored seq at progress p (0..1), tweened, in image-normalized coords.
function ghostFrameAt(cur, p, ax, ay, asc) {
  const f = Math.max(0, Math.min(1, p)) * (FIXED_LEN - 1);
  const i = Math.floor(f);
  const frac = f - i;
  const a = cur.seq[i], b = cur.seq[Math.min(i + 1, FIXED_LEN - 1)];
  const ghost = [];
  const A = imgAspect(); // stored x is in square (height) units; images are not
  for (let k = 0; k < NUM_LMS; k++) {
    const vx = a[k * 2] * (1 - frac) + b[k * 2] * frac;
    const vy = a[k * 2 + 1] * (1 - frac) + b[k * 2 + 1] * frac;
    // Slots a 17-point code never filled stay at the origin; mark them
    // invisible so the painters skip them instead of drawing to (0,0).
    const filled = a[k * 2] !== 0 || a[k * 2 + 1] !== 0 || b[k * 2] !== 0 || b[k * 2 + 1] !== 0;
    ghost.push({ x: ax + (vx * asc) / A, y: ay + vy * asc, visibility: filled ? 1 : 0 });
  }
  return ghost;
}
// Ghost coords are normalized (0..1); the painters want pixels.
const ghostPx = (ghost) => (i) => {
  const p = ghost[i];
  return p && p.visibility > 0
    ? { x: p.x * overlay.width, y: p.y * overlay.height, v: true }
    : { x: 0, y: 0, v: false };
};
function ghostTorso(GP) {
  const ls = GP(11), rs = GP(12), lh = GP(23), rh = GP(24);
  if (!ls.v || !rs.v || (!lh.v && !rh.v)) return 0;
  const hipX = lh.v && rh.v ? (lh.x + rh.x) / 2 : (lh.v ? lh.x : rh.x);
  const hipY = lh.v && rh.v ? (lh.y + rh.y) / 2 : (lh.v ? lh.y : rh.y);
  return Math.hypot((ls.x + rs.x) / 2 - hipX, (ls.y + rs.y) / 2 - hipY);
}
// Live-follow feedback: for each focus joint, where the performer's SAME
// joint currently is relative to the ghost's. `on` within about half a
// torso of the target.
function pbTargets(GP, gsc, joints, now) {
  const res = new Map();
  const lv = liveLmsNow && now - liveLmsNow.at < 300 ? liveLmsNow.lms : null;
  if (!lv) return res;
  for (const j of joints) {
    const g = GP(j);
    const p = lv[j];
    if (!g.v || !p || (p.visibility ?? 0) < 0.3) continue;
    const x = p.x * overlay.width, y = p.y * overlay.height;
    const d = Math.hypot(x - g.x, y - g.y);
    res.set(j, { x, y, d, on: d < gsc * 0.55 });
  }
  return res;
}
// A red arrow from the performer's joint toward the ghost's ring: which way
// to move, without words.
function drawGuideArrow(from, to, gsc) {
  const dx = to.x - from.x, dy = to.y - from.y;
  const len = Math.hypot(dx, dy);
  if (len < gsc * 0.3) return;
  const ux = dx / len, uy = dy / len;
  const ex = to.x - ux * gsc * 0.28, ey = to.y - uy * gsc * 0.28;
  const ah = Math.min(14, Math.max(7, gsc * 0.12));
  octx.save();
  octx.strokeStyle = "rgba(255,0,42,0.85)";
  octx.fillStyle = "rgba(255,0,42,0.85)";
  octx.lineWidth = 3;
  octx.beginPath();
  octx.moveTo(from.x, from.y);
  octx.lineTo(ex, ey);
  octx.stroke();
  octx.beginPath();
  octx.moveTo(ex, ey);
  octx.lineTo(ex - ux * ah - uy * ah * 0.5, ey - uy * ah + ux * ah * 0.5);
  octx.lineTo(ex - ux * ah + uy * ah * 0.5, ey - uy * ah - ux * ah * 0.5);
  octx.closePath();
  octx.fill();
  octx.restore();
}
// Anticipation cue: a dashed arc tracing where each focus joint travels over
// the NEXT count, with an arrowhead, so the learner can lead instead of lag.
function drawPathHints(cur, p, ax, ay, asc, joints) {
  octx.save();
  octx.setLineDash([6, 6]);
  octx.strokeStyle = "rgba(255,255,255,0.55)";
  octx.fillStyle = "rgba(255,255,255,0.55)";
  octx.lineWidth = 2;
  for (const j of joints) {
    const pts = [];
    for (let s = 0; s <= 5; s++) {
      const g = ghostFrameAt(cur, Math.min(1, p + (s / 5) * 0.125), ax, ay, asc)[j];
      if (!g || g.visibility <= 0) { pts.length = 0; break; }
      pts.push({ x: g.x * overlay.width, y: g.y * overlay.height });
    }
    if (pts.length < 2) continue;
    const a = pts[pts.length - 2], b = pts[pts.length - 1];
    if (Math.hypot(b.x - pts[0].x, b.y - pts[0].y) < 12) continue; // barely moves: no cue
    octx.beginPath();
    octx.moveTo(pts[0].x, pts[0].y);
    for (const q of pts.slice(1)) octx.lineTo(q.x, q.y);
    octx.stroke();
    const dx = b.x - a.x, dy = b.y - a.y;
    const l = Math.hypot(dx, dy) || 1;
    const ux = dx / l, uy = dy / l;
    octx.save();
    octx.setLineDash([]);
    octx.beginPath();
    octx.moveTo(b.x, b.y);
    octx.lineTo(b.x - ux * 10 - uy * 5, b.y - uy * 10 + ux * 5);
    octx.lineTo(b.x - ux * 10 + uy * 5, b.y - uy * 10 - ux * 5);
    octx.closePath();
    octx.fill();
    octx.restore();
  }
  octx.restore();
}
// Score a rehearse attempt against the word's examples, same matcher as
// Perform (DTW + the word's own threshold).
function scoreRehearse(pb) {
  const frames = trimStillEnds(pb.capture || []);
  if (!frames || frames.length < MIN_SEG_FRAMES) {
    return { ok: false, big: "TOO LITTLE", msg: "Too little movement seen. Watch once more, then go bigger. Again…" };
  }
  const live = resampleSeq(frames, FIXED_LEN);
  let best = Infinity;
  for (const t of pb.allItems) best = Math.min(best, dtw(live, t.seq));
  const thr = thresholdFor(pb.label);
  const pct = Math.max(0, Math.min(100, Math.round((1 - best / (thr * 1.8)) * 100)));
  const ok = best < thr;
  const hint = pb.focus?.label ? ` Watch the ${pb.focus.label}.` : "";
  return ok
    ? { ok, pct, big: `${pct}%`, msg: `Nailed it: ${pct}% match. Again…` }
    : { ok, pct, big: `${pct}%`, msg: `${pct}%, not quite.${hint} Again…` };
}
function drawPlayback(now) {
  const pb = playback;
  // Playback runs on its OWN clock: the speed chips stretch it, Steps mode
  // pauses it, and wall time only feeds it.
  const dt = pb.lastNow == null ? 0 : Math.min(100, now - pb.lastNow);
  pb.lastNow = now;
  if (!pb.waiting) pb.tRel += dt * pbSpeed;

  const routineMode = pb.mode === "routine";
  const step = routineMode ? pb.steps[Math.min(pb.stepIdx, pb.steps.length - 1)] : null;
  const cur = routineMode ? step.rep : pb.items[Math.min(pb.idx, pb.items.length - 1)];
  const dur = Math.min(5000, Math.max(800, cur.durMs || 2000));
  const beat = dur / 8;
  const lead = beat * 4; // the "5 6 7 8" count-in
  if (pb.focusFor !== cur) {
    pb.focusFor = cur;
    pb.focus = focusJoints(cur.seq);
  }
  const rehearse = pb.mode === "rehearse" || routineMode;
  const ph = rehearse ? pb.phase : "demo";
  const nextPhase = (phase) => {
    pb.phase = phase;
    pb.tRel = 0;
    pb.lastCount = 0;
    pb.stepDone = 0;
    pb.cueShown = false;
  };

  // Routine: each step opens with a cue card naming the next move.
  if (ph === "cue") {
    if (!pb.cueShown) {
      pb.cueShown = true;
      pb.label = step.word;
      flashBigStatus(`NEXT: ${step.word}`, "", 1400);
      speak(step.word); // voice cue, so eyes can stay off the screen
    }
    if (pb.tRel >= 1600) { nextPhase("turnLead"); return; }
  }

  // ---- phase / item transitions ----
  if (ph === "demo" && pb.tRel >= lead + dur) {
    if (rehearse) { nextPhase("turnLead"); return; }
    pb.idx++;
    pb.tRel = 0;
    pb.lastCount = 0;
    pb.stepDone = 0;
    if (pb.idx >= pb.items.length) {
      if (pbLoop) { pb.idx = 0; return; }
      stopPlayback();
      return;
    }
    return;
  }
  if (ph === "turnLead" && pb.tRel >= lead) {
    if (routineMode) pb.allItems = step.items; // score against this step's word
    nextPhase("turn");
    pb.capture = []; // runFrame fills this with normalized poses
    return;
  }
  if (ph === "turn" && pb.tRel >= dur + beat) {
    pb.result = scoreRehearse(pb);
    pb.capture = null;
    if (routineMode) pb.scores.push(pb.result);
    if (pb.result.ok) { flashRiso(now); ping(); }
    flashBigStatus(pb.result.big, pb.result.ok ? "" : "stop", 2000);
    nextPhase("result");
    return;
  }
  if (ph === "result" && pb.tRel >= 2300) {
    if (pb.mode === "routine") {
      pb.stepIdx++;
      if (pb.stepIdx >= pb.steps.length) {
        const hits = pb.scores.filter((s) => s.ok).length;
        const avg = Math.round(pb.scores.reduce((s, r) => s + r.pct, 0) / pb.scores.length);
        pb.summary = `Routine done: ${hits}/${pb.scores.length} hits, average ${avg}%.`;
        flashBigStatus(`${hits}/${pb.scores.length} · ${avg}%`, hits === pb.scores.length ? "" : "stop", 2800);
        if (hits === pb.scores.length) flashRiso(now);
        nextPhase("done");
        return;
      }
      nextPhase("cue");
      return;
    }
    nextPhase("demo"); // rehearse loops until stopped
    return;
  }
  if (ph === "done" && pb.tRel >= 3000) { stopPlayback(); return; }

  // ---- timeline for this frame ----
  let leading = false, p = 0, count = null, showGhost = true, ghostAlpha = 1;
  if (ph === "demo") {
    leading = pb.tRel < lead;
    p = leading ? 0 : (pb.tRel - lead) / dur;
    count = leading
      ? 5 + Math.min(3, Math.floor(pb.tRel / beat))
      : 1 + Math.min(7, Math.floor(p * 8));
  } else if (ph === "cue") {
    leading = true;
    ghostAlpha = 0.3; // the next move's start pose, faintly, under the cue card
  } else if (ph === "turnLead") {
    leading = true;
    ghostAlpha = 0.3; // the start pose lingers faintly while you get set
    count = 5 + Math.min(3, Math.floor(pb.tRel / beat));
  } else if (ph === "turn") {
    p = Math.min(1, pb.tRel / dur);
    count = 1 + Math.min(7, Math.floor(Math.min(0.999, pb.tRel / dur) * 8));
    showGhost = false; // from memory: only the counts carry you
  } else {
    showGhost = false; // result: the verdict is on screen
  }

  // Steps mode (plain playback only): freeze at the top of every count until
  // the performer hits the pose, or Space skips.
  if (!rehearse && pbSteps && !leading && count > pb.stepDone) {
    pb.waiting = true;
    pb.tRel = lead + (count - 1) * beat;
    p = (pb.tRel - lead) / dur;
  }

  // Anchor the ghost on the live body when one is visible (so it dances on
  // the performer at their position and size), else a fixed spot.
  const live = liveFrame && (now - liveFrame.at < 300);
  const ax = live ? liveFrame.cx : GHOST_CX;
  const ay = live ? liveFrame.cy : GHOST_CY;
  const asc = live ? liveFrame.torso : GHOST_SCALE;

  if (showGhost) {
    // Onion-skin echoes (screened flat ink) under the moving figure.
    if (!leading && !pb.waiting) {
      for (const [back, alpha] of [[0.12, 0.1], [0.06, 0.2]]) {
        if (p - back <= 0) continue;
        const GPe = ghostPx(ghostFrameAt(cur, p - back, ax, ay, asc));
        const sce = ghostTorso(GPe);
        if (sce > 8) {
          const ge = inkLayer();
          ge.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
          paintBody(ge, GPe, sce, "#E8452C");
          screenInk(ge, 0.4, 0.55);
          octx.save();
          octx.globalAlpha = alpha;
          octx.drawImage(inkCanvas, 0, 0);
          octx.restore();
        }
      }
    }
    const ghost = ghostFrameAt(cur, p, ax, ay, asc);
    const GP = ghostPx(ghost);
    const gsc = ghostTorso(GP);
    if (gsc > 8) {
      if (ghostAlpha < 1) {
        octx.save();
        octx.globalAlpha = ghostAlpha;
        paintBody(octx, GP, gsc, "#E8452C");
        octx.restore();
      } else {
        paintRisoGhost(GP, gsc);
      }
      // Where each focus joint goes NEXT (dashed), then the rings with live
      // feedback: filled solid when your joint is in place, hollow with a
      // red guide arrow when it is not.
      if (!leading && !pb.waiting && p < 0.999) drawPathHints(cur, p, ax, ay, asc, pb.focus.joints);
      const targets = pbTargets(GP, gsc, pb.focus.joints, now);
      const pulse = 1 + 0.15 * Math.sin(now / 130);
      let ringCount = 0, onCount = 0;
      for (const j of pb.focus.joints) {
        const q = GP(j);
        if (!q.v) continue;
        ringCount++;
        const t = targets.get(j);
        if (t?.on) onCount++;
        const r = Math.max(10, gsc * 0.22) * (t?.on ? 1 : pulse);
        octx.beginPath();
        octx.arc(q.x, q.y, r, 0, Math.PI * 2);
        if (t?.on) {
          octx.fillStyle = "rgba(236,255,0,0.3)";
          octx.fill();
          octx.lineWidth = 5;
          octx.strokeStyle = "#ECFF00";
          octx.stroke();
        } else {
          octx.lineWidth = 3;
          octx.strokeStyle = "rgba(236,255,0,0.9)";
          octx.stroke();
          octx.beginPath();
          octx.arc(q.x, q.y, r * 1.35, 0, Math.PI * 2);
          octx.lineWidth = 2;
          octx.strokeStyle = "rgba(236,255,0,0.35)";
          octx.stroke();
          if (t) drawGuideArrow(t, q, gsc);
        }
      }
      // Steps: release the freeze when every visible focus joint is in place
      // (and at least one live joint was actually seen), or on Space.
      if (pb.waiting && (pb.skipStep || (ringCount > 0 && targets.size > 0 && onCount === ringCount))) {
        pb.stepDone = count;
        pb.waiting = false;
        pb.skipStep = false;
        countTick(false);
      }
    } else {
      const conns = connectionsForFamily(cur.family || "blaze");
      drawSkeleton(ghost, conns, "rgba(0,0,0,0.55)", "rgba(0,0,0,0.55)", 11, 7);
      drawSkeleton(ghost, conns, "rgba(255,255,255,0.95)", "#ffffff", 5, 4);
      if (pb.waiting && pb.skipStep) { pb.stepDone = count; pb.waiting = false; pb.skipStep = false; }
    }
  }

  // Counts: numeral + dot strip, updated only when the count changes.
  if (count == null) {
    danceCount.hidden = true;
    countDots.hidden = true;
  } else if (pb.lastCount !== count || pb.lastLead !== leading) {
    pb.lastCount = count;
    pb.lastLead = leading;
    danceCount.hidden = false;
    danceCount.textContent = String(count);
    danceCount.classList.toggle("lead", leading);
    danceCount.classList.remove("tick");
    void danceCount.offsetWidth; // restart the pop animation
    danceCount.classList.add("tick");
    countDots.hidden = false;
    const kids = countDots.children;
    for (let k = 0; k < kids.length; k++) {
      kids[k].className = !leading && k < count ? (k === count - 1 ? "cur" : "on") : "";
    }
    countTick(!leading && (count === 1 || count === 5));
  }

  // Status line per phase.
  if (ph === "cue") {
    statusEl.textContent = `Routine ${pb.stepIdx + 1}/${pb.steps.length}: next is “${step.word}”…`;
  } else if (ph === "turnLead") {
    statusEl.textContent = `Your turn: perform “${pb.label}” from memory after the count-in…`;
  } else if (ph === "turn") {
    statusEl.textContent = `Go! Perform “${pb.label}”.`;
  } else if (ph === "result") {
    statusEl.textContent = pb.result.msg;
  } else if (ph === "done") {
    statusEl.textContent = pb.summary;
  } else {
    statusEl.textContent =
      (rehearse ? `Rehearsing “${pb.label}”: watch first` : `Playing “${cur._label || pb.label}”`) +
      (!rehearse && pb.items.length > 1 ? ` (${pb.idx + 1}/${pb.items.length})` : "") +
      (pb.focus.label ? ` · watch the ${pb.focus.label}` : "") +
      (pb.waiting ? " · hit the pose to continue (or press Space)" : "");
  }
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
    const rehearsing = playback && playback.key === "rh:" + g.word.toLowerCase();
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
          <button class="btn small ${rehearsing ? "playing" : ""}" data-act="rehearse" data-word="${w}" title="Watch it, then perform it from memory and get scored">${rehearsing ? "Stop" : "Rehearse"}</button>
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
  renderRoutine(); // keep the routine builder's word list in sync
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
  } else if (act === "rehearse") {
    if (playback && playback.key === "rh:" + word.toLowerCase()) stopPlayback();
    else startRehearse(word);
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
function exportCodes(filename = "queercoded-codes.json") {
  const blob = new Blob([JSON.stringify(templates, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}
exportBtn.addEventListener("click", () => exportCodes());
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
      templates.push(migrateSq({ ...t, id: newId(), createdAt: t.createdAt || Date.now() }));
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
  if (!confirm("Delete all saved codes? A backup file downloads first, so you can Import it if you change your mind.")) return;
  // Safety net: the wipe always leaves a file behind.
  if (templates.length) exportCodes("queercoded-codes-backup.json");
  templates = [];
  saveTemplates();
  renderCodeList();
});

// ---------- AlgoDance zine viewer ----------
// The PDF is rendered page by page with pdf.js (loaded on demand from a CDN
// the first time the tab opens), so reading it feels like flipping a zine:
// one lit page against a dark room, turned by click, chevron, or arrow key.
// If pdf.js can't load, the plain embedded-PDF iframe takes over. Nothing
// downloads until the tab is opened (the PDF is ~24 MB).
const PDFJS_BASE = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/";
const zineEl = document.getElementById("zine");
const zineCanvas = document.getElementById("zineCanvas");
const zinePrevBtn = document.getElementById("zinePrev");
const zineNextBtn = document.getElementById("zineNext");
const zineNum = document.getElementById("zineNum");
const pdfFallback = document.getElementById("pdfFallback");
const headerToggle = document.getElementById("headerToggle");
let zine = null; // {doc, page, seq, task} once opened; {failed:true} on fallback

async function openZine() {
  if (zine) return;
  zine = { doc: null, page: 1, seq: 0, task: null };
  const loading = document.createElement("div");
  loading.className = "zine-load";
  loading.textContent = "Loading the zine…";
  zineEl.appendChild(loading);
  try {
    const pdfjs = await import(PDFJS_BASE + "pdf.min.mjs");
    pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_BASE + "pdf.worker.min.mjs";
    zine.doc = await pdfjs.getDocument("AlgoDance.pdf").promise;
    loading.remove();
    renderZinePage(1, 0);
  } catch (e) {
    console.warn("zine viewer unavailable, embedding the PDF instead:", e);
    loading.remove();
    zine = { failed: true };
    zineEl.hidden = true;
    pdfFallback.src = "AlgoDance.pdf#toolbar=0&navpanes=0&scrollbar=1&view=FitH";
    pdfFallback.hidden = false;
  }
}

// Per-page rotation. Some pages hold sideways text; where the PDF has a text
// layer, the dominant baseline direction detects and fixes that automatically.
// Old manual per-page rotations are dropped: the defaults below are correct.
try { localStorage.removeItem("queercoded.zineRot.v1"); } catch {}
// Pages known to be sideways in AlgoDance.pdf, verified against raw renders:
// 4-6 hold sideways content and need a quarter-turn right; 7 is ALREADY
// upright and needs none. Absolute viewport rotation, ignoring metadata.
const ZINE_ABS_ROT = { 4: 90, 5: 90, 6: 90, 7: 0 };
const zineAutoRot = new Map();
async function zineExtraRot(page, n) {
  let auto = zineAutoRot.get(n);
  if (auto == null) {
    auto = 0;
    try {
      const tc = await page.getTextContent();
      let horiz = 0, up = 0, down = 0;
      for (const it of tc.items) {
        const [a, b] = it.transform; // text baseline direction in PDF space
        if (Math.abs(a) >= Math.abs(b)) horiz++;
        else if (b > 0) up++;
        else down++;
      }
      const total = horiz + up + down;
      if (total >= 5 && up > total * 0.6) auto = 90;         // reads bottom-to-top
      else if (total >= 5 && down > total * 0.6) auto = 270; // reads top-to-bottom
    } catch {}
    zineAutoRot.set(n, auto);
  }
  return auto;
}

// Pages render into OFFSCREEN canvases, cached as promises (so concurrent
// requests share one render) and swapped in only when ready: the previous
// page stays visible instead of flashing white, and prerendered neighbours
// make most flips instant.
const zineCache = new Map(); // "page@rot@height" -> Promise<canvas>
async function paintZinePage(n, boxH) {
  if (!zine?.doc || n < 1 || n > zine.doc.numPages) return null;
  const page = await zine.doc.getPage(n);
  const rot = ZINE_ABS_ROT[n] != null
    ? ZINE_ABS_ROT[n]
    : (page.rotate + await zineExtraRot(page, n) + 360) % 360;
  const key = `${n}@${rot}@${boxH}`;
  if (!zineCache.has(key)) {
    zineCache.set(key, (async () => {
      const vp1 = page.getViewport({ scale: 1, rotation: rot });
      const scale = (boxH / vp1.height) * Math.min(2, window.devicePixelRatio || 1);
      const vp = page.getViewport({ scale, rotation: rot });
      const off = document.createElement("canvas");
      off.width = vp.width;
      off.height = vp.height;
      await page.render({ canvasContext: off.getContext("2d"), viewport: vp }).promise;
      return off;
    })());
    while (zineCache.size > 6) zineCache.delete(zineCache.keys().next().value);
  }
  return zineCache.get(key);
}

async function renderZinePage(n, dir) {
  if (!zine?.doc) return;
  n = Math.max(1, Math.min(zine.doc.numPages, n));
  zine.page = n;
  const my = ++zine.seq; // newer requests win; stale renders are dropped
  const boxH = zineEl.clientHeight || 640;
  let canvas = null;
  try { canvas = await paintZinePage(n, boxH); } catch (e) { console.warn("zine render:", e); }
  if (my !== zine.seq || !canvas) return;
  zineCanvas.width = canvas.width;
  zineCanvas.height = canvas.height;
  zineCanvas.getContext("2d").drawImage(canvas, 0, 0);
  zineNum.textContent = `${n} / ${zine.doc.numPages}`;
  if (dir) {
    zineCanvas.classList.remove("flip-left", "flip-right");
    void zineCanvas.offsetWidth; // restart the animation
    zineCanvas.classList.add(dir > 0 ? "flip-right" : "flip-left");
  }
  // Warm the neighbours so the next flip swaps in without a wait.
  paintZinePage(n + 1, boxH)?.catch?.(() => {});
  paintZinePage(n - 1, boxH)?.catch?.(() => {});
}

function zineFlip(dir) {
  if (!zine?.doc) return;
  renderZinePage(zine.page + dir, dir);
}

zinePrevBtn.addEventListener("click", () => zineFlip(-1));
zineNextBtn.addEventListener("click", () => zineFlip(1));
// Click the page itself: right half flips forward, left half back.
zineCanvas.addEventListener("click", (e) => {
  const r = zineCanvas.getBoundingClientRect();
  zineFlip(e.clientX > r.left + r.width / 2 ? 1 : -1);
});
window.addEventListener("keydown", (e) => {
  if (document.getElementById("pane-algodance").hidden || !zine?.doc) return;
  if (e.key === "ArrowRight" || e.key === "PageDown") { e.preventDefault(); zineFlip(1); }
  else if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); zineFlip(-1); }
});
// Re-fit the page when the window changes size (debounced).
let zineResizeTimer = null;
window.addEventListener("resize", () => {
  if (document.getElementById("pane-algodance").hidden || !zine?.doc) return;
  clearTimeout(zineResizeTimer);
  zineResizeTimer = setTimeout(() => renderZinePage(zine.page, 0), 200);
});

// ---------- Kiosk mode ----------
// Fullscreen performance view for installations: just the camera and the
// matched words, no panel, no header. Opt-in from the Perform tab; Esc or
// the floating Exit button leaves it. Never the default.
const kioskBtn = document.getElementById("kioskBtn");
const kioskExit = document.getElementById("kioskExit");
function setKiosk(on) {
  document.body.classList.toggle("kiosk", on);
  kioskExit.hidden = !on;
  if (on) {
    activateTab(document.getElementById("tab-perform")); // kiosk IS performing
    document.documentElement.requestFullscreen?.().catch(() => {});
  } else if (document.fullscreenElement) {
    document.exitFullscreen?.().catch(() => {});
  }
}
kioskBtn.addEventListener("click", () => setKiosk(true));
bgBtn.addEventListener("click", startBgCapture);
kioskExit.addEventListener("click", () => setKiosk(false));
document.addEventListener("fullscreenchange", () => {
  // Esc leaves browser fullscreen; drop the kiosk chrome with it.
  if (!document.fullscreenElement && document.body.classList.contains("kiosk")) {
    document.body.classList.remove("kiosk");
    kioskExit.hidden = true;
  }
});

// Header collapse while reading: the zine wants the whole screen. The
// floating Menu button toggles the header back for switching tabs.
headerToggle.addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("header-collapsed");
  headerToggle.textContent = collapsed ? "Menu" : "Hide menu";
});

// ---------- UI wiring ----------
const tabs = [...document.querySelectorAll(".tab")];
function activateTab(tab, focus = false) {
  const name = tab.dataset.tab;
  currentTab = name;
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
  // AlgoDance is nothing but the zine, filling the viewport edge to edge:
  // the header collapses (Menu floats to bring it back) and the viewer loads
  // on first open.
  const isZine = name === "algodance";
  layout.classList.toggle("pdf-full", isZine);
  document.body.classList.toggle("header-collapsed", isZine);
  headerToggle.hidden = !isZine;
  headerToggle.textContent = "Menu";
  if (isZine) openZine();
  // Leaving the Teach tab mid-recording abandons it, so an active teach can
  // never keep "recording" (REC) into Perform. Also drop any manual capture.
  if (teach && name !== "teach") cancelTeach();
  if (bgCapture && name !== "perform") { bgCapture = null; bgBtn.disabled = false; }
  manualCapturing = false;
  moving = false; // drop any half-finished Perform move
  hideClosest();
  barEl.style.width = "0%";
  updatePbControls(); // the practice strip only shows on Perform
  // First Perform visit each session: offer a warm-up run of your codes.
  if (name === "perform" && !warmupOffered) {
    warmupOffered = true;
    if (templates.some((t) => (t.family || "blaze") === currentFamily)) warmupOffer.hidden = false;
  }
  if (name !== "perform") warmupOffer.hidden = true;
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

// Horizontal vs vertical presentation. Vertical rotates the ENTIRE page 90
// degrees clockwise (pure CSS on <body>) for a monitor physically turned on
// its side. Always starts horizontal; the switch lives in the header so it
// stays reachable from either mode.
const orientSel = document.getElementById("orientSel");
localStorage.removeItem("queercoded.orientation.v1"); // orientation no longer persists
function applyOrientation(mode) {
  // Vertical is a desktop/installation layout; on small screens it fights
  // the fullscreen mobile layout, so it is forced off there.
  if (window.innerWidth <= 900) mode = "landscape";
  document.body.classList.toggle("rot90", mode === "rotated");
}
orientSel.addEventListener("change", () => applyOrientation(orientSel.value));
window.addEventListener("resize", () => applyOrientation(orientSel.value));

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
  // Build tag, so "which version am I actually running?" has an answer.
  console.log("Queercoded build v50 (2026-07-13)");
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
    const cam = initCamera().catch((e) => { e.isCamera = true; throw e; });
    const pose = initPose().catch((e) => { throw new Error("pose engine: " + e.message); });
    await Promise.all([cam, pose]);
    ready = true;
    clearTimeout(slow);
    statusEl.textContent = "Ready.";
    setPerformState();
  } catch (err) {
    clearTimeout(slow);
    console.error(err);
    // Camera failures get plain instructions, not developer-speak.
    if (err.isCamera && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
      statusEl.textContent = "Camera access was denied. Click the camera icon in your browser's address bar, allow the camera, then reload this page.";
    } else if (err.isCamera && (err.name === "NotFoundError" || err.name === "OverconstrainedError")) {
      statusEl.textContent = "No usable camera was found. Connect or enable one, then reload this page.";
    } else if (err.isCamera) {
      statusEl.textContent = "The camera could not start (" + err.message + "). Note it only works over https or localhost.";
    } else {
      statusEl.textContent = "Error loading " + err.message + ".";
    }
  }
})();
