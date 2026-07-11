// Pose-estimation backends. Each backend exposes the SAME interface so the app
// never cares which algorithm is running:
//
//   backend.family                 "blaze" | "movenet" | "yolo"
//   backend.connections            [[i,j], ...] slot pairs to draw as bones
//   await backend.load()           fetch weights / build the session
//   await backend.detect(video,ts) -> array of poses; each pose is a 33-slot
//                                     array of {x, y, visibility} in image-
//                                     normalized coords (0..1, origin top-left)
//   backend.close()                release resources
//
// Every backend outputs the SAME 33-slot layout as MediaPipe BlazePose, so the
// rest of the pipeline (normalization, rest detection, matching, drawing) is
// identical regardless of algorithm. 17-keypoint models (MoveNet, YOLO) fill
// only the slots they have; the unused slots stay at visibility 0. Because a
// code is only ever matched against codes from the SAME family, the constant
// empty slots do not affect recognition.
//
// Only BlazePose is imported statically (it is the default and must always
// work). MoveNet and YOLO are loaded with dynamic import() the first time they
// are selected, so a CDN or model failure disables just that algorithm instead
// of breaking the whole app.

import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

export const NUM_LMS = 33;

// COCO-17 keypoint index -> BlazePose 33-slot index. This places COCO joints
// into the same slots BlazePose uses, so the app's key indices (nose 0, ears
// 7/8, shoulders 11/12, wrists 15/16, hips 23/24) line up for every backend.
const COCO_TO_BLAZE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

// Bones to draw for a 17-keypoint (COCO) skeleton, in BlazePose slot indices.
export const COCO_CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [0, 2], [0, 5], [2, 7], [5, 8],
];

// Full BlazePose bone list, derived from the model's own connection table.
export const BLAZE_CONNECTIONS = (PoseLandmarker.POSE_CONNECTIONS || []).map(
  (c) => [c.start, c.end]
);

function emptyPose() {
  const p = new Array(NUM_LMS);
  for (let i = 0; i < NUM_LMS; i++) p[i] = { x: 0, y: 0, visibility: 0 };
  return p;
}

// Turn 17 normalized COCO keypoints ({x,y,score}) into a 33-slot pose.
function cocoToPose(kpts) {
  const p = emptyPose();
  for (let i = 0; i < 17; i++) {
    const k = kpts[i];
    if (!k) continue;
    p[COCO_TO_BLAZE[i]] = { x: k.x, y: k.y, visibility: k.score ?? 0 };
  }
  return p;
}

// ---------- BlazePose (MediaPipe Tasks-Vision) ----------
const MP_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/";
const BLAZE_URLS = {
  "blaze-lite": MP_BASE + "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  "blaze-full": MP_BASE + "pose_landmarker_full/float16/1/pose_landmarker_full.task",
  "blaze-heavy": MP_BASE + "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
};
let mpFileset = null;

class BlazeBackend {
  constructor(key) { this.key = key; this.family = "blaze"; this.connections = BLAZE_CONNECTIONS; this.lm = null; }
  async load() {
    mpFileset = mpFileset || await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    const make = (delegate) => PoseLandmarker.createFromOptions(mpFileset, {
      baseOptions: { modelAssetPath: BLAZE_URLS[this.key], delegate },
      runningMode: "VIDEO",
      numPoses: 3,
    });
    try { this.lm = await make("GPU"); }
    catch (e) { console.warn("BlazePose GPU failed, using CPU:", e); this.lm = await make("CPU"); }
  }
  // MediaPipe returns 33 landmarks {x,y,z,visibility} already in image space.
  async detect(video, ts) { return this.lm.detectForVideo(video, ts).landmarks || []; }
  close() { try { this.lm?.close(); } catch {} }
}

// ---------- MoveNet (TensorFlow.js) ----------
class MoveNetBackend {
  constructor(key) { this.key = key; this.family = "movenet"; this.connections = COCO_CONNECTIONS; this.detector = null; }
  async load() {
    const tf = await import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.22.0/+esm");
    await import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter@4.22.0/+esm");
    await import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.22.0/+esm");
    await tf.setBackend("webgl");
    await tf.ready();
    const pd = await import("https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/+esm");
    const modelType = this.key === "movenet-thunder"
      ? pd.movenet.modelType.SINGLEPOSE_THUNDER
      : pd.movenet.modelType.SINGLEPOSE_LIGHTNING;
    this.detector = await pd.createDetector(pd.SupportedModels.MoveNet, { modelType });
  }
  async detect(video) {
    const w = video.videoWidth || 1, h = video.videoHeight || 1;
    const poses = await this.detector.estimatePoses(video, { flipHorizontal: false });
    return poses.map((p) =>
      cocoToPose(p.keypoints.map((k) => ({ x: k.x / w, y: k.y / h, score: k.score })))
    );
  }
  close() { try { this.detector?.dispose(); } catch {} }
}

// ---------- YOLO-Pose (ONNX Runtime Web + WebGPU) ----------
// Expects a YOLOv8/YOLO11-pose model exported to ONNX at 640x640. Because there
// is no universal public URL for such a file, the URL is supplied by the app
// (a constant or a value the user pastes) and passed into the constructor.
class YoloBackend {
  constructor(key, modelUrl) {
    this.key = key; this.family = "yolo"; this.connections = COCO_CONNECTIONS;
    this.modelUrl = modelUrl; this.session = null; this.ort = null;
    this.size = 640;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.size; this.canvas.height = this.size;
    this.cctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }
  async load() {
    if (!this.modelUrl) throw new Error("no YOLO model URL configured");
    this.ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.webgpu.bundle.min.mjs");
    this.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
    this.session = await this.ort.InferenceSession.create(this.modelUrl, {
      executionProviders: ["webgpu", "wasm"],
    });
  }
  // Letterbox the frame into a 640x640 RGB CHW float tensor.
  _preprocess(video) {
    const W = video.videoWidth, H = video.videoHeight, S = this.size;
    const scale = Math.min(S / W, S / H);
    const nw = Math.round(W * scale), nh = Math.round(H * scale);
    const padX = (S - nw) / 2, padY = (S - nh) / 2;
    this.cctx.fillStyle = "rgb(114,114,114)";
    this.cctx.fillRect(0, 0, S, S);
    this.cctx.drawImage(video, padX, padY, nw, nh);
    const data = this.cctx.getImageData(0, 0, S, S).data;
    const chw = new Float32Array(3 * S * S);
    const plane = S * S;
    for (let i = 0; i < plane; i++) {
      chw[i] = data[i * 4] / 255;               // R
      chw[i + plane] = data[i * 4 + 1] / 255;   // G
      chw[i + 2 * plane] = data[i * 4 + 2] / 255; // B
    }
    return { tensor: new this.ort.Tensor("float32", chw, [1, 3, S, S]), scale, padX, padY, W, H };
  }
  async detect(video) {
    if (!this.session) return [];
    const pre = this._preprocess(video);
    const feeds = {}; feeds[this.session.inputNames[0]] = pre.tensor;
    const out = await this.session.run(feeds);
    const o = out[this.session.outputNames[0]];
    const dims = o.dims, d = o.data;
    // Accept [1,56,N] or [1,N,56]; C=56 is 4 box + 1 conf + 17*3 kpts.
    let C, N, colMajor;
    if (dims[1] === 56) { C = dims[1]; N = dims[2]; colMajor = true; }
    else { N = dims[1]; C = dims[2]; colMajor = false; }
    const at = (row, col) => colMajor ? d[col * N + row] : d[row * C + col];
    let bestI = -1, bestConf = 0.25;
    for (let i = 0; i < N; i++) {
      const conf = at(i, 4);
      if (conf > bestConf) { bestConf = conf; bestI = i; }
    }
    if (bestI < 0) return [];
    const kpts = [];
    for (let k = 0; k < 17; k++) {
      const px = at(bestI, 5 + k * 3);
      const py = at(bestI, 5 + k * 3 + 1);
      const ps = at(bestI, 5 + k * 3 + 2);
      const x = (px - pre.padX) / pre.scale / pre.W;
      const y = (py - pre.padY) / pre.scale / pre.H;
      kpts.push({ x, y, score: ps });
    }
    return [cocoToPose(kpts)];
  }
  close() { try { this.session?.release?.(); } catch {} }
}

// Factory. `opts.yoloModelUrl` is only needed for the yolo family.
export function createBackend(key, opts = {}) {
  if (key.startsWith("blaze")) return new BlazeBackend(key);
  if (key.startsWith("movenet")) return new MoveNetBackend(key);
  if (key === "yolo") return new YoloBackend(key, opts.yoloModelUrl);
  throw new Error("unknown algorithm: " + key);
}

export function familyOf(key) {
  if (key.startsWith("movenet")) return "movenet";
  if (key === "yolo") return "yolo";
  return "blaze";
}
