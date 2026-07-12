# Queercoded

Code a movement. Move to speak.

A webcam app that turns body movement into words. You **teach** the app a
movement (a pose, a gesture, a short dance) bound to a word or phrase, then
**perform** it to make the word appear on screen. Everything runs in the
browser; only pose **coordinates** are stored, never webcam video.

## What the code does

The whole app is three static files (`index.html`, `style.css`, `app.js`) with
no build step. `app.js` does all the work:

1. **Pose tracking.** Each video frame goes through a pose-estimation model
   running entirely in the browser. A picker under the video chooses between
   three genuinely different algorithms (see `backends.js`):
   - **MediaPipe BlazePose** (Lite / Full / Heavy): 33 landmarks, the default
     and the best all-round single-dancer accuracy in the browser. Full is
     recommended; Heavy is most accurate but heavier; Lite is fastest.
   - **MoveNet** (Lightning / Thunder, via TensorFlow.js): 17 keypoints,
     loaded from a CDN. Thunder is more accurate, Lightning faster.
   - **YOLO-Pose** (via ONNX Runtime Web + WebGPU): 17 keypoints. Needs a
     model file you supply (see "Adding a YOLO model" below).

   Internally every algorithm is mapped to the same 33-slot skeleton, so the
   rest of the app is identical regardless of choice. Because the three do not
   agree on scale, **codes are saved per algorithm family**: a code taught with
   BlazePose is only matched against other BlazePose codes, and switching
   algorithms means re-teaching. Each code shows a small badge (BlazePose /
   MoveNet / YOLO) in the Codes list. Your algorithm choice is remembered.
   If several people are in frame, only the nearest (largest) skeleton is
   tracked, so a bystander in the background does not steal the tracking.

2. **Normalization.** Raw landmarks depend on where you stand and how big you
   appear. Each frame is re-centered on the hip midpoint and scaled by torso
   length, so a code works whether you are near or far from the camera.

3. **Teach vs Perform.** Teaching a code is a deliberate recording: cover your
   face with your RIGHT hand (a big R marks it) and hold for a 3-second
   countdown to start, perform the movement, then cover your face with your
   LEFT hand (a big L) and hold to stop and save. The hand-to-face
   moments are trimmed so a code spans only the movement itself. Performing, by
   contrast, is continuous: the app keeps a rolling buffer of your recent poses
   and, several times a second, matches it against your saved codes. When a
   movement matches, its label is shown and spoken. No recording step, no
   button, no cover pose. (A Manual hold-to-capture mode is available as a
   fallback for noisy backgrounds.)

4. **Matching.** A captured movement is resampled to a fixed 20 frames and
   compared to every saved code with Dynamic Time Warping, which tolerates
   performing the movement faster or slower than you taught it. Each word with
   two or more examples gets its own threshold, auto-calibrated from how
   consistent those examples are (the sensitivity slider scales it globally),
   and a match is rejected when a different word is nearly as close, so
   ambiguous movements do not fire the wrong word. When a match wins, the word
   flashes above your head, pings, is spoken aloud (toggle in Perform), and
   joins the phrase strip.

5. **Storage.** Codes live in `localStorage` as JSON (word + 20 normalized
   skeleton frames). They can be exported/imported as JSON files, replayed as
   a "ghost" skeleton over the live video, renamed, and deleted.

6. **Riso look.** While a pose is being saved (Teach recording), and for a
   moment whenever a movement is matched (Perform), the performer is rendered in
   a red/yellow duotone with film grain and a brief chromatic glitch, so the
   live body looks the way a saved pose does. Each saved code also shows a small
   "pose card": a triptych of red-with-yellow-offset silhouette figures drawn
   from the stored skeleton frames (coordinates only, never a webcam image), in
   the Codes list and as a preview right after teaching.

## Run locally

The camera needs a secure context, so serve over `localhost` (opening the file
directly will not get camera access in most browsers):

```bash
cd queercoded
python3 -m http.server 8000
# open http://localhost:8000
```

Any static server works (`npx serve`, etc.). Allow camera access when prompted.

## Adding a YOLO model

The YOLO-Pose option needs a YOLOv8 or YOLO11 pose model exported to ONNX at
640x640. There is no universal public URL for one, so you supply it: the first
time you pick YOLO-Pose, the app prompts for a URL to the `.onnx` file (it must
be served with permissive CORS, e.g. from the same GitHub Pages repo). To make
one with [Ultralytics](https://docs.ultralytics.com/):

```bash
pip install ultralytics
yolo export model=yolov8n-pose.pt format=onnx imgsz=640
```

Commit the resulting `yolov8n-pose.onnx` to the repo (note it is several MB) and
give the app its URL (e.g. `https://<user>.github.io/queercoded/yolov8n-pose.onnx`).
The URL is remembered in the browser. MoveNet and BlazePose need no setup.

> Note: MoveNet and YOLO-Pose are newer additions and depend on third-party
> CDN builds (TensorFlow.js, ONNX Runtime Web). BlazePose is the most
> thoroughly tested path; if an algorithm fails to load, the app reverts to
> the previous one and keeps running.

## Use

- **Teach:** name a word, click Record, cover your face with your right hand
  to start, perform your movement, then cover your face with your left hand.
  It saves by itself.
- **Perform:** just perform. The app watches continuously and, when your
  movement matches a saved code, the word appears above your head, is spoken
  aloud, and joins the phrase strip. Speak the whole phrase, undo the last
  word, or clear it with the buttons above the strip. Use the sensitivity
  slider and the live distance readout to calibrate.
- **About:** what the app does and a plain-language privacy summary (everything
  runs locally; no data is sent anywhere).
- **Codes:** play back a whole word or a single example (dance along with the
  ghost skeleton), rename, delete a word or just one of its examples, and
  export/import all codes as JSON.
- **AlgoDance:** the zine, read full screen one page at a time. Flip with a
  click, the chevrons, or the arrow keys; the header collapses while reading
  (the floating Menu button brings it back). Rendered with pdf.js, loaded only
  when the tab is opened; if that fails, the PDF embeds directly instead.

### If it says "No clear movement captured"

- Check the white outline is actually drawn around your body; if not, adjust
  framing so at least your head and shoulders are clearly in view.
- Make the movement bigger. Very small movements are treated as jitter.
- Try BlazePose Full or Heavy; Lite (and Lightning) can lose fast limbs.
- As a fallback, switch Trigger to Manual in the Perform tab and hold the
  button (or Spacebar) while recording.

## Hosting (GitHub Pages)

`index.html` is at the repo root, so Pages can serve it directly:

1. Push to `main`.
2. Repo **Settings > Pages > Build and deployment > Deploy from a branch**.
3. Branch: `main`, folder: `/ (root)`. Save.
4. Site publishes at `https://artificialnouveau.github.io/queercoded/`.

## Docs

- [DESIGN.md](DESIGN.md): architecture, pose pipeline, matching, open questions.
- [SHARING.md](SHARING.md): how a shared code library could work later.

[mp]: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
