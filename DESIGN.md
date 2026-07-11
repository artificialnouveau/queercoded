# Queer Coded — Design

A browser app that turns body movement into language. Using a webcam and
MediaPipe pose tracking, you can **teach** the app a movement (a static pose or
a short dance) and bind it to a word or phrase, then **perform** that movement
to make the word appear on screen.

The name plays on "queer coded": movement as a private, embodied code.

---

## Goals

1. **Teach** — record a body movement and assign it a word/phrase.
2. **Perform** — recognize a performed movement live and display the word.
3. Runs entirely in the browser, webcam-enabled, no install.
4. Stores only pose **coordinates**, never webcam video.

Non-goals (for v1): multi-person, full sign-language grammar, cloud sync
(see [SHARING.md](SHARING.md) for the future shared-database plan).

---

## Architecture

```
webcam ─▶ MediaPipe PoseLandmarker ─▶ 33 landmarks/frame
                                         │
                    normalize (torso-relative)  ─▶ pose vector (66 numbers)
                                         │
        ┌────────────── Teach ──────────┴────────── Perform ──────────────┐
        │ record frames between rest      rest-delimited segmentation      │
        │ poses, trim, resample to 20     capture movement between rests   │
        │ ─▶ template stored in           ─▶ resample ─▶ DTW vs templates  │
        │    localStorage                 ─▶ best < threshold ⇒ show word  │
        └─────────────────────────────────────────────────────────────────┘
```

Single-page, no build step. Files: `index.html`, `style.css`, `app.js`, and
`backends.js` (ES modules). MediaPipe Tasks-Vision loads from CDN; MoveNet
(TensorFlow.js) and YOLO-Pose (ONNX Runtime Web) are loaded on demand via
dynamic `import()` only when selected, so a failure disables just that
algorithm rather than the whole app.

---

## Pose pipeline

- **Algorithms:** user-selectable across three families (default
  `blaze-full`, persisted in localStorage), all normalized to the same 33-slot
  layout by `backends.js`:
  - **BlazePose** (MediaPipe): 33 native points, GPU delegate with CPU
    fallback, `numPoses: 3` then keep the most prominent person.
  - **MoveNet** (TensorFlow.js): 17 COCO points mapped into the 33 slots.
  - **YOLO-Pose** (ONNX Runtime Web + WebGPU): 17 COCO points; the app
    letterboxes the frame to 640x640, runs the session, and decodes the
    top-confidence detection. Model URL supplied by the user.
  Because the three disagree on scale, codes are tagged with their family and
  matched only within it (switching algorithm means re-teaching). The mapped
  key indices (nose 0, ears 7/8, shoulders 11/12, wrists 15/16, hips 23/24)
  line up across all families, so rest detection and normalization are shared.
- **Visibility gate:** we only use a frame when the shoulders (landmarks 11,
  12) have visibility > 0.35. Hips are NOT required to be visible: the model
  estimates their position even out of frame, which is enough for
  normalization, and requiring them blocked close face-and-torso framings.

### Normalization (why matching is position/size invariant)

Raw landmark coordinates depend on where you stand and how close you are. We
remove that:

- Translate so the **hip midpoint** is the origin.
- Scale by **torso length** (hip-midpoint to shoulder-midpoint).
- Keep `x, y` only (MediaPipe `z` is noisier). Result: a 66-number vector per
  frame that is the same whether you are near/far, left/right in frame.

---

## The resting pose (gesture boundaries)

Gestures are **bracketed by the hand-over-face rest pose**: covering the face
arms a capture, moving the hand away starts it, and covering the face again
ends it. Nothing else can start a recording, so ordinary movement never fires
by accident. Frames at either end where a wrist is still near the face (the
trip to and from the pose) are trimmed, and a capture whose trimmed frames
never travel a minimum distance is dropped as a false start. For anyone who
prefers explicit control, a **Manual trigger** mode (hold a button or Spacebar)
captures instead.

`isResting()` is true when either wrist sits near the face (distance normalized
by shoulder width) AND is roughly under the face centre horizontally. The
horizontal-centring gate exists because the pose tracks the wrist, which sits
below the face when the palm covers it, so a radius alone cannot distinguish a
palm over the face from a hand beside it. Both conditions use hysteresis (easier
to enter than to leave) to stop boundary flicker. The face anchor averages whichever of nose/ears are visible,
with a low visibility gate, because the covering hand itself occludes the nose.
The face was chosen over the hips because close framings often crop the hips or
track them weakly, while the face stays solid. The app draws a **target circle
over the face** that lights up when a wrist is close enough, so the threshold
is visible rather than guessed. Only the upper body (face, shoulders, wrists,
and roughly-estimated hips for normalization) needs to be in frame, so close
framing and seated use both work.

- **Perform:** a small state machine: `idle` (waiting for the face to be
  covered) -> `armed` (hand on face) -> `recording` (hand moved away) -> back
  to `armed` when the face is covered again, which trims and matches the
  capture. A capture that never returns to the face is abandoned after 10 s.
- **Teach:** the identical bracket. After the countdown it waits for the hand
  on the face, records from the moment it leaves, and saves when the face is
  covered again. Near-face frames are trimmed from both ends, so every stored
  template spans only the movement itself. In Manual trigger mode the record
  button becomes a start/stop toggle.

---

## Matching (Dynamic Time Warping)

- Each captured movement is **resampled to 20 frames** so lengths are uniform.
- Distance between two poses = mean per-landmark euclidean distance.
- Distance between two sequences = **DTW**, normalized by path length. DTW is
  robust to tempo differences (fast vs slow performance of the same move).
- The performed movement is compared against every saved template; the lowest
  distance wins if it is below the **sensitivity threshold**.

### Calibration

The Perform panel shows the live **best match** and its **distance**, plus a
confidence bar. The sensitivity slider is the threshold. Lower = stricter.
Users watch the number while performing to find a good threshold for their
codes and their space.

---

Ghost playback anchors on the live body when one is in frame: because the
stored seq is normalized to hip-centre/torso-length, it is reprojected using
the performer's current hip midpoint and torso length so the ghost dances on
them at their size, falling back to a fixed centre-screen spot when no body is
detected.

## Data model

A template (one code):

```json
{
  "id": "c_ab12cd3",
  "word": "hello",
  "seq": [[x0,y0, x1,y1, ... 66 numbers], ... 20 frames],
  "durMs": 2000,
  "createdAt": 1720600000000
}
```

`seq` is the only movement data: normalized pose coordinates. No imagery.

**Storage:** `localStorage` under `queercoded.templates.v1`. Export/import as a
JSON file is supported for backup and manual sharing today.

---

## UI / theme

- Three tabs: **Perform**, **Teach**, **Codes**.
- Big webcam stage with skeleton overlay (mirrored, selfie-style); matched word
  animates large over the video; a phrase strip collects matched words.
- **Look:** bathroom-tile background (white ceramic tiles + grout), with the
  app content as dark, floating cards. Rainbow gradient wordmark. The bathroom
  motif nods to queer space; "Queer Coded" as the title.

---

## Limitations / known trade-offs

- 2D only (ignores depth); movements that differ mainly in depth can collide.
- Single pose; no two-person duets.
- Static poses work but must still be bracketed by rest to trigger.
- Recognition quality depends on lighting, framing, and threshold calibration.
- Similar movements bound to different words will be hard to disambiguate.

---

## Decisions (confirmed)

1. **Use / audience:** audience-facing — performance and installation. Favors a
   clean big-word display, a match sound, and reliable triggering in front of
   people.
2. **Dynamic** capture (not static-only): gestures are movement sequences
   delimited by the rest pose. Implemented.
3. **Full body — arms and legs:** matching compares all 33 landmarks, so leg and
   whole-body dance movement counts, not just arms. Implemented.
4. **Output:** on-screen text **plus a ping** (a short confirmation tone) on each
   match. A "Ping on match" toggle is in the Perform panel. Implemented.
5. **Multiple examples per word:** yes. Record the same word several times; each
   take is stored as an example and the best (nearest) match across a word's
   examples wins. The Codes list groups examples under one word. Implemented.
6. **Seated / accessibility:** yes. The rest pose (a hand over the face) only
   needs the head, shoulders, and one wrist in frame, so it works standing or
   seated. A **Manual trigger** mode (hold a button or Spacebar) is provided as a
   fallback for anyone who cannot reliably reach or hold the rest pose.
   Implemented.
7. **Shared library:** yes, planned for later. See [SHARING.md](SHARING.md).
   v1 ships with local storage + JSON export/import.

Still open: text-to-speech output (in addition to the ping), a dedicated
full-screen performance view, and per-code thresholds.

---

## Roadmap

- [x] v1: teach + perform, rest segmentation, localStorage, export/import
- [x] Ping on match; sound toggle
- [x] Multiple examples per word (nearest-of-N matching)
- [x] Seated support + manual (hold) trigger for accessibility
- [x] Ghost playback: click a saved code to replay its skeleton movement
  (reprojected from stored coordinates, no video involved)
- [ ] Optional text-to-speech output per word
- [ ] Full-screen performance/installation view
- [ ] Per-code threshold
- [ ] Shared code library (see [SHARING.md](SHARING.md))
