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

Single-page, no build step. Three files: `index.html`, `style.css`, `app.js`
(ES module). MediaPipe Tasks-Vision and the pose model load from CDN.

---

## Pose pipeline

- **Model:** `pose_landmarker_lite` (fast, GPU delegate), `runningMode: VIDEO`,
  one pose. 33 landmarks per frame, each with `x, y, z, visibility` in
  normalized image coordinates.
- **Visibility gate:** we only use a frame when shoulders + hips (landmarks
  11, 12, 23, 24) have visibility > 0.5. This avoids garbage when the person is
  partly out of frame.

### Normalization (why matching is position/size invariant)

Raw landmark coordinates depend on where you stand and how close you are. We
remove that:

- Translate so the **hip midpoint** is the origin.
- Scale by **torso length** (hip-midpoint to shoulder-midpoint).
- Keep `x, y` only (MediaPipe `z` is noisier). Result: a 66-number vector per
  frame that is the same whether you are near/far, left/right in frame.

---

## The resting pose (gesture boundaries)

Every gesture starts and ends in a **resting pose**: hands on hips. This was
chosen over arms-at-sides so the hands stay inside a close upper-body camera
frame (no need to see the legs), and because it is a distinct, easy-to-hold
pose. It gives clean, automatic segmentation without a button press. For anyone
who cannot reliably reach or hold it, a **Manual trigger** mode (hold a button
or Spacebar) captures the movement instead.

`isResting()` is true when each wrist sits near its hip: the wrist-to-hip
distance, normalized by shoulder width, is below a threshold for both hands. It
needs only the upper body (shoulders, hips, wrists) to be visible, so it works
with the camera close in and works seated too.

- **Perform:** a small state machine. In `rest`, leaving rest for a couple of
  frames starts a capture. In `move`, returning to rest for a few frames ends
  it. The trailing rest frames are trimmed, then the captured movement is
  matched. A capture that never returns to rest is abandoned after 6 s.
- **Teach:** uses the same movement-delimited capture as Perform (no fixed
  timer). After you hit record it waits until you are at rest, starts capturing
  when you move, and ends when you return to rest. Leading and trailing rest is
  trimmed, so every stored template begins and ends at the same neutral pose.
  In Manual trigger mode the record button becomes a start/stop toggle.

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
6. **Seated / accessibility:** yes. The rest pose is defined relative to the
   hips (wrists at/below hip height, close to the body), so it works standing or
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
