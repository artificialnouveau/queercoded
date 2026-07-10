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

Every gesture starts and ends in a **resting pose**: standing, arms at your
sides. This gives clean, automatic segmentation without a button press.

`isResting()` is true when both wrists are at/below hip height and horizontally
close to the body (not extended outward), relative to shoulder width.

- **Perform:** a small state machine. In `rest`, leaving rest for a couple of
  frames starts a capture. In `move`, returning to rest for a few frames ends
  it. The trailing rest frames are trimmed, then the captured movement is
  matched. A capture that never returns to rest is abandoned after 6 s.
- **Teach:** the fixed-length recording is trimmed of leading and trailing rest
  frames, so every stored template begins and ends at the same neutral pose.

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

## Open questions (input needed)

These are the decisions that would change the design. Answers from you go here:

1. **Audience/use:** performance art, an accessibility/AAC tool, a game, an
   installation? This drives the tone and which failure modes matter most.
2. **Static vs dynamic:** should we add an explicit "hold a pose" mode, or is
   rest-delimited dynamic capture enough?
3. **Full body vs arms-only:** do legs/dance matter, or should matching weight
   arms and torso? (Affects which landmarks we compare.)
4. **Output:** just on-screen text, or also speech (text-to-speech), sound, or
   visuals per word?
5. **Multiple examples per word:** allow recording several takes of one word to
   improve robustness (average/nearest-of-N)?
6. **Sharing model:** personal only, or a shared library of codes people can
   browse and load? (See SHARING.md — decide later.)
7. **Hosting/target:** GitHub Pages is set up; any custom domain?

---

## Roadmap

- [x] v1: teach + perform, rest segmentation, localStorage, export/import
- [ ] Multiple examples per code; per-code threshold
- [ ] Optional text-to-speech output
- [ ] Shared code library (see [SHARING.md](SHARING.md))
