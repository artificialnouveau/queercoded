# Queer Coded

Code a movement. Move to speak.

A webcam app that turns body movement into words. You **teach** the app a
movement (a pose, a gesture, a short dance) bound to a word or phrase, then
**perform** it to make the word appear on screen. Everything runs in the
browser; only pose **coordinates** are stored, never webcam video.

## What the code does

The whole app is three static files (`index.html`, `style.css`, `app.js`) with
no build step. `app.js` does all the work:

1. **Pose tracking.** Each video frame goes through a
   [MediaPipe Pose Landmarker][mp] model running in the browser (WebAssembly,
   GPU when available). The model returns 33 body landmarks (shoulders, elbows,
   wrists, hips, knees, and so on) as x/y coordinates plus a visibility score.
   A picker under the video lets you choose between three model sizes:
   - **Lite**: fastest, but loses track of fast-moving limbs.
   - **Full** (default): the best accuracy/speed balance for dance.
   - **Heavy**: most accurate, but can drop the frame rate on slower machines.

   All three output the same 33 landmarks, so codes saved with one model still
   work after switching. Your choice is remembered in the browser.

2. **Normalization.** Raw landmarks depend on where you stand and how big you
   appear. Each frame is re-centered on the hip midpoint and scaled by torso
   length, so a code works whether you are near or far from the camera.

3. **Segmentation.** The app watches for real motion to start a capture and
   ends it when you settle: either holding still, or putting a hand over your
   face (the circle drawn on the video). The face is used because it stays
   reliably tracked even when a close camera framing crops the lower body.
   Captures that never really go anywhere (a twitch, tracker jitter) are
   dropped quietly. The rest at both ends is trimmed so a saved code spans
   exactly the movement.

4. **Matching.** A captured movement is resampled to a fixed 20 frames and
   compared to every saved code with Dynamic Time Warping, which tolerates
   performing the movement faster or slower than you taught it. If the best
   match beats the sensitivity threshold, the word fires: it flashes on
   screen, pings, and joins the phrase strip.

5. **Storage.** Codes live in `localStorage` as JSON (word + 20 normalized
   skeleton frames). They can be exported/imported as JSON files, replayed as
   a "ghost" skeleton over the live video, renamed, and deleted.

## Run locally

The camera needs a secure context, so serve over `localhost` (opening the file
directly will not get camera access in most browsers):

```bash
cd queercoded
python3 -m http.server 8000
# open http://localhost:8000
```

Any static server works (`npx serve`, etc.). Allow camera access when prompted.

## Use

- **Teach:** name a word, click Record. Hold still until the indicator under
  the video says MOVE, perform your movement, then hold still again. It saves
  by itself.
- **Perform:** from rest, perform a saved code and return to rest. When it
  matches, the word appears and joins the phrase strip. Use the sensitivity
  slider and the live distance readout to calibrate.
- **Codes:** play back, rename, delete, and export/import your codes as JSON.

### If it says "No clear movement captured"

- Check the skeleton overlay is actually drawn on your body; if not, adjust
  framing so at least your head and shoulders are clearly in view.
- Make the movement bigger. Very small movements are treated as jitter.
- Try the Full or Heavy tracking model; Lite can lose fast limbs.
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
