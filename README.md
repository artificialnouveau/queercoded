# Queer Coded

Code a movement. Move to speak.

A webcam app that turns body movement into words. Using [MediaPipe][mp] pose
tracking in the browser, you **teach** the app a movement (a pose or a short
dance) bound to a word or phrase, then **perform** it to make the word appear
on screen. Every gesture starts and ends in a resting pose (standing, arms at
your sides).

Only pose **coordinates** are stored, never webcam video.

## Run locally

The camera needs a secure context, so serve over `localhost` (opening the file
directly will not get camera access in most browsers):

```bash
cd queercode
python3 -m http.server 8000
# open http://localhost:8000
```

Any static server works (`npx serve`, etc.). Allow camera access when prompted.

## Use

- **Teach:** name a word, click Record, start at rest, perform the movement,
  return to rest. The code is saved to your browser.
- **Perform:** from rest, perform a saved code and return to rest. When it
  matches, the word appears and joins the phrase strip. Use the sensitivity
  slider and the live distance readout to calibrate.
- **Codes:** rename, delete, and export/import your codes as JSON.

## Hosting (GitHub Pages)

`index.html` is at the repo root, so Pages can serve it directly:

1. Push to `main`.
2. Repo **Settings → Pages → Build and deployment → Deploy from a branch**.
3. Branch: `main`, folder: `/ (root)`. Save.
4. Site publishes at `https://artificialnouveau.github.io/queercoded/`.

## Docs

- [DESIGN.md](DESIGN.md) — architecture, pose pipeline, matching, open questions.
- [SHARING.md](SHARING.md) — how a shared code library could work later.

[mp]: https://ai.google.dev/edge/mediapipe
