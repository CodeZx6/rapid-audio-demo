# PACE listening examples

Static demo page: open `index.html` in a browser, or host the folder on any
static web server (GitHub Pages, anonymous.4open.science, etc.). No build
step, no external dependencies, no author-identifying information.

- 15 utterances, 3 per benchmark (VoiceBank-DEMAND, DNS no-reverb, LSM, ESM, EARS-WHAM).
- Per utterance 4 clips: noisy input / frozen backbone at default NFE /
  +PACE at reduced NFE / clean reference (16 kHz mono wav, `audio/`).
- `samples.json` holds per-clip metadata: source filename, backbone, NFE,
  PESQ (noisy/backbone/+PACE), UTMOS and DNSMOS OVRL (backbone vs +PACE),
  and the transcript shown on the page.
- Selection rule: +PACE beats the backbone on PESQ and on the majority of
  non-intrusive perceptual scores (UTMOS, DNSMOS OVRL, SQUIM MOS), with a
  clearly audible denoising margin over the noisy input.
