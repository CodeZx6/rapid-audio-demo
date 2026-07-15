"""
Build the ESM (EARS-MUSAN) shift benchmark.

ESM isolates a speech-style shift. It uses the exact recipe of build_lsm.py and
changes only the speech source: expressive EARS speech (speakers p102-p107)
replaces the read LibriSpeech speech, while the MUSAN noise, SNR schedule,
mixing and anti-clip stay identical. Comparing LSM and ESM therefore separates a
noise-family shift from a speech-style shift under one shared protocol. The build
is seeded and bit-exact; verify with esm_sha256.txt.

Recipe:
  - 300 pairs, utterances >= 3 s after resampling, 16 kHz, PCM_16
  - SNR in {2.5, 7.5, 12.5, 17.5} dB, round-robin by pair index
  - seed 4321

Sources (download separately):
  - EARS         https://sp-uhh.github.io/ears_dataset/   (CC BY-NC 4.0)
  - MUSAN noise  https://www.openslr.org/17               (CC BY 4.0)

The released ESM mixtures inherit the CC BY-NC 4.0 license of EARS.

Usage:
  python build_esm.py --ears /path/to/EARS --musan /path/to/musan/noise --out ./esm_ood
"""
import os, glob, argparse
import numpy as np
import soundfile as sf
import soxr
from multiprocessing import Pool

SPEAKERS = ["p102", "p103", "p104", "p105", "p106", "p107"]
N_PAIRS, SNRS, SEED = 300, [2.5, 7.5, 12.5, 17.5], 4321
EPS = 1e-10


def resample_and_check(path):
    """Keep an EARS utterance only if it is >= 3 s after the 48k->16k resample."""
    try:
        s, sr = sf.read(path)
        if s.ndim > 1:
            s = s.mean(1)
        if sr != 16000:
            s = soxr.resample(s, sr, 16000)
        if len(s) / 16000.0 >= 3.0:
            return path
    except Exception:
        pass
    return None


def build_pair(job):
    i, sp_path, noise_path, snr, off_frac, out = job
    s, sr = sf.read(sp_path)
    if s.ndim > 1:
        s = s.mean(1)
    if sr != 16000:
        s = soxr.resample(s, sr, 16000)
    n, sr2 = sf.read(noise_path)
    if n.ndim > 1:
        n = n.mean(1)
    if sr2 != 16000:
        n = soxr.resample(n, sr2, 16000)
    if len(n) < len(s):
        n = np.tile(n, int(np.ceil(len(s) / len(n))))
    off = int(off_frac * max(1, len(n) - len(s)))
    n = n[off:off + len(s)]
    ps, pn = np.mean(s ** 2) + EPS, np.mean(n ** 2) + EPS
    n = n * np.sqrt(ps / (pn * 10 ** (snr / 10)))
    y = s + n
    peak = max(np.abs(y).max(), np.abs(s).max()) + EPS
    if peak > 0.99:
        s, y = s / peak * 0.99, y / peak * 0.99
    name = f"esm_{i:04d}_snr{snr}.wav"
    sf.write(os.path.join(out, "clean", name), s, 16000, subtype="PCM_16")
    sf.write(os.path.join(out, "noisy", name), y, 16000, subtype="PCM_16")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ears", required=True, help="EARS root (speaker subfolders)")
    ap.add_argument("--musan", required=True, help="MUSAN noise root")
    ap.add_argument("--out", default="./esm_ood")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    for d in ("clean", "noisy"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    speech_all = []
    for spk in SPEAKERS:
        speech_all += sorted(glob.glob(os.path.join(args.ears, spk, "*.wav")))
    print(f"EARS p102..p107 candidate wavs: {len(speech_all)}")
    with Pool(min(48, os.cpu_count())) as p:
        results = p.map(resample_and_check, speech_all)
    keep = [r for r in results if r is not None]
    noises = sorted(glob.glob(os.path.join(args.musan, "*", "*.wav")))
    print(f"speech>=3s (post-resample): {len(keep)} | noises: {len(noises)}")
    sel = rng.choice(len(keep), N_PAIRS, replace=False)
    jobs = [(i, keep[si], noises[rng.integers(0, len(noises))],
             SNRS[i % 4], rng.random(), args.out) for i, si in enumerate(sel)]
    with Pool(min(48, os.cpu_count())) as p:
        n = sum(p.map(build_pair, jobs, chunksize=4))
    print("built", n, "pairs ->", args.out)


if __name__ == "__main__":
    main()
