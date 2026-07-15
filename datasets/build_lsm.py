"""
Build the LSM (LibriSpeech-MUSAN) shift benchmark.

LSM isolates a noise-family shift: read speech from LibriSpeech test-clean is
mixed with out-of-family MUSAN noise, holding every other factor fixed. The
build is fully seeded and bit-exact, so running this script against the public
source corpora reproduces the released set; verify with lsm_sha256.txt.

Recipe (identical to build_esm.py, only the speech source differs):
  - 300 pairs, utterances >= 3 s, sample rate 16 kHz, PCM_16
  - SNR in {2.5, 7.5, 12.5, 17.5} dB, round-robin by pair index
  - noise power scaled to the target SNR, shared anti-clip rescale keeps
    the clean/noisy pair aligned
  - seed 1234

Sources (download separately, both permissive):
  - LibriSpeech test-clean   https://www.openslr.org/12   (CC BY 4.0)
  - MUSAN noise              https://www.openslr.org/17   (CC BY 4.0)

Usage:
  python build_lsm.py --librispeech /path/to/LibriSpeech/test-clean \
                      --musan /path/to/musan/noise --out ./lsm_ood
"""
import os, glob, argparse
import numpy as np
import soundfile as sf
import soxr
from multiprocessing import Pool

N_PAIRS, SNRS, SEED = 300, [2.5, 7.5, 12.5, 17.5], 1234
EPS = 1e-10


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
    if peak > 0.99:                          # shared rescale keeps the pair aligned
        s, y = s / peak * 0.99, y / peak * 0.99
    name = f"lsm_{i:04d}_snr{snr}.wav"
    sf.write(os.path.join(out, "clean", name), s, 16000, subtype="PCM_16")
    sf.write(os.path.join(out, "noisy", name), y, 16000, subtype="PCM_16")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True, help="LibriSpeech test-clean root")
    ap.add_argument("--musan", required=True, help="MUSAN noise root")
    ap.add_argument("--out", default="./lsm_ood")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    for d in ("clean", "noisy"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    speech = sorted(glob.glob(os.path.join(args.librispeech, "*", "*", "*.flac")))
    keep = [f for f in speech if sf.info(f).frames / sf.info(f).samplerate >= 3.0]
    noises = sorted(glob.glob(os.path.join(args.musan, "*", "*.wav")))
    print(f"speech>=3s: {len(keep)} | noises: {len(noises)}")
    sel = rng.choice(len(keep), N_PAIRS, replace=False)
    jobs = [(i, keep[si], noises[rng.integers(0, len(noises))],
             SNRS[i % 4], rng.random(), args.out) for i, si in enumerate(sel)]
    with Pool(min(64, os.cpu_count())) as p:
        n = sum(p.map(build_pair, jobs, chunksize=4))
    print("built", n, "pairs ->", args.out)


if __name__ == "__main__":
    main()
