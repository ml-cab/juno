# Context policy: LoRA and vision no-regression (unpinned, both builds in one session)

**Purpose.** Context shifting and sliding-window attention touch the attention paths of every handler, including
the LoRA handlers (training and playback) and moondream2's Phi-2 text half (vision). These gates are looser than
0.90x, so they are read unpinned, median of three, both builds in one session.

**Pinned: no.**

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| baseline (A) | HEAD `de98ff6`, the last build before context shifting and sliding windows | `620a5caf606538b6` |
| candidate (B) | HEAD `0f87603` plus sliding windows and the in-place device KV shift | `16f8794d833b1dff` |

**Commands.** Each tree's own scripts, GPU, `--skip-build --no-publish`: `compare-lora.sh --gpu --reps 3`
(baseline, then candidate), and `compare-vision.sh --gpu` three times per build, alternating A B A B A B
(2026-10-09, 13:19 to 14:23 local).

## Results

| Gate | Baseline (median) | Candidate (median) | B/A | Threshold | Result |
|---|---|---|---|---|---|
| LoRA train, total ms | 44,000 (43,000 / 44,000 / 44,000) | 41,000 (43,000 / 41,000 / 41,000) | 0.932 | `<= 1.25x` | met |
| LoRA playback, wall-clock tps | 12.52 (12.41 / 12.75 / 12.52) | 13.36 (13.31 / 13.54 / 13.36) | 1.067 | `>= 0.80x` | met |
| Vision latency, ms | 486,974 (480,255 / 486,974 / 496,897) | 484,801 (484,297 / 496,140 / 484,801) | 0.996 | `<= 1.25x` | met |
| Vision decode, `TokenProduced` tps | 1.426 (1.505 / 1.421 / 1.426) | 1.463 (1.463 / 1.442 / 1.471) | 1.026 | `>= 0.80x` | met |

## How to read it

- Both builds train to the same target (15 passes, final loss 1.1797) and recall the trained name on playback.
- All six vision replies are identical text on both builds.
- Vision runs about eight minutes per request on both builds, as in the last published vision gate
  (`20261004T091233Z-packed-kquant-close-vision-lora`, about 506 s): the CLIP encoder and moondream2's text half run on
  the CPU path. Not new, and the same on both sides.
- Local paths are replaced by `<repo>/`, `<baseline-tree>` and `<scratch>`.
