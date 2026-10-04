# Packed K-quant prefill, closing vision and LoRA readings

**Purpose.** The vision and LoRA no-regression readings for the packed K-quant prefill work, taken on the
closing build. The vision gate (latency `<= 1.25x`, decode tps `>= 0.80x`) and the LoRA playback gate
(wall-clock tps `>= 0.80x`) are looser than 0.90x, so they may be read unpinned, median of three, both
builds in one session. The LoRA train gate (`>= 0.95x`) is tighter, so the train figure here is for
reading only. The scored train reading is a pinned same-hour A/B in the owner's closing gate.

**Pinned: no.**

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| baseline | HEAD `f02bdae` (the last build before the packed K-quant work), from `git archive`, built with package only | `cdd4de12314667d2` |
| candidate | HEAD `59c53cc` plus the attention barrier fix and the decode GEMV scratch held at upload (working tree) | `5ab4c78c505a4cef` |

Host: GTX 1080 (8 GiB), desktop session sharing the card; JDK 25. 2026-10-04 09:12Z to 10:19Z, one engine at
a time.

**Commands.** Each build's own script from its own tree:
- `compare-vision.sh --gpu --skip-build --no-publish`, three runs per build, alternated base, candidate,
  base, candidate, base, candidate;
- `compare-lora.sh --gpu --reps 3 --skip-build --no-publish`, base then candidate.

## Results

| Gate | Baseline (median of 3) | Candidate (median of 3) | Ratio | Threshold | Result |
|---|---|---|---|---|---|
| vision `latency_ms` | 506,325 | 507,239 | 1.002x | `<= 1.25x` | met |
| vision decode tps | 1.3954 | 1.3886 | 0.995x | `>= 0.80x` | met |
| LoRA playback, wall-clock tps | 10.687 | 10.938 | 1.023x | `>= 0.80x` | met |
| LoRA train ms per pass (reading only) | 3,333 | 3,333 | 1.000x | `>= 0.95x` speed, pinned | owed to the owner's pinned A/B |

Six vision captions are byte-identical across both builds. Every LoRA repetition trained 15 passes to loss
1.1797 on both builds.

## How to read it

- **Vision does not reach the changed code.** The CLIP encoder runs on the CPU, and moondream2's text half
  is Phi-2, which has no device weight path, so the packed prefill matmul, the attention fix and the GEMV
  scratch change cannot touch it. The readings confirm it: equal times and identical captions.
- **LoRA training keeps its frozen weights FP32-resident** and never takes the packed path. Playback with
  Q4-resident weights multiplies row by row with the decode kernel (the tiled kernel is a later item).
  Neither moved.
- The train figure is coarse: the REPL log reports the 15-pass total in whole seconds (50 s, 52 s or 51 s
  per repetition), so ms per pass moves in steps of about 67 ms.
- Files: `vision-<base|cand>-<n>.json`, `lora-<base|cand>.json` (median of three) and
  `lora-<base|cand>-rep<n>.json`. Local paths are replaced by `<repo>/`, `<baseline-tree>/` and
  `<scratch>/`.
