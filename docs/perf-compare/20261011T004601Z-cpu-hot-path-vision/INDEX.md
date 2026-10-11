# Vision gate after the CPU hot-path work (2026-10-11)

**Purpose.** The vision no-regression gate for the CPU tier's changes (integer CPU kernels, the CPU worker pool, the
decode scratch): moondream2 image-to-text latency and decode throughput, pre-tier build against this tree, on the GPU
lane (as earlier tiers ran it) and the CPU lane (where the changes are).

**Pinned: no.** The vision gate is looser than 0.90x (latency at most 1.25x, decode at least 0.80x), so it may be read
unpinned, median of three, both builds in one session (README, "Test infrastructure").

**Builds.** Baseline: `aabc42cb`, jar sha256 `7ba4a3142c3ab0f4` (built from `git archive`). Candidate: this tree,
`aabc42cb` plus the uncommitted CPU work, jar sha256 `6ae135b0e21839cf`. OpenJDK 25.0.3.

## Command

Each tree's own script, alternating A B A B A B per lane:

```bash
(cd <tree> && scripts/performance-tests/compare-vision.sh --gpu --skip-build --no-publish --out <dir>)
(cd <tree> && scripts/performance-tests/compare-vision.sh --cpu --skip-build --no-publish --out <dir>)
```

## Result (moondream2, default `--prefill single`, 32 tokens)

| Lane | Latency, baseline (three runs) | Latency, candidate | Latency B/A (gate <= 1.25) | Decode tps, baseline | Decode tps, candidate | Decode B/A (gate >= 0.80) |
|---|---|---|---|---|---|---|
| GPU | 478.0 / 480.1 / 479.1 s | 270.6 / 271.4 / 269.3 s | **0.565** | 1.468 / 1.470 / 1.469 | 2.385 / 2.409 / 2.380 | **1.624** |
| CPU | 478.1 / 480.5 / 478.5 s | 269.6 / 267.8 / 271.0 s | **0.563** | 1.475 / 1.475 / 1.465 | 2.383 / 2.429 / 2.383 | **1.616** |

Both gates are met on both lanes; the candidate is about 1.8x faster end to end. The two lanes read almost the same
because moondream2's request is dominated by host-side matmuls in both, which are what the tier sped up.

**Output.** Each build's reply is identical across its three runs. The two builds' replies differ in one phrase ("each
containing different colors" against "each with different colors"): the 8-bit activation rounding of the integer
kernels in the text model's host-side matmuls. Same meaning, same image description.

Per-run `vision-perf.json` and `current-response.json` are in the `<lane>-<side>-<run>/` directories.
