# Vision gate after the CPU decode and prefill kernels work (2026-10-26)

**Purpose.** The vision no-regression gate for the CPU kernel work (vector Q6_K decode, Q4_K activation split, pooled
vector prefill attention, the prefill matmul tile): moondream2 image-to-text latency and decode throughput, the build
before the work against the final build, on the GPU lane and the CPU lane. moondream2's request is dominated by
host-side matmuls in both lanes.

**Pinned: no.** The vision gate is looser than 0.90x (latency at most 1.25x, decode at least 0.80x), so it is read
unpinned, median of three, both builds in one session, alternating A B A B A B per lane.

**Builds.**

| Side | Jar sha256 (first 16) | Build |
|---|---|---|
| baseline | `d3c4f655130ad917` | `git archive 01027ee9`, built with `mvn package -DskipTests` (the closing gate's baseline tree; same commit as the gate's baseline jar `b9a0440b3a1b597a`) |
| candidate | `4e480ab9131887eb` | `c2633a68` plus the in-recording allocation field |

The `git_commit` field in the baseline runs' `vision-perf.json` reads the parent repository's HEAD (the archived tree has
no `.git`); the jar hash above identifies the build.

## Command

Each tree's own script, alternating per lane:

```bash
(cd <tree> && scripts/performance-tests/compare-vision.sh --gpu --skip-build --no-publish --out <dir>)
(cd <tree> && scripts/performance-tests/compare-vision.sh --cpu --skip-build --no-publish --out <dir>)
```

## Result (moondream2, default `--prefill single`, 32 tokens)

| Lane | Latency, baseline (three runs) | Latency, candidate | Latency B/A (gate <= 1.25) | Decode tps, baseline | Decode tps, candidate | Decode B/A (gate >= 0.80) |
|---|---|---|---|---|---|---|
| GPU | 262.5 / 262.0 / 261.4 s | 145.6 / 145.8 / 147.6 s | **0.556** | 2.467 / 2.481 / 2.500 | 5.303 / 5.225 / 5.208 | **2.106** |
| CPU | 264.6 / 258.6 / 261.0 s | 144.1 / 145.0 / 146.2 s | **0.555** | 2.448 / 2.524 / 2.481 | 5.143 / 5.284 / 5.220 | **2.104** |

Both gates are met on both lanes; the candidate is about 1.8x faster end to end and decodes about 2.1x faster.

**Output.** Each build's reply is identical across all six of its runs (both lanes). The two builds' replies agree for
the first sentences and part after about 28 tokens ("Some squares overlap" against "There is also"): this work scales
sub-block sums in float lanes and vectorizes the prefill attention dot, which changes float rounding by design. Model
quality is scored by the NLL test (`20261027T025856Z-cpu-kernels-quality`).

Per-run `vision-perf.json` and `current-response.json` are in the `<lane>-<side>-<run>/` directories (local paths
replaced with `<repo>`, `<baseline-tree>` and `<scratch>`).
