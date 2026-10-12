# Vision and LoRA gates after the GPU host-overhead work (2026-10-11)

**Purpose.** The vision and LoRA no-regression gates for the GPU host-overhead tier's changes: Qwen2 and Qwen2.5 in
the device-resident decode region, host KV tensors reused across requests, and the GPU prefill window writing K and V
into the host KV cache straight from its download. moondream2's text half is a Phi-2 handler, which none of the changes
reach; LoRA training and playback decline the decode region. Both are expected unchanged.

**Pinned: no.** Both gates are looser than 0.90x (vision latency at most 1.25x and decode at least 0.80x; LoRA training
time at most 1.25x and playback at least 0.80x), so they may be read unpinned, median of three, both builds in one
session.

**Builds.** Baseline: `198467ad`, the last build before this tier changed code, built from `git archive` (jar sha256
`9af96224967a0057`; the tier's own pre-work copy of the same source is `6ae135b0e21839cf`, builds are not
byte-reproducible). Candidate: this tree, `198467ad` plus the uncommitted tier work, jar sha256 `921ab742ba3f362a`.
OpenJDK 25.0.3. Both report `git_commit` `198467ad`, the candidate because its changes are uncommitted.

## Command

Each tree's own script, alternating A B A B A B; vision on the GPU lane, LoRA on both lanes:

```bash
(cd <tree> && scripts/performance-tests/compare-vision.sh --gpu --skip-build --no-publish --out <dir>)
(cd <tree> && scripts/performance-tests/compare-lora.sh --gpu --skip-build --no-publish --out <dir>)
(cd <tree> && scripts/performance-tests/compare-lora.sh --cpu --skip-build --no-publish --out <dir>)
```

## Vision (moondream2, default `--prefill single`, 32 tokens, GPU)

| | Run 1 | Run 2 | Run 3 | Median | B/A | Gate |
|---|---|---|---|---|---|---|
| Latency, baseline (s) | 269.7 | 266.2 | 261.7 | 266.2 | | |
| Latency, candidate (s) | 265.4 | 265.4 | 262.9 | 265.4 | **0.997** | <= 1.25: met |
| Decode tps, baseline | 2.400 | 2.434 | 2.474 | 2.434 | | |
| Decode tps, candidate | 2.421 | 2.443 | 2.472 | 2.443 | **1.004** | >= 0.80: met |

The reply is identical on all six runs.

## LoRA (TinyLlama Q4_K_M, rank 8, q and v, to loss 1.2)

| Lane | Train ms per pass, baseline | Candidate | Time B/A (gate <= 1.25) | Playback tps, baseline | Candidate | B/A (gate >= 0.80) |
|---|---|---|---|---|---|---|
| GPU | 2733 / 2733 / 2733 | 2733 / 2733 / 2733 | **1.000** | 13.54 / 13.23 / 13.49 | 13.51 / 13.26 / 13.23 | **0.983** |
| CPU | 49800 / 50067 / 49733 | 49667 / 49667 / 49733 | **0.997** | 0.877 / 0.875 / 0.882 | 0.887 / 0.881 / 0.886 | **1.011** |

Every run trains in 15 passes to the target loss and recalls "My name is Juno". All four gates are met, and every
reading is within the run-to-run spread.

## How to read it

Medians of three per side; B/A is the candidate over the baseline. A latency or training-time ratio under 1 is faster.
Per-run `vision-perf.json`, `lora-perf.json` and `current-response.json` are in the `<kind>-<lane>-<side>-<run>/`
directories, with local paths replaced by `<repo>`, `<run-dir>` and `<scratch>`.
