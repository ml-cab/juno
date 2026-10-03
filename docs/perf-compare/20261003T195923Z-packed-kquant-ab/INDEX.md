# Packed K-quant prefill matmul: same-hour pinned A/B and its confirmation (2026-10-03, pinned)

Purpose: the switch's gate. Prefill windows wider than 8 rows used to expand packed Q4_K/Q5_K/Q6_K weights
to FP16 and run an FP16 GEMM; the candidate multiplies the packed weights with a tiled integer kernel.
Scored on Juno absolute t/s, candidate (B) over baseline (A), median of three, same hour, clocks pinned
(README "No-regression gates tighter than the floor"). Owner run, 2026-10-03 19:59 to 20:40 UTC
(`dist/packed-kquant-ab/run-gate.sh`).

Each invocation: `compare-llama-cpp.sh --gpu --pin-clocks --models <models> --n-prompt 512 --juno-warmup 2
--juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <jar>` (plus `--prefill-batch 16` or `64` in
part W), alternating A B three times per part. Every run `clock_pinned: true` (CPU governor performance,
turbo off, GPU graphics clock locked at 1911 MHz); every prefill 512 of 512 prompt tokens; 64 generated
tokens each. `ab-readings.json` holds every reading, with each request's `jdk.GCPhasePause` count and
maximum and `jdk.ThreadAllocationStatistics` bytes.

Builds: A = HEAD `f02bdae` from `git archive` (jar `c9f8e7a74c4187b7`); B = the tree with the packed
prefill matmul routed (jar `5393faf01b594879`).

## Part P: default window, n_prompt 512

| Model | Prefill A (min / max) | Prefill B (min / max) | B/A | Threshold | Generation A (min / max) | Generation B (min / max) | B/A | Threshold |
|---|---|---|---|---|---|---|---|---|
| TinyLlama | 950.59 (934.92 / 998.79) | 1165.28 (1143.74 / 1166.43) | **1.226** | >= 1.10 met | 64.41 (63.52 / 67.19) | 62.82 (61.89 / 63.66) | **0.975** | >= 0.95 met |
| Qwen2.5-3B | 418.69 (396.86 / 442.74) | 591.48 (590.73 / 625.23) | 1.413 | recorded | 30.14 (28.14 / 30.61) | 28.49 (27.60 / 28.57) | **0.945** | >= 0.95 **missed** |
| Phi-3.5-mini | 172.72 (164.28 / 175.38) | 236.24 (234.58 / 237.79) | 1.368 | recorded | 32.24 (30.38 / 32.27) | 30.66 (29.62 / 30.97) | **0.951** | >= 0.95 met |
| Mistral 7B | 189.20 (186.67 / 200.43) | 294.00 (293.00 / 302.02) | **1.554** | >= 1.30 met | 21.05 (21.00 / 22.67) | 21.41 (20.97 / 21.41) | **1.017** | >= 0.95 met |

## Part W: fixed window widths, n_prompt 512 (prefill B/A >= 1.10)

| Model | Width | Prefill A (min / max) | Prefill B (min / max) | B/A |
|---|---|---|---|---|
| TinyLlama | 16 | 192.00 (190.95 / 192.22) | 606.17 (594.53 / 627.62) | **3.157** met |
| Mistral 7B | 16 | 30.61 (29.43 / 31.76) | 157.35 (157.17 / 164.82) | **5.140** met |
| TinyLlama | 64 | 557.80 (486.79 / 566.80) | 1017.30 (838.34 / 1029.07) | **1.824** met |
| Mistral 7B | 64 | 101.69 (95.65 / 104.76) | 256.92 (239.34 / 259.88) | **2.527** met |

## How to read it

**Every prefill threshold is met, most by a wide margin; the generation gate is missed on Qwen2.5-3B,
0.945 against 0.95, and Phi-3.5-mini sits at 0.951.** As scored, the gate is missed.

What is known about the generation figure:
- The change does not reach decode. A generated token is one activation row, which never takes the
  batched K-quant path (wider than 8 rows); the decode kernels and their PTX module are unchanged.
- The JVM is not the difference. Per request, GC pauses peak at 8 to 15 ms on both builds and allocation
  is equal to within 1% (`ab-readings.json`).
- The baseline swings more than the gap. Its own Qwen2.5-3B readings span 28.14 to 30.61 (8.4%); run 2 is
  low on every model, while all three candidate runs sit near that low level.
- An unpinned follow-up (`unpinned-decode-probe.txt`, four alternations, TinyLlama and Qwen2.5-3B,
  started about 21:15 UTC) read generation B/A at 0.986 and 0.994 at 512 prompt tokens, and 1.00 at an 8-token prompt.
  It is unpinned, so it does not score the gate. It does argue against a reproducible 5% decode cost.

The plan's rule is that a gate is scored on the pinned A/B and never overruled by an unpinned run, so the
owner chose a confirmation run with its rule fixed before it started (below).

## Confirmation run: part P, five alternations (owner run, 21:48 to 22:21 UTC)

Rule, declared before the run: same jars and invocation, A B alternated five times, scored on the median of
five; generation B/A >= 0.95 on every sweep model, prefill B/A >= 1.30 on mistral-7b and >= 1.10 on
tinyllama. If any model missed again, the change would be treated as a decode regression and investigated.
Command: `PART=P N=5 RUN_LABEL=-confirm bash dist/packed-kquant-ab/run-gate.sh`. All ten runs pinned; every
prefill 512 of 512, 64 tokens generated. `ab-readings-confirm.json` holds every reading.

| Model | Prefill A | Prefill B | B/A | Generation A, five runs | Generation B, five runs | Gen. median A / B | B/A |
|---|---|---|---|---|---|---|---|
| TinyLlama | 915.11 | 1122.55 | **1.227** | 64.07, 59.18, 57.38, 62.43, 58.31 | 62.61, 55.44, 57.15, 62.20, 58.56 | 59.18 / 58.56 | **0.989** |
| Qwen2.5-3B | 374.58 | 559.75 | 1.494 | 29.27, 25.34, 25.51, 27.57, 25.67 | 27.20, 25.44, 24.22, 25.64, 25.68 | 25.67 / 25.64 | **0.999** |
| Phi-3.5-mini | 155.63 | 225.12 | 1.447 | 31.82, 27.80, 30.03, 27.92, 28.09 | 28.32, 27.84, 30.10, 28.43, 27.79 | 28.09 / 28.32 | **1.009** |
| Mistral 7B | 186.64 | 277.59 | **1.487** | 22.58, 19.10, 19.22, 20.51, 19.48 | 20.31, 19.52, 20.62, 19.78, 19.34 | 19.48 / 19.78 | **1.016** |

**Gate met under the declared rule**: generation 0.989x to 1.016x, prefill 1.227x (TinyLlama) and 1.487x
(Mistral 7B). GC pauses peak at 9.6 to 18.0 ms on both builds; allocation per request is equal within 1%.

How to read the two runs together. The host's absolute generation drifted down across the session on both
builds alike (Qwen2.5-3B 30.1 to 25.7 t/s from the first run to the confirmation), and within the
confirmation the first pair reads high on both sides; the alternation is what keeps the comparison fair.
Across both runs the generation ratio sat between 0.945 and 1.017 with no model consistently below 1.0,
which is what an unchanged decode path looks like at this host's run-to-run spread (up to about 15% across
runs). The first run's 0.945 is kept on record, not discarded.
