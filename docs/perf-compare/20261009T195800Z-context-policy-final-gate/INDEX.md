# Context policy: final gate - GPU no-regression on the final jar, and the shift-step latency (owner run)

**Purpose.** Close the context-policy work (context shifting and sliding-window attention): the GPU no-regression A/B
re-run on the final jar (the first gate, `20261009T151747Z-context-policy-close-gate`, measured an earlier jar; the
change since then shifts the device KV copy in place and runs only when a request shifts), and the bound on the
decode step that performs a context shift. The reference-relative sweep of the same run is `20261009T195529Z`.

**Pinned: yes** (part A: `clock_pinned: true` on all six runs; part D: CPU governor performance, turbo off; the GPU
clock is recorded, not fixed, on this card).

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| baseline (A) | HEAD `de98ff6`, the last build before context shifting and sliding windows | `620a5caf606538b6` |
| candidate (B) | HEAD `0f87603` plus sliding windows and the in-place device KV shift | `16f8794d833b1dff` |

**Command.** `dist/context-policy-final/run-gate.sh` parts A, D and L, started by `wait-and-run.sh` (2026-10-09, 14:24
to about 15:20 local). Part A: as in the first gate (`compare-llama-cpp.sh --gpu --pin-clocks --n-prompt 512`, A B A B
A B, four sweep models). Part D: `context-shift-step-bench.sh --jar <candidate> --pin-clocks --reps 3 --heap 16g`.

## Part A: GPU no-regression (gate: tg and pp B/A >= 0.95; allocation per token <= 1.10x; GC pause in span <= 1.25x or <= 5 ms under 5 ms; greedy output equal)

| Model | pp A | pp B | pp B/A | tg A | tg B | tg B/A | alloc/token B/A | GC ms A / B | Greedy equal | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | 3017.24 | 2997.75 | 0.994 | 140.97 | 145.29 | 1.031 | 1.000 | 0.0 / 0.0 | 3 of 3 | met |
| qwen2.5-3b | 1246.47 | 1246.05 | 1.000 | 29.85 | 29.61 | 0.992 | 0.994 | 0.0 / 0.0 | 3 of 3 | met |
| Phi-3.5-mini | 700.92 | 699.52 | 0.998 | 50.88 | 50.73 | 0.997 | 1.004 | 13.7 / 13.5 | 3 of 3 | met |
| mistral-7b | 557.22 | 551.90 | 0.990 | 37.66 | 37.66 | 1.000 | 1.001 | 0.0 / 0.0 | 3 of 3 | met |

## Part D: shift step at 32,768 positions (gate: median of three of shift step / median decode step <= 3.0)

Per repetition: five decode steps ending at position 32,767 (median taken), then the shift the generation loop
performs (keep 32, discard 16,368) plus the next decode step, timed together. The host KV is filled directly (what a
step costs does not depend on what the cache holds).

| Model | Backend | Median decode step (ms, per rep) | Shift (ms) | Shift step (ms) | Ratio per rep | Median | Result |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | CPU | 5,482 / 5,517 / 5,464 | 87 / 80 / 78 | 2,966 / 2,990 / 2,947 | 0.541 / 0.542 / 0.539 | **0.541** | met |
| tinyllama-1.1b | GPU | 50.6 / 48.9 / 43.8 | 92 / 89 / 89 | 119 / 116 / 115 | 2.362 / 2.361 / 2.620 | **2.362** | met |
| mistral-7b | CPU | 17,319 / 17,368 / 17,323 | 471 / 467 / 473 | 10,155 / 10,070 / 10,066 | 0.586 / 0.580 / 0.581 | **0.581** | met |
| mistral-7b | GPU | 3,927 / 3,924 / 3,916 | 500 / 490 / 495 | 2,476 / 2,472 / 2,466 | 0.631 / 0.630 / 0.630 | **0.630** | met |

## How to read it

- Every row of both parts is met.
- The step after a shift attends over half the context, which pays for most of the shift on the CPU and on Mistral 7B's
  GPU lane (ratios below 1). TinyLlama's GPU lane is the tight one: its decode step is about 50 ms and the shift itself
  is about 90 ms, nearly all of it the host shift moving about 1.5 GB of float rows; the device shift is a few ms.
- Mistral 7B's GPU lane decodes at about 3.9 s at 32,768 positions because its device KV copy does not fit next to the
  weights on this 8 GiB card, so attention runs on the host; that was so before this work.
- Before the in-place device shift (unpinned, one repetition): TinyLlama GPU 16.3x, Mistral 7B GPU shift 2.9 s.
- Local paths are replaced by `<repo>/`.
