# Packed K-quant prefill closing gate: two fixes' no-regression A/B and LoRA

**Purpose.** Parts A and L of the packed K-quant prefill closing gate (`dist/packed-kquant-close/run-gate.sh`,
owner run). Part A scores whether two fixes made during the closing work cost decode or prefill
throughput: a barrier in the GPU attention kernel's block reduction (it removed run-to-run variance in
greedy output), and holding the decode matrix-vector scratch at weight upload (it repaired a failed
request when several processes share one GPU). Part L scores the LoRA train and playback gates against
the build before the packed K-quant work. Part B, the closing sweeps, is published separately
(`20261004T113210Z`, `20261004T114812Z`, `20261004T120622Z-partial`).

**Pinned: yes.** Part A through `compare-llama-cpp.sh --pin-clocks` (governor performance, turbo off, GPU
graphics clock locked at 1911 MHz; `clock_pinned` true on all six runs). Part L by hand (governor
performance, turbo off on every run; `part_l.pinning` in `ab-readings.json`). 2026-10-04 11:05Z to 11:31Z.

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| prefix (A) | HEAD `59c53cc`, before either fix | `425285a3daef839e` |
| candidate (A, L) | HEAD `59c53cc` plus both fixes (working tree) | `5ab4c78c505a4cef` |
| baseline (L) | HEAD `f02bdae`, before the packed K-quant work | `cdd4de12314667d2` |

**Commands.** Part A: prefix and candidate alternated three times, each `compare-llama-cpp.sh --gpu
--pin-clocks --n-prompt 512 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish`, all four
sweep models. Part L: each tree's own `compare-lora.sh --gpu --reps 1 --skip-build --no-publish`,
alternated three times. Gates read the median of three per side. All readings: `ab-readings.json`.

## Part A: the two fixes (candidate over prefix, median of three)

| Model | Prefill t/s (prefix / candidate) | Prefill B/A | Generation t/s (prefix / candidate) | Generation B/A |
|---|---|---|---|---|
| tinyllama-1.1b | 1043.17 / 1038.85 | 0.996 | 54.68 / 55.02 | 1.006 |
| qwen2.5-3b | 534.96 / 527.06 | 0.985 | 25.02 / 23.83 | **0.952** |
| Phi-3.5-mini | 214.19 / 215.97 | 1.008 | 26.25 / 26.22 | 0.999 |
| mistral-7b | 260.48 / 253.22 | 0.972 | 18.39 / 18.33 | 0.996 |

Gate: both ratios `>= 0.95` on every model. **Met.**

## Part L: LoRA (candidate over baseline, median of three)

| Reading | Baseline | Candidate | Ratio | Gate | Result |
|---|---|---|---|---|---|
| train ms per pass | 3400.0 | 3400.0 | speed 1.000 | `>= 0.95` | met |
| playback wall-clock tps | 10.72 | 11.04 | 1.030 | `>= 0.80` | met |

Every repetition trained 15 passes to loss 1.1797.

## How to read it

- **Qwen2.5-3B generation is the one tight reading.** The candidate is lower in each of the three pairs
  (23.83 against 25.02, 23.29 against 24.22, 24.69 against 25.10 t/s), so the 0.952 median is not one
  outlier. No mechanism is known. The barrier adds two block-wide synchronizations per attention block,
  and the scratch change removes work from decode rather than adding it. The same model produced the same
  pattern in the packed matmul's own A/B (first run 0.945, declared five-alternation confirmation 0.999,
  `20261003T195923Z-packed-kquant-ab`). It meets the gate as declared, and is recorded so a later A/B can
  watch it.
- Train time is coarse: the REPL reports the 15-pass total in whole seconds (50 or 51 s).
