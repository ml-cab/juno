# Attention and long context: closing gate (owner run) and a decode diagnostic

**Purpose.** The closing no-regression gates for the tiled GPU attention kernel, the decode residency region
and its `auto` default, scored against the last build before that work. One gate row misses (Qwen2.5-3B GPU
generation); the diagnostic below attributes the miss to the attention kernel at decode width.

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| baseline (A) | HEAD `05a17de`: full-materialization GPU attention, decode region through RoPE only, `--gpu-residency` default `off`; from `git archive`, package only | `d1f50669dfc23d55` |
| candidate (B) | HEAD `51fa5d4` plus test-only changes: tiled online-softmax attention, whole decode layer in the region, `--gpu-residency` default `auto` | `fefd30371caf535a` |

Host: GTX 1080 (8 GiB), Xeon E5-1650 v2; desktop session sharing the card. Reference-tool build `ac4cdde`.
2026-10-08 05:44Z onward. The closing sweeps of the same run are published separately (see the end).

**Commands.** Each invocation `compare-llama-cpp.sh --juno-jar <jar> --juno-reps 1 --juno-warmup 2 --reps 1
--no-tuned-lane --no-publish`, A B A B A B, medians of three per side:
- Part A, `--gpu --pin-clocks --n-prompt 512`, four sweep models, default flags (`gpu-ab-*`);
- Part C, `--cpu --pin-clocks --n-prompt 128`, TinyLlama and Mistral 7B (`cpu-ab-*`);
- Part S, `--gpu --device-spans --n-prompt 2048`, unpinned by the milestone's own rule, Phi-3.5-mini in its own
  invocation at `COMPARE_HEAP=8g` (`attn-*`, `attn-phi-*`);
- Diagnostic, unpinned, Qwen2.5-3B, `--gpu --n-prompt 512 --gpu-attention on|off`, both jars, alternated three
  times (`diag-*`).

**Pinned:** parts A and C yes (`clock_pinned: true` on all twelve runs); part S and the diagnostic no.

## Part A: GPU throughput, allocation and GC (gate: t/s B/A >= 0.95; alloc/token <= 1.10x; GC in span <= 1.25x, or <= 5 ms under a 5 ms baseline)

| Model | pp A | pp B | pp B/A | tg A | tg B | tg B/A | alloc/token A | B | B/A | GC ms A | B | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | 1274.44 | 3007.97 | 2.360 | 69.79 | 130.85 | 1.875 | 48.5M | 39.0M | 0.803 | 6.7 | 0.0 | met |
| qwen2.5-3b | 641.37 | 1220.39 | 1.903 | 30.38 | 28.81 | **0.948** | 143.5M | 143.3M | 0.998 | 0.0 | 0.0 | **missed (tg)** |
| Phi-3.5-mini | 246.91 | 688.97 | 2.790 | 32.39 | 47.64 | 1.471 | 210.0M | 174.3M | 0.830 | 10.7 | 11.9 | met |
| mistral-7b | 302.22 | 523.89 | 1.733 | 22.69 | 35.09 | 1.546 | 224.2M | 193.6M | 0.863 | 13.6 | 0.0 | met |

Qwen2.5-3B tg per run: A 29.71 / 30.38 / 30.59, B 29.06 / 28.64 / 28.81: lower in all three pairs.

## Part C: standing CPU and allocation gate (same thresholds)

| Model | pp A | pp B | pp B/A | tg A | tg B | tg B/A | alloc/token B/A | GC ms A | B | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | 6.27 | 6.25 | 0.997 | 3.36 | 3.37 | 1.000 | 1.000 | 7.5 | 7.4 | met |
| mistral-7b | 0.88 | 0.88 | 1.001 | 0.53 | 0.53 | 1.000 | 1.000 | 0.0 | 0.0 | met |

Top 10 `jdk.ExecutionSample` methods are the same on both jars and both models (the Q6_K and Q4_K CPU kernels,
the weight-stationary Q4_K GEMM, `GgufReader.f16ToF32`, `GqaMath.attend`, then common-pool frames), with
sample counts within a few percent; see each run's `*-juno.json` `.jfr.top_methods`.

## Part S: attention at 2048 tokens, clock-normalised (gate: A over B >= 6.0x on every sweep model)

Attention per window = `region_attention_compute + attention_compute + attention_copies + attention_host` from
`prefill-breakdown.sh`, times the prefill repetition's in-window median SM MHz; median of three per side.

| Model | A ms (median run) | B ms (median run) | A over B, normalised | Result |
|---|---|---|---|---|
| tinyllama-1.1b | 4541.6 @ 1607 MHz | 323.0 @ 1607 MHz | **14.06x** | met |
| qwen2.5-3b | 6512.3 @ 1828 MHz | 712.6 @ 1607 MHz | **10.40x** | met |
| Phi-3.5-mini | 19661.4 @ 1607 MHz | 1148.5 @ 1607 MHz | **17.12x** | met |
| mistral-7b | 18928.1 @ 1506 MHz | 1260.0 @ 1607 MHz | **14.08x** | met |

Every run's reading: `attn-readings.txt`.

## Diagnostic: where Qwen2.5-3B's decode time went (unpinned)

| Qwen2.5-3B, n_prompt 512 | A decode ms/token (3 runs) | B decode ms/token (3 runs) | B/A (medians) |
|---|---|---|---|
| GPU attention on (default) | 32.21 / 30.66 / 31.07 | 32.03 / 32.43 / 32.29 | 1.039 (tg 0.965x) |
| GPU attention off | 33.54 / 33.59 / 33.38 | 33.91 / 33.44 / 33.24 | 0.997 |

With GPU attention off the two builds decode at the same speed, so the miss sits in the attention kernel. With it
on, B spends about 1.2 ms more per token (about 33 us per layer over 36 layers), and still less than with GPU
attention off. Qwen2.5-3B is the only sweep model whose decode the residency region declines, so it is the only one
where the attention kernel's decode cost is not offset by the region's gains.

## How to read it

- Both kernels launch one 128-thread block per (row, query head), so at decode width both run `numHeads` blocks
  (16 on Qwen2.5-3B, on a 20-SM card): the block count does not differ. What differs is the work inside a block.
  The old kernel scores every key in parallel into a global scores buffer, then runs the softmax and the V sum;
  the tiled kernel stages K and V tiles in shared memory and merges online-softmax partials across 32 slots per
  tile, which buys its flat scratch and its long-context speed. Which of those per-block costs makes up the extra
  33 us per layer at a short decode context is not established: no kernel-level profile was taken, and the other
  models' attention time at decode was not isolated (the region hides it in their end-to-end reading).
- Part S and the prefill gains are large; the miss is a decode-width cost of the same kernel.
- Closing sweeps from the same run: `20261008T083312Z` (128), `20261008T084614Z` (512), `20261008T090011Z` (2048,
  three models), `20261008T091353Z` (2048, Phi-3.5-mini at 8 GiB). All pinned, every row scorable, prompt parity
  exact.
- Local paths are replaced by `<repo>/`.

## Attention share of a 512-token window (candidate, unpinned, `--device-spans`, default lane, median of three)

`spans512/`: `compare-llama-cpp.sh --gpu --device-spans --n-prompt 512 --no-tuned-lane --juno-jar <candidate>`,
then `prefill-breakdown.sh`. Read against the pre-tier decomposition (`20261005T003146Z`, same terms).

| Model | Window ms | Attention ms | Attention share | Pre-tier attention ms / share | Residue |
|---|---|---|---|---|---|
| tinyllama-1.1b | 174 | 23 | 13.0% | 248 / 62.1% | 6.2% (flagged) |
| qwen2.5-3b | 410 | 48 | 11.7% | 408 / 53.3% | 3.7% |
| Phi-3.5-mini | 733 | 74 | 10.1% | 943 / 45.5% | 12.0% (flagged) |
| mistral-7b | 957 | 87 | 9.1% | 757 / 47.6% | 3.3% |

`prefill-breakdown.sh` exits 1 because two models' unattributed residue is above its 5% limit; residue is window time
no span covers, so it does not move the attention terms, but those two windows are not fully accounted for.
