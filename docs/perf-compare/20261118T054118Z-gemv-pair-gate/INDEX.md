# FFN gate and up projections in one launch, pinned gate: decode after over before >= 1.00, FP16 and `q8_0` device KV, every sweep model

**Purpose.** The owner's pinned gate for decision 4 (a) (2026-11-15): the decode region's FFN gate and up GEMVs run as
one launch (`q4k_gemv_pair`), reading the Q8_1 input once for both matrices
([`20261117T082501Z-gemv-gate-up-pair`](../20261117T082501Z-gemv-gate-up-pair/INDEX.md): kernel bench, unpinned decode
A/B, hot-path gate). Two bars per sweep model, set before implementation: GPU decode tg at a 2,048-token context, this
build over the build before (`9c33ab93`), **>= 1.00** with FP16 device KV and **>= 1.00** with `q8_0` device KV.

**Pinned:** CPU yes (governor performance, turbo off). GPU clock recorded, not fixed: the 1,300 MHz lock was accepted and
not applied (every CLOCK line reads "lock NOT applied"), as on every earlier gate on this card. The card was cooled to
65 C before every JVM. Per model the before and after sides ran at the same median loaded clock, except TinyLlama's FP16
after side, 13 MHz (0.7%) above its before side, and Qwen2.5-3B's two after sides, 13 MHz below theirs (section 2).

**Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB (shared with the desktop). **Builds:** after (B, D): HEAD `9c33ab93` plus
the uncommitted change (`tree.txt`: `head 9c33ab93...`, `diff 2213b73c2a755981`), the hash `READY` recorded and
`wait-and-run.sh` checked before the run; before (A, C): `git archive 9c33ab93`, built in its own directory. Files changed
after the gate are this record's documents only: no file outside `docs/` is newer than `READY`.

**Method** (`dist/gemv-pair/run-gate.sh`, owner run 2026-11-18T05:41Z to 07:26Z): per model twelve JVMs, A B C D three
times (A `f16` before, B `f16` after, C `q8_0` before, D `q8_0` after), each running `DeviceKvDecodeTgLiveTest` from its
own tree: prefill 2,048 tokens, then 128 timed greedy steps through the serving decode path, one warm-up and three
measured repetitions, the JVM's reading their median. A bar's ratio is the median of the three after readings over the
median of the three before readings.

## 1. Result (`results.txt`)

| Model | `f16` before (A), tok/s | `f16` after (B) | **B over A** | `q8_0` before (C) | `q8_0` after (D) | **D over C** |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 154.740 | 166.349 | **1.0750** pass | 154.124 | 158.229 | **1.0266** pass |
| Qwen2.5-3B | 62.759 | 63.786 | **1.0164** pass | 58.099 | 60.685 | **1.0445** pass |
| Phi-3.5-mini | 46.483 | 47.846 | **1.0293** pass | 45.397 | 46.091 | **1.0153** pass |
| Mistral 7B | 37.467 | 37.975 | **1.0136** pass | 36.470 | 37.156 | **1.0188** pass |

8 of 8 pass (bar >= 1.00). JVM readings (`<model>.txt`), cycles 1 2 3:
- TinyLlama: A 151.900 160.108 154.740; B 165.124 167.552 166.349; C 155.697 154.124 148.766; D 165.145 158.229 151.484
- Qwen2.5-3B: A 63.073 62.759 59.921; B 63.786 62.907 65.522; C 59.156 58.099 56.787; D 64.130 60.685 59.672
- Phi-3.5-mini: A 47.055 46.442 46.483; B 47.949 47.846 47.764; C 45.355 45.769 45.397; D 46.091 46.354 45.795
- Mistral 7B: A 37.352 37.528 37.467; B 37.947 37.975 38.045; C 36.424 36.470 36.528; D 37.124 37.172 37.156

Per cycle, after over the before JVM of the same cycle: `f16` TinyLlama 1.087, 1.047, 1.075; Qwen2.5-3B 1.011, 1.002, 1.094;
Phi-3.5-mini 1.019, 1.030, 1.028; Mistral 7B 1.016, 1.012, 1.015. `q8_0` TinyLlama 1.061, 1.027, 1.018; Qwen2.5-3B 1.084,
1.045, 1.051; Phi-3.5-mini 1.016, 1.013, 1.009; Mistral 7B 1.019, 1.019, 1.017. No cycle reads under 1.00. On Phi-3.5-mini and
Mistral 7B (both types), TinyLlama `f16` and Qwen2.5-3B `q8_0`, every after JVM is faster than every before JVM; the slowest
after over the fastest before reads 0.973 on TinyLlama `q8_0` (D3 against C1) and 0.997 on Qwen2.5-3B `f16` (B2 against A1).

Scorable: a side's three JVM readings agree within 8.6% of their median (largest, TinyLlama D; Phi-3.5-mini and Mistral 7B
within 1.3%), under the 15% dispersion rule. One JVM's own three repetitions spread 15.3% (TinyLlama C2, a before JVM:
136.18, 154.12, 159.80 tok/s); its reading is the C side's median, and against the fastest C JVM instead, D over C reads
1.016.

Mistral 7B, the expected thin margin (1.008x and 1.005x unpinned, sides overlapping): pinned 1.014 and 1.019, its readings
within 0.5% per side. Against the executor's unpinned A/B: `f16` 1.060, 1.045, 1.028, 1.008 then, 1.075, 1.016, 1.029, 1.014
now; `q8_0` 1.077, 1.041, 1.011, 1.005 then, 1.027, 1.045, 1.015, 1.019 now (TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B).

The before side reproduces the 2026-11-17 gate, which measured the same bytes
([`20261117T053104Z-quant-decode-loads-gate`](../20261117T053104Z-quant-decode-loads-gate/INDEX.md)), within 0.6% on
Qwen2.5-3B `f16`, Phi-3.5-mini and Mistral 7B; TinyLlama read 3.7% (`f16`) and 2.6% (`q8_0`) lower and Qwen2.5-3B `q8_0`
3.2% lower (A 154.740 against 160.645, C 154.124 against 158.281, C 58.099 against 60.049).

## 2. The clock (`gpu-clock-summary.txt`)

Samples above 70 W (prefill and decode), all three JVMs per side:

| Model | A / B / C / D median | Range, temperature | Throttle reasons (loaded samples) |
|---|---|---|---|
| TinyLlama 1.1B | 1,847 / 1,860 / 1,847 / 1,847 MHz | 1,771 to 1,885 MHz, 51 to 75 C | none |
| Qwen2.5-3B | 1,835 / 1,822 / 1,835 / 1,822 MHz | 1,607 to 1,860 MHz, 64 to 79 C | none |
| Phi-3.5-mini | 1,607 on every side | 1,468 to 1,847 MHz, 65 to 94 C | software thermal slowdown (0x20) on 76% to 78% |
| Mistral 7B | 1,809 on every side | 1,607 to 1,847 MHz, 65 to 84 C | software thermal slowdown on 16% to 17%; software power cap (0x4) on 2 samples of A and C |

## 3. `q8_0` over FP16 (exit criterion 7's ratio), read, not scored

This gate scores each type against itself; criterion 7's bar (`q8_0` decode >= 0.95x FP16, pinned) was scored on
`9c33ab93` by the 2026-11-17 gate (0.958 to 0.985). Read from this run's after side, D over B is 0.951 (TinyLlama), 0.951
(Qwen2.5-3B), 0.963 (Phi-3.5-mini) and 0.978 (Mistral 7B). The same reading on the before side, the code that gate
measured, is 0.996, **0.926**, 0.977 and 0.973 against its 0.985, 0.958, 0.974 and 0.972: in this run's order (the `f16`
and `q8_0` JVMs two slots apart), TinyLlama's and Qwen2.5-3B's `q8_0` JVMs slowed across the three cycles on both sides
(TinyLlama C 155.70, 154.12, 148.77 and D 165.15, 158.23, 151.48; Qwen2.5-3B C 59.16, 58.10, 56.79 and D 64.13, 60.69, 59.67)
while their `f16` JVMs did not, so a cross-type ratio from this run moves by up to 3.4% on unchanged code.

The change takes the same work out of both types' decode step: the paired launch computes the FFN gate and up rows and
reads no KV. With `t` the time per token at the 2,048-token context and `s` the time saved, `q8_0` over FP16 goes from
`t_f16 / t_q8` to `(t_f16 - s) / (t_q8 - s)`. From the 2026-11-17 gate's rates and the largest saving measured per model
(any side, pinned or unpinned: 0.48, 0.73, 0.61 and 0.51 ms per token), the ratio can fall at most to 0.984 (TinyLlama),
0.956 (Qwen2.5-3B), 0.973 (Phi-3.5-mini) and 0.971 (Mistral 7B). Both readings are at or above 0.95 on every model;
criterion 7 stays met.

## 4. Prefill, not a bar (`prefill-from-logs.txt`)

From the JVM logs' per-layer prefill records, a 2,048-token prefill's time (the warm-up request excluded), median of the
three JVMs per side; the paired launch does not run in prefill:

| Model | A `f16` before | B `f16` after | C `q8_0` before | D `q8_0` after | B over A | D over C |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 1,019.8 ms | 997.4 ms | 970.9 ms | 951.8 ms | 0.978 | 0.980 |
| Qwen2.5-3B | 2,432.2 ms | 2,482.5 ms | 2,332.7 ms | 2,297.8 ms | 1.021 | 0.985 |
| Mistral 7B | 4,944.3 ms | 4,940.7 ms | 4,693.3 ms | 4,647.3 ms | 0.999 | 0.990 |

Within 2.1% either way. Phi-3.5-mini's handler writes no per-layer prefill records.

## How to read it

Decision 4 (a) passes its decode gate: at a 2,048-token context, pinned, GPU decode with the FFN gate and up projections in
one launch reads 1.014x (Mistral 7B) to 1.075x (TinyLlama) the build before with FP16 device KV and 1.015x (Phi-3.5-mini)
to 1.045x (Qwen2.5-3B) with `q8_0`, on every sweep model and in every cycle. The gain follows the kernel: largest at
width 2,048, where the paired launch also puts the idle warps to work, smallest at widths 3,072 and 4,096, where only the
input reads halve. Outputs are bit for bit the two launches', so no quality reading moves.

Files: `results.txt`, `pin.txt`, `tree.txt`, per model `<model>.txt` (every JVM's DEVICE-KV-TG and CLOCK lines),
`gpu-samples/<model>-<A|B|C|D><n>.gpu.csv`, `gpu-clock-summary.txt`, `prefill-from-logs.txt`. The JVM logs stay in
`dist/gemv-pair/runs/20261118T054118Z/`.
