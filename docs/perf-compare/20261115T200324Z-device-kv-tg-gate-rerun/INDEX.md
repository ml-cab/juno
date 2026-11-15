# Quantized device KV, decode rate gate, re-run: every sweep model passes

**Purpose.** Score the quantized device KV's decode bar again after the decode path was reworked
([`20261115T045828Z-device-kv-decode-path`](../20261115T045828Z-device-kv-decode-path/INDEX.md)): GPU decode tg at a
2,048-token context with `q8_0` device KV **>= 0.95x** the FP16 device copy, every sweep model, pinned, same hour. The first
run ([`20261113T215326Z-device-kv-tg-gate`](../20261113T215326Z-device-kv-tg-gate/INDEX.md)) read Qwen2.5-3B at 0.918x.

**Pinned:** CPU yes (governor performance, turbo off). GPU clock recorded, not fixed: `nvidia-smi -lgc 1300,1300` returned
success and the card did not apply it (loaded samples at 1,480 to 1,885 MHz, section 2), as the driver does on this
consumer Pascal card; the gate script now says so per JVM. The card was cooled to 65 C before every JVM. Within each model
the two sides ran at the same median loaded clock and the same memory clock (4,513 MHz, P2).

**Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB (shared with the desktop). **Build:** HEAD `79fa87b0` plus the
uncommitted quantized device KV work with the warp KV encoder and the wide decode block, run from the tree's test classes;
the tree hash the waiter checked (`head 79fa87b0...`, `diff e458312807298dc0`, `tree.txt`) still matched after the run.

**Method** (`dist/device-kv-tg/run-gate.sh`, owner run 2026-11-15T20:03Z): per model six JVMs, A B A B A B (A `f16`, B
`q8_0` keys and values), each running `DeviceKvDecodeTgLiveTest`: prefill 2,048 tokens of the quality text, then 128 timed
greedy steps through `ForwardPassHandler.forward`, one warm-up and three measured repetitions, the JVM's reading their
median. The model's ratio is the median of the three B readings over the median of the three A readings.

## 1. Result (`results.txt`)

| Model | FP16 tg, three JVMs (median) | `q8_0` tg, three JVMs (median) | Ratio | Bar 0.95x | First run |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 119.460, 125.249, 124.074 (124.074) | 153.177, 153.552, 151.753 (153.177) | **1.2346** | pass | 1.0853 |
| Qwen2.5-3B | 56.759, 56.799, 51.054 (56.759) | 58.394, 58.483, 58.554 (58.483) | **1.0304** | pass | 0.9184 |
| Phi-3.5-mini | 43.449, 43.094, 43.369 (43.369) | 45.746, 46.244, 45.810 (45.810) | **1.0563** | pass | 1.0035 |
| Mistral 7B | 35.035, 34.857, 35.011 (35.011) | 36.589, 36.576, 36.567 (36.576) | **1.0447** | pass | 0.9993 |

Per cycle (B over the A before it): TinyLlama 1.282, 1.226, 1.223; Qwen2.5-3B 1.029, 1.030, 1.147; Phi-3.5-mini 1.053,
1.073, 1.056; Mistral 7B 1.044, 1.049, 1.044. Every `q8_0` JVM is faster than every FP16 JVM on every model: the slowest
`q8_0` reading over the fastest FP16 reading is 1.212, 1.028, 1.053 and 1.044.

Capacity lines (free device memory after the weights, 2,048-token sessions whose device KV fits): `q8_0` holds 1.87x to
2.00x the FP16 sessions (TinyLlama 281 against 149, Qwen2.5-3B 133 to 135 against 71 to 72, Phi-3.5-mini 12 against 6,
Mistral 7B 22 against 11).

## 2. The clock (`gpu-clock-summary.txt`)

nvidia-smi sampled `clocks.sm`, `clocks.mem`, pstate, temperature, power and throttle reasons every 500 ms over each JVM.
Samples above 70 W (prefill and decode), per side:

| Model | FP16 JVMs, median | `q8_0` JVMs, median | Range, temperature | Throttle reasons (loaded samples) |
|---|---|---|---|---|
| TinyLlama 1.1B | 1,860 MHz | 1,860 MHz | 1,607 to 1,885 MHz, 45 to 68 C | none |
| Qwen2.5-3B | 1,835 MHz | 1,835 MHz | 1,733 to 1,847 MHz, 63 to 78 C | none |
| Phi-3.5-mini | 1,607 MHz | 1,607 MHz | 1,480 to 1,847 MHz, 65 to 94 C | software thermal slowdown (0x20) on about 80% |
| Mistral 7B | 1,809 MHz | 1,809 MHz | 1,607 to 1,847 MHz, 65 to 83 C | software thermal slowdown on 6% to 31% |

The memory clock read 4,513 MHz (P2) in every loaded sample. The CLOCK lines' `below_lock=0` counted only samples under
the requested 1,300 MHz and so read clean; the script now also counts samples above it and prints "lock NOT applied" when
most loaded samples are (a re-check over this run's samples prints it for every JVM).

## 3. Qwen2.5-3B's third FP16 JVM

A3 read 51.054 against 56.759 and 56.799, all three of its repetitions 50.1 to 51.1. Its prefill was slower by the same
share (2,671 ms against 2,420 and 2,426, `prefill-from-logs.txt`), at the same median clock (1,835 MHz), temperature (64
to 78 C) and power (137 W against 138 to 139 W) as A1 and A2, with no warning in its log. Something slowed that whole JVM;
the samples do not say what (the desktop shares the card and the host). The median rule leaves the result unchanged: the
two other FP16 JVMs give 1.028 to 1.031.

## 4. Prefill, not a bar (`prefill-from-logs.txt`)

The JVM logs' per-layer prefill records give each 2,048-token prefill's time. With `q8_0` device KV it takes **1.31x**
(TinyLlama), **1.21x** (Qwen2.5-3B) and **1.31x** (Mistral 7B) the FP16 time, and the first run's logs read the same
(1.32x, 1.24x, 1.29x), before the decode path was reworked: the cost comes from the quantized prefill path (the tiled
attention kernel reading `q8_0` K and V in 128-row windows), unchanged since item 7 landed. Phi-3.5-mini's handler writes
no per-layer prefill records. No threshold covers prefill with a quantized device KV.

## How to read it

The decode bar is met on every sweep model, at matched clocks per model, with every `q8_0` JVM faster than every FP16
JVM. Against the first run, Qwen2.5-3B moved from 0.918x to 1.030x and the other three models by 4% to 14%. Part of the
margin is the wider decode block, which the FP16 kernel does not use yet. A 2,048-token prefill with `q8_0` device KV takes
1.21x to 1.31x the FP16 time; that is outside the bar and recorded for the owner.

Files: `results.txt`, `pin.txt`, `tree.txt`, per model `<model>.txt` (every JVM's lines), `gpu-samples/<model>-<A|B><n>.gpu.csv`,
`gpu-clock-summary.txt`, `prefill-from-logs.txt` (both runs). The JVM logs stay in `dist/device-kv-tg/runs/20261115T200324Z/`.
