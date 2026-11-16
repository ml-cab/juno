# Quantized device KV, prefill time gate: every sweep model passes at both widths

**Purpose.** Score the quantized device KV's prefill bar (owner decisions 6 (a) and 8 (a)): a 2,048-token prefill with
`q8_0` device KV takes **<= 1.10x** the FP16 device copy's time, in one 2,048-row window and in 128-row windows, every sweep
model, pinned, same hour. It follows the host KV taking the device copy's stored rows
([`20261116T032708Z-device-kv-prefill-encoded-rows`](../20261116T032708Z-device-kv-prefill-encoded-rows/INDEX.md), unpinned
0.895x to 0.989x) and the attention's quantized tile staging
([`20261116T004641Z-device-kv-prefill-staging`](../20261116T004641Z-device-kv-prefill-staging/INDEX.md)); before both, the
same prefill read 1.21x to 1.31x (2.04x on Phi-3.5-mini).

**Pinned:** CPU yes (governor performance, turbo off). GPU clock recorded, not fixed: `nvidia-smi -lgc 1300,1300` returned
success and the card did not apply it (every JVM's CLOCK line reads "lock NOT applied"; loaded samples at 1,607 to 1,873
MHz, section 2), as the driver does on this consumer Pascal card. The card was cooled to 65 C before every JVM. Within each
model the two sides ran within 25 MHz of each other's median loaded clock, the `q8_0` side the lower and warmer of the two,
and every loaded sample read the same memory clock (4,513 MHz, P2).

**Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB (shared with the desktop). **Build:** HEAD `a4a91182` plus the uncommitted
quantized device KV prefill work (the tile staging and the stored-row download), run from the tree's test classes; the tree
hash the waiter checked before the gate (`head a4a91182...`, `diff 023fd7eb880c6512`, `tree.txt`) still matched after it.

**Method** (`dist/device-kv-prefill/run-gate.sh`, owner run 2026-11-16T16:38Z): per model six JVMs, A B A B A B (A `f16`, B
`q8_0` keys and values), each running `DeviceKvPrefillTimeLiveTest`: a new session prefills the model's 2,048-token quality
text through `forwardBatch` (the serving prefill path), once in one window and once in 128-row windows, one warm-up and five
measured repetitions per width; the JVM's reading per width is their median. The model's ratio per width is the median of
the three B readings over the median of the three A readings. All 24 JVMs passed their test (one model each, the other
three skipped) and logged no Juno warning.

## 1. Result (`results.txt`)

| Model | Width | FP16, three JVMs (median) | `q8_0`, three JVMs (median) | Ratio | Bar 1.10x | Unpinned, same tree |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | one window | 796.6, 791.7, 850.4 (796.6 ms) | 763.2, 765.4, 771.0 (765.4 ms) | **0.9608** | pass | 0.938 |
| TinyLlama 1.1B | 128-row windows | 1,093.6, 1,065.6, 1,080.8 (1,080.8) | 1,027.0, 1,046.1, 1,038.5 (1,038.5) | **0.9609** | pass | 0.926 |
| Qwen2.5-3B | one window | 2,005.9, 2,074.9, 2,169.5 (2,074.9) | 1,927.0, 1,968.0, 1,959.3 (1,959.3) | **0.9443** | pass | 0.989 |
| Qwen2.5-3B | 128-row windows | 2,710.3, 2,707.2, 2,798.2 (2,710.3) | 2,425.9, 2,472.3, 2,472.9 (2,472.3) | **0.9122** | pass | 0.971 |
| Phi-3.5-mini | one window | 3,200.5, 3,095.3, 3,094.8 (3,095.3) | 2,778.2, 2,715.0, 2,709.2 (2,715.0) | **0.8772** | pass | 0.895 |
| Phi-3.5-mini | 128-row windows | 3,841.1, 3,716.8, 3,736.4 (3,736.4) | 3,381.1, 3,276.6, 3,272.7 (3,276.6) | **0.8769** | pass | 0.895 |
| Mistral 7B | one window | 4,510.1, 4,505.7, 4,503.1 (4,505.7) | 4,291.7, 4,272.9, 4,308.7 (4,291.7) | **0.9525** | pass | 0.938 |
| Mistral 7B | 128-row windows | 5,317.1, 5,318.6, 5,331.0 (5,318.6) | 5,012.4, 5,060.9, 5,060.3 (5,060.3) | **0.9515** | pass | 0.948 |

Per cycle (B over the A before it), one window / 128-row windows: TinyLlama 0.958, 0.967, 0.907 / 0.939, 0.982, 0.961;
Qwen2.5-3B 0.961, 0.948, 0.903 / 0.895, 0.913, 0.884; Phi-3.5-mini 0.868, 0.877, 0.875 / 0.880, 0.882, 0.876; Mistral 7B
0.952, 0.948, 0.957 / 0.943, 0.952, 0.949. Every `q8_0` JVM is faster than every FP16 JVM on every model at both widths:
the slowest `q8_0` reading over the fastest FP16 reading is at most 0.982 (TinyLlama, 128-row windows).

## 2. The clock (`gpu-clock-summary.txt`)

nvidia-smi sampled `clocks.sm`, `clocks.mem`, pstate, temperature, power and throttle reasons every 500 ms over each JVM.
Samples above 70 W (the prefills), per side:

| Model | FP16 JVMs, median | `q8_0` JVMs, median | Range, temperature | Software thermal slowdown (0x20), share of loaded samples |
|---|---|---|---|---|
| TinyLlama 1.1B | 1,835 MHz | 1,822 MHz | 1,607 to 1,873 MHz, 58 to 78 C | none (power cap 0x4 on a few, both sides) |
| Qwen2.5-3B | 1,809 MHz | 1,809 MHz | 1,607 to 1,835 MHz, 63 to 83 C | 1% both sides |
| Phi-3.5-mini | 1,822 MHz | 1,797 MHz | 1,607 to 1,835 MHz, 65 to 83 C | FP16 7%, `q8_0` 16% |
| Mistral 7B | 1,721 MHz | 1,708 MHz | 1,607 to 1,835 MHz, 70 to 84 C | FP16 40%, `q8_0` 46% |

Where the sides differ, the `q8_0` JVMs ran the slower and hotter clock, which can only have raised their times: the
ratios are not flattered by the clock.

## How to read it

The prefill bar is met on every sweep model at both widths, with a margin of 0.14 to 0.22 under it, at matched clocks per
model, and every `q8_0` JVM faster than every FP16 JVM. A quantized device KV now costs no time to first token on this
host: the quantized attention launch is faster than the FP16 one (0.90x to 0.95x), and the host copy takes the device's
stored rows (a quarter of the FP32 bytes, no CPU encode). Pinned ratios sit within 0.06 of the unpinned session's.

Files: `results.txt`, `pin.txt`, `tree.txt`, per model `<model>.txt` (every JVM's lines and CLOCK line),
`gpu-samples/<model>-<A|B><n>.gpu.csv`, `gpu-clock-summary.txt`. The JVM logs stay in
`dist/device-kv-prefill/runs/20261116T163811Z/`.
