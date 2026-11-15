# Quantized device KV, decode rate gate: owner run, Qwen2.5-3B under the bar

**Purpose.** Score the quantized device KV's decode bar: GPU decode tg at a 2,048-token context with `q8_0` device KV
**>= 0.95x** the FP16 device copy, every sweep model, pinned, same hour. The other bars of the change (bytes, session
capacity, greedy agreement) are in [`20261113T194534Z-device-kv-quant`](../20261113T194534Z-device-kv-quant/INDEX.md).

**Pinned:** CPU yes (governor performance, turbo off). GPU: the graphics clock was locked at the card's maximum, 1,911
MHz, and not held: at 63 to 94 C the card ran below it. Within each model the two sides ran at the same loaded clock
(section 2), so the ratios compare like with like; the absolute rates are lower than a cool card's.

**Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB (shared with the desktop). **Build:** HEAD `79fa87b0` plus the
uncommitted quantized device KV work, run from the tree's test classes (`tree.txt`); the tree hash in the waiter's
`READY` matched (`head 79fa87b0...`, `diff b33e22ab988a070e`).

**Method** (`dist/device-kv-tg/run-gate.sh`, owner run 2026-11-13T21:53Z): per model six JVMs, A B A B A B (A `f16`,
B `q8_0` keys and values), each running `DeviceKvDecodeTgLiveTest`: prefill 2,048 tokens of the quality text, then 128
timed greedy steps through `ForwardPassHandler.forward`, one warm-up and three measured repetitions, the JVM's reading
their median. The model's ratio is the median of the three B readings over the median of the three A readings.

## 1. Result (`results.txt`)

| Model | FP16 tg, three JVMs (median) | `q8_0` tg, three JVMs (median) | Ratio | Bar 0.95x |
|---|---|---|---|---|
| TinyLlama 1.1B | 125.094, 125.225, 125.289 (125.225) | 135.910, 135.995, 135.183 (135.910) | 1.0853 | pass |
| Qwen2.5-3B | 57.559, 57.047, 54.399 (57.047) | 53.864, 52.394, 51.396 (52.394) | **0.9184** | **fail** |
| Phi-3.5-mini | 41.608, 39.751, 39.779 (39.779) | 39.920, 39.888, 39.987 (39.920) | 1.0035 | pass |
| Mistral 7B | 32.338, 33.225, 33.453 (33.225) | 33.173, 33.203, 33.402 (33.203) | 0.9993 | pass |

Per cycle on Qwen2.5-3B: 0.936, 0.918, 0.945; every `q8_0` JVM is slower than every FP16 JVM. Capacity lines (free
device memory after the weights, 2,048-token sessions that fit): `q8_0` holds 1.88x to 2.00x the FP16 sessions on every
model (TinyLlama 276-277 against 146-147, Qwen2.5-3B 134-135 against 71, Phi-3.5-mini 12 against 6, Mistral 7B 22
against 11).

## 2. The clock (`gpu-clock-summary.txt`)

nvidia-smi sampled `clocks.sm`, temperature and power every 500 ms over each whole JVM. Samples above 70 W (prefill and
decode; model loads draw about 50 W at a flat 1,607 MHz), median clock per side:

| Model | FP16 JVMs | `q8_0` JVMs | Range, temperature |
|---|---|---|---|
| TinyLlama 1.1B | 1,860 MHz | 1,847 MHz | 1,607 to 1,885 MHz, 48 to 72 C |
| Qwen2.5-3B | 1,809 MHz | 1,809 MHz | 1,607 to 1,847 MHz, 63 to 83 C |
| Phi-3.5-mini | 1,404 MHz | 1,417 MHz | 873 to 1,822 MHz, 79 to 94 C |
| Mistral 7B | 1,607 MHz | 1,607 MHz | 1,328 to 1,733 MHz, 84 to 94 C |

The lock did not hold (Phi-3.5-mini and Mistral 7B ran throttled at up to 94 C; earlier owner gates on this card held
about 1,810 to 1,845 MHz at 52 to 67 C), but each model's two sides match within 13 MHz. Qwen2.5-3B's shortfall is not
the clock.

## 3. Where Qwen2.5-3B's time goes (`qwen-decode-trace.txt`, unpinned, traced)

Qwen2.5-3B JVMs under `nsys` (CUDA trace), one warm-up and one measured repetition each, in both orders (FP16 first,
then `q8_0` first); both sides at the same loaded clock (median about 1,850 MHz, memory 4,513 MHz, P2). Per token, over
tokens 1 to 127 of the measured repetition:

| Per token | FP16 | `q8_0` | Difference |
|---|---|---|---|
| decode attention, 36 launches | 3,010 / 3,018 us | 3,259 / 3,268 us | +248 / +250 us (1.08x) |
| KV write, 72 launches | `to_half` 124 / 125 us | `kv_encode_rows` 341 / 340 us | +217 / +215 us (4.7 against 1.7 us a launch) |
| FFN gate and up GEMVs, 72 launches | | | +5.6% each, every layer: about +330 / +350 us |
| every other kernel | | | within about 1% |
| GPU idle (wall minus kernels) | 6,399 / 5,878 us | 6,109 / 5,715 us | -290 / -163 us |

- **No host-side cost.** The GPU waits on the host less with `q8_0`, not more: the host's per-layer work (each layer
  writes its K and V rows to the host KV, a `q8_0` encode of 2 x 256 values where FP16 copies floats) is not where the
  time goes.
- **Decode attention, 1.08x.** Qwen2.5-3B has 16 query heads over 2 KV heads at width 128: a grid of 4 x 16 blocks
  reading about 2 MB in about 83 us, bound by latency, not bandwidth, so the dequantizing adds to every step. In
  isolation, with the KV warm in L2, the same launch read 0.955x the FP16 time (`decode-kernel-bench.txt` of the run
  above).
- **KV write, 2.7x a launch.** The encoder runs one thread per 32-value block: 8 threads a launch on Qwen2.5-3B, two
  launches a layer.
- **The FFN gate and up GEMVs** (unchanged code, identical launches: 11,008 x 2,048, the same grid, block and registers)
  take 1.053x to 1.059x in all 36 layers, in both orders; the other GEMVs, the down projection included, do not move.
  Not the clock (matched), and not the placement of the weights: an FP16 JVM that loads the quantized attention module
  before its weight upload, as a `q8_0` JVM does, reads 0.994x to 0.997x on the same GEMVs (a scratch build, not
  published code). What in the quantized decode slows the next layer-wide weight reads is not yet found.
- Together about +0.85 ms of GPU time on a 17.3 ms token (the traced GPU busy time differs by +843 / +865 us), a ratio of
  about 0.95x before the pinned run's own conditions; the run read 0.918x.

## How to read it

Qwen2.5-3B misses the bar: every `q8_0` JVM slower than every FP16 JVM, at matched clocks. TinyLlama, Phi-3.5-mini and
Mistral 7B pass. The cost is on the GPU, in three places, two of them the quantized path's own kernels (the attention
launch at few KV heads, the KV write) and one an interaction with the FFN weight reads still to be explained. For a
re-run, `dist/device-kv-tg/run-gate.sh` (changed after this run) locks the graphics clock at 1,300 MHz, below what the
card held loaded at 94 C, lets the card cool to 65 C before every JVM, and prints a CLOCK line per JVM (loaded samples,
how many fell below the lock, the hottest sample, the throttle reasons). A lower core clock does not favour the
quantized side: the weights' memory-bound reads do not slow with it; the dequantizing and encoding it adds do.

Follow-up, the decode path reworked after this run: [`20261115T045828Z-device-kv-decode-path`](../20261115T045828Z-device-kv-decode-path/INDEX.md).
The re-run: [`20261115T200324Z-device-kv-tg-gate-rerun`](../20261115T200324Z-device-kv-tg-gate-rerun/INDEX.md), every sweep model passes.

*Correction 2026-11-15: the re-run showed `nvidia-smi -lgc` is accepted and not applied on this card (a 1,300 MHz request read 1,480 to 1,885 MHz loaded), so the 1,911 MHz lock here most likely never applied either: the GPU clock in this run was recorded, not fixed, as the plan README says of consumer Pascal parts. Section 2's matched clocks per model stand.*

Files: `results.txt`, `pin.txt`, `tree.txt`, per model `<model>.txt` (every JVM's lines), `gpu-samples/<model>-<A|B><n>.gpu.csv`,
`gpu-clock-summary.txt`, `qwen-decode-trace.txt`. The JVM logs stay in `dist/device-kv-tg/runs/20261113T215326Z/`.
