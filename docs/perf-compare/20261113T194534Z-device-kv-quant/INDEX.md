# Quantized device KV: bytes, session capacity, greedy agreement and decode rate (unpinned)

**Purpose.** Read the quantized device KV (`--cache-type-k/-v` reaching the GPU's copy of each request's KV, stored as
`q8_0` / `q4_0` blocks and read by the attention kernels) against its bars: device bytes per position **<= 0.55x** the
FP16 copy at `q8_0` and **<= 0.30x** at `q4_0`; 2,048-token Mistral 7B sessions fitting beside the weights **>= 1.8x**
the FP16 figure at `q8_0`; greedy agreement with the FP16 copy **>= 99%** at `q8_0` and **>= 97%** at `q8_0` keys with
`q4_0` values, every sweep model; GPU decode tg at a 2,048-token context with `q8_0` **>= 0.95x** the FP16 copy
(pinned, same hour).

**Pinned:** no. The decode-rate bar is the owner's pinned run (`dist/device-kv-tg/`); section 4 is a preview. The other
readings are deterministic (greedy, fixed text) or arithmetic, and are not clock-sensitive.

**Host:** Xeon E5-1650 v2 (6 cores, 12 threads), 64 GiB, GTX 1080 8 GiB (shared with the desktop).

**Build.** One tree, both sides of every comparison: HEAD `79fa87b0` plus the uncommitted quantized device KV work
(`DeviceKvFormat`, `DeviceKvFormats`, `GqaAttentionKvKernel`, `kv_format.cuh`, `gqa_attention.cuh`,
`gqa_attention_kv.cu`, `kv_encode_rows`, `kv_append_rows_fmt`). Run from the tree's test classes, not a jar: the readings
are opt-in tests in `node`. After these readings the tree changed only in which head widths may take a quantized format
(a multiple of 32; every sweep model's heads are 64, 96 or 128 wide), the Javadoc, and the docs. Side A is
`--cache-type-k/-v f16` (the FP16 device copy, as before this change), side B the quantized type.

## 1. Bytes per position (bars: `q8_0` <= 0.55x, `q4_0` <= 0.30x)

A property of the formats, asserted in `DeviceKvFormatTest` (no GPU) on every sweep model's K/V width and read live
from the handler's allocation (`DeviceKvCache.allocatedBytes()` after one 128-token window, `device_bytes_window`
below, equal to layers x 128 x bytes per position on every model):

| Device KV | Bytes per 32 values, K + V | Against FP16 | Bar |
|---|---|---|---|
| FP16 (`f16`) | 64 + 64 | 1.000x | |
| `q8_0` K and V | 34 + 34 | **0.531x** | <= 0.55x, met |
| `q4_0` K and V | 18 + 18 | **0.281x** | <= 0.30x, met |
| `q8_0` K, `q4_0` V (the 4-bit setting held to the quality bar) | 34 + 18 | 0.406x | reported |

Live, TinyLlama (22 layers): 2,883,584 bytes FP16, 1,531,904 `q8_0`, 1,171,456 `q8_0`/`q4_0` for one 128-token window.

## 2. Session capacity, Mistral 7B, 2,048 tokens (bar: >= 1.8x the FP16 figure at `q8_0`)

`DeviceKvDecodeTgLiveTest` prints the device memory free once the handler has loaded (all 32 layers on the GPU, the
prefill window's reserve held) and the whole 2,048-token sessions whose device KV fits in it
(`decode-tg-and-capacity.txt`, `DEVICE-KV-CAPACITY` lines):

| Free after the weights | FP16 sessions (268,435,456 bytes each) | `q8_0` sessions (142,606,336 bytes each) | Ratio |
|---|---|---|---|
| 3,219,652,608 to 3,224,895,488 (three runs) | 11 to 12 | 22 | **1.83x to 2.00x** (bar 1.8x, met) |

The bytes ratio is 1.882x; whole sessions round it down to 1.83x where the free figure sits just above a twelfth FP16
session.

## 3. Greedy agreement with the FP16 device copy (bars: >= 99% `q8_0`, >= 97% `q8_0` K + `q4_0` V)

`DeviceKvQuantQualityLiveTest` (node, opt-in, GPU, one model and one cache type per JVM; the FP16 run saves its
reference): the reading of `20261113T051804Z-kv-q4-0` taken on the device. The FP16 handler continues the quality text's
first 1,024 tokens greedily for 512 tokens; that sequence is teacher-forced through both handlers in windows of 128
(prefill width), and the gate is the share of the 512 generated positions where their argmax agree. NLL over the
2,048-token text, and the free-running quantized reply's matching prefix (decode width), are reported beside it.

```bash
mvn -o test -pl node -am -Dgroups=gpu -Dtest=DeviceKvQuantQualityLiveTest -Dsurefire.failIfNoSpecifiedTests=false \
  -Djuno.test.deviceKvQuality=true -Djuno.test.deviceKvQuality.models=<tinyllama|qwen25|phi35|mistral> \
  -Djuno.test.deviceKvQuality.type=<f16 first, then q8_0, q8_0/q4_0> -Djuno.test.deviceKvQuality.dir=<dir>
```

| Model | `q8_0`: agreement, NLL ratio, free-running prefix | `q8_0` K, `q4_0` V: agreement, NLL ratio, free-running prefix |
|---|---|---|
| TinyLlama 1.1B | **99.41%**, 1.00047, 1/512 | **98.83%**, 1.00077, 20/512 |
| Qwen2.5-3B | **99.61%**, 1.00112, 1/512 | **98.63%**, 1.00398, 1/512 |
| Phi-3.5-mini | **99.61%**, 0.99979, 4/512 | **99.41%**, 1.00325, 28/512 |
| Mistral 7B | **100.00%**, 0.99914, 512/512 | **99.61%**, 0.99735, 3/512 |

Every bar met on every model. The FP16 run's own teacher-forced argmax equals the tokens it generated at decode width on
509, 509, 512 and 512 of 512 positions (TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B): the prefill and decode paths
of one FP16 handler already disagree on up to 3 positions, the floor this reading sits on. A free-running reply leaves
the FP16 one at its first differing token and then shares nothing, which is why the prefix column is short; it is the
reading the plan's definition replaced (Tier 03's perf-gate note, 2026-11-12). The table is the final kernel's. A run on
the first quantized kernel read the same values on TinyLlama, Qwen2.5-3B and Mistral 7B, and on Phi-3.5-mini 99.41%
(`q8_0`) and 99.22% (`q8_0`/`q4_0`), both above the bars; why only Phi-3.5-mini's teacher-forced reading moved with the
decode-path change was not traced.

## 4. Decode rate at a 2,048-token context, preview (bar: `q8_0` >= 0.95x FP16, pinned, the owner's run)

`DeviceKvDecodeTgLiveTest`: prefill 2,048 tokens of the quality text, then 128 timed greedy steps through
`ForwardPassHandler.forward` (the serving decode path); one warm-up and three measured repetitions per JVM, the JVM's
reading their median. Mistral 7B A B A B A B; the other models one A B pair. Unpinned, desktop sharing the GPU.

| Model | FP16 tg (A) | `q8_0` tg (B) | B / A |
|---|---|---|---|
| Mistral 7B | 35.579, 34.318, 33.879 (median 34.318) | 34.562, 33.709, 33.307 (median 33.709) | **0.982** |
| TinyLlama 1.1B | 115.996 | 130.256 | 1.123 |
| Qwen2.5-3B | 54.468 | 51.729 | 0.9497 |
| Phi-3.5-mini | 42.875 | 40.712 | 0.9496 |

Mistral 7B and TinyLlama clear the bar; Qwen2.5-3B and Phi-3.5-mini read 0.9497 and 0.9496 on one unpinned pair each,
0.03% under it. Within this session the FP16 side alone moved from 31.1 to 35.6 t/s on Mistral 7B between runs an hour
apart (the first preview, on the first kernel, read 31.0 to 31.3), so one pair does not score this; the pinned run with
three cycles per model does. Where the decode rate goes with a quantized type, beside attention (section 5): the host KV
is `q8_0` as well, so every decoded token encodes its K and V rows on the host per layer (6,144 values a layer on
Phi-3.5-mini) where `f16` copies floats, and each layer launches the encode kernel instead of the FP16 cast.

The owner's run: [`20261113T215326Z-device-kv-tg-gate`](../20261113T215326Z-device-kv-tg-gate/INDEX.md) (TinyLlama 1.085,
Qwen2.5-3B 0.918, Phi-3.5-mini 1.004, Mistral 7B 0.999; Qwen2.5-3B under the bar, its cost traced there).

## 5. The attention kernel alone (`decode-kernel-bench.txt`)

`GqaAttentionDecodeBench` (node, opt-in, device events), one decode row per launch: at 2,048 keys the quantized kernel
takes 0.58x to 1.02x the FP16 kernel's time (`q8_0`: TinyLlama 0.576x, Qwen2.5-3B 0.955x, Phi-3.5-mini 0.604x,
Mistral 7B 0.825x). The first version took 2.9x to 8.2x; three changes brought it here, each recorded in the file:
the format pair chosen once per launch (decode and staging loops compiled per pair, no format branch inside them), each
decode lane holding whole blocks of the head (one scale load per block, the quants as realigned 4-byte words, the scale
applied once per block), and every KV read through the read-only cache.

## How to read it

Sections 1 to 3 meet their bars on this tree and are what the exit criterion cites; section 4 is evidence, and its bar
is scored only by `dist/device-kv-tg/run-gate.sh`, pinned. The FP16 kernels are unchanged by this work: their PTX,
rebuilt from the shared header, differs from the committed one only in internal symbol names, so side A is the build
before the change.
