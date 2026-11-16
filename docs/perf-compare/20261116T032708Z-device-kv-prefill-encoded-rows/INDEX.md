# Quantized device KV: the host KV takes the device copy's encoded rows; prefill time after

**Purpose.** The previous entry ([`20261116T004641Z-device-kv-prefill-staging`](../20261116T004641Z-device-kv-prefill-staging/INDEX.md))
traced what was left of a `q8_0` prefill's extra time over FP16 to the host copy of the KV: after each whole-layer prefill
window the region downloaded the window's K and V as FP32 and the CPU encoded every row to `q8_0` on one thread (about 9 ns
a value) while the device waited. Owner decision 7 (a): where a side's device format is that side's host type, the region
downloads the window's stored blocks, which the device's KV write already encoded with the host codec's arithmetic, into
the host tensor's packed rows as they are. This run is the unpinned A/B after that change, before the owner's pinned gate
(decision 8 (a): q8_0 over f16 <= 1.10x at one 2,048-row window and at 128-row windows, every sweep model).

**Pinned:** no. Desktop sharing the GPU and the CPU; one JVM per model and type; every ratio is within one session, run
in the order f16, `q8_0`, `q8_0`/`q4_0` per model. The f16 side read lower than in the previous entry's session (TinyLlama
820 ms against 973 ms in one window; nothing on the f16 path changed): compare ratios across sessions, never times.

**Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB, sm_61. **Build:** HEAD `a4a91182` plus the previous entry's kernel change
and this change, run from the tree's classes through `mvn -o test -pl node -am`.

**Command.** `mvn -o test -pl node -am -Dgroups=gpu -Dtest=DeviceKvPrefillTimeLiveTest -Dsurefire.failIfNoSpecifiedTests=false
-Djuno.test.deviceKvPp=true -Djuno.test.deviceKvPp.models=<tinyllama|qwen25|phi35|mistral> -Djuno.test.deviceKvPp.type=<f16|q8_0|q8_0/q4_0>`:
the model's 2,048-token quality text prefilled through `forwardBatch` (the serving prefill path), once in one window and once
in 128-row windows, one warm-up and five measured repetitions per width; a JVM's reading is the median.

## 1. Files

| File | What |
|---|---|
| `prefill-time.txt` | every `DEVICE-KV-PP` line, four models, three types |
| `quality-recheck.txt` | `DeviceKvQuantQualityLiveTest` on this tree: TinyLlama, Qwen2.5-3B, Phi-3.5-mini, three types |

## 2. What changed

For a whole-layer window, `ResidentActivation.materializeToKv` queues each side's download from the device copy's rows
(`DeviceKvCache.encodedRowsK/V`, `count x rowBytes`: 34 bytes per 32 values at `q8_0`, 18 at `q4_0`) instead of the FP32
activation (128 bytes per 32 values), and writes them with `SessionKvTensor.writeEncodedRows`: one bulk copy on the dense
tensor, one copy per page on the paged one. An `f16` side, and a quantized host side the device holds as FP16 (a head width
not a multiple of 32), download FP32 rows as before. `PrefillRegionEncodedKvTest` holds the host bytes equal to the old
path's (the FP32 rows written as floats), every device format pair, dense and paged tensors, and counts the downloads by
JFR staging site (`materialize(prefill k encoded)` against `materialize(prefill k)`).

## 3. Prefill time (median of five; ratio over the same session's f16)

| Model | Width | f16 | `q8_0` | ratio (before) | `q8_0`/`q4_0` | ratio |
|---|---|---|---|---|---|---|
| TinyLlama | one 2,048-row window | 820.4 ms | 769.2 ms | **0.938** (1.094) | 784.0 ms | 0.956 |
| TinyLlama | 128-row windows | 1,113.1 ms | 1,031.2 ms | **0.926** (1.046) | 1,032.3 ms | 0.927 |
| Qwen2.5-3B | one window | 2,038.2 ms | 2,016.7 ms | **0.989** (1.150) | 2,074.8 ms | 1.018 |
| Qwen2.5-3B | 128-row windows | 2,579.3 ms | 2,505.7 ms | **0.971** (1.076) | 2,583.4 ms | 1.002 |
| Phi-3.5-mini | one window | 3,285.7 ms | 2,941.2 ms | **0.895** (1.835) | 2,929.1 ms | 0.891 |
| Phi-3.5-mini | 128-row windows | 4,003.8 ms | 3,581.5 ms | **0.895** (1.650) | 3,584.4 ms | 0.895 |
| Mistral 7B | one window | 4,652.6 ms | 4,362.6 ms | **0.938** (1.172) | 4,340.0 ms | 0.933 |
| Mistral 7B | 128-row windows | 5,568.2 ms | 5,280.0 ms | **0.948** (1.103) | 5,284.0 ms | 0.949 |

"Before" is the previous entry's ratio, after its kernel change and before this one. With the host encode gone, a
`q8_0` prefill is now faster than the FP16 one on every sweep model at both widths: its attention launch already was
(0.90x to 0.95x), and its KV download is a quarter of the bytes. The proposed bar is 1.10x; the owner's pinned gate
scores it.

## 4. Unchanged outputs

`quality-recheck.txt`: `DeviceKvQuantQualityLiveTest` at `f16`, `q8_0` and `q8_0`/`q4_0` prints the figures of the
previous kernels, every one: TinyLlama `q8_0` NLL 2.45773, 509 of 512, first disagreement at 11; Qwen2.5-3B 1.69806, 510
of 512; Phi-3.5-mini 1.36070 and 1.36423, the kit's `gpu.kv-quality` readings of 2026-11-15
([`20261115T232030Z-kit-device-kv-probes`](../20261115T232030Z-kit-device-kv-probes/INDEX.md)), on the same text. The
host KV holds the bytes it held before (`PrefillRegionEncodedKvTest`), so nothing a later step reads from it changed.

## How to read it

Section 3 is the whole of decision 6's prefill question on this host, unpinned: the quantized copy no longer costs prefill
time. The host KV holds the bytes it held before, byte for byte (`PrefillRegionEncodedKvTest`), so every output is
unchanged (section 4).
