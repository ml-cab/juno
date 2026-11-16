# Quantized device KV: prefill time, traced per launch; the prefill attention's tile staging

**Purpose.** A 2,048-token prefill with `q8_0` device KV took 1.21x to 1.31x the FP16 time in the owner's decode gate logs
([`20261115T200324Z-device-kv-tg-gate-rerun`](../20261115T200324Z-device-kv-tg-gate-rerun/INDEX.md)). The work set for it:
trace the prefill per launch, rework the tiled attention kernel's quantized read at prefill width, and score the prefill
time against a bar of 1.10x the FP16 time. This run is the trace, the kernel change and the unpinned A/B before any pinned
run. The trace found a second cost, off the device, that the kernel cannot remove (section 2).

**Pinned:** no. Desktop sharing the GPU and the CPU; one JVM per model and type; the FP16 side varies by up to 10% between
sessions (Mistral 7B FP16 read 4,768 ms under the profiler and 5,230 ms later). Every ratio below is within one session.

**Host:** Xeon E5-1650 v2, 64 GiB, GTX 1080 8 GiB, nvcc 12.0, sm_61. **Builds:** HEAD `a4a91182` (before) and HEAD plus
this change (after), run from the tree's classes through `mvn -o test -pl node -am`. The kernel builds:
`gqa_attention_kv.ptx` before `bc6be797...6c657548`, after `e91f6952...181322b8`; `gqa_attention.ptx` (FP16) before
`96a01e41...377d5a89`, after `d1359455...44d97d95` (code identical; only the mangled names of its shared arrays changed).

**Commands.**
- Prefill time: `mvn -o test -pl node -am -Dgroups=gpu -Dtest=DeviceKvPrefillTimeLiveTest -Dsurefire.failIfNoSpecifiedTests=false
  -Djuno.test.deviceKvPp=true -Djuno.test.deviceKvPp.models=<tinyllama|qwen25|phi35|mistral> -Djuno.test.deviceKvPp.type=<f16|q8_0>`
  (a 2,048-token prompt through `forwardBatch`, windows 2048 and 128, one warm-up and five measured prefills).
- Per launch: the same under `nvprof --profile-child-processes --print-gpu-summary`, `.windows=2048 .reps=2`.
- Host encode: `java -cp kvcache/target/classes host-kv-encode-bench.java`.

## 1. Files

| File | What |
|---|---|
| `prefill-time.txt` | every `DEVICE-KV-PP` line: before (TinyLlama), before under the profiler (four models), after (four models) |
| `attention-trace.txt` | `nvprof` GPU summaries: FP16 before, `q8_0` before, `q8_0` after, per model |
| `host-kv-encode.txt`, `host-kv-encode-bench.java` | `DenseKvTensor.writeRows` of a 2,048-row window per host KV type and KV width |
| `quality-recheck.txt` | `DeviceKvQuantQualityLiveTest` on the after build, TinyLlama and Qwen2.5-3B, three types |
| `registers.txt` | `ptxas -v` registers per entry, before and after |

## 2. Where the prefill time went (before; one 2,048-row window, under the profiler)

| Model | Prefill FP16 -> `q8_0` | GPU busy difference | of which attention | Off the device |
|---|---|---|---|---|
| TinyLlama | 1,002 -> 1,339 ms (1.34x) | +152 ms | 14.17 -> 21.08 ms a layer (1.49x), +152 ms | +185 ms |
| Qwen2.5-3B | 2,267 -> 2,766 ms (1.22x) | +189 ms | 19.76 -> 25.35 ms (1.28x), +201 ms | +310 ms |
| Phi-3.5-mini | 3,419 -> 6,985 ms (2.04x) | +293 ms | 36.13 -> 45.70 ms (1.26x), +306 ms | +3,274 ms |
| Mistral 7B | 4,768 -> 6,234 ms (1.31x) | +365 ms | 39.87 -> 51.01 ms (1.28x), +356 ms | +1,101 ms |

The device-side difference is the attention launch; the KV write (`kv_encode_kv_rows`, 0.06 ms a layer on TinyLlama)
costs less than FP16's two casts. The rest is off the device and scales with layers x 2,048 x KV width x 2 at roughly
8 ns per value on every model: after each layer the prefill region downloads the window's K and V as FP32 and writes the
host KV tensors, and a `q8_0` host tensor encodes every row on the CPU, one thread, while the device waits for the next
layer. `host-kv-encode.txt` reads that write alone: 9.0 to 9.3 ns a value for `q8_0` against 0.27 to 0.58 ns for the
`f16` host type (FP32-backed, a copy); on Phi-3.5-mini (KV width 3,072) 57 ms per tensor per layer, about 3.6 s a prefill.

## 3. The kernel change and its reading

Tile staging over a quantized side: eight values of a key per thread (`kv_load8_t`: one scale and two realigned quant
words), where it read a scale and two bytes per pair, the index stepped without a division. The 64- and 128-wide
quantized entries no longer contain the decode loop (their decode rows go to the wide decode entries): 196 -> 127
registers at width 64 (FP16's count; four blocks per multiprocessor again instead of two), 226 -> 217 at width 128.
The tile holds the same floats, and the arithmetic after staging is the FP16 kernel's.

Per-layer attention launch at a 2,048-row window (`attention-trace.txt`):

| Model | FP16 | `q8_0` before | `q8_0` after |
|---|---|---|---|
| TinyLlama | 14.17 ms | 21.08 ms (1.49x) | **13.40 ms (0.95x)** |
| Qwen2.5-3B | 19.76 ms | 25.35 ms (1.28x) | **17.77 ms (0.90x)** |
| Phi-3.5-mini | 36.13 ms | 45.70 ms (1.26x) | **33.26 ms (0.92x)** |
| Mistral 7B | 39.87 ms | 51.01 ms (1.28x) | **36.16 ms (0.91x)** |

Prefill time after (median of five, same session per model, `prefill-time.txt`):

| Model | One 2,048-row window: FP16 / `q8_0` | Ratio | 128-row windows: FP16 / `q8_0` | Ratio |
|---|---|---|---|---|
| TinyLlama | 973.0 / 1,064.0 ms | **1.094** (before 1.259) | 1,280.9 / 1,339.3 ms | **1.046** (before 1.210) |
| Qwen2.5-3B | 2,207.0 / 2,537.8 ms | **1.150** | 2,834.2 / 3,050.0 ms | **1.076** |
| Phi-3.5-mini | 3,714.0 / 6,814.1 ms | **1.835** | 4,465.7 / 7,370.2 ms | **1.650** |
| Mistral 7B | 5,230.5 / 6,132.4 ms | **1.172** | 6,441.0 / 7,103.1 ms | **1.103** |

Against the proposed bar of 1.10x: TinyLlama under it, the others over it by what the host encode costs.

**Unchanged outputs.** `quality-recheck.txt`: every printed figure equals the readings on the previous kernels (TinyLlama
`q8_0` NLL 2.45773, 509 of 512, first disagreement at 11; Qwen2.5-3B 1.69806, 510 of 512; both `q8_0`/`q4_0` rows).
`DeviceKvQuantParityTest#prefillTileOverExactValues_isTheFp16Kernel` holds the staging to the bit: blocks whose values
FP16 stores exactly give a prefill equal to the FP16 kernel's, every format pair, head widths 64, 96, 128 and 256.

## How to read it

Section 2 is why the bar is not met by the kernel alone: the prefill's extra time over FP16 was 9% to 45% attention and
the rest the host KV encode. Section 3 is the kernel's part, done: the quantized attention is now faster than the FP16
one. The remaining gap is the host write, and how it is closed is an owner decision recorded in the plan.
