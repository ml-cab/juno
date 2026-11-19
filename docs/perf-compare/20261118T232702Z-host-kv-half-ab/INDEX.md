# Real FP16 host KV and growth toward the request's length: throughput and allocation, before and after

**Purpose.** Evidence for the KV-cache tier's item 6 (an `f16` host KV row holds IEEE halves, and dense KV grows toward
the request's length): does it move prefill or decode throughput, allocation per token or GC pauses? This is the
standing throughput and allocation gate's shape (tg and pp >= 0.95x, allocation per token <= 1.10x, in-span GC pause
<= 1.25x), measured unpinned as evidence only; the scored run is the owner's pinned run at the tier's close.

**Not pinned.** One session, A and B interleaved run by run, desktop session active (about 15% of the CPU used by
other processes throughout). Medians of three (two at 2,048 tokens on the CPU).

**Builds.** A = `c607e926` (the tree before item 6), player jar `6e2f714daae0c2e2`, kit jar `f5b973c10bff8f0c`.
B = `c607e926` plus item 6 (uncommitted), player jar `244271ec4dd6b154`, kit jar `ff526c3d25fa1bce`. Default cache
types (`f16`) on both: A stores float32 rows, B stores halves. JDK 25.0.4.1, GTX 1080, Xeon E5-1650 v2 (12 threads).

**Command** (each tree's own harness from its own root, on its own shaded jar; `<ref bin>` is the reference tool's
CPU or CUDA build directory):

```bash
scripts/performance-tests/compare-llama-cpp.sh --cpu|--gpu --models <model> --n-prompt 512|2048 \
  --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --llama-bin <ref bin> --out <dir>
# 2,048-token lanes again with COMPARE_HEAP=8g on both sides; the CPU ones with --juno-warmup 1
```

## 1. Throughput, B over A

| Lane | Heap | pp A | pp B | pp B/A | tg A | tg B | tg B/A |
|---|---|---|---|---|---|---|---|
| CPU TinyLlama, 512 | 4g | 33.30 | 32.31 | **0.970** | 11.98 | 11.74 | **0.980** |
| CPU Phi-3.5-mini, 512 | 6g | 9.32 | 9.37 | **1.006** | 3.00 | 2.99 | **0.998** |
| GPU TinyLlama, 512 | 4g | 3,178.8 | 3,147.2 | **0.990** | 166.43 | 163.68 | **0.983** |
| GPU Phi-3.5-mini, 512 | 6g | 841.0 | 942.9 | **1.121** | 59.80 | 59.41 | **0.993** |
| CPU Phi-3.5-mini, 2,048 | 6g | out of heap | 6.50 | - | 2.88 | 2.87 | 0.997 |
| GPU Phi-3.5-mini, 2,048 | 6g | out of heap | 708.3 | - | 59.08 | 58.83 | 0.996 |
| CPU Phi-3.5-mini, 2,048 | 8g | 7.11 | 6.89 | **0.969** | withheld | 2.86 | - |
| GPU Phi-3.5-mini, 2,048 | 8g | 664.5 | 678.0 | **1.020** | 59.42 | 59.65 | **1.004** |

Tokens per second, medians. The generate lane decodes 64 tokens after a 10-token prompt at every `--n-prompt`, so its
tg is the same workload in every row.

## 2. A 2,048-token prompt in the harness's 6 GiB heap

A fails the 2,048-token lane with HTTP 500, `java.lang.OutOfMemoryError: Java heap space`, on the CPU and on the GPU, in
both runs; B answers it in both. The first request of the lane, the harness's prompt calibration, is 2,057 prompt tokens
with one to generate: A's dense KV passes 2,048 positions and doubles to 4,096 float32 positions (3,221,225,472 bytes on
Phi-3.5-mini) beside 3.4 GiB of weights, so calibration fails and the uncalibrated prompt fails again. B's KV grows to
the 2,058 positions the request can write, as halves (809,238,528 bytes).

## 3. Allocation and GC

| Lane | Allocation per token, A | B | GC pause in the token span | Prefill GC pause, A / B |
|---|---|---|---|---|
| CPU TinyLlama, 512 | 290,951 to 295,549 | 303,885 to 310,546 (**1.048x**) | 0 / 0 | 0 / 0 |
| CPU Phi-3.5-mini, 512 | 341,386 to 352,260 | 341,586 to 385,067 (0.997x) | 0 / 0 | 0 / 0 to 4.9 ms |
| CPU Phi-3.5-mini, 2,048, 8g | 352,362, 391,363 | 353,404, 359,305 (0.958x) | 0 / 0 | 2.3 to 5.0 / 0 ms |
| GPU Phi-3.5-mini, 2,048, 8g | 412,820 to 434,281 | 431,024 to 435,834 (1.039x) | 0 / 0 | 16.7 to 19.1 / 10.9 to 11.2 ms (**0.60x**) |
| GPU TinyLlama and Phi-3.5-mini, 512 | 0, 0, 418,956; 280,222 to 425,741 | 421,554 to 500,352; 281,121 to 468,273 | 0 / 0 | 0 / 0 |

Bytes per generated token over the measured request (`allocated_bytes_per_token_in_recording`). The GPU 512 readings sit
at two levels on both builds (about 280 KB or 415 to 500 KB, two A runs read 0), so their medians are not compared.

**The handler alone** (`alloc-probe/AllocProbeMain.java`, the calling thread's allocated bytes per request: a 10-token
prefill, 63 decode steps, evict; four requests on one handler, the last two read; B told the request's length):

| Handler | Prefill A | Prefill B | Decode per token A | Decode per token B |
|---|---|---|---|---|
| TinyLlama, CPU | 1,848,648 / 1,848,304 | 1,847,592 / 1,847,552 | 162,920 / 162,422 | 162,920 / 162,427 |
| TinyLlama, CUDA | 1,706,352 / 1,706,424 | 1,708,168 / 1,708,240 | 376,177 / 374,802 | 376,177 / 374,794 |
| Phi-3.5-mini, CUDA | 2,364,648 / 2,355,360 | 2,367,264 / 2,357,976 | 486,262 / 485,297 | 472,436 / 471,473 |

```bash
javac --add-modules jdk.incubator.vector -cp <kit jar> -d classes docs/perf-compare/20261118T232702Z-host-kv-half-ab/alloc-probe/AllocProbeMain.java
java -Xms8g -Xmx8g -XX:+UseG1GC --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp classes:<kit jar> cab.ml.juno.kit.AllocProbeMain models/<model>.gguf 10 64 4 reserve|none cpu|gpu
```

## 4. LoRA hot-path gate

`scripts/performance-tests/compare-lora.sh --gpu --skip-build --no-publish` from each tree (TinyLlama, CUDA, train to the
loss target, then one `--lora-play` request of 22 prompt tokens and a 7-token reply, a fresh JVM per rep). Bars: train
<= 1.25x, playback >= 0.80x.

| Run | Train ms per pass, A / B | Playback tps, A / B | B/A | Prefill pass ms, A / B | Decode ms (8 passes), A / B | Recall |
|---|---|---|---|---|---|---|
| `--reps 3`, A then B | 2,800 / 2,733 (0.976x) | 12.87 / 11.49 | **0.893** | 246 to 274 / 301 to 330 | 253 to 261 / 247 to 250 | 6 of 6 |
| `--reps 1` x3, interleaved | 3,000 / 3,000 to 3,067 | 11.33 / 9.30 | **0.821** | 266 to 308 / 343 to 432 | 272 to 316 / 279 to 313 | 6 of 6 |
| `--reps 1` x3, interleaved, B on `f32` | - | 11.20 / 11.02 | 0.984 | 279 to 325 / 279 to 315 | 286 to 302 / 296 to 308 | 6 of 6 |

The gate passes in every run. The playback difference is all in the request's one prefill pass, and only on `f16`: B on
`f32` host KV plays back as A does. The eight decode passes after the prefill read the same `f16` rows and do about as
much attention work (about 146 thousand head-key products against the prefill's 170 thousand) at A's speed, so the
prefill pays a first execution, not a per-row cost: it is the first code in the JVM to read `f16` rows (training runs
the float attention path), and those reads run before the JIT has compiled them. A server pays it once, on its first
request; the warmed lanes above do not see it. The GPU warmed from 74 C to 81 C over the interleaved runs (graphics clock
1,835 to 1,733 MHz), which is why both sides read lower there than in the first run.

## How to read it

- **Throughput.** Every lane is at or above 0.95x. The lowest are the CPU prefills, TinyLlama at 512 (0.970x) and
  Phi-3.5-mini at 2,048 (0.969x): CPU attention now decodes each key and value head slice from halves (as it does for
  `q8_0`) instead of reading floats, a cost that grows with the context. GPU prefill gains (1.121x at 512, 1.020x at
  2,048): the host KV takes the device's FP16 rows as they are, half the bytes over the bus and no float copy.
- **Allocation.** The handler allocates the same bytes per token on both builds (CPU, CUDA), and fewer on Phi-3.5-mini
  CUDA. The harness's CPU TinyLlama reading (+4.8%, about 14 KB a token, every B run above every A run) is on the
  generating thread but not in the handler. Nothing item 6 added above the handler runs per token (the length hint
  runs once per request), and the recording holds too few allocation samples (12) to place it. It is inside the 1.10x
  bar.
- **GC.** No pause falls inside a token span on either build. The 2,048-token GPU prefill pauses for 0.60x A's time:
  less heap to scan with the KV a quarter the size.
- **Withheld readings.** The harness withholds a run's tg when its JFR timestamps disagree with the engine clock by
  more than its allowance (here 28 to 74 ms in a 23 s request); `runs.txt` marks them.

`runs.txt` has every run's reading.
