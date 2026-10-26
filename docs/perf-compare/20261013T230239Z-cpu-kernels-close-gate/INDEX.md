# CPU decode and prefill kernels: closing gate, run 1 (2026-10-13 23:02Z to 2026-10-14 13:37Z)

**Purpose.** The owner's pinned closing gate for the CPU kernel work: the decode kernel microbenchmark, the closing
sweeps, same-hour A/Bs against the build before the work on the CPU and GPU lanes, and the LoRA gates.

**Pinned:** yes. CPU governor `performance`, turbo off (3.50 GHz on every core; mean core clock after each kernel run
3,492 to 3,496 MHz); every A/B run reports `clock_pinned: true`. The GPU clock is recorded, not locked (this host's
Pascal card refuses locking).

**Builds.**

| Side | Jar sha256 (first 16) | Build |
|---|---|---|
| baseline | `b9a0440b3a1b597a` | HEAD `01027ee9`, before any change to the CPU kernels (the jar of the step 2 sweeps) |
| candidate | `e37e16c004f4c213` | `01027ee9` plus the uncommitted CPU kernel work as of 2026-10-13 18:00 |

**Command.** `PART=KSCAL bash dist/cpu-kernels-close/run-gate.sh` (started by its `wait-and-run.sh`); the full log is
`gate.log` here. Part K runs `scripts/performance-tests/cpu-decode-kernel-variants.sh --skip-build` three times; Parts C
and A alternate the two jars A B A B A B through `compare-llama-cpp.sh ... --juno-warmup 2 --juno-reps 1 --reps 1
--no-tuned-lane --no-publish --juno-jar <jar>`; Part L runs each tree's own `compare-lora.sh --reps 1 --skip-build`
three times per side and lane. Part S's sweeps are published in their own directories (below).

## Part K: decode kernels, Mistral 7B FFN shapes (Q4_K 14336x4096, Q6_K 4096x14336)

Cycles per weight = median ms per call x 3.5e6 / 58,720,256 weights (x 6 cores at 12 threads). Raw reports:
`kernel/run<i>-threads<t>.md`.

| Kernel | Threads | Runs (ms per call) | Cycles per weight | Bound | Before this work (`20261012T225531Z-cpu-kernel-baseline`) |
|---|---|---|---|---|---|
| Q4_K | 1 | 15.704, 15.449, 15.626 | **0.931** | <= 1.1 | 1.52 |
| Q6_K | 1 | 35.713, 35.405, 36.071 | **2.13** | <= 2.7 | 5.41 |
| Q4_K | 12 (per core) | 3.108, 3.082, 3.029 | **1.102** | <= 1.1 | 1.52 |
| Q6_K | 12 (per core) | 6.534, 6.964, 7.232 | **2.49** | <= 2.7 | 5.24 |

The gate's table printed the Q4_K 12-thread figure as 1.10 and passed it, because it compared the rounded value. The
unrounded median is 1.1022, 0.2% above the bound (the three runs read 1.083, 1.102 and 1.111). The tier records the
owner's decision on it.

## Part S: closing sweeps (published separately)

| Model | CPU tg ratio @128 | pp ratio @128 | pp ratio @512 | pp ratio @2048 | pp t/s @512 against the pre-tier sweep | pp t/s @2048 against the pre-tier sweep |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 0.391x | 0.321x | 0.321x | 0.315x | 12.38 -> 24.81 (2.00x) | 6.10 -> 22.12 (3.63x) |
| Qwen2.5-3B | 0.366x | 0.320x | 0.312x | 0.313x | 5.14 -> 8.56 (1.67x) | 3.18 -> 8.14 (2.56x) |
| Phi-3.5-mini | 0.308x | 0.348x | 0.369x | 0.314x | 3.81 -> 7.07 (1.86x) | 1.68 -> 5.70 (3.40x) |
| Mistral 7B | 0.351x | 0.315x | 0.319x | 0.328x | 2.25 -> 3.70 (1.65x) | 1.29 -> 3.34 (2.58x) |

Sweeps: [`20261013T231340Z`](../20261013T231340Z/INDEX.md) (128), [`20261013T235235Z`](../20261013T235235Z/INDEX.md) (512),
[`20261014T011910Z`](../20261014T011910Z/INDEX.md) (2048) and [`20261014T050210Z`](../20261014T050210Z/INDEX.md) (2048, Phi-3.5-mini at
8 GiB). Pre-tier sweeps: `20261012T045931Z` (512), `20261012T071213Z` and `20261012T162344Z` (2048).

## Part C: CPU same-hour A/B, pinned (per-run results in `ab/cpu128-*`, `ab/cpu512-*`)

Medians of three; B/A is candidate over baseline. Allocation is the harness's `jfr.allocated_bytes_per_token`.

| `n_prompt` | Model | pp t/s A -> B (B/A, bar) | tg t/s A -> B (B/A, bar) | Allocation per token A -> B (B/A, bar <= 1.10) |
|---|---|---|---|---|
| 128 | TinyLlama | 16.63 -> 24.28 (1.460, >= 0.95) | 7.87 -> 11.94 (1.516, >= 0.95) | 77.2 MB -> 543.4 MB (**7.04x**) |
| 128 | Qwen2.5-3B | 6.19 -> 8.30 (1.342, >= 1.10) | 2.59 -> 4.17 (1.608, >= 1.10) | 145.7 MB -> 180.6 MB (**1.24x**) |
| 128 | Phi-3.5-mini | 4.84 -> 6.96 (1.439, >= 0.95) | 2.29 -> 3.00 (1.313, >= 0.95) | 162.4 MB -> 919.2 MB (**5.66x**) |
| 128 | Mistral 7B | 2.59 -> 3.63 (1.403, >= 1.10) | 1.28 -> 2.13 (1.661, >= 1.10) | 217.2 MB -> 331.8 MB (**1.53x**) |
| 512 | TinyLlama | 12.45 -> 25.10 (2.016, >= 1.10) | 7.80 -> 11.92 (1.529) | 79.9 MB -> 502.3 MB (6.28x, not gated at 512) |
| 512 | Qwen2.5-3B | 5.09 -> 8.41 (1.652, >= 1.10) | 2.60 -> 4.26 (1.640) | 148.0 MB -> 199.7 MB (1.35x) |
| 512 | Phi-3.5-mini | 3.84 -> 6.89 (1.796, >= 1.10) | 2.32 -> 3.01 (1.300) | 171.2 MB -> 914.0 MB (5.34x) |
| 512 | Mistral 7B | 2.29 -> 3.55 (1.555, >= 1.10) | 1.28 -> 2.12 (1.653) | 200.0 MB -> 338.6 MB (1.69x) |

In-span GC pause total: 0.0 ms on every run of both builds. Two candidate runs are outliers against their own side
(Phi-3.5-mini pp 1.38 t/s at 128 and 3.52 t/s at 512); the medians are unaffected.

**Throughput met on every model; allocation missed on every model.** The candidate's JFR allocation samples put almost
all of the excess in the Vector API's generic path (`jdk.incubator.vector.AbstractVector.convert0` and
`ShortVector.lanewiseTemplate`, tens of GB per request), reached from the prefill matmul `CpuIntegerGemm`: its one-row
remainder loop passed a vector to a helper (`widenPairs(ShortVector)`) that C2 no longer inlined once the tile method
had grown, so the vector was boxed and every operation on it allocated. The tier record carries the fix and its
verification. The harness's allocation figure also counts each thread's allocation from its start (model load and the
two warmups included), not only the measured request; the tier record raises that with the owner.

## Part A: GPU same-hour A/B, pinned, `n_prompt=512` (per-run results in `ab/gpu512-*`)

| Model | pp t/s A -> B (B/A) | tg t/s A -> B (B/A) | Allocation B/A | GC in span | Greedy output |
|---|---|---|---|---|---|
| TinyLlama | 3172.6 -> 3197.7 (1.008) | 150.5 -> 152.5 (1.013) | 1.000 | 0 / 0 ms | equal, 3 of 3 |
| Qwen2.5-3B | 1286.2 -> 1280.4 (0.996) | 58.80 -> 58.94 (1.002) | 1.000 | 0 / 0 ms | equal, 3 of 3 |
| Phi-3.5-mini | 863.2 -> 848.5 (0.983) | 58.12 -> 58.47 (1.006) | 1.000 | 0 / 0 ms | equal, 3 of 3 |
| Mistral 7B | 583.0 -> 564.8 (0.969) | 39.56 -> 39.00 (0.986) | 1.000 | 0 / 0 ms | equal, 3 of 3 |

Bars: t/s B/A >= 0.95, allocation <= 1.10x, GC <= 1.25x (<= 5 ms under 5 ms). All met.

## Part L: LoRA, pinned by hand (per-run results in `lora/`)

| Lane | Training ms per pass A / B (throughput B/A, bar >= 0.95) | Playback wall-clock tps A / B (B/A, bar >= 0.80) | Recall |
|---|---|---|---|
| GPU | 2866.7 / 2800.0 (1.024) | 12.939 / 12.635 (0.977) | true in every run |
| CPU | 51466.7 / 51533.3 (0.999) | 0.853 / 0.849 (0.995) | true in every run |

Medians of three. All met.

## How to read this

Part K's figures are unaffected by the matmul fix that followed (it times the decode kernels, which the fix does not
touch), and so are Part L's training figures (LoRA training's frozen forward uses the float kernels); LoRA playback's
prompt may pass through the fixed matmul. Part C is re-run on the fixed build and published in its own directory. Part
S's ratios are for this candidate jar.
