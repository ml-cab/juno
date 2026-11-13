# Bounded GPU prefill launches: kernel timing at the longest window, greedy parity against the previous build

**Purpose.** A GPU prefill window's attention and its tiled K-quant matmuls now run as several launches of bounded work
(`BoundedLaunches`; attention splits between whole blocks of query rows, the GEMM between whole 64-row tiles, each launch
with the whole window's geometry). This entry holds the readings that change rests on: the single-launch device time at a
2048-token window and at the longest window the device KV holds (32,768 tokens), and greedy replies against the build
before the change on prompts long enough to split.

**Not pinned.** Clocks were not pinned (no prompt-free sudo). None of these readings is scored against a no-regression gate
at 0.90x or tighter; the kernel-timing check is an absolute bar (500 ms, a quarter of the 2 s display-driver watchdog), and
the time-to-first-token figures are single readings, for orientation only.

**Builds.**

| Build | What | Jar (SHA-256, first 16) |
|---|---|---|
| baseline | the working tree before the change (HEAD `6ad5c728` plus the uncommitted page-reclaim work), built from a copy | `8c0edb793628f2e4` |
| candidate (smoke) | the working tree with the change | `52ff3d6cbf143534` |
| kit run | the final tree; differs from the smoke's candidate only in `GpuKernelTimings` (the timing class, off the inference path: it also times a window's last launch) and comments | `2e7ab50ca9c2cdf4` |

Host: Xeon E5-1650 v2, GTX 1080 (8 GiB), Linux, driver 580.178.04.

## 1. Kernel timing (`gpu.kernel-timing`)

Command: `./juno-kit.sh --probes env,gpu.kernel-timing --yes --models-dir models` (5 timed launches each, after 2 untimed;
CUDA events around each launch). Files: `gpu-kernel-timing.json`, `kit-SUMMARY.txt`.

| Kernel and shape | 2048 tokens: one launch, median (max) ms | 32,768 tokens: launches | 32,768 tokens: slowest launch, median (max) ms |
|---|---|---|---|
| Prefill attention, TinyLlama 32/4/64 | 16.34 (16.45) | 98 | 41.27 (41.74) |
| Prefill attention, Qwen2.5-3B 16/2/128 | 19.79 (20.20) | 98 | 57.51 (57.94) |
| Prefill attention, Phi-3.5-mini 32/32/96 | 35.21 (35.27) | 212 | 51.43 (51.93) |
| Prefill attention, Mistral 7B 32/8/128 | 37.58 (37.75) | 212 | 54.41 (54.95) |
| Tiled GEMM, Mistral 7B gate/up 14336x4096 Q4_K | 22.77 (23.66) | 10 (3,392 rows each) | 38.16 (38.49) |
| Tiled GEMM, Mistral 7B down 4096x14336 Q6_K | 25.34 (25.63) | 10 (3,392 rows each) | 40.90 (42.26) |
| LM head GEMV, Qwen2.5-3B 151936x2048 Q6_K | 1.90 (1.92) | not window-dependent | |
| **Check: slowest launch <= 500 ms** | **met** (37.75) | | **met** (57.94) |

Before the change, one launch over the whole window read 596.51 ms for Mistral 7B's attention at 8,192 tokens on this card
(`20261106T045718Z-gpu-kernel-timing`), and extrapolates to about 9.5 s at 32,768. At 2048 tokens every sweep shape is one
launch, as before. GPU memory peaked at 514 MiB as sampled (the catalog's figure is updated from 152).

**Against the slowest GPU a kit run has measured** (GeForce MX150, `20261112T001051Z-remote-linux-amd64-core-i7-8550u-geforce-mx150`),
scaling its 2048-token single launches by work (time follows work; `BoundedLaunchesTest` holds this):

| Kernel | MX150 at 2048 tokens | Work at 2048 over the budget | Projected heaviest launch at any length |
|---|---|---|---|
| Attention, Mistral 7B heads | 268.75 ms | 8.59e9 / 1.2e10 | 375 ms |
| Tiled GEMM, Mistral 7B down | 209.69 ms | 1.20e11 / 2.0e11 | 349 ms |

The slowest launch at 32,768 tokens on the GTX 1080 reads 1.45x its 2048-token single launch for Mistral 7B's attention
(1.40x the work); the small-head shapes read up to 2.9x (Qwen2.5-3B), because their 2048-token window is about a third of a budget's work (Qwen2.5-3B's MX150 reading projects to 377 ms).

## 2. Greedy parity on long prompts (`smoke-long-prompt-prefill.sh`)

Command: `SMOKE_HEAP=16g JUNO_JAR=<candidate> scripts/performance-tests/smoke-long-prompt-prefill.sh --models mistral
--lengths "2048 8192" --baseline-jar <baseline>` (GPU, both schedules). Files: `smoke-long-prompt.log`,
`smoke-long-prompt-results.jsonl`. At 8,192 tokens Mistral 7B's attention splits (more than one launch per window or chunk)
and so does the GEMM (3,392 rows a launch); at 2048 nothing splits.

| Schedule | Prompt tokens | Greedy text vs baseline | TTFT candidate (ms) | TTFT baseline (ms) |
|---|---|---|---|---|
| static | 2044 | identical | 4,955 | 5,100 |
| static | 8472 | identical | 347,788 | 347,935 |
| continuous | 2044 | identical | 10,779 | 11,381 |
| continuous | 8472 | identical | 74,932 | 76,329 |

All 16 checks pass (HTTP 200, prompt length within 10%, streamed text equal to non-streamed, greedy text equal to the
baseline build). The first attempt at the smoke's default 8 GiB heap failed the 8,192-token calibration request with
`OutOfMemoryError: Java heap space` before either build was compared: the dense host KV doubles past the request and stores
`f16` as 32-bit floats, the host footprint item the KV tier still owes. Hence the 16 GiB heap.

**Finding, both builds:** on the `static` schedule an 8,472-token Mistral 7B prompt reads 348 s to the first token, against
75 s on `continuous`. It is the same in the build before the change, so it is not caused by the split.

## How to read this

- The split changes no result: section 2's replies are identical, and `BoundedLaunchParityTest` holds every split launch to
  one launch's bits.
- The kernel-timing check is absolute: each reading is one kernel launch, and the display-driver watchdog acts on a single
  launch. A window's total time is not bounded by it, and is not what it measures.
- TTFT readings are single and unpinned; they say the split costs nothing visible at 8,192 tokens, not a ratio to score.
