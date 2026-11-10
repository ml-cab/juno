# Tier 02H: Batched GPU decode

Status: **complete** (2026-11-09). Every exit criterion is checked. `--parallel 8` over `1` reads 1.77x to 2.32x (bar 1.50x) at
0.43x to 0.58x the mean latency (bar 1.00x), with 0 of 8 replies differing on every sweep model; the owner's pinned standing gate
is met; row 7 of the cross-surface table was restated by **Owner decision** 2026-11-09. Kit probe `gpu.unit-tests` did not pass on
this host: see README execution rule 10 under 2026-11-09 "the standing gate and the cross-surface runs". Added 2026-10-31 by owner
decision; begun 2026-11-07 by owner decision while Tier 02G is postponed until it is run on Windows (README execution rule 1).
Gap analysis refs: none — see "Why this tier, why now"

**Added 2026-10-31 (owner decision, Tier 02F record of that date).** A decode step of several requests runs
on the GPU through the host-driven path, while one request's decode runs in the device-resident region. On
this host's GTX 1080 a full static batch of 8 is therefore slower than 8 requests decoded one at a time.
This tier runs the batched step in the device region, so that batching gains on the GPU as it already does
on the CPU.

## Objective

A decode step of N requests (a static `--parallel` batch, or the continuous schedule's running set) runs on
the GPU the way one request's step does: each request's residual row is uploaded once per step and stays on
the device across the layers; norms, RoPE, SwiGLU and residual adds run on the device; each weight is read
once per layer for all N rows; attention reads each request's device KV at its own position; the host waits
once per layer. Static `--parallel 8` then reads above `--parallel 1` on the GPU.

## Why this tier, why now

- **The reading.** `docs/perf-compare/20261031T040116Z-gpu-parallel-window/` (`compare-parallel.sh --gpu
  --sessions 8`, 64 tokens, median of three): `--parallel 8` over `--parallel 1` is 0.748x (TinyLlama) and
  0.676x (Qwen2.5-3B) in aggregate t/s, with mean latency 1.38x and 1.50x. Every P = 8 run was one batch of
  8. On the CPU the same change reads 1.174x and 1.084x
  (`docs/perf-compare/20261031T041246Z-cpu-parallel-window/`), because `CpuIntegerDecodeBatch` reads each
  weight once for the whole batch.
- **The code.** `LocalInferencePipeline.forwardBatch` calls each handler's `forwardMultiDecode`. In
  `LlamaTransformerHandler.runLayersMultiDecode` the activations stay on the host: per layer it normalizes,
  rotates and packs K and V on the host, round-trips every projection through `sgemmLayerInto`, and uploads
  queries and downloads attention output around `CudaGqaAttention.attendBatched`, each copy synchronous.
  [Tier 01](TIER-01-gpu-activation-residency.md)'s record states it: "Prefill windows and batched decode
  (`transformerLayerBatch`) are not wired". Tier 02D later put the prefill window in the region; the batched
  decode step is still outside it. One request's step uploads its residual row once and downloads K, V and
  the layer output once per layer.
- **Two schedules sit on it.** `ContinuousBatchEngine` decodes its running set through the same
  `forwardBatch` call, so [Tier 07](TIER-07-continuous-batching.md)'s gate (continuous at least 1.10x
  static on the 8x64 GPU benchmark) is measured through this path. Its 0.86x finding is the same shape as
  the static reading above.
- **Why after Tier 02G and before Tier 03.** Tier 02G was placed directly after 02F by owner decision, and
  the plan runs one tier at a time. [Tier 03](TIER-03-kv-cache-maturity.md) changes the device KV that
  batched attention reads (the block-table read path, and item 7's quantized device KV). Building the batched
  region first means Tier 03 carries one device decode design, single and batched, through that change,
  rather than building on a host-driven path this tier replaces. Tier 02D ran before Tier 03 for the same
  reason.

## Scope

### In scope

1. **Decompose first.** Profile one batched decode step at N = 1, 2, 4 and 8 on TinyLlama and Qwen2.5-3B
   (JFR spans plus CUDA event timing): host compute, transfers, kernel time and waits per layer, against the
   single-request step. Publish it before any change. If the decomposition puts a threshold below out of
   reach, the threshold is restated with a recorded **Owner decision** before the work continues.
2. **Batched decode region.** Every handler whose single-request decode runs in the device region today gets
   a batched step in the region: N residual rows uploaded once per step, the projections as one batched
   K-quant matmul over N rows per weight (the packed path where the single region uses it), norms, RoPE,
   SwiGLU and residual adds on the device, attention through `attendBatched` over each request's device KV,
   and one download per layer for K, V and the layer outputs while the host KV stays the source of truth.
   New functionality goes through `GpuBindings`, not a vendor class, unless it is the CUDA-only packed path.
3. **Handlers outside the region** keep the host-driven batch and say so once at startup, as the single
   region's `announceUnsupported` does. Nothing falls back silently.
4. **Batched equals single.** Greedy replies of a batched step match the single-request replies (threshold
   below). On the CPU's integer path this already holds bit for bit; on the GPU today a batch of 8 at 64
   tokens gives a different reply from a single request.
5. **Docs follow the reading.** The `--parallel` row in `docs/howto.md` and the launcher's `--help` say `1`
   on the GPU today; once the gate is met they recommend `8` for GPU API servers too, with the reading's
   ratios.

### Out of scope

- The continuous scheduler's policy (token budget, admission, concurrency ceiling): Tier 07.
- The device KV layout (block table, quantized device KV): Tier 03, which carries this region through it.
  *(2026-11-10, owner decision recorded in [Tier 03](TIER-03-kv-cache-maturity.md): Tier 03's block-table read is the CPU
  attention's, over the host pages, so this region's device KV stays a dense copy per request; Tier 03 carries it through item
  7's quantized device KV only. A paged device KV is a candidate in [Tier 07](TIER-07-continuous-batching.md). The same holds
  for "the block-table read path" in "Why after Tier 02G and before Tier 03" above.)*
- Prefill: the prefill window is already in the region.
- ROCm kernels beyond what `GpuBindings` already dispatches: verified through the kit on AMD hardware
  (NEEDS-AMD-HARDWARE on this host).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | unchanged; the CPU batch path is `CpuIntegerDecodeBatch` |
| 2 | CUDA GPU inference | primary target |
| 3 | ROCm GPU inference | through `GpuBindings`; NEEDS-AMD-HARDWARE here, verified by the kit's GPU probes on a tester's AMD card |
| 4 | Static schedule | primary: a `--parallel` batch decodes in the region |
| 5 | Continuous schedule | the running set's decode step takes the same region; Tier 07 scores its gate |
| 6 | Single-node local mode | primary surface |
| 7 | Pipeline-parallel cluster | keeps its current path, one request at a time on each node, stated at startup; live checks 1 to 6. *Restated 2026-11-09 by **Owner decision**; it read "each node's stage runs its layers' batched step in the region", which no code did before or after this tier* |
| 8 | Tensor-parallel cluster | batched decode across shards keeps its current path or fails closed with a stated reason; live checks 7 and 8 |
| 9 | LoRA training | unchanged (CPU, float kernels) |
| 10 | LoRA playback | a batch with `--lora-play` either runs the adapter in the region or keeps the host-driven batch, stated at startup |
| 11 | Vision | moondream2 (phi2) has no device decode path; unchanged, stated |
| 12 | OpenAI REST surface | batched requests through `/v1/chat/completions`; grammar and tools requests batch as today |
| 13 | Native REST surface | batched `/v1/inference` |
| 14 | CLI | `--parallel` help text follows the reading |
| 15 | JVM embedding facade | `JunoHttpClient` against a batched server (the kit's consistency probe) |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first.
2. Pre-tier record: `compare-parallel.sh --gpu --sessions 8` at `--parallel` 1 and 8 on the four sweep
   models, and the standing gate's GPU lanes, with the pre-tier jar.
3. Item 1, the decomposition, published.
4. Tests first, then item 2 on the Llama family, then the other handlers in the region, then item 3.
5. Item 4's equality on every sweep model; the thresholds below; the standing gate; the docs.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other test
  or code in this tier (README execution rule 7).
- **Region against the host-driven batch**, GPU test group: per handler, logits of a batched step in the
  region against the current batched path on random K-quant weights, N = 1, 2 and 8, requests at different
  positions; the tolerance is the handlers' existing multi-decode tolerance.
- **Batched against single on real weights**: a GPU test that decodes the consistency prompts as one batch
  of 8 and one at a time, greedy, 64 tokens, and compares the replies.
- **Fail-closed**: a handler outside the region announces the host-driven batch once.
- **Kit probes**: `gpu.consistency.tinyllama` decodes 64 tokens in a batch of 8 as well as its current 8
  tokens; `gpu.parallel.tinyllama` records the `--parallel 8` over `--parallel 1` ratio as a reading.
- **Standing throughput and allocation gate** (README, "Test infrastructure"): run against the pre-tier jar
  and score it before closing this tier. The single-request region is shared code.
- **Perf gate (required)**: `compare-parallel.sh --gpu` at `--parallel` 1 and 8, `compare-schedule.sh --gpu`
  as a reading for Tier 07, `compare-lora.sh`, and the standing gate. Published under `docs/perf-compare/`
  with descriptive directory names.

  **Threshold.**
  - `compare-parallel.sh --gpu --sessions 8`, 64 tokens: `--parallel 8` aggregate t/s **>= 1.50x**
    `--parallel 1`, every sweep model (today 0.748x on TinyLlama and 0.676x on Qwen2.5-3B), median of three,
    the two settings alternating in one session, the method of the reading above.
  - Mean latency per session at `--parallel 8` **<= 1.00x** `--parallel 1`'s on the same runs.
  - Batched equals single: a batch of 8 gives the single-request greedy reply at 64 tokens on the
    consistency prompts, every sweep model; differing replies **<= 0**. A difference keeps the tier open
    unless an **Owner decision** records why it is acceptable.
  - Standing throughput and allocation gate: GPU tg and pp **>= 0.95x** the pre-tier build, allocation per
    token **<= 1.10x**, in-span GC pause total **<= 1.25x**.

## Kit probes

README execution rule 10.

- Keep passing: `gpu.consistency.tinyllama`, `gpu.parallel.tinyllama`, `gpu.schedule.tinyllama`,
  `gpu.mixed-prefill.tinyllama`, `gpu.live.*`, `gpu.residency.tinyllama`, `gpu.unit-tests`; on the CPU
  `consistency.tinyllama` and `parallel.tinyllama`, unchanged.
- Add: `gpu.consistency.tinyllama` decodes 64 tokens in a batch of 8 and compares each reply with the
  single-request reply; `gpu.parallel.tinyllama` records the `--parallel 8` over `--parallel 1` ratio as a
  reading.

## Models needed

The four sweep models, all present here (`INVENTORY.md`). Hardware: this host's GTX 1080.

## Exit criteria

- [x] The decomposition (item 1) is published before the first change, and any threshold it moved carries
      an **Owner decision**.
      *(2026-11-08: [`20261108T230602Z-gpu-batched-decode-decomposition`](../perf-compare/20261108T230602Z-gpu-batched-decode-decomposition/INDEX.md),
      on the pre-tier jar, before any code change. No threshold moved: the estimate reaches about 2.5x.)*
- [x] Every handler whose single-request decode runs in the device region decodes a batch there; the others
      announce the host-driven batch at startup.
      *(2026-11-09: the Llama family (TinyLlama, Mistral 7B, Qwen2.5-3B), Phi-3 and Qwen3 handlers; `BatchDecodeRegionHandlerTest`,
      `ResidentBatchDecodeTest`. Phi-2 and Qwen3-MoE keep the host-driven batch and say so once: `BatchDecodeAnnouncementTest`.)*
- [x] Every threshold above is met, with the run directories cited. A miss keeps the tier open; closing on
      one takes an explicit **Owner decision** recorded here with the measured number.
      *(2026-11-09: `--parallel 8` over `--parallel 1` 1.77x to 2.32x against 1.50x, mean latency 0.43x to 0.58x against 1.00x, differing
      replies 0: [`20261109T005616Z-gpu-batched-decode-parallel`](../perf-compare/20261109T005616Z-gpu-batched-decode-parallel/INDEX.md). The standing gate:
      the next criterion.)*
- [x] Standing throughput and allocation gate met (README, "Test infrastructure").
      *(2026-11-09, owner run, pinned: [`20261109T035315Z-gpu-batched-decode-gate`](../perf-compare/20261109T035315Z-gpu-batched-decode-gate/INDEX.md): GPU pp 0.976x to 1.017x and
      tg 0.963x to 1.011x, CPU pp 0.969x to 1.025x and tg 0.988x to 1.000x, bars 0.95x; allocation per token 0.851x to 1.033x, bar 1.10x; median GC pause 0 ms.)*
- [x] Kit probes for this tier's behaviour are added (64-token batched consistency on the GPU, the
      `--parallel` ratio reading).
      *(2026-11-09: `ConsistencyProbe`'s GPU run adds a batch of eight at 64 tokens; `gpu.parallel.tinyllama` already recorded the ratio
      as `parallel8OverParallel1`; `coverage.tsv` line `feature:batched-gpu-decode`; `ProbeSelfTests`, `CoverageTest`.)*
- [x] Cross-surface checklist fully resolved.
      *(2026-11-09: all 15 rows, in [`20261109T151847Z-gpu-batched-decode-cross-surface`](../perf-compare/20261109T151847Z-gpu-batched-decode-cross-surface/INDEX.md). Row 7 restated by
      **Owner decision**: a pipeline cluster keeps its per-request path, stated at startup. See this date's record "the standing gate and the cross-surface runs".)*
- [x] Docs (`docs/howto.md`, the launcher's `--help`, `docs/agent-arch.txt`, `README.md`) updated in
      Juno-native language; `CHANGELOG.md` entry added.
      *(2026-11-09: `README.md` does not state a `--parallel` recommendation, so it had nothing to change.)*

## Execution record

### 2026-10-31: tier created (owner decision)

- **Owner decision, 2026-10-31**, on the Tier 02F step 6 report: plan a device-resident batched GPU decode
  as its own entry after Tier 02G, not started now.
- **Findings at creation**: as listed under "Why this tier, why now", from the two readings of 2026-10-31
  and the code at `06d6d22d` plus that branch's uncommitted work.

### 2026-11-07: tier begun (owner decision); steps 1 and 2, the pre-tier record

**Owner decision, 2026-11-07.** Tier 02G is postponed until it is run on Windows, and this tier begins while it is open
(README execution rule 1). This tier's place after Tier 02G was sequencing only: none of its work depends on Tier 02G's.

**Plan against code** (HEAD `d8012b11` plus the uncommitted work of 2026-11-05 to 2026-11-07, none of it in the decode
path). Every claim of "Why this tier, why now" holds: `LocalInferencePipeline.forwardBatch` calls each stage's
`forwardMultiDecode`; `LlamaTransformerHandler.runLayersMultiDecode` runs a batched step's projections through
`sgemmLayerInto` and its attention through `CudaGqaAttention.attendBatched`; `GenerationLoop` (static) and
`ContinuousBatchEngine` (continuous) both decode through `forwardBatch`; `Phi3TransformerHandler` and
`Qwen3MoeTransformerHandler` carry their own batch entry points. `compare-parallel.sh`, `compare-schedule.sh` and the kit
probes `gpu.consistency.tinyllama` and `gpu.parallel.tinyllama` exist. No drift.

**Step 1.** `scripts/performance-tests/check-plan-thresholds.sh`: ok (26 tier files).

**Step 2, the pre-tier jar.** `dist/batched-gpu-decode-baseline/`: `baseline-shaded.jar` (sha256 `796de19984b3e9ae`), a
clean build (`mvn clean package -DskipTests`) of HEAD `d8012b11` plus the uncommitted work, which `baseline-tree.patch`
(`git diff HEAD`) and `baseline-untracked.tar.gz` (the untracked files) reproduce on that commit. `BASELINE.txt` says so.
Once the owner commits that work, the commit is the tree.

**Step 2, the `--parallel` reading** (unpinned, as the threshold's method is):

[`20261108T032429Z-gpu-parallel-baseline`](../perf-compare/20261108T032429Z-gpu-parallel-baseline/INDEX.md):
`compare-parallel.sh --gpu --sessions 8`, 64 tokens, P = 1 and 8 alternating, median of three, the pre-tier jar,
2026-11-08 03:24 to 03:45 UTC. Every run succeeded; every P = 8 run is one batch of 8 (server latencies within 19 ms).

| Model | P = 1 t/s | P = 8 t/s | 8 over 1 (bar >= 1.50) | Mean latency 8 over 1 (bar <= 1.00) | P = 8 reply equals P = 1 |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 122.63 | 79.90 | 0.652 | 1.559 | no |
| Qwen2.5-3B | 49.54 | 33.08 | 0.668 | 1.502 | no |
| Phi-3.5-mini | 47.41 | 36.46 | 0.769 | 1.338 | yes |
| Mistral 7B | 32.27 | 23.05 | 0.714 | 1.434 | yes |

The bar is 2.0x to 2.3x above today's ratios. Whether the batched region can reach it on this GPU is what the
decomposition (step 3) reads first; if it cannot, the threshold is restated with an owner decision before the work
continues (item 1). TinyLlama's P = 1 reads higher than on 2026-10-31 (122.63 against 108.66 t/s, another build,
unpinned) and its P = 8 the same (79.90 against 81.23), so its ratio fell from 0.748 to 0.652.

**Step 2, the standing gate's GPU lane on the pre-tier jar** (pinned, owner run 2026-11-08, 16:42 to 16:57 local;
published as [`20261108T224241Z`](../perf-compare/20261108T224241Z/INDEX.md), `clock_pinned` true, exit 0;
log `dist/batched-gpu-decode-baseline/gate-20261108T164241.log`). Juno, median of three cycles (each cycle's value), and
the ratio to the reference engine:

| Model | tg t/s (cycles) | tg ratio | pp t/s (cycles) | pp ratio | Allocation per token in the recording, B | GC pause in token span |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 134.03 (130.67 / 134.03 / 144.62) | 0.716 | 3145.67 (3164.61 / 3145.67 / 2985.63) | 0.766 | 585,286 | 0 ms |
| Qwen2.5-3B | 55.00 (54.80 / 56.87 / 55.00) | 0.795 | 1181.37 (1181.37 / 1170.20 / 1184.60) | 0.804 | 1,835,317 | 0 ms |
| Phi-3.5-mini | 55.45 (55.45 / 55.40 / 55.71) | 0.974 | 830.70 (834.29 / 804.44 / 830.70) | 0.713 | 601,601 | 0 ms |
| Mistral 7B | 37.07 (37.45 / 36.93 / 37.07) | 1.089 | 538.56 (512.86 / 538.56 / 539.99) | 0.865 | 760,248 | 0 ms |

Hot methods (top 10 per model, both histograms, in the log): on every model the Java samples are few (a decode step is
driver downcalls) and led by `SoftmaxStep.apply`, with `LlamaTransformerHandler.transformerLayerBatch` second on Qwen2.5-3B
and Mistral 7B; the native samples are led by `FileInputStream.readBytes` and `EPoll.wait`, idle waits of the harness's
own threads, then the downcall stubs. This is the single-request lane; the batched step is step 3's to decompose.

The sweep's setup, as handed over: `dist/batched-gpu-decode-baseline/
run-gate.sh`: `compare-llama-cpp.sh --gpu --pin-clocks`, the four sweep models, `n_prompt=512`, three measured Juno cycles
and three reference runs per model, published; it prints `clock_pinned`, tg and pp per model with each cycle's value,
allocation per token in the recording, GC pause in the token span, and the top 10 Java and native hot methods. It refuses to
run unless the tree's shaded jar is the baseline jar (a published sweep runs the tree's jar). `wait-and-run.sh` starts it
once `READY` (the jar's hash) exists and no Maven process runs.

**Owed to the owner:** commit the uncommitted work (Tier 02G's Linux side), so the pre-tier tree is a commit.

### 2026-11-08: step 3, the decomposition (item 1)

Published before any change:
[`20261108T230602Z-gpu-batched-decode-decomposition`](../perf-compare/20261108T230602Z-gpu-batched-decode-decomposition/INDEX.md).
One decode step of N requests on TinyLlama 1.1B and Qwen2.5-3B, N = 1, 2, 4 and 8, and 9 and 16 to read the tiled
K-quant GEMM that a wider batch takes; the pre-tier jar (`796de19984b3e9ae`); a CUDA activity trace (Nsight Systems) for
the device terms and an untraced JFR run for the step and the host spans, the method of the 2026-10-11 GPU host-overhead
decomposition extended to a batch (`tools/` in the reading). Unpinned. The scope asked for CUDA event timing; a CUDA trace
times every kernel and copy on the device without adding events to the measured path, and this repository's earlier
decode decompositions read it the same way.

| Model | N | Step ms | t/s | Device-busy ms (kernels) | Host share | Launches / copies / syncs per step |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 1 | 6.92 | 144.6 | 5.55 (5.49) | 20% | 486 / 25 / 23 |
| TinyLlama 1.1B | 8 | 106.26 | 75.3 | 43.95 (39.74) | 59% | 2,502 / 2,942 / 1,240 |
| TinyLlama 1.1B | 9 | 43.74 | 205.7 | 13.18 (10.11) | 70% | 332 / 816 / 156 |
| Qwen2.5-3B | 1 | 17.62 | 56.8 | 14.70 (14.54) | 17% | 902 / 39 / 37 |
| Qwen2.5-3B | 8 | 267.34 | 29.9 | 122.84 (113.67) | 54% | 4,084 / 4,804 / 2,024 |
| Qwen2.5-3B | 9 | 119.97 | 75.0 | 34.14 (27.10) | 72% | 542 / 1,334 / 254 |

(N = 2, 4 and 16, and the host spans per term, are in the reading.)

- **Up to 8 rows, `CudaMatVec.sgemmInto` on packed Q4_K weights runs one GEMV per row** (`HALF_SGEMM_BATCH_MAX` = 8),
  each with its own upload, download and stream sync: kernel time grows with N, every weight is read once per row, and
  per layer at N = 8 a step makes 114 launches and 134 copies (the single region: 22 and 1.1).
- **The host does 54% to 62% of a batched step**: the per-row matmul round trips (`juno.MatVec` 81.80 of TinyLlama's
  106.26 ms at N = 8), the SwiGLU as a scalar host loop (12.56 ms; 40.31 on Qwen2.5-3B), the norms, RoPE and residual
  adds.
- **The tiled K-quant GEMM reads each weight once**: about twice one row's kernel time for 9 or 16 rows (TinyLlama 10.11
  and 11.30 ms against 5.49). A static batch of 9 reads 2.7x (TinyLlama) and 2.5x (Qwen2.5-3B) the aggregate of a batch
  of 8 on this build.

**The threshold is in reach; no threshold moves, no owner decision is needed.** By the threshold's own method, 1.50x
needs a step of 8 requests of at most 32.5 ms (TinyLlama) and 83.7 ms (Qwen2.5-3B), 4.7x the single step. Estimated
from the measured terms (the model reads today's ratio about 15% low, so it errs low): today's per-row kernels with no
host time at all reach 1.19x and 1.10x; the tiled GEMM on today's host-driven path 1.19x and 1.12x; a weight-shared
kernel in a device-resident batched step (device-busy at N = 9 plus the single step's host time) about 2.52x and 2.65x,
with the latency bar met (sessions finish in about 1.66 s and 3.84 s against P = 1's mean of 4.09 and 10.12 s). So
item 2 needs both halves: the region's host-side work and a weight-shared batched K-quant matmul for 2 to 8 rows.

**A constraint for item 4.** A weight-shared kernel sums in another order than the single-row `q4k_gemv`, so its rows
can differ from the single request's in the last bits. Item 2's batched matmul either reproduces the single-row kernel's
per-row arithmetic exactly, or the single step takes the same kernel; "batched equals single" (differing replies <= 0)
is read against whichever single step ships.

**No code changed** in this step, so no test ran.

**Next for the executor:** step 4, tests first: the region-against-host-driven batch test (GPU group, random K-quant
weights, N = 1, 2 and 8, requests at different positions), the batched-against-single test on real weights, the
fail-closed announcement test; then item 2 on the Llama family, starting from the weight-shared batched K-quant matmul.

### 2026-11-09: step 4 (items 2 and 3), tests first, and step 5's readings

**Plan against code.** The decomposition's claims held: `CudaMatVec.sgemmInto` on packed weights ran one GEMV per row up to 8 rows,
and `kq_gemm` sums in a different order from `q4k_gemv`, as its "constraint for item 4" said. No drift.

**The design.** A new kernel module, `batched_decode.cu` (so every PTX the single-request path uses stays byte-identical) and a new
class, `ResidentBatchDecode`, over the single region's weights and kernels (`ResidentQkvPath.Parts`, no second copy). The constraint
is met the first way it names: the batched matmul reproduces the single-row kernel's per-lane accumulation and reduction, so the
single step is unchanged. `q?k_gemv_rows_NC` reads each lane's weight bytes once per super-block for up to 8 activation rows (a wider
batch takes several launches). The rest of a layer needed no new arithmetic: the norm, Q8_1 quantization, SwiGLU and residual adds are
per row or per element, and attention runs one block per row. RoPE takes a per-row position and the KV append writes each row into
its own mirror. Host: one upload of the residual rows (none when the last layer ran in the region), one attention table copy and one
wait per layer, downloading K, V and the layer output.

**Tests, in the order written.**

| Test | First run | Result |
|---|---|---|
| `KQuantGemvRowsTest` (GPU, new): every row of the N-row matmul against one request's GEMV, bit for bit, Q4_K/Q5_K/Q6_K, N up to 17, four shapes | seen failing: `sgemvRowsInto` did not exist (compile error) | 7 pass |
| `ResidentBatchDecodeTest` (GPU, new): every row's K, V and layer output against that request's single region, bit for bit, at different positions over four shapes (plain, Qwen2, Qwen3, Phi-3) and batches of 1, 2, 8 and 9, plus declines and no device allocation per step | seen failing: 15 of 15 on a stub throwing "not implemented" | 15 pass |
| `BatchDecodeRegionHandlerTest` (GPU, new): 64 greedy tokens, every logit of every step against decoding alone, batches of 2 and 8, the four sweep models and Qwen3-1.7B; one batched step against the host-driven batch (N = 1, 2, 8) within the host-norm tolerance | seen failing: 7 of 7 on `batchDecodeRegionActive()` being false | 11 pass |
| `BatchDecodeAnnouncementTest` (new, no GPU): a handler outside the region announces the host-driven batch once | regression-style: written with the wiring; failed once in the full suite for its own reason (an earlier test in the JVM had taken the once-per-surface slot), fixed in the test | 1 pass |
| `ProbeSelfTests#consistencyProbeBatchOfEightFillsABatchFromThePrompts`, `CoverageTest` | written with the probe | pass |

*Deviation from "Tests to write", recorded:* the region-against-host-driven test asked for random K-quant weights. Random weights hold the
stronger equality (`ResidentBatchDecodeTest` against the single region, bit for bit); against the host-driven batch, whose norms run
another path, the comparison runs on real weights at its tolerance.

**Microbench** (`KQuantGemvRowsMicrobenchTest`, prints only, unpinned; ms per matmul, kernels on stream, no copies, Q4_K, batch of 8): the
256-wide model's shapes 0.036 to 0.045 ms (8 single-row GEMVs 0.32 to 0.60, the tiled GEMM 0.067 to 0.085); the 2048-wide model's
2048x2048 0.099 (0.260, 0.081), 8192x2048 0.266 (0.527, 0.171), 2048x8192 0.247 (0.433, 0.184). So it is 1.7x to 2.0x faster than one GEMV per
row at the sizes that bind (5x to 13x on the small ones), and within 1.2x to 1.6x of the tiled GEMM on the large shapes, which it beats on the small ones, while
staying bit for bit.

**The threshold readings** (unpinned, the threshold's method, jar `edf0cea7f8e24b74`; the gate's jar `789c5204c9af15ab` differs only in `--help` text):
[`20261109T005616Z-gpu-batched-decode-parallel`](../perf-compare/20261109T005616Z-gpu-batched-decode-parallel/INDEX.md).

| Model | P = 1 t/s | P = 8 t/s | 8 over 1 (>= 1.50) | Latency 8 over 1 (<= 1.00) | Replies at P = 8 vs P = 1 |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 135.27 | 313.34 | 2.316 | 0.433 | equal |
| Qwen2.5-3B | 54.47 | 107.50 | 1.973 | 0.521 | equal |
| Phi-3.5-mini | 50.08 | 96.01 | 1.917 | 0.533 | equal |
| Mistral 7B | 33.20 | 58.65 | 1.767 | 0.580 | equal |

Batched equals single on the consistency prompts through REST (64 tokens, eight concurrent requests cycling the three prompts): 0 of 8
differ, every sweep model (`rest-consistency/`). The handler test holds the same at the logit level.

**Unit and GPU suite.** `mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player,juno-kit`:
registry, lora, kvcache, health pass; `node` 1,313 tests, 2 failures, 50 skipped (47 min), which skipped the modules after it. The two:
`BatchDecodeAnnouncementTest` (fixed, above; passes alone) and `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` (98,566,144 bytes
free with the device full against <= 67,108,864). That is the failure Tier 02G's record of 2026-11-07 read on this tree and on committed
`d8012b11` alike (84,410,368 there), with the driver's holdback moving with the desktop's GPU use; this work allocates nothing in that test's path. Not loosened.
`PrefillRegionHandlerParityTest`: Qwen2.5-3B's free-VRAM check failed twice (26 MB and 16.7 MB under the start, 16 MB allowed); its other three models pass, every logit bit for bit. A diagnostic loaded and released
Qwen2.5-3B five times with the batch region on and off and held -2.31 to 0.19 MB, so it is not a leak of this work; that model's logit comparison in that test was not reached, and
`BatchDecodeRegionHandlerTest` holds its batched decode. The remaining modules (tokenizer, sampler, coordinator, vision, metrics, juno-player, juno-kit), re-run with `-am` and a package filter so they resolve the build tree's `node` (a run without `-am` resolved a stale `node` jar from the local repository and exercised none of this work): 1,451 tests, 0 failures, 0 errors, 6 skipped.

**Cross-surface, as of this date.**

| # | Surface | State |
|---|---|---|
| 1 | CPU inference | unchanged (the CPU batch path); the CPU lane of the standing gate is the owner's run |
| 2 | CUDA GPU inference | PASS (tests above; `20261109T005616Z-gpu-batched-decode-parallel`) |
| 3 | ROCm GPU inference | FAIL-CLOSED: the region is CUDA-only (`ResidentQkvPath.unsupportedReason`), so ROCm keeps the host-driven batch; NEEDS-AMD-HARDWARE for a kit run |
| 4 | Static schedule | PASS (`20261109T005616Z-gpu-batched-decode-parallel`) |
| 5 | Continuous schedule | the same `forwardMultiDecode` path; `compare-schedule.sh --gpu` reading OWED |
| 6 | Single-node local mode | PASS |
| 7, 8 | Pipeline and tensor-parallel cluster | OWED (`mvn verify -pl juno-master -Pgpu`, live checks 1 to 8); the region is per node and inherits the single region's eligibility |
| 9, 11 | LoRA training, vision | N/A: CPU float kernels (LoRA training), moondream2 has no device decode path |
| 10 | LoRA playback | OWED: with `--lora-play` the handler is `LoraTrainableHandler`, which does not reach the region; to be confirmed with a batch |
| 12, 13 | OpenAI and native REST | PASS for OpenAI (`20261109T005616Z-gpu-batched-decode-parallel`, `rest-consistency/`); native covered by the kit's `gpu.consistency.tinyllama`, not run on this build |
| 14 | CLI | PASS (`--help` text) |
| 15 | JVM embedding facade | kit's consistency probe (`JunoHttpClient`); OWED on this build |

**Out-of-tier changes:** none. Inside the tier: `LlamaTransformerHandler.releaseGpuResources` now also closes `residentQkv` and the batch region (Phi-3 and Qwen3 already closed theirs).

**Owed to the owner:** the pinned standing gate: `bash dist/batched-gpu-decode-gate/wait-and-run.sh` (asks for sudo once, about 2.5 hours; pre-tier jar against the candidate on both lanes, four sweep models). Once it has run, the
evidence goes here and the "Standing gate" criterion closes. Then the cross-surface runs above.

**Next for the executor:** the cross-surface runs (`mvn verify -pl juno-master -Pgpu`, `compare-schedule.sh --gpu`, `--lora-play` with a batch, the kit's GPU probes), `compare-lora.sh` on this build, then close the tier.

### 2026-11-09: the standing gate and the cross-surface runs

**Standing gate** (owner run, pinned, pre-tier `796de19984b3e9ae` against `789c5204c9af15ab`, A/B x3 on the four sweep models, `GATES MET`):
[`20261109T035315Z-gpu-batched-decode-gate`](../perf-compare/20261109T035315Z-gpu-batched-decode-gate/INDEX.md). The ratios, recomputed from the per-run JSON, agree with the script.

| Lane | pp B/A (TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B), bar 0.95 | tg B/A, bar 0.95 |
|---|---|---|
| GPU, `n_prompt` 512 | 1.013, 1.017, 0.982, 0.976 | 1.011, 1.007, 0.993, 0.963 |
| CPU, `n_prompt` 128 | 1.025, 0.989, 1.014, 0.969 | 0.992, 1.000, 0.988, 0.995 |

Allocation per token 0.851x to 1.033x (bar 1.10x); median GC pause 0 ms; clocks pinned in 12 of 12 runs; greedy output equal in 3 of 3 on every model of both lanes. The
lowest ratio, Mistral 7B's GPU tg at 0.963, is inside the baseline's own 38.33 to 40.08 t/s spread. Hot methods (reported, not gated): on the CPU lane the same methods lead on both jars
(`CpuIntegerKernels.q4kRow`/`q5kRow`, the worker pool, `Objects.checkIndex`, `CpuIntegerGemm.fourRows32`) within run-to-run spread; on the GPU lane Java samples are few (at most 15 for
any method), led by `SoftmaxStep.apply`.

**The final jar stands on that gate.** Against the gated jar, the final build differs only in `ConsoleMain`, `ConsoleMain$2`, `JunoPlayer$Builder` and two new statement classes,
by CRC of all 18,700 entries: startup text and `--help`, nothing on a hot path. (`juno-kit.sh --help` builds the tree first; run once by mistake while no test was running, it rebuilt
the tree's jar with identical entries. The gate's jars in `dist/` were untouched.)

**Two surfaces that did not say what they do, fixed.** Both keep their behaviour and now state it at startup, per item 3's "nothing falls back silently".

- Row 10: with `--lora-play` a batch decodes one request at a time, on the host. `LoraPlaybackBatchNotice`, in `ConsoleMain` and `JunoPlayer`.
- Rows 7 and 8: a cluster decodes a batch one request at a time on each node. `ClusterBatchNotice` (coordinator), in `ConsoleMain`'s cluster path and `CoordinatorMain`. `--help` and `docs/howto.md` now recommend `8` for a local server and `1` on a cluster.

**Drift between the plan and the code, one row.** Row 7 says "each node's stage runs its layers' batched step in the region". Nothing does: `LocalInferencePipeline` is the only production caller of
`forwardMultiDecode`, `ProcessPipelineClient` and `TensorParallelPipelineClient` inherit `InferencePipeline.forwardBatch`'s serial default, and the `ForwardPass` RPC carries one request. That was already so before this tier. Row 8
allows it ("keeps its current path"); row 7 as written did not, hence the decision below.

| Cluster, TinyLlama, GPU, 8 concurrent clients, 64 tokens | `--parallel` 8 over 1, t/s (pre-tier) | t/s (this build) | Mean latency, 8 over 1 (this build) | Replies differing at 8 |
|---|---|---|---|---|
| pipeline | 0.314 | 0.383 | 2.64 | 0 of 8, both sessions |
| tensor | 0.815 | 0.813 | 1.25 | 0 of 8, both sessions |

So a batch on a cluster was slower than `--parallel 1` before this tier and still is; why was not decomposed. With one stream per request (`--parallel 1`), this build against pre-tier, the jars alternating, reads 1.000x pipeline and 0.999x tensor.

**Cross-surface, final.** Evidence: [`20261109T151847Z-gpu-batched-decode-cross-surface`](../perf-compare/20261109T151847Z-gpu-batched-decode-cross-surface/INDEX.md) unless another is named.

| # | Surface | State |
|---|---|---|
| 1 | CPU inference | PASS: the standing gate's CPU lane; kit `consistency.tinyllama`, `parallel.tinyllama` (`--parallel` 8 over 1 1.17x) |
| 2 | CUDA GPU inference | PASS: `20261109T005616Z-gpu-batched-decode-parallel`, the standing gate |
| 3 | ROCm GPU inference | FAIL-CLOSED: the region is CUDA-only (`ResidentQkvPath.unsupportedReason`), so ROCm keeps the host-driven batch. `ResidentQkvPathTest` asserts the refusal with a null context, the same branch a non-CUDA backend takes; no ROCm-labelled context is tested. NEEDS-AMD-HARDWARE for a kit run |
| 4 | Static schedule | PASS: `20261109T005616Z-gpu-batched-decode-parallel` |
| 5 | Continuous schedule | PASS: `compare-schedule.sh --gpu` (continuous over static 0.967x aggregate, a reading; largest decode batch 8), kit `gpu.schedule.tinyllama`, `gpu.mixed-prefill.tinyllama` |
| 6 | Single-node local mode | PASS |
| 7 | Pipeline-parallel cluster | PASS on the row as restated (**Owner decision** below): keeps its current path, stated at startup (`ClusterBatchNoticeTest`); live checks 1 to 6 pass on five models (kit `gpu.live.*`), stub integration tests pass, replies equal |
| 8 | Tensor-parallel cluster | PASS on the row's own terms: keeps its current path, stated at startup; live checks 7 and 8 pass on five models; replies equal |
| 9 | LoRA training | PASS: CPU float kernels unchanged; `compare-lora.sh` train 1.000x (bar 1.25x), recall ok |
| 10 | LoRA playback | PASS: keeps the serial batch, stated at startup (`LoraPlaybackBatchNoticeTest`); 0 of 8 batched replies differ; playback 1.013x (bar 0.80x) |
| 11 | Vision | N/A: moondream2 has no device decode path; kit `gpu.vision.moondream2` passes |
| 12, 13 | OpenAI and native REST | PASS: `rest-consistency/` of `20261109T005616Z-gpu-batched-decode-parallel`, kit `gpu.consistency.tinyllama` |
| 14 | CLI | PASS: `--help` |
| 15 | JVM embedding facade | PASS: `JunoHttpClient` in kit `gpu.consistency.tinyllama` |

**Kit, 15 probes.** 12 pass, 2 are readings (`gpu.parallel.tinyllama` 2.67x, `parallel.tinyllama` 1.17x), and `gpu.unit-tests` fails: 405 tests, 4 failures. None of the four comes from this work; none was loosened.
`PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` is the failure Tier 02G's record of 2026-11-07 read on the committed tree. The other three are the device-wide free-memory check in `GpuAttentionHandlerParityTest` and
`PrefillRegionHandlerParityTest`. Five repeat runs of those two classes with every process's GPU memory sampled once a second: the test JVM returns to 118 to 122 MiB after every case while the desktop's own use moves by up to about 1 GiB; which cases miss changes
from run to run; run 5 passes all six, every logit comparison reached.

**README execution rule 10, not met to the letter.** `gpu.unit-tests` is on this tier's keep-passing list, and the rule says such a probe passes on the tier's build before the tier closes. On this host it does not: `PrefillReserveDeviceTest` fails here on the committed tree as well, and the
free-memory check depends on what the desktop does with the GPU. The tier closed on the owner's decision (a) below, taken with these failures reported; the owner has not ruled on rule 10 separately. A run of this probe on a host with no desktop on the GPU has not been made.

**Tests of this part.**

| Test | First run | Result |
|---|---|---|
| `LoraPlaybackBatchNoticeTest` (new, no GPU): when the statement applies; all four LoRA handlers keep the serial default, so the statement stays true | seen failing: class absent (compile error) | 5 pass |
| `ClusterBatchNoticeTest` (new, no GPU): both cluster clients keep the serial `forwardBatch`, so the statement stays true | seen failing: class absent (compile error) | 2 pass |

Last code change to the end: `coordinator` 402 tests, 0 failures, 1 skipped; `juno-player` 151, 0 failures, 2 skipped; `juno-master` 13 unit tests and 20 stub integration tests, 0 failures. Earlier on this work's code: `mvn verify -pl juno-master -Pgpu` 10 tests, 0 failures. `CoordinatorMain`'s
statement was compiled and its text confirmed in the jar; that standalone coordinator was not started on its own.

**Out-of-tier changes:** none.

**Owner decision (2026-11-09), row 7.** Put to the owner with the readings above: (a) restate row 7 as row 8 reads, "keeps its current path, stated at startup", which closes this tier and leaves a batched cluster decode as later work; or (b) require a batched step on each node, which needs a new `ForwardPass` RPC carrying N activations and each
node's stage calling the batched decode, and keeps this tier open. Recommended: (a), because the tier's objective and every threshold are about the local server, where they are met, and (b) changes the wire protocol with no reading yet showing what it would gain. **The owner chose (a).** A batched cluster decode is later work, not scheduled; it would need its own threshold.

**Owed to the owner:** commit this work. ROCm stays NEEDS-AMD-HARDWARE until a tester's kit run on an AMD card.

**Next for the executor:** nothing in this tier. In running order, Tier 02F waits on testers' zips and Tier 02G on the owner's Windows runs (its steps 2 and 3); no decision yet lets Tier 03 begin while they are open.
