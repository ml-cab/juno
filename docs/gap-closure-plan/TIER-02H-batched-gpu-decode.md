# Tier 02H: Batched GPU decode

Status: not started (added 2026-10-31 by owner decision; runs after Tier 02G)
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
| 7 | Pipeline-parallel cluster | each node's stage runs its layers' batched step in the region; live checks 1 to 6 |
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

- [ ] The decomposition (item 1) is published before the first change, and any threshold it moved carries
      an **Owner decision**.
- [ ] Every handler whose single-request decode runs in the device region decodes a batch there; the others
      announce the host-driven batch at startup.
- [ ] Every threshold above is met, with the run directories cited. A miss keeps the tier open; closing on
      one takes an explicit **Owner decision** recorded here with the measured number.
- [ ] Standing throughput and allocation gate met (README, "Test infrastructure").
- [ ] Kit probes for this tier's behaviour are added (64-token batched consistency on the GPU, the
      `--parallel` ratio reading).
- [ ] Cross-surface checklist fully resolved.
- [ ] Docs (`docs/howto.md`, the launcher's `--help`, `docs/agent-arch.txt`, `README.md`) updated in
      Juno-native language; `CHANGELOG.md` entry added.

## Execution record

### 2026-10-31: tier created (owner decision)

- **Owner decision, 2026-10-31**, on the Tier 02F step 6 report: plan a device-resident batched GPU decode
  as its own entry after Tier 02G, not started now.
- **Findings at creation**: as listed under "Why this tier, why now", from the two readings of 2026-10-31
  and the code at `06d6d22d` plus that branch's uncommitted work.
