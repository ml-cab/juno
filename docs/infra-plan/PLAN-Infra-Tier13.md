# Tier 13: Flash Attention / Fused Quant (Gated)

## Agent handoff

Read and follow `models/CLAUDE.md` before implementing:

1. Unit tests first, only for valuable business logic.
2. Implementation details designed with performance in mind.
3. Follow KISS.
4. Prefer adding new Java classes over extending existing ones.
5. Update `docs/agent-arch.txt`, `docs/howto.md`, `README.md` when applicable.
6. No emojis; be strict and precise.
7. Output: list changed files for preview; never zip files back.

Also read:

- `PLAN-Infra-ROADMAP.md`
- [`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md) — **Phase A preliminary memo (go for MMQ)**
- `PLAN-Infra-Tier5.md`, `PLAN-Infra-Tier6.md`, `PLAN-Infra-Tier8.md` (baselines)
- `docs/performance.md` and recent JFR profiles
- `GpuMatVec` / attention hot paths in handlers

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P0 step 1 (13A ✓) / P0 step 5 (13B MMQ) / P5 (FlashAttn subset) |
| **Exec step** | 13A complete; 13B after P0 step 3; FlashAttn after Tier 8 baselines |
| **Depends on** | Phase A: none (complete). Phase B MMQ: Phase A go memo. FlashAttn: Tier 8 |
| **Blocks** | Interactive latency peer claims until Phi-3.5 P0 0.5× (tile-kernel 1.3× **met**; next lever is fuller device-resident activations) |
| **Parallel with** | P0 steps 2–4 for 13B prep |

Phase A **complete** (2026-08-31 bake-off JFR; record final memo in `docs/performance.md`). **Go** for Phase B scoped to fused Q4 MMQ first — not FlashAttn (P5).

**Phase B status (feature complete — VRAM-fit ship):** `--mmq on|off|auto` / `JUNO_MMQ` (default **off**); CUDA Driver API + classpath PTX `q4k_gemv.ptx`; `DeviceQ4KMatrix` + `Q4KMmqKernel`; shared `Q4KResidentUpload`; wired in `LlamaTransformerHandler`, `Phi3TransformerHandler` (fused QKV/gate_up = one GEMV + host slice), and `Qwen3TransformerHandler` for Q4_K projections; parity tests green. LoRA play Phase 1 **complete**. Bake-off [`20260910T025804Z`](../perf-compare/20260910T025804Z/).

**Tile-kernel follow-on (2026-09-11):** Q8_1 activation + `dp4a` integer-dot GEMV landed (`q4k_gemv.cu` / `quantize_q8_1`). Bake-off [`20260911T235203Z`](../perf-compare/20260911T235203Z/) + pair [`20260911T235353Z`](../perf-compare/20260911T235353Z/): Phi-3.5 `--mmq on` **19.34** tg vs `--mmq off` **12.83** (**1.51×**, ≥1.3× **met**). Mistral packed-Q4 **0.43×** llama (P0 0.15× **met**). Phi-3.5 vs peer **0.33×** (P0 0.5× **unmet**). Default remains **off**. User-facing docs may claim a measured speed win vs `--mmq off` on CUDA — not peer latency.

**Follow-ons (not blocking 13B feature-complete):**

1. **Shared-activation GEMV (Phase 1 landed)** — `MatVec.sgemvSameX` uploads `x` once and coalesces sync for Q/K/V and gate/up on Llama decode (`CudaMatVec` / `RocmMatVec`). Full device-resident hidden-state chain (norm / attn / residual on GPU) remains future work.
2. **Tile Q4_K kernel** — **landed** (Q8_1 + `dp4a`; ≥1.3× vs `--mmq off` on Phi-3.5).
3. Phi-2 / Qwen3-MoE GPU residency; optional Phi fused Q4 split like FP16.

**LoRA adjacency (Phase 1 complete):** [`PLAN-Infra-LoRA-MMQ.md`](PLAN-Infra-LoRA-MMQ.md). Not part of Tier 13B exit.

## Feature × surface interaction matrix (`--mmq`)

Per ROADMAP **§6**. Cells filled for Phase B current state:

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | CUDA | ROCm | Default |
|--------------------|----------------|-------------|------------|--------|------------|--------------|-----------------|------|------|---------|
| `--mmq` | **wired** (Llama-family + Phi-3 + Qwen3 dense Q4_K) | **wired** (Phase 1: `LoraTrainableHandler` / Qwen2) → [`PLAN-Infra-LoRA-MMQ.md`](PLAN-Infra-LoRA-MMQ.md) | **explicit no-op** + warn (train stays FP16/FP32) | N/A (text MatVec only) | **wired** if decode uses same MMQ projections | **wired** with partial offload (Q4 upload only for resident layers) | **wired** (batched path uses Q4 `sgemm` serial GEMVs) | **wired** | **explicit no-op** (`supportsQ4KMmq` false) | **off** |

Phi-2 / Qwen3-MoE: **follow-up** (no GPU residency path yet). User-facing docs may claim `--mmq` for **VRAM fit** and a **measured decode-throughput win vs `--mmq off`** on Llama-family, Phi-3, and Qwen3 dense text inference plus LoRA playback — **not** peer latency (P0 0.5× still open).

## Overview

Flash Attention and fused quantized matmul (MMQ) are central to peer engine speed. Juno’s strategy is Panama + vendor BLAS first. Phase B ships packed Q4 residency for fit; the tile-kernel speed exit vs `--mmq off` is **met**. Peer 0.5× decode remains open.

**Phase A status (2026-08-31):** Bake-off JFR on GTX 1080 shows `juno.MatVec` is **93–96%** of GPU decode time on resident models. **Go** for Phase B scoped to **fused Q4_K MMQ** — not FlashAttn first. Full memo: [`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md).

## Scope and compatibility

Goals:

1. **Phase A (mandatory):** Profile decode/prefill. **Done.**
2. **Phase B:** Flag-gated fused Q4 path with parity tests; ship as VRAM-fit (amended gate).
3. Parity tests if anything ships.

Non-goals:

- Porting the full ggml FlashAttn / MMQ stacks.
- Claiming MMQ decode TPS wins vs FP16-resident without a measured tile-kernel bake-off.
- Metal / Vulkan backends.

## Chosen design

Go criteria for Phase B (met for prototype): MatVec **>40%** of decode → fused Q4 on resident path.

**Ship bar:** wiring + parity + published bake-off + honest docs. VRAM-fit **met**; tile-kernel ≥1.3× vs `--mmq off` **met** (Phi-3.5 **1.51×**). 13B stays **feature complete**; P0 0.5× peer is a program gate, not a 13B exit.

## Implementation

### Phase A

1. ~~Profile + memo~~ **Done** — [`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md).

### Phase B (conditional)

1. Prototype behind a flag — **landed** (default off).
2. Parity vs oracle — **landed**.
3. Perf gate ≥1.3× vs `--mmq off` — **met** (Phi-3.5 **1.51×**, 2026-09-11).
4. Docs and ROADMAP — **updated** for VRAM-fit + measured CUDA decode win vs `--mmq off` (not peer latency).

### Shared-activation Phase 1

- `MatVec.sgemvSameX` + CUDA/ROCm overrides; Llama `transformerLayer` / `ffn` use it for QKV and gate/up when all slots share a resident backend.
- Parity: `SgemvSameXParityTest`.

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.

Exit (Phase B MMQ — **met under amendment**):

1. **Ship (VRAM-fit):** flag-gated packed Q4 path with parity tests, bake-off artifacts, and docs that do **not** claim speed vs FP16-resident; or
2. **Close:** Phase A memo shows BLAS path sufficient (not used — prototype shipped).

**Deferred speed exit (tile kernel follow-on):** ≥1.3× decode TPS vs `--mmq off` on Phi-3.5 or TinyLlama, or ≥1.5× long-context prefill, with a new bake-off row. **Met** — Phi-3.5 **1.51×** ([`20260911T235203Z`](../perf-compare/20260911T235203Z/) vs [`20260911T235353Z`](../perf-compare/20260911T235353Z/)).

## Implementation todos

1. ~~Phase A~~
2. ~~Phase B prototype + parity + amended ship~~
3. Shared-activation Phase 1 — **landed** (Llama); extend other handlers as follow-up
4. Tile Q4 kernel — **landed** (2026-09-11 bake-off)
5. Preview files; no zip

## Preview files (expected)

Phase B: PTX / Panama bindings, handlers, CLI flag, tests, ROADMAP / howto / performance notes

## Phase C — GPU-resident attention (`--gpu-attention`)

**Motivation:** the new `juno.Attention` JFR span (added during the Phase B/Tier 17 follow-on,
see [`docs/performance.md`](../performance.md) "Prefill microbatching") showed attention —
not the GEMM/MatVec path Tier 17 fixed — is the dominant GPU decode/prefill cost at realistic
context length: **78.3%** of prefill wall time and **64.2%** of decode wall time at ~512 tokens
context (TinyLlama, GTX 1080). Attention (`gqaInto`/`gqa`: QK^T + softmax + weighted-V-sum) ran
entirely on scalar CPU Java even when the model's weights were fully GPU-resident. Phase C moves
attention onto the GPU.

**Design decision:** v1 is a straightforward parallel kernel (materialize the score row, softmax,
weighted-V-sum, parallelized across GPU threads) — one block per (batch-row, head), 3-pass
(QK^T+max, softmax, weighted-V-sum) — **not** a tiled/online-softmax FlashAttention-2 kernel. The
fuller design remains the P5 FlashAttn follow-on (gated on Tier 8 baselines per the ROADMAP);
Phase C does not change that gate, it is additional evidence for it.

**Scope:** `LlamaTransformerHandler` only (`llama`/`mistral`/`qwen2` GGUF architectures). CUDA
only. `Phi2TransformerHandler`/`Phi3TransformerHandler`/`Qwen3TransformerHandler`/
`Qwen3MoeTransformerHandler`, ROCm, and `--lora-play`/LoRA train are named follow-ups (each owns
its own attention math / KV map — see `LoraTrainableHandler`), matching how Phase B's MMQ rollout
handled the same handlers. Vision is wired automatically (no vision-specific code) since
`VisionAwareForwardPassHandler` delegates every forward call to an internal
`LlamaTransformerHandler`.

**Status: feature complete.** `--gpu-attention on|off|auto` / `JUNO_GPU_ATTENTION` (default
**off**); CUDA Driver API + classpath PTX `gqa_attention.ptx`; `DeviceKvCache` (device-resident FP16
KV mirror, dual-write alongside host `SessionKvTensor`, grow-and-preserve via D2D copy) +
`GqaAttentionKernel` (PTX loader/launcher) + `CudaGqaAttention` (handler-facing
`attendBatched(...)`, batched-pointer design serving prefill window / single decode / `--parallel`
multi-decode in one launch); `GqaMath` (attention math extracted from `LlamaTransformerHandler`,
zero behavior change, also the parity-test CPU oracle); wired into all three
`LlamaTransformerHandler` attention call sites (prefill batch, single decode, multi-decode) with
fallback to `GqaMath` when GPU dispatch returns `false`. CLI flag mirrors `--mmq` exactly in
`ConsoleMain`, `scripts/run.sh`, and `scripts/performance-tests/compare-llama-cpp.sh` (+
`compare-prefill-batch.sh` passthrough for its own bake-off). Bake-off:
[`20260916T035952Z-prefill`](../perf-compare/20260916T035952Z-prefill/) (off) /
[`20260916T040113Z-prefill`](../perf-compare/20260916T040113Z-prefill/) (on); regression gate
[`20260916T034621Z`](../perf-compare/20260916T034621Z/) /
[`20260916T035124Z`](../perf-compare/20260916T035124Z/); LoRA regression
[`20260916T035640Z-lora`](../perf-compare/20260916T035640Z-lora/). Full write-up:
[`docs/perf-compare/README.md`](../perf-compare/README.md) → "GPU-resident attention bake-off —
Tier 13 Phase C".

**Speed exit — met.** At `--prefill-batch 32` (the window size where `juno.Attention` events are
classified as `prefill`, not folded into decode accounting at window=1): pp throughput
**31.04 -> 119.56 t/s (3.85x)**, attention's share of prefill wall time **78.7% -> 11.0%**
(`compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32`, TinyLlama Q4_K_M, GTX
1080). Standing regression gate (short default prompt, 4-model set) stays flat to modestly
improved, never regressed — expected, since attention's share of cost is naturally small at short
context; the dedicated long-prompt repro above is this feature's real signal, same honesty
standard as `--mmq`'s "measured decode-throughput win, not peer latency" framing.

**Known limitation (documented in `DeviceKvCache` javadoc, not a defect):** multi-token
greedy-decode sequences can occasionally diverge between `--gpu-attention on` and `off` on some
prompts after 15+ tokens — FP16 KV rounding occasionally flips a close greedy decision, same class
of behavior already accepted for `--mmq` and other reduced-precision paths in this codebase.
Single-step logits match tightly (parity test). Not pursued further — bit-identical multi-step
generation is not the bar here or anywhere else in this codebase.

### Feature × surface interaction matrix (`--gpu-attention`)

Per ROADMAP **§6**.

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | --mmq | CUDA | ROCm | Default |
|--------------------|----------------|-------------|------------|--------|------------|--------------|-----------------|-------|------|------|---------|
| `--gpu-attention` | **wired** (Llama-family/Mistral/Qwen2); Phi-2/Phi-3/Qwen3/Qwen3-MoE **follow-up** | **explicit no-op + warn** (separate handler class, own KV map/attention math) | **explicit no-op + warn** | **wired automatically** (delegates to internal `LlamaTransformerHandler`) | **wired** (per-stream device pointers, one batched launch) | **wired** — activates only for GPU-resident layers, scalar fallback below cutover | **wired** | **wired**, orthogonal (attention only ever consumes host `float[]` Q/K/V regardless of which device dtype produced it) | **wired** | **follow-up** — no kernel; `CudaGqaAttention.tryCreate` returns `null` on non-CUDA backends, falls back to scalar CPU, never silent (logged at handler construction) | **off** |

### Cross-feature smoke (before feature complete)

- [x] Base inference **wired**: `LlamaTransformerHandlerGpuAttentionLiveTest` (real GGUF + CUDA)
      proves `gpuAttentionActive()` true and greedy-token parity; JFR `cuda-resident-*` /
      `gpuAttentionActive()` log line names the active policy.
- [x] `--lora-play` / LoRA train **explicit no-op + warn**: `LoraTrainableHandler.
      warnIfGpuAttentionIgnored()` logs once and records `LoraTrainNotices.GPU_ATTENTION_IGNORED`
      when `--gpu-attention` is preferred outside the base handler.
- [x] Vision **wired automatically**: no vision-specific code — delegation via
      `VisionAwareForwardPassHandler` inherited from Phase B's same pattern; not independently
      re-verified in this stage (no vision-specific bake-off required per this stage's exit
      criteria — vision shares `LlamaTransformerHandler`'s code path with no divergent call site).
- [x] `--parallel` **wired**: batched-pointer `attendBatched(...)` design serves multi-decode in
      one launch; covered by the existing multi-decode parity tests.
- [x] `--gpu-layers` **wired**: `layerGpuResident(...)` gate in `LlamaTransformerHandler` — only
      GPU-resident layers get a `DeviceKvCache` entry, non-resident layers fall to `GqaMath`.
- [x] `--prefill-batch` **wired**: prefill batch call site builds a `B`-sized batch and calls
      `attendBatched`, same as single/multi-decode call sites.
- [x] `--mmq` **wired**, orthogonal: confirmed by code inspection — attention consumes host
      `float[]` Q/K/V regardless of which device dtype (FP16/Q4_K) produced them.
- [x] CUDA **wired**; ROCm **follow-up**: `CudaGqaAttention.tryCreate` returns `null` on non-CUDA
      backends (checked in `GpuBindings`/`GpuContext.selectBindings()` dispatch), logged, falls
      back to `GqaMath`.
- [x] §2 compares run: inference regression (off/on), LoRA regression, and the dedicated
      `compare-prefill-batch.sh` bake-off — see run links above. Vision compare not required this
      stage (no MatVec/vision-specific code changed; delegates unchanged).

### Exit checklist (compatibility)

- [x] Interaction matrix complete (no empty cells)
- [x] No silent flag ignore on any surface that accepts the flag in the launcher
- [x] Launcher (`scripts/run.sh`) forwards `--gpu-attention` for the command mode that honors it
      (`local`; `cluster`/`lora` share `ConsoleMain`'s CLI parser and env-var fallback, matching
      exactly how `--mmq` is scoped — `--mmq` itself is only CLI-wired in `run.sh`'s `local`
      subcommand today)
- [x] User-facing docs (`docs/howto.md`) state which modes honor the feature and which
      architectures/surfaces are follow-ups
- [x] ROADMAP §5 architectures covered (Llama-family) or named follow-up (Phi-2/Phi-3/Qwen3/
      Qwen3-MoE)
