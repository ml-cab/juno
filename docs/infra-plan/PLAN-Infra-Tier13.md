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
| **Blocks** | Interactive latency peer claims (13B); gather-tax escape hatch (separate memo only) |
| **Parallel with** | P0 steps 2–4 for 13B prep |

Phase A **complete** (2026-08-31 bake-off JFR; record final memo in `docs/performance.md`). **Go** for Phase B scoped to fused Q4 MMQ first — not FlashAttn (P5).

**Phase B status (in progress):** `--mmq on|off|auto` / `JUNO_MMQ` (default off); CUDA Driver API + classpath PTX `q4k_gemv.ptx`; `DeviceQ4KMatrix` + `Q4KMmqKernel`; wired in `LlamaTransformerHandler` for Q4_K projections; `Q4KMmqParityTest` green. Bake-off + perf gate (≥1.3× tg) still open before feature complete.

**Architecture follow-up (before 13B exit):** Phi-2 / Phi-3 / Qwen3 handlers still use FP16-resident upload; extend the same Q4_K packed path (shared upload helper preferred) per ROADMAP §5.

**LoRA adjacency (plan ready, not started):** `--lora-play` / `LoraTrainableHandler` still dequant→FP16 via `LoraResidentWeights` and ignores `--mmq`. Wiring plan (risk register; recommended playback-only MMQ): [`PLAN-Infra-LoRA-MMQ.md`](PLAN-Infra-LoRA-MMQ.md) (from [`PROMPT-LoRA-MMQ.md`](PROMPT-LoRA-MMQ.md)). Not part of Tier 13B exit; implement only after that plan is reviewed.

## Feature × surface interaction matrix (`--mmq`)

Per ROADMAP **§6**. Cells filled for Phase B current state:

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | CUDA | ROCm | Default |
|--------------------|----------------|-------------|------------|--------|------------|--------------|-----------------|------|------|---------|
| `--mmq` | **wired** (Llama-family Q4_K) | **follow-up** → [`PLAN-Infra-LoRA-MMQ.md`](PLAN-Infra-LoRA-MMQ.md) | **explicit no-op** (train must stay FP16/FP32 until transpose policy) — **warn TODO** | N/A (text MatVec only) | **wired** if decode uses same Llama MMQ projections | **wired** with partial offload (Q4 upload only for resident layers) | **wired** (batched path uses Q4 `sgemm` serial GEMVs) | **wired** | **explicit no-op** (`supportsQ4KMmq` false) | **off** |

Until LoRA follow-up lands and warn-on-ignore ships: do not claim `--mmq` accelerates `--lora-play` in user-facing docs.

## Overview

Flash Attention and fused quantized matmul (MMQ) are central to llama.cpp speed. Juno’s strategy is Panama + vendor BLAS first. This tier closes the **measured** remaining gap — or formally closes the tier as unnecessary.

**Phase A status (2026-08-31):** Bake-off JFR on GTX 1080 shows `juno.MatVec` is **93–96%** of GPU decode time on resident models (Phi-3.5, Qwen2.5-3B, TinyLlama). **Go** for Phase B scoped to **fused Q4_K MMQ / batched decode GEMV** — not FlashAttn first (decode seq len = 1; FA wins on long prefill after Tier 8). Full memo: [`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md).

## Scope and compatibility

Goals:

1. **Phase A (mandatory):** Profile decode/prefill. **Preliminary memo complete** — see [`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md). Record final section in `docs/performance.md`.
2. **Phase B (only if go):** Minimal acceleration — either FlashAttn for GQA decode via a small native lib + Panama or a vendor library, **or** a narrow fused quant matmul — not a full ggml port.
3. Parity tests if anything ships.

Non-goals:

- Porting the full ggml FlashAttn / MMQ stacks.
- Doing FlashAttn and fused MMQ in one blind push (separate go/no-go).
- **Page-native / PagedAttention kernels as the default gather-tax mitigation** for Tier 14 — dual KV + larger pages first; page-aware attention only via a **separate** go memo after Tier 14 microbench.
- Metal / Vulkan backends.
- Claiming wins without JFR evidence.

## Chosen design

Go criteria for Phase B (example bar; adjust only with roadmap amendment):

- Attention or dequant+GEMM accounts for **>40%** of decode time on the declared mid-size model bench after prior infra tiers; and
- No further BLAS / batching / KV / offload tweak is an obvious cheaper fix.

If go (preliminary: **yes** for MMQ):

- **First target:** fused Q4_K dequant+GEMV on `cuda_resident_fp16` path — closes the 4–6× single-stream decode gap vs llama.cpp.
- **Second target (after Tier 8):** FlashAttn for long prefill / context if profiling shows attention >40% on 512+ token prompts.
- Prefer the smallest ABI-stable native surface.
- Keep CPU and non-FA paths as correctness oracles.

If no-go:

- Mark Tier 13 deferred/closed in the ROADMAP with the memo link; no production code required.

## Implementation

### Phase A

1. ~~Collect JFR / timers on TinyLlama and one mid-size model (Phi-3.5 or Qwen).~~ **Done** — see [`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md).
2. ~~Break down matVec, attention, softmax, transfer, sampling.~~ MatVec **>90%** decode; attention secondary on decode bench.
3. ~~Write memo: go or no-go with numbers.~~ **Go for MMQ**; FlashAttn deferred to post–Tier 8.
4. Record final memo section in `docs/performance.md` when that file is created.

### Phase B (conditional)

1. Prototype behind a flag (`--mmq on|off|auto` / `JUNO_MMQ`) — **landed** (default off).
2. Parity vs oracle path — **landed** (`Q4KMmqParityTest`).
3. Perf gate: ≥1.3× decode TPS **or** ≥1.5× long-context prefill on the declared bench — **open**.
4. Docs and ROADMAP — **in progress** (howto / performance / ROADMAP updated; bake-off pending).

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.


Exit only when **one of**:

1. **Ship:** flag-gated acceleration meets the perf gate with parity tests and docs; or
2. **Close:** Phase A memo shows BLAS path sufficient; ROADMAP marks Tier 13 deferred/closed without shipping kernels.

## Implementation todos

1. Phase A profile + performance.md memo (mandatory).
2. If go: minimal prototype + parity + perf gate.
3. If no-go: ROADMAP deferral note only.
4. Preview files; no zip.

## Preview files (expected)

Phase A: `docs/performance.md` (memo section), possibly scripts under `scripts/performance-tests/`

Phase B (only if go): native shim / Panama bindings, handler attention path, CLI flag, tests, ROADMAP status
