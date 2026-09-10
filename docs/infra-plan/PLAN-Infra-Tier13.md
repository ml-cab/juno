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
| **Blocks** | Interactive latency peer claims until tile kernel / activation residency land |
| **Parallel with** | P0 steps 2–4 for 13B prep |

Phase A **complete** (2026-08-31 bake-off JFR; record final memo in `docs/performance.md`). **Go** for Phase B scoped to fused Q4 MMQ first — not FlashAttn (P5).

**Phase B status (feature complete — VRAM-fit ship):** `--mmq on|off|auto` / `JUNO_MMQ` (default **off**); CUDA Driver API + classpath PTX `q4k_gemv.ptx`; `DeviceQ4KMatrix` + `Q4KMmqKernel`; shared `Q4KResidentUpload`; wired in `LlamaTransformerHandler`, `Phi3TransformerHandler` (fused QKV/gate_up = one GEMV + host slice), and `Qwen3TransformerHandler` for Q4_K projections; parity tests green. LoRA play Phase 1 **complete**. Bake-off [`20260910T025804Z`](../perf-compare/20260910T025804Z/).

**Gate amendment (2026-09-10):** The original ≥1.3× tg vs FP16-resident bar is **deferred**. Landed PTX is compute-bound vs cuBLAS FP16 on GTX 1080 (Phi-3.5 MMQ ≈ **7.4** tg vs FP16 ≈ **12.2**). **Shipped claim:** `--mmq` is a **VRAM-fit** path (packed Q4 residency; Mistral ≈ **0.14×** llama near P0 **0.15×** fit). **Not claimed:** decode speedup vs `--mmq off` until a tile/`mul_mat_vec`-class kernel lands. Default remains **off**.

**Follow-ons (not blocking 13B feature-complete):**

1. **Shared-activation GEMV (Phase 1 landed)** — `MatVec.sgemvSameX` uploads `x` once and coalesces sync for Q/K/V and gate/up on Llama decode (`CudaMatVec` / `RocmMatVec`). Full device-resident hidden-state chain (norm / attn / residual on GPU) remains future work.
2. **Tile Q4_K kernel** — required before any speed claim vs FP16-resident.
3. Phi-2 / Qwen3-MoE GPU residency; optional Phi fused Q4 split like FP16.

**LoRA adjacency (Phase 1 complete):** [`PLAN-Infra-LoRA-MMQ.md`](PLAN-Infra-LoRA-MMQ.md). Not part of Tier 13B exit.

## Feature × surface interaction matrix (`--mmq`)

Per ROADMAP **§6**. Cells filled for Phase B current state:

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | CUDA | ROCm | Default |
|--------------------|----------------|-------------|------------|--------|------------|--------------|-----------------|------|------|---------|
| `--mmq` | **wired** (Llama-family + Phi-3 + Qwen3 dense Q4_K) | **wired** (Phase 1: `LoraTrainableHandler` / Qwen2) → [`PLAN-Infra-LoRA-MMQ.md`](PLAN-Infra-LoRA-MMQ.md) | **explicit no-op** + warn (train stays FP16/FP32) | N/A (text MatVec only) | **wired** if decode uses same MMQ projections | **wired** with partial offload (Q4 upload only for resident layers) | **wired** (batched path uses Q4 `sgemm` serial GEMVs) | **wired** | **explicit no-op** (`supportsQ4KMmq` false) | **off** |

Phi-2 / Qwen3-MoE: **follow-up** (no GPU residency path yet). User-facing docs may claim `--mmq` for **VRAM fit** on Llama-family, Phi-3, and Qwen3 dense text inference plus LoRA playback — **not** as a throughput win vs FP16-resident.

## Overview

Flash Attention and fused quantized matmul (MMQ) are central to peer engine speed. Juno’s strategy is Panama + vendor BLAS first. Phase B ships packed Q4 residency for fit; speed parity with a tuned kernel remains a follow-on.

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

**Amended ship bar:** wiring + parity + published bake-off + honest docs (fit yes / speed deferred). Original ≥1.3× vs FP16 remains a **follow-on exit** for the tile-kernel workstream, not for marking 13B feature-complete.

## Implementation

### Phase A

1. ~~Profile + memo~~ **Done** — [`PLAN-Infra-PERF-ANALYSIS.md`](PLAN-Infra-PERF-ANALYSIS.md).

### Phase B (conditional)

1. Prototype behind a flag — **landed** (default off).
2. Parity vs oracle — **landed**.
3. Perf gate ≥1.3× vs FP16 — **amended / deferred** (fit ship; see status).
4. Docs and ROADMAP — **updated** for VRAM-fit claim + shared-activation Phase 1.

### Shared-activation Phase 1

- `MatVec.sgemvSameX` + CUDA/ROCm overrides; Llama `transformerLayer` / `ffn` use it for QKV and gate/up when all slots share a resident backend.
- Parity: `SgemvSameXParityTest`.

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.

Exit (Phase B MMQ — **met under amendment**):

1. **Ship (VRAM-fit):** flag-gated packed Q4 path with parity tests, bake-off artifacts, and docs that do **not** claim speed vs FP16-resident; or
2. **Close:** Phase A memo shows BLAS path sufficient (not used — prototype shipped).

**Deferred speed exit (tile kernel follow-on):** ≥1.3× decode TPS vs `--mmq off` on Phi-3.5 or TinyLlama, or ≥1.5× long-context prefill, with a new bake-off row.

## Implementation todos

1. ~~Phase A~~
2. ~~Phase B prototype + parity + amended ship~~
3. Shared-activation Phase 1 — **landed** (Llama); extend other handlers as follow-up
4. Tile Q4 kernel — open
5. Preview files; no zip

## Preview files (expected)

Phase B: PTX / Panama bindings, handlers, CLI flag, tests, ROADMAP / howto / performance notes
