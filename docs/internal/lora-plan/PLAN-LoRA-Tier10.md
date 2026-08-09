# Tier 10: Multi-Arch GPU LoRA Parity and Production Gates

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

- `PLAN-LoRA-ROADMAP.md`
- `PLAN-LoRA-Tier3.md` (DoRA norm-refresh production gate)
- `PLAN-LoRA-Tier4.md`, `PLAN-LoRA-Tier9.md` (resident transpose + train-device contracts)
- `PLAN-LoRA-Tier6.md` (factory allowlist, Phi fused layout, Qwen3 norms)
- `PLAN-LoRA-Tier7.md` (JFR lifecycle exit criteria)
- `node/.../LoraTrainingHandlerFactory.java`
- `LoraTrainableHandler.java` (reference resident upload path)
- `Phi3LoraTrainableHandler.java`, `Qwen3LoraTrainableHandler.java`, `Qwen2LoraTrainableHandler.java`
- `LoraTrainingMath.java` (CPU transpose used by Phi-3/Qwen3 today)
- `docs/LoRA.md`, `docs/performance.md`, `CHANGELOG.md`

Prerequisite: Tier 6 CPU handlers exist. Tier 9 (or equivalent Milestone 1 resident path on LLaMA) should be available so Phi-3/Qwen3 can reuse the same residency helper rather than inventing a third GPU stack.

## Overview

Tier 6 delivered CPU-oracle LoRA for LLaMA-family, Qwen2/2.5, Phi-3, and dense Qwen3. GPU resident forward/transpose is wired for LLaMA-family and Qwen2 (delegate). Phi-3 and Qwen3 still call CPU `LoraTrainingMath.transposedMatVec` for frozen transpose. Gated live LoRA smokes on real GGUFs are missing. Tier 3 DoRA refresh budget and Tier 7 CHANGELOG “start” wording remain open.

This tier closes **architecture GPU parity** and **production/documentation gates**. It does not add MoE, Gemma, or Qwen3.5 LoRA.

## Scope and compatibility

Goals:

1. Share resident weight upload + `sgemvTranspose` with Phi-3 and dense Qwen3 handlers.
2. Add `@EnabledIf`-style gated live LoRA smokes for available fixtures.
3. Close or explicitly document DoRA norm-refresh as correctness-only (Tier 3 leftover).
4. Confirm Tier 7 exit criteria and mark complete in CHANGELOG when green.
5. Explicitly defer Tier 5 held-out research matrix unless product requires projected-merge SLAs.

Non-goals:

- Qwen3-MoE, Qwen3.5, Gemma, multimodal LoRA (roadmap deferrals).
- Replacing Tier 9 microbatch work.
- Exact QA-LoRA K-quant merge.
- Tensor-parallel / pipeline-parallel LoRA.

## Architecture matrix (target after this tier)

| Capability | LLaMA / Mistral / TinyLlama | Qwen2 / 2.5 | Phi-3 | Dense Qwen3 |
|------------|-----------------------------|-------------|-------|-------------|
| CPU train / play / F32 merge | Yes | Yes | Yes | Yes |
| Resident GPU forward + transpose | Yes | Yes | **Yes (this tier)** | **Yes (this tier)** |
| Gated live LoRA smoke | **Yes** | **Yes** | **Yes** | **Yes** (when fixture present) |

## 1. Shared residency helper

Prefer a **new** focused class over copy-paste into each handler, e.g.:

- `LoraResidentWeights.java` — upload dequantized projections to `ResidentWeightMatrix[]` keyed by logical projection; close on failure; expose `transposedMatVec(quant, resident, g, rows, cols)`.

Requirements:

- Reuse `GpuMatVec` / `supportsHalfResident()` behavior from `LoraTrainableHandler.uploadResidentWeights`.
- Honor `--lora-train-device` from Tier 9 when present.
- Phi-3 fused QKV / FFN-up layouts must upload and transpose the **physical** tensors then apply logical slice adapters as today (Tier 6 layout bindings). Do not break fused-slice F32 merge.
- Qwen3 must preserve per-head Q/K RMSNorm and `qDim` vs `hiddenDim` differences; residency is for dense projections only.
- On VRAM failure: close partial buffers, log, fall back to CPU quantized transpose (existing policy).

Refactor `LoraTrainableHandler` to call the shared helper so LLaMA/Qwen2 and Phi-3/Qwen3 do not diverge.

## 2. Handler integration

### Phi3LoraTrainableHandler

- After load, if backend is `GpuMatVec` and device policy allows, upload fused and unfused projections needed for train backward.
- Route frozen transpose through resident matrices; keep architecture-specific RoPE (NeoX) and norm adjoints on host.
- Zero-adapter logit parity with inference handler must remain green.

### Qwen3LoraTrainableHandler

- Same residency pattern for dense projections.
- Do not claim MoE support.

### Qwen2LoraTrainableHandler

- Already delegates; verify shared helper refactor does not regress biases / frozen QKV bias path.

## 3. Gated live LoRA smokes

Add integration tests gated on fixture presence (pattern used elsewhere for live GGUFs), one per available model:

Flow per architecture:

1. Open trainer with small rank (e.g. 8), `qv` targets.
2. `/train-qa`-equivalent programmatic train until modest loss drop or fixed few updates.
3. Save `.lora`, reload with `--lora-play` / inference overlay.
4. Assert completion-only training did not mode-collapse every prompt (smoke assertion: answer tokens appear for the trained question under greedy decode when applicable).

Fixtures (enable when files exist under `models/` or CI cache):

- TinyLlama Q4_K_M
- Qwen2.5 dense GGUF (if present)
- Phi-3.5-mini instruct GGUF (if present)
- Dense Qwen3 GGUF (if present)

Missing fixtures must skip, not fail CI.

## 4. DoRA production gate (Tier 3 leftover)

Choose exactly one outcome and document it:

**A. Measure and gate:** benchmark exact DoRA norm refresh time and peak heap on TinyLlama all-linear; record budget in `docs/performance.md`; fail CI if exceeded without opt-in flag.

**B. Correctness-only:** CHANGELOG + `docs/LoRA.md` state DoRA is correctness-complete but not production-perf-gated; operators should prefer standard LoRA/rsLoRA for large all-linear jobs until a budget exists.

Do not leave the Session 37 “pending” wording without a decision.

## 5. Tier 7 completion check

Verify against `PLAN-LoRA-Tier7.md` exit gate:

- Programmatic LoRA `--jfr` + auto-extract on exit
- Mode identity on train events for `lora` / `rslora` / `dora` / `qa-lora`
- Extractor series with guarded reads
- Docs list metrics contract

If green: update `CHANGELOG.md` from “Tier 7 (start)” to complete; fix any missing pretty-printer section only if listed in Tier 7 as required.

Tier-4 timing subsets may remain zero on CPU-only runs; on GPU runs after Tier 9 they should be non-zero.

## 6. Tier 5 research matrix

Do **not** expand scope into the full held-out QA-LoRA quality matrix unless product explicitly requires it. Record in ROADMAP / CHANGELOG:

- Tier 5 **code** complete
- Held-out quality thresholds / experiment matrix **deferred**
- Exact K-quant affine merge remains unsupported

## 7. Tests first

- Unit: residency helper upload/close/fallback with mocked or small dense matrices where feasible
- Handler: Phi-3 / Qwen3 finite-difference or adjoint checks still pass with residency enabled (GPU group)
- Zero-adapter logit parity unchanged
- Gated live smokes as above
- Factory allowlist regressions unchanged (`qwen3moe`, `qwen35`, `gemma` rejected)

## 8. Expected files

Likely modified:

- `node/.../LoraTrainableHandler.java`
- `node/.../Phi3LoraTrainableHandler.java`
- `node/.../Qwen3LoraTrainableHandler.java`
- `node/.../Qwen2LoraTrainableHandler.java` (if refactor touch)
- `docs/LoRA.md`, `docs/performance.md`, `docs/howto.md`, `docs/agent-arch.txt`
- `CHANGELOG.md`, `RELEASE_NOTES.md`
- `PLAN-LoRA-ROADMAP.md` status notes if maintained inline

Likely new:

- `node/.../LoraResidentWeights.java` (name may vary; keep focused)
- `node/src/test/...` GPU and gated live smoke tests
- Optional DoRA benchmark harness under `scripts/` or test group

## Verification and exit gate

Exit only when:

1. Phi-3 and dense Qwen3 use resident GPU transpose when device policy and VRAM allow; CPU fallback remains correct.
2. LLaMA/Qwen2 still pass existing GPU/CPU tests after helper extraction.
3. Gated live LoRA smokes exist and pass or skip cleanly per fixture.
4. DoRA gate decided (measured or documented correctness-only).
5. Tier 7 marked complete in CHANGELOG if criteria met; else list remaining gaps precisely.
6. Tier 5 research matrix explicitly deferred in docs/roadmap.
7. `mvn test -pl node -am` and `mvn test -pl juno-player -am` pass; GPU-group tests pass on reference hardware when `-Dgroups=gpu` / ROCm group is run.

## Implementation todos

1. Extract shared `LoraResidentWeights` (or equivalent); refactor LLaMA handler to use it; tests for close/fallback.
2. Integrate into Phi-3 and Qwen3 handlers; GPU adjoint / parity tests.
3. Add gated live LoRA smokes for available GGUFs.
4. Resolve DoRA refresh gate (measure or document).
5. Confirm Tier 7; update CHANGELOG/docs; defer Tier 5 research explicitly.
6. List preview files; no zip.
