# Tier 9: GPU LoRA Productization and Microbatching

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
- **`PLAN-LoRA-Tier4.md` in full** — this tier executes the *remaining* Tier 4 milestones against the current codebase
- `PLAN-LoRA-Tier8.md` (recommended first: longer chunks make microbatch worthwhile)
- `docs/performance.md` (LoRA GPU baseline section — still says hybrid is not production GPU training)
- `docs/LoRA.md`, `CHANGELOG.md` Session 38
- GPU stack: `GpuBindings`, `CudaBindings`, `RocmBindings`, `GpuMatVec`, `CudaMatVec`, `RocmMatVec`, `ResidentWeightMatrix`
- `LoraTrainableHandler` (`uploadResidentWeights`, `transposedMatVecLayer`, `sgemvTranspose`)
- `LoraCliOptions`, `LoraTrainingConfig` (`trainDevice` field may exist but is not CLI-wired)

Prerequisite: Tier 4 **primitives** (resident transpose API + LLaMA-family handler wiring) already exist. Do **not** re-implement transpose from scratch. Do not begin with tiny per-call LoRA GEMVs on GPU without batching.

## Overview

Session 38 shipped resident FP32/FP16 transpose primitives and LLaMA-handler integration. Product gaps remain:

- No `--lora-train-device auto|gpu|cpu`
- `LoraTrainEvent` timing subsets often zero / unfilled
- No microbatched frozen forward/transpose (`DeviceActivationBatch` / batched GEMV)
- No published CPU↔GPU parity IT or 2× / 1.5× speed gates
- Docs correctly refuse to call the path “GPU LoRA training”

This tier finishes measured GPU training for the LLaMA-family / Qwen2 path (Qwen2 delegates to `LoraTrainableHandler`). Phi-3 and Qwen3 resident GPU parity is **Tier 10**.

## Current state (do not redo)

Already present (verify before coding):

- `GpuMatVec.sgemvTranspose` on CUDA and ROCm
- `LoraTrainableHandler.uploadResidentWeights` + `transposedMatVecLayer` → device when resident
- Adjoint / transpose contract tests under GPU groups
- `MatVecBackend` resident-transpose labels

Still missing (this tier):

1. CLI/env device selection and fail-closed `gpu` mode
2. Populated train-step timing fields
3. Microbatching (§5 of Tier 4)
4. Selective device adapter math (§6) and optional GPU Adam (§7) — only after measured win
5. Product gate evidence in `docs/performance.md`

## Scope and compatibility

Goals:

1. Productize `--lora-train-device` / `LORA_TRAIN_DEVICE`.
2. Fill Tier-4 timing subsets on `juno.LoraTrainStep`.
3. Implement microbatched frozen linears; keep token-weighted normalize + clip.
4. Pass Milestone 1 and microbatch gates from `PLAN-LoRA-Tier4.md`.
5. Only then update docs to allow “GPU LoRA training” language with measured numbers.

Non-goals:

- Phi-3 / Qwen3 resident upload (Tier 10).
- Custom GPU attention / softmax / RoPE kernels.
- Tensor-parallel LoRA.
- Claiming GPU training before gates pass.
- Changing Tier 8 corpus/chunk CLI semantics.

## 1. Establish / refresh the baseline

- Instrument `LoraTrainEvent` fields: `frozenForwardMs`, attention/nonlinear, `frozenTransposeBackwardMs`, `adapterBackwardMs`, `transferMs`, `optimizerMs` (names as already reserved; fill from handler, not zeros).
- Benchmark on TinyLlama Q4_K_M, seq 64/128 (use Tier 8 `--lora-chunk-tokens` when available), rank 8, `qv` and `all`.
- Record tokens/s, timings, H2D/D2H when counters exist, peak heap/VRAM.
- Keep baseline tables in `docs/performance.md` before claiming wins.

## 2. Wire `--lora-train-device`

Values: `auto` | `gpu` | `cpu`.

| Mode | Behavior |
|------|----------|
| `cpu` | Force CPU MatVec; no resident upload required |
| `gpu` | Require GPU backend + resident transpose path; fail closed if unavailable |
| `auto` | Current LoRA default: try GPU when present; fall back to CPU on OOM/unsupported |

Wire:

- `LoraCliOptions` + env `LORA_TRAIN_DEVICE`
- `LoraTrainingConfig.trainDevice`
- Handler / `ForwardPassHandlerLoader.selectLoraBackend` (or equivalent) must honor the mode
- `scripts/run.sh` / `run.bat` help + passthrough
- Metrics identity `trainDevice` field must reflect the resolved device

Do not silently ignore `gpu` when bindings are missing.

## 3. Milestone 1 productization (frozen backward)

Confirm resident transpose covers `wq,wk,wv,wo,wgate,wup,wdown` and output projection on the LLaMA handler. Keep `LayerState`, attention, LoRA A/B, clip, Adam on host unless later sections enable device adapters.

Milestone 1 gate (must pass before microbatch marketing):

- CPU/GPU loss and adapter gradients agree within declared FP32/FP16 tolerances
- GPU backward ≥ 2× CPU backward on reference benchmark
- End-to-end training ≥ 1.5× on reference GPU
- CUDA and ROCm both pass; do not ship CUDA-only as public “GPU training”

## 4. Microbatched frozen linear algebra

Follow Tier 4 §5 lowest-risk path:

1. Extend HSS strided-batched GEMV to `batchCount > 1`, or
2. Bind FP32 `cublasSgemm_v2` / `rocblas_sgemm`

Add new classes (prefer new over bloating MatVec):

- `GpuBlasOps.java` — explicit batched forward/transpose contracts
- `DeviceActivationBatch.java` — reusable device scratch

Integration rules:

- Microbatch inside a chunk / accumulation group
- Normalization remains by **total prediction count** (Tier 1)
- Keep attention/nonlinear on host initially
- Measure transfer reduction; do not assume GEMM is faster

Microbatch gate:

- Batched loss and summed gradients agree with sequential CPU within tolerance
- Batch 8 ≥ 2× Milestone 1 **or** ≥ 4× CPU
- H2D/D2H bytes per token decline as batch grows
- Document max batch size and VRAM for TinyLlama and one 7B model

## 5. Device-resident adapters and GPU Adam (conditional)

Implement Tier 4 §6–7 **only if** profiling shows host adapter or transfer dominates after microbatch:

- `DeviceLoraAdapter` / `LoraDeviceState`: upload A/B once; batched GEMM; preserve dropout, LoRA+, rsLoRA/DoRA scaling
- At batch 1, host adapter math must not regress wall time by more than 10%; otherwise keep host
- `LoraAdamOptimizerGpu` only if gradient download/host update is material; match host global-norm, AdamW, LoRA+, schedules

If gates fail, document “frozen batched GPU + host adapters” as the supported production path and stop.

## 6. Fallback policy

Ordered paths (unchanged intent from Tier 4 §8):

1. CPU quantized full path — oracle
2. GPU forward + CPU transpose — compatibility, not advertised as full GPU training
3. GPU forward + resident transpose
4. GPU batched frozen linears + host adapters
5. Fully resident adapters — only after intensity gates

Fail closed on adjoint/layout mismatch. Reject tensor-parallel adapter train/play until designed.

## 7. Tests first

Extend / add (GPU groups as required):

- Device CLI parse + fail-closed `gpu` without bindings
- Timing fields non-zero on a short instrumented train step (unit or gated)
- `LoraTrainableHandlerGpuBackwardTest`: CPU/GPU loss, A/B grads, clip input, updates
- Batched GEMV/GEMM parity vs sequential
- OOM / unsupported FP16 / CPU fallback
- Gated TinyLlama: loss decreases; save/playback; checkpoint drift within tolerance
- Dropout / LoRA+ / rsLoRA parity if device adapters land

## 8. Expected files

Likely modified:

- `node/.../LoraTrainableHandler.java`, `LoraTrainEvent.java`
- `node/.../CudaMatVec.java`, `RocmMatVec.java`, `GpuMatVec.java`, `GpuBindings.java`, …
- `juno-player/.../LoraCliOptions.java`, `LoraTrainingConfig.java`, `ConsoleMain.java`
- `scripts/run.sh`, `scripts/run.bat`
- `docs/performance.md`, `docs/LoRA.md`, `docs/howto.md`, `docs/agent-arch.txt`
- `CHANGELOG.md`, `RELEASE_NOTES.md`, `README.md` as applicable

Likely new:

- `node/.../GpuBlasOps.java`
- `node/.../DeviceActivationBatch.java`
- Optional `DeviceLoraAdapter.java`, `LoraAdamOptimizerGpu.java`
- Corresponding `*Test.java` under GPU groups

## Verification and exit gate

Exit only when Tier 4 product gates are met:

1. `--lora-train-device` works for `auto|gpu|cpu` with documented fail-closed behavior.
2. Train-step timing subsets are populated on the GPU path (not permanently zero).
3. CUDA and ROCm transpose + microbatch correctness tests pass.
4. Published TinyLlama numbers show ≥2× backward and ≥1.5× e2e (or microbatch gate equivalents).
5. `docs/performance.md` and `docs/LoRA.md` state what is and is not “GPU LoRA training” with evidence.
6. Device adapter / GPU Adam either gated-on with thresholds or explicitly deferred in CHANGELOG.

## Implementation todos

1. Wire `--lora-train-device` + tests; refresh baseline instrumentation.
2. Confirm Milestone 1 resident backward gates; fill timing fields.
3. Tests for batched frozen linears; implement `GpuBlasOps` + `DeviceActivationBatch`.
4. Integrate microbatch into handler train path; measure transfer and speed.
5. Conditionally add device adapters / GPU Adam only after intensity proof.
6. Update docs/CHANGELOG; publish benchmark table; list preview files.
