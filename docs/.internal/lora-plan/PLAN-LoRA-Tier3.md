# Tier 3: rsLoRA, Kaiming Initialization, and DoRA

## Agent handoff

Read and follow `models/CLAUDE.md` before implementing:

1. Unit tests first, only for valuable business logic.
2. Implementation details designed with performance in mind.
3. Follow KISS.
4. Prefer adding new Java classes over extending existing ones.
5. Update `docs/agent-arch.txt`, `docs/howto.md`, `README.md` when applicable.
6. No emojis; be strict and precise.
7. Output: list changed files for preview; never zip files.

Also read `README.md`, `CHANGELOG.md`, `docs/`, `scripts/`, and the LoRA-related source/tests before coding.

Prerequisite: implement after Tiers 1 and 2. Reuse Tier 1 projection/backward contracts and Tier 2 parameter-group optimizer/training-loop contracts rather than creating parallel APIs.

## Overview

Add explicit scaling and initialization metadata, rsLoRA, PEFT-compatible Kaiming-uniform initialization, and canonical DoRA with safe checkpointing and merge support. DoRA is gated by an explicit norm-refresh performance benchmark before being described as production-ready.

## Prerequisites and delivery gates

- Implement after Tiers 1 and 2. DoRA magnitude updates must use Tier 2 parameter groups, scheduling, clipping, and shared orchestration.
- Deliver in three internal phases: metadata/rsLoRA/Kaiming; canonical DoRA; CLI/docs/integration.
- Keep standard scaling as the compatibility default; use Kaiming-uniform for newly created adapters while preserving legacy-normal as an explicit reproducibility option.
- Treat exact DoRA norm-refresh performance as a go/no-go gate for production claims.
- Explicitly reject tensor-parallel adapter overlays until shard-aware LoRA/DoRA semantics are separately implemented; local and pipeline-parallel execution are in scope.

## 1. Add explicit adapter configuration

- Add immutable types in the lora module:
  - `LoraScaling.java`: `STANDARD`, `RANK_STABILIZED`.
  - `LoraInitialization.java`: `KAIMING_UNIFORM`, `LEGACY_NORMAL`.
  - `LoraMode.java`: `LORA`, `DORA`.
  - `LoraAdapterConfig.java`: rank, declared alpha, scaling, initialization, mode.
- Keep `LoraAdapterConfig` separate from Tier 1's builder-based `LoraTrainingConfig`; adapter identity belongs in checkpoints, while training policy does not.
- Keep old constructors/factories as standard-LoRA compatibility overloads using legacy-normal initialization. New config-based creation may default to Kaiming; do not silently change the legacy overload.
- Validate rank, finite alpha, and non-null modes explicitly; store declared alpha separately from effective scale.

## 2. Implement rsLoRA and Kaiming-uniform

- Update `LoraAdapter.java`:
  - Standard scale: `alpha / rank`.
  - rsLoRA scale: `alpha / sqrt(rank)`.
  - Kaiming-uniform A: `U(-1/sqrt(inDim), +1/sqrt(inDim))`, matching PyTorch `kaiming_uniform_(a=sqrt(5))`.
  - Preserve B as exact zeros.
  - Use the supplied seeded `Random`; avoid unseeded construction.
  - Replace `fromWeights` random-initialize-then-overwrite with a private direct-loading constructor.
- Ensure forward, backward, merge, status, and serialization consume the adapter’s effective scale rather than recomputing formulas.

## 3. Introduce checkpoint v2 safely

- Update `LoraAdapterSet.java` to read v1 and v2 and normally write v2.
- v1 loads as standard scaling, legacy/unknown initialization provenance, plain LoRA.
- Make v2 entries length-delimited and store key, dimensions, rank, declared alpha, scaling, initialization, mode, A/B, optional DoRA magnitude, and adapted-base tensor fingerprint metadata.
- Reserve length-delimited extension fields for Tier 5 algorithm, pooling, quantization-layout, encoder-version, and merge-capability metadata before freezing v2.
- Validate lengths before allocation, enum IDs, duplicate keys, truncation, dimensions, finite values, and trailing bytes.
- Do not serialize gradients or optimizer state.
- Optional `saveLegacyV1` may encode rsLoRA’s effective scale as a transformed legacy alpha; reject DoRA legacy export because v1 cannot represent magnitude semantics.

## 4. Add canonical detached-norm DoRA state

- Add `DoraMagnitude.java` holding magnitude and gradient vectors with shape/finite validation and zeroing.
- Extend `LoraAdapterSet` with magnitude state keyed identically to adapters; require a matching LoRA entry and exact `outDim` length.
- Add `DoraProjection.java` to bind frozen quantized tensor, low-rank adapter, magnitude, cached row norms/coefficient, and dirty state.
- Use canonical parameterization per output row:
  - `direction = W + scale*B*A`
  - `coefficient = magnitude / max(norm(direction), epsilon)` with norm detached from gradients
  - `output = coefficient * (W*x + scale*B*A*x)`
- Initialize magnitude from row norms of Juno’s F32 GGUF dequantization so B=0 reproduces the base output.
- Backward with `scaledGradient = coefficient * gradOut`; accumulate `gradMagnitude += gradOut * directionOutput / norm`; feed `scaledGradient` to frozen transpose and LoRA backward.
- Do not differentiate through row norm; doing so would be a different algorithm than canonical PEFT-style DoRA.

## 5. Integrate DoRA in handlers and optimizer

- Build model-aware DoRA state in a new initializer such as `DoraInitializer.java`, because base row norms/fingerprints require GGUF tensors and do not belong in the lora module.
- Update `LoraTrainableHandler.java` to compose plain LoRA or DoRA per selected projection using Tier-1 projection metadata.
- Store compact per-projection direction output/norm state needed for magnitude gradients; for Q retain pre-RoPE DoRA state, and for V use current-position value state.
- Scale gradients by DoRA coefficient before both frozen transpose matvec and adapter backward.
- Register magnitude with Tier 2's optimizer parameter-group mechanism, using independent moments and default magnitude weight decay off. Do not create separate optimizer semantics in this tier.
- Mark DoRA norm caches dirty after optimizer updates; refresh once before the next forward. In immutable inference-only playback, refresh once at load.
- Document direct A/B array mutation as requiring explicit invalidation, or narrow mutable array exposure if source compatibility allows.

## 6. Implement exact norm refresh with a benchmark gate

- First implementation: dequantize one adapted projection at a time, construct/accumulate effective rows, calculate exact F32 row norms, and release temporary storage before the next projection.
- Keep initialization, refresh, and merge on the same Juno dequantization semantics; document the small runtime difference when GPU weights are FP16-resident.
- Record norm-refresh duration separately in JFR.
- Benchmark memory and time for qv and all-linear on TinyLlama and at least one 7B model.
- If projection-at-a-time refresh is unacceptable, add row-streaming dequantization before considering GEMM backend expansion. Do not substitute a frozen-`W` norm approximation and call it DoRA.

## 7. Base-model binding and distributed safety

- Store SHA-256 of each adapted raw GGUF tensor plus tensor type/dimensions in v2 DoRA entries.
- Verify fingerprints during playback, training resume, and merge; fail by default on mismatch.
- In pipeline parallelism, initialize/refresh only local-layer DoRA projections while retaining global checkpoint keys.
- In `ForwardPassHandlerLoader.java` and relevant node startup paths, explicitly reject LoRA/rsLoRA/DoRA overlays under tensor parallelism rather than producing incorrect sliced results.

## 8. Merge support

- Generalize `LoraMerge.java`:
  - LoRA/rsLoRA: `W + scale*B*A`.
  - DoRA: rowwise `magnitude/norm(direction) * direction`.
- Validate model fingerprint, dimensions, magnitude presence, and finite values.
- Continue writing adapted tensors as F32; preserve untouched tensor bytes.
- Report mode, scaling, target count, and exact tensors patched without fixed-count assumptions.

## 9. API, CLI, reset, and scripts

- Extend the Tier 1 config-based `LoraTrainer.open` through `LoraAdapterConfig`; do not add a competing overload or positional configuration type.
- Add creation-time flags in `ConsoleMain.java`:
  - `--lora-mode lora|dora`
  - `--lora-scaling standard|rslora`
  - `--lora-init kaiming-uniform|legacy-normal`
- Loaded checkpoint metadata is authoritative; reject or warn on explicit conflicting creation flags.
- Make startup and `/status` report actual per-checkpoint mode, declared alpha, effective scale, initialization, and targets.
- Make `/reset` recreate adapters from selected config; for DoRA reread base row norms/fingerprints.
- Mirror options as `LORA_MODE`, `LORA_SCALING`, and `LORA_INIT` in `scripts/run.sh` and `scripts/run.bat`.
- Playback/deployment reads algorithm metadata from checkpoint; require compatible binaries rather than separate inference flags.

## 10. Tests first

- Extend `LoraAdapterTest.java`: standard/rs scales, finite-difference gradients in both modes, Kaiming bounds/determinism/statistics, exact zero B, legacy initialization.
- Add `DoraMagnitudeTest.java`: shape, gradients, zeroing, finite validation.
- Extend `LoraAdapterSetTest.java`: hard-coded v1 fixture, bit-exact v2 LoRA/rsLoRA/DoRA round trips, corruption/duplicate/truncation tests, legacy export behavior.
- Extend `LoraAdamOptimizerTest.java`: magnitude moments/update/reset, no default magnitude decay, dirty-cache notification.
- Add `DoraProjectionTest.java`: B=0 identity, row axis, dense reference forward, detached-norm finite differences for magnitude/A/B/X, epsilon, cache invalidation, representative quantized tensors.
- Extend `LoraTrainableHandlerTest.java`: rsLoRA and DoRA loss reduction, Q inverse-RoPE, V state, coefficient-scaled frozen transpose, norm refresh, immutable concurrent inference.
- Add `LoraMergeTest.java`: dense references for all modes, initial DoRA identity, F32 adapted tensors, untouched-byte preservation, fingerprint failure, quantization coverage.
- Add pipeline-local mapping and explicit tensor-parallel rejection tests.
- Verify Tier 2 LoRA+ ratio, scheduling, clipping, and validation continue to work for rsLoRA and DoRA parameter groups.

## 11. Documentation and verification

- Update `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `docs/LoRA.md`, `docs/howto.md`, `docs/features.md`, `docs/arch.md`, `docs/agent-arch.txt`, `docs/model_support_summary.md`, `docs/performance.md`, and review `docs/legal.md` for arbitrary-base wording.
- Document formulas, Kaiming convention, v1/v2 compatibility, base fingerprinting, norm-refresh cost, merge behavior, optimizer-state limitation, and local/pipeline-only support.
- Run module tests and full Maven tests, then gated TinyLlama standard/rsLoRA/DoRA train-playback-merge equivalence.
- Benchmark exact norm refresh separately from token inference. Do not mark DoRA production-ready unless the measured refresh time and peak memory meet an agreed budget.
- Treat this tier's CPU path as the correctness oracle for Tier 4 GPU training.

## Implementation todos

1. Write scaling, initialization, checkpoint-v2, and compatibility tests first.
2. Implement explicit metadata, rsLoRA scaling, Kaiming initialization, and checkpoint v2.
3. Implement canonical DoRA magnitude/projection math, optimizer state, and norm cache lifecycle.
4. Add base fingerprinting, distributed validation, and LoRA/rsLoRA/DoRA merge behavior.
5. Wire APIs/CLI/scripts, run integration tests and norm-refresh benchmarks, then update docs.
