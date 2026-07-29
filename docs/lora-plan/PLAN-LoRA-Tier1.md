# Tier 1: Projection Coverage, Accumulation, and Clipping

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

## Overview

Implement the correctness foundation for the five-tier roadmap in `PLAN-LoRA-ROADMAP.md`: configurable projection targets, complete backward math, token-weighted gradient accumulation, and global gradient-norm clipping. This tier preserves existing qv checkpoints and public APIs and establishes the CPU oracle used by Tiers 2–5.

## Scope and compatibility

- Preserve current `.lora` version-1 files and existing `LoraTrainer.open(model, adapter, rank, alpha, lr)` behavior.
- Preserve the legacy overload's current initialization, qv targets, one-chunk update behavior, and clipping-disabled behavior. New config-based defaults must not silently change legacy runs.
- Keep `qv` as the default target preset; add `all` and explicit target lists.
- Restrict training/playback to architectures actually supported by the dense LLaMA LoRA handler; fail clearly for fused or structurally incompatible Phi/Qwen/MoE paths.
- Follow test-first development, keep the implementation allocation-conscious, and prefer focused new classes over growing `ConsoleMain` further.

## 1. Define target projections and initialization

- Add `node/src/main/java/cab/ml/juno/node/LoraProjection.java` as the single source of truth for key, dimensions, and GGUF tensor suffix:
  - `wq`: hidden→hidden, `attn_q.weight`
  - `wk`: hidden→kv, `attn_k.weight`
  - `wv`: hidden→kv, `attn_v.weight`
  - `wo`: hidden→hidden, `attn_output.weight`
  - `wgate`: hidden→intermediate, `ffn_gate.weight`
  - `wup`: hidden→intermediate, `ffn_up.weight`
  - `wdown`: intermediate→hidden, `ffn_down.weight`
- Keep all external keys lowercase and stable in CLI/checkpoints: `wq,wk,wv,wo,wgate,wup,wdown`; enum constants may use Java naming internally.
- Parse `qv`, `all`/`all-linear`, and comma-separated keys deterministically; reject unknown, duplicate, or empty target sets.
- Add `node/src/main/java/cab/ml/juno/node/LoraInitializer.java` to create adapters in stable layer/projection order.
- Keep `LoraQvInitializer.java` as a compatibility facade delegating to the new initializer.
- Validate loaded adapter keys, layers, and dimensions against `LlamaConfig` before handler construction.

## 2. Generalize forward and backward paths

- Refactor `LoraTrainableHandler.java` around one projection helper used by inference and training, preventing target drift.
- Apply adapters at the correct inputs: Q/K/V←`xNorm1`, O←`attnOut`, gate/up←`xNorm2`, down←`hiddenAct`.
- Extend `LayerState` only with values required for backward, notably post-RoPE Q and `attnOut`; avoid retaining duplicate large arrays.
- Add complete adapter backward contributions:
  - `wDown.backward(gradFfnOut, hiddenAct)` into `gradHidden`.
  - `wGate.backward(gradGate, xNorm2)` and `wUp.backward(gradUp, xNorm2)` into `gradXNorm2`.
  - `wo.backward(gradAttnProj, attnOut)` into `gradAttnOut`.
  - Compute current-position K gradients, inverse-RoPE both Q and K gradients, then combine frozen and adapter Q/K/V input gradients before RMSNorm backward.
- Correct `rmsNormBackward` to use `cfg.rmsNormEps()` instead of a hard-coded epsilon.
- Add explicit architecture validation in `ForwardPassHandlerLoader.java`; do not silently route incompatible architectures through the LLaMA training handler.

## 3. Separate gradient computation from optimizer updates

- Add `LoraGradientResult(lossSum, predictionCount, forwardMs, backwardMs)` in the node module; use this exact type name across handler and orchestration APIs.
- Split `trainStep` into:
  - `computeGradients(tokens)`: forward/backward only, accumulates unnormalized summed gradients, never clears gradients or steps the optimizer.
  - Legacy `trainStep(tokens, optimizer)`: zeroes gradients, computes one sequence, normalizes by predictions, steps once, and returns mean loss.
- Add `node/src/main/java/cab/ml/juno/node/LoraGradientBatch.java` to aggregate `LoraGradientResult` values, prediction counts, chunks, and timing.
- In CLI and programmatic loops: zero once per accumulation group, process up to N chunks, normalize by total prediction tokens, clip, step once, and flush the final partial group.
- Report token-weighted pass loss rather than the last chunk’s loss; never average per-chunk means.

## 4. Add global gradient preparation

- Add `lora/src/main/java/cab/ml/juno/lora/LoraGradients.java` with a deterministic two-pass operation over all A/B gradients.
- Accumulate squared norm in `double`, reject non-finite values before optimizer mutation, then apply one combined normalization/clipping scale.
- Clip all adapter gradients jointly after prediction-count normalization and before Adam/moment/weight-decay processing.
- Define `maxNorm == 0` as clipping disabled while still normalizing gradients.
- Return norm, applied scale, and clipped status for logs/JFR.

## 5. Centralize configuration and CLI wiring

- Add builder-based `juno-player/src/main/java/cab/ml/juno/player/LoraTrainingConfig.java` with targets, LR, accumulation steps, and max gradient norm. Use a builder so later tiers can add scheduling, validation, GPU, and algorithm fields without breaking positional constructors.
- Reserve `lora/src/main/java/cab/ml/juno/lora/LoraAdapterConfig.java` for Tier 3 adapter identity: rank, declared alpha, scaling, initialization, and mode. Until Tier 3 lands, Tier 1 may carry rank/alpha in the training builder but must keep the API separable.
- Add an `open(..., LoraTrainingConfig)` overload in `LoraTrainer.java`; retain the old overload with qv, accumulation 1, and clipping disabled.
- This tier owns the config-based `open` overload. Later tiers extend its configuration rather than adding competing overloads.
- Extract/test LoRA option parsing rather than adding more untestable static parsing to `ConsoleMain.java`.
- Wire flags and environment variables through `scripts/run.sh` and `scripts/run.bat`:
  - `--lora-targets`, `LORA_TARGETS`
  - `--lora-gradient-accumulation`, `LORA_GRADIENT_ACCUMULATION`
  - `--lora-max-grad-norm`, `LORA_MAX_GRAD_NORM`
- Recommended CLI defaults: `qv`, accumulation `1`, max norm `1.0`; legacy Java overloads keep clipping disabled for numerical compatibility.

## 6. Merge and observability

- Extend `LoraMerge.java` through `LoraProjection` so all seven target names map consistently; reject unsupported keys and stop assuming 44 tensors.
- Limit this tier's merge ownership to validated projection mapping and safe F32 output. Tier 3 adds rsLoRA/DoRA formulas; Tier 5 owns requantization policy.
- Continue writing adapted tensors as F32 and warn that all-linear merging increases output size substantially.
- Update `LoraTrainEvent.java` to emit one event per optimizer update with chunk count, prediction count, global norm, clip scale, and clipped status.
- Update `JfrMetricsExtractor.java` defensively so old recordings remain readable.

## 7. Tests first

- Add `LoraGradientsTest.java`: exact norm, token normalization, clipping, disabled/zero gradients, NaN/Inf rejection.
- Add `LoraProjectionTest.java`: presets, CSV parsing, dimensions, mappings, invalid targets.
- Extend `LoraQvInitializerTest.java`: compatibility, all-linear count/order, GQA K/V dimensions.
- Extend `LoraTrainableHandlerTest.java`: finite-difference adapter/input gradients for every projection, K/RoPE path, summed-gradient equivalence, unequal-chunk weighted loss, legacy one-step semantics, model RMS epsilon.
- Extend `LoraTrainerTest.java`: optimizer update counts, final partial group, all-linear save/playback, configured keys.
- Add merge tests for all target mappings and unknown-key behavior.

## 8. Documentation and handoff

- Update `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `docs/LoRA.md`, `docs/howto.md`, `docs/features.md`, `docs/arch.md`, `docs/agent-arch.txt`, and `docs/performance.md`.
- Document target costs, token-weighted accumulation, clipping order, architecture restrictions, merge expansion, and update-vs-chunk counters.
- Describe the feature as LoRA on a quantized GGUF base, not QLoRA.
- Correct `/reset`: recreate adapters from the selected config instead of only zeroing B while retaining trained A.

## Verification

- Run `mvn test -pl lora`.
- Run `mvn test -pl node -am`.
- Run `mvn test -pl juno-player -am`.
- Run the full Maven suite.
- With a gated TinyLlama model, compare: qv compatibility baseline; accumulation >1 with clipping; all-linear train/save/playback/merge; JFR event count equals optimizer updates.

## Implementation todos

1. Write projection, gradient preparation, accumulation, and compatibility tests first.
2. Implement target metadata/initialization and complete all-linear forward/backward support.
3. Separate gradient computation from stepping and add token-weighted accumulation plus clipping.
4. Wire configuration, merge mappings, scripts, CLI, and JFR metrics.
5. Update documentation and run unit, reactor, and gated model verification.
