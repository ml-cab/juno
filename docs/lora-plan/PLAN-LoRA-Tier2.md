# Tier 2: Scheduling, AdamW, Dropout, Validation, and LoRA+

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

Prerequisite: implement after Tier 1 (`PLAN-LoRA-Tier1.md`) so this work uses separated gradient/update phases and `LoraTrainingConfig`.

## Overview

Build on Tier 1 with warmup/cosine scheduling, true decoupled AdamW, deterministic train-only LoRA dropout, held-out validation early stopping, and LoRA+ parameter-group learning rates. Existing APIs and compatibility defaults remain available.

## Prerequisite and compatibility

- Implement after Tier 1 so this work uses separated gradient/update phases and `LoraTrainingConfig` rather than introducing competing orchestration.
- Preserve constant LR, zero dropout, and no validation as compatibility defaults.
- Correct coupled L2 decay to true decoupled AdamW while retaining the established A-only decay policy; document this intentional numerical change.
- Promise source/checkpoint compatibility, not numerical trajectory compatibility, when switching coupled L2 to decoupled AdamW.
- Keep checkpoint v1 unchanged: optimizer, scheduler, dropout, and validation state are invocation state, not adapter inference state.

## 1. Learning-rate schedules

- Add `lora/src/main/java/cab/ml/juno/lora/LoraLearningRateSchedule.java` with constant and warmup-cosine factories.
- Use one-based optimizer-update numbering.
- Define warmup as `peakLr * update / warmupUpdates`; after warmup, cosine-decay from peak to minimum; clamp at minimum after total updates.
- Validate finite/non-negative rates, `minimum <= peak`, non-negative warmup, and a total-update count compatible with warmup.
- Build a schedule per training invocation from planned optimizer updates after Tier-1 accumulation grouping.

## 2. True A-only AdamW

- Update `LoraAdamOptimizer.java` with `step(adapters, learningRate)` while preserving `step(adapters)` as fixed-base-LR delegation.
- Feed only raw gradients into first/second moments.
- Apply decoupled decay to A using the scheduled, uncorrected LR and the pre-update parameter value; do not decay B.
- Expose base LR, last applied LR, and decay for status/JFR; validate epsilon, decay, and per-step LR.
- Keep gradient clipping before optimizer entry, so decay is never included in the global norm.

## 3. Deterministic train-only LoRA dropout

- Extend `LoraAdapter.java` with training-specific forward/backward methods; leave inference `forward()` unchanged.
- Apply inverted dropout to the LoRA branch input before A; use the same masked/scaled input for A/B gradients and mask the LoRA branch’s input gradient.
- Generate masks from a stateless index hash of `(root seed, optimizer update, global chunk/prediction ordinal, token position, absolute layer, projection, input index)` so forward/backward regenerate exactly without retaining large masks.
- Include the global chunk/prediction ordinal so equal token positions in different chunks of one accumulation group do not reuse identical masks.
- Fast-path dropout zero through existing math for bitwise compatibility and no added allocations.
- Store only compact per-projection seeds in training state; never use dropout in inference or validation.

## 4. Add LoRA+ parameter groups

- Extend `LoraAdamOptimizer.java` with explicit A and B learning-rate groups:
  - A uses the scheduled base learning rate.
  - B uses `scheduledBaseLearningRate * loraPlusRatio`.
  - `loraPlusRatio == 1.0` is exactly equivalent to ordinary non-LoRA+ optimizer behavior.
- Keep weight decay policy independent from the learning-rate ratio: A uses decoupled AdamW decay; B remains undecayed unless a future plan changes that policy explicitly.
- Validate a finite ratio greater than zero and expose the actual A/B learning rates in status and JFR.
- Add `--lora-plus-ratio` and `LORA_PLUS_RATIO`; default to `1.0` for compatibility. Recommend a larger value only after model-level benchmarks, rather than hard-coding 16 as a universal default.

## 5. Evaluation and validation data flow

- Add a forward-only `evaluateLoss(tokens)` to `LoraTrainableHandler.java`.
- Evaluation must use teacher forcing with local K/V buffers, no activation retention, no gradients, no optimizer mutation, no dropout, and no pollution of persistent inference caches.
- Return loss sum and prediction count so validation is token-weighted.
- For `/train-qa`, expose the four formatted variants separately in `ChatTrainingFormats.java`; hold out complete variants rather than splitting inside chat turns.

## 6. Shared training loop and early stopping

- Add `juno-player/src/main/java/cab/ml/juno/player/LoraTrainingLoop.java` and move duplicated orchestration out of `ConsoleMain`/`LoraTrainer`.
- Deterministically split chunk/document indices with the configured seed, clamp validation count to `[1, count-1]`, and disable validation with an explicit warning/result if fewer than two units exist.
- Evaluate validation once after each full pass, not after each chunk.
- Track improvement by `validationMinDelta`; stop after `validationPatience` checks without improvement.
- Snapshot best A/B weights, restore them on exit, and reset optimizer state after restoring older parameters.
- Keep the current low-training-loss guard separate from validation patience; expose a stop reason distinguishing target reached, patience exhausted, low-loss guard, and max iterations.
- Return final/best losses, best iteration, pass count, optimizer-update count, and stop reason through a richer result type while retaining legacy `TrainUntilResult` adapters.

## 7. Configuration, API, and CLI

- Extend Tier-1 builder-based `LoraTrainingConfig.java` with schedule mode, minimum LR, warmup updates, weight decay, LoRA+ ratio, dropout, seed, validation split, patience, minimum delta, and restore-best behavior.
- Keep `LoraTrainer.java` legacy overloads; add config-based train methods and delegate all chunking/training to `LoraTrainingLoop`.
- Update `ConsoleMain.java` to consume the shared loop and report actual LR, schedule, dropout, decay, train/validation loss, best pass, and stop reason.
- Wire through both launchers:
  - `--lora-lr-schedule constant|cosine`
  - `--lora-warmup-steps`
  - `--lora-min-lr`
  - `--lora-weight-decay`
  - `--lora-plus-ratio`
  - `--lora-dropout`
  - `--lora-seed`
  - `--lora-validation-split`
  - `--lora-validation-patience`
  - `--lora-validation-min-delta`
- Mirror environment variables and validation exactly in `scripts/run.sh` and `scripts/run.bat`.

## 8. Observability

- Extend `LoraTrainEvent.java` with actual A learning rate, B learning rate, LoRA+ ratio, and dropout while preserving existing fields.
- Optionally add `LoraValidationEvent.java` for validation loss, predictions, duration, and best-so-far status.
- Update `JfrMetricsExtractor.java` using guarded field reads for backward-compatible JFR parsing.

## 9. Tests first

- Add `LoraLearningRateScheduleTest.java`: warmup boundaries, cosine midpoint/end, no warmup, clamping, invalid parameters, deterministic calls.
- Update `LoraAdamOptimizerTest.java`: zero-gradient A decay, B exclusion, moments unaffected by decay, scheduled-LR scaling, LoRA+ ratio-1 equivalence, B-only ratio scaling, reset, validation.
- Update `LoraAdapterTest.java`: dropout-zero equivalence, same/different seeds, different accumulated chunks receive different masks, expectation over masks, fixed-mask finite differences for A/B/X, dropped-coordinate input gradients, invalid rates.
- Update `LoraTrainableHandlerTest.java`: LR propagation, deterministic dropout, zero-dropout legacy equivalence, evaluation purity and weighted loss.
- Add `LoraTrainingLoopTest.java` using test doubles: deterministic split, train/validation disjointness, weighted aggregation, patience/min-delta, best restoration, optimizer reset, schedule update count, one-unit fallback, distinct stop reasons.
- Extend `LoraTrainerTest.java` with gated validation/dropout integration tests.

## 10. Documentation and verification

- Update `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `docs/LoRA.md`, `docs/howto.md`, `docs/features.md`, `docs/arch.md`, and `docs/agent-arch.txt`.
- Document exact schedule equations, A-only AdamW semantics, LoRA+ parameter groups, train-only deterministic dropout, validation split units, restore-best behavior, and the fact that resumed checkpoints start fresh optimizer/scheduler state.
- Run `mvn test -pl lora`, `mvn test -pl node -am`, `mvn test -pl juno-player -am`, then the full suite.
- Run launcher help smoke tests and two identical seeded TinyLlama runs; compare loss/LR traces and adapter bytes. Run a held-out Q&A phrasing test and verify inference is dropout-free.

## Implementation todos

1. Write schedule, AdamW, LoRA+, dropout, evaluation, and training-loop tests first.
2. Implement schedule primitives, true A-only decoupled AdamW, and LoRA+ parameter groups.
3. Implement deterministic train-only dropout and pure forward-only loss evaluation.
4. Centralize orchestration and add deterministic held-out early stopping with best-weight restoration.
5. Wire CLI/scripts/JFR, update docs, and run deterministic and held-out verification.
