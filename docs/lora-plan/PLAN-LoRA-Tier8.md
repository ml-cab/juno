# Tier 8: Train-File Scheduling and Corpus Caps

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
- `PLAN-LoRA-Tier1.md`, `PLAN-LoRA-Tier2.md` (config builder, `LoraTrainingLoop`, validation units)
- `docs/LoRA.md`, `docs/howto.md`, `CHANGELOG.md`
- `juno-player/.../ConsoleMain.java` (`/train-file`, `trainOnMasked`, hardcoded `CHUNK = 32`)
- `juno-player/.../LoraCliOptions.java`, `LoraTrainingConfig.java`, `LoraTrainingLoop.java`, `LoraTrainingSequences.java`
- `scripts/run.sh`, `scripts/run.bat`

Prerequisite: Tiers 1–2 are complete. This tier does **not** require Tier 4 GPU work. Do not change loss math, masks, clipping, or optimizer semantics.

## Overview

`/train-file` is slow on large corpora primarily because the REPL hardcodes 32-token chunks, splits the file into thousands of tiny `TrainUnit`s before the loop, and has no way to cap or subsample tokens. Docs and `/help` incorrectly say ~128.

This tier adds quality-safe **scheduling** knobs: configurable chunk length and seeded max-token / max-chunk corpus caps. Longer windows reduce launch overhead per token under truncated BPTT and usually improve or leave unchanged LM SFT quality. Caps control wall-clock without changing per-token CE.

## Scope and compatibility

Goals:

1. Expose `--lora-chunk-tokens` / `LORA_CHUNK_TOKENS` (default **32** for reproducibility with historical runs).
2. Expose `--lora-max-train-tokens` / `LORA_MAX_TRAIN_TOKENS` (`0` = unlimited).
3. Remove hardcoded `32` from `ConsoleMain` train paths; pass config into `/train`, `/train-file`, and `/train-qa`.
4. Prefer **one `TrainUnit` per document** (or capped document), then chunk inside `LoraTrainingLoop.flattenChunks`.
5. Fix doc/help drift: code default remains 32; **recommend 128 for `/train-file`**.

Non-goals:

- GPU microbatching (Tier 4 / Tier 9).
- Changing truncated BPTT, accumulation math, or loss targets as a “speed fix.”
- Document-level multi-file datasets or shuffling across files.
- Raising the default chunk to 128 silently (would change historical REPL behavior).

## Current behavior (baseline)

```
/train-file path
  -> Files.readString
  -> trainOnText -> encode (no second BOS) -> all-true mask
  -> trainOnMasked: CHUNK = 32 hardcoded
  -> each 32-token window becomes a TrainUnit
  -> LoraTrainingLoop.train(..., chunkTokens=32)
  -> flattenChunks re-chunks (usually 1:1)
```

Large files (e.g. ~100k+ tokens) produce thousands of units and thousands of full forward+backward passes per epoch.

## 1. Config and CLI

Extend `LoraTrainingConfig` builder (do not add a competing config type):

| Field | Type | Default | Semantics |
|-------|------|---------|-----------|
| `chunkTokens` | `int` | `32` | Prediction positions per chunk window (`tokens.length == chunkTokens + 1` after chunking). Must be `>= 1`. |
| `maxTrainTokens` | `int` | `0` | Cap on supervised prediction tokens (or total tokens — pick one and document). `0` = no cap. Must be `>= 0`. |

CLI / env (mirror existing `LoraCliOptions` patterns):

- `--lora-chunk-tokens N` / `LORA_CHUNK_TOKENS`
- `--lora-max-train-tokens N` / `LORA_MAX_TRAIN_TOKENS`

Validation:

- Reject non-positive chunk size.
- Reject negative max-train-tokens.
- Cap chunk size softly if needed for heap: document that very large `N` increases activation memory `O(N * layers * hidden)`; do not silently clamp without a log warning. Prefer reject above a documented hard ceiling (e.g. model context or `8192`) rather than silent truncation.

Wire through `scripts/run.sh` and `scripts/run.bat` help + flag passthrough. Update `ConsoleMain` help and LoRA banner if other training knobs are shown there.

## 2. Document-level units and chunking

In `ConsoleMain.trainOnMasked` / `trainOnText` / `trainOnQA`:

1. Build one (or few) `LoraTrainingLoop.TrainUnit` values from the full masked sequence — **do not** pre-split into 32-token units for validation/training.
2. Apply `maxTrainTokens` **before** training:
   - Prefer a dedicated helper class (KISS, new class over growing `ConsoleMain`), e.g. `LoraCorpusLimit.java` in `juno-player`.
   - Algorithm (deterministic, seeded with `--lora-seed`):
     - If `maxTrainTokens == 0` or sequence already within budget: no-op.
     - Else: chunk the sequence with `chunkTokens`, then take a **seeded subsample** of whole chunks until the prediction-token budget is met (Fisher–Yates shuffle of chunk indices with `Random(seed)`, then take prefix of shuffled list). Concatenate selected chunk windows back into train units **or** pass selected chunks as units — prefer keeping selected windows as the train set so validation split still operates on coherent units.
   - Document that this is **epoch sizing**, not identical to a full-corpus pass.
3. Pass `config.chunkTokens()` into `LoraTrainingLoop.train(..., chunkTokens)`.

`/train-qa` must use the same `chunkTokens` for consistency (usually one short unit; no behavior change when under the limit).

Do not change `LoraTrainingSequences.chunk` math (stride, `chunkTokens + 1` window, drop empty masks) except via the caller-supplied size.

## 3. Programmatic API

- `LoraTrainer.trainRawText*` already accepts `chunkTokens` in some overloads — route config defaults through `LoraTrainingConfig.chunkTokens()` when the REPL/trainer open path is used.
- Add or extend overloads only if needed; do not break legacy signatures.
- Prefer config-based methods over new positional overload sprawl.

## 4. Tests first

Add/extend:

- `LoraCliOptionsTest` (or equivalent): parse chunk/max-tokens; env override; reject invalid values.
- `LoraCorpusLimitTest` (new): unlimited path; prefix/subsample respects budget; same seed → same selected chunks; different seed → different selection when multiple chunks exist; empty/short sequences.
- `LoraTrainingSequencesTest`: existing chunk invariants still hold for 32/64/128.
- `LoraTrainingLoopTest`: document-as-single-unit + `chunkTokens=128` still token-weights loss correctly; validation split with one unit falls back safely (existing one-unit behavior).
- Optional ConsoleMain-level test only if cheap; prefer unit tests on the helper.

Do not require live TinyLlama for Tier 8 exit. Optional smoke: `/train-file` on a small fixture with `--lora-chunk-tokens 128 --lora-max-train-tokens 2048` completes and prints unit/token counts consistent with the cap.

## 5. Documentation

Update:

- `docs/LoRA.md` — `/train-file` chunk description; recommended 128 for files; max-train-tokens semantics.
- `docs/howto.md` — flag table defaults and env names.
- `docs/agent-arch.txt` — brief note on corpus limiting helper if architecture notes list training knobs.
- `CHANGELOG.md`, `RELEASE_NOTES.md` as applicable.
- `README.md` only if LoRA flag summary lives there.
- Fix `/help` string that still says “chunks of ~128 tokens” while default remains 32.

State explicitly:

- Default chunk **32** (repro).
- Recommended for large `/train-file`: `--lora-chunk-tokens 128`.
- Caps reduce work; they do not change the CE objective on included tokens.

## 6. Expected files

Likely modified:

- `juno-player/src/main/java/cab/ml/juno/player/LoraCliOptions.java`
- `juno-player/src/main/java/cab/ml/juno/player/LoraTrainingConfig.java`
- `juno-player/src/main/java/cab/ml/juno/player/ConsoleMain.java`
- `juno-player/src/main/java/cab/ml/juno/player/LoraTrainer.java` (if wiring defaults)
- `scripts/run.sh`, `scripts/run.bat`
- `docs/LoRA.md`, `docs/howto.md`, `docs/agent-arch.txt`
- `CHANGELOG.md`, `RELEASE_NOTES.md`

Likely new:

- `juno-player/src/main/java/cab/ml/juno/player/LoraCorpusLimit.java`
- `juno-player/src/test/java/cab/ml/juno/player/LoraCorpusLimitTest.java`
- CLI/config test updates

## Verification and exit gate

Exit only when:

1. REPL no longer hardcodes `CHUNK = 32`; value comes from config/CLI.
2. `--lora-chunk-tokens 128` changes printed training unit/chunk behavior on a short file.
3. `--lora-max-train-tokens` with a fixed seed is deterministic across two runs.
4. Unit tests for CLI bounds and corpus limit pass (`mvn test -pl juno-player -am`).
5. Docs/help match defaults (32) and recommendation (128 for files).
6. No change to gradient normalization, clipping, AdamW, or LoRA+ semantics (Tier 1–2 contracts intact).

## Implementation todos

1. Write `LoraCorpusLimitTest` and CLI parse tests.
2. Add config fields + CLI/env/script wiring.
3. Implement `LoraCorpusLimit` and switch `ConsoleMain` to document-level units + config chunk size.
4. Update docs/help/CHANGELOG.
5. Run `mvn test -pl juno-player -am` and a short `/train-file` smoke with chunk 128 and a token cap.
