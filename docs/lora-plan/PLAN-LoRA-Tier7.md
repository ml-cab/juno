# Tier 7: LoRA JFR Metrics for All Modes

## Agent handoff

Read and follow `models/CLAUDE.md`, `PLAN-LoRA-ROADMAP.md`, and Tiers 1–6 before implementing. Also read:

- Existing events: `LoraTrainEvent`, `LoraValidationEvent`, `MatVecEvent`, `ForwardPassEvent`
- Extraction: `JfrMetricsExtractor`, `MetricsMain`, `MetricsSnapshot`
- Emission sites: `LoraTrainingLoop`, `LoraTrainableHandler`, `ConsoleMain` (local/cluster JFR lifecycle)
- Launchers: `scripts/run.sh` / `scripts/run.bat` LoRA `--jfr` path
- Docs: `docs/LoRA.md`, `docs/performance.md`, `docs/features.md`, `docs/howto.md`

Prefer focused new event classes and guarded extractor field reads over growing `LoraTrainEvent` into an unbounded bag. Preserve existing JFR field names; new fields are additive and optional for older recordings.

## Overview

Inference modes (`local`, `cluster`) already manage programmatic JFR and auto-print metrics on exit. LoRA training emits `juno.LoraTrainStep` and optional `juno.LoraValidation`, but coverage is incomplete:

1. **Lifecycle gap.** LoRA `--jfr` uses JVM `-XX:StartFlightRecording` via `run.sh`/`run.bat` and does not auto-extract metrics on exit the way local/cluster do.
2. **Mode identity gap.** Train-step events do not record adapter algorithm (`lora` / `dora` / `qa-lora`), scaling (`standard` / `rslora`), initialization, architecture, train device, rank, alpha, targets, or QA group width.
3. **Extractor gap.** `JfrMetricsExtractor` only aggregates train-step forward/backward/optimizer p95. It ignores validation events, Tier-4 timing subsets, clip/LR fields, and has no merge/playback/DoRA-refresh series.
4. **Operation gap.** Merge (`juno merge`), sidecar playback (`--lora-play`), DoRA norm refresh, and checkpoint save/load have no first-class JFR events, so quality and latency cannot be compared across modes from one metrics JSON.

This tier makes every supported LoRA adapter mode and operational path emit, extract, and document comparable JFR metrics.

Supported adapter modes (algorithm × scaling):

| Mode label | `LoraMode` / scaling | Notes |
|------------|----------------------|-------|
| `lora` | `LORA` + `standard` | Plain LoRA |
| `rslora` | `LORA` + `rslora` | Same train event; scaling field distinguishes |
| `dora` | `DORA` | Plus norm-refresh event |
| `qa-lora` | `QA_LORA` | Plus group-width / merge-capability fields |

Supported operational paths:

| Path | Primary events |
|------|----------------|
| Train (optimizer update) | `juno.LoraTrainStep` |
| Held-out validation | `juno.LoraValidation` |
| Sidecar playback / chat with adapters | `juno.ForwardPass` + `juno.LoraPlayback` (session summary) |
| F32 / projected / sidecar merge | `juno.LoraMerge` |
| DoRA norm refresh | `juno.LoraNormRefresh` |
| Checkpoint save / load | `juno.LoraCheckpoint` (optional, low volume) |

## Prerequisites and delivery gates

- Tiers 1–3 contracts for train-step and validation emission are required.
- Tier 4 timing subset fields on `LoraTrainEvent` already exist; this tier must fill and extract them when the instrumented path runs (zeros remain valid for CPU-only).
- Tier 5 merge metrics (`QuantizedMergeMetrics`) feed `juno.LoraMerge` fields; do not invent a second metrics type.
- Tier 6 architecture identity (when present) is recorded as a string field; Tier 7 must not hard-code LLaMA-only assumptions.
- Tier 7 is independent of remaining Tier 4/5/6 product work once the corresponding code paths exist. Implement extraction and events behind guarded field reads so older `.jfr` files still parse.

## Scope

Goals:

1. Unify LoRA JFR lifecycle with local/cluster: programmatic recording, dump on exit/duration, auto-extract metrics JSON.
2. Tag every train/validation/merge/playback event with stable mode identity fields.
3. Extend `JfrMetricsExtractor` so metrics JSON covers all LoRA modes and operations with percentiles, counts, and last/mean quality scalars.
4. Emit dedicated events for merge, DoRA norm refresh, and playback session summaries.
5. Document the LoRA metrics contract in `docs/performance.md` and `docs/LoRA.md`.

Non-goals:

- Replacing JDK Mission Control browsing; JFR remains the source of truth.
- Distributed multi-node LoRA training metrics (training stays single-process).
- Claiming exact QA-LoRA K-quant merge quality from JFR alone.
- Changing optimizer or adapter math; this tier is observability only.
- Building a full Grafana/Prometheus exporter (JSON + console summary is enough).

## 1. Shared mode identity contract

Add a small immutable descriptor used when committing events (prefer a package-private helper over duplicating strings at every call site):

```text
LoraMetricsIdentity
  algorithm: lora | dora | qa-lora
  scaling: standard | rslora
  init: kaiming-uniform | legacy-normal | …
  architecture: llama | qwen2 | phi3 | qwen3 | …
  trainDevice: cpu | cuda | rocm | auto-resolved label
  rank, alpha, effectiveScale
  targets: comma-separated logical keys (wq,wv,…)
  groupWidth: QA-LoRA only (0 otherwise)
  mergeCapability: sidecar | f32 | source-type-projected | …
```

Rules:

- Values must match CLI/checkpoint vocabulary (`--lora-mode`, `--lora-scaling`, `AdapterAlgorithm`, `MergeCapability`).
- Never invent a second spelling (e.g. do not emit `QA_LORA` in JFR if CLI uses `qa-lora`).
- Identity fields are copied onto train, validation, merge, norm-refresh, and playback events so one recording can be filtered by mode without correlating external logs.

## 2. Phase A — LoRA JFR lifecycle parity

Bring LoRA mode in line with local/cluster:

1. In `ConsoleMain`, when `loraMode && jfrDuration != null`, start a programmatic `jdk.jfr.Recording` (mirror `startLocalJfr`), run the REPL, stop/dump on duration expiry or REPL exit, then call `MetricsMain.extractToJson`.
2. Stop relying on JVM `-XX:StartFlightRecording` for LoRA in `scripts/run.sh` / `scripts/run.bat`; pass `--jfr DURATION` as an app argument exactly like local mode.
3. Keep filename convention: `juno-<modelStem>-<timestamp>.jfr`.
4. Print the same "JFR Metrics Summary" banner used by local mode, including the metrics JSON path.
5. Preserve a documented escape hatch for manual JVM-flag recording if needed for Mission Control-only workflows; product default is programmatic + auto-extract.

Gate A:

- `./juno lora --model-path … --jfr 30s` writes a `.jfr`, extracts `target/metrics/metrics.json`, and prints LoRA train-step counts after at least one `/train` or `/train-qa`.
- Scripts no longer inject `-XX:StartFlightRecording` for the LoRA path.
- Local/cluster JFR behavior is unchanged.

## 3. Phase B — Enrich train and validation events

### `juno.LoraTrainStep` (extend, do not rename)

Add additive fields (guarded reads in extractor):

- Mode identity: `algorithm`, `scaling`, `initialization`, `architecture`, `trainDevice`, `rank`, `alpha`, `effectiveScale`, `targets`, `groupWidth`
- Quality already present: `loss`, `globalGradNorm`, `clipScale`, `clipped`, A/B LR, LoRA+ ratio, dropout
- Timing already present: forward/backward/optimizer/total + Tier-4 subsets (`frozenForwardMs`, `frozenTransposeBackwardMs`, `adapterBackwardMs`, `attentionNonlinearMs`, `transferMs`)
- Fill Tier-4 subset timings wherever the instrumented handler path can attribute time; leave zeros when not attributable (CPU path without split timers)

Wire identity from `LoraTrainingConfig` / adapter set / handler layout at every emission site:

- `LoraTrainingLoop.stepOptimizer`
- `ConsoleMain` legacy train-step helper (if still used)
- `LoraTrainableHandler.trainStep` legacy path

### `juno.LoraValidation` (extend)

Add: identity fields, `passIndex` / `optimizerStep`, `trainLossAtEval` (optional), `durationMs` (already present), `bestSoFar`.

Ensure every validation evaluation in `LoraTrainingLoop` commits the event (already present via `commitValidation`); verify REPL and `LoraTrainer` both hit that path.

Gate B:

- Synthetic recording with LoRA, rsLoRA, DoRA, and QA-LoRA each produce train-step events whose `algorithm`/`scaling`/`groupWidth` match the config.
- Validation-enabled run produces `juno.LoraValidation` events with matching identity.
- Older recordings without new fields still extract without error.

## 4. Phase C — New operation events

### `juno.LoraNormRefresh` (DoRA)

Emit once per full or partial norm-cache refresh:

- Identity fields
- `layerCount`, `projectionCount`
- `durationMs`
- `bytesTouched` (optional estimate)
- `reason`: `load` | `post-step` | `reset` | `explicit`

Gate: DoRA train emits ≥1 refresh event; plain LoRA emits none.

### `juno.LoraMerge`

Emit once per `LoraMerge.merge` / `LoraMergeMain` completion:

- Identity + `mergeCapability` / output policy
- `tensorsPatched`, `bytesWritten`, `durationMs`
- From `QuantizedMergeMetrics` when projected: `rmse`, `maxAbsError`, `changedBlocks`, `totalBlocks`, `saturationRate`, `deltaRetention`
- `success` boolean; on failure commit with error label string (keep short, no stack traces in JFR)

Cover F32 preserve, sidecar-only report path if applicable, and `SOURCE_TYPE_PROJECTED`. Never imply exact affine K-quant merge unless `MergeCapability.EXACT_AFFINE` is actually used.

### `juno.LoraPlayback`

Low-frequency session/summary event (not per token):

- Fired when `--lora-play` handler loads adapters, and/or when a LoRA REPL inference burst ends if cheap to attribute
- Fields: identity, `adapterCount`, `loadMs`, optional `forwardCount` / `tokensGenerated` for the session window
- Per-token timing remains `juno.ForwardPass` / `juno.MatVec`; do not duplicate those

### `juno.LoraCheckpoint` (optional but recommended)

Emit on save and load:

- `operation`: `save` | `load`
- `version`, `entryCount`, `durationMs`, `bytes`
- Identity fields from checkpoint metadata

Gate C:

- `juno merge` with F32 and projected policies each produce one `LoraMerge` event; projected includes finite RMSE/deltaRetention.
- DoRA refresh events appear only for DoRA.
- Playback load produces a `LoraPlayback` event; chat tokens still appear as ForwardPass.

## 5. Phase D — Metrics extraction and console contract

Update `JfrMetricsExtractor` with guarded `hasField` reads. Emit stable JSON keys:

Train:

```text
juno.LoraTrainStep.count
juno.LoraTrainStep.forward_ms.p95
juno.LoraTrainStep.backward_ms.p95
juno.LoraTrainStep.optimizer_ms.p95
juno.LoraTrainStep.total_ms.p95
juno.LoraTrainStep.frozen_forward_ms.p95
juno.LoraTrainStep.frozen_transpose_ms.p95
juno.LoraTrainStep.adapter_backward_ms.p95
juno.LoraTrainStep.transfer_ms.p95
juno.LoraTrainStep.loss.last
juno.LoraTrainStep.loss.mean
juno.LoraTrainStep.grad_norm.p95
juno.LoraTrainStep.clipped.fraction
juno.LoraTrainStep.tokens.total
juno.LoraTrainStep.predictions.total
```

Validation:

```text
juno.LoraValidation.count
juno.LoraValidation.loss.last
juno.LoraValidation.loss.best
juno.LoraValidation.duration_ms.p95
```

Mode-tagged aggregates (when identity fields present):

```text
juno.LoraTrainStep.by_algorithm.<algo>.count
juno.LoraTrainStep.by_algorithm.<algo>.total_ms.p95
```

Merge / DoRA / playback:

```text
juno.LoraMerge.count
juno.LoraMerge.duration_ms.p95
juno.LoraMerge.rmse.last
juno.LoraMerge.delta_retention.last
juno.LoraNormRefresh.count
juno.LoraNormRefresh.duration_ms.p95
juno.LoraPlayback.count
juno.LoraPlayback.load_ms.p95
juno.LoraCheckpoint.count
```

Rules:

- Missing series → count `0` and percentile `0` (existing MatVec/ForwardPass style), never NaN in JSON.
- Do not require Mission Control to interpret results; console summary should list the LoRA keys when any LoRA event was present.
- Keep inference-only keys unchanged so the performance matrix stays compatible.

Optional: `MetricsMain` / console pretty-printer section titled `LoRA` that prints algorithm, step count, loss last/mean, train p95 breakdown, validation best, merge RMSE when present.

## 6. Tests first

Extractor / metrics module:

- Fixture or synthetic `Recording` / recorded-event doubles covering old train-step shape (no identity fields) and new shape.
- Validation, merge, norm-refresh, playback events aggregate correctly.
- Projected merge metrics map to JSON keys with finite values.
- Empty recording yields zeros, not exceptions.
- Mode-tagged `by_algorithm` keys only appear when identity fields exist (or always with 0 — pick one rule and test it).

Emission / player-node:

- LoRA / rsLoRA / DoRA / QA-LoRA train each set distinct identity fields.
- Validation commit path covered in `LoraTrainingLoopTest`.
- DoRA refresh emits `LoraNormRefresh`; LoRA does not.
- `LoraMerge` event fields match `QuantizedMergeMetrics` for a tiny synthetic tensor.
- Programmatic LoRA JFR lifecycle smoke (gated or short unit with `Recording` API): start → emit one train event → stop → extract → assert count ≥ 1.

Script smoke:

- Help text for `./juno lora --help` documents `--jfr` parity with local mode.
- Launcher no longer sets `-XX:StartFlightRecording` for LoRA when `--jfr` is passed as an app arg.

## 7. Expected files

Likely new:

- `node/src/main/java/cab/ml/juno/node/LoraMetricsIdentity.java` (or player-side helper if identity is assembled only at commit sites)
- `node/src/main/java/cab/ml/juno/node/LoraNormRefreshEvent.java`
- `node/src/main/java/cab/ml/juno/node/LoraMergeEvent.java`
- `node/src/main/java/cab/ml/juno/node/LoraPlaybackEvent.java`
- `node/src/main/java/cab/ml/juno/node/LoraCheckpointEvent.java`
- `metrics/src/test/java/cab/ml/juno/metrics/JfrMetricsExtractorLoraTest.java`
- node/player tests for event emission as needed

Likely modified:

- `node/src/main/java/cab/ml/juno/node/LoraTrainEvent.java`
- `node/src/main/java/cab/ml/juno/node/LoraValidationEvent.java`
- `node/src/main/java/cab/ml/juno/node/LoraMerge.java`
- `node/src/main/java/cab/ml/juno/node/LoraTrainableHandler.java` (timing fill + identity; DoRA refresh commit)
- `juno-player/src/main/java/cab/ml/juno/player/LoraTrainingLoop.java`
- `juno-player/src/main/java/cab/ml/juno/player/ConsoleMain.java` (programmatic LoRA JFR)
- `juno-player/src/main/java/cab/ml/juno/player/LoraMergeMain.java`
- `metrics/src/main/java/cab/ml/juno/metrics/JfrMetricsExtractor.java`
- `metrics/src/main/java/cab/ml/juno/metrics/MetricsMain.java` (pretty-print if needed)
- `scripts/run.sh`, `scripts/run.bat`
- `docs/LoRA.md`, `docs/performance.md`, `docs/features.md`, `docs/howto.md`, `docs/arch.md`, `docs/agent-arch.txt`
- `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`
- `PLAN-LoRA-ROADMAP.md` (Tier 7 entry; keep this file authoritative for exit gates)

## 8. Documentation and verification

Document:

- LoRA `--jfr` lifecycle parity with local mode.
- Full event catalog and JSON key table for all adapter modes and operations.
- How to compare LoRA vs DoRA vs QA-LoRA train-step p95 from one metrics file.
- That projected-merge JFR metrics are approximate requantization quality, not exact QA-LoRA closure proofs.
- That Tier-4 timing subsets may be zero on CPU-only runs.

Verification commands:

```bash
mvn test -pl metrics
mvn test -pl node -am
mvn test -pl juno-player -am
mvn test
```

Manual smokes (gated models when available):

```bash
./juno lora --model-path <gguf> --jfr 1m --lora-mode lora
# /train-qa … then exit → metrics.json has LoraTrainStep.count > 0

./juno lora --model-path <gguf> --jfr 1m --lora-mode dora
# confirm LoraNormRefresh.count > 0

./juno lora --model-path <gguf> --jfr 1m --lora-mode qa-lora --lora-group-width 32
# confirm algorithm=qa-lora and groupWidth=32 in events / by_algorithm keys

./juno merge --model <gguf> --lora <adapter> --output out.gguf --jfr 30s
# or merge under an active recording; confirm LoraMerge fields
```

## Implementation todos

1. Define `LoraMetricsIdentity` and extend train/validation events; write extractor tests for old and new shapes.
2. Implement programmatic LoRA JFR lifecycle + launcher script parity; auto-extract on exit.
3. Emit and extract DoRA norm-refresh, merge, playback, and checkpoint events.
4. Fill Tier-4 timing subsets where attributable; aggregate all LoRA JSON keys.
5. Update docs and run synthetic plus gated mode-by-mode verification.

## Exit gate

Exit only when:

- LoRA `--jfr` uses programmatic recording and prints extracted metrics on exit (parity with local).
- Train-step events for `lora`, `rslora`, `dora`, and `qa-lora` carry correct identity fields.
- `JfrMetricsExtractor` aggregates train, validation, merge, norm-refresh, and playback series with guarded field reads.
- DoRA-only refresh events and projected-merge quality fields appear in metrics JSON when those paths run.
- Older `.jfr` files without new fields still extract successfully.
- Docs list the LoRA metrics contract; launcher help matches behavior.
