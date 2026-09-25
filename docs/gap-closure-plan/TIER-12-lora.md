# Tier 12: LoRA

Status: not started
Gap analysis refs: §1.11

## Objective

Add native GGUF-LoRA inference (no offline conversion step required), per-request LoRA hot-swapping
(finally wiring up `x_juno_loras`, which fails closed on every schedule today), and gradient/
activation checkpointing plus QLoRA-style quantized-base-weight training.

## Why this tier, why now

Per-request LoRA hot-swapping interacts directly with whatever Tier 07 decided about continuous
batching's scheduling model (a per-request adapter needs to be selected per-slot, which only makes
sense once the scheduler's slot/token-budget model is settled) and with Tier 03's KV/prefix-cache
work (a LoRA-modified tokenized prefix already invalidates cross-turn cache hits — hot-swapping
makes this interaction more frequent, so it needs to be correct against the final KV design, not an
interim one).

## Scope

### In scope

1. **Native GGUF-LoRA inference**: load a standard GGUF-LoRA adapter directly at request/startup
   time, applying its rank-r delta the same way Juno's own `.lora` format does today
   (`LoraTrainableHandler.applyLoraInPlace`-style composition), without requiring the offline
   `GgufLoraImporter`/`juno lora-import` conversion step first. The importer/converter can remain
   available for users who want a persistent Juno-native copy, but it stops being a hard
   requirement.
2. **Per-request LoRA hot-swapping**: wire `x_juno_loras` into the forward-pass pipeline for real,
   for both static and continuous schedules — replacing `ContinuousLoraPolicy`'s current
   unconditional fail-closed rejection with actual per-request adapter selection, falling back to
   fail-closed only for combinations genuinely not supported (e.g. if QA-LoRA/DoRA adapters still
   can't participate in multi-adapter merges, per the existing `LoraPlaybackMerge` restriction —
   that specific restriction can stay, but it must stay for a real, current, re-verified reason, not
   because the whole per-request feature is unbuilt).
3. **Gradient/activation checkpointing** for LoRA training, reducing memory needed to train larger
   models/longer sequences.
4. **QLoRA-style quantized-base-weight training**: keep frozen base weights in their original
   quantized form during training (not dequantized to FP16/FP32), closing the gap flagged in
   `LoraMmqPolicy`'s existing "training ignores --mmq" comment and `QaLoraAdapter`'s explicit "this
   is not QLoRA" disclaimer.

### Out of scope

- Removing or changing the existing Juno-native `.lora` format — native GGUF-LoRA inference is
  additive, not a replacement.
- Multi-adapter merge support for QA-LoRA/DoRA specifically, unless investigation during this tier
  shows it's now tractable given the other changes — otherwise the existing restriction stays,
  re-verified and re-documented rather than silently carried forward unexamined.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | native GGUF-LoRA inference and per-request hot-swapping must work on CPU |
| 2 | CUDA GPU inference | QLoRA-style quantized training and native GGUF-LoRA inference under `--mmq` must be verified together |
| 3 | ROCm GPU inference | same, NEEDS-AMD-HARDWARE for final validation; training already has "no Q4 transpose kernel" on any backend per the gap analysis — confirm whether this tier's QLoRA-style work changes that for CUDA, ROCm, or both |
| 4 | Static schedule | per-request LoRA hot-swapping across a static micro-batch with different requests wanting different adapters in the same batch |
| 5 | Continuous schedule | same, across `ContinuousBatchEngine` slots |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | per-request adapter selection must correctly propagate to whichever node holds the relevant layer range |
| 8 | Tensor-parallel cluster | adapter deltas must be correctly sliced/composed consistently with Tier 09's real tensor-parallel weight slicing (a LoRA delta on a column-sliced weight must be sliced the same way) |
| 9 | LoRA training | primary target for checkpointing and QLoRA-style work |
| 10 | LoRA playback | primary target for native GGUF-LoRA inference and hot-swapping |
| 11 | Vision | confirm LoRA (native GGUF or hot-swapped) applies only to the text backbone and doesn't accidentally touch vision encoder weights |
| 12 | OpenAI REST surface | `x_juno_loras` actually works now — needs request-shape documentation and validation (which adapters are visible/selectable, error behavior for an unknown adapter name) |
| 13 | Native REST surface | same |
| 14 | CLI | `--lora-play` gains the ability to reference a raw GGUF-LoRA file directly, not just a converted `.lora`; `./juno lora` gains checkpointing/QLoRA-training flags |
| 15 | JVM embedding facade | per-request adapter selection and native GGUF-LoRA loading must be reachable from `JunoPlayer`/`LoraTrainer`, which are the facades an embedder uses for exactly this; a hot-swap feature available only over REST is half-wired for the surface most likely to want it |

## Implementation steps

1. Write correctness tests for the existing `.lora`-format playback path as the regression oracle.
2. Implement native GGUF-LoRA inference, reusing `GgufLoraImporter`'s existing tensor-name-mapping
   knowledge but applying deltas directly rather than writing an intermediate file.
3. Wire per-request hot-swapping into both schedules, replacing `ContinuousLoraPolicy`'s blanket
   rejection with real per-request selection plus narrower, re-justified fail-closed cases.
4. Implement gradient/activation checkpointing for training.
5. Implement QLoRA-style quantized-base-weight training.
6. Full cross-surface smoke matrix, with particular attention to mixed-adapter static/continuous
   batches.

## Tests to write/upgrade before implementation

- **New `GgufLoraImporterTest`-adjacent test** (or a new `NativeGgufLoraInferenceTest`): load a
  real converter-produced GGUF-LoRA file directly (not via the offline importer) and confirm output
  matches the equivalent offline-converted-then-loaded `.lora` path bit-for-bit — this also finally
  addresses the importer's own documented caveat that it "was never verified against a real
  converter-produced file."
- **New `ContinuousLoraPolicyTest`**: per-request adapter selection correctness across concurrent
  slots requesting different adapters; confirm the narrowed fail-closed cases (if any remain) still
  reject correctly.
- **New static-batch LoRA hot-swap test**: multiple concurrent requests, different adapters, same
  micro-batch, correct per-request output.
- **New checkpointing test**: memory-usage assertion (lower peak memory with checkpointing enabled)
  plus correctness (training converges equivalently, within tolerance, to the non-checkpointed run).
- **New QLoRA-style training test**: correctness of a quantized-frozen-weight training run compared
  to the existing FP16/FP32-frozen-weight run, and a memory-usage comparison.
- **`ModelLiveRunnerIT`**: add a native-GGUF-LoRA-inference check and a per-request hot-swap check.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier12-lora.sh` — drives
  multi-adapter hot-swapped requests through both schedules, and a QLoRA-style training run,
  end to end.
- **Perf gate (required)**: `compare-lora.sh` rerun (this script already exists specifically for
  LoRA train+playback comparisons, and it takes `--reps`; pass `--reps 3` per the README's
  noise-floor rule, since it defaults to 1) plus new checkpointing/QLoRA-specific memory and
  throughput measurements; publish under `docs/perf-compare/`.

  **Threshold.**
  - Existing `.lora` train and playback must not regress, per the standing LoRA rule: train
    **>= 0.95x** and wall-clock playback tps **>= 0.80x** of the last published baseline.
  - Gradient/activation checkpointing must reduce peak training memory by **>= 30%** on
    `tinyllama-1.1b` at the longest sequence length that fits without it, while costing **<= 1.4x**
    the training step time — checkpointing that saves no meaningful memory, or that halves training
    speed to save a little, is not worth shipping and the number is how that gets decided.
  - QLoRA-style quantized-base-weight training must reduce peak training memory by **>= 25%**
    against the current FP16/FP32-frozen path, with final validation loss within **2%** of it after
    an equal number of steps. A memory win bought with a convergence regression is a failure.
  - Per-request hot-swapping must not slow the single-adapter path: playback tps with one adapter
    selected per-request **>= 0.95x** the `--lora-play` startup-flag path it replaces.

## Models needed

`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` and its existing `.lora` adapter cover the regression oracle
and hot-swap testing. Native GGUF-LoRA inference testing needs a real GGUF-LoRA adapter file
produced by a standard converter (the existing `GgufLoraImporter` code comment notes it was never
tested against one) — ask the user for one at the point this tier starts, since without it the
native-inference path can only be validated against Juno's own offline-converted output, not
against a truly independent real-world file.

## Exit criteria

- [ ] Native GGUF-LoRA inference works, validated against a real (not Juno-produced) converter
      output file.
- [ ] Per-request LoRA hot-swapping (`x_juno_loras`) works for both schedules, with any remaining
      fail-closed restrictions re-justified and re-documented.
- [ ] Gradient/activation checkpointing implemented, measured to reduce peak training memory.
- [ ] QLoRA-style quantized-base-weight training implemented and correctness-validated.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published, every threshold above met or explicitly missed with its number.
- [ ] `x_juno_loras`'s now-working request shape, and any new LoRA fields, are declared in
      `api/src/main/resources/openapi.yaml` and `juno-api.yaml` (README feature-complete rule) —
      this field has been accepted-and-rejected by the server without ever being specified.
- [ ] Docs (`docs/howto.md` LoRA sections, `docs/agent-arch.txt`) updated.
- [ ] `CHANGELOG.md` entry added.
