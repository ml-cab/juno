# Tier 5: QA-LoRA and Quantization-Preserving Merge

## Agent handoff

Read and follow `models/CLAUDE.md`, `PLAN-LoRA-ROADMAP.md`, and Tiers 1–4 before implementing. Read the QA-LoRA paper and reference implementation, then verify every assumption against Juno’s actual GGUF Q4_K/Q5_K/Q6_K codecs.

This is a research-gated tier. Do not promise exact QA-LoRA merge into GGUF K-quants. Geometric group alignment does not imply that learned affine shifts are representable by those encodings.

## Executive decision

- Do not implement NF4 QLoRA as part of this roadmap.
- Call Juno’s existing design LoRA on a quantized GGUF base, not QLoRA.
- Implement QA-LoRA only as a distinct grouped-adapter algorithm.
- Keep sidecar and F32 merge as the safe paths.
- Treat original-type Q4_K/Q5_K/Q6_K merge as approximate projected requantization until quality and codec gates pass.

## Why exact K-quant merge is not assumed

QA-LoRA pools contiguous input groups and learns a constant per-output-row shift over each group:

```text
pooled[group] = sum(input[groupStart:groupEnd])
delta = scale * B * A * pooled
```

This expands to an effective delta where every input weight in one group receives the same output-row-specific additive shift.

For an independently affine-quantized group, that shift can be absorbed into a floating zero/offset. GGUF K-quants are more constrained:

- Q4_K/Q5_K use 256-element super-block factors with discretized nested scale/min multipliers. A 32-element group aligns geometrically, but arbitrary new offsets are not generally representable.
- Q6_K uses symmetric scaled groups and no additive min/zero term. An arbitrary constant shift cannot generally be absorbed.

Therefore:

- dequantize → add grouped delta → requantize can preserve tensor type and size approximately;
- it is not the paper’s exact zero-point merge;
- exact claims require a formal closure proof and exhaustive block tests.

## 1. Establish one conformant quantization codec layer

Before QA-LoRA training integration, extract shared GGUF quantization logic from reader, matvec, and merge paths.

Add:

- `QuantizationLayout.java`: block width, sub-block width, alignment, affine/symmetric capability, type ID.
- `GgufQuantCodec.java`: decode block/row, encode through a named strategy, validate dimensions, and report reconstruction metrics.
- `GgufKQuantCodec.java`: Q4_K/Q5_K/Q6_K implementations.
- `QuantizedMergeMetrics.java`: RMSE, maximum error, saturation, changed blocks, delta retention.

Refactor:

- `GgufReader.java`
- `LlamaTransformerHandler.java`
- `LoraTrainableHandler.java`
- `LoraMerge.java`

Requirements:

- Pin llama.cpp reference vectors/version.
- Decoders must agree with reference values.
- Encoders must be explicitly versioned; do not assume the current merge encoders are canonical without differential tests.
- A no-op merge must copy raw adapted tensor bytes unchanged rather than decode/re-encode.
- Reject malformed dimensions, partial blocks, overflowed lengths, unsupported types, and corrupt data before allocation.

Gate A: stop if codec conformance and byte-preserving no-op behavior are not proven.

## 2. Implement grouped QA-LoRA math independently

Add `QaLoraAdapter.java` in the lora module:

- `inDim` must be divisible by `groupWidth`.
- `groupCount = inDim / groupWidth`.
- A shape: `[rank, groupCount]`.
- B shape: `[outDim, rank]`.
- Use sum pooling as the canonical operation to avoid hidden normalization ambiguity.
- Forward: pool input, apply A then B, then effective scaling.
- Backward:
  - accumulate A/B gradients against pooled input;
  - backpropagate the pooled gradient equally to every original input element in its group;
  - preserve Tier 1 normalization/clipping and Tier 2 optimizer semantics.

Add `AdapterAlgorithm.java` with at least `LORA`, `DORA`, and `QA_LORA`, or extend the Tier 3 mode model without creating conflicting checkpoint fields.

Tests:

- grouped forward equals multiplication by densely expanded delta;
- finite differences for A, B, and input;
- strict group/divisibility validation;
- sum-versus-average normalization protection;
- deterministic initialization and checkpoint round trip.

Gate B: stop if grouped math does not agree with the dense oracle.

## 3. Model merge capability explicitly

Add an explicit capability/policy model:

- `SIDECAR_ONLY`
- `F32_PRESERVE`
- `SOURCE_TYPE_PROJECTED`
- `EXACT_AFFINE`
- `UNSUPPORTED`

Rules:

- Never silently fall back from exact to projected merge.
- `F32_PRESERVE` remains the default.
- `EXACT_AFFINE` is unavailable for standard Q4_K/Q5_K/Q6_K unless later proof establishes closure.
- `SOURCE_TYPE_PROJECTED` always reports that it is requantization.

Tier ownership:

- Tier 1 mapping remains authoritative for projection names.
- Tier 3 formulas/fingerprints remain authoritative for LoRA/rsLoRA/DoRA F32 merge.
- Tier 5 owns output quantization and QA-LoRA expansion.

## 4. Integrate QA-LoRA with actual tensor layouts

Add a model-aware `QaLoraInitializer.java` in the node module:

- Inspect each adapted tensor’s actual GGML type; do not infer it from model filename suffix such as `_M`.
- Initial candidate group widths:
  - Q4_K/Q5_K: 32, matching sub-block geometry.
  - Q6_K: 16, matching scale-group geometry.
- Record grouping per adapter entry because mixed-quantization models can use different tensor types.
- Reject unsupported/misaligned dimensions instead of padding silently.
- Reuse Tier 1 target projections and complete backward paths.
- Reuse Tier 2 scheduling, AdamW, LoRA+, dropout, validation, and accumulation.
- Reuse Tier 3 scaling/initialization metadata where mathematically compatible.
- Treat Tier 4 GPU FP16-resident execution as a distinct numerical backend; compare it with CPU quantized execution rather than assuming equality.

Keep sidecar playback working before any quantized merge path is approved.

## 5. Extend checkpoint metadata

Extend Tier 3’s length-delimited v2 entries, or introduce the next version if v2 is already frozen.

Store per adapter:

- algorithm ID;
- rank, declared alpha, effective scaling policy;
- pooling operation;
- input group width and count;
- in/out dimensions;
- canonical projection/tensor name;
- source GGML type and quantization layout ID;
- quantizer/encoder ID and version;
- requested merge capability;
- architecture metadata;
- raw source tensor SHA-256, dimensions, and byte count;
- initialization provenance;
- A/B values.

Reject v1 export for QA-LoRA: v1 cannot distinguish grouped A from dense A safely.

Loaded metadata is authoritative. Fail on model fingerprint, tensor type, dimensions, grouping, or encoder mismatch unless an explicitly unsafe research mode is added later.

## 6. Implement explicit merge paths

### Overlay

- Apply grouped adapter at inference.
- No model mutation.
- Reference behavior for quality comparisons.

### F32 preserve

- Densely expand the grouped delta and add it to dequantized weights.
- Write adapted tensors as F32.
- Must agree with overlay logits within declared FP32 tolerance.

### Source-type projected

- Decode source tensor.
- Add grouped delta.
- Encode using the versioned codec into the original tensor type.
- Preserve non-adapted tensor bytes.
- Report per tensor:
  - source/destination type;
  - target delta norm;
  - retained-delta projection;
  - reconstruction RMSE/max error;
  - saturation/clipping rate;
  - changed block count;
  - source/output fingerprints.

### Exact affine

- Reserve for a future encoding with independently mutable floating group offsets.
- Do not expose it for Q4_K/Q5_K/Q6_K without proof.

## 7. Baseline experiment matrix

For Q4_K, Q5_K, and Q6_K compare:

1. Raw base copied unchanged.
2. Decode/encode with zero delta.
3. Ordinary LoRA overlay.
4. Ordinary LoRA F32 merge.
5. Ordinary LoRA projected source-type merge.
6. QA-LoRA overlay.
7. QA-LoRA F32 merge.
8. QA-LoRA projected source-type merge.

Sweep:

- ranks 4/8/16/32;
- qv and all-linear;
- aligned and invalid/misaligned dimensions;
- delta magnitudes below, near, and above quantization step size;
- CPU quantized and GPU FP16-resident evaluation.

Metrics:

- reconstructed-weight relative error;
- delta relative error;
- delta-retention projection;
- saturation;
- logit RMSE and KL divergence;
- top-token agreement;
- validation perplexity;
- task accuracy/recall;
- output size;
- merge time and peak heap.

Use the same base converted independently to Q4_K_M, Q5_K_M, and Q6_K with identical train/validation splits. Inspect actual per-tensor type in `_M` models.

## 8. Go/no-go gates

### Codec gate

- Decode agrees with pinned llama.cpp vectors.
- No-op merge is byte-identical.
- Encoder behavior is named/versioned and differential-tested.
- Malformed inputs fail safely.

### QA math gate

- Grouped forward/backward passes dense equivalence and finite differences.

### Exact K-merge gate

- Default no-go for Q4_K/Q5_K/Q6_K.
- Change only after formal representability proof and exhaustive block tests.

### Projected deployment gate

Initial thresholds, to be finalized before experiments:

- validation perplexity regression no greater than 1% versus QA overlay;
- task-score loss no greater than 0.5 percentage points;
- at least 99% top-token agreement on the evaluation corpus;
- materially better delta retention than ordinary LoRA projected merge;
- no NaN/Inf and saturation below an agreed limit.

Failure keeps sidecar and F32 merge as supported production paths.

### Naming gate

- Use “QA-LoRA” only for grouped input pooling plus grouped A and quantized-base training.
- Use “projected K-quant merge” for requantization.
- Never call this QLoRA.

## 9. Tests first

Required tests:

- QA pooling/dense-expansion equivalence.
- A/B/input finite differences.
- Invalid grouping and metadata rejection.
- Checkpoint corruption, duplicate keys, fingerprints, and version compatibility.
- llama.cpp golden decode and encoder differential vectors.
- Zero-delta byte preservation.
- Q4_K/Q5_K nested-scale non-closure counterexamples.
- Q6_K additive-shift non-closure counterexamples.
- Projected merge metrics and saturation.
- Overlay/F32 equivalence.
- Original-type preservation in projected mode.
- Mixed-type `_M` model handling.
- CPU/GPU comparison.
- End-to-end train/save/load/play/merge.

## 10. Expected files

Likely new:

- `lora/src/main/java/cab/ml/juno/lora/QaLoraAdapter.java`
- `lora/src/main/java/cab/ml/juno/lora/AdapterAlgorithm.java`
- `node/src/main/java/cab/ml/juno/node/QuantizationLayout.java`
- `node/src/main/java/cab/ml/juno/node/GgufQuantCodec.java`
- `node/src/main/java/cab/ml/juno/node/GgufKQuantCodec.java`
- `node/src/main/java/cab/ml/juno/node/QaLoraInitializer.java`
- `node/src/main/java/cab/ml/juno/node/QuantizedMergeMetrics.java`
- Corresponding unit/integration tests.

Likely modified:

- `LoraAdapterSet.java` and Tier 3 config/checkpoint classes.
- `LoraTrainableHandler.java`.
- `LoraMerge.java`.
- `GgufReader.java`.
- `LlamaTransformerHandler.java`.
- `LoraTrainer.java`.
- Tier 1–2 training configuration and CLI classes.
- `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`.
- `docs/LoRA.md`, `docs/arch.md`, `docs/howto.md`, `docs/features.md`, `docs/performance.md`, `docs/agent-arch.txt`.

## Implementation todos

1. Extract and verify conformant Q4_K/Q5_K/Q6_K codecs.
2. Implement grouped QA-LoRA math and dense reference tests.
3. Add explicit merge capability/policy and checkpoint metadata.
4. Integrate QA-LoRA training and sidecar/F32 playback.
5. Implement projected source-type merge with mandatory metrics.
6. Run the full experiment matrix and apply product gates.
7. Document results precisely; do not promote unsupported exact-merge or QLoRA claims.
