# Tier 6: Multi-Architecture LoRA Training

## Agent handoff

Read and follow `models/CLAUDE.md`, `PLAN-LoRA-ROADMAP.md`, and Tiers 1–3 before implementing. Also read inference handlers for Phi-3, Qwen2 (via Llama), and Qwen3, plus `ForwardPassHandlerLoader`, `LoraTrainableHandler`, `LoraProjection`, `LoraInitializer`, `LoraMerge`, tokenizer/chat-template code, tests, and docs.

Prefer new architecture-specific training handlers over growing `LoraTrainableHandler` with conditionals.

## Overview

Current LoRA training and `--lora-play` route every adapter set through `LoraTrainableHandler` after an architecture allowlist check. That handler assumes separate LLaMA-family Q/K/V/FFN tensors and LLaMA adjacent-pair RoPE. Phi-3 uses fused QKV and gate-up tensors plus NeoX RoPE. Qwen2/Qwen2.5 need frozen QKV biases. Dense Qwen3 needs per-head Q/K RMSNorm and possibly `qDim != hiddenDim`.

This tier adds architecture-specific LoRA training and playback for:

1. Dense Qwen2 / Qwen2.5
2. Phi-3 / Phi-3.5
3. Dense Qwen3

Qwen3-MoE, Qwen3.5, Qwen2-VL, and Gemma training are deferred.

## Prerequisites and delivery gates

- Tier 1 projection/layout contracts and complete LLaMA forward/backward are required.
- Tiers 2–3 are recommended so new handlers reuse shared training orchestration, optimizer groups, and checkpoint metadata.
- Tier 6 is independent of GPU training (Tier 4) and QA-LoRA (Tier 5). Ship architecture handlers on the CPU oracle first; GPU residency can follow later.
- Keep logical checkpoint keys stable: `wq,wk,wv,wo,wgate,wup,wdown`.

## Scope

Goals:

1. Replace single-handler LoRA routing with an architecture factory.
2. Keep adapter math and checkpoint keys logical and architecture-independent.
3. Map logical projections to physical GGUF tensors through layout bindings.
4. Make training tokenization and chat templates match each architecture's inference path.
5. Support F32 merge for every Tier 6 architecture, including fused Phi slices.

Non-goals:

- Qwen3-MoE router/expert training.
- Qwen3.5 hybrid DeltaNet/SSM/recurrent training.
- Qwen2-VL / multimodal training.
- Gemma LoRA until an audited dense layout exists.
- Distributed pipeline-parallel training (cross-node backward).
- Tensor-parallel adapter overlays.

## 1. Shared foundation

Prefer composition and new classes:

- `LoraTrainingHandler` interface: `computeGradients`, inference `forward`, adapters, GPU release.
- `LoraTrainingHandlerFactory` selected by `general.architecture`.
- `LoraModelLayout` / `LoraProjectionBinding`: logical key, physical GGUF tensor name, row offset/count, input/output dimensions.
- `LoraTrainingMath`: shared quantized transpose, RMSNorm backward, GQA/softmax backward, SwiGLU backward.
- Keep `LoraAdapter`, `LoraAdapterSet`, optimizer, gradient preparation, and training-loop classes unchanged where possible.

Refactor ownership:

- Keep `LoraProjection` as the stable logical-key enum.
- Move physical GGUF mapping and architecture-specific dimensions into layout bindings.
- Update `LoraInitializer` and validation to consume layouts rather than assuming every architecture matches `LlamaConfig` LLaMA dims.
- Update `ForwardPassHandlerLoader` to dispatch adapters through the factory instead of always constructing `LoraTrainableHandler`.

Architecture routing after this tier:

```text
llama / mistral / tinyllama-compatible -> LlamaLoraTrainableHandler
qwen2 / qwen2.5 dense                 -> Qwen2LoraTrainableHandler
phi3                                  -> Phi3LoraTrainableHandler
qwen3 dense                           -> Qwen3LoraTrainableHandler
qwen3moe / qwen35 / gemma / unknown   -> explicit unsupported error
```

Replace default-accept gating with an explicit allowlist. Do not silently route unverified architectures into the LLaMA trainer.

Training tokenization and templates:

- Use `tokenizer.encode(...)` and GGUF `add_bos_token` metadata. Do not inject a hard-coded BOS token id.
- Train and infer with the same chat template key from `ChatModelType` / `ChatTrainingFormats`.
- For Qwen3, training text must match inference reasoning formatting (including the closed empty `<think>` block where used).

## 2. Phase A — Qwen2 / Qwen2.5 dense

Lowest-risk target: same dense SwiGLU, GQA, and adjacent-pair RoPE structure as the LLaMA path, plus frozen QKV biases.

Implement `Qwen2LoraTrainableHandler`:

- Load optional:
  - `blk.L.attn_q.bias [H]`
  - `blk.L.attn_k.bias [KV]`
  - `blk.L.attn_v.bias [KV]`
- Forward:
  - `q = Wq * x + LoRAq(x) + bq`
  - equivalently for K and V
- Backward:
  - biases remain frozen;
  - bias backward is identity into the projection output gradient;
  - no bias parameter gradients are stored.
- Reuse LLaMA RoPE, attention, SwiGLU, and projection dimensions.

Gate A:

- Zero-adapter logits match `LlamaTransformerHandler` on a bias-bearing Qwen2 fixture.
- Finite-difference adapter gradients pass for qv and all-linear.
- Tiny synthetic overfit decreases loss.
- Exclude Qwen2-MoE and Qwen2-VL.

## 3. Phase B — Phi-3 / Phi-3.5

Dedicated handler based on `Phi3TransformerHandler`.

Physical layout:

- `attn_qkv.weight`: `[H + 2*KV, H]`
  - Q rows `[0, H)`
  - K rows `[H, H+KV)`
  - V rows `[H+KV, H+2*KV)`
- fused FFN up/gate: `[2*I, H]`
  - gate rows `[0, I)`
  - up rows `[I, 2*I)`
- separate `attn_output.weight` and `ffn_down.weight`

Forward:

1. Compute fused QKV, then apply logical `wq`/`wk`/`wv` LoRA deltas to the corresponding slices before RoPE.
2. Apply `Phi3Rope.ropeExt` with `Phi3RopeConfig`.
3. Apply `wo` LoRA to the attention output projection.
4. Slice fused gate/up, apply `wgate`/`wup` LoRA, then SwiGLU.
5. Apply `wdown` LoRA.

Backward:

1. Existing CE, output projection, RMSNorm, residual, and SwiGLU derivatives.
2. Separate adapter gradients for `wgate` and `wup`.
3. Concatenate `[gradGate, gradUp]` and compute one frozen transpose: `W_gateUp^T * concat(...)`.
4. Attention backward under truncated-KV semantics.
5. Apply the Jacobian transpose of Phi NeoX RoPE, retaining `attnFactor` in the cosine/sine scale:
   - `dx0 = cos*g0 + sin*g1`
   - `dx1 = -sin*g0 + cos*g1`
6. Separate adapter gradients for Q/K/V.
7. Concatenate `[gradQ, gradK, gradV]` and compute `W_qkv^T * concat(...)`.
8. Sum LoRA input-gradient contributions and continue through pre-attention RMSNorm.

Merge:

- Multiple logical adapters may patch one physical tensor.
- `wq`/`wk`/`wv` all patch F32 `attn_qkv.weight` at different row ranges.
- `wgate`/`wup` both patch F32 fused `ffn_up.weight`.
- Do not use a map keyed only by physical tensor name that overwrites earlier slices.

Gate B:

- Zero-adapter logits match `Phi3TransformerHandler`.
- Phi RoPE adjoint and fused transpose adjoint tests pass.
- Finite-difference gradients pass for all seven logical projections.
- Fused merge preserves both QKV and gate/up slices; merged logits match sidecar playback.
- Gated real-model smoke against Phi-3.5-mini when available.

## 4. Phase C — dense Qwen3

Dedicated handler based on `Qwen3TransformerHandler`.

Tensor layout:

- Q: `[qDim, H]`, where `qDim = numHeads * headDim`
- K/V: `[kvDim, H]`
- O: `[H, qDim]`
- Q/K norm weights: `[headDim]`, shared across heads
- Dense FFN remains gate/up `[I,H]`, down `[H,I]`

Forward state must retain:

- residual and pre-attention normalized input;
- Q and K before per-head normalization;
- Q/K after normalization and RoPE;
- attention probabilities and attention output;
- FFN gate, up, and activated hidden values.

Backward order:

1. Output projection, final RMSNorm, residual, and dense SwiGLU.
2. O projection using shape `[H,qDim]`.
3. GQA/softmax backward producing post-RoPE Q/K gradients and V gradient.
4. Exact adjoint of Qwen3 RoPE / YaRN, preserving `attnFactor` / mscale.
5. Independent per-head Q/K RMSNorm backward.
6. Q/K/V frozen matrices and LoRA adapters.
7. Pre-attention RMSNorm and residual.

Update layout/initializer/merge so WQ output and WO input use `qDim`, not a hard-coded `hiddenDim`.

Gate C:

- Zero-adapter logits match `Qwen3TransformerHandler`.
- Per-head RMSNorm and RoPE/YaRN adjoint tests pass.
- Shape test with `qDim != hiddenDim`.
- Training template token parity with inference.
- Finite-difference adapter gradients and tiny overfit succeed.
- Gated real-model smoke against dense Qwen3 when available.

## 5. Merge and playback

- Playback uses the same architecture factory as training.
- F32 remains the default merge policy for Tier 6.
- Layout-aware merge must:
  - validate logical keys against the architecture layout;
  - support one-to-one tensors for Qwen2/Qwen3/LLaMA;
  - support multi-adapter fused-slice patching for Phi-3;
  - preserve untouched tensor bytes;
  - report exact tensors and slices patched.
- Checkpoint metadata remains logical-key based. Physical layout is resolved at load time from the GGUF architecture.

## 6. Tests first

Shared:

- Factory allowlist and rejection tests.
- Layout binding tests for each architecture.
- Tokenizer BOS metadata tests: no synthetic token-id injection when `add_bos_token=false`.
- Chat-template train/inference parity tests.

Qwen2:

- Bias-bearing synthetic GGUF: base and zero-adapter logits match.
- Finite-difference adapter gradients.
- Loss decreases on a tiny synthetic model.
- Loader routes Qwen2 adapters to the Qwen2 handler.

Phi-3:

- Projection layout names, row ranges, and dimensions.
- `Phi3Rope` adjoint and finite-difference tests, including non-1 `attnFactor`.
- Fused QKV and gate/up transpose adjointness.
- Zero-adapter parity with `Phi3TransformerHandler`.
- Finite-difference gradients for all seven projections.
- Fused merge slice preservation and sidecar parity.

Qwen3 dense:

- Per-head RMSNorm finite-difference backward.
- Standard and YaRN RoPE adjoint tests.
- `qDim != hiddenDim` forward/backward shape test.
- Zero-adapter parity with `Qwen3TransformerHandler`.
- Qwen3 training-template token parity.
- Finite-difference adapter gradients and overfit.

Gated live models when present:

- Qwen2.5 instruct Q4_K_M
- Phi-3.5-mini instruct Q4_K_M
- dense Qwen3 instruct Q4_K_M

## 7. Expected files

Likely new:

- `node/src/main/java/cab/ml/juno/node/LoraTrainingHandler.java`
- `node/src/main/java/cab/ml/juno/node/LoraTrainingHandlerFactory.java`
- `node/src/main/java/cab/ml/juno/node/LoraModelLayout.java`
- `node/src/main/java/cab/ml/juno/node/LoraProjectionBinding.java`
- `node/src/main/java/cab/ml/juno/node/LoraTrainingMath.java`
- `node/src/main/java/cab/ml/juno/node/Qwen2LoraTrainableHandler.java`
- `node/src/main/java/cab/ml/juno/node/Phi3LoraTrainableHandler.java`
- `node/src/main/java/cab/ml/juno/node/Qwen3LoraTrainableHandler.java`
- corresponding unit and gated tests under `node/src/test/...` and player tests as needed

Likely modified:

- `node/src/main/java/cab/ml/juno/node/ForwardPassHandlerLoader.java`
- `node/src/main/java/cab/ml/juno/node/LoraTrainableHandler.java`
- `node/src/main/java/cab/ml/juno/node/LoraInitializer.java`
- `node/src/main/java/cab/ml/juno/node/LoraProjection.java`
- `node/src/main/java/cab/ml/juno/node/LoraMerge.java`
- `juno-player/src/main/java/cab/ml/juno/player/LoraTrainer.java`
- `juno-player/src/main/java/cab/ml/juno/player/ConsoleMain.java`
- `juno-player/src/main/java/cab/ml/juno/player/ChatTrainingFormats.java`
- `juno-player/src/main/java/cab/ml/juno/player/ChatModelType.java`
- `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`
- `docs/LoRA.md`, `docs/howto.md`, `docs/features.md`, `docs/arch.md`, `docs/agent-arch.txt`, `docs/model_support_summary.md`

## 8. Documentation and verification

- Document which architectures are trainable, which are inference-only, and which are rejected.
- Document fused Phi merge behavior and Qwen2 bias handling.
- Document Qwen3 template parity requirements.
- Do not claim MoE or Qwen3.5 LoRA support from this tier.

Verification commands:

```bash
mvn test -pl lora
mvn test -pl node -am
mvn test -pl juno-player -am
mvn test
```

Then run gated real-model smokes for each enabled architecture: train `/train-qa`, save, `--lora-play`, and merge parity.

## Implementation todos

1. Write shared factory, layouts, math helpers, and strict architecture allowlist.
2. Fix tokenization/template assumptions so BOS and chat formatting follow model metadata.
3. Implement Qwen2/Qwen2.5 dense training and parity tests.
4. Implement Phi-3 fused training, NeoX RoPE adjoint, and fused-slice merge.
5. Implement dense Qwen3 training with Q/K norm and `qDim` shapes.
6. Update docs and run synthetic plus gated real-model verification.
