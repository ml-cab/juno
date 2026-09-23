# Tier 06: Speculative decoding expansion

Status: not started
Gap analysis refs: §1.5

## Objective

Fix draft-model speculative decoding's measured 0.52x regression using Tier 01's activation
residency work, add lookahead decoding as a third strategy, and wire speculative decoding into the
surfaces that currently fall back to serial generation without it: `generateBatch()` (static
multi-request batching), `ContinuousBatchEngine`, and the non-`LlamaTransformerHandler`
architectures (Phi-2, Phi-3, Qwen3, Qwen3-MoE) and cluster/tensor-parallel pipelines.

## Why this tier, why now

Draft-model speculative decoding's regression is explicitly, in the gap analysis, attributed to the
same per-launch GPU dispatch overhead Tier 01 was built to fix (§1.6) — attempting to fix it before
Tier 01 lands would very likely reproduce the same "measured, doesn't help, shelved" outcome the
project has already hit twice. This tier is sequenced after sampling/grammar (Tier 05) because
speculative decoding's verify step needs to interact correctly with whatever sampler chain is
active (grammar-masked speculative decoding in particular — verifying drafted tokens against a
grammar-constrained target is a real edge case that must be tested, not assumed to work).

## Scope

### In scope

1. Re-measure draft-model speculative decoding using Tier 01's residency primitive for the draft
   model's own sequential forward passes (the exact mechanism identified as the regression's root
   cause). Confirm speedup, not just parity, before calling this done — if residency alone doesn't
   flip it positive, investigate further (e.g. a smaller/faster draft model choice, or capping
   draft-window length adaptively) rather than shipping a still-negative result silently.
2. Lookahead decoding (Jacobi-iteration n-gram-pool approach) as a third `--spec-type` option,
   alongside the existing `ngram-simple` and `draft-simple`.
3. Wire speculative decoding (whichever strategies make sense per-surface) into `generateBatch()`,
   `ContinuousBatchEngine`, and every `ForwardPassHandler` implementation — Phi-2, Phi-3, Qwen3,
   Qwen3-MoE need their own `forwardVerify` override (today only `LlamaTransformerHandler` has
   one), and cluster/tensor-parallel pipelines need a verify path that works across the gRPC
   boundary.
4. Re-verify the grammar+speculative-decoding interaction explicitly (draft tokens that would be
   grammar-illegal must be correctly rejected at verify time, not accepted).

### Out of scope

- Medusa-style multi-head speculation or self-speculative (early-exit) decoding — not named in the
  gap analysis as a currently-planned gap beyond "llama.cpp has a broader menu"; if the user wants
  these added, they should be scoped as a new tier rather than folded in here silently.
- Per-request draft-model selection via the API — the existing `--model-draft` is a process-wide
  startup flag; making it per-request is a separate, larger feature (analogous to LoRA's
  per-request hot-swap gap in Tier 12) and is out of scope here unless it turns out to be trivial
  once the rest of this tier lands.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | speculative decoding must still produce byte-identical output to `--spec-type none` on CPU (existing correctness guarantee, must not regress) |
| 2 | CUDA GPU inference | primary target for the residency-based fix |
| 3 | ROCm GPU inference | NEEDS-AMD-HARDWARE for the residency-dependent fix; the algorithmic wiring into batch/continuous/other-architecture paths should work on ROCm's existing (slower) path too, just without the residency speedup |
| 4 | Static schedule | `generateBatch()` gets draft/verify wired in — must not violate `BatchConfig`'s "all requests start at step 0" constraint in a way that breaks the batch abstraction |
| 5 | Continuous schedule | `ContinuousBatchEngine` gets draft/verify wired in — must correctly interact with mixed prefill/decode step planning |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | verify step must work when target/draft models' layers are split across nodes — likely requires the draft model to run entirely on one node (document this constraint explicitly if it's required, rather than attempting full pipeline-splitting of the draft model in this tier) |
| 8 | Tensor-parallel cluster | same consideration for tensor-sliced draft/target models |
| 9 | LoRA training | N/A |
| 10 | LoRA playback | speculative decoding + `--lora-play` interaction must be tested — a LoRA-modified target model's accept/reject decisions must stay correct |
| 11 | Vision | speculative decoding is text-token generation; vision's image-token positions must be excluded from drafting (they're not sampled) — verify this explicitly |
| 12 | OpenAI REST surface | `--spec-type`/`--model-draft` remain CLI/startup flags per current design; confirm response correctness is unaffected regardless of speculative decoding being on |
| 13 | Native REST surface | same |
| 14 | CLI | new `lookahead` value for `--spec-type`; updated `--help` text |

## Implementation steps

1. Re-run the exact Phase-2/draft-simple benchmark methodology from `docs/performance.md` with
   Tier 01's residency wired into the draft model's forward pass, before any other change in this
   tier — this is the tier's key go/no-go checkpoint.
2. If positive: proceed to wire draft-simple and ngram-simple into `generateBatch()` and
   `ContinuousBatchEngine`.
3. Add `forwardVerify` overrides for Phi-2, Phi-3, Qwen3, Qwen3-MoE (mirroring
   `LlamaTransformerHandler`'s existing batched-verify design).
4. Implement lookahead decoding as a new strategy.
5. Add cluster/tensor-parallel verify-path support, with the single-node-draft-model constraint
   documented if that's what implementation reveals is necessary.
6. Explicitly test grammar + speculative decoding together.

## Tests to write/upgrade before implementation

- **Re-run existing draft-simple JFR-based benchmark** (the one that produced the 0.52x finding) as
  a before/after comparison — this is a test in the sense of a go/no-go gate, tracked the same way
  as a perf-compare run.
- **New `forwardVerify` unit tests** for Phi-2, Phi-3, Qwen3, Qwen3-MoE handlers, mirroring the
  existing `LlamaTransformerHandlerVerifyParityTest`.
- **New `GenerationLoop.generateBatch()` speculative-decoding test**: multiple concurrent requests,
  draft/verify wired in, output byte-identical to `--spec-type none` for each.
- **New `ContinuousBatchEngine` speculative-decoding test**: same correctness guarantee across
  mixed prefill/decode steps.
- **New grammar+speculative-decoding test**: a grammar that would reject some drafted tokens;
  confirm verify correctly falls back to the grammar-legal token, not the raw drafted one.
- **New LoRA-playback+speculative-decoding test**.
- **`ModelLiveRunnerIT`**: add checks for each newly-wired surface (batch, continuous, each
  architecture, cluster).
- **New bash smoke script**: `scripts/performance-tests/smoke-tier06-speculative-decoding.sh` —
  exercises `lookahead`, batch-mode drafting, and continuous-mode drafting end to end, diffing
  output against `--spec-type none` for correctness.
- **Perf gate (required)**: this tier's entire point is a performance fix — full
  `compare-lora.sh` plus the dedicated draft-model/ngram/lookahead comparison, plus
  `compare-llama-cpp.sh` for a llama.cpp-relative reading (per README's llama.cpp-relative gate, and
  directly relevant here since llama.cpp has the same sequential-draft-forward structure — see gap
  analysis §1.5); publish under `docs/perf-compare/`.

## Models needed

`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` as the draft model and `mistral-7b-instruct-v0.1-q4_k_m
.gguf` as the target (matching the existing benchmark setup that produced the 0.52x finding) are
already present and sufficient. Phi-2/Phi-3/Qwen3/Qwen3-MoE `forwardVerify` testing needs a working
model per architecture — `Phi-3.5-mini-instruct-Q4_K_M.gguf` is present; a plain Phi-2, a plain
`qwen3` (non-MoE, non-3.5), and a working `qwen3moe` model are not (see
[`INVENTORY.md`](INVENTORY.md)) — flag to the user at the point this tier reaches those specific
sub-checks if they're still missing.

## Exit criteria

- [ ] Draft-model speculative decoding measured *faster* than `--spec-type none` on the GTX 1080
      benchmark hardware (not just improved — actually positive), or the tier documents explicitly
      why residency alone wasn't sufficient and what alternative was shipped instead.
- [ ] Lookahead decoding implemented, correctness-verified (byte-identical to `--spec-type none`).
- [ ] `generateBatch()` and `ContinuousBatchEngine` both support speculative decoding.
- [ ] Phi-2, Phi-3, Qwen3, Qwen3-MoE each have a `forwardVerify` override.
- [ ] Cluster/tensor-parallel verify path works or is explicitly, documentedly constrained.
- [ ] Grammar + speculative decoding interaction tested and correct.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published showing the fix.
- [ ] Docs updated.
- [ ] `CHANGELOG.md` entry added.
