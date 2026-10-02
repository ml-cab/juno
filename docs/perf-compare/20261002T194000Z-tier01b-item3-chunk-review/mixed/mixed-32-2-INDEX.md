# Mixed chunked prefill bake-off — 20261002T195807Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · schedule=continuous · n_prompt=512 · shorts=3 · prefill-batch=32 · parallel=8

## Long-prompt + short-decode concurrency (SSE)

| mode | short mean TTFT ms | short mean TPOT ms | short max TTFT ms | long TTFT ms | wall ms | proof |
|------|-------------------:|-------------------:|------------------:|-------------:|--------:|-------|
| mixed (default) | 898.0083333333333 | 197.006 | 902.788 | 3544.375 | 3685 | pass |
| admit-time baseline | 2216.9753333333333 | 119.23361111111113 | 2219.312 | 2207.448 | 3119 | n/a_admit_baseline |

Short TTFT mixed/admit: **0.40506013749062914**
Short TPOT mixed/admit: **1.6522690050577644**

## JFR ContinuousStep (mixed)

| metric | value |
|--------|------:|
| `juno.ContinuousStep.count` | 19.0 |
| `juno.ContinuousStep.max_decode_batch` | 3.0 |
| `juno.ContinuousStep.max_prefill_chunks` | 4.0 |
| `juno.ContinuousStep.max_running_set` | 4.0 |
| `juno.ContinuousStep.prefill_chunks` | 20.0 |
| `juno.ContinuousStep.prefill_tokens` | 607.0 |
| `juno.ContinuousStep.shared_steps` | 8.0 |
| `juno.ContinuousStep.steps_with_prefill` | 17.0 |
| `juno.TokenProduced.tps` | 8.34050732593124 |

## Short-decode latency bound

Under this recipe, short-request TTFT max was **902.788 ms** with mixed chunked prefill.
Documented bound for this SKU/recipe: short TTFT ≤ **1.25×** that max on re-runs (≤ **1128.4850000000001 ms**).

Script: `scripts/performance-tests/compare-mixed-prefill.sh`

### Verdict

- **PASS (TTFT):** mixed improves short TTFT vs admit-time full prefill on this load.
- **TPOT tradeoff:** short TPOT 1.65× admit-time under mix (shared steps with long ubatch); expected capacity sharing.
- Mixed prefill JFR proof **PASS** (`ContinuousStep.prefill_chunks` / `steps_with_prefill`).
