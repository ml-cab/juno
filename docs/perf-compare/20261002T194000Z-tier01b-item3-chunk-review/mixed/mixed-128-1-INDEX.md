# Mixed chunked prefill bake-off — 20261002T195833Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · schedule=continuous · n_prompt=512 · shorts=3 · prefill-batch=128 · parallel=8

## Long-prompt + short-decode concurrency (SSE)

| mode | short mean TTFT ms | short mean TPOT ms | short max TTFT ms | long TTFT ms | wall ms | proof |
|------|-------------------:|-------------------:|------------------:|-------------:|--------:|-------|
| mixed (default) | 1003.7356666666666 | 202.44977777777777 | 1006.346 | 1977.876 | 2399 | pass |
| admit-time baseline | 1455.323 | 121.82161111111112 | 1457.062 | 1450.765 | 2363 | n/a_admit_baseline |

Short TTFT mixed/admit: **0.6896995832998355**
Short TPOT mixed/admit: **1.6618543781458222**

## JFR ContinuousStep (mixed)

| metric | value |
|--------|------:|
| `juno.ContinuousStep.count` | 9.0 |
| `juno.ContinuousStep.max_decode_batch` | 4.0 |
| `juno.ContinuousStep.max_prefill_chunks` | 4.0 |
| `juno.ContinuousStep.max_running_set` | 4.0 |
| `juno.ContinuousStep.prefill_chunks` | 8.0 |
| `juno.ContinuousStep.prefill_tokens` | 607.0 |
| `juno.ContinuousStep.shared_steps` | 8.0 |
| `juno.ContinuousStep.steps_with_prefill` | 5.0 |
| `juno.TokenProduced.tps` | 18.08729496491128 |

## Short-decode latency bound

Under this recipe, short-request TTFT max was **1006.346 ms** with mixed chunked prefill.
Documented bound for this SKU/recipe: short TTFT ≤ **1.25×** that max on re-runs (≤ **1257.9325 ms**).

Script: `scripts/performance-tests/compare-mixed-prefill.sh`

### Verdict

- **PASS (TTFT):** mixed improves short TTFT vs admit-time full prefill on this load.
- **TPOT tradeoff:** short TPOT 1.66× admit-time under mix (shared steps with long ubatch); expected capacity sharing.
- Mixed prefill JFR proof **PASS** (`ContinuousStep.prefill_chunks` / `steps_with_prefill`).
