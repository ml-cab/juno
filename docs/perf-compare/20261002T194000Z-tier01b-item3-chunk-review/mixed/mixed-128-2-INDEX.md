# Mixed chunked prefill bake-off — 20261002T195857Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · schedule=continuous · n_prompt=512 · shorts=3 · prefill-batch=128 · parallel=8

## Long-prompt + short-decode concurrency (SSE)

| mode | short mean TTFT ms | short mean TPOT ms | short max TTFT ms | long TTFT ms | wall ms | proof |
|------|-------------------:|-------------------:|------------------:|-------------:|--------:|-------|
| mixed (default) | 1033.0596666666668 | 201.8157222222222 | 1039.342 | 2000.145 | 2420 | pass |
| admit-time baseline | 1459.1553333333334 | 121.65505555555553 | 1460.909 | 1467.533 | 2364 | n/a_admit_baseline |

Short TTFT mixed/admit: **0.7079847107893015**
Short TPOT mixed/admit: **1.6589176775317829**

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
| `juno.TokenProduced.tps` | 18.164082306493334 |

## Short-decode latency bound

Under this recipe, short-request TTFT max was **1039.342 ms** with mixed chunked prefill.
Documented bound for this SKU/recipe: short TTFT ≤ **1.25×** that max on re-runs (≤ **1299.1775000000002 ms**).

Script: `scripts/performance-tests/compare-mixed-prefill.sh`

### Verdict

- **PASS (TTFT):** mixed improves short TTFT vs admit-time full prefill on this load.
- **TPOT tradeoff:** short TPOT 1.66× admit-time under mix (shared steps with long ubatch); expected capacity sharing.
- Mixed prefill JFR proof **PASS** (`ContinuousStep.prefill_chunks` / `steps_with_prefill`).
