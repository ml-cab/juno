# Mixed chunked prefill bake-off — 20260911T204721Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · schedule=continuous · n_prompt=256 · shorts=3 · prefill-batch=32 · parallel=8

## Long-prompt + short-decode concurrency (SSE)

| mode | short mean TTFT ms | short mean TPOT ms | short max TTFT ms | long TTFT ms | wall ms | proof |
|------|-------------------:|-------------------:|------------------:|-------------:|--------:|-------|
| mixed (default) | 4552.299666666667 | 928.8653208333332 | 4561.708 | 15155.325 | 17419 | pass |
| admit-time baseline | 13929.161 | 189.3152444444444 | 13938.065 | 13912.295 | 16858 | n/a_admit_baseline |

Short TTFT mixed/admit: **0.327×**
Short TPOT mixed/admit: **4.906×**

## JFR ContinuousStep (mixed)

| metric | value |
|--------|------:|
| `juno.ContinuousStep.count` | 25.0 |
| `juno.ContinuousStep.max_decode_batch` | 3.0 |
| `juno.ContinuousStep.max_prefill_chunks` | 4.0 |
| `juno.ContinuousStep.max_running_set` | 4.0 |
| `juno.ContinuousStep.prefill_chunks` | 12.0 |
| `juno.ContinuousStep.prefill_tokens` | 351.0 |
| `juno.ContinuousStep.shared_steps` | 16.0 |
| `juno.ContinuousStep.steps_with_prefill` | 9.0 |
| `juno.TokenProduced.tps` | 3.9949854857583067 |

## Short-decode latency bound

Under this recipe, short-request TTFT max was **4561.708 ms** with mixed chunked prefill.
Documented bound for this SKU/recipe: short TTFT ≤ **1.25×** that max on re-runs (≤ **5702 ms**).

Script: `scripts/performance-tests/compare-mixed-prefill.sh`

### Verdict

- **PASS (TTFT):** mixed short mean TTFT **0.327×** admit-time (4552 vs 13929 ms) — short decode is not starved by long prefill.
- **TPOT tradeoff:** short mean TPOT **4.9×** admit-time under mix (shared steps with long ubatch chunks); expected capacity sharing, not a regression of the fairness goal.
- Mixed prefill JFR proof **PASS** (`prefill_chunks=12`, `steps_with_prefill=9`, `max_decode_batch=3`).
