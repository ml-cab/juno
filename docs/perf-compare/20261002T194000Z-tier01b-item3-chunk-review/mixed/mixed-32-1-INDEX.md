# Mixed chunked prefill bake-off — 20261002T195742Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · schedule=continuous · n_prompt=512 · shorts=3 · prefill-batch=32 · parallel=8

## Long-prompt + short-decode concurrency (SSE)

| mode | short mean TTFT ms | short mean TPOT ms | short max TTFT ms | long TTFT ms | wall ms | proof |
|------|-------------------:|-------------------:|------------------:|-------------:|--------:|-------|
| mixed (default) | 903.8413333333333 | 195.58933333333334 | 907.971 | 3539.211 | 3701 | pass |
| admit-time baseline | 2226.67 | 120.15033333333332 | 2227.603 | 2226.173 | 3133 | n/a_admit_baseline |

Short TTFT mixed/admit: **0.40591615880814547**
Short TPOT mixed/admit: **1.6278717495581927**

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
| `juno.TokenProduced.tps` | 8.361320274879741 |

## Short-decode latency bound

Under this recipe, short-request TTFT max was **907.971 ms** with mixed chunked prefill.
Documented bound for this SKU/recipe: short TTFT ≤ **1.25×** that max on re-runs (≤ **1134.96375 ms**).

Script: `scripts/performance-tests/compare-mixed-prefill.sh`

### Verdict

- **PASS (TTFT):** mixed improves short TTFT vs admit-time full prefill on this load.
- **TPOT tradeoff:** short TPOT 1.63× admit-time under mix (shared steps with long ubatch); expected capacity sharing.
- Mixed prefill JFR proof **PASS** (`ContinuousStep.prefill_chunks` / `steps_with_prefill`).
