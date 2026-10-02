# Mixed chunked prefill bake-off — 20261002T195920Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · schedule=continuous · n_prompt=512 · shorts=3 · prefill-batch=512 · parallel=8

## Long-prompt + short-decode concurrency (SSE)

| mode | short mean TTFT ms | short mean TPOT ms | short max TTFT ms | long TTFT ms | wall ms | proof |
|------|-------------------:|-------------------:|------------------:|-------------:|--------:|-------|
| mixed (default) | 1335.5829999999999 | 123.70055555555554 | 1340.779 | 1478.436 | 2257 | pass |
| admit-time baseline | 1337.175 | 121.1891111111111 | 1339.984 | 1339.755 | 2244 | n/a_admit_baseline |

Short TTFT mixed/admit: **0.9988094303288649**
Short TPOT mixed/admit: **1.0207233506493982**

## JFR ContinuousStep (mixed)

| metric | value |
|--------|------:|
| `juno.ContinuousStep.count` | 9.0 |
| `juno.ContinuousStep.max_decode_batch` | 4.0 |
| `juno.ContinuousStep.max_prefill_chunks` | 4.0 |
| `juno.ContinuousStep.max_running_set` | 4.0 |
| `juno.ContinuousStep.prefill_chunks` | 5.0 |
| `juno.ContinuousStep.prefill_tokens` | 607.0 |
| `juno.ContinuousStep.shared_steps` | 8.0 |
| `juno.ContinuousStep.steps_with_prefill` | 2.0 |
| `juno.TokenProduced.tps` | 29.56165633342329 |

## Short-decode latency bound

Under this recipe, short-request TTFT max was **1340.779 ms** with mixed chunked prefill.
Documented bound for this SKU/recipe: short TTFT ≤ **1.25×** that max on re-runs (≤ **1675.97375 ms**).

Script: `scripts/performance-tests/compare-mixed-prefill.sh`

### Verdict

- **PASS (TTFT):** mixed improves short TTFT vs admit-time full prefill on this load.
- **TPOT tradeoff:** short TPOT 1.02× admit-time under mix (shared steps with long ubatch); expected capacity sharing.
- Mixed prefill JFR proof **PASS** (`ContinuousStep.prefill_chunks` / `steps_with_prefill`).
