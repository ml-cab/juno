# Mixed chunked prefill bake-off — 20261002T195944Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · schedule=continuous · n_prompt=512 · shorts=3 · prefill-batch=512 · parallel=8

## Long-prompt + short-decode concurrency (SSE)

| mode | short mean TTFT ms | short mean TPOT ms | short max TTFT ms | long TTFT ms | wall ms | proof |
|------|-------------------:|-------------------:|------------------:|-------------:|--------:|-------|
| mixed (default) | 1383.2093333333332 | 125.94194444444445 | 1388.52 | 1532.927 | 2325 | pass |
| admit-time baseline | 1329.5040000000001 | 120.62838888888886 | 1333.044 | 1323.133 | 2241 | n/a_admit_baseline |

Short TTFT mixed/admit: **1.040395014481591**
Short TPOT mixed/admit: **1.0440489639669308**

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
| `juno.TokenProduced.tps` | 29.056791403703585 |

## Short-decode latency bound

Under this recipe, short-request TTFT max was **1388.52 ms** with mixed chunked prefill.
Documented bound for this SKU/recipe: short TTFT ≤ **1.25×** that max on re-runs (≤ **1735.65 ms**).

Script: `scripts/performance-tests/compare-mixed-prefill.sh`

### Verdict

- **HONEST:** mixed did not improve short TTFT/TPOT vs admit-time on this recipe; artifacts published; check JFR prefill_chunks proof and arrival timing.
- Mixed prefill JFR proof **PASS** (`ContinuousStep.prefill_chunks` / `steps_with_prefill`).
