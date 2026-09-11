# Continuous vs static schedule — 20260911T194430Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · sessions=8 · max_tokens=64 · parallel=8 · backend=gpu

## Multi-session TPS (blocking)

| schedule | aggregate tg t/s | mean latency ms | wall ms | ok/fail |
|----------|----------------:|----------------:|--------:|--------:|
| static | 29.6571 | 15385.25 | 17264 | 8/8 |
| continuous | 25.5655 | 16081.38 | 20027 | 8/8 |

Continuous / static aggregate TPS: 0.8620364094938481

## Concurrent SSE TTFT / TPOT

| schedule | mean TTFT ms | mean TPOT ms | shared-step proof | ContinuousStep max_decode_batch | ContinuousStep shared_steps |
|----------|-------------:|-------------:|-------------------|--------------------------------:|----------------------------:|
| continuous | 6414.36275 | 208.03727777777777 | pass | 8.0 | 64.0 |
| static | 5136.628624999999 | 168.0689920634921 | n/a_static_isolated | - | - |

## Prefix cache (shared system prompt, multi-turn `x_juno_session_id`)

| schedule | lookups | hits | hit rate | first latency ms | mean later latency ms |
|----------|--------:|-----:|---------:|-----------------:|----------------------:|
| continuous | 8 | 7 | 0.875 | 5966 | 9882.86 |
| static | 8 | 7 | 0.875 | 1984 | 1566.14 |

Script: `scripts/performance-tests/compare-schedule.sh`

### Verdict (honest)

- **TPS:** continuous **0.86×** static on synchronized 8-way blocking (gather + paged path tax; not a win on this recipe).
- **SSE:** continuous **shared-step proof PASS** (`juno.ContinuousStep` max_decode_batch=8, shared_steps=64). Wall TTFT/TPOT still behind static on synchronized arrival.
- **Prefix:** continuous hit rate **0.875** (7/8) with session multi-turn; trie survives across turns (does not wipe after cohort).
- **P1 phase gate** (continuous SSE beats static): **unmet** on this synchronized workload; feature bake-off artifacts published.
