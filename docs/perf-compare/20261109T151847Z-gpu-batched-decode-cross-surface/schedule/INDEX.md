# Continuous vs static schedule — 20261109T152544Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · sessions=8 · max_tokens=64 · parallel=8 · backend=gpu

## Multi-session TPS (blocking)

| schedule | aggregate tg t/s | mean latency ms | wall ms | ok/fail |
|----------|----------------:|----------------:|--------:|--------:|
| static | 274.8256 | 1747.62 | 1863 | 8/8 |
| continuous | 265.8359 | 1794.38 | 1926 | 8/8 |

Continuous / static aggregate TPS: 0.9672894373741019

## Concurrent SSE TTFT / TPOT

| schedule | mean TTFT ms | mean TPOT ms | shared-step proof | ContinuousStep max_decode_batch |
|----------|-------------:|-------------:|-------------------|--------------------------------:|
| continuous | 410.401 | 25.367680555555555 | pass | 8.0 |
| static | 515.043625 | 67.99079166666667 | n/a_static_isolated | - |

## Prefix cache (shared system prompt, multi-turn session)

| schedule | lookups | hits | hit rate | first latency ms | mean later latency ms |
|----------|--------:|-----:|---------:|-----------------:|----------------------:|
| continuous | 8 | 7 | 0.875 | 427 | 328.14 |
| static | 8 | 7 | 0.875 | 160 | 110.71 |

Script: `scripts/performance-tests/compare-schedule.sh`

Notes:
- TPS: synchronized 8-way blocking arrival (same recipe as compare-parallel).
- SSE continuous: JFR `juno.ContinuousStep` with decodeBatchSize≥2 proves shared steps.
- Prefix: `x_juno_session_id` multi-turn; counters from `GET /v1/cluster/health`.
