# LoRA perf compare — 20260918T204959Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| release-0.1.2 | 51a3b90 | 44000 | 2933.333333 | 15 | 14.056225 | 37.845055605929424 | true |
| HEAD | 0c519f1 | 44000 | 2933.333333 | 15 | 12.195122 | 36.718302317836226 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (0c519f1)
- train_total_ms ratio: 1
- train_ms_per_pass ratio: 1
- playback_tps ratio (wall): 0.8675958160886013
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
