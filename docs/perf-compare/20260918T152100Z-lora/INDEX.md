# LoRA perf compare — 20260918T152100Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| release-0.1.2 | 51a3b90 | 44000 | 2933.333333 | 15 | 14.112903 | 38.124386040121664 | true |
| HEAD | 5995597 | 43000 | 2866.666667 | 15 | 12.389381 | 37.05411843668645 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (5995597)
- train_total_ms ratio: 0.9772727272727273
- train_ms_per_pass ratio: 0.9772727274974173
- playback_tps ratio (wall): 0.8778761534745899
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
