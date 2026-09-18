# LoRA perf compare — 20260918T063739Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| release-0.1.2 | 51a3b90 | 45000 | 3000.000000 | 15 | 13.833992 | 37.95845347057329 | true |
| HEAD | 8776a3f | 44000 | 2933.333333 | 15 | 12.455516 | 38.4892779802502 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (8776a3f)
- train_total_ms ratio: 0.9777777777777777
- train_ms_per_pass ratio: 0.9777777776666666
- playback_tps ratio (wall): 0.9003558770310116
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
