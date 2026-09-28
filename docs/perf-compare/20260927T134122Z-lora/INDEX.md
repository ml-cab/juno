# LoRA perf compare — 20260927T134122Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| cc94c53 | cc94c53 | 44000 | 2933.333333 | 15 | 12.089810 | 37.119653286996524 | true |
| HEAD | cc94c53 | 41000 | 2733.333333 | 15 | 13.207547 | 40.01596188700518 | true |

## Comparison

- baseline: cc94c53 (cc94c53)
- current: HEAD (cc94c53)
- train_total_ms ratio: 0.9318181818181818
- train_ms_per_pass ratio: 0.9318181818104339
- playback_tps ratio (wall): 1.092452817703504
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
