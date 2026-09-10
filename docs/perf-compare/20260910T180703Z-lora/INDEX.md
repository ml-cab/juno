# LoRA perf compare — 20260910T180703Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| release-0.1.2 | 51a3b90 | 49000 | 3266.666667 | 15 | 12.544803 | 34.142824885691695 | true |
| HEAD | e137c15 | 49000 | 3266.666667 | 15 | 10.989011 | 33.48630861373139 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (e137c15)
- train_total_ms ratio: 1
- train_ms_per_pass ratio: 1
- playback_tps ratio (wall): 0.8759811533110563
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
