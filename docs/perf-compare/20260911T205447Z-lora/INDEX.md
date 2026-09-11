# LoRA perf compare — 20260911T205447Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| release-0.1.2 | 51a3b90 | 54000 | 3600.000000 | 15 | 11.705686 | 31.00227823341826 | true |
| HEAD | 40c0afd | 54000 | 3600.000000 | 15 | 10.101010 | 31.053366010666174 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (40c0afd)
- train_total_ms ratio: 1
- train_ms_per_pass ratio: 1
- playback_tps ratio (wall): 0.8629148261793457
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
