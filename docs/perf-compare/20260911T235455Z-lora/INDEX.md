# LoRA perf compare — 20260911T235455Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| release-0.1.2 | 51a3b90 | 47000 | 3133.333333 | 15 | 13.461538 | 36.02172236655166 | true |
| HEAD | 60c4b85 | 47000 | 3133.333333 | 15 | 11.784512 | 36.149392192795524 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (60c4b85)
- train_total_ms ratio: 1
- train_ms_per_pass ratio: 1
- playback_tps ratio (wall): 0.875420921443003
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
