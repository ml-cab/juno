# LoRA perf compare — 20260924T185248Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| HEAD~1 | 15cb582 | 44000 | 2933.333333 | 15 | 12.411348 | 38.472907501650646 | true |
| HEAD | 1f90b68 | 45000 | 3000.000000 | 15 | 12.259194 | 37.334059346385004 | true |

## Comparison

- baseline: HEAD~1 (15cb582)
- current: HEAD (1f90b68)
- train_total_ms ratio: 1.0227272727272727
- train_ms_per_pass ratio: 1.0227272728434917
- playback_tps ratio (wall): 0.9877407353335029
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
