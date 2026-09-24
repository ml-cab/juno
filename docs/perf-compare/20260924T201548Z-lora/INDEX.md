# LoRA perf compare — 20260924T201548Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| 1f90b68 | 1f90b68 | 45000 | 3000.000000 | 15 | 12.152778 | 37.0685620642901 | true |
| HEAD | 1f90b68 | 45000 | 3000.000000 | 15 | 12.152778 | 37.43660793664024 | true |

## Comparison

- baseline: 1f90b68 (1f90b68)
- current: HEAD (1f90b68)
- train_total_ms ratio: 1
- train_ms_per_pass ratio: 1
- playback_tps ratio (wall): 1
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
