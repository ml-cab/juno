# LoRA perf compare — 20260927T235655Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| cc94c53 | cc94c53 | 44000 | 2933.333333 | 15 | 12.323944 | 37.92196749724852 | true |
| HEAD | cc94c53 | 40000 | 2666.666667 | 15 | 13.333333 | 40.0990173054126 | true |

## Comparison

- baseline: cc94c53 (cc94c53)
- current: HEAD (cc94c53)
- train_total_ms ratio: 0.9090909090909091
- train_ms_per_pass ratio: 0.9090909093078512
- playback_tps ratio (wall): 1.0819047051820425
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
