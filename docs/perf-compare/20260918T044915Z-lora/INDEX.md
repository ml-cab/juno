# LoRA perf compare — 20260918T044915Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| release-0.1.2 | 51a3b90 | 57000 | 3800.000000 | 15 | 10.086455 | 30.28585491937957 | true |
| HEAD | cb3367d | 57000 | 3800.000000 | 15 | 9.067358 | 27.65835637053152 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (cb3367d)
- train_total_ms ratio: 1
- train_ms_per_pass ratio: 1
- playback_tps ratio (wall): 0.898963808394525
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
