# LoRA perf compare — 20260916T035640Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| release-0.1.2 | 51a3b90 | 44000 | 2933.333333 | 15 | 14.462810 | 40.7570780494295 | true |
| HEAD | c70c4dc | 42000 | 2800.000000 | 15 | 12.844037 | 39.9112237090102 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (c70c4dc)
- train_total_ms ratio: 0.9545454545454546
- train_ms_per_pass ratio: 0.9545454546539256
- playback_tps ratio (wall): 0.8880734103538663
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
