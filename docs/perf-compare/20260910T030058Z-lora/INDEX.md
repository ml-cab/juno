# LoRA perf compare — 20260910T030058Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps | recall |
|-----|--------|----------:|--------:|-------:|-------------:|:------:|
| release-0.1.2 | 51a3b90 | 45000 | 3000.000000 | 15 | 38.25383953404865 | true |
| HEAD | 8b78382 | 44000 | 2933.333333 | 15 | 38.290114795755244 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (8b78382)
- train_total_ms ratio: 0.9777777777777777
- train_ms_per_pass ratio: 0.9777777776666666
- playback_tps ratio: 1.0009482776669856
- status: **ok**

Train/playback timings from REPL log; playback tps prefers `juno.TokenProduced.tps` from JFR when present.
