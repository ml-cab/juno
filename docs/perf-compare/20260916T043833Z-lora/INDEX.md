# LoRA perf compare — 20260916T043833Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| release-0.1.2 | 51a3b90 | 43000 | 2866.666667 | 15 | 14.028056 | 39.111525380387945 | true |
| HEAD | 8e73d5e | 43000 | 2866.666667 | 15 | 12.704174 | 40.27658478533644 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (8e73d5e)
- train_total_ms ratio: 1
- train_ms_per_pass ratio: 1
- playback_tps ratio (wall): 0.9056261252450091
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
