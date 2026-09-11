# LoRA perf compare — 20260911T195711Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| release-0.1.2 | 51a3b90 | 51000 | 3400.000000 | 15 | 11.945392 | 33.5983890083166 | true |
| HEAD | 57839dd | 50000 | 3333.333333 | 15 | 10.558069 | 32.74506947010566 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (57839dd)
- train_total_ms ratio: 0.9803921568627451
- train_ms_per_pass ratio: 0.9803921567647059
- playback_tps ratio (wall): 0.8838612412217196
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
