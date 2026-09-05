# LoRA perf compare — 20260905T031520Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps | recall |
|-----|--------|----------:|--------:|-------:|-------------:|:------:|
| release-0.1.2 | 51a3b90 | 50000 | 3333.333333 | 15 | 36.109947829141795 | true |
| HEAD | e0245ae | 48000 | 3200.000000 | 15 | 32.16281403112282 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (e0245ae)
- train_total_ms ratio: 0.96
- train_ms_per_pass ratio: 0.960000000096
- playback_tps ratio: 0.8906912350941277
- status: **ok**

Train/playback timings from REPL log; playback tps prefers `juno.TokenProduced.tps` from JFR when present.
