# LoRA perf compare — 20260902T200210Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall

| ref | commit | train ms | ms/pass | passes | playback tps | recall |
|-----|--------|----------:|--------:|-------:|-------------:|:------:|
| release-0.1.2 | 51a3b90 | 56722 | 3781.466667 | 15 | 10.590015 | true |
| HEAD | 5fee573 | 326470 | 21764.666667 | 15 | 7.352941 | true |

## Comparison

- baseline: release-0.1.2 (51a3b90)
- current: HEAD (5fee573)
- train_total_ms ratio: 5.755615105250167
- train_ms_per_pass ratio: 5.755615104830964
- playback_tps ratio: 0.6943277228596939
- status: **regression**
- regressions: train_total_ms, train_ms_per_pass, playback_tps
