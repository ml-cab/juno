# LoRA perf compare — 20260927T213908Z

> **Superseded** by [`20260927T235655Z-lora`](../20260927T235655Z-lora/INDEX.md): taken on the intermediate build (batched FP16 pack loop still inline). Passed the gate (train 0.932, playback 1.081x).

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| cc94c53 | cc94c53 | 44000 | 2933.333333 | 15 | 12.522361 | 38.36935549466475 | true |
| HEAD | cc94c53 | 41000 | 2733.333333 | 15 | 13.539652 | 39.846491868235525 | true |

## Comparison

- baseline: cc94c53 (cc94c53)
- current: HEAD (cc94c53)
- train_total_ms ratio: 0.9318181818181818
- train_ms_per_pass ratio: 0.9318181818104339
- playback_tps ratio (wall): 1.0812379550469755
- status: **ok**

Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
