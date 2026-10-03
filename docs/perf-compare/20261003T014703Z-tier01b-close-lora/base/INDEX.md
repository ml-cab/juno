# LoRA perf compare — 20261003T015000Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| HEAD | unknown | 41000 | 2733.333333 | 15 | 13.307985 | 38.90871179675179 | true |
| HEAD | unknown | 41000 | 2733.333333 | 15 | 13.307985 | 38.90871179675179 | true |


Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
