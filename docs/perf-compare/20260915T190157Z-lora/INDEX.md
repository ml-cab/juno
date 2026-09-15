# LoRA perf compare — 20260915T190157Z

Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` · backend=gpu · scenario: train-qa name recall · JFR `10m`

| ref | commit | train ms | ms/pass | passes | playback tps (wall) | tps_jfr | recall |
|-----|--------|----------:|--------:|-------:|---------------------:|--------:|:------:|
| HEAD | 4981c10 | 44000 | 2933.333333 | 15 | 12.367491 | 38.69481042383091 | true |
| HEAD | 4981c10 | 44000 | 2933.333333 | 15 | 12.367491 | 38.69481042383091 | true |


Train timings from REPL log. Playback gate uses wall-clock tokens/ms (REPL `Generated` line).
`juno.TokenProduced.tps` is informational only here (short decode after train; GC spikes dominate first→last span).
