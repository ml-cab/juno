# JFR and metrics

Every launcher mode accepts `--jfr DURATION` to record Java Flight Recorder with custom events (`juno.MatVec`, `juno.ForwardPass`, `juno.TokenProduced`, tokenizer events, `juno.LoraTrainStep`). Coordinator and forked nodes each emit `.jfr` files in cluster runs; `MetricsMain.extractToJsonMerged()` merges them into `target/metrics/metrics.json`.

Aggregate throughput can be read from `juno.TokenProduced` spans without extra counters; see [arch.md](../arch.md). Publishable scenario tables and CPU/GPU comparisons are in [juno_test_matrix.html](../juno_test_matrix.html); extraction CLI remains in [howto.md](../howto.md) and [performance.md](../performance.md).
