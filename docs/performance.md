# Performance reporting

The primary Juno performance artifact is the interactive HTML matrix **[juno_test_matrix.html](juno_test_matrix.html)** (model, CPU vs GPU scenarios, throughput and latency insights). Open it from a checkout in a browser; refresh or regenerate the file when harness inputs or hardware baselines change.

Measurements tie back to JFR custom events (especially `juno.TokenProduced`, `juno.MatVec`, `juno.ForwardPass`): extract `.jfr` snapshots with the metrics module as described in [howto.md](howto.md). Cluster runs merge per-JVM recordings via `cab.ml.juno.metrics.MetricsMain`.
