(ch-7-3)=
# 7.3. Performance Report

The primary Juno performance artifact is the interactive HTML matrix
[juno_test_matrix.html](https://ml.cab/juno_test_matrix.html) (model, CPU vs GPU scenarios, throughput
and latency data). Open it from a checkout in a browser; refresh or regenerate the file when
harness inputs or hardware baselines change.

Measurements tie back to JFR custom events, especially `juno.TokenProduced`, `juno.MatVec`, and
`juno.ForwardPass`. Extract `.jfr` snapshots with the metrics module as described in
[JFR and metrics](#ch-7-1). Cluster runs produce one file per JVM; the launcher
prints a per-file summary on exit. For combined percentile math across JVMs, use
`MetricsMain.extractToJsonMerged()` programmatically.

For the methodology behind these numbers, baseline hardware, and how to reproduce them, see
[Performance methodology](#ch-7-2).

## See also

- [Chapter 7.2 -- Performance Methodology](#ch-7-2)
- [Chapter 7.1 -- JFR and Metrics](#ch-7-1)
- [Chapter 2.4 -- GPU Acceleration](#ch-2-4)

---

[<- 7.2 Performance Methodology](#ch-7-2) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [8.1 Build and Test ->](#ch-8-1)