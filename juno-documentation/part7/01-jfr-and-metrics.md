(ch-7-1)=
# 7.1. JFR and Metrics

Every launcher mode accepts `--jfr DURATION` to record Java Flight Recorder data with custom
event types (`juno.MatVec`, `juno.ForwardPass`, `juno.TokenProduced`, tokenizer events,
`juno.LoraTrainStep`, and more). Coordinator and forked nodes each emit separate `.jfr` files in
cluster runs. See [Key design decisions](#ch-2-5) for how
this is wired internally.

### Metrics

```bash
# Automatic in local mode (single JVM: all juno.* events in one .jfr file)
./juno local --model-path /path/to/model.gguf --jfr 5m

# Cluster mode: coordinator + each node write separate .jfr files. On exit the launcher
# calls MetricsMain.extractToJson() once per existing file and prints each summary;
# target/metrics/metrics.json reflects the last processed file. For throughput (TPS),
# use the coordinator recording (juno.TokenProduced lives on the coordinator JVM).

# Manual extraction from .jfr files in the project root
mvn package -pl metrics -am -DskipTests
java -cp metrics/target/metrics-*.jar cab.ml.juno.metrics.MetricsMain
# Output: target/metrics/metrics.json (one snapshot per mapped .jfr in project root)
```

The JSON report includes the following `juno.TokenProduced` fields derived from the coordinator
JFR file. These are the primary throughput metrics for performance comparison:

| Field | Description |
|-------|-------------|
| `juno.TokenProduced.count` | Total tokens delivered to clients in the recording window |
| `juno.TokenProduced.elapsed_seconds` | Wall-clock span from first to last delivered token |
| `juno.TokenProduced.tps` | Aggregate tokens per second (`count / elapsed_seconds`) |

AWS cluster JFR:

```bash
./launcher.sh juno-deploy.sh setup --jfr 2m ...
# Ctrl+C -> recordings collected from all nodes -> metrics printed -> instances stopped
```

## LoRA training event catalog

LoRA training uses the same programmatic JFR recording as `./juno local --jfr`, not the JVM
`-XX:StartFlightRecording` flag. On exit, metrics are auto-extracted to
`target/metrics/metrics.json` and a console summary is printed.

```bash
./juno lora --model-path /path/to/model.gguf --jfr 1m --lora-mode dora
# train, then quit: prints JFR Metrics Summary and writes target/metrics/metrics.json
```

**Event catalog:**

| Event | Emitted when |
|-------|-------------|
| `juno.LoraTrainStep` | Once per optimizer update |
| `juno.LoraValidation` | Once per validation pass (requires `--lora-validation-split > 0`) |
| `juno.LoraNormRefresh` | Once per DoRA norm-cache refresh (DoRA only) |
| `juno.LoraMerge` | Once per `juno merge` completion |
| `juno.LoraPlayback` | Once per `--lora-play` adapter load |
| `juno.LoraCheckpoint` | Once per `/save` or checkpoint load |
| `juno.ForwardPass` | Per transformer layer forward pass |
| `juno.MatVec` | Per matVec call (shared with inference path) |

Every LoRA event carries `LoraMetricsIdentity` fields: `algorithm` (lora / rslora / dora /
qa-lora), `scaling`, `initialization`, `architecture`, `trainDevice` (cpu / cuda / rocm),
`rank`, `alpha`, `targets`, `groupWidth` (QA-LoRA only).

**Key JSON keys** (extracted to `target/metrics/metrics.json`):

```
juno.LoraTrainStep.count
juno.LoraTrainStep.forward_ms.p95
juno.LoraTrainStep.backward_ms.p95
juno.LoraTrainStep.optimizer_ms.p95
juno.LoraTrainStep.total_ms.p95
juno.LoraTrainStep.frozen_forward_ms.p95       # non-zero only on GPU path
juno.LoraTrainStep.frozen_transpose_ms.p95     # non-zero only on GPU path
juno.LoraTrainStep.adapter_backward_ms.p95
juno.LoraTrainStep.loss.last
juno.LoraTrainStep.loss.mean
juno.LoraTrainStep.grad_norm.p95
juno.LoraTrainStep.clipped.fraction
juno.LoraTrainStep.by_algorithm.<algo>.total_ms.p95

juno.LoraValidation.count
juno.LoraValidation.loss.best
juno.LoraValidation.duration_ms.p95

juno.LoraMerge.count
juno.LoraMerge.duration_ms.p95
juno.LoraMerge.rmse.last                       # non-zero for source-type-projected merges
juno.LoraMerge.delta_retention.last            # non-zero for source-type-projected merges

juno.LoraNormRefresh.count
juno.LoraNormRefresh.duration_ms.p95

juno.LoraPlayback.count
juno.LoraPlayback.load_ms.p95

juno.LoraCheckpoint.count
```

Rules:
- Missing series return 0, never NaN.
- Frozen forward/transpose timing fields are zero on CPU-only runs.
- Projected-merge RMSE and delta-retention are approximate requantization quality, not exact
  QA-LoRA closure proofs.
- Older `.jfr` files without new fields still extract cleanly via guarded field reads.

## See also

- [Chapter 2.5 -- Key Design Decisions](#ch-2-5)
- [Chapter 4.3 -- Training Guide](#ch-4-3)
- [Chapter 7.2 -- Performance Methodology](#ch-7-2)

---

[<- 6.3 Windows Notes](#ch-6-3) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [7.2 Performance Methodology ->](#ch-7-2)
