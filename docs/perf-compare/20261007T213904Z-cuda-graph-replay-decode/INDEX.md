# Decode layer: CUDA graph replay against launched kernels

**Purpose.** Decide whether replaying each decode layer's device work from a captured CUDA graph
is worth wiring into the decode residency region (`--gpu-residency`). Decision rule, fixed before the
measurement: wire it if it saves at least 5% of decode forward-pass time on TinyLlama and Mistral 7B
with output unchanged; otherwise delete the graph wrapper and its test.

**Outcome: not wired; deleted.** With every kernel argument held fixed, which is the most replay can
save, it saves 2.0% to 3.1% of a TinyLlama decode forward pass and nothing on Mistral 7B, where
replay was 1.9% to 2.3% slower. Output bit-identical on every layer of both models.

**Pinned: no.** Prompt-free sudo was not available. Both lanes run in one process, alternated per
repetition (launched first on even repetitions, replayed first on odd ones), median of five with
min/max published. The GPU was hot and throttling (SM 1607 MHz against a 1911 MHz maximum, 83 C, read
after the second run). The rule needs at least 5% on both models, and Mistral 7B reads a loss at both
positions, so pinning cannot change the outcome.

## What was measured

`DecodeLayerGraphReplayBench`, a `@Tag("gpu")` test-scope measurement on a real model loaded through
`LlamaTransformerHandler` with `--gpu-residency on`, every layer run whole in the decode region, one
lease for the whole run, one KV mirror per layer holding `pos` random rows.

- **Launched lane**: `ResidentQkvPath.run` per layer, the shipped path: about 24 kernel launches per
  layer, then one download of the packed row and one wait.
- **Replayed lane**: `ResidentQkvPath.runReplayed` per layer: the same work captured once per layer
  into a CUDA graph on the region's stream (`cuStreamBeginCapture` .. `cuGraphInstantiateWithFlags`),
  then per call one `cuGraphLaunch`, followed by the same download and wait. The input upload, when
  needed, is issued as the launched lane issues it.
- **Ceiling**: a captured graph fixes every kernel argument, so the replay holds the position, the
  mirror's write offset and the attention length fixed. A wired version would also pay to move them
  every token (exec-node parameter updates, or kernels reading them from device memory). The saving
  read here is therefore an upper bound.
- **Parity**: before timing, one token runs both ways and every layer's k, v, attention output and
  layer output must match bit for bit. They did, on all 22 and 32 layers. Identical layer outputs
  give identical logits, so greedy output is unchanged.

Per token: 100 warm-up tokens per lane, then five repetitions of 200 tokens per lane.

## Results

`juno.graphReplay.pos=512` ([pos512.md](pos512.md)):

| Model | Layers | Launched ms/token (min-max) | Replayed ms/token (min-max) | Saved ms/token | Saved per layer | Replayed / launched |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B Q4_K_M | 22 | 9.532 (9.466-9.697) | 9.301 (9.265-9.555) | 0.231 | 10.5 us | 0.976 |
| Mistral 7B Q4_K_M | 32 | 34.761 (33.543-35.143) | 35.565 (33.709-35.845) | -0.804 | -25.1 us | 1.023 |

`juno.graphReplay.pos=128`, close to the sweep's decode depth ([pos128.md](pos128.md)):

| Model | Layers | Launched ms/token (min-max) | Replayed ms/token (min-max) | Saved ms/token | Saved per layer | Replayed / launched |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B Q4_K_M | 22 | 7.042 (7.002-7.434) | 6.891 (6.836-6.961) | 0.152 | 6.9 us | 0.978 |
| Mistral 7B Q4_K_M | 32 | 29.423 (28.632-29.496) | 29.990 (29.385-30.057) | -0.567 | -17.7 us | 1.019 |

Scored against the decode forward-pass time (`jfr.forward_pass_decode_total_ms / count`, median of
the three region-on runs) of the owner's pinned final-region A/B,
[`20261007T174646Z-gpu-residency-final-region-ab`](../20261007T174646Z-gpu-residency-final-region-ab/INDEX.md):
TinyLlama 7.552 ms, Mistral 7B 28.344 ms.

| Model | Saved at pos 128 | Saved at pos 512 | Threshold | Result |
|---|---|---|---|---|
| TinyLlama | 2.0% | 3.1% | >= 5% | missed |
| Mistral 7B | -2.0% | -2.8% | >= 5% | missed |

Against each lane's own launched time the TinyLlama saving is 2.2% (pos 128) and 2.4% (pos 512).

## How to read it

Replacing about 24 launch calls per layer with one graph launch removes almost all of the host's
launch work: 528 calls per TinyLlama token and 768 per Mistral 7B token. It changes a layer's time by
7 to 11 us out of about 320 to 430 us (TinyLlama), and not measurably for the better on Mistral 7B.
So the host issues a layer's launches while the device is still running the earlier ones, and a
decode layer's time is set by the device work and the one wait per layer, not by the launch calls.
The lever for decode on this card is the device kernels themselves. At pos 128, a replayed TinyLlama
layer still takes about 313 us per token with almost no launch work left on the host, while reading
its roughly 26 MB of weights at the card's 320 GB/s peak bandwidth takes about 80 us. That is an
estimate from the weight size, not a device-timed reading.

## Reproduce

The measurement code is not in the tree, since the decision removed it. It is in
[`measurement.patch`](measurement.patch) (sha256 `266be5ced677a8c5151e2a560495f9e98f4972df316b6df9ad29936c0518ae21`),
against HEAD `0a149c1`: `DecodeLayerGraphReplayBench`, the `runReplayed` hook in `ResidentQkvPath`, the
external-stream form of `CudaGraphSession`, the `cuGraphInstantiateWithFlags` binding and the replay
parity case in `ResidentQkvPathTest`.

```
git apply docs/perf-compare/20261007T213904Z-cuda-graph-replay-decode/measurement.patch   # on 0a149c1
mvn -o -pl node test -Dgroups=gpu -Dtest=DecodeLayerGraphReplayBench -Dsurefire.failIfNoSpecifiedTests=false \
  -Djuno.graphReplay.models=$PWD/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf,$PWD/models/mistral-7b-instruct-v0.1-q4_k_m.gguf \
  -Djuno.graphReplay.pos=512 -Djuno.graphReplay.out=pos512.md
```

## Build and host

- Code: HEAD `0a149c1` plus `measurement.patch`, run from the Maven test classpath (no jar).
- JDK 25.0.3; NVIDIA GeForce GTX 1080, driver 580.173.02, CUDA toolkit 12.0; clocks not pinned.
- `node` GPU tests on the measured tree: `ResidentQkvPathTest` 20 of 20 (including the new replay
  case, shown to fail when a replay is skipped and a stale row is downloaded) and
  `CudaGraphSessionTest` 3 of 3.
