# Tier 11: Vision

Status: not started
Gap analysis refs: §1.10

## Objective

Add dynamic/high-resolution tiling ("anyres"-style tile+thumbnail composition), multi-image-per-
request support, and cluster-mode wiring for vision — today `--local` mode only. Native
dynamic-resolution ViT (2D-RoPE variable patch grids) is evaluated but only built if a real model
needing it becomes available for testing (see Models needed).

## Why this tier, why now

Vision reuses the same `MatVec`/attention primitives as the main transformer (`VisionEncoder` is
explicitly built this way), so it benefits from having Tiers 01-03's residency, attention, and KV
work already landed — a tiling implementation that runs each tile through the encoder is exactly
the kind of repeated-small-batch workload that pays for good GPU dispatch amortization. It's
sequenced after Tier 09 (tensor parallelism) so vision's cluster-mode wiring can be built against a
tensor-parallel path that's actually real, rather than the current stub.

## Scope

### In scope

1. **Dynamic/high-resolution tiling**: extend `ImagePatchEmbedder` beyond its current single
   fixed-square-resize design to support splitting a high-resolution image into tiles at native
   aspect ratio plus a global thumbnail, each independently encoded by `VisionEncoder`, matching the
   LLaVA-1.6/NeXT-style approach conceptually (implementation is Juno-native, not a port).
2. **Multi-image-per-request**: `VisionChatHandler` currently reads exactly one `image` part;
   extend it (and `VisionAwareForwardPassHandler`'s per-request patch-array storage) to accept and
   correctly splice multiple images at their respective `<image>` token positions.
3. **Cluster-mode wiring**: make vision work under `./juno cluster`, not just `--local` — this
   means `VisionAwareForwardPassHandler`'s wrapping must happen correctly on whichever node holds
   the relevant layer range (pipeline-parallel) or on every node (tensor-parallel), and image data
   must be correctly transported to the right node(s) over gRPC.

   **This requires a build-graph change, and it is not optional.** `VisionAwareForwardPassHandler`
   lives in the `vision` module, and `juno-node/pom.xml` does not depend on `vision` — only
   `juno-master` does. So today's node executable cannot wrap a handler in it at all. Add the
   `vision` dependency to `juno-node`; this does not create a cycle, since `vision` depends on
   `node`, `coordinator`, `registry`, `tokenizer` and `sampler` but not on `juno-node`. Budget for
   the shaded-jar size increase and confirm the shade plugin's relocations still behave. Do this
   first in the tier — nothing else in item 3 can be tested until the node jar can load the class.

### Out of scope

- Native dynamic-resolution ViT (Qwen2-VL-style 2D-RoPE variable patch grids) — no test model is
  currently available (see Models needed); this tier evaluates the design but only implements it if
  a suitable model is obtained, tracked as an explicit sub-decision rather than silently skipped.
- New vision-tower architectures beyond CLIP/SigLIP-style (e.g. MiniCPM-V's perceiver-resampler) —
  out of scope without a real model to validate against.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | tiling and multi-image must work without a GPU (slower, but correct) |
| 2 | CUDA GPU inference | primary target for tiling throughput |
| 3 | ROCm GPU inference | `VisionEncoder` uses the shared `MatVec` abstraction, so ROCm support should follow automatically from whatever ROCm parity Tier 10 already established — verify, don't assume |
| 4 | Static schedule | multi-image/tiled requests must compose correctly with static micro-batching alongside plain text requests |
| 5 | Continuous schedule | same, though continuous is local-mode only today — vision's continuous-mode support (if any) inherits whatever Tier 07 decided; if continuous+vision was never tested, treat it as N/A here unless Tier 07 explicitly enabled it |
| 6 | Single-node local mode | primary existing surface, must not regress |
| 7 | Pipeline-parallel cluster | primary new target for this tier — vision must work correctly end to end across a sharded pipeline |
| 8 | Tensor-parallel cluster | same, using Tier 09's real tensor-parallel slicing |
| 9 | LoRA training | N/A — vision encoder isn't LoRA-trained today; confirm this stays explicitly out of scope, not silently broken |
| 10 | LoRA playback | confirm `--lora-play` on the text backbone still works correctly for a vision-capable model (LoRA applies to the text side; vision encoder weights are untouched) |
| 11 | Vision | primary target of this entire tier |
| 12 | OpenAI REST surface | N/A directly — vision uses its own `/v1/vision/chat` endpoint, not OpenAI's chat completions; confirm this remains the case or document if this tier changes it |
| 13 | Native REST surface | `/v1/vision/chat` gets multi-image and tiling support |
| 14 | CLI | `--mmproj-path` flag behavior unchanged; cluster-mode vision flags (if any new ones are needed to enable it) documented in `--help` |

## Implementation steps

1. Write correctness tests for the current single-image, single-tile behavior as the regression
   oracle before changing `ImagePatchEmbedder`/`VisionEncoder`.
2. Implement tiling; validate against known-good reference outputs if obtainable (compare visual
   description quality/coherence, since exact numerical parity with any external reference isn't
   the goal — internal consistency and the existing `assets/Vision-I2T.md` bug-hunting methodology
   is the right validation approach, per that file's own track record of catching real bugs this
   way).
3. Implement multi-image support.
4. Wire vision into cluster mode; test pipeline-parallel first (simpler, since vision-token
   splicing only needs to happen on the node holding the input-embedding layer), then
   tensor-parallel.
5. Evaluate native dynamic-resolution ViT; implement only if a suitable model is available.

## Tests to write/upgrade before implementation

- **`ImagePatchEmbedderTest`**: new tiling cases — high-aspect-ratio images, tile-count boundary
  cases, thumbnail composition correctness.
- **`VisionEncoderTest`**: confirm per-tile encoding is independent and correctly recombined.
- **`VisionAwareForwardPassHandlerTest`**: multi-image splicing at multiple `<image>` token
  positions.
- **New cluster-mode vision integration test**: extend `ThreeNodeClusterIT` (or add a sibling) with
  a vision request routed through the pipeline-parallel cluster.
- **`ModelLiveRunnerIT`**: add a tiled/high-res image check and a multi-image check, using
  `moondream2-q5_k.llamafile`.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier11-vision.sh` — drives
  `/v1/vision/chat` with a high-resolution image (tiling path), multiple images, and a cluster-mode
  request, asserting coherent output in each case; the existing `compare-vision.sh` must continue
  passing unmodified as a regression guard for the pre-existing single-image local-mode path.
- **Perf gate (required)**: `compare-vision.sh` rerun, plus a new tiling-specific latency
  measurement (N tiles vs. single fixed-resize); publish under `docs/perf-compare/`.

  **Threshold.**
  - Existing single-image local-mode path must not regress, per the standing vision rule:
    `latency_ms` **<= 1.25x** baseline, decode tps **>= 0.80x** baseline.
  - Tiling must amortize, not merely work: an N-tile encode must cost **<= 1.3 * N** times a
    single-tile encode. A linear-or-worse scaling means each tile is paying full per-call dispatch
    and staging cost, which is the same amortization failure Tier 01B measured on the prefill path
    and is the specific risk of running the encoder once per tile.
  - Multi-image requests scale the same way against image count.
  - Cluster-mode vision end-to-end latency is **recorded with no threshold** on first delivery — it
    has no prior baseline, and this run establishes it as the reference for later tiers.

## Models needed

`moondream2-q5_k.llamafile` (present) covers multi-image and cluster-wiring validation, though it's
a lower-resolution model so may not meaningfully exercise tiling. Per [`INVENTORY.md`](INVENTORY.md),
no LLaVA-1.5/1.6 model is present — LLaVA-1.6 in particular is the architecture whose "anyres"
design this tier's tiling work is conceptually modeled on, so ask the user for a LLaVA-1.6 GGUF +
its `mmproj` file before finalizing the tiling implementation's tile-selection heuristics, so they
can be validated against the model family that actually needs them.

## Exit criteria

- [ ] Dynamic/high-resolution tiling implemented and validated (LLaVA-1.6-family model obtained and
      tested, or the tier explicitly documents why validation proceeded without one).
- [ ] Multi-image-per-request works correctly.
- [ ] `juno-node` depends on `vision`, the shaded node jar loads `VisionAwareForwardPassHandler`,
      and the jar-size change is recorded.
- [ ] Vision works end to end under pipeline-parallel and tensor-parallel cluster mode.
- [ ] Multi-image and any new cluster-vision request shape are declared in
      `api/src/main/resources/openapi.yaml` and `juno-api.yaml` (README feature-complete rule).
- [ ] Native dynamic-resolution ViT evaluated; implemented only if a suitable model was obtained,
      otherwise explicitly deferred with reasoning (not silently dropped).
- [ ] Cross-surface checklist fully resolved.
- [ ] `compare-vision.sh` passes with no regression; new tiling perf data published.
- [ ] Docs (`docs/howto.md`, `assets/Vision-I2T.md`-style tracking) updated, Juno-native language
      only.
- [ ] `CHANGELOG.md` entry added.
