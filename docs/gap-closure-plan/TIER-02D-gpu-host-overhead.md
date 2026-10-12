# Tier 02D: GPU host overhead and decode-region coverage

Status: **complete** (2026-10-12). Every exit criterion is met: the owner's pinned closing gate met every gate
(`docs/perf-compare/20261012T002824Z-gpu-host-overhead-close-gate/`) and the closing sweeps met all three raised GPU rows.
The history of the tier follows. As of 2026-10-11: Steps 1 and 2 are done (decomposition published as
`docs/perf-compare/20261011T055415Z-gpu-host-overhead-decomposition/`). Owner decisions 1 and 2 taken: the Phi-3.5-mini
generation row is retired as met on arrival, and item 3 removes the host KV allocation by reusing tensors. Step 3 is
under way: item 2 (Qwen2 in the region) done; item 3 done (residue and host KV write removed; the two terms left by
decision 3 (a)); item 4 done (the Phi-3.5-mini decode step is kernel-bound). Step 4 done (2026-10-11): the
cross-surface smoke matrix passes, and the vision and LoRA gates are met. Next: the owner's pinned closing gate
(`dist/gpu-host-overhead-close/`), then `docs/performance.md` and the tier close.

**Added 2026-10-08 (owner decision, Tier 02 decision 15).** Tier 02's closing sweeps met every GPU end-of-plan
target with room (GPU tg 0.858x on Phi-3.5-mini and 1.030x on Mistral 7B against 0.70x; GPU pp 0.594x at 512 and
0.540x at 2048 against 0.40x and 0.25x). By this plan's rule a target met before the work meant to reach it measures
nothing, so the owner raised them, and placed the work that has to reach them in this tier, after Tier 02C and
before Tier 03.

## Objective

Bring the GPU path to the raised end-of-plan targets by removing what is left between the device kernels and the
wall clock: Qwen2's decode outside the device region, the host work around Phi-3's prefill window, and Phi-3's
remaining per-token decode cost.

## Why this tier, why now

**Three levers had no owner.** At Tier 02's close (reference column 2026-10-08, `docs/perf-compare/20261008T173153Z/`
and following):

- **Qwen2.5-3B decodes at 0.416x**, the lowest GPU tg of the four sweep models, because the device-resident decode
  region declines Qwen2: `GpuResidencyOptions` and `ResidentQkvPath` give "uses the split-half RoPE layout (and Q/K/V
  biases), which the device region does not implement". Both operations already exist as device kernels:
  `rope_split_half` (`rope.cu`) runs in Qwen3's decode region, and `bias_add` runs in the prefill window
  (`PrefillWindowRegion`). The LLaMA-family decode region uses neither. On the three models the region covers, it
  lifted decode 1.56x to 1.99x (Tier 02 item 6 sweeps).
- **Phi-3.5-mini binds both prefill rows** (0.594x at 512, 0.540x at 2048), and its 512-token window spends only
  45% of its time in GEMM, against 71% on Mistral 7B. Of the rest, device attention and elementwise kernels are 19%,
  and 35.9% is host-side or unattributed: unattributed residue 12.0%, the region's host side 9.9%, the host KV write
  8.2%, matmul staging 5.8%
  (`docs/perf-compare/20261008T054413Z-attention-close-gate/spans512/`). Mistral 7B's same terms total about 10%.
- **Phi-3.5-mini's decode** reads 0.858x. No item in a later tier moves default-flag decode: Tier 06's speculation is
  opt-in, and the reference engine does not speculate, so it does not score these rows.

It runs after Tier 02C because CPU is still the largest gap in the program (about 0.1x), and before Tier 03 because
Tier 03 changes the KV layout the host KV write sits on (Tier 03 item 6 owns the host KV storage; this tier owns the
time the prefill window spends writing it).

## Scope

### In scope

1. **Decompose before fixing.** On the current build, publish:
   (a) a Phi-3.5-mini and a Qwen2.5-3B decode step: device kernel time, host time between launches, and wait time
   per layer, from `--device-spans` at decode width plus `jdk.ExecutionSample`, at context 64 and 512;
   (b) the Phi-3.5-mini 512- and 2048-token prefill window with `prefill-breakdown.sh`, with the 12% residue
   attributed (add the span that covers it, or name the code it runs);
   (c) for each raised row, the ratio the decomposition says this tier's items can reach with everything else held.
   If a raised row is out of reach by that arithmetic, write the numbers here and restate the row with the owner
   before implementing, as Tier 02's step 2 did.
   *Corrected 2026-10-11 (step 2): (a) cannot come from `--device-spans`. `juno.DeviceCompute` and
   `juno.DeviceStaging` count decode-width launches and copies but do not time them (`DeviceComputeEvent`'s javadoc;
   `timed_count = 0` on every decode site of every earlier span run). The device side of a decode step was read from a
   CUDA activity trace (Nsight Systems, `-t cuda`) and the host side from JFR on an untraced run of the same
   configuration; method and scripts are in the published directory.*
2. **Qwen2 in the decode region.** Run the LLaMA-family decode region on split-half RoPE layouts with
   `rope_split_half`, and add the Q, K and V biases on the device with the prefill window's bias kernel, so
   Qwen2/Qwen2.5 decode runs the whole layer in the region. Same scope and fallbacks as the region today
   (single-sequence decode, CUDA, K-quant device weights), the same notice rules (`auto` silent where declined).
   Bit-identity against the op-at-a-time path by test, as for every region handler.
3. **Phi-3's prefill window host work.** Remove or overlap the terms the decomposition ranks: the region's host side
   (`projection_and_region_host`), the host KV write (`kv_host_write`, the time only; its storage is Tier 03 item 6),
   matmul staging, and whatever the attributed residue turns out to be. Apply the same change to the LLaMA family and
   Qwen3 where the term exists there.
   *Widened 2026-10-11 (owner decision 2, revised the same day): item 3 also removes the prefill residue's cause,
   the per-request allocation of the host KV tensors, by reusing a finished request's dense tensors for the next
   request. Exact growth and real FP16 storage stay in Tier 03 item 6.*
4. **Phi-3's decode step.** From item 1 (a), remove the largest term of a Phi-3.5-mini decode step that is not device
   kernel time; if the step is already bound by device kernels, say so with the number and name the kernel.

### Out of scope

- CPU work (Tier 02C), the KV layout and host KV storage (Tier 03; apart from the growth half moved into item 3), new quantization formats (Tier 04), speculative
  decoding (Tier 06), ROCm (Tier 10).
- Phi-2 and Qwen3-MoE device paths (Tier 08 item 6).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | N/A for the region and the prefill window (CUDA only); unchanged, held by the standing CPU gate |
| 2 | CUDA GPU inference | primary target |
| 3 | ROCm GPU inference | N/A or NEEDS-AMD-HARDWARE: the region is CUDA-only; `auto` resolves off on ROCm |
| 4 | Static schedule | prefill windows (item 3) at every `--parallel` width the tests cover |
| 5 | Continuous schedule | prefill chunks reach item 3's changes; the decode region stays single-sequence |
| 6 | Single-node local mode | primary dev surface |
| 7 | Pipeline-parallel cluster | forked nodes run the region on their layers; greedy output matches local mode |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | declines the region, as today; unchanged |
| 10 | LoRA playback | declines the region, as today; unchanged |
| 11 | Vision | moondream2 (Phi-2 text half) does not reach these paths; `compare-vision.sh` confirms unchanged |
| 12 | OpenAI REST surface | verified through the smoke runs |
| 13 | Native REST surface | same |
| 14 | CLI | no new flag; `--gpu-residency` help and `docs/howto.md` name Qwen2 as covered |
| 15 | JVM embedding facade | N/A: no new embedder-invocable capability |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first.
2. Item 1, the decomposition, published; restate any row it shows out of reach, with the owner.
3. Items 2, 3 and 4 in the order the decomposition ranks them, re-measuring after each.
4. Run the full cross-surface smoke matrix.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other test or code
  in this tier (README execution rule 7).
- **Qwen2 in the region (item 2)**: a `ResidentQkvPath` case per new operation (split-half rotation, Q/K/V bias)
  bit-identical to the op-at-a-time path at several positions and across concurrent threads; a handler test that a
  Qwen2.5-3B decoded token makes 2 uploads and layers + 1 downloads (`juno.DeviceStaging`), as the other region
  handlers' tests do; greedy parity on against off. `smoke-gpu-residency.sh` (unmodified) then reports Qwen2.5-3B as
  active rather than declined.
- **Prefill window (item 3)**: logits bit-identical, or within the existing `PrefillRegionHandlerParityTest` bounds,
  region on against off, on every changed handler; a byte or span assertion for each removed term.
- **`ModelLiveRunnerIT`**: check 9 at 512 and 2048 on Qwen2.5-3B and Phi-3.5-mini, unchanged contract.
- **No new smoke script**: `smoke-gpu-residency.sh` and `smoke-long-prompt-prefill.sh` already exercise both paths;
  re-run them unmodified.
- **Standing CPU and allocation gate** (README, "Test infrastructure"): run against the pre-tier jar and score it
  before closing this tier.
- **Perf gate (required)**: hot-path change (forward pass, GPU residency). `compare-llama-cpp.sh --gpu` closing
  sweeps at 128, 512 and 2048 (Phi-3.5-mini at 2048 at `COMPARE_HEAP=8g`), `compare-lora.sh` and `compare-vision.sh`,
  published under `docs/perf-compare/`.

  **Threshold** (the README's raised end-of-plan rows, scored on this tier's pinned closing sweeps):
  - GPU tg **>= 0.70x** on every sweep model (reference 0.416x, Qwen2.5-3B binding);
  - GPU tg **>= 0.90x** on Phi-3.5-mini (reference 0.858x);
  - GPU pp **>= 0.70x** at `n_prompt=512` (reference 0.594x) and **>= 0.60x** at `n_prompt=2048` (reference 0.540x),
    every sweep model, Phi-3.5-mini binding.

  Throughput must not regress: Juno tg and pp t/s **>= 0.95x** the pre-tier build on every sweep model, from a
  same-hour interleaved A/B with pinned clocks (README, "No-regression gates tighter than the floor are
  Juno-against-Juno"). Vision latency **<= 1.25x** and decode **>= 0.80x**; LoRA playback **>= 0.80x**.

## Models needed

The four sweep models and Qwen3-1.7B (the other split-half region handler, held unchanged). All present.

## Exit criteria

- [x] Item 1's decomposition published before any fix, with each raised row's reachable ratio; any row it puts out
      of reach restated with the owner first.
      *2026-10-11:* [`20261011T055415Z-gpu-host-overhead-decomposition`](../perf-compare/20261011T055415Z-gpu-host-overhead-decomposition/INDEX.md),
      on the pre-work jar `6ae135b0e21839cf`, before any code change. No row is out of reach; the Phi-3.5-mini
      generation row is already met by projection (decision 1).
- [x] Qwen2 decodes in the region: bit-identity by test, 2 uploads and layers + 1 downloads per decoded token,
      greedy identical on against off, `smoke-gpu-residency.sh` reports it active.
      *2026-10-11:* `ResidentQkvPathQwen2Test` (bit for bit, concurrent leases), `LlamaTransformerHandlerGpuResidencyTest`
      (2 up, 37 down per token; greedy equal over 19 generated tokens), the unmodified smoke (active, identical over 32
      tokens); [`20261011T125030Z-qwen2-decode-region`](../perf-compare/20261011T125030Z-qwen2-decode-region/INDEX.md).
- [x] Phi-3's prefill window host terms reduced as item 3 states, each by a span or byte assertion; logits within
      the parity bounds.
      *2026-10-11:* the residue (byte assertion, `HostKvReuseLiveTest`) and the host KV write (span: 69.3 to 0.1 ms at 512
      tokens) are removed; logits and host KV are bit-identical, region on against off (`PrefillRegionHandlerParityTest`).
      The region host side and matmul staging are left by owner decision 3 (a).
      [`20261011T162329Z-phi3-prefill-host-work`](../perf-compare/20261011T162329Z-phi3-prefill-host-work/INDEX.md).
- [x] Phi-3's decode step: the largest non-kernel term removed, or the step shown kernel-bound with the number.
      *2026-10-11:* kernel-bound, 94.6% device-busy at 55 prompt tokens and 93.6% at 503 on this tier's build; largest
      kernel `q4k_gemv` (43%); [`20261011T183740Z-phi3-decode-kernel-bound`](../perf-compare/20261011T183740Z-phi3-decode-kernel-bound/INDEX.md).
- [x] Raised end-of-plan rows (GPU tg >= 0.70x every sweep model, >= 0.90x Phi-3.5-mini; GPU pp >= 0.70x at 512,
      >= 0.60x at 2048) each reported met or missed with its number from the pinned closing sweeps.
      *2026-10-12: all met, pinned.* GPU tg 0.776x (TinyLlama, binding) to 1.066x, Qwen2.5-3B 0.828x
      ([`20261012T013642Z`](../perf-compare/20261012T013642Z/INDEX.md)); Phi-3.5-mini tg 0.973x (retired row, confirmed);
      pp 0.726x at 512 ([`20261012T015002Z`](../perf-compare/20261012T015002Z/INDEX.md)) and 0.602x at 2048
      ([`20261012T021800Z`](../perf-compare/20261012T021800Z/INDEX.md), [`20261012T020410Z`](../perf-compare/20261012T020410Z/INDEX.md)),
      Phi-3.5-mini binding at both.
      *2026-10-11: owed to the owner.* `dist/gpu-host-overhead-close/run-gate.sh` Part B runs and scores the sweeps; its
      scoring was dry-run against the Tier 02 sweeps and reproduces the reference cells (step 4 record).
- [x] Perf gate published: no regression (>= 0.95x A/B, pinned), vision and LoRA gates, and the standing CPU and
      allocation gate met.
      *2026-10-12: all met.* Pinned GPU A/B tg 0.999x to 1.843x, pp 1.017x to 1.195x, allocation 0.903x to 0.992x, no GC
      pause in span; standing CPU gate tg 0.999x and 1.005x, pp 0.983x and 1.030x, allocation 0.966x and 0.920x:
      [`20261012T002824Z-gpu-host-overhead-close-gate`](../perf-compare/20261012T002824Z-gpu-host-overhead-close-gate/INDEX.md).
      *2026-10-11:* vision (latency 0.997x, decode 1.004x) and LoRA (training time 1.000x GPU, 0.997x CPU; playback
      0.983x and 1.011x) met, unpinned, as the README allows for gates looser than 0.90x:
      [`20261011T201902Z-gpu-host-overhead-vision-lora`](../perf-compare/20261011T201902Z-gpu-host-overhead-vision-lora/INDEX.md).
      Owed to the owner: the pinned GPU A/B (Part A) and the standing CPU and allocation gate (Part C).
- [x] Cross-surface checklist fully resolved.
      *2026-10-11:* every row resolved (step 4 record, cross-surface table); the smoke matrix and the vision and LoRA
      gates are in [`20261011T201902Z-gpu-host-overhead-vision-lora`](../perf-compare/20261011T201902Z-gpu-host-overhead-vision-lora/INDEX.md)
      and the step 4 record. The throughput halves of rows 1 and 2 are the owner's gate (Parts C and A), as in Tier 02C.
- [x] Docs (`docs/howto.md` `--gpu-residency`, `docs/agent-arch.txt`, `docs/performance.md`) updated, Juno-native
      language only.
      *2026-10-12:* `docs/performance.md` gains "Qwen2 decode on the device and host KV reuse" with the pinned numbers; its
      earlier sentence on Qwen2.5-3B is corrected.
      *2026-10-11:* `docs/howto.md` and `docs/agent-arch.txt` done with items 2 and 3. `docs/performance.md` is owed after
      the pinned gate: it still says Qwen2.5-3B decodes outside the region, and its numbers come from the closing sweeps.
- [x] `CHANGELOG.md` entry added.
      *2026-10-11:* Session 126 (items 2 and 3, unpinned figures labelled as such); the pinned figures are added at close.

## Execution record

### 2026-10-11: steps 1 and 2 (plan check, item 1's decomposition)

**Step 1.** `scripts/performance-tests/check-plan-thresholds.sh`: ok (22 tier files). Plan claims re-verified against
the code: the region declines Qwen2 with the quoted reason (`GpuResidencyOptions`, `ResidentQkvPath`);
`rope_split_half` is a kernel in `rope.cu`; `bias_add` is a prefill-window region site; `prefill-breakdown.sh` defines
`projection_and_region_host` and `kv_host_write`. One claim did not hold, item 1 (a)'s method (corrected in place above).
All five models present.

**Pre-work build.** Jar `6ae135b0e21839cf` (HEAD `198467ad`, Tier 02C's candidate, no source newer), saved as
`dist/gpu-host-overhead/baseline-shaded.jar` for this tier's A/B.

**Published:** [`20261011T055415Z-gpu-host-overhead-decomposition`](../perf-compare/20261011T055415Z-gpu-host-overhead-decomposition/INDEX.md),
unpinned. Decode step, untraced, median of 127 steps, 128 generated tokens:

| Model, prompt | Step ms | Device-busy | In-forward host | Between passes |
|---|---|---|---|---|
| Phi-3.5-mini, 55 | 17.48 | 16.37 (93.7%) | 0.60 (3.4%) | 0.52 (3.0%) |
| Phi-3.5-mini, 503 | 18.43 | 17.42 (94.5%) | 0.51 (2.8%) | 0.50 (2.7%) |
| Qwen2.5-3B, 54 | 36.17 | 14.35 (39.7%) | 19.56 (54.1%) | 2.26 (6.2%) |
| Qwen2.5-3B, 502 | 35.59 | 14.75 (41.4%) | 18.29 (51.4%) | 2.55 (7.2%) |

Phi-3.5-mini prefill window (median of three): residue 11.0% at 512 tokens (84 ms of 767) and 6.2% at 2048 (226 ms of
3628); region host side 11.0% and 7.3%; host KV write 9.0% and 6.9%; matmul staging 5.5% and 4.5%.

**Findings.**

- **Phi-3.5-mini decode is bound by device kernels** (item 4's alternative outcome, measured): 93.7% of the step at 55
  tokens. The kernels are `q4k_gemv` 43%, `q5k_gemv` 26%, `q6k_gemv` 13%, the FP16 LM head GEMV 5.8% (cuBLAS over
  the dense FP16 output matrix, 0.93 ms) and attention 3.9%. The largest non-kernel term is the in-forward host time,
  0.60 ms: the device idles at each of the 32 per-layer downloads while the host takes it and launches the next layer.
- **Qwen2.5-3B decode spends more than half its step on the host**: 650 copies per token outside the region, and host
  SwiGLU (87 of about 130 execution samples). The region runs SwiGLU on the device, so item 2 removes both.
- **The prefill residue is the host KV growth before the first layer.** 96% to 98% of it is one interval in
  `runLayersBatch` before the first `device_layer` span. There the request's 64 host KV tensors grow from 64 positions
  by doubling copies to the window length (`DenseKvTensor.grow`, every allocation sample in the interval), and the
  allocation triggers 24 to 56 ms of GC pauses, all inside it. Phi-3.5-mini's KV row (3072 floats per layer) is 3x Mistral 7B's (1024).
- **Greedy decoding computes a full-vocabulary softmax** (`Sampler` always runs `SoftmaxStep`; greedy then takes the
  argmax and never reads the probabilities). It is the most-sampled frame between passes on both models: about
  2 ms of Qwen2.5-3B's 2.26 ms (151,936 entries), a smaller part of Phi-3.5-mini's 0.52 ms (32,064).

**Reachable ratios (item 1 (c)), projected onto the pinned reference column:**

| Row | Reference | Projection | Outcome |
|---|---|---|---|
| GPU tg >= 0.70x every sweep model | 0.416x (Qwen2.5-3B) | about 17.4 ms per token with item 2: 0.80x to 0.83x | reachable by item 2 alone |
| GPU tg >= 0.90x Phi-3.5-mini | 0.858x | 0.858x x 1.127 (Tier 02C pinned A/B) = 0.967x; unpinned same-session readings 0.977x and 0.955x | **already met before this tier's work** (decision 1) |
| GPU pp >= 0.70x at 512 | 0.594x (Phi-3.5-mini) | residue only 0.667x; residue plus half the other host terms 0.78x | reachable, not by the residue alone |
| GPU pp >= 0.60x at 2048 | 0.540x (Phi-3.5-mini) | residue only 0.576x; residue plus half the other host terms 0.64x | reachable, not by the residue alone |

**Order for step 3 (from the decomposition):** item 2, then item 3 (residue first, then region host side, host KV write,
staging), then item 4.

**Decision 1 (the owner's): the Phi-3.5-mini generation row is met before the tier's work.** The README's rule: a target
met before the work meant to reach it measures nothing.
- **(a)** Mark the row `retired: met on arrival` in the README's end-of-plan table, with this projection as its
  reference and the closing sweep confirming it pinned. Item 4 closes on "kernel-bound, 93.7%, `q4k_gemv` the largest
  kernel".
- **(b)** Raise it, for example to `>= 1.00x`. The bound with every non-device term removed is about 1.03x, so 1.00x
  needs nearly all of them gone or a kernel change (a packed LM head instead of the FP16 GEMV, about 0.5 ms). No item
  here owns that kernel change.
- **(c)** Keep it as written and score it at the closing sweep.

Recommendation: **(a)**. The step is kernel-bound, so a raised row would be scored on kernel work this tier does not
own.

**Decision 2 (the owner's): the prefill residue's cause is Tier 03 item 6's.** "Grow toward the known request length
instead of doubling" is Tier 03 item 6, and this tier's scope excludes host KV storage. Both prefill rows need the
residue removed.
- **(a)** Move the growth half of Tier 03 item 6 here: the window's KV tensors are sized once to the request length
  before the first layer, with no doubling copies. Real FP16 storage stays in Tier 03.
- **(b)** Leave it in Tier 03. The prefill rows then need 15.1% (512) and 10.0% (2048) from the region host side, host
  KV write and staging alone (25.5% and 18.7% of the window). That is possible on paper, but it means removing most of
  those terms.
- **(c)** Restate the prefill rows.

Recommendation: **(a)**. It is the single largest removable term on both rows.

**Raised, not a decision:** the greedy softmax is a sampler change outside items 2 to 4. It is worth about 2 ms per token
on Qwen2.5-3B once item 2 lands, about 12% of the projected step. Unless the owner says otherwise, it is recorded here
and left to Tier 05 (sampling).

**Out-of-tier changes:** none. No code changed.

**Decided 2026-10-11 (owner): decision 1 (a), decision 2 (a).** The README's Phi-3.5-mini GPU tg row is `retired: met on
arrival`, with reference 0.967x (projected) and the closing sweep reporting the pinned ratio. Item 4 closes on the measured
kernel-bound step. Tier 03 item 6's growth half is item 3's (scope widened above, and noted in Tier 03). The greedy softmax
stays with Tier 05.

### 2026-10-11: step 3, item 2 (Qwen2 in the decode region)

**The plan overstated the work.** Split-half rotation was already a parameter of the region (`ResidentRope`, a
`CudaRope` in either `RopePairing`, used by Qwen3). The LLaMA handler only built its region with the adjacent default
and declined Qwen2 before reaching it. What was missing was the bias add.

**Tests first.**
- `ResidentQkvPathQwen2Test` (new; GPU): a Qwen2-shaped layer (split-half pairs, Q/K/V biases, 128-wide heads, Q6_K V
  and down) held bit for bit against the op-at-a-time GPU path with the host bias add and rotation. It covers q, k
  and v at five positions up to 30,000, the whole layer at six, a 30-token leased decode through three layers, three
  threads in leases of their own, and a layer missing a bias being declined. Before the change it did not compile
  (no `ResidentQkvPath.Biases`).
- `LlamaTransformerHandlerGpuResidencyTest`: the "qwen2.5 is declined" case became two cases, seen failing first
  (region not active). They check that `on` activates the region and that greedy tokens equal `off` over the model's
  own 19-token continuation of a 5-token prompt, with logits within 1.0 (measured 0.507). They also check that a
  decoded token makes exactly 2 uploads and layers + 1 = 37 downloads.
- `GpuResidencyOptionsTest`: `qwen2`/`qwen2.5` are no longer named unsupported (seen failing first).
  `ResidentQkvPathTest.namesWhatItCannotRun` was rewritten to the remaining reasons (CUDA, query width) and passed on
  its first run, a regression test.
- A fixed run of TinyLlama's token ids put a near tie on Qwen2.5 at one position (top two logits 0.026 apart, under the
  0.2 to 0.4 CPU-versus-GPU norm difference), where on and off picked different tokens. The test uses the model's own
  greedy text instead, as the Qwen3 handler's test already does for the same reason.

**Change.**
- `ResidentQkvPath.Biases` (per-layer q, k, v; a layer without all three is not eligible), uploaded per layer and added
  by the prefill window's `add_bias` kernel after the three projections and before RoPE; `bias_add` sites tallied at
  decode width.
- Head norms and biases together are refused.
- The pairing and bias overloads of `unsupportedReason` are removed.
- `LlamaTransformerHandler.openResidentQkv` passes a `CudaRope` in the file's pairing and its biases.
- `GpuResidencyOptions` no longer names Qwen2.
- CLI help, `docs/howto.md`, `docs/agent-arch.txt` and `CHANGELOG.md` are updated.

**Results.**
- **Region tests:** 5 Qwen2 + 19 + 5 Qwen3 + 5 Phi-3 + 1 device-full, and the handler test's 4, all pass, as does
  `GpuResidencyOptionsTest`'s 10.
- **Smoke:** `smoke-gpu-residency.sh` (unmodified, TinyLlama with its cluster runs, and Qwen2.5-3B) passes. Qwen2.5 is
  active on every layer, greedy output is identical over 32 tokens, and device memory does not grow.
- **Re-measured**, unpinned,
  [`20261011T125030Z-qwen2-decode-region`](../perf-compare/20261011T125030Z-qwen2-decode-region/INDEX.md): Qwen2.5-3B
  decode 17.11 ms per token, was 36.17 (2.11x; 2.04x at a 502-token prompt); 2 uploads and 37 downloads per token,
  were 361 and 289. The decomposition projected 17.4 ms. The binding generation row is scored by the pinned closing
  sweep.
- **Greedy divergence:** the 128-token replies to the degenerate "x x x" prompts part from the earlier build's after
  about 25 and about 4 tokens. This is the norm-order difference documented for TinyLlama and Mistral 7B, now stated in
  `docs/howto.md`.

**Out-of-tier changes:** none.

**Unit suite after item 2:** `mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`,
BUILD SUCCESS on all 11 modules, no failure or error (the GPU-tagged classes above were run separately with `-Dgroups=gpu`).

### 2026-10-11: step 3, item 3 (Phi-3's prefill window host work)

**The decomposition's mechanism was corrected first.** `DenseKvTensor.ensureCapacity` grows a tensor once, to the next
power of two; it does not double repeatedly. So the residue is the allocation of the window's 64 full-size host KV
arrays (403 MB at 512 tokens, 1.6 GB at 2048, each 6.3 MB or more and so a G1 humongous object) plus the collections
that allocation triggers. Every GC in the window is "G1 Humongous Allocation". Exact sizing (decision 2 (a)) would not
remove it, which was put to the owner.

**Decided 2026-10-11 (owner), revising decision 2: reuse a finished request's dense host KV tensors for the next
request.** Exact growth stays in Tier 03 item 6, and Tier 03's note was corrected.

**Tests first.**
- `SessionKvReuseTest` (kvcache, new): retired tensors go to the next request with their capacity; a reused tensor
  reads zero wherever the new request has not written (F16 and Q8_0); the larger retired set is kept; paged layers
  are released, not reused. Written before the code; it did not compile until `retireKLayers` existed. The Q8_0 case
  then failed on the test's own exact 2.0 (Q8_0 stores 1.9998779) and was corrected to the stored value.
- `HostKvReuseLiveTest` (node, new; TinyLlama, Phi-3.5-mini, Qwen3-1.7B on the CPU backend): seen failing first on all
  three (new tensors). After the change, the second request takes the evicted request's tensors, its logits equal
  those on new tensors bit for bit, and its window allocates at least the KV's bytes less (byte assertion: Phi-3.5-mini
  59.9 to 9.6 MB, TinyLlama 8.6 to 5.8, Qwen3 20.9 to 6.2).
- `InFlightRequestsTest` (new): a regression test that passed on its first run, written after the class.
- `KvWriteRowsTest` (kvcache, new): `writeRows` from native memory equals token-at-a-time writes on dense F16, Q8_0 and
  paged, and the zero gap holds on a reused tensor. Written before the method but not run until it existed (it could
  not compile).
- `PrefillRegionHandlerParityTest`: now also holds every layer's host K and V bit-identical, region on against off, on
  four models. A regression guard for the direct write; it would also have passed before it.

**Change.**
- **kvcache:** `DenseKvRecycler` (new); `SessionKvLayout.retireKLayers`/`retireVLayers`; `newKLayers`/`newVLayers` take
  retired layers first. `DenseKvTensor` keeps a written watermark and zeroes a gap before it can be read or written
  past. `SessionKvTensor.writeRows` (default row by row; dense F16 one bulk copy).
- **node:** `InFlightRequests` (new). The LLaMA, Phi-3 and Qwen3 handlers count running passes per request (forward,
  forwardBatch, forwardMultiDecode, forwardVerify on LLaMA, lastRmsHiddenForEmbedding, shiftKv), and evict reuses only
  when none is running. A cluster retry broadcasts the evict to every node, including one whose timed-out pass may
  still run; that case drops the tensors as before. `ResidentActivation.materializeToKv` and a
  `PrefillWindowRegion.Window.runLayer` variant write a whole-layer window's K and V from the download buffer into the
  host tensors. The three handlers pass their layer's tensors and no longer write those rows again.
- **Docs:** `docs/howto.md` (the idle heap a server keeps; the direct K/V write), `docs/agent-arch.txt`, `CHANGELOG.md`.

**Results**, unpinned, span recording on,
[`20261011T162329Z-phi3-prefill-host-work`](../perf-compare/20261011T162329Z-phi3-prefill-host-work/INDEX.md):

| Phi-3.5-mini window | Before | Reuse | Reuse + direct write |
|---|---|---|---|
| 512 tokens | 767.4 ms (residue 11.0%, host KV write 69.3 ms) | 659.6 ms (residue 2.3%) | **602.2 ms** (host KV write 0.1 ms) |
| 2048 tokens | 3628.2 ms (residue 6.2%, host KV write 249.8 ms) | 3348.5 ms (residue 1.5%) | **3096.5 ms** (host KV write 0.1 ms) |

- **Projection onto the pinned reference:** 0.757x at 512 (row >= 0.70x) and 0.633x at 2048 (row >= 0.60x).
- **Same-session readings:** 0.690x and 0.623x, with span recording on.
- **Left unchanged:** the region host side (72.8 and 260.6 ms) and matmul staging (44.0 and 164.3 ms).

**GPU tests:** `PrefillRegionHandlerParityTest` (4), `PrefillWindowKernelsParityTest` (54),
`PrefillWindowRegionResidualTest` (5) and `ResidentActivationTest` (8) all pass, with the free-VRAM check passing this run.

**Regression runs after item 3.**
- **Unit suite:** `mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`,
  BUILD SUCCESS on all 11 modules, no failure.
- **`smoke-long-prompt-prefill.sh`** (unmodified, its default TinyLlama and Mistral 7B, static and continuous, 128/512/2048
  tokens, `--baseline-jar` the pre-tier jar `6ae135b0e21839cf`): all checks pass, and greedy text equals the pre-tier
  build on every model, schedule and length. Time to first token on Mistral 7B static was 1056 to 889 ms at 512 tokens
  and 5033 to 4877 ms at 2048. A first attempt passed Phi-3.5-mini to `--models`, which the script does not accept (it
  takes `tinyllama` or `mistral`); it failed on that argument and was re-run as above.

**Decision 3 (the owner's): close item 3 with two of its four terms left.** Item 3 names the region host side, the host
KV write, matmul staging and the residue. The residue and the host KV write are removed. The other two are unchanged,
and both prefill rows are projected met without them.
- **(a)** Close item 3 on the removed terms. If the pinned closing sweep misses a prefill row, the region host side is
  the next lever: the per-layer download wait, about 2.3 ms a layer at 512.
- **(b)** Work on the region host side and staging now.

Recommendation: **(a)**. Exit criterion 3 stays unticked until the owner decides.

**Decided 2026-10-11 (owner): (a).** Item 3 closes on the removed terms; exit criterion 3 ticked. If the pinned closing
sweep misses a prefill row, the region host side is the next lever.

**Out-of-tier changes:** none. The KV reuse is item 3's by owner decision.

### 2026-10-11: step 3, item 4 (Phi-3's decode step)

By owner decision 1 (a), item 4 closes on the measured step. Re-measured on this tier's build (jar `921ab742ba3f362a`),
same method as the decomposition:
[`20261011T183740Z-phi3-decode-kernel-bound`](../perf-compare/20261011T183740Z-phi3-decode-kernel-bound/INDEX.md).

| Prompt | Step | Device-busy | In-forward host | Between passes |
|---|---|---|---|---|
| 55 | 17.25 ms | 16.32 ms (94.6%) | 0.42 ms | 0.51 ms |
| 503 | 18.76 ms | 17.56 ms (93.6%) | 0.65 ms | 0.55 ms |

**The step is bound by device kernels.** `q4k_gemv` is 43% of kernel time, `q5k_gemv` 26%, `q6k_gemv` 13%, and the FP16 LM
head GEMV 5.8%. The non-kernel remainder is about 6% of the step, in two terms of about 0.5 ms each: the per-layer
download wait, and the full-vocabulary greedy softmax between passes, which is left to Tier 05. Removing all of it would
give at most about 1.06x. No code changed for item 4.

**Out-of-tier changes:** none.

### 2026-10-11: step 4 (cross-surface smoke matrix)

**Plan check:** `check-plan-thresholds.sh` ok (22 tier files). Plan claims re-verified: the region takes Qwen2
(`GpuResidencyOptions`, `ResidentQkvPath.Biases`), the three handlers retire and reuse host KV (`InFlightRequests`,
`SessionKvLayout.retireKLayers`), and the CLI help names Qwen2. All five models are present. No source file is newer than
the tree's shaded jar.

**Build under test.** Candidate jar `921ab742ba3f362a` (the jar item 4 measured), frozen as
`dist/gpu-host-overhead-close/candidate-shaded.jar` before anything ran. Every smoke ran it (`JUNO_JAR`). The Maven
install for the ITs rebuilt the tree's jar from the same source; the candidate's bytes were then copied back so the
owner's gate can check the tree jar against it. Baseline for the A/B runs: `198467ad` from `git archive`, built in a
scratch directory (jar `9af96224967a0057`; same source as the pre-work jar `6ae135b0e21839cf`, builds are not
byte-reproducible).

Every run was sequential, and only one GPU workload ran at a time. All smoke scripts ran unmodified.

| Check | Result | Notes |
|---|---|---|
| `smoke-gpu-residency.sh --models qwen2.5-3b,Phi-3.5-mini,Qwen3-1.7B` (cluster runs on Qwen2.5-3B, the first) | 13 PASS, 0 FAIL | region active on every layer of all three; greedy identical on against off over 32 tokens; no device-memory growth; pipeline and tensor cluster with the region on answer and equal local mode's output, no node JVM left |
| `smoke-context-policy.sh` (Phi-3.5-mini, GPU, both schedules) | all checks pass | failing requests at the limit next to KV reuse: a failed request keeps its KV, a later request is not affected |
| `smoke-packed-kquant-matmul.sh` (four sweep models, both schedules, 512 + 512 tokens) | all checks pass | |
| `smoke-consistency.sh` (CPU and GPU) | 54 PASS, 0 FAIL | `--parallel 2` batches and session requests equal the single-request response over `/v1/chat/completions` and `/v1/inference`; zero prefix-cache hits for stateless traffic. This is the check for a reused KV tensor exposing an earlier request's rows |
| `smoke-grammar.sh`, `smoke-tools.sh` | 19 and 14 PASS, 0 failures | |
| `smoke-cpu-hot-path.sh` (four models at 4 and 12 threads, vision) | all checks pass | the KV reuse runs on the CPU path too |
| `smoke-long-prompt-prefill.sh --baseline-jar <pre-tier>` | all checks pass (item 3 record, same jar) | greedy text equal to the pre-tier build, both schedules, 128/512/2048; not re-run |
| GPU-tagged node tests, `mvn test -pl node -Dgroups=gpu` | 367 run, 0 failures, 7 skipped | the free-VRAM check in `PrefillRegionHandlerParityTest` passed this run |
| Unit suite, eleven modules | 2,336 tests, 0 failures, 0 errors (52 skipped) | BUILD SUCCESS on all 11 modules |
| `ModelLiveRunnerIT`, five models (TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B, Qwen3-1.7B) | 5 of 5, **50 of 50 checks** (19 min) | check 9 on CUDA at about 560 and about 2,100 tokens (TinyLlama capped at 1,968): first token exact everywhere; first divergence over 16 tokens none, except Qwen2.5-3B at 2048, static, token 12 (reported, not asserted; Tier 02 read the same token on both schedules) |
| `mvn verify -pl juno-master` (stub cluster ITs) | 30 tests, 0 failures | |
| `mvn verify -pl juno-master -Pgpu` (TinyLlama) | 10 tests, 0 failures | `GpuForwardPassIT`, `GpuAttentionDivergenceIT`, `PrefillRegionGreedyIT` |
| Vision A/B, GPU, x3 | latency **0.997x**, decode **1.004x**; reply identical on all six runs | [`20261011T201902Z-gpu-host-overhead-vision-lora`](../perf-compare/20261011T201902Z-gpu-host-overhead-vision-lora/INDEX.md) |
| LoRA A/B, GPU and CPU, x3 | training time **1.000x** (GPU), **0.997x** (CPU); playback **0.983x** and **1.011x**; recall correct every run | same directory |

**Cross-surface checklist (closing):**

| # | Surface | Status | Evidence |
|---|---|---|---|
| 1 | CPU inference | PASS | the region and window are CUDA only; the host KV reuse runs on the CPU path (`HostKvReuseLiveTest`, CPU backend), `smoke-cpu-hot-path.sh`, `smoke-consistency.sh` on CPU; throughput is the owner gate's Part C |
| 2 | CUDA GPU inference | PASS | the smokes above, node GPU tests, GPU ITs, `ModelLiveRunnerIT` checks 9 and 10 on CUDA; throughput is the owner gate's Part A |
| 3 | ROCm GPU inference | NEEDS-AMD-HARDWARE | the region is CUDA only and `auto` resolves off on ROCm; the KV reuse is backend-neutral host code, unit-tested; no AMD device here |
| 4 | Static schedule | PASS | `smoke-consistency.sh` (`--parallel 2`, session re-prefill), `smoke-long-prompt-prefill.sh`, check 9 static, packed smoke static |
| 5 | Continuous schedule | PASS | paged KV is released, not reused (`SessionKvReuseTest`); check 9 continuous, packed and context-policy smokes on continuous |
| 6 | Single-node local mode | PASS | every smoke above |
| 7 | Pipeline-parallel cluster | PASS | residency smoke: Qwen2.5-3B with the region on, output equals local mode's; `ModelLiveRunnerIT` checks 1 to 6; `ThreeNodeClusterIT` |
| 8 | Tensor-parallel cluster | PASS | residency smoke tensor run, same; checks 7 and 8; `TensorParallelClusterIT` |
| 9 | LoRA training | PASS | declines the region as before; training time 1.000x (GPU) and 0.997x (CPU) |
| 10 | LoRA playback | PASS | declines the region; playback 0.983x and 1.011x, recall correct; grammar smoke's LoRA cell |
| 11 | Vision | PASS | moondream2 does not reach the changed paths; latency 0.997x, decode 1.004x, reply identical |
| 12 | OpenAI REST surface | PASS | every smoke over `/v1/chat/completions`, streaming included (long-prompt, packed), tools and grammar |
| 13 | Native REST surface | PASS | `smoke-consistency.sh` over `/v1/inference`, CPU and GPU |
| 14 | CLI | PASS | no new flag; `--gpu-residency` help and `docs/howto.md` name Qwen2 as covered |
| 15 | JVM embedding facade | N/A | no new embedder-invocable capability |

**Owner's closing gate, prepared:** `dist/gpu-host-overhead-close/` (git-ignored). It holds `baseline-shaded.jar`
(`6ae135b0e21839cf`, `198467ad`), `candidate-shaded.jar` (`921ab742ba3f362a`, this tree, byte-identical to the tree's
shaded jar), `run-gate.sh`, `wait-and-run.sh` and `READY`. About three hours, default `PART=ACB`:

- **Part A:** GPU no-regression, pinned A/B on the four sweep models at 512 tokens. Gates: tg and pp at least 0.95x,
  allocation per token at most 1.10x, GC pause within bounds. Greedy equality and the top 10 hot methods are reported.
- **Part C:** the standing CPU and allocation gate, pinned A/B on TinyLlama and Mistral 7B at 128 tokens, with the same
  thresholds. The top 10 hot methods are reported.
- **Part B:** the closing sweeps at 128, 512 and 2048, with Phi-3.5-mini at 2048 at an 8 GiB heap, pinned and published.
  They score the raised rows (tg at least 0.70x on every model, pp at least 0.70x at 512 and 0.60x at 2048) and report
  Phi-3.5-mini's tg against the retired 0.90x row and the 512-over-128 milestone.

Part B's scoring was dry-run against Tier 02's published closing sweeps and reproduces the README's reference cells
(0.416x, 0.594x, 0.540x, 512 over 128 0.997 on Mistral 7B).

After the run, the results go into this file and `docs/performance.md`. The README's GPU reference column moves to the
closing sweeps (its reference rule), and the tier closes if every gate is met. If a prefill row is missed, the next
lever is the region host side (decision 3 (a)).

**Out-of-tier changes:** none. No code changed in step 4.

### 2026-10-12: closing gate (owner run, pinned) and tier close

**Published:**
- [`20261012T002824Z-gpu-host-overhead-close-gate`](../perf-compare/20261012T002824Z-gpu-host-overhead-close-gate/INDEX.md):
  Parts A and C, the gate log and hot methods.
- The closing sweeps, published by the harness: [`20261012T013642Z`](../perf-compare/20261012T013642Z/INDEX.md) (128),
  [`20261012T015002Z`](../perf-compare/20261012T015002Z/INDEX.md) (512),
  [`20261012T020410Z`](../perf-compare/20261012T020410Z/INDEX.md) (2048) and
  [`20261012T021800Z`](../perf-compare/20261012T021800Z/INDEX.md) (2048, Phi-3.5-mini at 8 GiB).

Baseline jar `6ae135b0e21839cf` (`198467ad`), candidate `921ab742ba3f362a`. Clocks were pinned on every run, and the
script printed `GATES MET`.

| Part | Gate | Result |
|---|---|---|
| A, GPU no-regression (four sweep models, 512 tokens, A/B x3) | tg and pp >= 0.95x, allocation <= 1.10x, GC within bounds | **met**: tg 0.999x to **1.843x** (Qwen2.5-3B), pp 1.017x to **1.195x** (Phi-3.5-mini), allocation 0.903x to 0.992x, no GC pause in span on either side |
| C, standing CPU and allocation gate (TinyLlama, Mistral 7B, 128 tokens, A/B x3) | same | **met**: tg 0.999x and 1.005x, pp 0.983x and 1.030x, allocation 0.966x and 0.920x; Mistral 7B's tg median is over 2 baseline runs (one withheld by the span check), and the request-timed median of three agrees (1.007x) |
| B, closing sweeps (pinned, every row scorable, prompt parity exact) | tg >= 0.70x every model; pp >= 0.70x at 512 and >= 0.60x at 2048 every model | **met**: tg 0.776x to 1.066x; pp 0.726x to 0.833x at 512 and 0.602x to 0.730x at 2048 (Phi-3.5-mini binding at both); Phi-3.5-mini tg 0.973x (retired row, confirmed) |
| Vision and LoRA (step 4, unpinned) | latency <= 1.25x, decode >= 0.80x; training time <= 1.25x, playback >= 0.80x | **met** (step 4 record) |

**Notes.**
- **Greedy output, GPU A/B:** equal to the baseline in 3 of 3 runs on TinyLlama, Phi-3.5-mini and Mistral 7B. On
  Qwen2.5-3B the candidate's reply is identical across its three runs and agrees with the baseline's for 230 characters
  (about 48 tokens), then takes the other of two near-equal continuations: the region's RMS norm order. Reported, not
  gated.
- **The 2048 prefill row is met by 0.3%** (0.602x against 0.60x). It is met as scored, but the margin is inside the
  reference engine's own run-to-run movement. A later tier that moves Phi-3.5-mini's prefill should expect this row to
  read either side of 0.60x. The next lever is the region host side (decision 3 (a)), about 2.3 ms per layer at 512
  tokens.
- **Hot methods:** Qwen2.5-3B's GPU decode lost its host FFN frames (35 samples); its largest remaining host frame is
  the full-vocabulary greedy softmax (Tier 05). The CPU lane's frames are the same on both jars.
- **512 over 128:** 1.036 (Mistral 7B, binding) to 1.149, so the milestone row Tier 02 missed (0.998) is met and retired.

**Done in this close:**
- The README's GPU reference column moves to the four closing sweeps. The 2026-10-08 sweeps are marked superseded in
  their INDEX files and in `docs/perf-compare/README.md` (the README's reference rule).
- The three active GPU end-of-plan rows are retired as met, and the Phi-3.5-mini row's reference is updated to the
  pinned 0.973x. The 512-over-128 milestone is retired as met. `check-plan-thresholds.sh` passes.
- `docs/performance.md` gains the section "Qwen2 decode on the device and host KV reuse". `CHANGELOG.md` carries the
  pinned figures. Every number in both was read off the published directories above.

**Decision 4 (the owner's): the GPU end-of-plan rows are all met.** The only active end-of-plan row is CPU tg (>= 0.25x,
0.215x today). When Tier 02 met its GPU rows, the owner raised them (Tier 02 decision 15) and created this tier.
- **(a)** Leave the GPU rows retired. What is left on the GPU is device-kernel time, which no tier owns: Phi-3.5-mini
  decode is 94% kernels, `q4k_gemv` 43% of it. Tier 14 reports the distance to parity.
- **(b)** Raise them again (for example tg >= 0.85x every model, pp >= 0.80x at 512 and >= 0.70x at 2048), and name a
  tier that owns the kernel work (a faster K-quant GEMV, the packed LM head) before Tier 03 starts, as Tier 02D was
  created.

Recommendation: **(a)**. A raised row needs an owner, and the lever is new kernel work, not host overhead. CPU, at
0.215x, is still the largest gap. Tier 03's own perf gate and the standing gates hold the GPU numbers where they are.

*Corrected 2026-10-11 (plan review): the last sentence was not true when written.* The standing gate's throughput half
was CPU-only, and Tier 03's thresholds (`gather_ms`, `paged_vs_dense%`, defrag pause) sit on the paged `continuous`
path while every sweep reads `static` dense KV, so nothing held GPU t/s in Tier 03 or Tier 04. The README's standing
gate (renamed the "Standing throughput and allocation gate") now has a GPU lane: Juno GPU tg and pp **>= 0.95x** the
pre-tier build on all four sweep models at `n_prompt=512`, same-hour A/B, pinned. With it, option (a)'s premise holds.
The decision itself stays the owner's. The CPU gap this recommendation names is now owned by
[Tier 02E](TIER-02E-cpu-decode-kernels.md), inserted between this tier and Tier 03, so **the next tier is Tier 02E**,
not Tier 03 as the closing line below says.

**Carried forward, unowned:** the region host side in the prefill window (decision 3 (a)), the full-vocabulary greedy
softmax (Tier 05), and the Phi-3 decode kernels (`q4k_gemv`, the FP16 LM head GEMV).

**Out-of-tier changes:** none.

**Next tier:** Tier 03 (KV cache maturity), the next row of the README's running order.
