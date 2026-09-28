# Tier 01B: Prefill throughput

Status: not started
Gap analysis refs: none directly — this tier exists because the gap analysis has no prefill section
at all, while the published measurements under `docs/perf-compare/` show prompt processing to be the
single largest gap Juno has. See "Why this tier, why now".

## Objective

Raise batched-prefill (prompt-processing) throughput on the GPU path, which is the largest measured
gap against llama.cpp anywhere in this repository and which no other tier in this plan owns. Find
where prefill time actually goes using the JFR spans that already exist (`juno.PrefillBatch`,
`juno.ForwardPass`, `juno.MatVec`, `juno.Attention`), attack the dominant term, and move the
Juno/llama.cpp pp ratio by a stated multiple rather than incidentally.

## Why this tier, why now

The plan as originally written had no tier whose objective was prefill throughput. Tier 02 touches
attention but targets peak memory at long context; Tier 04 adds fused kernels for formats that are
not Q4_K; Tier 07 targets multi-session throughput. Meanwhile the numbers say this:

| Model (GPU, Q4_K_M) | llama.cpp pp t/s | Juno pp t/s | Juno/llama pp | Juno/llama tg | Juno tokens actually prefilled |
|---|---|---|---|---|---|
| tinyllama-1.1b | 3654.97 | 59.25 | 0.016 | 0.239 | **30** |
| qwen2.5-3b | 1482.11 | 28.82 | 0.019 | 0.284 | **20** |
| Phi-3.5-mini | 1199.81 | 28.79 | 0.024 | 0.330 | **21** |
| mistral-7b | 668.57 | 20.89 | 0.031 | 0.513 | **20** |

(`docs/perf-compare/20260918T204809Z/INDEX.md`, `n_gen=64`, GTX 1080.)

**Read that last column before drawing a conclusion from the pp ratios.** That run has
`raw_prompt: 0`, so llama-bench prefilled the 128 tokens it was asked for while Juno prefilled the
20-to-30-token sentence `compare-llama-cpp.sh` hard-codes. A 20-token prefill on a cold JVM is the
single worst shape Juno has: the smallest possible batch, executed once, inside C2 warmup. The
0.016-to-0.031x figures are a real gap plus a batch-width mismatch plus a cold-start penalty, and
the three are not separable from this run. They are not a valid before-measurement for this tier and
must not be quoted as one.

The parity-corrected numbers are fewer, and they are worse rather than better:

| Measurement | Juno pp t/s | llama.cpp pp t/s | Ratio |
|---|---|---|---|
| mistral-7b, real 520-token prompt (`20260915T043143Z`, `raw_prompt: 1`) | 11.79 | 709.73 | **0.0166x** |
| tinyllama, real 530-token prompt (same run) | 31.83 | 4225.02 | **0.0075x** |
| tinyllama, real 512-token prompt, `--gpu-attention on`, `--prefill-batch 32` (`20260916T040113Z-prefill/`) | **119.56** | 4225.02 | **0.028x** |

That third row is the best GPU prefill figure anywhere in this repository. Everything this tier does
is measured against it, not against 0.0166x.

Note also that the two runs previously cited as proof that "pp gets worse with prompt length"
differ in **two** variables besides prompt length: the 128 run has `raw_prompt: 0` (20 Juno tokens)
while the 512 run has `raw_prompt: 1` (520 Juno tokens), and the 128 run used `--vector 1` while the
512 run used `--vector 0`. The claim may well be true — a fixed per-chunk cost paid sixteen times
for a 512-token prompt predicts exactly that shape — but it is **not established by those two
runs**, and step 1 below is what actually establishes it.

Decode is 2 to 4 times slower than llama.cpp. Prefill, measured like-for-like, is 35 to 130 times
slower depending on configuration. A serving engine whose time-to-first-token scales that badly with
prompt length has a product problem, not only a benchmark problem, and every tier after this one
that measures a llama.cpp-relative pp ratio will keep reporting a number nobody is working on.

**One prefill lever exists on part of the model set, and what it is worth elsewhere is unmeasured.**
The GPU-resident attention kernel behind `--gpu-attention` was measured taking pp from **31.04 to
119.56 t/s (3.85x)**, with attention's share of prefill wall time dropping from **78.7% to 11.0%**,
at `--prefill-batch 32` on a real 512-token prompt — on **`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`**
(`docs/perf-compare/20260916T035952Z-prefill/INDEX.md` off versus
`20260916T040113Z-prefill/INDEX.md` on; `docs/performance.md` labels the same table "TinyLlama
Q4_K_M"). An earlier draft of this tier attributed that result to Phi-3.5. It did not come from
Phi-3.5, and three things follow that change what item 0 is worth:

- **TinyLlama is Llama-family, so it already runs with the kernel on by default.** The 3.85x is the
  performance of the *current default*, not a lever waiting to be switched on. There is no
  measurement, anywhere in this repository, of what Phi-2, Phi-3, Qwen3 or Qwen3-MoE would gain.
- **The derived ratio was wrong by roughly 3x.** "Roughly 120 t/s against llama.cpp's ~1200 t/s is
  about 0.10x" used Phi-3.5's llama.cpp figure against TinyLlama's Juno figure. Against the correct
  denominator (4225 t/s on the same run shape) 119.56 t/s is **0.028x**.
- **Item 0 is therefore a bet, not a banked win**, and this tier's threshold cannot lean on it.

The flag's default is `auto`, not `off`, and `auto` resolves to on whenever CUDA is present —
`GpuAttentionOptions.preferGpuAttention()` is literally `CudaAvailability.isAvailable()` for `AUTO`.
**There is no per-architecture resolver.** The class has no architecture awareness at all; the
reason the kernel is inactive for Phi-2, Phi-3, Qwen3 and Qwen3-MoE is simpler and more expensive to
fix than a resolver would be: those handlers never read `GpuAttentionOptions`. Only
`LlamaTransformerHandler` and `LoraTrainableHandler` reference it, and each of the four uncovered
handlers owns its own KV map and attention math. ROCm is inactive for a third reason again — `AUTO`
keys off CUDA availability specifically.

So item 0 is **four kernel integrations plus a capability-reporting mechanism that does not exist
yet**, not "existing kernel code reaching more architectures." Size it accordingly. It still goes
first, because every other measurement in this tier would otherwise be taken against a baseline that
is fast on four architectures and slow on four others — but it is no longer the cheap win this tier
was originally written around.

This tier is sequenced immediately after Tier 01 **and depends on it.** An earlier draft claimed
independence from Tier 01's go/no-go outcome on the grounds that "prefill is dominated by large-batch
GEMM and host-device transfer, not by the per-op dispatch overhead Tier 01 targets." That reasoning
defeats itself: host-device transfer *is* what Tier 01's activation-residency primitive removes, and
scope item 2 below — now the largest item in this tier — is built directly on it. If Tier 01 comes
back negative and downgrades to partial-complete, item 2 cannot proceed as written and this tier
escalates rather than silently substituting a smaller scope. Items 0, 1, 3 and 4 remain executable
on today's op-at-a-time GPU path either way.

## Scope

### In scope

0. **Make `--gpu-attention` genuinely default.** Close the architecture gap so the GPU-resident
   attention kernel is on by default everywhere it is correct, rather than resolving to the scalar
   CPU path for half the supported model set:
   - Add GPU-attention support to `Phi2TransformerHandler`, `Phi3TransformerHandler`,
     `Qwen3TransformerHandler` and `Qwen3MoeTransformerHandler`, which today keep the scalar CPU
     path unconditionally.
   - Decide and document the ROCm answer. If the kernel can be ported, port it (subject to this
     plan's `NEEDS-AMD-HARDWARE` rule, since there is no AMD device here). If it cannot land this
     tier, `auto` must say so — a startup notice naming the backend and the resulting path, not a
     silent resolution to scalar.
   - Once coverage is complete, change the default from `auto` to `on` and keep `auto` as an
     explicit opt-in for anyone who wants per-architecture resolution. Where a path genuinely cannot
     support the kernel, it fails loudly to the documented fallback rather than resolving quietly.
   - Keep the two existing, deliberate exemptions and re-verify them rather than assuming: LoRA
     **training** and `--lora-play` ignore `--gpu-attention` (attention stays scalar CPU, the train
     REPL warns), and that stays true unless this tier explicitly changes it.
   - Carry the documented consequence honestly: the FP16 KV mirror can produce occasional
     multi-token greedy-decode divergence from the scalar path. That is acceptable as a default only
     if it is stated plainly in `docs/howto.md` and `--help`, and `off` remains available as the
     bit-identical CPU-parity baseline. Re-verify the divergence characteristics on each newly
     covered architecture before turning its default on — do not inherit the Llama-family result.
   - **Fix the benchmark's stale description in the same change.** `compare-llama-cpp.sh` documents
     its default lane as `--gpu-attention off` (its `--help` text and the comment above the
     default-flags lane), but it only passes the flag when the variable is non-empty, so the default
     lane has actually been running whatever Juno's own default resolves to. Every published default
     lane number is therefore labelled wrong. Correct the labelling, and state in the first
     re-baselined run which lanes were affected — this sits alongside the benchmark-parity
     preconditions in [`README.md`](README.md) and is the same class of defect.
1. **Measure before changing anything — after building the instrumentation that makes the
   measurement possible.** An earlier draft of this item said the breakdown could be read "from the
   JFR spans that already exist." It cannot, and the difference is not cosmetic, so this item has two
   halves and the first one is net-new code.

   **1a. Add the two spans the breakdown needs.** Four spans do exist and are useful —
   `juno.PrefillBatch` (window size, start position), `juno.ForwardPass` (prefill total),
   `juno.MatVec` (call count and time), `juno.Attention` (prefill share) — but there is **no** event
   anywhere in `src/main` for host-to-device staging, for device-to-host readback, or for weight
   dequantization. The twenty `@Name("juno.*")` declarations in the tree are `Attention`,
   `ContinuousStep`, `ForwardPass`, `GrammarConstrained`, five `Lora*`, `MatVec`, `PrefillBatch`,
   `ResidualAdd`, `RmsNorm`, `Rope`, `Speculation`, `SwiGlu`, `TemplateFormat`, `Tokenizer` and
   `TokenProduced`; none of them is a staging or dequant span. `juno.MatVec` wraps staging,
   dequantization and compute in a single span, so the very terms 1b has to attribute are *inside*
   it, not beside it. Asking for a breakdown with "no unattributed residue" off those spans is asking
   for something nobody can deliver. Add:
   - **`juno.DeviceStaging`** — direction (`H2D`/`D2H`), bytes and duration, around every
     `gpuMemcpy` on the activation and weight paths. The staged-bytes assertion under "Tests to
     write/upgrade" reads from this same event, so it is instrumentation both halves of this tier
     need rather than extra work for the breakdown alone.
   - **`juno.WeightDequant`** — format, rows, cols and duration, around `Q4KMmqKernel.launchDequant`
     and `LlamaTransformerHandler.dequantize`.

   Register both in `scripts/performance-tests/juno-perf.jfc` and in `JfrMetricsExtractor`, with
   tests in the `metrics` module, following the pattern Tier 01 established for `JdkEventBucket` —
   including its rule that every key is written on every run, zero or not, so a consumer never has to
   distinguish "absent" from "none". [Tier 04C](TIER-04C-packed-weight-matmul.md) item 1 reads its
   `launchDequant`-versus-`gemmHalf`-versus-staging split off these same two spans; it was written
   believing Tier 01 had widened the extractor far enough for that, and Tier 01 widened only the
   `jdk.*` bucket. Building them here covers both tiers, and `metrics` must be in this tier's own
   `mvn test -pl` line (the documented command omits it — see [`README.md`](README.md)'s test
   infrastructure section).

   **1b. Produce the breakdown.** A per-term breakdown of prefill wall time for all four sweep models
   at `n_prompt` of 128 and 512, on GPU, from the four existing spans plus the two added in 1a.
   Attribute what remains — layout packing, per-chunk fixed cost — explicitly rather than leaving it
   as an unlabelled residue. This breakdown is the tier's primary artifact and decides what items 2-4
   are actually worth doing; publish it under `docs/perf-compare/` before writing any kernel code.
2. **Stop staging the activation batch to host between every matmul.** An earlier draft of this item
   proposed extending batched `sgemm` to "every other quantized residency type." That set is empty:
   `DeviceFloatMatrix`, `DeviceHalfMatrix` and `DeviceQ4KMatrix` are the *only* device matrix types
   in the module, `CudaMatVec` already overrides `sgemm` for all three, and it already dispatches to
   a real tiled `cublasGemmEx` path above `HALF_SGEMM_BATCH_MAX = 8` — which a prefill window always
   exceeds. Every other quant format is dequantized on the host at load and uploaded as FP16
   (`uploadFp16Layer`). There is no dtype falling through to a serial GEMV loop on CUDA.

   The real cost is the data movement around those GEMMs. `MatVec.sgemm` takes `float[][] X` and
   returns a **new** `float[][] Y`, so every projection stages the entire activation window
   host-to-device and the entire result device-to-host. At a 512-token window on a 4096-dim model
   that is roughly 8 MB moved each way per matmul, seven matmuls per layer, every layer, every
   chunk. `LlamaTransformerHandler.sgemmLayerInto` then allocates the returned `float[][]` and
   `System.arraycopy`s it into the output buffer it was handed, adding a full-batch allocation and
   copy per matmul on top.

   Extend Tier 01's residency primitive to the prefill window: upload the activation batch once per
   layer (once per forward pass where the boundary permits), keep the GEMM outputs on-device across
   the projections, and materialize to host only where something that is not GPU-resident needs the
   data — the attention boundary until Tier 02 lands its kernel, the gRPC boundary in cluster mode,
   or the LM head. Give `MatVec` a non-allocating batched form so `sgemmLayerInto` can stop
   allocating and copying; coordinate the contract with Tier 10 item 4, which adds the same
   output-parameter shape for the CPU path, so the two do not land two different spellings of it.

   **This item is the one that depends on Tier 01.** If Tier 01's residency primitive did not ship,
   escalate rather than proceeding — see "Why this tier, why now."

   *2026-09-27, hand-off from Tier 01.* Tier 01 closed with the primitive-threshold reading "decode
   width missed (0.26x the CPU chain), prefill width at parity (1.002x / 0.995x), chaining met (0.55 at
   prefill)". Per Tier 01's contingency this item **proceeds on the prefill result**. The evidence is
   `docs/perf-compare/20260927T115107Z-tier01-resident-chain/`; the device-only lane (0.157 ms against
   2.13 ms on the CPU at batch 512, 13.6x) is the case this item rests on. `ResidentQkvPath` is the
   working decode-width template: chain-owned scratch (`ResidentChain.allocateScratch`), device regions
   pooled by concurrent callers rather than per thread, one wait per region.
3. **Chunk sizing and staging cost.** `--prefill-batch` defaults to adaptive whole-prompt sizing only
   on GPU + `static` + local single-shard, and to a fixed `32` everywhere else — including CPU, the
   `continuous` schedule, cluster, and `juno lora` (`docs/howto.md:63`). A fixed 32-token chunk pays
   the per-chunk fixed cost 16 times for a 512-token prompt. Extend adaptive sizing to the surfaces
   that are still pinned at 32 where the VRAM query that drives it is meaningful, and make the
   per-chunk fixed cost itself smaller (the pinned host-staging work already published at
   `docs/perf-compare/20260918T153900Z-prefill-adaptive/` measured -30% prefill and -27% request wall
   on a real 488-token mistral-7b prompt — continue that line rather than restarting it).
4. **Residual prefill attention.** On TinyLlama, turning the kernel on dropped attention from 78.7%
   to 11.0% of prefill wall time, so on Llama-family models item 0 has already taken most of this —
   but that is the one architecture where the kernel was already the default, so nothing here is
   banked for the four this tier newly covers. Re-read item 1's post-item-0 breakdown and confirm,
   **per architecture**, whether attention is still a material share of prefill at 512 tokens. If it
   is, state plainly that the remainder belongs to Tier 02's tiled kernel and record the number so
   Tier 02 can be held to it; if item 0 already reduced it to noise, record that too, so Tier 02 is
   not later credited with a prefill win that this tier already banked. Do not extrapolate the
   78.7%-to-11.0% figure onto any architecture that was not measured — it is one model's result.
5. **Reap the engine-keepalive subshell in the eight sibling scripts.** Keeping the console REPL alive
   requires stdin never to reach end of file, and eight scripts do that with
   `< <(while true; do sleep 3600; done)` — a process substitution whose subshell nothing reaps, which
   respawns a fresh `sleep` every hour and therefore persists indefinitely:
   `compare-vision.sh`, `compare-prefill-batch.sh`, `compare-schedule.sh`, `compare-parallel.sh`,
   `compare-mixed-prefill.sh`, `smoke-tools.sh`, `smoke-grammar.sh` and `smoke-tier00-consistency.sh`.
   [Tier 01](TIER-01-gpu-activation-residency.md) fixed the same line in `compare-llama-cpp.sh` — it
   now holds a named pipe open on a descriptor it owns and releases both with the engine — and
   deliberately left these eight, because none of them sources `perf-lib.sh` today and several cannot be
   exercised without a GPU, real models and a long run. That was the right call for a tier that could not
   run them. **It is the wrong call for this tier, which runs three of them as required gates**:
   `compare-vision.sh` (mandatory here, not optional), `compare-prefill-batch.sh` and, through the
   cross-surface matrix, `compare-schedule.sh`.

   Move Tier 01's named-pipe helper into `perf-lib.sh` — which is a safe host for it, since it defines
   functions only and its one assignment is guarded — and have all nine scripts source it, rather than
   making eight separate copies of the same edit. This is not hygiene for its own sake: Tier 01's record
   documents that the accumulated orphans (some two days old, from Tier 00's smoke script, still
   spawning) made `pgrep` checks for "is a sweep still running" return false positives, which cost real
   time while stopping a sweep. This tier runs more sweeps than any tier before it, and every later tier
   inherits both the scripts and the problem.

   **Exit condition:** a full run of each of the nine scripts leaves zero leftover shells and zero
   leftover pipes, checked with `pgrep` and `lsof` immediately after the script returns, on the same host.
   For the scripts this tier cannot fully exercise (no AMD hardware, or a model this host will not fit),
   run them far enough to launch and stop an engine at least once — which is all that is needed to
   demonstrate the leak is gone, since the leak is per engine launch rather than per run.

### Out of scope

- CPU prefill throughput. The CPU pp ratio is 0.050 to 0.099 (`docs/perf-compare/20260918T031702Z/`),
  a serious gap, but its causes are the allocation, threading and kernel issues Tier 10 owns after
  its scope was widened. Re-measure CPU pp at the end of Tier 10, not here.
- ROCm prefill. `RocmMatVec` has no `sgemm` override at all, so ROCm prefill is a serial GEMV loop
  today; closing that is Tier 10 item 1 (ROCm tiled-GEMM), which is where the hardware-gated work is
  already concentrated. This tier must not leave the ROCm path worse than it found it.
- New quantization formats (Tier 04) and new architectures (Tier 08).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | must not regress; CPU prefill throughput itself is Tier 10's, but the correctness oracle and the fixed-`32` chunking default both live here |
| 2 | CUDA GPU inference | primary target |
| 3 | ROCm GPU inference | N/A for the new batched paths (CUDA-only this tier); must verify ROCm's existing serial-GEMV prefill still works and is not regressed by any shared dispatch change. **Not N/A for item 0**: ROCm resolves `--gpu-attention` to scalar today, and this tier must either port the kernel (`NEEDS-AMD-HARDWARE`) or make the fallback explicit and announced rather than silent |
| 4 | Static schedule | primary target — this is where adaptive whole-prompt prefill already runs |
| 5 | Continuous schedule | chunked prefill mixed with decode is the continuous engine's core loop; any per-chunk cost reduction must be verified there too, and the fixed-`32` default re-examined |
| 6 | Single-node local mode | primary dev/test surface |
| 7 | Pipeline-parallel cluster | prefill chunks cross the gRPC boundary per shard — confirm the per-chunk fixed cost being reduced here is not simply relocated into serialization |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | training prefills its own microbatches; confirm unaffected, or improved, but not silently changed in numerics. Training's existing `--gpu-attention` exemption (attention stays scalar CPU, REPL warns) is **re-verified and kept** by item 0, not quietly swept into the new default |
| 10 | LoRA playback | playback prefills a LoRA-modified prefix; confirm the delta-add still composes correctly against a wider batched path. Same `--gpu-attention` exemption as row 9 — re-verified, kept, and still warned about |
| 11 | Vision | `VisionEncoder` runs the widest batches in the system (B around 741) and reuses the same `MatVec` primitives — a batched-dispatch change is exactly the kind that regressed vision once before; run `compare-vision.sh` as a required gate, not an optional one. Vision already inherits `--gpu-attention` by delegating to `LlamaTransformerHandler`, so item 0 changes nothing for it directly — verify that stays true rather than assuming it |
| 12 | OpenAI REST surface | time-to-first-token is the user-visible form of this tier's metric; measure TTFT, not only aggregate pp |
| 13 | Native REST surface | same |
| 14 | CLI | `--prefill-batch` semantics change on any surface where the default moves off `32`; this is user-visible and must be documented |

## Implementation steps

0. **Ship the two script fixes before the first sweep, because both of them affect running one.**
   `scripts/performance-tests/check-plan-thresholds.sh` (execution rule 7's enforcement, hoisted out of
   Tier 14 — see the test list below) and scope item 5's engine-keepalive reaping. The second one comes
   first in practice: this tier runs more engine launches than any tier before it, and until the leak is
   fixed a `pgrep` check for "is a sweep still running" returns false positives, which is a problem
   whose cost is paid during step 1 rather than after it. Neither needs a model, a GPU or a build.

   **A third item, handed over by Tier 01 on 2026-09-27: find and fix the one fast prefill
   repetition before step 1's re-baseline.** In `compare-llama-cpp.sh` sweeps, one of three prefill
   repetitions sometimes reads four to five times faster than the other two: the tinyllama default lane
   read 898 and then 819 t/s in two separate runs against about 168 for the others, and the qwen2.5-3b
   tuned lane 130.6 against about 80. On tinyllama it was **repetition 1 both times**, the measured
   request that directly follows the warm-up. Medians are unaffected (median of three), but the row
   breaks the 15% spread rule and forces re-runs, and step 1's re-baseline is this tier's
   before-measurement, so it must not need re-runs for a harness artifact. Suspects, not verified: (a)
   prefix-cache or KV reuse between the warm-up and the measured request, so the measured request does
   not actually prefill every prompt token; (b) a timing boundary in how the prefill lane reads
   `prompt_eval_tps`. Evidence: `docs/perf-compare/20260927T091155Z/` and `20260927T093054Z/` (the INDEX
   banners, and the per-repetition `*-prefill-rep*-juno.json`: `prompt_eval_tps`). If the cause is (a),
   the fix belongs to the harness (a fresh session or a distinct prompt per measured request), not to
   the engine's reuse; if the engine turns out to reuse a prefix the request did not ask to share,
   that is a correctness finding and is raised with the owner before the re-baseline.
1. Re-baseline first. Run `compare-llama-cpp.sh --gpu` at `n_prompt` 128 and 512 on all four sweep
   models on current HEAD (post-Tier-01), under the parity-corrected harness required by this plan's
   "Benchmark parity preconditions" (README), and with each lane's actual resolved
   `--gpu-attention` value recorded rather than assumed — the historical numbers quoted above were
   taken before those corrections, with at least one lane mislabelled, and are not a valid
   before-measurement for this tier's gate.
2. **Land item 0 next, before anything else that changes the forward pass.** Not because it is cheap —
   it is four kernel integrations against four handlers that each own their own KV map and attention
   math, plus a capability-reporting mechanism that does not exist yet — but because leaving it until
   later would mean every subsequent measurement in this tier is taken against a baseline that is fast
   on four architectures and slow on four others. Re-measure immediately after, so the default change
   has its own attributable number, and record that number per architecture: the Llama-family 3.85x
   says nothing about what these four will do. Three of the four have a real file to measure on — see
   "Models needed", and check `models/` rather than trusting any table.
3. **Build the `juno.DeviceStaging` and `juno.WeightDequant` spans (scope item 1a) before
   attempting the breakdown.** This is net-new instrumentation with `metrics` tests, not a
   measurement step, and the breakdown's "no unattributed residue" criterion is unreachable without
   it. Land it on the post-item-0 build so the spans are present for every measurement from here on.
4. Produce the per-term prefill breakdown (scope item 1b) and publish it.
   Decide which of scope items 2, 3 and 4 the breakdown actually justifies, and record the decision here
   — item 0 may well have moved which term dominates, which is the point of sequencing it here. The
   expected ranking going in is that host-device staging (item 2) dominates once attention is on the
   GPU, since a 512-token window moves roughly 8 MB each way per matmul; if the breakdown says
   otherwise, follow the breakdown.
5. Write down the expected contribution of each remaining item against the threshold, per the
   "Decompose the ask before implementing it" clause below, and escalate here if they do not sum.
6. Implement in the order the breakdown ranks, largest term first.
7. Re-measure after each change rather than only at the end, so a negative result is attributable to
   one change instead of the batch.
8. Run the full cross-surface smoke matrix, including the vision gate.

## Tests to write/upgrade before implementation

- **New GPU-attention parity tests, one per newly covered architecture** (Phi-2, Phi-3, Qwen3,
  Qwen3-MoE): kernel output against that handler's scalar CPU attention within stated tolerance, and
  a greedy-decode divergence characterisation on a real model — how often, and by how many tokens,
  the FP16 KV mirror diverges. This is the evidence that decides whether that architecture's default
  flips on; do not inherit the Llama-family answer for it.
- **New GPU-attention capability tests.** Note what these can and cannot be: `GpuAttentionOptions`
  has no architecture awareness today (`AUTO` is just `CudaAvailability.isAvailable()`), so there is
  no resolver to test until item 0 builds one. The tests are therefore (a) each covered handler
  reports its GPU-attention capability truthfully through the new mechanism, (b) an uncovered
  handler or backend produces the documented explicit notice rather than a silent scalar fallback,
  (c) `off` still gives the bit-identical CPU-parity baseline, and (d) the LoRA training and
  `--lora-play` exemptions still hold and still warn.
- **New residency/staging tests for item 2**: a prefill window produces output equal within float
  tolerance to today's stage-per-matmul path, at window sizes spanning the `HALF_SGEMM_BATCH_MAX = 8`
  dispatch threshold (1, 2, 8, 9, 32, 512) and at the vision batch width (B around 741); device
  memory is not leaked across repeated windows (reuse the accounting assertion Tier 01 adds); and
  the non-allocating batched form produces bit-identical results to the allocating one for every
  backend implementation.
- **New `metrics` tests for the two added spans** (item 1a), mirroring
  `JfrMetricsExtractorJdkEventsTest`: `juno.DeviceStaging` aggregates bytes and duration per
  direction, `juno.WeightDequant` aggregates per format, and both emit their keys on every run
  including when the count is zero. These must fail on the pre-item-1a build for the right reason
  (the keys do not exist), the same evidence standard Tier 01 applied to `JdkEventBucket`.
- **A staged-bytes assertion**: read H2D/D2H bytes per prefill window off `juno.DeviceStaging` and
  assert the post-item-2 figure against the pre-item-2 baseline, so item 2's win is measured in bytes
  moved and not only in wall time.
- **New `PrefillBatchOptions` tests** for any surface whose chunk-sizing default changes: assert the
  new default per surface, and assert that an explicit `--prefill-batch N` still overrides on every
  surface.
- **`ModelLiveRunnerIT`**: add a long-prompt (512-token) prefill check asserting correct output, for
  both schedules.
- **New bash smoke script**: `scripts/performance-tests/smoke-tier01b-prefill.sh` — drives
  `/v1/chat/completions` with 128-, 512- and 2048-token prompts against `tinyllama` and `mistral-7b`,
  on both schedules, asserting correct output and recording TTFT; and asserts greedy-decode output is
  identical to the pre-tier build for the same prompt and seed.
- **`scripts/performance-tests/check-plan-thresholds.sh` (new, and this tier ships it).** Execution
  rule 7 in [`README.md`](README.md) says every tier file with a `Perf gate` must carry a `**Threshold`
  block with a numeral and a comparison operator, and says the rule is machine-checked. The check was
  originally specified inside Tier 14's doc-consistency script, which left a rule governing seventeen
  tiers enforced by nothing until the last of them. This tier is the next to execute, so it ships the
  script: grep every `TIER-*.md` in this tree, fail on any file containing `Perf gate` without a
  matching `**Threshold` block, exclude `TIER-14-*.md` (it names the string to describe the check), and
  exit non-zero with the offending filenames listed. Every later tier runs it as the first item of its
  own test list, before its own tests, and Tier 14 calls it rather than restating it. It needs no model,
  no GPU and no build — it is `grep` and an exit code, and it should run in under a second.
- **Every prefill repetition is a full prefill (implementation step 0, third item).** A harness selftest
  or a live check, run before step 1's re-baseline: for every measured prefill repetition, the prompt
  tokens the engine actually prefilled equal `prompt_tokens` (no reused prefix reported for the
  measured request), and the repetition's `prompt_eval_tps` comes from that request's own prefill span.
  Show it catching the artifact on the evidence runs' shape (a warm-up followed by repetition 1 on
  tinyllama) before the fix, and passing after.
- **Perf gate (required)**: this is by definition a forward-pass change. `compare-lora.sh`,
  `compare-vision.sh` (mandatory here, not optional — vision runs the widest batches in the system),
  `compare-prefill-batch.sh`, and `compare-llama-cpp.sh` at both `n_prompt` 128 and 512; publish under
  `docs/perf-compare/<timestamp>-tier01b-prefill/`.

  Every number below is a median of at least three runs with min/max published, per the README's
  noise-floor rule — this host resolves to about ±15%, and two of these thresholds sit inside that.

  **Threshold, item 0 on its own.** Every architecture newly covered by the default change (Phi-2,
  Phi-3, Qwen3, Qwen3-MoE) must show a measured prefill gain against its own pre-item-0 baseline on
  the same host and prompt length — a default flipped on that buys nothing on a given architecture is
  a finding to report and investigate, not a checkbox. Three of the four have a file on disk to
  measure on (Phi-3, Qwen3, Qwen3-MoE — see "Models needed"), so three of the four readings are
  obtainable in this tier without asking for anything. **There is no reference point for any of
  them.** The 3.85x figure is TinyLlama's, on the one architecture family where the kernel was already
  the default, so it predicts nothing here; record what each one actually does. Decode is also
  expected to move on these architectures, since attention has been measured at 64.2% of decode wall
  time at ctx around 512; a flat decode result here is a signal that the kernel is not actually being
  taken, not a pass.

  **Threshold, item 2 on its own.** Bytes staged host-to-device and device-to-host per 512-token
  prefill window must drop by **>= 70%** against the step-1 baseline. This is the item's primary
  number because it is the one that is not confounded by clock state or noise: a residency change
  either stops moving the bytes or it does not.

  **Threshold, the tier overall.** mistral-7b Q4_K_M on GPU must reach **pp ratio >= 0.10x**
  llama.cpp at `n_prompt=512` **and** pp must not fall with prompt length — the 512-token ratio must
  be greater than or equal to the 128-token ratio for every sweep model. Both are measured under the
  parity-corrected harness (`RAW_PROMPT=1`, warmup, median of three), and **both are re-derived from
  step 1's re-baseline before implementation starts**, because neither has ever been measured
  like-for-like. Understand the size of the 0.10x ask honestly: the best parity-corrected prefill
  figure in this repository is TinyLlama at 0.028x with the attention kernel on, so 0.10x is roughly
  a three- to four-fold improvement over the best result this project has produced, not a six-fold
  improvement over a 0.0166x reading that was partly an artefact. If step 1's re-baseline puts
  mistral-7b materially below 0.028x, restate this threshold against the re-baselined number and say
  so here rather than carrying a target that was set against the wrong denominator. Decode must not
  regress: tg ratio within 0.95x of the step-1 baseline for every sweep model. Vision gate per the
  existing rule: `latency_ms` <= 1.25x baseline, decode tps >= 0.80x baseline.

  **Decompose the ask before implementing it, and escalate if it does not add up.** This is the
  largest single performance ask in the plan and the only milestone tier that previously had no
  stop-before-implementing clause; [Tier 10](TIER-10-gpu-backend-breadth-cpu-simd.md) already carries
  one for its own 0.20x CPU milestone and this tier now matches it. Once the re-baseline and the
  breakdown exist, and **before** implementing the staging and chunk-sizing items, write down the
  expected contribution of each named item — the GPU-attention default change, the removal of
  host-device staging, chunk sizing, residual attention — as a multiple on the re-baselined pp ratio,
  read off the breakdown rather than estimated. If the named items do not plausibly sum to the
  threshold, say so in this file **before** implementing rather than after, and escalate. Shipping
  four items that were never expected to reach the number and reporting the miss afterwards is the
  outcome this clause exists to prevent; reporting up front that the number needs a mechanism no item
  here owns is a useful result, and it is what tells the user whether the missing mechanism belongs
  to [Tier 04C](TIER-04C-packed-weight-matmul.md), to Tier 02, or to a tier that does not exist yet.

  **Contingency, in the same spirit as Tier 01's.** If the breakdown in step 3 shows prefill time is
  dominated by a term this tier cannot move without work owned by a later tier (for example: residual
  attention at long context, which is Tier 02; or per-format kernels, which is Tier 04), do not
  iterate indefinitely. Publish the breakdown, ship whatever items the breakdown does justify, state
  the measured ratio honestly, and mark the tier **partial-complete** with a named successor tier for
  the dominant term — then escalate to the user, since that re-scopes another tier. Item 0 is
  excluded from this contingency: it ships regardless, because removing a silent per-architecture
  degrade is worth doing whatever the measurement says. Item 2 has its own escalation path instead
  of this one — if Tier 01's residency primitive did not ship, item 2 does not proceed on a
  substitute design; escalate.

## Models needed

The four standing sweep models (`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`,
`qwen2.5-3b-instruct-q4_k_m.gguf`, `Phi-3.5-mini-instruct-Q4_K_M.gguf`,
`mistral-7b-instruct-v0.1-q4_k_m.gguf`) plus `moondream2-q5_k.llamafile` for the vision gate cover
items 1 through 4. All present. Item 5 (the script-leak fix) needs no model file, though verifying it
means launching an engine at least once per script.

**Item 0's model coverage, corrected.** An earlier draft of this section listed Qwen3 and Qwen3-MoE
as untestable for want of a file. Both files are on disk, and have been since before this tier was
written — [`INVENTORY.md`](INVENTORY.md) was corrected on 2026-09-24 and
[Tier 01](TIER-01-gpu-activation-residency.md)'s execution record notes the correction, but this
table was not updated with it. Read from the file headers:

| Architecture | Testable today | With what |
|---|---|---|
| Phi-3 | yes | `Phi-3.5-mini-instruct-Q4_K_M.gguf` |
| Phi-2 | yes | `phi-2.Q4_K_M.gguf` (1.7 GB, header reports `general.architecture=phi2`). On disk since 2026-09-23 and first noticed on 2026-09-26, while this table still said the only Phi-2 was `moondream2-q5_k.llamafile`'s backbone reached through the vision path |
| Qwen3 | yes | `Qwen3-1.7B-Q4_K_M.gguf` (1.1 GB, header reports `general.architecture=qwen3`) |
| Qwen3-MoE | yes, with a caveat | `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (18.6 GB, `qwen3moe`). Too large to fully offload to 8 GiB of VRAM, so its per-architecture gain is measured at partial offload and the offloaded layer count is recorded beside the figure. A partial-offload measurement is a real measurement; absence of a full-offload one is not grounds for leaving the default off |

So item 0 can ship kernel support for all four **and** produce the required per-architecture measured
gain for every one of them — Phi-2, Phi-3, Qwen3 and Qwen3-MoE (the last at partial offload). No file
is missing for this item, and nothing here needs asking the user for.

**No architecture may therefore ship with its default resolved off for want of a measurement.** An
earlier version of this section pre-authorised exactly that for Phi-2, on the strength of a file gap
that had already closed. Every one of the four has a real file, so every one gets a measured default.
Turning on an unvalidated default is exactly the silent-degrade this item exists to remove — and so is
leaving a default off
because a table said a file was missing when it was not.

**Lesson recorded for the rest of the tree, not just this tier.** A correction to
[`INVENTORY.md`](INVENTORY.md) is propagated to every tier whose "Models needed" section cites the
corrected row, in the same pass that makes the correction — otherwise a stale availability claim
silently narrows a later tier's scope, which is what happened here and what would have shipped two
architectures' defaults off on a false premise.

## Exit criteria

- [ ] `--gpu-attention` defaults to on for every architecture whose gain was actually measured on a
      real model — Llama-family, Mistral, Qwen2, **Phi-2, Phi-3, Qwen3 and Qwen3-MoE**, since a file
      exists on disk for each of those last four (see "Models needed"; Qwen3-MoE is measured at
      partial offload and the offloaded layer count is recorded beside its figure). **No architecture
      on this list has a file gap to fall back on**, so none may keep its default resolved off for
      want of a measurement; the Phi-2 exception this criterion used to carry was written against a
      gap that had already closed. The ROCm answer decided and either implemented (`NEEDS-AMD-HARDWARE`) or
      made an explicit announced fallback. No path resolves to scalar silently, whichever way each one
      landed.
- [ ] Each newly covered architecture shows a measured prefill gain against its own pre-item-0
      baseline. "Unmeasured" is only acceptable for an architecture with no file on disk, and the
      claim is checked against `models/` at the time this tier runs rather than against this file's
      table — that table was wrong once already.
- [ ] Greedy-decode divergence from the FP16 KV mirror characterised per newly covered architecture
      and documented in `docs/howto.md` and `--help`, with `off` retained as the bit-identical
      CPU-parity baseline.
- [ ] LoRA training and `--lora-play` exemptions re-verified as still holding and still warning.
- [ ] `compare-llama-cpp.sh`'s stale default-lane `--gpu-attention` labelling corrected, and the
      first re-baselined run states which previously published lanes were mislabelled.
      `docs/performance.md`'s GPU-resident-attention section is corrected in the same pass: it states
      the default is **off** while `GpuAttentionOptions.fromEnv()` defaults to `auto`.
- [ ] Item 0's own attributable measurement published separately from the rest of the tier's, so the
      default change's effect is visible on its own.
- [ ] `juno.DeviceStaging` and `juno.WeightDequant` exist, are enabled in
      `scripts/performance-tests/juno-perf.jfc`, are aggregated by `JfrMetricsExtractor`, and have
      `metrics` tests that failed on the pre-tier build for the right reason. Until this is checked,
      the breakdown criterion below cannot be satisfied by anyone.
- [ ] Per-term prefill breakdown published for all four sweep models at `n_prompt` 128 and 512, with
      no unattributed residue — every term named, including host-device staging and dequantization.
- [ ] The threshold decomposition written down before implementation, with each item's expected
      contribution read off the breakdown, and an escalation recorded here if they did not sum.
- [ ] Prefill activations stay device-resident across a layer's projections, with the materialization
      boundary documented; bytes staged per 512-token window down >= 70% against the step-1 baseline;
      `sgemmLayerInto`'s per-matmul allocate-and-copy removed via the non-allocating batched form,
      whose contract matches the one Tier 10 item 4 adds for CPU.
- [ ] Chunk-sizing defaults reviewed per surface; any surface still pinned at `32` has a measured
      reason, not an inherited one.
- [ ] `scripts/performance-tests/check-plan-thresholds.sh` exists, passes against this tree, and fails
      against a deliberately broken copy of a tier file with its threshold block removed — a check that
      cannot fail is not a check. Recorded here as shipped, so later tiers run it rather than re-deriving
      execution rule 7's enforcement.
- [ ] The engine-keepalive subshell leak is gone from all eight sibling scripts, via a shared helper in
      `perf-lib.sh` rather than eight copies of the edit, and a full run of each of the nine scripts
      leaves zero leftover shells and zero leftover pipes. Until this is checked, a `pgrep` check for a
      running sweep is unreliable, which already cost real time once.
- [ ] Prefill throughput no longer degrades with prompt length: the `n_prompt=512` pp ratio is greater
      than or equal to the `n_prompt=128` pp ratio for every sweep model — **both measured under
      `RAW_PROMPT=1` with the same `--vector` setting**, which no published pair of runs has ever
      been. Step 1 establishes whether the degradation is real before this criterion can be scored.
- [ ] Threshold above met, or the tier is explicitly marked partial-complete with the dominant term
      named and assigned to a successor tier (not silently marked complete).
- [ ] Decode (tg) and vision both verified not regressed, with published numbers.
- [ ] Cross-surface checklist fully resolved.
- [ ] Docs (`docs/howto.md` for any `--prefill-batch` default change, `docs/performance.md`,
      `docs/agent-arch.txt`) updated, Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
