# Tier 01B: Prefill throughput

Status: in progress — implementation steps 0 and 1 complete (2026-09-30); step 2 (re-baseline) next
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
runs**, and step 2 below (the re-baseline) is what actually establishes it.

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
yet**, not "existing kernel code reaching more architectures." Size it accordingly. It is still the
first forward-pass change in this tier (only the instrumentation of step 1 and the re-baseline of
step 2 precede it), because every other measurement in this tier would otherwise be taken against a baseline that
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
   whose cost is paid during step 2 rather than after it. Neither needs a model, a GPU or a build.

   **A third item, handed over by Tier 01 on 2026-09-27: find and fix the one fast prefill
   repetition before step 2's re-baseline.** In `compare-llama-cpp.sh` sweeps, one of three prefill
   repetitions sometimes reads four to five times faster than the other two: the tinyllama default lane
   read 898 and then 819 t/s in two separate runs against about 168 for the others, and the qwen2.5-3b
   tuned lane 130.6 against about 80. On tinyllama it was **repetition 1 both times**, the measured
   request that directly follows the warm-up. Medians are unaffected (median of three), but the row
   breaks the 15% spread rule and forces re-runs, and step 2's re-baseline is this tier's
   before-measurement, so it must not need re-runs for a harness artifact. Suspects, not verified: (a)
   prefix-cache or KV reuse between the warm-up and the measured request, so the measured request does
   not actually prefill every prompt token; (b) a timing boundary in how the prefill lane reads
   `prompt_eval_tps`. Evidence: `docs/perf-compare/20260927T091155Z/` and `20260927T093054Z/` (the INDEX
   banners, and the per-repetition `*-prefill-rep*-juno.json`: `prompt_eval_tps`). If the cause is (a),
   the fix belongs to the harness (a fresh session or a distinct prompt per measured request), not to
   the engine's reuse; if the engine turns out to reuse a prefix the request did not ask to share,
   that is a correctness finding and is raised with the owner before the re-baseline.

   *Two harness changes already landed, 2026-09-27, after Tier 01's close-out (owner-approved,
   measurement time only, no check weakened):* `smoke-gpu-residency.sh` generates the full `--n-gen`
   only on the first and last request (which must match) and `--mem-n-gen` (default 8) on the
   requests between, which only feed the per-request memory reading (the 16-request llama-1-30b run
   took 3 h 11 min; the memory reading is per request, not per token); and `compare-llama-cpp.sh
   --juno-jar PATH` runs another build's jar from this checkout, logs it, and never publishes, for
   same-hour A/B runs. `docs/performance.md` gives the pre-gate and A/B recipe. **Lesson for this
   tier's item 1a:** the Tier 01 close-out spent about 75 minutes of A/B, bisect and hand-added
   per-phase timing to find a prefill cost that a `juno.DeviceStaging`-style span would have shown in
   the first sweep. *Decided by the owner 2026-09-27, in the plan review: item 1a now runs first, as step 1 below.*

   *`check-plan-thresholds.sh` shipped 2026-09-27; see the execution record.*
1. **Build the `juno.DeviceStaging` and `juno.WeightDequant` spans (scope item 1a) first, before the
   re-baseline.** It is instrumentation, not a forward-pass change, so it does not have to wait behind
   item 0, and two things depend on it being in place before any measurement: the item-2 threshold
   reads bytes staged per 512-token window "against the step-2 baseline", which has no staged-bytes
   figure unless the spans exist when that baseline is taken; and Tier 01's close-out spent about 75
   minutes finding by bisect a prefill cost these spans would have shown in the first sweep. The spans
   do add work around every staging copy, so land them with a same-hour A/B (pinned clocks, README
   "No-regression gates tighter than the floor") of the build with the spans against the build without,
   on two sweep models at `n_prompt` 512: prefill **>= 0.98x**, or the spans are made cheaper before the
   re-baseline is taken on top of them.
2. Re-baseline. Run `compare-llama-cpp.sh --gpu --pin-clocks` at `n_prompt` 128 and 512 on all four sweep
   models on current HEAD (post-Tier-01), under the parity-corrected harness required by this plan's
   "Benchmark parity preconditions" (README), and with each lane's actual resolved
   `--gpu-attention` value recorded rather than assumed — the historical numbers quoted above were
   taken before those corrections, with at least one lane mislabelled, and are not a valid
   before-measurement for this tier's gate. This run carries the staged-bytes and dequant figures
   from step 1's spans, which are item 2's before-measurement. *Since step 1 the spans are opt-in: take
   the ratios from the default run and the staged bytes and dequant figures from a separate
   `--device-spans` run of the same build, which costs about 6% of prefill.*
3. **Land item 0 next, before anything else that changes the forward pass.** Not because it is cheap —
   it is four kernel integrations against four handlers that each own their own KV map and attention
   math, plus a capability-reporting mechanism that does not exist yet — but because leaving it until
   later would mean every subsequent measurement in this tier is taken against a baseline that is fast
   on four architectures and slow on four others. Re-measure immediately after, so the default change
   has its own attributable number, and record that number per architecture: the Llama-family 3.85x
   says nothing about what these four will do. All four have a real file to measure on — see
   "Models needed", and check `models/` rather than trusting any table.
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
  flips on; do not inherit the Llama-family answer for it. *Build on
  `juno-master/.../GpuForwardPassIT` (rebuilt 2026-09-27, see Tier 01's out-of-tier table): it already
  holds the Llama-family GPU path to the CPU oracle by relative L2, cosine, top-1 and top-5 logits and
  a 16-token greedy decode, with bounds calibrated against planted faults. Extend it per architecture
  with each one's own calibration rather than writing a second oracle.*
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
  or a live check, run before step 2's re-baseline: for every measured prefill repetition, the prompt
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
  a finding to report and investigate, not a checkbox. All four have a file on disk to
  measure on (Phi-2, Phi-3, Qwen3, and Qwen3-MoE at partial offload — see "Models needed"), so all four
  readings are obtainable in this tier without asking for anything. **There is no reference point for any of
  them.** The 3.85x figure is TinyLlama's, on the one architecture family where the kernel was already
  the default, so it predicts nothing here; record what each one actually does. Decode is also
  expected to move on these architectures, since attention has been measured at 64.2% of decode wall
  time at ctx around 512; a flat decode result here is a signal that the kernel is not actually being
  taken, not a pass.

  **Threshold, item 2 on its own.** Bytes staged host-to-device and device-to-host per 512-token
  prefill window must drop by **>= 70%** against the step-2 baseline. This is the item's primary
  number because it is the one that is not confounded by clock state or noise: a residency change
  either stops moving the bytes or it does not.

  **Threshold, the tier overall** (the README's Tier 01B milestone rows, restated 2026-09-27).
  At `n_prompt=512`, GPU:
  - pp ratio **>= 0.10x** llama.cpp on every sweep model except Phi-3.5-mini (tinyllama, qwen2.5-3b,
    mistral-7b; binding reference 0.062x on tinyllama, read at `n_prompt=128`);
  - pp ratio **>= 0.08x** on Phi-3.5-mini (reference 0.040x at `n_prompt=128`) — the model whose
    handler item 0 newly gives the GPU attention kernel, and the binding constraint on the program's
    end-of-plan pp target;
  - pp must not fall with prompt length: the 512-token ratio **>= 1.00x** the 128-token ratio for
    every sweep model.

  All three are measured under the parity-corrected harness (`RAW_PROMPT=1`, warmup, median of three,
  `--pin-clocks`), and **the first two are re-read against step 2's 512-token re-baseline before
  implementation starts**: their references were taken at 128 tokens because no parity-corrected 512
  reading exists. If the 512 readings sit materially below the 128 ones, restate both rows against the
  512 figures, here and in the README table, keeping the same multiples (about 1.6x on the binding
  non-Phi model and 2.0x on Phi-3.5-mini) rather than carrying a target set against the wrong
  denominator.

  *Why this changed from "mistral-7b >= 0.10x".* The current reference (`20260927T232837Z`) already reads
  mistral-7b at 0.101x at 128 tokens, after the out-of-tier memory fix and the extracted FP16 pack loop
  that landed at Tier 01's close-out. A mistral-only milestone would have credited this tier with work
  done before it began, while leaving the model that binds the program target untested. The earlier
  framing ("0.10x is a three- to four-fold improvement over TinyLlama's 0.028x") predates both the
  parity-corrected re-baseline and that fix, and no longer describes the ask.

  Decode must not regress: Juno tg t/s **>= 0.95x** the step-2 build on every sweep model, from a
  same-hour interleaved A/B with pinned clocks (README, "No-regression gates tighter than the floor are
  Juno-against-Juno"); the tg ratio is recorded against the program target, not gated. Vision gate per the
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

  **Contingency, in the same spirit as Tier 01's.** If the breakdown in step 4 shows prefill time is
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

## Execution record

Status of the tier: in progress. Implementation steps 0 (threshold check, keepalive reaping, the fast
prefill repetition) and 1 (the copy and dequantization spans, opt-in, gate met) are complete; step 2 (the
re-baseline, with `--device-spans` for the staged-bytes figures) is next. This section also records the
plan-review pass of 2026-09-27, which landed ahead of step 1.

### 2026-09-27 — plan-review pass: threshold check, CI trigger, reference and gate rules, clock pinning

Scope, as the owner directed from the plan review: re-point the program target's reference column,
ship and extend the rule-7 check, record the CI trigger, correct Tier 02 item 1, make tight
no-regression gates same-hour Juno-against-Juno A/B, move item 1a ahead of item 0, restate this tier's
milestone, and pin clocks plus record build identity in the comparison harness. No forward-pass,
MatVec, KV or batching code changed.

**`check-plan-thresholds.sh` shipped** (`scripts/performance-tests/`), implementation step 0's first
item and this tier's exit criterion. It fails on a tier mentioning a perf gate in any capitalization
without a numeric `**Threshold` block (a tier with no gate declares `**No perf gate**`), on an exit
criterion reading "no unexplained regression", and on a malformed intermediate-milestone row or an
active milestone its reference reading already meets. Evidence, both directions:
- Against the tree as it stood when written: **7 failures** — Tiers 04 and 04C ("no unexplained
  regression" in an exit criterion), 04B (a threshold block with no comparison operator: "exact
  equality", "within 0.95x"), 05 (a lowercase "perf gate" with neither threshold nor declaration), and
  all three rows of the old free-text milestone table.
- After the fixes (the milestone table made machine-readable, the Tier 04 milestone retired as met on
  arrival, the 04B threshold rewritten, Tier 05 declaring `**No perf gate**`, both exit criteria
  stating their numbers): `check-plan-thresholds: ok (18 tier files, milestone table checked)`.
- Against a scratch copy with four planted defects (Tier 03's `**Threshold` markers removed, a "no
  unexplained regression" exit criterion added to Tier 11, the Tier 10 milestone lowered below its
  reference, Tier 05's declaration removed): exit 1 naming exactly those four.

**The CI revisit trigger fired during Tier 01 and is recorded here** (README, "Owner and revisit
trigger": a tier whose smoke matrix exceeds thirty minutes of hands-on execution records the
re-examination in its own file). Evidence from Tier 01's record: the eleven-module unit reactor took
22:46 to 27:14 per pass across seven passes; `smoke-gpu-residency.sh` on llama-1-30b took 3 h 11 min
before its request count was trimmed; the RoPE-pairing A/B took 36 minutes. What changed since the
README's original judgement: every one of those runs was started and read by hand, one run was
misread from a pipeline's exit code and a second reactor started on top of it (Tier 01, four-hour
run), and the rule-7 check shipped here is exactly the kind of GPU-free gate a CI job would run on
every change. **Outcome: open — the decision belongs to the owner and is due before step 2's
re-baseline.** *2026-09-28: the owner deferred it to the end of step 1 and asked to be asked
again then; the recommendation on the table is a GPU-free two-job workflow (script checks on every
push; the unit reactor and `mvn verify -pl juno-master` on pull requests and pushes to `main`), with
real-model and performance gates staying manual.* The two options, as Tier 07 item 5 frames them: adopt a `.github/workflows/` job
covering `mvn test` on the eleven unit-test modules, `mvn verify -pl juno-master`,
`check-plan-thresholds.sh` and `compare-llama-cpp.sh --selftest` (GPU and real-model gates stay
manual on this host), stating that fraction; or decline, naming what changed or that nothing did. Tier
07 still carries its own re-examination.

**Harness: clock pinning and build identity** (`compare-llama-cpp.sh`). `--pin-clocks` sets the
performance governor and turns turbo off for the run and restores both on exit, and locks the GPU
graphics clock where the driver allows (`--pin-gpu-mhz`); it needs prompt-free sudo and refuses to run
if it cannot pin the CPU. `host.json` and `INDEX.md` now carry `clock_pinned` and what was pinned, the
Juno commit and dirty flag, the jar's hash, the JDK build, the JVM flags, the GPU driver and the
reference tool's build commit, and an unpinned run's index says it is not usable for a gate tighter
than the noise floor. The Juno heap is now fixed with `-Xms` equal to `-Xmx` (was `-Xms512m`).
Verified: `bash -n` and `--selftest` (all checks pass). **Not yet verified on a live run**: a dry run
exercising the pinned and unpinned paths was not executed in this pass, so the first real sweep is
also the first live exercise of both; check its `host.json` fields before trusting them.

### 2026-09-28 — implementation step 0: engine-keepalive reaping, and the fast prefill repetition

Scope: implementation step 0's two remaining items (scope item 5, and the third item handed over by
Tier 01). No forward-pass, MatVec, KV or batching code changed. One `metrics` change (two new keys) and
harness changes only.

**Plan-versus-code drift found before starting, and corrected here rather than worked around.**
- Scope item 5 lists **eight** sibling scripts. There are **nine**: `smoke-gpu-residency.sh`, added in
  Tier 01, uses the same `< <(while true; do sleep 3600; done)` keepalive and was not in the list. With
  `compare-llama-cpp.sh` that makes ten scripts on the shared helper, not nine.
- Item 5 says `perf-lib.sh`'s "one assignment is guarded". It has several top-level assignments, all of
  the guarded `${X:-default}` form, so the conclusion (a safe host to source) holds.
- `compare-prefill-batch.sh --help` also documents `--gpu-attention`'s default as "off", the same stale
  label item 0 names for `compare-llama-cpp.sh`. Left for item 0 to correct with the other one.

**Tier 01's named-pipe helper had two defects of its own**, both found while moving it into `perf-lib.sh`
and both shown by the new selftest failing against a copy of it:
- The engine inherited the script's read-write descriptor on the pipe, so it held a writer on its own
  stdin and never saw end of file. Its comment ("the engine exits on its own once the pipe closes") was
  false. Harmless while `stop_juno` kills the engine, but a script that died without stopping it left
  the engine running.
- `exec {fd}>&- 2>/dev/null` applies the redirect to the shell itself, so from its first engine stop
  onward **`compare-llama-cpp.sh` discarded all of its own stderr** — every `warn` and `die` message of
  every sweep after the first engine of that sweep. Published result files are unaffected (they are
  written to files, not stderr); what was lost is warnings a reader never saw.

**The engine-keepalive fix.** `perf-lib.sh` gains `perf_engine_stdin_open` / `perf_engine_exec` /
`perf_engine_stdin_release`: a named pipe under `$TMPDIR`, held read-write by the launching shell, with
the child's inherited copy closed so releasing it delivers end of file. All ten scripts source it and
call the release from their `stop_juno` on every path. `selftest-engine-stdin.sh` (new, no model or GPU)
checks, over three launches, that the engine stays up while the pipe is held, holds only its stdin on
it, exits on release, and leaves no pipe on disk, no descriptor in the shell and no helper process;
that stderr survives a release; and that an engine whose script exits without stopping it is not
orphaned. Its negative control runs the old process substitution and requires the leftover check to
see the sleep loop it leaves. Evidence: passes (23 checks); against a copy of Tier 01's helper it fails
8 checks naming the inherited writer, the engine outliving release, the orphaned engine, and stderr
redirected to `/dev/null`.

**Exit condition, run 2026-09-28 on this host: all ten scripts ran to completion with an engine
launched and stopped at least once, and left nothing behind.** Small parameters (TinyLlama, or the
script's default model where it has one; CPU where the script defaults to it), `--no-publish`, one
after another, checked 3 s after each returned:

| Script | Exit | Wall | New keepalive loops | Pipe files | Pipe descriptors held (`lsof`) | Engine JVMs left |
|---|---|---|---|---|---|---|
| `compare-parallel.sh` | 0 | 22 s | 0 | 0 | 0 | 0 |
| `compare-prefill-batch.sh` | 0 | 28 s | 0 | 0 | 0 | 0 |
| `compare-mixed-prefill.sh` | 0 | 59 s | 0 | 0 | 0 | 0 |
| `compare-schedule.sh` (`--mode tps`) | 0 | 37 s | 0 | 0 | 0 | 0 |
| `smoke-grammar.sh` | 0 | 117 s | 0 | 0 | 0 | 0 |
| `smoke-tools.sh` | 0 | 510 s | 0 | 0 | 0 | 0 |
| `smoke-tier00-consistency.sh` (`--no-gpu`) | 1 (see below) | 350 s | 0 | 0 | 0 | 0 |
| `smoke-gpu-residency.sh` (tinyllama, 2 requests, no cluster) | 0 | 30 s | 0 | 0 | 0 | 0 |
| `compare-vision.sh` (moondream2, GPU, `--skip-build`) | 0 | 552 s | 0 | 0 | 0 | 0 |
| `compare-llama-cpp.sh` (tinyllama, CPU) | 0 | 47 s | 0 | 0 | 0 | 0 |

"New keepalive loops" counts `sleep 3600` processes whose parent shell started after the run began. A
PID baseline does not work: 24 orphaned loops from earlier sessions (parents are old
`smoke-tier01-gpu-residency.sh` and `smoke-gpu-residency.sh` runs) respawn their `sleep` every hour, so
a first version of this check reported 5 and then 14 "new" loops that were all respawns. Those 24 are
still on the host; they are what this item exists to stop accumulating, and killing them is left to the
owner. The CPU `compare-llama-cpp.sh` run also showed the stderr fix working: its warnings reached the
terminal after the first engine stop.

**The fast prefill repetition: root cause, and it is neither suspect.** Not a prefix reuse (suspect a)
and not the harness's arithmetic (suspect b). The fast repetitions were not faster requests: the
tinyllama repetition reading 898 t/s had a request latency of 808 ms against 782 ms for its sibling
reading 170 t/s. What was wrong was the recorded span. On this host **CPU0's timestamp counter reads
633 ms ahead of CPUs 1 to 11** (probe: rdtsc against `CLOCK_MONOTONIC` per pinned CPU; offsets
-632.8 to -633.2 ms on every other core). The kernel detected it at boot ("Measured 208056 cycles TSC
warp between CPUs, turning off TSC clock", clocksource switched to `hpet`), but the CPU advertises an
invariant counter, and the JVM enables `UseFastUnorderedTimeStamps` ergonomically on that flag alone,
so **JFR stamps events with the raw counter while `System.nanoTime` uses the kernel clock**. A span
that begins on CPU0 and ends elsewhere reads 633 ms short; the reverse reads 633 ms long. Reproduced
deterministically with a one-event program re-pinned mid-span (`taskset`): a real 1500 ms span reads
866 ms (CPU0 to CPU1), 2128 ms (CPU1 to CPU0) and 1497 ms (CPU1 to CPU2); with
`-XX:+UnlockExperimentalVMOptions -XX:-UseFastUnorderedTimeStamps` it reads 1501 ms both ways.

What this explains beyond the fast repetition:
- **The ~633 to 636 ms "GC pauses"** the README's noise-control section used to withdraw the pause rule
  (and Tier 01's record discusses at length) were this offset, not pauses. The withdrawal stands — the
  dispersion rule is the better gate — but its stated reason ("the pause counter does not measure
  stopped time on this host") was the symptom, not the cause. Corrected in the README and in
  `docs/performance.md`.
- **Readings that were too slow**, not only too fast, and in the generation lane as well as prefill.
  Applying the new check to every published repetition of `20260927T091155Z`, `093054Z`, `232837Z`
  and `234659Z` withholds **23 of 156**. Every withheld residual sits near +633 ms or near the normal
  overhead minus 633 ms; the healthy residuals are 9 to 170 ms (forward passes) and 22 to 106 ms (token
  span).
- **Effect on the current reference column: none at its stated precision.** Recomputing each row's
  median without the withheld repetitions moves no reference row by more than 2.2% (tinyllama pp
  224.89 to 229.78; qwen2.5-3b tuned pp 88.93 to 90.24; the rest within 0.4%). The binding Phi-3.5-mini
  pp ratio is unchanged at 0.040x; tinyllama's pp ratio, the milestone's binding non-Phi reference, moves
  from 0.062x to 0.064x (229.78 / 3599.25). Neither changes the milestone rows, which step 2's 512-token
  re-baseline restates anyway; the reference column is left as published. One published row is
  wrong outright but is not in the reference: `234659Z` qwen2.5-3b default generation has **all three**
  repetitions withheld; two of its token spans were misread long and its published 23.93 t/s median
  should be about 31 (its engine-clock `api_token_gen_tps` is 26.9 to 27.0 on all three, identical to
  the clean sweep).

**The fix, as the owner chose it (2026-09-28): the JVM flag, applied automatically.** `perf-lib.sh`
gains `perf_jfr_clock_jvm_flags`, which returns the two flags above whenever the kernel clocksource is
not `tsc` (`PERF_JFR_OS_CLOCK=auto`, the default; `1` forces it, `0` keeps the JVM default for an A/B).
`compare-llama-cpp.sh` adds them to Juno's JVM flags and records `jfr_timestamp_source` in `host.json`
and the INDEX; `compare-vision.sh` passes them through `JUNO_JVM_OPTS` (honored by launchers from
`release-0.1.2` on, so both sides of a baseline comparison get the same clock). Cost, measured with a
microbenchmark on this host: one JFR event from about 0.7 to about 3.0 us, which at about 260 events
per decoded token is on the order of 3 to 4% of decode while a recording runs; prefill spans are per
layer, not per token.

**Live check, same hour, 2026-09-28 00:36 to 00:46** (`compare-llama-cpp.sh --gpu`, TinyLlama
Q4_K_M, `n_prompt` 128, 2 warm-ups, 6 repetitions per lane, no tuned lane, `--no-publish`, clocks not
pinned; `PERF_JFR_OS_CLOCK` alternated 0, 1, 0, 1, so 24 repetitions per setting):

| JFR clock | Withheld | Forward-pass residual | Token-span residual | pp median (min/max) | JFR tg median (min/max) | Engine-clock tg median (min/max) |
|---|---|---|---|---|---|---|
| raw counter (JVM default) | **1 of 24** (residual -581 ms) | 6 to 214 ms | 9 to 187 ms | 224.33 (202.9 / 230.3) | 70.36 (68.4 / 71.5) | 57.94 (56.1 / 59.0) |
| operating-system clock | **0 of 24** | 4 to 38 ms | 5 to 13 ms | 225.21 (216.5 / 231.0) | 67.62 (64.6 / 68.3) | 55.72 (53.2 / 56.2) |

The operating-system clock removes the misreads and tightens every residual. It costs **3.9% of
generation read under JFR** (and 3.8% on the engine's own clock, since the recording is running during
the request); prefill does not move (+0.4%, inside the spread). That matches the microbenchmark
estimate. Generation readings from this harness are therefore about 4% below what the same build read
before the change, which is the measurement boundary noted below; a Juno-against-Juno A/B is unaffected
because both sides carry the same cost.

**A guard, independent of the fix.** `jfr_summary_json` now checks every repetition against its own
request: the engine latency minus the forward-pass spans, and minus prefill and the token span, must
each lie between -25 ms and 300 + 3 ms per generated token (a negative residual beyond that is
physically impossible, since the spans are sequential and inside the request). A failing repetition
has its readings withheld — not replaced by the API figure, which is a different measurement — with the
reason in its JSON, a warning, and a line in the INDEX; the median is taken over the rest. It also
checks that the measured request prefilled its whole prompt from position 0, read off two new
`JfrMetricsExtractor` keys, `juno.PrefillBatch.tokens` and `juno.PrefillBatch.min_start_position`
(written on every run; -1 when there was no prefill window), with four `metrics` tests that failed on
the unchanged extractor (keys absent) and pass now. On the evidence, suspect (a) did not happen: every
measured prefill is one 127-token window at position 0 plus one forward pass at position 127, for a
128-token prompt. `--selftest` gains 17 cases built from the evidence runs' own numbers.

*Corrected the same day:* the first version of the token-span leg flagged a healthy CPU repetition
(8 tokens, 314 ms per decode step: 484 ms left outside the token span). The span starts at the first
token, so the gap before it includes one whole decode step, which is 14 to 43 ms on the GPU sweeps the
bounds were calibrated on and hid the mistake. The leg now subtracts one mean decode step. Re-applied to
the four evidence sweeps it withholds the same 23 repetitions, and healthy token-span residuals sit at
-1 to 29 ms (one at 267 ms, inside the bound).

**Found while verifying: `smoke-tier00-consistency.sh` failed 3 of 36 checks at HEAD** (fixed the
same day at the owner's direction; recorded under "Out-of-tier changes" below). Its local-mode audit
expects `Unsupported model architecture '<arch>'` for qwen35, mistral3 and minimax-m2. Since commit
`a906a8d` (Tier 01, pre-tokenizer dispatch), `ConsoleMain`'s local REPL loaded the tokenizer before the
handler loader, and the tokenizer refused each of those files' pre-tokenizer type first (`qwen35`,
`tekken`, `minimax-m2`), as an uncaught `IllegalArgumentException` stack trace. The files were still
refused and the cluster legs still named the architecture, so nothing loaded that should not; what
regressed was the reason given. Tier 01 closed without re-running this smoke's local leg. The same
ordering existed at six other entry points (three more `ConsoleMain` paths, `CoordinatorMain`,
`JunoPlayer`, `LoraTrainer`); the cluster legs passed only because the nodes rejected first.

**Measurement boundary.** Juno readings from `compare-llama-cpp.sh` and `compare-vision.sh` taken with
the flag are on a different JFR clock from every published run. Generation readings move by the flag's
own cost under JFR; prefill is expected not to. The step 2 re-baseline is the first run on this side of
it, so no gate in this tier straddles it.

### 2026-09-28 to 2026-09-30 — implementation step 1: `juno.DeviceStaging` and `juno.WeightDequant`

Scope: scope item 1a, its same-hour A/B gate, and one harness correction the gate exposed. No change to
what any copy, kernel or matmul computes. **Gate met** (prefill 0.999x and 1.000x); two owner decisions
changed the design on the way, recorded below.

**Plan-versus-code drift found before starting.**
- **"Duration around every `gpuMemcpy`" measures nothing on the prefill path.** Every activation copy
  the CUDA batched GEMM issues is `cudaMemcpyAsync` on one stream, and so is `Q4KMmqKernel.launchDequant`,
  with one `cudaStreamSynchronize` at the end of the call. A host clock around each call times the
  enqueue and would have reported staging and dequantization as free. No GPU event bindings existed.
  **Owner decision 1 (2026-09-28): time asynchronous work on the device with stream events**, added
  vendor-neutrally to `GpuBindings` (`gpuEventCreate`/`Record`/`ElapsedTime`/`Destroy`, CUDA and HIP).
- Three synchronous copies read a kernel's result straight back (attention output, norm output, the
  FP32 BLAS batch), so a host clock would also time the kernel; a timed device-to-host copy drains
  the default stream first. The host waits for that kernel either way.
- `LlamaTransformerHandler.dequantize` is **load-time only** (every caller is an FP16 upload), not a
  per-prefill term. It is counted as the item asks, under `timing=host`, apart from the per-call device
  dequantization (`timing=device`) the breakdown needs.
- The copies a prefill window issues are not only matmul activations: the attention kernel's pointer
  and length tables, the norm weight and the KV mirror's per-position rows (22,528 of the 22,726 H2D
  copies of a TinyLlama window) happen inside it too. Each copy is therefore classified by the width
  of the forward call that issued it; `DeviceKvCache.appendToken` gained that parameter.
- `juno-perf.jfc` also ships inside the `juno-player` jar (resource copied from `scripts/performance-tests/`).

**The gate, four same-hour pinned A/B runs** (`docs/perf-compare/20260930T030253Z-tier01b-step1-spans/`,
per-pass readings in its `ab-readings.json`; TinyLlama and Mistral 7B Q4_K_M, `n_prompt` 512, A = the
jar of `38b1c6d`, median of three, all 24 invocations pinned):

| Run | Candidate | TinyLlama prefill | Mistral prefill | TinyLlama gen | Mistral gen |
|---|---|---|---|---|---|
| 1 | one JFR event per copy | 0.802x | 0.948x | 0.818x | 0.902x |
| 2 | totals per site and phase, decode untimed | 0.896x | 0.975x | 1.016x | 1.000x |
| 3 | as run 2, harness corrected (below) | 0.933x | 1.006x | 1.026x | 0.984x |
| 4 | spans opt-in, small copies timed 1 in 16 | **0.999x** | **1.000x** | 1.000x | 0.998x |

- **Run 1 failed on cost.** One event per copy, at a few microseconds each on this host's
  operating-system JFR clock, across 22,726 copies per prefill window and several hundred per token.
  Owner decision 2 (2026-09-29): **totals per site and phase**. `DeviceSpanTally` counts into
  lock-free cells and both events became periodic (`endChunk`), one event per non-empty cell; starting
  a recording ends the previous chunk, so a recording holds exactly its own window's work
  (`DeviceSpanTallyTest.workBeforeARecordingStartsIsNotInIt`). Decode copies are counted, not timed.
- **Run 2's prefill miss was not the copies.** Per-layer `juno.SwiGlu` read 251 and 161 ms for the first
  two of 22 layers, then 35 ms like the baseline, and `jdk.Deoptimization` named `DeviceStaging.copy`
  and `DeviceSpanTally.staging` (`unstable_if`): the harness warmed the engine up with no recording, so
  the recording-only branches compiled as uncommon traps and the measured request's recording tripped
  them, deoptimizing the batched-layer method they are inlined into. The baseline pays a smaller share
  of the same effect from JFR instrumenting its own event classes (85 and 42 ms). **Harness fix:** the
  last warmup runs under a discarded recording with the measurement settings (`compare-llama-cpp.sh`,
  `JFR_WARMUP_RECORDING_NAME`); verified every layer flat and no Juno method deoptimized in the window.
  This is a **measurement boundary for prefill** (TinyLlama baseline about 240 to 248 t/s); the step-2
  re-baseline is the first published run on this side of it, so no gate straddles it.
- **Run 3's remaining TinyLlama gap was the counting work.** The same candidate jar with the two
  events disabled in the settings matched the baseline in three interleaved unpinned rounds (248.7
  against 246.5 t/s median); enabled it read 237.6. Isolated checks ruled out the enabled check (5 ns),
  the matmul's stream-event timing (no measurable change over 60 calls at FFN shape) and the
  small-copy clock reads (sampling them 1 in 16 did not close it). Owner decision 3 (2026-09-30):
  **the spans are opt-in**. `juno-perf.jfc` ships them disabled; `juno-perf-spans.jfc` is an overlay
  that enables only those two, layered with a second `settings=` on `jcmd JFR.start` or by
  `compare-llama-cpp.sh --device-spans` (recorded as `device_spans` in `host.json`). A spans run costs
  about 6% of TinyLlama prefill (241.5 against 258.0 t/s, unpinned): read throughput from a run without
  the flag, staged bytes and the breakdown from a run with it. Bytes are exact either way.

**What shipped.**
- `node`: `DeviceStagingEvent` (`juno.DeviceStaging`: `site`, `direction`, `phase`, `copies`, `bytes`,
  `timedCopies`, `transferNanos`) and `WeightDequantEvent` (`juno.WeightDequant`: `format`, `timing`,
  `count`, `timedCount`, `dequantNanos`), both periodic; `DeviceSpanTally` (the totals and the periodic
  hooks); `DeviceSpanTimer` (pooled stream events per owner, `CudaMatVec`/`RocmMatVec` per call under the
  context lock, `ResidentChain` at each sync); `DeviceStaging` (the synchronous copy every other site
  uses; copies under 64 KB timed one in sixteen). Instrumented: all 22 asynchronous and 3 synchronous
  copy sites in `CudaMatVec`, all 15 in `RocmMatVec`, the resident-activation copies, and the 19
  synchronous sites in `DeviceKvCache`, `CudaGqaAttention`, `CudaRmsNorm`, `CudaRope`,
  `DeviceActivationBatch` and the three device matrix uploads. With the events off every site is the
  plain call: nothing counted, no stream event, no extra synchronize.
- `metrics`: `DeviceSpanBucket`: `juno.DeviceStaging.{H2D,D2H,D2D}[.{prefill,decode,other}]` and
  `.site.<site>.<phase>` as `count`/`bytes`/`timed_count`/`total_ms`/`estimated_total_ms` (each site's
  measured mean scaled to all its copies); `juno.WeightDequant` overall, per timing and per format.
- Harness: `juno-perf-spans.jfc`; `--device-spans`; the warmup recording; a `device_staging` object in
  each result (null without the flag or on an older build); `JUNO_JFR_SETTINGS_FILE` override;
  three new `--selftest` cases.

**Tests, written first and shown failing for the right reason.**
- `JfrMetricsExtractorDeviceSpansTest` (`metrics`, 7 cases): the first version failed on the unchanged
  extractor with every key absent (5 of 5); the estimated-duration case failed with its keys absent
  before `estimated_total_ms` existed (3 of 7). All pass.
- `DeviceStagingSpansTest` (`node`, `@Tag("gpu")`, 6 cases, GTX 1080): exact bytes and device timing for
  a 32-row FP16 GEMM; decode-width copies counted with bytes, untimed; the Q4_K batched GEMM's device
  dequantization apart from its copies; a weight upload as a host-timed `other` copy; a resident
  chain's upload and materialize; bit-identical GEMM output with and without a recording. 5 failed
  before any instrumentation existed (no events), the bit-identity control passed; all pass.
- `DeviceSpanTallyTest` (`node`, CPU, 4 cases): totals per site and phase, recording scoping, nothing
  counted without a recording, dequant totals. `WeightDequantEventTest` (CPU): the load-time host
  dequantization. `GpuBindingsDelegationTest`: the four new handles on both vendors.
- Regression: see the verification lines at the end of this section. ROCm: bindings, `RocmMatVec`
  sites and the vendor-neutral timer compile and are covered by the handle test's ROCm case, which
  skips here — **NEEDS-AMD-HARDWARE**.

**CI revisit, due at the end of this step (owner deferred it here on 2026-09-28): still open, put to the
owner in this step's report.** Recommendation unchanged (GPU-free two-job workflow).

**Verification, on the final tree (2026-09-30).** `mvn test` on the eleven unit-test modules: 1,832 tests,
0 failures, 49 skipped (existing assumptions), 26:22. `mvn test -pl node -Dgroups=gpu`: 145 tests, 0
failures, 7 skipped. `compare-llama-cpp.sh --selftest` and `check-plan-thresholds.sh` pass. Not run:
`mvn verify -pl juno-master` (stub-mode cluster ITs; step 1 changes no gRPC, scheduling or handler
contract) and the real-model live runner; both belong to the tier's closing matrix (implementation step
8). No `compare-lora.sh` or `compare-vision.sh` run: with the events off every copy site is the plain
call, and the pinned A/B above is the step's own gate.

### Out-of-tier changes (recorded per execution rule 9)

| Change | What it touched | Measurement boundary? |
|---|---|---|
| Working tree, 2026-09-28: architecture checked first at every entry point | `ModelFileGate.requireLoadable` (new, `node`) reads `general.architecture` and refuses an unverified one with `UnsupportedModelException` (new, an `IOException`, now also what `LlamaFamilyArchitectures` throws) before the config or tokenizer is read; called once in `ConsoleMain.main` before the mode dispatch (covers local, cluster, lora and their JFR variants), and in `CoordinatorMain`, `JunoPlayer.build` and `LoraTrainer.open`. The tokenizer's refusal becomes `UnsupportedPreTokenizerException` (new, still an `IllegalArgumentException`). `ConsoleMain` prints either as one `ERROR:` line and exits 1; cluster mode now refuses before forking nodes. Tests first: `ModelFileGateTest` (4 cases) and `BpePreTokenizerTest` tightened to the new type, both failing to compile before the change and passing after; `smoke-tier00-consistency.sh`, unmodified, 54 of 54 checks with the GPU legs (was 33 of 36 without them).; `mvn test` on the eleven unit-test modules passes (25:20 min), and `mvn verify -pl juno-master` passes, including `ThreeNodeClusterIT`, `TensorParallelClusterIT` and the unsupported-architecture IT. | **No.** One metadata read per model load, before any weights; nothing in the forward pass, MatVec, KV, batching or quantization. No published baseline is affected. |
| Working tree, 2026-09-27: `compare-llama-cpp.sh` heap and clock pinning | Juno launched with `-Xms` equal to `-Xmx` (was `-Xms512m`); optional `--pin-clocks`. | **Yes, for Juno readings from this harness**: a fixed-size heap changes when and how often G1 collects, and a pinned run runs at different clocks from an unpinned one (turbo off lowers absolute throughput for both engines). No published reference is invalidated by the code change itself, because none has been taken with it; the **step 2 re-baseline is the first run on this side of it** and every gate in this tier reads against that run, so no gate straddles the boundary. Do not compare a pinned run's absolute t/s with an unpinned run's. |

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
- [x] `juno.DeviceStaging` and `juno.WeightDequant` exist, are enabled in
      `scripts/performance-tests/juno-perf.jfc`, are aggregated by `JfrMetricsExtractor`, and have
      `metrics` tests that failed on the pre-tier build for the right reason. Until this is checked,
      the breakdown criterion below cannot be satisfied by anyone.
      *Checked 2026-09-30, with one owner-approved change to its wording: the events are registered in
      `juno-perf.jfc` but **disabled** there, and enabled by the overlay `juno-perf-spans.jfc`
      (`compare-llama-cpp.sh --device-spans`), because counting every copy costs about 6% of TinyLlama
      prefill. Step 1's gate is met on that design (prefill 0.999x and 1.000x, pinned A/B). Every run
      that feeds the breakdown or the item-2 staged-bytes threshold passes `--device-spans`.*
- [ ] Per-term prefill breakdown published for all four sweep models at `n_prompt` 128 and 512, with
      no unattributed residue — every term named, including host-device staging and dequantization.
- [ ] The threshold decomposition written down before implementation, with each item's expected
      contribution read off the breakdown, and an escalation recorded here if they did not sum.
- [ ] Prefill activations stay device-resident across a layer's projections, with the materialization
      boundary documented; bytes staged per 512-token window down >= 70% against the step-2 baseline;
      `sgemmLayerInto`'s per-matmul allocate-and-copy removed via the non-allocating batched form,
      whose contract matches the one Tier 10 item 4 adds for CPU.
- [ ] Chunk-sizing defaults reviewed per surface; any surface still pinned at `32` has a measured
      reason, not an inherited one.
- [x] `scripts/performance-tests/check-plan-thresholds.sh` exists, passes against this tree, and fails
      against a deliberately broken copy of a tier file with its threshold block removed — a check that
      cannot fail is not a check. Recorded here as shipped, so later tiers run it rather than re-deriving
      execution rule 7's enforcement.
      *Shipped 2026-09-27: passes against this tree; fails against a scratch copy with four planted
      defects, naming each (see the execution record). Extended beyond this box's wording to catch
      lowercase gate mentions, "no unexplained regression" exit criteria, and already-met milestones.*
- [x] The engine-keepalive subshell leak is gone from all eight sibling scripts, via a shared helper in
      `perf-lib.sh` rather than eight copies of the edit, and a full run of each of the nine scripts
      leaves zero leftover shells and zero leftover pipes. Until this is checked, a `pgrep` check for a
      running sweep is unreliable, which already cost real time once.
      *Checked 2026-09-28: nine siblings, not eight (`smoke-gpu-residency.sh` was missing from the list),
      ten scripts on the helper with `compare-llama-cpp.sh`; `selftest-engine-stdin.sh` passes and fails
      against Tier 01's helper; all ten ran to completion leaving zero new loops, pipes, descriptors and
      engines (table in the execution record). The 24 pre-existing orphans are not removed by this.*
- [ ] Prefill throughput no longer degrades with prompt length: the `n_prompt=512` pp ratio is greater
      than or equal to the `n_prompt=128` pp ratio for every sweep model — **both measured under
      `RAW_PROMPT=1` with the same `--vector` setting**, which no published pair of runs has ever
      been. Step 2 establishes whether the degradation is real before this criterion can be scored.
- [ ] Threshold above met, or the tier is explicitly marked partial-complete with the dominant term
      named and assigned to a successor tier (not silently marked complete).
- [ ] Decode (tg) and vision both verified not regressed, with published numbers.
- [ ] Cross-surface checklist fully resolved.
- [ ] Docs (`docs/howto.md` for any `--prefill-batch` default change, `docs/performance.md`,
      `docs/agent-arch.txt`) updated, Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
