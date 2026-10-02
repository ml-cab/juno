# Tier 01B: Prefill throughput

Status: in progress — implementation steps 0 to 3 complete (2026-09-30; step 3 is item 0, pinned gate met); plan amended 2026-09-30 after the plan review (scope items 6 to 10, step 3a, the `juno.DeviceCompute` span); step 3a complete 2026-10-01: items 1a-ii, 7, 8 and 9 (pinned gate and CPU reference taken by the owner); item 10 (CI) removed as an exit criterion by the owner; step 4a (spans for the whole prefill window, added 2026-10-01 by owner decision) implemented and tested, its pinned A/B gate met 2026-10-02 (owner run); steps 4 (the per-term prefill breakdown, published unpinned, taken ahead of the 4a gate by owner decision) and 5 (the threshold decomposition) complete 2026-10-01, with the third milestone row (512 over 128) moved to Tier 02 by owner decision because no item here moves it; step 6 started 2026-10-02 under the owner's "one region, one change" decision: the prefill-window device region (item 6 with item 2's operand residency) is implemented and tested, logits bit-identical to the host path, its unpinned breakdown published, and its pinned A/B gate met 2026-10-02 (TinyLlama prefill 3.899x, owner run); item 6 is complete; item 2's remainder (the residual kept on the device across layers and the non-allocating `MatVec` form) implemented and tested 2026-10-02, measured unpinned (prefill 1.045x and 1.047x, staged bytes down 97% to 99% on the three region models); item 2's criterion met (owner decision 2026-10-02: its bytes threshold is scored on the three models whose attention runs in the region; Phi-3.5-mini's prefill bytes moved to Tier 02 item 8); this change's pinned no-regression gate met 2026-10-02 (owner run: generation 0.997x and 0.993x, prefill 1.073x and 1.049x); item 3 (the chunk-sizing review per surface) complete 2026-10-02: the `JunoPlayer` facade sizes from free VRAM, every other fixed-32 surface has a measured reason, cluster launchers accept `--prefill-batch`; item 4 recorded in the step 4 record; remaining: step 8, the closing cross-surface matrix and pinned closing sweeps (threshold, decode and vision criteria, docs and CHANGELOG close-out)
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
     *Amended 2026-09-30 (owner decision, see "Before step 3" in the execution record): Phi-3 and Qwen3
     only. Phi-2 and Qwen3-MoE run every matmul on the CPU even on a GPU run, so there is no device
     path to put the kernel on. They announce that at startup through the capability mechanism, and
     their GPU path moves to Tier 08 scope item 6.*
   - Decide and document the ROCm answer. If the kernel can be ported, port it (subject to this
     plan's `NEEDS-AMD-HARDWARE` rule, since there is no AMD device here). If it cannot land this
     tier, `auto` must say so — a startup notice naming the backend and the resulting path, not a
     silent resolution to scalar.
     *Decided 2026-09-30 (owner): announced fallback in this tier; the port is
     [Tier 10](TIER-10-gpu-backend-breadth-cpu-simd.md) scope item 8, targeting the attention kernel
     current when that tier runs.*
   - Once coverage is complete, change the default from `auto` to `on` and keep `auto` as an
     explicit opt-in for anyone who wants per-architecture resolution. Where a path genuinely cannot
     support the kernel, it fails loudly to the documented fallback rather than resolving quietly.
     *Amended 2026-09-30 (owner decision): the default stays `auto`. This bullet assumed `auto`
     resolved per architecture and could fall back silently, so `on` was to be the loud mode. After
     item 0, `auto` already announces every case where the kernel cannot run (CPU-only handler,
     non-CUDA backend), and on CUDA `auto` and `on` execute identically. Flipping would change only
     the label, and would make every CPU launch warn unless an explicit `on` were told apart from a
     default one. The "fails loudly" half of this bullet is what is kept.*
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
   distinguish "absent" from "none". [Tier 01C](TIER-01C-packed-kquant-matmul.md) item 1 (split out of Tier 04C on 2026-09-30) reads its
   `launchDequant`-versus-`gemmHalf`-versus-staging split off these same two spans; it was written
   believing Tier 01 had widened the extractor far enough for that, and Tier 01 widened only the
   `jdk.*` bucket. Building them here covers both tiers, and `metrics` must be in this tier's own
   `mvn test -pl` line (the documented command omits it — see [`README.md`](README.md)'s test
   infrastructure section).

   **1a-ii. Add the span that times the GEMM itself (amended 2026-09-30, plan review).** The two
   spans above time copies and dequantization; nothing times the matmul kernels, so 1b's "no
   unattributed residue" cannot be met: the GEMM would be whatever is left of `juno.MatVec` after
   staging and dequantization are subtracted, which also contains the host FP16 packing of the
   activation window (`packFp16Rows`, 17 to 38 ms per matmul when it was deoptimizing in Tier 01's
   close-out). Add **`juno.DeviceCompute`** — `site` (`gemm_half`, `gemm_fp32`, `mmq_packed`,
   `gqa_attention`, and the item 6 kernels as they land), `phase`, `count`, `timedCount`,
   `computeNanos` — timed with the same `DeviceSpanTimer` stream events, aggregated the same way
   (totals per site and phase, periodic), registered disabled in `juno-perf.jfc` and enabled by
   `juno-perf-spans.jfc`, and extracted by `DeviceSpanBucket` as
   `juno.DeviceCompute.site.<site>.<phase>.total_ms` with every key written on every run. Add the host
   packing as a `site=pack_fp16_host` entry of `juno.DeviceStaging` timed on the host clock (it is
   host work that exists only to stage). Same evidence standard as step 1: `metrics` tests that fail on
   the build without the keys, a `@Tag("gpu")` test that a 32-row FP16 GEMM's compute time is non-zero
   and its output bit-identical with and without a recording, and a same-hour pinned A/B with the
   spans **off**: prefill **>= 0.98x** the build without the new event.

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

   *Items 6 to 10 were added on 2026-09-30 by the plan review. Item 6 is forward-pass work and the
   tier's largest remaining lever; items 7 to 10 are small harness, correctness and repository items
   the review found unowned or stuck. See "2026-09-30 — plan-review amendments" in the execution
   record.*
6. **Keep the prefill window's elementwise work and its KV append on the device.** This tier's own
   records show where prefill time goes, and staging is not most of it. Step 2's spans run puts
   staging plus dequantization at 12% to 15% of prefill on the three Llama-family models, so removing
   both entirely is worth at most about 1.13x to 1.18x, against the 1.6x the non-Phi milestone asks.
   Step 1's record puts `juno.SwiGlu` at about 35 ms per layer on TinyLlama at a 512-token window, about
   770 ms of a 2,157 ms prefill (about 36%). The code explains it: the prefill window's SwiGLU is a
   single-threaded scalar loop over `W x I` elements calling `Math.exp` in double precision
   (`LlamaTransformerHandler`, the window path after the gate and up projections), and both RMS norms
   take the CPU fallback because `rmsNormGpu` is null. The gate and up outputs (the widest tensors in
   the layer) are copied device-to-host only to be multiplied on the host and copied straight back for
   the down projection, which is why device-to-host bytes (898 MB) exceed host-to-device bytes (507 MB)
   on TinyLlama. Separately, `DeviceKvCache.appendToken` copies one K row and one V row per token per
   layer, 22,528 of the 22,902 host-to-device copies in a TinyLlama window.

   Scope, built on item 2's prefill-window region and Tier 01's `ResidentChain`:
   - SwiGLU on the device for the window (a fused `silu(gate) * up` kernel), so gate and up are never
     materialized to host;
   - both RMS norms on the device through `CudaRmsNorm.normalizeResident`, which already runs at batch
     512 (Tier 01's microbenchmark);
   - the residual adds on the device, so the residual stream crosses the boundary once per layer at
     most, and once per window where item 2's boundary permits;
   - RoPE on the device for the window through `CudaRope`, which first needs the split-half pairing Tier
     01 left unbuilt (Qwen2 and Qwen3 use it); until it has it, those models keep the CPU table rotation
     and the startup log says so;
   - the KV mirror appended once per layer per window (`DeviceKvCache` gains a window append taking the
     `W` rows as one contiguous copy), keeping the host KV written first and the written-prefix
     watermark (`c91f879`) intact.

   Phi-3 and Qwen3 take the same path where their handlers share the operation (norm, SwiGLU, residual);
   their own RoPE variants stay on the host and are announced if not moved. Phi-2 and Qwen3-MoE run on
   the CPU (Tier 08 item 6) and are unaffected. The CPU backend's window path is unchanged, so the CPU
   remains the correctness oracle. This item and item 2 meet at the same region: implement them as one
   design (item 2 moves the matmul operands, item 6 moves the operations between them), and measure them
   separately, item 6 first, because the breakdown says it is the larger term.
7. **Matched thread count in the comparison harness, now** (README benchmark-parity precondition 4,
   amended 2026-09-30). `compare-llama-cpp.sh` passes
   `-Djava.util.concurrent.ForkJoinPool.common.parallelism=$((N_THREADS - 1))` to Juno, so the
   common-pool hot path runs the same thread count the reference tool is given with `-t`, records
   `juno_threads` and the property in `host.json` and `INDEX.md`, and drops the stated-mismatch note
   when they match. A `--selftest` case asserts the property is passed and recorded. One CPU sweep is
   re-taken with it as the new CPU reference (a measurement boundary for CPU readings), and the
   README's reference column moves in the same change. Tier 10 item 5 still owns the product
   `--threads` flag.
8. **Phi-3.5 LongRoPE factor selection — carried here as an out-of-tier correctness fix.** Found
   2026-09-28 and filed as [Tier 02](TIER-02-attention-long-context.md) scope item 7, which has not
   started: `Phi3RopeConfig.selectFactors()` always returns the long-context factors, so Phi-3.5
   rarely produces `<|end|>` (P = 0.502 against 0.992 with the short factors) and about half of its
   console replies run on to the token limit. Phi-3.5-mini is a sweep model and the binding model on
   two of this tier's milestones. The Qwen RoPE-pairing defect was fixed out of tier in Tier 01 on the
   same reasoning. The owner picks the policy first (Tier 02 item 7: option (a), short factors unless
   the session is configured above 4096 tokens and fail closed at the crossing; or option (b),
   per-sequence switching with re-rotation, which shares machinery with context shift). Recommendation:
   (a) here, since it is small and complete, with (b) left to Tier 02 if wanted. Applied identically on
   every `Phi3Rope` caller (CPU, GPU, batched prefill, LoRA `ropeExtBackward`). The long-factor gap to
   the reference engine (0.502 against 0.675) stays Tier 02's to explain. Recorded under "Out-of-tier
   changes" with its measurement-boundary reading: rotation angles change, work does not, so throughput
   readings stand and greedy-parity readings on Phi-3.5 do not.
9. **Rule 7's results-side check** (README execution rule 7, check 4). Extend
   `check-plan-thresholds.sh` so a ticked exit criterion that states a threshold must cite a
   `docs/perf-compare/<dir>` that exists, or carry an explicit `**Evidence (not published):**` marker.
   Write the check against a scratch copy with planted defects first (a ticked threshold criterion with
   no citation, one citing a directory that does not exist), show it failing on both, then bring the
   tree's existing ticked criteria into compliance by adding citations or markers. Do not weaken the
   check to make the tree pass.
10. **CI that actually runs.** `.github/workflows/ci.yml` is untracked (README, "Checked 2026-09-30").
    The owner commits it (git writes are the owner's), and this tier records the first green run of both
    jobs with its URL. If the `build-and-test` job fails on the hosted runner (no GPU, no model files),
    that is a finding to fix here: every GPU-, ROCm- and model-gated test must skip rather than fail.
    *Removed as an exit criterion 2026-10-01 (owner decision).* CI is not a gate of this tier or of the plan. `.github/workflows/ci.yml` stays in the working tree, uncommitted, and nothing in the plan relies on it running: rule 7's check, `compare-llama-cpp.sh --selftest` and the unit and stub-IT runs are executed by hand at each step, as they have been throughout.

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
| 7 | Pipeline-parallel cluster | ~~prefill chunks cross the gRPC boundary per shard — confirm the per-chunk fixed cost being reduced here is not simply relocated into serialization~~ *Corrected 2026-10-02 (item 3 record): cluster prefill is not chunked at all; the gRPC clients send one `ForwardPass` per prompt token, so nothing this tier builds reaches a cluster node. Batched cluster prefill is [Tier 09](TIER-09-tensor-parallelism-multi-gpu.md) item 5 (owner decision). This tier's closing matrix records the row as correct but not reached, owned by Tier 09, not as a prefill PASS* |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | training prefills its own microbatches; confirm unaffected, or improved, but not silently changed in numerics. Training's existing `--gpu-attention` exemption (attention stays scalar CPU, REPL warns) is **re-verified and kept** by item 0, not quietly swept into the new default |
| 10 | LoRA playback | playback prefills a LoRA-modified prefix; confirm the delta-add still composes correctly against a wider batched path. Same `--gpu-attention` exemption as row 9 — re-verified, kept, and still warned about |
| 11 | Vision | `VisionEncoder` runs the widest batches in the system (B around 741) and reuses the same `MatVec` primitives — a batched-dispatch change is exactly the kind that regressed vision once before; run `compare-vision.sh` as a required gate, not an optional one. ~~Vision already inherits `--gpu-attention` by delegating to `LlamaTransformerHandler`~~ *Corrected 2026-09-30: the only vision model on disk, moondream2, has a Phi-2 backbone and runs `Phi2TransformerHandler` (CPU matmuls, scalar attention). Item 0 leaves Phi-2 on that path, so vision's text half must stay bit-identical; `compare-vision.sh` verifies it* |
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
3a. **Before the breakdown (added 2026-09-30):** land the `juno.DeviceCompute` span (scope item
   1a-ii) with its A/B gate, and the small items that do not touch the forward pass: item 7 (matched
   thread count, with its CPU re-baseline), item 9 (rule 7 check 4) and item 10 (CI committed and
   green). Item 8 (Phi-3.5 LongRoPE) lands here too once the owner has picked the policy; it changes
   rotation angles, not work, so it does not move the breakdown, but it does move Phi-3.5 greedy
   output and must be in before any divergence or greedy-parity reading this tier takes.
4a. **Spans for the whole prefill window (added 2026-10-01, owner decision; see "before step 4" in the
   execution record).** The per-op spans existed on the LLaMA-family handler only, and the KV write,
   each matmul's copy-out, the embedding and the LM head sat outside every span, so the breakdown's
   `<= 5%` residue bound was unreachable (7.0% on TinyLlama, 44.8% on Phi-3.5-mini). Add
   `juno.WindowStep`, put the per-op events on the Phi-3 and Qwen3 window paths, split `juno.MatVec`
   by phase, tests first, and pass a same-hour pinned A/B (prefill **>= 0.98x** the build without them,
   spans off) before the breakdown is taken on top of them.
4. Produce the per-term prefill breakdown (scope item 1b) and publish it.
   Decide which of scope items 2, 3, 4 and 6 the breakdown actually justifies, and record the decision
   here — item 0 may well have moved which term dominates, which is the point of sequencing it here.
   *The expected ranking going in was that host-device staging (item 2) dominates. Step 2's spans run
   already contradicts that (staging plus dequantization 12% to 15% of prefill), and step 1's record
   puts host SwiGLU alone at about 36% on TinyLlama.* The expected ranking is now: host elementwise work
   (item 6) first, staging and the per-token KV copies (items 2 and 6) second, and the GEMM compute
   itself — read from `juno.DeviceCompute`, not inferred — third on the small models and larger on
   mistral-7b, where the roofline in the README's "Post-plan anchor" puts the FP32-compute GEMM floor
   at about 0.8 to 1.1 s of a 7.5 s window. If the breakdown says otherwise, follow the breakdown.
   Whatever the GEMM's measured share is, record it per model: it is the number
   [Tier 01C](TIER-01C-packed-kquant-matmul.md)'s throughput threshold is conditioned on.
5. Write down the expected contribution of each remaining item against the threshold, per the
   "Decompose the ask before implementing it" clause below, and escalate here if they do not sum.
   Item 6 is one of the named items, and the GEMM term is attributed to Tier 01C (the next tier) rather
   than to "a tier that does not exist yet".
6. Implement in the order the breakdown ranks, largest term first. Items 2 and 6 share the
   prefill-window region; land item 6's operations and item 2's operand residency as separately
   measured changes on that one region, not as two regions.
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
- **Item 6 tests (added 2026-09-30)**: the device SwiGLU, norm and residual kernels against the CPU
  window path within float tolerance at `W` = 1, 8, 9, 32, 512 and `I` from every sweep model; the
  split-half RoPE mode bit-identical to the CPU table rotation (the adjacent mode already is); the
  window KV append leaves the host KV and the device mirror equal and the watermark at the window end,
  including when a device allocation fails mid-window (host written first, mirror retired, CPU
  continues); a `@Tag("gpu")` handler test that gate and up produce no device-to-host copy on the
  window path (read off `juno.DeviceStaging`); device memory returns to its starting level across
  repeated windows.
- **`juno.DeviceCompute` tests (item 1a-ii)**: as described in that item; must fail on the build
  without the keys.
- **Item 7 (thread parity)**: a `compare-llama-cpp.sh --selftest` case that the common-pool property
  is passed with `N_THREADS - 1` and recorded, and a live check that a CPU run's
  `jdk.ExecutionSample` threads on the matmul methods number `N_THREADS`, not
  `availableProcessors()`.
- **Item 8 (Phi-3.5 LongRoPE)**: Tier 02's end-of-turn live test, moved here — teacher-force the 19 ids
  listed in Tier 02 and assert P(32007) **>= 0.95** at the last position (0.502 before the fix); a unit
  test of the chosen policy's factor selection at and across 4096 tokens.
- **Item 9 (rule 7 check 4)**: the planted-defect scratch copy described in that item, failing before
  and passing after.
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

  **Threshold, item 6 on its own** (added 2026-09-30). At `n_prompt=512` on tinyllama, qwen2.5-3b
  and mistral-7b, from a `--device-spans` run:
  - host elementwise time — `juno.SwiGlu` + `juno.RmsNorm` + `juno.Rope` + `juno.ResidualAdd` within
    the prefill window — **<= 10%** of `juno.ForwardPass` prefill time on each model (step 1's record
    puts SwiGLU alone at about 36% on TinyLlama);
  - KV mirror host-to-device copies per window
    (`juno.DeviceStaging.H2D.site.memcpy_k_row_h2d.prefill.count` plus the V-row site, or their window
    replacement) **<= 2 x the layer count** — one K and one V copy per layer, against 22,528 today on
    TinyLlama;
  - Juno prefill t/s **>= 1.25x** the pre-item-6 build on tinyllama (the model where the measured
    elementwise share is largest), from a same-hour pinned A/B, Juno absolute t/s (README,
    "No-regression gates tighter than the floor"); on qwen2.5-3b and mistral-7b the gain is recorded,
    not gated, until the breakdown gives their elementwise shares;
  - greedy output: identical to the pre-item-6 build over 64 tokens on tinyllama and mistral-7b, or
    characterised the way item 0 characterised the FP16 KV mirror (first divergent step per prompt over
    six prompts), never earlier than the item-0 baseline's earliest divergence.

  **Threshold, the GEMM compute span (item 1a-ii).** Same-hour pinned A/B with the spans off: prefill
  **>= 0.98x** the build without `juno.DeviceCompute`; with `--device-spans`, the per-term breakdown's
  unattributed residue (`juno.ForwardPass` prefill minus every named term) **<= 5%** of prefill on
  each sweep model.

  **Threshold, the tier overall** (the README's Tier 01B milestone rows, restated 2026-09-27).
  At `n_prompt=512`, GPU:
  - pp ratio **>= 0.10x** llama.cpp on every sweep model except Phi-3.5-mini (tinyllama, qwen2.5-3b,
    mistral-7b; binding reference 0.062x on tinyllama, read at `n_prompt=128`);
  - pp ratio **>= 0.08x** on Phi-3.5-mini (reference 0.040x at `n_prompt=128`) — the model whose
    handler item 0 newly gives the GPU attention kernel, and the binding constraint on the program's
    end-of-plan pp target;
  - ~~pp must not fall with prompt length: the 512-token ratio **>= 1.00x** the 128-token ratio for
    every sweep model.~~ *Moved to [Tier 02](TIER-02-attention-long-context.md) on 2026-10-01 (owner
    decision, step 5 record): attention is what falls with prompt length, and no item here moves it.*

  All three are measured under the parity-corrected harness (`RAW_PROMPT=1`, warmup, median of three,
  `--pin-clocks`), and **the first two are re-read against step 2's 512-token re-baseline before
  implementation starts**: their references were taken at 128 tokens because no parity-corrected 512
  reading exists. If the 512 readings sit materially below the 128 ones, restate both rows against the
  512 figures, here and in the README table, keeping the same multiples (about 1.6x on the binding
  non-Phi model and 2.0x on Phi-3.5-mini) rather than carrying a target set against the wrong
  denominator.

  *Re-read 2026-09-30 (step 2, owner decision): neither row is restated.* The non-Phi 512 reading
  (0.064x, qwen2.5-3b) is not materially below its 128 reference (0.062x), so 0.10x stands. Phi-3.5-mini's
  (0.011x) is 3.6x below, but that is a collapse caused by the scalar attention item 0 removes, not the
  modest dip this rule was written for. Restating at 2.0x would give 0.022x, which item 0 alone would meet
  by restoring the 128-token behaviour (about 0.037x). **0.08x is kept**, now a 7.2x move from 0.011x.
  **The restatement rule applies to a modest dip, not to a collapse that an item scoped in this tier
  removes.**

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
  to [Tier 01C](TIER-01C-packed-kquant-matmul.md) (the GEMM operand and kernel, next in the running
  order), to [Tier 04C](TIER-04C-packed-weight-matmul.md) (formats beyond Q4_K/Q5_K/Q6_K), to Tier 02
  (attention at long context), or to a tier that does not exist yet.

  **Contingency, in the same spirit as Tier 01's.** If the breakdown in step 4 shows prefill time is
  dominated by a term this tier cannot move without work owned by a later tier (for example: residual
  attention at long context, which is Tier 02; the GEMM kernel itself, which is Tier 01C and runs
  next; or per-format kernels, which is Tier 04), do not
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

*Amended 2026-09-30: the files are all present, but Phi-2 and Qwen3-MoE turned out to have no GPU path
to measure. Their handlers never use the GPU backend, so "partial offload" offloads nothing. By the
owner's decision, item 0 measures Phi-3 and Qwen3, and Phi-2 and Qwen3-MoE announce their CPU path (see
"Before step 3" in the execution record).*

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
prefill repetition), 1 (the copy and dequantization spans, opt-in, gate met) and 2 (the pinned
re-baseline at 128 and 512, with a spans run for staged bytes) are complete. Step 3 (item 0) is
implemented and tested for Phi-3 and Qwen3 under an owner-amended scope. The default and the ROCm
answer are decided; its pinned gate is met (see its record). Step 3a (2026-10-01) landed the
`juno.DeviceCompute` span (pinned gate met), harness thread parity with a pinned CPU reference, the
Phi-3.5 LongRoPE fix and rule 7 check 4; item 10 (CI) was removed as an exit criterion by the owner. Step 4a (spans for the whole window) is
complete, its pinned gate met (0.993x and 1.005x). Steps 4 (the breakdown, residue 0.4% to 2.2%) and 5 (the threshold
decomposition) are complete; the decomposition's third row moved to Tier 02 (owner decision). Step 6's first
change (the prefill-window device region, items 6 and 2 as one region by owner decision) is implemented,
tested, measured, and its pinned gate met (TinyLlama prefill 3.899x); item 6 is complete. Step 6's second
change (item 2's remainder) is implemented, tested and measured unpinned, and item 2's criterion is met under
the owner's decision on Phi-3.5-mini's bytes (moved to Tier 02 item 8); its pinned gate is met (owner run).
Next: item 3 (chunk sizing per surface), then step 8's closing matrix.
The two milestone decisions step 2 raised were taken by the owner on 2026-09-30 (recorded under step 2). This section also records the
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
*Decided 2026-09-29 (owner): adopted.* See "Before step 2" below.

### 2026-09-29 — before step 2: CI adopted, resolved GPU attention recorded

**CI decision (the README's revisit trigger, due before this step): adopted.** `.github/workflows/ci.yml`
has two GPU-free jobs. `script-checks` runs on every push: `check-plan-thresholds.sh`,
`compare-llama-cpp.sh --selftest` and `selftest-engine-stdin.sh`. `build-and-test` runs on pull requests
and pushes to `main`: `mvn -B clean verify` at the root, which is every unit-test module plus the stub
cluster ITs in `juno-master`. GPU- and ROCm-tagged tests skip themselves without a device. Real-model runs
(`ModelLiveRunnerIT`, the smoke scripts) and every performance gate stay manual on this host. So CI
covers the unit and stub-IT layer and the script checks, and none of the gates this plan scores. **Not
yet verified on GitHub**: the workflow was checked for YAML validity only, and its first run happens on
the first push that carries it. Tier 07's re-examination still stands.

**Harness: the resolved `--gpu-attention` value is recorded, as step 2 requires.** Before this change
the harness recorded only the flag it passed, which is empty for the default lane. Each per-repetition
result now carries `gpu_attention_requested` (`engine default` when no flag was passed) and
`gpu_attention_resolved`, read from the engine log:
- `on`: `LlamaTransformerHandler` wrote its activation line.
- `off-backend-fallback`: the kernel was requested but the backend lacks it.
- `off`: a GPU run with no activation line. This is sound on current code, because the only handler that
  can activate the kernel logs when it does. The Phi-3, Phi-2, Qwen3 and Qwen3-MoE handlers never read the
  option.
- `n/a`: a CPU run.

*Corrected 2026-09-30, after step 2's sweep: the `off` rule was wrong.* The console resets library
logging to `OFF` unless `--verbose`, so the activation line never reaches the engine log, and the first
sweep read `off` on every row, including the three Llama-family models that ran the kernel. The rule is
now evidence-only. A log line is used when present. On a `--device-spans` run the result is `on` when
the kernel's own copy sites (`memcpy_gqa_*`) are in the recording and `off` when they are not.
Otherwise it is `unknown`. Tests first again: 4 new `--selftest` cases; 3 failed on the first version,
and the fourth passed only because that version answered `off` everywhere. The three published runs
were re-derived from their own recordings, with a correction note in each `INDEX.md` (see step 2
below).

The aggregate reads `mixed` when repetitions or lanes disagree. The prefill and generation lanes each
keep their own value under `lanes`, and `INDEX.md` lists the value per row. The `--help` text and the
tuned-lane comment no longer say the default lane runs `--gpu-attention off`. That is the stale label
scope item 0 names, and it is corrected here because step 2's run is the one the exit criterion asks
to state the mislabelling. The engine's own `--help` and `docs/performance.md` corrections stay with
item 0. Tests first: 8 new `--selftest` cases. Seven failed on the unchanged script (the function was
absent, and neither the aggregate nor the lane merge carried the field); all pass now, as do the
existing cases.

**Verification, on the final tree (2026-09-30).** `mvn test` on the eleven unit-test modules: 1,832 tests,
0 failures, 49 skipped (existing assumptions), 26:22. `mvn test -pl node -Dgroups=gpu`: 145 tests, 0
failures, 7 skipped. `compare-llama-cpp.sh --selftest` and `check-plan-thresholds.sh` pass. Not run:
`mvn verify -pl juno-master` (stub-mode cluster ITs; step 1 changes no gRPC, scheduling or handler
contract) and the real-model live runner; both belong to the tier's closing matrix (implementation step
8). No `compare-lora.sh` or `compare-vision.sh` run: with the events off every copy site is the plain
call, and the pinned A/B above is the step's own gate.

### 2026-09-30 — implementation step 2: the re-baseline

Scope: measurement only, on the build of `ffd0ca7` (jar `juno-player-0.1.2-shaded.jar`, sha256
`dc94bd797c8a3219`; the tree was dirty only in harness scripts and plan docs). The owner ran the three
sweeps from a local terminal, because prompt-free sudo was not available in the agent shell. No
engine code changed. Every row of all three runs is scorable (repetitions within 15%), every Juno
prefill is 128/128 or 512/512 tokens, and no repetition was withheld by the span check.

| Run | `n_prompt` | Flags | Role |
|---|---|---|---|
| [`20260930T135225Z`](../perf-compare/20260930T135225Z/INDEX.md) | 128 | `--gpu --pin-clocks` | GPU reference at 128 |
| [`20260930T141026Z`](../perf-compare/20260930T141026Z/INDEX.md) | 512 | `--gpu --pin-clocks` | GPU reference at 512 |
| [`20260930T144658Z`](../perf-compare/20260930T144658Z/INDEX.md) | 512 | `--gpu --pin-clocks --device-spans` | staged bytes and dequantization (item 2's before-measurement) |

**Resolved GPU attention, per lane:** `on` for tinyllama, qwen2.5-3b and mistral-7b (default and tuned
lanes); `off` for Phi-3.5-mini on both lanes. This is read from the spans run's copy sites. The two ratio
runs record `unknown`, since they carry no spans, but they are the same build, flags and models, so the
same resolution applies to them.

**Default lanes, median of three (min to max); ratios against the reference tool's own reading in the same run:**

| Model | pp t/s @128 | pp ratio @128 | pp t/s @512 | pp ratio @512 | 512/128 ratio | tg t/s | tg ratio |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | 239.08 (238.9 to 244.7) | 0.0676x | 242.35 (242.2 to 244.6) | 0.0662x | 0.979 | 68.76 | 0.377x |
| qwen2.5-3b | 95.24 (93.6 to 96.1) | 0.0668x | 91.74 (90.3 to 93.3) | 0.0638x | 0.955 | 29.23 | 0.423x |
| Phi-3.5-mini | 43.07 (43.0 to 43.2) | 0.0372x | **13.06** (13.04 to 13.06) | **0.0112x** | **0.300** | 22.80 | 0.394x |
| mistral-7b | 64.28 (63.6 to 64.3) | 0.1014x | 68.35 (67.8 to 68.5) | 0.0984x | 0.970 | 21.46 | 0.610x |

(tg from the 128 run. The 512 run's tg is within 6% of it on every model, and generation does not depend
on `n_prompt` in this harness.) Tuned lanes are within each row's spread of the default lane.

What this establishes:
- **Prefill really does degrade with prompt length, and only on the model without the GPU attention
  kernel.** Phi-3.5-mini's own throughput falls 3.3x from 128 to 512 tokens, with tight spreads. That is
  the quadratic scalar attention the earlier runs could only suggest. On the three Llama-family models
  Juno's own prefill t/s is flat (tinyllama +1.4%, mistral +6.3%, qwen2.5 -3.7%). Their 512/128 ratios
  of 0.955 to 0.979 come from the reference tool reading faster at 512 (tinyllama +3.5%, mistral +9.6%)
  and, for qwen2.5, from Juno's own -3.7% (the reference moved +0.9%). All sit inside the 15% noise
  floor.
- **The binding reference moved.** At 512 the binding non-Phi model is qwen2.5-3b at 0.064x (reference
  was tinyllama 0.062x at 128), which is not materially different. Phi-3.5-mini is at 0.011x, against its
  0.040x reference at 128: 3.6x below it, so it is materially below.
- **Spans cost less than step 1 estimated.** At pinned clocks the spans run reads 0.980x to 1.013x of the
  unspanned 512 run's Juno prefill, within each row's spread (step 1's 6% was unpinned).

**Staged bytes and dequantization per 512-token prefill window** (spans run; bytes identical across all
three repetitions; times are device-event totals and their share of that window's prefill wall time):

| Model | H2D MB | D2H MB | H2D + D2H ms (share) | Device dequant ms (share) | Copies | Prefill ms |
|---|---|---|---|---|---|---|
| tinyllama-1.1b | 507 | 898 | 294 (13.6%) | 38 (1.8%) | 22,902 | 2,157 |
| qwen2.5-3b | 1,027 | 2,261 | 564 (10.2%) | 98 (1.8%) | 37,476 | 5,512 |
| Phi-3.5-mini | 569 | 2,076 | 249 (0.6%) | 119 (0.3%) | 256 | 39,474 |
| mistral-7b | 1,608 | 3,081 | 695 (9.2%) | 200 (2.7%) | 33,312 | 7,522 |

These byte counts are item 2's before-measurement for its `>= 70%` threshold. The Llama-family copy
count is dominated by the KV mirror's per-position row copies (`memcpy_k_row_h2d`/`memcpy_v_row_h2d`,
one per layer per token), which item 2's description does not mention. **Early warning for step 5,
not a decision:** staging plus dequantization is 12% to 15% of prefill wall time on the three
Llama-family models. So removing it entirely is worth at most about 1.13x to 1.18x on those models,
against the 1.6x the non-Phi milestone asks. Step 4's breakdown has to find the rest, or step 5
escalates.

**Three harness faults found and corrected while reading the runs** (all in `compare-llama-cpp.sh`):
1. The resolved-attention rule; see the correction under "Before step 2" above.
2. `jar sha256` was recorded as `missing` in every run since the plan-review pass: the hash function
   and `host.json`'s `juno_jar` named `juno-player/target/juno-player.jar`, which the build does not
   produce, while the engine ran the shaded jar `find_juno_jar` selects. Both now use `find_juno_jar`.
   The three runs' `host.json` and `INDEX.md` carry the hash of the jar they launched. It was unchanged on
   disk from before the first run, so this is the same file, and the correction is noted in each.
3. The mislabelled-lanes statement the exit criterion asks for is in `20260930T135225Z/INDEX.md`: from
   `8776a3f` (2026-09-17, where the engine default moved from `off` to `auto`) every GPU run's default
   lane ran the kernel on the three Llama-family models while the harness called it `off`. The data
   recorded the flag as empty, so no published figure changes.

**Reference and milestones.** Per the README rule, the program-target table gains this run as its
current-reference column in the same change, and the two 2026-09-27 reference runs carry
superseded banners. The Tier 01B milestone rows now cite the 512 readings (0.064x, 0.011x, and 0.30 for
the 512/128 ratio, which was unmeasured before). Their thresholds are unchanged, by the owner's decision below.

**Decisions taken by the owner (raised and decided 2026-09-30):**
- **Phi-3.5-mini milestone.** The plan says that if the 512 reading is materially below the 128 one,
  the row is restated against the 512 figure at the same 2.0x multiple. That gives 0.022x, well under
  today's 0.08x, because the degradation is 3.6x, not the modest dip the rule was written for. The cause
  is exactly what item 0 removes. Recommendation: keep 0.08x, now a 7.2x move, because a target lowered
  to 0.022x would be met by item 0 restoring the 128-token behaviour, with nothing else required. The
  non-Phi row stays at 0.10x either way (0.064x is not materially below 0.062x). **Decided: keep 0.08x.**
  The restatement rule under "Threshold, the tier overall" now carries the collapse carve-out. If 0.08x
  proves unreachable in this tier (Phi-3.5-mini's staging is 0.6% of its prefill, so item 2 barely
  reaches it), the tier's contingency applies: partial-complete, with the dominant term named.
- **The retired Tier 04 milestone** (Phi-3.5-mini GPU tg >= 0.40x, "met on arrival" at 0.416x) reads
  0.394x at pinned clocks, so it no longer meets its threshold. The gap is inside the noise floor, and
  pinning changed both engines' clocks. **Decided: keep retired.** Its reference cell keeps the 0.416x
  it was retired on, and the README records the pinned reading beside it. Reactivating it would gate on
  a 1.5% gap, a tenth of the noise floor, in a tier whose work cannot move the number.

**Verification.** `compare-llama-cpp.sh --selftest` (all cases, including the 12 new ones) and `bash -n`
pass; `check-plan-thresholds.sh` passes after the milestone edits. No Java changed, so no `mvn` run.

### 2026-09-30 — before step 3: plan-versus-code drift in item 0, raised with the owner

Read-only pass over the claims item 0 depends on, against HEAD `ffd0ca7` plus the uncommitted step 2
tree. No code changed. **Step 3 is on hold until the owner decides how item 0 covers Phi-2 and Qwen3-MoE.**

Confirmed as the plan states:
- `GpuAttentionOptions.preferGpuAttention()` returns `CudaAvailability.isAvailable()` for `AUTO`; the
  class has no architecture awareness. Only `LlamaTransformerHandler` and `LoraTrainableHandler` (plus
  `ConsoleMain`'s flag plumbing) reference it. Its class javadoc ("enables on CUDA for supported
  architectures") overstates this; corrected with item 0.
- `CudaGqaAttention.tryCreate` returns `null` for any backend other than `cuda`, so ROCm never runs the
  kernel.
- All four uncovered handlers own their own `SessionKvTensor` maps and a private `gqaInto` with the same
  math as `GqaMath.attend` (scale `1/sqrt(headDim)`, grouped-query head mapping, causal by `seqLen`, no
  sliding window, no soft-cap). The kernel computes that math for any head dimension, so no kernel change
  is needed for Phi-2's head dimension of 80 or Qwen3's decoupled `qDim`.
- The LoRA training and `--lora-play` exemption is still in place and still warns
  (`LoraTrainableHandler.warnIfGpuAttentionIgnored`).
- All four item-0 model files are on disk (`phi-2.Q4_K_M.gguf`, `Phi-3.5-mini-instruct-Q4_K_M.gguf`,
  `Qwen3-1.7B-Q4_K_M.gguf`, `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`).

**Not as the plan states. These change item 0's scope:**
1. **Phi-2 and Qwen3-MoE never use the GPU.** Both handlers take a `MatVec backend` and store it, but no
   matmul goes through it. `Phi2TransformerHandler` runs every projection through its static
   `sgemmQuantBatch`, which calls `LlamaTransformerHandler`'s host weight-stationary kernels.
   `Qwen3MoeTransformerHandler` runs every projection, the router and the experts through
   `LlamaTransformerHandler.matVec` on the host. So on a `--gpu` run those two models run entirely on the
   CPU. No weights are on the device, and `--gpu-layers` has no effect on them. Item 0 assumed four
   GPU-weight handlers that lack only the attention kernel. That is true of Phi-3 and Qwen3, not of these two.
2. **"Qwen3-MoE measured at partial offload, with the offloaded layer count recorded" cannot be done as
   written.** The handler offloads zero layers whatever `--gpu-layers` says. A "partial offload" figure
   would be a CPU figure under a GPU label.
3. **Vision is not unaffected by item 0.** Cross-surface row 11 says vision "inherits `--gpu-attention` by
   delegating to `LlamaTransformerHandler`". `LlavaHandlerFactory` loads the text handler through
   `ForwardPassHandlerLoader`, and the only vision model on disk (`moondream2-q5_k.llamafile`) has a Phi-2
   backbone. So the vision gate runs `Phi2TransformerHandler`, CPU matmuls and scalar attention included.
   Whatever item 0 does for Phi-2, it does for the vision gate's text half.
4. `docs/howto.md`'s `--gpu-attention` row says `auto` "correctly resolves to off" on the four
   architectures. `auto` resolves to on (CUDA present); those handlers just never read it. That is the
   silent no-op item 0 exists to remove, so the correction belongs with item 0.
5. `Qwen3MoeTransformerHandler` has no `forwardBatch` override (Phi-2, Phi-3 and Qwen3 do), so its
   prefill does not run as a batched window. Item 0's prefill-gain threshold for Qwen3-MoE would be
   measured on the token-at-a-time path.

Phi-3 and Qwen3 match the plan: device FP16 or packed Q4_K weights through the backend, and the same
three attention call sites as the Llama handler (prefill window, single-token decode, multi-decode).
Item 0 can proceed for those two as written.

**Owner decision (2026-09-30): the kernel goes into Phi-3 and Qwen3; Phi-2 and Qwen3-MoE announce.**
Item 0 integrates the GPU attention kernel into `Phi3TransformerHandler` and `Qwen3TransformerHandler`
and measures both. `Phi2TransformerHandler` and `Qwen3MoeTransformerHandler` do not get the kernel in
this tier. On a GPU run each prints a startup notice, once, saying the handler runs its matmuls and
attention on the CPU and that `--gpu-attention` and `--gpu-layers` do not apply to it. The notice goes
through the same capability mechanism the covered handlers use, so neither resolves to scalar silently.
A GPU weight path for those two handlers, and then the kernel, is handed to
[Tier 08](TIER-08-model-architecture-breadth.md) (scope item 6 there), because that tier already owns the
MoE handler family. The same goes for a batched `forwardBatch` for Qwen3-MoE. Scope item 0, the "Models
needed" table, cross-surface row 11 and the first three exit criteria are amended to match. Vision keeps
its current behaviour: the moondream2 text half stays on the CPU handler, and item 0 must leave it
bit-identical.

### 2026-09-30 — implementation step 3 (item 0): the kernel in Phi-3 and Qwen3, and the capability mechanism

Scope as amended by the owner decision above. Code, tests and docs are done. The item's gate reading
(pinned clocks) was taken by the owner and meets the item-0 threshold; the owner decisions are
recorded at the end. **Step 3 is complete.**

**What shipped.**
- `node`: `ForwardPassHandler.gpuAttentionActive()` (default `false`; overridden by
  `LlamaTransformerHandler`, `Phi3TransformerHandler`, `Qwen3TransformerHandler`) is the capability
  report. `GpuAttentionSupport` (new) names the handlers that run the kernel. It builds the console
  notice and the once-per-process log lines for a CPU-only handler on a GPU backend (Phi-2,
  Qwen3-MoE), a backend without the kernel (ROCm; `on` or `auto`), and an explicit `on` with the CPU
  backend. `GpuAttentionMirror` (new) is the kernel plus per-request `DeviceKvCache` mirrors for the
  Phi-3 and Qwen3 handlers at all three call sites, with the LLaMA-family contract. That contract:
  host KV written first and always, the kernel reads only through the watermark, and a device OOM
  retires mirrors in place and continues on the CPU, logged once. The LLaMA-family handler's own
  inline path is unchanged. `ForwardPassHandlerLoader` logs the CPU-only notice.
  `LoraTrainingHandlerFactory.noteGpuAttentionIgnored` raises the LoRA exemption notice for every
  LoRA architecture. It used to be raised only by the LLaMA-family LoRA handler: with Phi-3 and Qwen3
  now running the kernel outside LoRA, LoRA on them would otherwise have dropped the flag silently.
  The same was already true, before this change, of Qwen2 LoRA.
- `juno-player`: `ConsoleMain` prints the GPU attention notice beside the residency notice; `--help`
  names the covered architectures and the divergence.
- ROCm answer: **announced fallback this tier**, no port. The kernel is PTX loaded through the CUDA
  driver API, and a HIP port could not be compiled or run here (`NEEDS-AMD-HARDWARE`). A GPU launch on
  ROCm with `on` or `auto` warns that the kernel is CUDA-only. Where a port should live is an owner
  decision (Tier 10 holds the other hardware-gated ROCm work).
- Docs: `docs/performance.md`'s GPU-resident-attention section (default was stated as off; coverage;
  divergence table), its recommended-flags section, `docs/howto.md`, `docs/agent-arch.txt`,
  `compare-prefill-batch.sh`'s stale "default off" label (found in step 0), `CHANGELOG.md` (Session 100).
  One internal tier number in the section edited ("see the Tier 17 follow-on finding") was replaced;
  others elsewhere in `docs/performance.md` were not audited in this step.

**Tests, written first and shown failing for the right reason.**
- `GpuAttentionSupportTest` (`node`, CPU, 7 cases): failed with the class a throwing skeleton (7
  errors); pass.
- `GpuAttentionHandlerParityTest` (`node`, `@Tag("gpu")`, Phi-3.5-mini and Qwen3-1.7B): kernel `on`
  (the default) against `off` on the same CUDA backend at the prefill window, single decode and a
  two-stream multi-decode at different positions; capability reported; device KV freed on evict.
  Failed before the wiring on "the default must activate the kernel on CUDA" for both models, with
  their `off` legs passing; passes. Measured logits relative L2 0.00012 to 0.0099 (Phi-3.5-mini) and
  0.00029 to 0.0149 (Qwen3-1.7B), top-1 equal everywhere; bound 0.025. **Planted fault** (one head's
  output zeroed after every launch): 0.153 and 0.090, failing the bound with top-1 unchanged.
- `LoraGpuAttentionNoticeTest` (`node`, CPU, 2 cases): failed against an empty method (notice absent);
  passes.
- `GpuAttentionDivergenceIT` (`juno-master`, `-Pgpu`, new): greedy divergence, below.

**Greedy-decode divergence** (six real prompts, 64 greedy tokens, `on` against `off`, same CUDA build):

| Model | Identical over 64 | First divergent step on the others | First token |
|---|---:|---|---|
| TinyLlama-1.1B (LLaMA family, kernel already default) | 3 of 6 | 8, 13, 23 | equal on all 6 |
| Phi-3.5-mini | 4 of 6 | 26, 50 | equal on all 6 |
| Qwen3-1.7B | 3 of 6 | 20, 23, 32 | equal on all 6 |

Neither newly covered architecture diverges earlier than the LLaMA-family default already shipping.
Documented in `docs/howto.md`, `docs/performance.md` and `--help`, with `off` kept as the bit-identical
baseline.

**Measurement, pinned (the gate reading).** The owner ran the pinned runs from a local terminal
(the agent shell has no prompt-free sudo), all on one jar (`2564f8f506273981`, the final tree), all
rows scorable, every prefill the full prompt, no repetition withheld.

*Same-build flag A/B*, twelve runs alternating off and on, published as
[`20260930T205554Z-tier01b-item0-ab`](../perf-compare/20260930T205554Z-tier01b-item0-ab/INDEX.md). Juno
t/s, median of three (min to max):

| Model | `n_prompt` | pp off | pp on | pp gain | tg off / on | tg gain |
|---|---|---|---|---|---|---|
| Phi-3.5-mini | 512 | 13.69 (12.96 to 13.75) | 89.71 (89.42 to 89.83) | **6.55x** | 24.08 / 32.03 | 1.33x |
| Qwen3-1.7B | 512 | 28.80 (28.32 to 29.07) | 175.56 (175.46 to 176.05) | **6.10x** | 34.96 / 38.43 | 1.10x |
| Phi-3.5-mini | 128 | 44.63 (44.43 to 44.65) | 98.07 (96.46 to 98.16) | **2.20x** | 24.20 / 31.87 | 1.32x |
| Qwen3-1.7B | 128 | 82.12 (81.94 to 82.41) | 177.72 (173.02 to 177.91) | **2.16x** | 34.87 / 38.63 | 1.11x |

**Item 0's threshold is met**: each newly covered architecture shows a measured prefill gain against
its own pre-item-0 path, far outside the noise floor (every spread under 3%), and decode moves with it,
the plan's evidence that the kernel is actually taken. The gain grows with prompt length because it
removes quadratic scalar attention: with the kernel, prefill is nearly flat from 128 to 512 tokens.
The unpinned run above (`20260930T180709Z-tier01b-item0-unpinned-ab`) is superseded by this one and
agreed with it.

*Item 0's attributable ratio* for the four standard models: the same build's default sweeps,
[`20260930T211637Z`](../perf-compare/20260930T211637Z/INDEX.md) (128) and
[`20260930T215547Z`](../perf-compare/20260930T215547Z/INDEX.md) (512), pinned, read against the step 2
reference (`20260930T135225Z`, `20260930T141026Z`):

| Model | pp ratio @128, step 2 → item 0 | pp ratio @512, step 2 → item 0 | 512/128 | tg ratio |
|---|---|---|---|---|
| tinyllama-1.1b | 0.0676x → 0.0658x | 0.0662x → 0.0672x | 1.02 | 0.358x |
| qwen2.5-3b | 0.0668x → 0.0666x | 0.0638x → 0.0607x | 0.91 | 0.424x |
| Phi-3.5-mini | 0.0372x → **0.0818x** | 0.0112x → **0.0749x** | 0.30 → **0.92** | 0.394x → **0.534x** |
| mistral-7b | 0.1014x → 0.1040x | 0.0984x → 0.0999x | 0.96 | 0.620x |

Only Phi-3.5-mini moves, as expected: the LLaMA-family handler is untouched, and the other three sit
within noise of step 2 (qwen2.5-3b's 512 ratio fell 4.9% because the reference tool read 7.8% faster
while Juno read 3.0% faster).

Against the milestones (read, not gated: ratios are reported, never gated below the noise floor):
- Phi-3.5-mini pp @512 **>= 0.08x: not met, 0.0749x** (from 0.0112x). Item 0 alone is a 6.7x move of
  the 7.2x asked for; the rest must come from items 2 to 4 or the contingency applies.
- Non-Phi pp @512 >= 0.10x: not met (tinyllama 0.0672x, qwen2.5-3b 0.0607x, mistral-7b 0.0999x);
  item 0 does not touch these handlers.
- 512/128 >= 1.00 on every model: met by tinyllama only (1.02); Phi-3.5-mini recovers from 0.30 to
  0.92.
- Program end-of-plan target GPU tg Phi-3.5-mini >= 0.50x: **met at 0.534x** (from 0.394x), because
  decode attention on Phi-3 now runs on the GPU. Recorded here and not moved into the README's
  reference column: that column moves with the tier's closing sweep, and step 2 stays the baseline
  every gate in this tier is scored against. *(2026-09-30: because this reading met the target, the
  owner raised both GPU tg end-of-plan targets to 0.70x and the GPU pp target to 0.25x; README, "End-of-plan
  targets raised".)*

**Verification, on the final tree (2026-09-30).** `mvn test` on the eleven unit-test modules: 1,843
tests, 0 failures, 0 errors, 49 skipped (existing assumptions), 26:07, GPU-tagged tests included (CUDA
present). The first pass stopped in `node` on `Q4KMmqMicrobenchTest` (`cudaMalloc` out of memory). The
cause was the new parity test leaving each `CudaMatVec`'s scratch allocated: 204 MB after Phi-3.5-mini
and 54 MB after Qwen3, measured, on top of the node JVM's other GPU tests. The test now releases its
backends' scratch, and asserts that free device memory returns to within 16 MB of where it started, so
the new mirror and kernel scratch are shown freed as well. `mvn verify -pl juno-master`: 20 ITs, 0
failures (in-process, unsupported-architecture, three-node pipeline, tensor-parallel).
`GpuAttentionDivergenceIT` (`-Pgpu`): 3 of 3. `compare-llama-cpp.sh --selftest` and
`check-plan-thresholds.sh` pass. **Not run in this step:** `compare-lora.sh` and `compare-vision.sh`.
The change does not touch the LLaMA-family handler's forward path, the LoRA handlers' compute or the
vision encoder, and moondream2's text half runs the unchanged Phi-2 handler, so both are expected
flat. Both belong to the tier's closing matrix (implementation step 8), and are needed there because
the tier's gate lists them. `ModelLiveRunnerIT` and `smoke-tier01b-prefill.sh` also belong to step 8
(the latter does not exist yet).

**Owner decisions, 2026-09-30:** (2) keep `auto`, and restate exit criterion 1 as behaviour;
(3) the ROCm port is Tier 10 scope item 8. Both are recorded at scope item 0 and at the criterion.
(1) The pinned gate run: done by the owner; results above.

**Originally open, for the owner:**
1. **The pinned gate run for item 0** (needs prompt-free sudo, as in step 2). Same-hour flag A/B,
   `--gpu-attention` alternating off, on, off, on, off, on with `--pin-clocks`:
   `scripts/performance-tests/compare-llama-cpp.sh --gpu --pin-clocks --models Phi-3.5-mini,Qwen3-1.7B
   --n-prompt 512 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --gpu-attention
   {off|on}`, then the same at `--n-prompt 128`. Then a published default sweep of the four sweep
   models at 128 and 512 (`--gpu --pin-clocks`, no `--gpu-attention`), which is item 0's attributable
   ratio against the step-2 reference.
2. **The `auto` to `on` default flip** that scope item 0 names. On CUDA the two now behave the same
   everywhere the kernel exists, and both announce everywhere it does not. The flip would change only
   the label, and it would make every CPU launch print the "`on` has no effect" notice unless explicit
   and default `on` were told apart. Recommendation: keep `auto` and restate the criterion as "the
   default runs the kernel on every covered architecture". Owner decision.
3. **Where a ROCm port of the kernel lives** (recommendation: Tier 10, beside its other hardware-gated
   ROCm work).

### 2026-09-30 — plan-review amendments

Plan text only; no code changed. A plan review taken after step 3 found that this tier's own
measurements contradict the ranking its remaining items were written around, and that several small
items were unowned. The owner approved the following:

| Change | Where | Why |
|---|---|---|
| Scope item 6: prefill-window SwiGLU, norms, residual adds and RoPE on the device; KV mirror appended once per layer per window | Scope, steps 4 to 6, Threshold, tests, exit criteria | Staging plus dequantization is 12% to 15% of prefill (step 2), while host SwiGLU alone is about 36% on TinyLlama (step 1's record). Items 2 to 4 could not reach the non-Phi milestone, and nothing in the plan owned the largest term |
| `juno.DeviceCompute` span (item 1a-ii), new step 3a | Scope item 1, steps | No event timed the GEMM, so the breakdown's "no unattributed residue" could not be met |
| Tier 01C split out of Tier 04C and placed next | README index, new [`TIER-01C`](TIER-01C-packed-kquant-matmul.md), Tier 04C | The GEMM term is the successor this tier's contingency names; it now runs next instead of after Tiers 02 to 04B |
| Scope item 7: harness thread parity via the common-pool property | README precondition 4 | Precondition 4 waited for Tier 10 although the hot path already honours the JVM property |
| Scope item 8: Phi-3.5 LongRoPE fix carried here, out of tier | Tier 02 item 7 | A correctness defect on the binding sweep model, in every reply, parked in a tier that has not started |
| Scope item 9: rule 7 check 4 | README rule 7 | The check proved a threshold was written down, not that a ticked box was scored |
| Scope item 10: CI committed and green | README CI note | The workflow is untracked and has never run |

Not changed: items 0 to 5, the milestone thresholds, the reference column.

### 2026-10-01 — implementation step 3a: `juno.DeviceCompute`, thread parity, Phi-3.5 LongRoPE, rule 7 check 4

Scope: step 3a as amended by the plan review (scope items 1a-ii, 7, 8, 9 and 10). Run on HEAD
`807bfea` plus the uncommitted plan-review tree. **Items 1a-ii, 7, 8 and 9 are complete**, the first two
with pinned measurements the owner ran from a local terminal (recorded at the end of this section).
**Item 10 (CI) was removed as an exit criterion by the owner on 2026-10-01**; see scope item 10.

**Plan-versus-code check before starting.** The prompt that started this session expected Tier 00;
Tier 00 (11 of 11) and Tier 01 (14 of 14) are ticked, and Tier 00's findings are still fixed in code
(`generateBatch` gated on `sessionId`, loader fails closed for unverified architectures, no `Tier NN`
or `com.hazelcast` in any `src/main`, `CLAUDE.md` lists `vision` and `metrics`). Step 3a's own
preconditions held: no `DeviceCompute` anywhere, the step-1 span classes present, `packFp16Rows`
present, no common-pool property in the harness, `check-plan-thresholds.sh` without check 4,
`.github/` untracked. Two drifts, both raised with the owner and decided:
- **Item 8: Juno has no configured session context.** Option (a) reads "short factors unless the
  session is configured above 4096 tokens", but nothing configures a context size: the KV cache grows
  on demand to `DenseKvTensor.MAX_SEQ_LEN` (32768) and no flag, request field or facade setting sets
  one. **Owner decision (2026-10-01): option (a) as a fixed cap.** A Phi-3 file carrying both factor
  sets uses the short set, and a position at or beyond `original_context_length` fails closed. A
  `--ctx-size` flag across every surface was the alternative, rejected as too large for a correctness
  fix.
- **Item 7: the default thread count already matched.** At the harness default `--threads` (`nproc`,
  12 here) the property is 11, which is the JVM's own default, and the calling thread joins as the
  twelfth. So every default-thread run already ran Juno's kernels on 12 threads, matching `-t 12`; the
  old INDEX note ("effective parallelism of 11") left out the calling thread. The change is a boundary
  only for runs with an explicit `--threads`. Recorded in the README at precondition 4.

**Item 9 (rule 7 check 4) — done.** `check-plan-thresholds.sh` reads each ticked criterion with its
indented notes; one that contains `Threshold`, `perf gate`, `>=`, `<=`, `≥` or `≤` must cite a
`docs/perf-compare/<dir>` or `../perf-compare/<dir>` that exists (new `--perf-compare DIR` to resolve
against another tree), or carry `**Evidence (not published):**`. Written against a scratch copy first:
a planted tier file with a ticked threshold criterion citing nothing and one citing a directory that
does not exist failed with exactly those two (`FAIL TIER-99-planted.md:5 ... cites no
docs/perf-compare/<dir>`, `FAIL ...:7 ... 20990101T000000Z-missing, which does not exist`), while four
controls passed (a real citation, the marker, a ticked criterion with no threshold, an unticked one).
On this tree it passes unchanged: only one ticked criterion in the tree states a threshold by those
patterns (Tier 01's RMSNorm/RoPE criterion), and it cites two existing directories. No criterion
needed editing, and the check was not weakened.

**Item 8 (Phi-3.5 LongRoPE) — done.** `Phi3RopeConfig.selectFactors()` returns the short factors
whenever the file has them; `requirePosition(pos)` throws `IllegalStateException` naming the limit at
`pos >= original_context_length` when the file also has long factors and its trained context exceeds
the original (Phi-3.5-mini: 4096). `Phi3Rope.buildCache` calls it, and every rotation goes through
there: the CPU and GPU handlers' three call sites, the batched prefill, LoRA forward and
`ropeExtBackward`. No other code reads the factor tensors. A file with only long factors, or trained
at its original context, is unchanged. Tests first, against a no-op `requirePosition` stub:
`Phi3RopeFactorPolicyTest` (5 cases, CPU) failed 2 (long factors selected; no throw at 4096), and
`Phi3EndOfTurnLiveTest` (the 19 teacher-forced ids, CPU, real file) read **P(32007) = 0.5016**
against `>= 0.95`. After the fix: 5 of 5, and **0.9924** (the recorded short-factor reading was 0.992;
the reference engine reads 0.996). All 29 Phi-3 tests in `node` pass, including
`Phi3GreedyDecodeIntegrationTest`'s reference continuation. Recorded under "Out-of-tier changes"
below; Tier 02 item 7 reduced to option (b) and the long-factor gap.

Because the change moves Phi-3.5 greedy output, step 3's divergence characterisation was re-taken on
the final tree (`GpuAttentionDivergenceIT`, `-Pgpu`, 3 of 3 pass, six prompts, 64 greedy tokens, kernel
`on` against `off`): **Phi-3.5-mini 4 of 6 identical, first divergence at steps 37 and 40** (was 26 and
50 on the long factors); TinyLlama and Qwen3-1.7B unchanged (3 of 6, earliest 8; 3 of 6, earliest 20).
Phi-3.5 still diverges no earlier than the LLaMA-family default.

**Item 1a-ii (`juno.DeviceCompute`) — implemented and tested; the pinned A/B gate is owed.**
- `DeviceComputeEvent` (`site`, `phase`, `count`, `timedCount`, `computeNanos`), periodic through
  `DeviceSpanTally` like the other two. `DeviceSpanTimer.compute` brackets `CudaMatVec`'s asynchronous
  kernels with stream events: `gemm_half` (the tiled FP16 GEMM on FP16 weights and on K-quant weights
  after `launchDequant`), `gemv_half_batched` (2 to 8 rows) and `mmq_packed` (decode GEMV, counted
  untimed). `DeviceComputeClock` (new) times the two default-stream kernels that sit between
  synchronous copies, on the host between two drains of the default stream: `gemm_fp32`
  (`GpuBlasOps.forward`) and `gqa_attention` (`CudaGqaAttention`, which also covers the Phi-3 and Qwen3
  mirrors). The host FP16 packing is `juno.DeviceStaging` site `pack_fp16_host` under a new direction
  `HOST`, so it stays out of the H2D/D2H bytes item 2's threshold is scored on. ROCm has no batched GEMM
  and so no compute site.
- `metrics`: `DeviceSpanBucket` writes `juno.DeviceCompute`, `.<phase>` and `.site.<site>.<phase>`
  (`count`, `timed_count`, `total_ms`) for the five known sites on every run, and `HOST` beside the
  three bus directions. `JfrMetricsExtractorDeviceComputeTest` (5 cases) failed 4 on the unchanged
  extractor with the keys absent; the fifth (the HOST case) passed already because the bucket creates
  unseen directions on demand, and its always-written zero key is asserted by the first case. All pass.
- `DeviceComputeSpansTest` (`node`, `@Tag("gpu")`, 7 cases, GTX 1080): each site's event with a
  measured duration at prefill width, `mmq_packed` counted and untimed at decode width, the packing as a
  HOST site with exact bytes, and a 32-row FP16 GEMM bit-identical with and without a recording. Before
  any instrumentation: 6 failed (no events), the bit-identity control passed. After: 7 of 7, and
  `DeviceStagingSpansTest` still 6 of 6.
- Registered disabled in `juno-perf.jfc`, enabled in `juno-perf-spans.jfc`. `compare-llama-cpp.sh`
  adds `prefill_compute_ms`, `prefill_compute_ms_by_site` and `prefill_pack_host_ms` to the
  `device_staging` object, with four new `--selftest` cases (written with the change, not shown failing
  first).

**Item 7 (thread parity) — harness done; the CPU reference sweep is owed.** `compare-llama-cpp.sh`
passes `-Djava.util.concurrent.ForkJoinPool.common.parallelism=$((N_THREADS - 1))` (at least 1),
records `juno_threads` and `juno_common_pool_parallelism` in `host.json` (replacing
`juno_effective_parallelism`) and the flag in `juno_jvm_flags`, and the INDEX states whether the counts
match instead of always calling them mismatched. Five new `--selftest` cases (the flag at `-t 4`, both
`host.json` fields, the JVM flags, `-t 1` keeping one worker). **Live check**, unpublished run
`--cpu --threads 4` on tinyllama at `n_prompt` 128: in the prefill repetition's recording, the
`jdk.ExecutionSample` events whose stack is in the CPU matmul kernels came from exactly four threads (the
request thread 3,913 samples, `commonPool-worker-1` to `-3` 3,905, 3,884 and 3,858), not twelve.

**Verification, on the final tree.** `mvn test` on the eleven unit-test modules: **1,861 tests, 0
failures, 0 errors, 49 skipped** (step 3: 1,843; the 18 new tests are this step's), 28:32, GPU-tagged
tests included (CUDA present). `mvn install -DskipTests`, then `mvn verify -pl juno-master`: 20 ITs, 0
failures. `GpuAttentionDivergenceIT`: 3 of 3. `compare-llama-cpp.sh --selftest` and
`check-plan-thresholds.sh` pass. Candidate jar sha256 `30d4d4937d1e3994`. No `compare-lora.sh`,
`compare-vision.sh` or `compare-llama-cpp.sh` sweep was taken in this step: with the events off every
new site is one enabled check before the plain call, and the step's own gate is the pinned A/B below,
which needs prompt-free sudo. No unpinned A/B was run as a substitute, since it could not score a 0.98x
gate. Item 8 changes rotation angles on Phi-3 only (no sweep work changes).

**Item 10 (CI) — owner action.** `.github/workflows/ci.yml` is still untracked (`git ls-files .github`
is empty). Committing it is a git write and the owner's.

**Pinned measurements, run by the owner on 2026-10-01** (the agent shell has no prompt-free sudo):

*1a-ii gate:* [`20261001T172351Z-tier01b-step3a-compute-ab`](../perf-compare/20261001T172351Z-tier01b-step3a-compute-ab/INDEX.md),
six pinned runs alternated A B A B A B, A = jar `fc01f42184810aa6` (before), B = `30d4d4937d1e3994`
(this step), `n_prompt` 512, spans off. Juno t/s, median of three (min / max):

| Model | Prefill A | Prefill B | B/A | Generation B/A |
|---|---|---|---|---|
| TinyLlama | 252.76 (249.60 / 254.69) | 251.36 (247.45 / 259.28) | **0.994** | 1.011 |
| Mistral 7B | 67.13 (66.97 / 68.05) | 68.23 (65.31 / 68.60) | **1.016** | 0.996 |

**Gate (>= 0.98x) met on both.** Every invocation pinned, every prefill 512 of 512, no repetition
withheld.

*Item 7's CPU reference:* [`20261001T180241Z`](../perf-compare/20261001T180241Z/INDEX.md), the four sweep
models at `n_prompt` 128, `--cpu --pin-clocks --reps 3 --juno-reps 3`, `juno_threads` 12 against `-t
12`, INDEX "Thread counts are matched", every row scorable, every prefill 128 of 128. Ratios against the
previous CPU reference (`20260927T094414Z`, unpinned):

| Model | pp ratio | tg ratio |
|---|---|---|
| tinyllama-1.1b | 0.090x → 0.103x | 0.135x → 0.121x |
| qwen2.5-3b | 0.077x → 0.076x | 0.094x → 0.099x |
| Phi-3.5-mini | 0.048x → 0.049x | 0.095x → 0.100x |
| mistral-7b | 0.075x → 0.075x | 0.090x → 0.092x |

Juno's own absolute t/s fell 3% to 7% (turbo off) and the reference tool's tinyllama prefill by 16%;
tinyllama's ratio moves are the reference tool's, inside the 15% noise floor. Since thread count did not
change at this default (see the drift note above), the boundary here is the clock pinning. The
README's program-target table gains this as the current CPU column and the Tier 10 milestone references
move to it (tg 0.092x on mistral-7b; pp 0.049x on Phi-3.5-mini), in the same change; the previous
reference carries a superseded banner.

**The original hand-off text, kept as the record of what was asked:**
1. *The 1a-ii gate.* Same-hour pinned A/B with the spans **off**, prefill `>= 0.98x` the build without
   `juno.DeviceCompute`, tinyllama and mistral-7b at `n_prompt` 512. Baseline jar: the build of this
   tree before step 3a (sha256 `fc01f42184810aa6`, saved as `target/tier01b-step3a-ab/baseline-shaded.jar`,
   which a root `mvn clean` deletes); candidate: this tree's build. Per pair, alternating A B A B A B:
   `compare-llama-cpp.sh --gpu --pin-clocks --models tinyllama-1.1b-chat-v1.0.Q4_K_M,mistral-7b-instruct-v0.1-q4_k_m --n-prompt 512 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <jar>`.
2. *The CPU reference for item 7.* `compare-llama-cpp.sh --cpu --pin-clocks --n-prompt 128 --reps 3
   --juno-reps 3` on the four sweep models, published, and the README's CPU row gains it as the
   reference column in the same change.
3. *Item 10.* Commit `.github/workflows/ci.yml`, push, and record the first green run of both jobs here.
   *Withdrawn 2026-10-01: the owner removed item 10 as an exit criterion.*

### 2026-10-01 — before step 4: the breakdown cannot reach its residue bound on today's spans

Read-only pass over what step 4 (scope item 1b) depends on, against HEAD `d47957f` (jar sha256
`2c163485a0071c66`, built from it; the tree is dirty only in `.gitignore` and the untracked `.github/`).
Tier 00's fixes re-checked and still in place. No code changed. **Step 4 is on hold for an owner
decision on instrumentation scope.**

**Drift that blocks the criterion "no unattributed residue, <= 5% of prefill per model":**
1. **`Phi3TransformerHandler` and `Qwen3TransformerHandler` emit only `juno.ForwardPass`.** Their window
   paths run RMS norm, `Phi3Rope.ropeExt`, the host KV write, attention, both residual adds and SwiGLU
   with no `juno.RmsNorm`, `Rope`, `Attention`, `ResidualAdd` or `SwiGlu` span. Item 1b's "four
   existing spans" exist on the LLaMA-family handler only. Phi-3.5-mini is a sweep model and the
   binding one on two milestones.
2. **Three pieces of LLaMA-family window work sit outside every span**: the per-token loop writing
   the host KV and calling `DeviceKvCache.appendToken` (only the mirror's copy time is seen, as
   `memcpy_k_row_h2d`/`memcpy_v_row_h2d`), `sgemmLayerInto`'s copy of each matmul result into the
   workspace (`juno.MatVec` is emitted inside `CudaMatVec.sgemm`), and the embedding lookup and
   final-norm/LM-head projection.
3. **`juno.MatVec` has no prefill/decode split** (no `windowSize` field; the extractor writes only a
   total), and a prefill repetition's recording also holds the decode-width pass at the last prompt
   position, so the MatVec term carries a few decode calls.
4. Nesting: `juno.DeviceStaging`, `WeightDequant` and `DeviceCompute` matmul terms are inside
   `juno.MatVec`, and the `gqa` copy and compute terms inside `juno.Attention`. Not a defect, but the
   breakdown has to subtract them from their parents rather than sum all spans.

**Diagnostic reading** (unpinned, unpublished, `--device-spans`, `n_prompt` 512, one repetition each,
nested terms subtracted from their parents; scratch only, not a gate reading):

| Term, share of prefill | TinyLlama (2,205 ms) | Phi-3.5-mini (5,700 ms) |
|---|---|---|
| GEMM compute (`gemm_half`) | 8.4% | 19.8% |
| Weight dequant (device) | 1.8% | 1.8% |
| Matmul staging copies | 5.4% | 4.4% |
| Host FP16 pack | 3.5% | 2.0% |
| `juno.MatVec` host remainder | 10.3% | 7.7% |
| Attention compute (`gqa_attention`) | 13.3% | 13.8% |
| Attention copies plus remainder | 4.1% | 1.3% (no `juno.Attention` span) |
| KV mirror row copies | 7.0% | 4.5% |
| Host SwiGLU / RmsNorm / Rope / ResidualAdd | 34.0 / 3.0 / 1.1 / 1.1% | not spanned |
| **Unattributed** | **7.0%** | **44.8%** |

Applying the same subtraction to step 2's spans run (no compute span then) gives 6.6% on TinyLlama,
9.2% on mistral-7b and 10.4% on qwen2.5-3b unattributed. The ranking step 4 expected holds where it can
be read: host SwiGLU is the largest single term on the LLaMA family, and the GEMM's share is larger on
the bigger model.

**Owner decisions (2026-10-01):** (1) **add the spans first**, as a new step 4a before the breakdown:
per-op spans on the Phi-3 and Qwen3 window paths, spans for the window work outside every span on all
three handlers, and a prefill/decode split for `juno.MatVec`, tests first, then a same-hour pinned A/B
(prefill >= 0.98x, the 1a-ii standard) run by the owner, then the breakdown. (2) The breakdown sweeps
themselves may be taken **unpinned** by the agent and published labelled as unpinned, since they report
shares within one run; any A/B gate still needs pinned clocks.

### 2026-10-01 — implementation step 4a: spans for the whole prefill window

Scope as decided above. No change to what any window computes; one new event, one new field, and the
existing per-op events placed on two more handlers. Build of this step: jar sha256 `513f57643c8fe97c`
(baseline, HEAD `d47957f`: `2c163485a0071c66`, saved as `target/tier01b-step4a-ab/baseline-shaded.jar`,
candidate as `candidate-shaded.jar` beside it; a root `mvn clean` deletes both).

**What shipped.**
- `node`: `WindowStepEvent` (`juno.WindowStep`: `step`, `windowSize`, `startPosition`), one event per
  call site per layer for `embed`, `projection` (each matmul call including its copy-out into the
  workspace; the backend's `juno.MatVec` nests inside it), `bias_add`, `kv_write` (host KV rows plus the
  device mirror append) and `lm_head`, on the `forwardBatch` window path of the LLaMA-family, Phi-3 and
  Qwen3 handlers (a `projectWindow` wrapper in the first and last, inline spans around Phi-3's fused
  calls). The Phi-3 and Qwen3 window paths gain `juno.RmsNorm` (Qwen3's per-head Q/K norm as a third),
  `Rope`, `Attention`, `ResidualAdd` and `SwiGlu` at the LLaMA-family granularity. Decode and
  multi-decode paths are unchanged. `MatVecEvent.windowSize` (default 1; the batch in `CudaMatVec`'s
  three batched paths).
- `metrics`: `WindowStepBucket` (`juno.WindowStep.<step>.{count,prefill.*,decode.*}`, every known
  step written on every run); `juno.MatVec.{prefill,decode}.{count,total_ms}` by the call's width, a
  recording without the field counting only in the total.
- Harness: `juno.WindowStep` enabled in `juno-perf.jfc` (like the per-op events, so the A/B's
  spans-off side carries its cost); `prefill-breakdown.sh` (new): reads a `--device-spans` run's prefill
  repetitions, subtracts each nested device term from its parent span, reports the per-term median and
  the residue, exits 1 above `--max-residue-pct` (default 5).
- Docs: `docs/howto.md` (prefill window breakdown section, keys), `docs/performance.md`,
  `docs/agent-arch.txt`, `CHANGELOG.md` (Session 102).

**Tests, written first and shown failing for the right reason.**
- `JfrMetricsExtractorWindowStepTest` (`metrics`, 5 cases): 5 of 5 failed on the unchanged extractor
  (keys absent); pass. `metrics` 82 of 82.
- `PrefillWindowSpansTest` (`node`, CPU, synthetic LLaMA, Phi-3 and Qwen3 models, 3 cases): event
  counts and window fields per handler; 3 of 3 failed before the change (no events), pass.
- `DeviceComputeSpansTest.matVec_recordsTheCallsBatchWidth` (`@Tag("gpu")`): widths 32, 4 and 1 for a
  GEMM, a small batch and a GEMV; failed before (field absent), passes. The class's other 7 still pass.
- `prefill-breakdown.sh --selftest` (10 cases, a synthetic 1,000 ms window whose terms are known):
  written with the script, not shown failing first.

**Residue, candidate build** (unpinned, unpublished, `--device-spans`, `n_prompt` 512, one repetition;
`prefill-breakdown.sh`, exit 0):

| Model | Prefill ms | Residue | Largest terms |
|---|---|---|---|
| tinyllama-1.1b | 2,064 | **0.7%** | host SwiGLU 36.9%, projection dispatch and copy-out 13.1%, attention kernel 12.1%, GEMM 7.9%, KV mirror copies 7.2% |
| qwen2.5-3b | 5,581 | **0.6%** | host SwiGLU 46.1%, projection dispatch and copy-out 13.7%, GEMM 10.7%, attention kernel 7.7% |
| Phi-3.5-mini | 6,375 | **1.5%** | host SwiGLU 28.4%, GEMM 18.9%, attention kernel 13.5%, projection dispatch and copy-out 12.2%, KV mirror copies 5.2% |
| mistral-7b | 7,553 | **0.5%** | host SwiGLU 38.2%, GEMM 17.1%, projection dispatch and copy-out 11.5%, attention kernel 9.9% |

Preliminary, not the step 4 artifact: one unpinned repetition at one prompt length. It shows the
instrumentation meets the `<= 5%` residue bound on all four sweep models, and that the ranking step 4
expected holds: host SwiGLU first everywhere, then the host side of each projection (dispatch, result
allocation and copy-out, which item 2's non-allocating form removes), then the GEMM and the attention
kernel.

**Informational A/B, unpinned** (spans off; three interleaved runs per side, A = baseline, B =
candidate, `n_prompt` 512; not scorable against a 0.98x gate): TinyLlama prefill 254.93 (254.20 to
255.73) against 248.22 (245.53 to 257.91), **0.974x**; Phi-3.5-mini 88.31 (85.63 to 90.65) against 89.87
(88.89 to 90.06), **1.018x**. The candidate's TinyLlama spread (5%) is wider than the gap. If the pinned
gate misses, the fallback is the step 1 design: move `juno.WindowStep` into `juno-perf-spans.jfc` so
it is on only for breakdown runs.

**Verification, on the final tree.** `mvn test` on the eleven unit-test modules: **1,870 tests, 0
failures, 0 errors, 49 skipped** (step 3a: 1,861; the 9 new tests are this step's), 28:32, GPU-tagged
tests included. `mvn install -DskipTests`, then `mvn verify -pl juno-master`: 20 ITs, 0 failures.
`compare-llama-cpp.sh --selftest`, `prefill-breakdown.sh --selftest` and `check-plan-thresholds.sh`
pass. The `install` rebuilt the working-tree jar from the same sources (sha256 `92643aec3c3b226a`; the
build is not byte-reproducible); the A/B uses the saved `candidate-shaded.jar`. Not run:
`compare-lora.sh` and `compare-vision.sh` (the LoRA handlers and the vision encoder are untouched, and
moondream2's text half runs the unchanged Phi-2 handler; both remain in step 8's closing matrix).

**Owed, for the owner (pinned, needs prompt-free sudo):** the step 4a gate, same-hour, alternating A B
A B A B, prefill `>= 0.98x` on both models:
`scripts/performance-tests/compare-llama-cpp.sh --gpu --pin-clocks --models tinyllama-1.1b-chat-v1.0.Q4_K_M,Phi-3.5-mini-instruct-Q4_K_M --n-prompt 512 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar target/tier01b-step4a-ab/{baseline|candidate}-shaded.jar`.
Phi-3.5-mini stands in for Mistral 7B here because its handler is the one whose window path gained the
most spans. Then step 4 proper: `compare-llama-cpp.sh --gpu --device-spans --juno-reps 3` on the four
sweep models at `n_prompt` 128 and 512, published, and `prefill-breakdown.sh` on each.

**Step 4a gate, taken by the owner 2026-10-01/02 (pinned, 23:59 to 00:09 UTC): met.** The command
above, alternating A B A B A B, `--out target/perf-compare/step4a-ab-{baseline|candidate}-{1|2|3}`.
Prefill B/A **0.993** on TinyLlama (251.67 against 249.88 t/s) and **1.005** on Phi-3.5-mini (89.13
against 89.58); generation 0.999 and 0.987, recorded, not gated. All six runs pinned (governor
performance, turbo off, GPU locked at 1911 MHz), jar hashes `2c163485a0071c66` and `513f57643c8fe97c`
as expected, every prefill 512 of 512, every row scorable, GC pause at most 9 ms. Published:
[`20261001T235933Z-tier01b-step4a-ab`](../perf-compare/20261001T235933Z-tier01b-step4a-ab/INDEX.md).
`juno.WindowStep` stays in the base `juno-perf.jfc`; the fallback is not needed. This was taken after
the step 4 breakdown (owner decision, step 4 record), on the same jars, so nothing the breakdown read
changes.

### 2026-10-01 — implementation step 4: the per-term prefill breakdown

**Owner decision (2026-10-01, this session): step 4 is taken before step 4a's pinned A/B gate**, which is
still owed (it needs prompt-free sudo; this session had none). Why the order does not affect the
breakdown: a breakdown run passes `--device-spans`, which records `juno.WindowStep` whichever `.jfc`
holds it, and step 4a's documented fallback (moving `juno.WindowStep` into `juno-perf-spans.jfc` if the
gate misses) changes only spans-off runs, through a harness file rather than the jar. The gate still
decides whether the spans-off build carries the event; it no longer orders this step. Taken **unpinned**
under the owner's 2026-10-01 decision (2): a breakdown reports shares within one run.

Build: HEAD `c42de6e` (step 4a), working-tree jar sha256 `92643aec3c3b226a`, tree dirty only in
`.gitignore` and the untracked `.github/`. Command, once per prompt length:
`compare-llama-cpp.sh --gpu --device-spans --models <the four sweep models> --n-prompt {512|128}
--juno-reps 3 --no-tuned-lane` (the tuned lane was skipped because the breakdown reads the default lane),
then `prefill-breakdown.sh RUN_DIR --json ...`. Both runs exit 0, every row scorable, prompt tokens
512/512 and 128/128; `prefill-breakdown.sh` exits 0 on both (every model under the `<= 5%` residue
bound). Published: [`20261001T224724Z`](../perf-compare/20261001T224724Z/INDEX.md) and
[`20261001T225929Z`](../perf-compare/20261001T225929Z/INDEX.md), each with `prefill-breakdown.md` and
`.json` beside the run, an INDEX banner saying unpinned and not a ratio reference, and a row in
`docs/perf-compare/README.md`. Each term is the median of three repetitions; nested device terms are
subtracted from their parent spans (see the script header), so the terms and the residue add up to the
window.

`n_prompt=512` ([`20261001T224724Z`](../perf-compare/20261001T224724Z/prefill-breakdown.md)), share of prefill:

| Term | tinyllama-1.1b (2,194 ms) | qwen2.5-3b (5,617 ms) | Phi-3.5-mini (5,920 ms) | mistral-7b (7,791 ms) |
|---|---|---|---|---|
| GEMM compute (`juno.DeviceCompute`) | 8.3% | 11.1% | 19.7% | 17.6% |
| Weight dequant (device) | 1.9% | 1.8% | 1.8% | 2.7% |
| Matmul staging copies | 5.5% | 5.1% | 4.3% | 5.0% |
| Host FP16 pack | 3.7% | 3.2% | 2.0% | 3.7% |
| Projection dispatch and copy-out | 12.3% | 13.2% | 12.2% | 11.5% |
| Attention kernel | 13.3% | 8.8% | 13.8% | 10.4% |
| Attention copies | 1.5% | 1.0% | 1.3% | 1.3% |
| Attention host part | 2.5% | 1.6% | 1.9% | 1.8% |
| KV mirror row copies | 7.4% | 4.1% | 4.7% | 2.8% |
| Host KV write | 1.1% | 0.4% | 2.8% | 0.8% |
| Host SwiGLU | 36.0% | 45.3% | 28.0% | 37.2% |
| Host RMS norm | 3.0% | 2.1% | 2.5% | 2.6% |
| Host RoPE | 1.1% | 0.6% | 2.8% | 0.9% |
| Host residual add | 1.1% | 0.8% | 1.0% | 1.1% |
| Bias add | 0.0% | 0.4% | 0.0% | 0.0% |
| Embedding | 0.0% | 0.0% | 0.0% | 0.0% |
| LM head | 0.0% | 0.0% | 0.0% | 0.0% |
| **Unattributed residue** | 0.8% | 0.4% | 1.3% | 0.5% |

`n_prompt=128` ([`20261001T225929Z`](../perf-compare/20261001T225929Z/prefill-breakdown.md)), share of prefill:

| Term | tinyllama-1.1b (524 ms) | qwen2.5-3b (1,334 ms) | Phi-3.5-mini (1,306 ms) | mistral-7b (1,916 ms) |
|---|---|---|---|---|
| GEMM compute (`juno.DeviceCompute`) | 10.9% | 11.6% | 18.4% | 20.4% |
| Weight dequant (device) | 6.7% | 6.9% | 7.8% | 10.1% |
| Matmul staging copies | 5.9% | 5.5% | 4.8% | 5.1% |
| Host FP16 pack | 2.9% | 2.6% | 1.7% | 2.9% |
| Projection dispatch and copy-out | 14.6% | 12.0% | 14.0% | 11.1% |
| Attention kernel | 4.0% | 2.6% | 2.8% | 2.5% |
| Attention copies | 2.2% | 1.5% | 1.7% | 1.4% |
| Attention host part | 2.7% | 1.5% | 2.3% | 1.9% |
| KV mirror row copies | 6.7% | 4.2% | 4.9% | 2.7% |
| Host KV write | 0.2% | 0.1% | 2.8% | 0.5% |
| Host SwiGLU | 36.8% | 46.3% | 29.8% | 36.1% |
| Host RMS norm | 3.1% | 2.0% | 2.8% | 2.5% |
| Host RoPE | 1.0% | 0.6% | 3.1% | 0.9% |
| Host residual add | 1.1% | 0.7% | 0.9% | 0.8% |
| Bias add | 0.0% | 0.2% | 0.0% | 0.0% |
| Embedding | 0.0% | 0.0% | 0.0% | 0.0% |
| LM head | 0.1% | 0.2% | 0.1% | 0.1% |
| **Unattributed residue** | 1.6% | 1.0% | 2.2% | 1.0% |

**GEMM share per model, for [Tier 01C](TIER-01C-packed-kquant-matmul.md)** (the number its throughput
threshold is conditioned on): 8.3% / 11.1% / 19.7% / 17.6% of a 512-token window (tinyllama, qwen2.5-3b,
Phi-3.5-mini, mistral-7b) and 10.9% / 11.6% / 18.4% / 20.4% at 128. The device dequantization in front
of it (the work a packed integer GEMM removes) adds 1.8% to 2.7% at 512 and 6.7% to 10.1% at 128. Both
shares grow once items 6 and 2 remove the host terms: on the step 5 estimate below the GEMM becomes about
20%, 30%, 36% and 39% of the 512-token window.

**What the breakdown justifies (the step 4 decision).**
- **Item 6: yes, first.** Host SwiGLU is the largest term on every model at both lengths (28.0% to 46.3%),
  and the elementwise group plus the KV mirror row copies is 39% to 53% of a 512-token window. The
  ranking the step text expected going in holds.
- **Item 2: yes, second, on the same region.** Matmul staging plus host FP16 pack plus projection
  dispatch and copy-out is 18.5% to 21.5% at 512. Only part of the dispatch term is removable (the result
  allocation and copy-out are; the launch is not).
- **Item 3: no throughput lever on the sweep shape.** The GPU static local surface already prefills a
  512-token prompt as one window, so the sweep contains no per-chunk cost to remove. What the breakdown
  does show is the size of the per-window cost a fixed `32` pays repeatedly: device weight dequantization
  falls from 6.7% to 10.1% of a 128-token window to 1.8% to 2.7% of a 512-token one, so a 32-token chunk
  pays it at a larger share again. That is the measured reason item 3's per-surface review starts from,
  and the term itself is removed by Tier 01C's packed GEMM rather than by chunk sizing. Item 3 stays a
  review, not an implementation item, unless that review finds a surface where the VRAM query is
  meaningful and the default is still `32`.
- **Item 4: attention is still material at 512 tokens on every architecture, and the remainder is Tier
  02's.** Kernel, copies and host part together: 17.3% (tinyllama), 11.4% (qwen2.5-3b), 17.0%
  (Phi-3.5-mini), 13.4% (mistral-7b) of a 512-token window, against 5.6% to 8.8% at 128. Per prefilled
  token it roughly triples from 128 to 512 (Phi-3.5-mini 0.69 to 1.97 ms per token). Item 0 already put
  Phi-3 and Qwen3 on the kernel; nothing in this tier changes the kernel itself, so these are the numbers
  Tier 02's tiled kernel is held to, and Tier 02 is not to be credited with the item-0 gain.

### 2026-10-01 — implementation step 5: the threshold decomposition, and an escalation

Read off the 512 and 128 breakdowns above, against the pinned post-item-0 reference ratios
([`20260930T215547Z`](../perf-compare/20260930T215547Z/INDEX.md) at 512,
[`20260930T211637Z`](../perf-compare/20260930T211637Z/INDEX.md) at 128). A removed share `f` of the
window is worth `1 / (1 - f)` on Juno's prefill t/s with the reference tool unchanged. Assumed removal
per term, stated so the estimate can be checked against the result: item 6 removes 95% of host SwiGLU,
RMS norm and residual add, 95% of host RoPE except on Phi-3.5-mini (its rotation stays on the host,
scope item 6) and 80% of the KV mirror row copies (one contiguous copy per layer keeps the bytes and
drops the per-row calls); item 2 removes 70% of matmul staging (the `>= 70%` bytes threshold), all of the
host FP16 pack, and half of projection dispatch and copy-out (allocation and copy-out go, the launch
stays). The shares come from a spans run, which inflates spanned host time by a few percent, so treat
every multiple below as an estimate with that much give.

| Model | Reference 512 ratio | Threshold | Multiple needed | Item 6 alone | Item 2 alone | Items 6 + 2 | Projected 512 ratio |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | 0.067x | >= 0.10x | 1.49x | 1.82x | 1.16x | 2.42x | 0.163x |
| qwen2.5-3b | 0.061x | >= 0.10x | 1.65x | 1.99x | 1.16x | 2.71x | 0.164x |
| Phi-3.5-mini | 0.075x | >= 0.08x | 1.07x | 1.51x | 1.12x | 1.81x | 0.136x |
| mistral-7b | 0.100x | >= 0.10x | 1.00x | 1.73x | 1.15x | 2.23x | 0.222x |

**The first two threshold rows sum, with margin, on item 6 alone.** Items 6 and 2 together are
estimated at 1.8x to 2.7x. mistral-7b already reads 0.0999x at the reference, so its row is met within
noise before any work; Phi-3.5-mini needs only 1.07x after item 0.

**The third row does not sum, and no item in this tier owns what it needs.** The third row is the 512
ratio at least 1.00 times the 128 ratio, on every model.

| Model | Reference 512/128 | After item 6 | After items 6 + 2 |
|---|---|---|---|
| tinyllama-1.1b | 1.021 | 1.016 | 1.000 |
| qwen2.5-3b | 0.911 | 0.895 | 0.913 |
| Phi-3.5-mini | 0.916 | 0.888 | 0.866 |
| mistral-7b | 0.961 | 0.988 | 1.018 |

The cause is attention, which grows with context. Items 6 and 2 remove costs that are fixed per token.
That makes the growing term a larger share of what remains, so on Phi-3.5-mini the row gets *worse* as
the tier's own items land. Mistral 7B is the exception: its per-window weight dequantization amortizes
at 512. Reaching 1.00 on qwen2.5-3b and Phi-3.5-mini needs attention whose per-token cost does not
roughly triple from 128 to 512 tokens. That is the tiled long-context kernel, which is
[Tier 02](TIER-02-attention-long-context.md)'s, and item 4 hands it there rather than building it. Every
model's reference reading already failed this row before step 5 except tinyllama, which passes by
0.021.

**Escalated to the owner, 2026-10-01; decided the same day: option 1, the row moves to Tier 02.** Per the decomposition clause, this is
recorded before items 6 and 2 are implemented. The options:
1. Re-home the third milestone row to Tier 02, whose own `>= 0.90` 2048-over-512 milestone measures the
   same mechanism.
2. Keep the row here and accept that the tier closes **partial-complete** on it under the contingency,
   with Tier 02 named as the successor for the attention term.
3. Pull a long-context attention item forward into this tier.

Implementation of items 6 and 2 (step 6) is not blocked by the decision. They are justified by the
first two rows whichever way the third goes.

**Owner decision (2026-10-01): option 1.** The 512-over-128 row moved to Tier 02, unchanged at `>= 1.00`, in
the README milestone table and in Tier 02's milestone and exit criteria. Its reference reading there is
the pinned post-item-0 one (0.911, qwen2.5-3b). Tier 01B is now scored on the first two rows only. This
tier still publishes the 512-over-128 reading in its closing sweeps, so Tier 02 starts from a number taken
after items 6 and 2 rather than from the estimate above.

### 2026-10-02 — before step 6: item 6 cannot be measured ahead of item 2, raised with the owner

Read-only pass over what item 6 depends on, against HEAD `c42de6e` plus the uncommitted step 4/5 tree.
Confirmed as the plan states: the window SwiGLU is a single-threaded scalar loop calling `Math.exp` in
double precision (`LlamaTransformerHandler.silu`, used by all three window paths); `rmsNormGpu` is left
null on purpose (the constructor comment cites the round-trip A/B), so both norms run on the host;
`DeviceKvCache.appendToken` copies one K row and one V row per token per layer; `CudaRmsNorm.normalizeResident`,
`ResidentChain` and `ResidentActivation` exist; `CudaRope` rotates adjacent pairs only; Phi-3
(`Phi3Rope.ropeExt`) and Qwen3 (`Qwen3Rope`, per-head Q/K norm) keep their own rotations. All four sweep
models are on disk; `nvcc` 12.0 is installed.

**Not as the plan states: "measure them separately, item 6 first."** Every operation item 6 moves sits
between two GEMMs, and `MatVec.sgemm` takes a host `float[][]` and returns a new host `float[][]`,
packing the input to FP16 on the host because `cublasGemmEx` runs on FP16 operands. A device SwiGLU fed
by host-returned gate and up outputs would add copies rather than remove them (Tier 01 measured the
round-trip norm slower than the CPU for the same reason). So item 6 needs GEMMs that read and write
device buffers, which is item 2's mechanism. Estimate for one region per layer (residual uploaded once,
q, k and v downloaded for the host KV, attention output uploaded, residual downloaded): TinyLlama's
staged bytes per 512-token window fall by about 71%, which would already meet item 2's `>= 70%`.

**Owner decision (2026-10-02): one region, one change.** The prefill-window region is built with
device-in, device-out GEMMs and item 6's operations in one change, and item 6's thresholds and item 2's
bytes threshold are scored off the same measurement; `>= 1.25x` on TinyLlama stays the speed gate.
Item 2's remainder (the residual kept on the device across layers, the non-allocating `MatVec` form and
its contract with Tier 10 item 4) follows as a second measured change. Step 7's "re-measure after each
change" applies to those two changes.

### 2026-10-02 — implementation step 6, first change: the prefill-window device region (item 6 with item 2's operand residency)

Scope as decided above (one region, one change). Build: HEAD `c42de6e` plus this change; baseline jar
`92643aec3c3b226a` (HEAD's build, saved as `target/tier01b-item6-ab/baseline-shaded.jar`), candidate as
`candidate-shaded.jar` beside it (a root `mvn clean` deletes both).

**What shipped.**
- `node`: `PrefillWindowRegion` (new). Per layer of a prefill window wider than 8 rows: upload the residual
  once, RMS norm, FP16 cast, Q/K/V GEMMs on device operands, bias add, RoPE, then either attention inside
  the region (the window's K and V cast straight into the `DeviceKvCache` mirror, the existing attention
  kernel reading it) followed by the rest of the layer and one download of k, v and x; or a download of q,
  k and v for the handler's own RoPE and attention, then `finishLayer` (O GEMM, residual add, norm, gate and
  up side by side, SwiGLU written as FP16, down GEMM, residual add, download x). LLaMA family and Qwen2
  run the whole layer in the region (adjacent and split-half RoPE, biases); Phi-3 (LongRoPE) and Qwen3
  (per-head Q/K norm) keep RoPE and attention on the host between the two calls. Pooled per concurrent
  prefill; each operation holds the serialization lock and waits for its stream before releasing it;
  device OOM redoes the layer on the host path, a mirror OOM retires the mirror. `PrefillWindowKernels`
  and `prefill_window.cu` (new): FP16 cast, SwiGLU to FP16, residual add, bias add, and an RMS norm in the
  host loop's order. `rope.cu` gains `rope_split_half`; `CudaRope` takes a pairing. `CudaMatVec` gains
  `gemmOnStream` / `dequantOnStream` / `gemmHalfOnStream` (device operands, caller's stream, a chosen output
  row stride; `CudaFp16GemmOps.gemmHalf` takes `ldc`). `DeviceKvCache` gains `appendWindow` (one copy per
  tensor), `writeWindowOnDevice` and `markWritten`; its `grow` no longer leaks the K buffer when the V
  allocation fails. Every window path (region or host, all three handlers) now appends the mirror once per
  layer per window. `ResidentActivation.materializeAll`, `KernelParams.i64`,
  `ForwardPassHandler.prefillRegionActive()`, `PrefillRegionOptions` (`JUNO_PREFILL_REGION`, on by
  default, forwarded by `ClusterHarness`). JFR: `juno.WindowStep` step `device_layer`; `juno.DeviceCompute`
  sites `rms_norm`, `convert_fp16`, `bias_add`, `rope`, `kv_append`, `gqa_attention_region`, `swiglu`,
  `residual_add`.
- `metrics`: the new sites and step written on every run.
- Harness: `prefill-breakdown.sh` treats `device_layer` as a parent beside `projection` (term
  `projection_and_region_host`, was `projection_dispatch_and_copy_out`) with `device_elementwise` and
  `region_attention_compute` as new terms; `compare-llama-cpp.sh` reads `gqa_attention_region` as evidence
  that GPU attention resolved on (in-region attention issues no `memcpy_gqa_*` copies, so a spans run would
  otherwise have read the LLaMA-family prefill lane as off).

**Bit identity, and how the first version missed it.** The region was first built on the existing device
norm (`rms_norm.cu`: tree reduction, `rsqrtf`). Prefill logits matched the host path within 0.002 to
0.005 relative L2, but one Qwen2.5-3B decode step after a region prefill read 0.090 with top-1 equal, at
the level of Tier 00's planted-fault calibration. Diagnosis, not a looser bound: the region's KV mirror
held exactly the FP16 of the host KV (0 mismatches over 36 layers x 64 positions), the host path is
bit-deterministic run to run, and per-layer K/V differed by a smooth 1e-6 to 1.3e-3 growing with depth
(no localized jump), so the cause was the norm's few-ulp difference flipping FP16 roundings of the GEMM
inputs. The accepted GPU attention kernel reads 0.0246 at the same step, so that step is sensitive to any
change. `rms_norm_host_order` sums in the host loop's order and is bit-identical to `rmsNormInto`; with it
**the region's logits equal the host path's bit for bit** on all four architectures, at every site.

**Tests, written first and shown failing for the right reason.**
- `PrefillWindowKernelsParityTest` (`node`, `@Tag("gpu")`, 49 cases): 29 failed against a throwing
  skeleton, then 20 more for the host-order norm; all pass. FP16 cast bit-identical to
  `Float.floatToFloat16` over 1M values (ties, subnormals, overflow); SwiGLU at W = 1, 8, 9, 32, 512 and
  I = 5632, 11008, 8192, 14336, 6144 bit-identical to the host loop then cast (bound was one FP16 ulp);
  residual and bias adds bit-identical; split-half RoPE max difference 0; norm bit-identical at 20 shapes.
  `RopeKernelParityTest` still 6 of 6.
- `DeviceKvWindowAppendTest` (`@Tag("gpu")`, 6 cases): 6 failed against the skeleton; pass. Window append
  equals per-token append; 2 copies for a 512-row window; device write casts in place with 0 host copies
  and leaves the watermark until marked; growth across a window keeps the prefix; a window past the prefix
  is not readable; a growth OOM (device filled to 4 MiB free) is a recognized VRAM OOM and leaves the
  watermark.
- `GemmOnStreamTest` (`@Tag("gpu")`, 7 cases): 7 failed against the skeleton; pass. Bit-identical to
  `sgemm` for FP16 and Q4_K weights at W = 9, 32, 512, and side by side with a doubled row stride.
- `PrefillRegionHandlerParityTest` (`@Tag("gpu")`, TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Qwen3-1.7B): 4
  failed on "the default must build the region" before the wiring; all pass. Region on against off on one
  CUDA backend: logits **bit-identical** at two fresh windows, a continuation window at position 48, a
  single decode and a two-stream decode; from a recording of the first window, no `juno.SwiGlu` or
  `juno.ResidualAdd` events, no matmul device-to-host copy (the item's "gate and up produce no
  device-to-host copy"), KV mirror copies <= 2 x layers; device memory back to its start after release,
  and no growth across a repeated window.
- `PrefillRegionGreedyIT` (`juno-master`, `-Pgpu`): six prompts, 64 greedy tokens, region on against off:
  **6 of 6 identical on TinyLlama and on Mistral 7B**.
- `metrics`: `JfrMetricsExtractorDeviceComputeTest` and `JfrMetricsExtractorWindowStepTest` failed on the
  absent keys; 82 of 82 pass. `prefill-breakdown.sh --selftest`: 6 of 16 failed on the old script; 16 of 16
  pass. `compare-llama-cpp.sh --selftest`: the new case failed (read `off`); all pass.

**Test-suite hygiene found on the way.** The first eleven-module reactor run with this change failed in
`PrefillRegionHandlerParityTest`: the region fell back on device OOM even on TinyLlama, and the Phi-3.5 and
Qwen3 cases ran out of memory on the host path. Two existing test classes loaded real models on CUDA and
never released them (`LlamaTransformerHandlerGpuAttentionLiveTest`, four TinyLlama loads;
`LlamaTransformerHandlerGpuResidencyTest`, two TinyLlama and one Qwen2.5-3B), and device matrices have no
GC cleaner, so about 6 GB stayed allocated for the rest of the test JVM. Both now release every handler
and backend after each test. No product code changed for this.

The rerun then failed once in `PrefillWindowSpansTest` (step 4a's CPU test: 13 projection events where 14
were expected). Not this change: alone it failed 3 of 6 runs, and 0 of 6 with JFR on the operating-system
clock. It is step 0's timestamp-counter skew (this host's kernel clocksource is `hpet`): a span crossing
cores reads a negative duration and JFR drops the event. The harness already passes
`-XX:-UseFastUnorderedTimeStamps` for perf runs; `node`'s surefire `argLine` now does too (6 of 6 pass).

**Verification, on the final tree (2026-10-02).** `mvn test` on the eleven unit-test modules: **1,936 tests,
0 failures, 0 errors, 49 skipped** (step 4a: 1,870), 29:53, GPU-tagged tests included. `mvn install
-DskipTests`, then `mvn verify -pl juno-master`: 20 ITs, 0 failures (three-node pipeline, tensor-parallel,
in-process, unsupported architecture). `-Pgpu`: `PrefillRegionGreedyIT` 2 of 2, `GpuAttentionDivergenceIT` 3
of 3 with the region on and the same divergence steps as step 3a (TinyLlama 8, 13, 23; Phi-3.5-mini 37, 40;
Qwen3-1.7B 20, 23, 32), `GpuForwardPassIT` 5 of 5 on TinyLlama against the CPU oracle (top-1 equal, top-5
overlap 5, logits relative L2 0.009). `check-plan-thresholds.sh`, `compare-llama-cpp.sh --selftest`,
`prefill-breakdown.sh --selftest` and `selftest-engine-stdin.sh` pass. Not run in this step, and owed to the
closing matrix (step 8): `compare-lora.sh` (the LoRA handlers are separate classes the region does not
touch), `compare-vision.sh` (moondream2's text half runs the Phi-2 handler, which has no device path),
`compare-prefill-batch.sh`, `ModelLiveRunnerIT`'s 512-token check and `smoke-tier01b-prefill.sh` (neither
exists yet).

**Measurement (agent, unpinned; the gates that need pinned clocks are owed to the owner).**

*Breakdown*, published as [`20261002T014419Z`](../perf-compare/20261002T014419Z/INDEX.md) (`--gpu
--device-spans --juno-reps 3 --no-tuned-lane`, `n_prompt=512`, unpinned under the owner's 2026-10-01
decision; jar `fb6d32e6c41cb3dd`; `prefill-breakdown.sh` exit 0, residue 1.1% to 2.8%). Per window, all
three repetitions identical in bytes and counts:

| Model | Prefill ms (step 4 → now) | Host elementwise share | KV mirror H2D copies (limit) | H2D + D2H MB (step 2 → now) | Bytes change |
|---|---|---|---|---|---|
| tinyllama-1.1b | 2,194 → 556 | 41.2% → **0%** | **0** (44) | 1,405 → 207 | **-85.3%** |
| qwen2.5-3b | 5,617 → 1,260 | 48.8% → **0%** | **0** (72) | 3,288 → 339 | **-89.7%** |
| Phi-3.5-mini | 5,920 → 3,121 | 34.3% → 5.3% (host RoPE) | **64** (64) | 2,645 → 1,809 | -31.6% |
| mistral-7b | 7,791 → 2,741 | 41.8% → **0%** | **0** (64) | 4,689 → 670 | **-85.7%** |

What remains on the region models is the GEMM (29.8% to 50.8%) and the attention kernel (30.2% to
46.3%), now the two dominant terms: the GEMM is Tier 01C's, attention at long context Tier 02's.
GPU attention resolved `on` for all four rows (the harness fix above).

*Informational A/B* (not a gate: unpinned), alternating baseline and candidate jars three times each,
`n_prompt=512`, median (min to max), Juno t/s, logs under `target/perf-compare/item6-unpinned-ab/`:

| Model | Prefill baseline | Prefill candidate | B/A | Generation B/A |
|---|---|---|---|---|
| TinyLlama | 258.97 (258.75 to 262.10) | 932.04 (872.80 to 934.18) | 3.60x | 0.995x |
| Mistral 7B | 68.80 (48.80 to 69.09) | 188.09 (187.38 to 193.59) | 2.73x | 0.997x |

Against step 5's estimate (items 6 and 2 together: 2.42x TinyLlama, 2.23x Mistral) the change reads
above it; the estimate assumed item 6 left attention and the GEMM where they were, and moving the
attention input and output onto the device removed copies the estimate charged to item 2's remainder.
The unpinned sweep above reads pp ratios of 0.230x, 0.259x, 0.143x and 0.283x at 512 (spans on): not a
milestone score, which comes from the tier's pinned closing sweep.

**Item 6 threshold, scored as far as an unpinned agent can:** host elementwise <= 10% of prefill on
tinyllama, qwen2.5-3b and mistral-7b: **met (0%)**; KV mirror H2D copies <= 2 x layers per window: **met
(0)**; greedy output identical over 64 tokens on tinyllama and mistral-7b: **met (6 of 6 each)**; TinyLlama
prefill >= 1.25x the pre-item-6 build from a same-hour **pinned** A/B: **owed** (the unpinned A/B reads
3.60x). **Item 2's bytes threshold** (>= 70% against step 2): met on tinyllama, qwen2.5-3b and mistral-7b,
**missed on Phi-3.5-mini (-31.6%)**, whose LongRoPE and attention keep Q, K and V crossing the bus every
layer; moving them needs a device LongRoPE kernel, which no item in this tier owns. Item 2's
non-allocating `MatVec` form (its contract with Tier 10 item 4) is the second change of step 6 and is
not done.

**Owed, for the owner (pinned, needs prompt-free sudo):** the item 6 gate, same-hour, alternating A B A B A B,
prefill `>= 1.25x` on TinyLlama (Mistral 7B recorded, not gated), and decode `>= 0.95x` on both:
`scripts/performance-tests/compare-llama-cpp.sh --gpu --pin-clocks --models tinyllama-1.1b-chat-v1.0.Q4_K_M,mistral-7b-instruct-v0.1-q4_k_m --n-prompt 512 --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar target/tier01b-item6-ab/{baseline|candidate}-shaded.jar --out target/perf-compare/item6-ab-{baseline|candidate}-{1|2|3}`
(baseline `92643aec3c3b226a`, candidate `fb6d32e6c41cb3dd`).

**Item 6 gate, taken by the owner 2026-10-02 (pinned, 03:05 to 03:16 UTC): met.** The command above, driven
by `target/tier01b-item6-ab/run-gate.sh`, alternating A B A B A B. Prefill B/A **3.899** on TinyLlama
(240.00 against 935.81 t/s) and **2.806** on Mistral 7B (68.35 against 191.76); generation **1.023** and
**1.003**. All six runs pinned (governor performance, turbo off, GPU locked at 1911 MHz), jar hashes
`92643aec3c3b226a` and `fb6d32e6c41cb3dd` as expected, every prefill 512 of 512, every row scorable, no
repetition withheld, GC pause at most 14.7 ms (Mistral, both sides alike). Published:
[`20261002T030500Z-tier01b-step6-region-ab`](../perf-compare/20261002T030500Z-tier01b-step6-region-ab/INDEX.md).
Every item 6 threshold is now met; the exit criterion is ticked. Item 2's criterion stays open (Phi-3.5-mini
bytes, the non-allocating form).

### 2026-10-02 — implementation step 6, second change: item 2's remainder (the residual across layers, the non-allocating `MatVec` form)

Scope as the owner decided before step 6: the residual kept on the device across layers, and the
non-allocating batched `MatVec` form with its contract for Tier 10 item 4. Build: HEAD `61c31cc` plus this
change. Baseline jar `d287e929f78c11ff` (HEAD, built from a `git archive` export), candidate `caaf5810752a1ecb`
(this tree; the jar the published sweep below measured). A root `mvn clean` deleted `target/` before the owner
ran the gate, so the jars and `run-gate.sh` now live in `dist/tier01b-item2-ab/` (gitignored, untouched by
`mvn clean`), with the candidate rebuilt from the same sources as `ba4ee78b6a77541e` (jar builds here are not
byte-reproducible: entry timestamps; no Java source changed between the two builds, only docs).

**Plan-versus-code check before starting (no drift found).** The tier's recorded state matched the code:
the region committed in `61c31cc`; `MatVec.sgemm` allocating its result batch and every handler's
`sgemmLayerInto` (and Phi-3's `sgemmProjInto`/`sgemmFusedInto`/fused Q/K/V and gate/up paths) copying it
into the workspace; `PrefillWindowRegion.Window` uploading x at every `runLayer` and downloading it at the end
of every layer. Tiers 00 and 01 are complete (all boxes ticked); the prompt that started this session
expected Tier 00 and was wrong about it, not the plan.

**What shipped.**
- `MatVec`: `sgemmInto(A, X, Y)` for all four batched overloads (host `float[]`, `DeviceFloatMatrix`,
  `DeviceHalfMatrix`, `DeviceQ4KMatrix`). Contract: writes `Y[b][0, rows)`, leaves the rest of the row alone,
  checks `Y` before computing (`SgemmOutput`, new), bit-identical to `sgemm`. The interface defaults copy from
  `sgemm`, so `RocmMatVec` (no override) stays correct; `CpuMatVec` and `CudaMatVec` implement it natively and
  their `sgemm` allocates and delegates, so the two cannot drift. This is the batched half Tier 10 item 4 names;
  its single-row `sgemvInto` is not in the contract (`CudaMatVec` has private ones for its small-batch
  branches). `GpuBlasOps.forwardInto` writes the FP32 path into the caller's rows; its packed host copies are
  still allocated (the FP32 device path is not used by any CUDA sweep model; Tier 10's allocation item).
- Handlers: `LlamaTransformerHandler.sgemmLayerInto`, `Qwen3TransformerHandler.sgemmLayerInto` and Phi-3's
  `sgemmProjInto`/`sgemmFusedInto` call `sgemmInto`; Phi-3's fused Q4_K Q/K/V and gate/up GEMMs write into a
  workspace buffer allocated once per call (it was one new batch per layer each) and are split from it.
- `PrefillWindowRegion.Window`: x is uploaded only when the device does not hold it (the window's first layer,
  or the first after a host-path layer) and is no longer downloaded per layer; `materializeResidual` is the
  boundary, called by the handlers at the end of the window and before any layer that runs on the host, inside a
  `juno.WindowStep` `device_layer` span. `recoverLayerInput` hands a failed layer's input back for the host
  redo: from the host rows when the layer uploaded them, from x before the layer's first in-place residual add,
  and otherwise from `xIn`, a device-to-device copy taken before that add (one W x hidden copy per layer, staging
  site `memcpy(prefill residual layer input D2D)`). The residual's copies are their own staging sites
  (`upload(prefill residual)`, `materialize(prefill residual)`; `ResidentChain.allocate(rows, dim, name)`).
- Docs: `docs/agent-arch.txt` (`MatVec`, `CudaMatVec`, `ResidentActivation`, `PrefillWindowRegion`),
  `docs/howto.md` ("Prefill windows on the device"), `docs/performance.md`, `CHANGELOG.md`.

**Tests, written first and shown failing for the right reason.**
- `MatVecSgemmIntoTest` (`node`, CPU, 9 cases): bit-identical to `sgemm` at B = 1, 2, 8, 9, 32, the caller's
  rows written in place, a wider row's tail untouched, a short/null/missing row rejected, the interface default
  correct for a backend that does not override it. Against the interface default, the allocation case failed:
  50,664 bytes for one call against a 49,152-byte output (bound: an eighth of it); passes on `CpuMatVec`'s own form.
- `CudaSgemmIntoTest` (`node`, `@Tag("gpu")`, 20 cases): FP16 and Q4_K at W = 1, 2, 8, 9, 32, 512 and 741 (the
  vision width), FP32 at 1, 2, 9, 32, 512, bit-identical to `sgemm`; at W = 512 the output is not allocated.
  Against the default the allocation case failed (537,960 bytes for a 524,288-byte output).
- `PrefillWindowRegionResidualTest` (`node`, `@Tag("gpu")`, 3 cases, synthetic layers): three layers with the
  residual kept on the device equal, bit for bit, the same layers with a download and upload after each, with 1
  residual upload and 1 download per window against 3 and 3; a device OOM after the layer's first residual add
  (a Q4_K gate whose dequantization scratch cannot grow, device filled to 2 MiB free) and one before it (a fused
  Q4_K Q/K/V) both recover the layer's input bit for bit, and the window then finishes the layer equal to a clean
  run. All three errored on the throwing skeleton.
- `PrefillRegionHandlerParityTest` extended: one residual upload and one download per region window. It failed
  before the change with one upload per layer (22, 36, 32 and 28 on TinyLlama, Qwen2.5-3B, Phi-3.5-mini and
  Qwen3-1.7B); passes after, logits still bit-identical region on against off on all four, at all six sites.

**Test-suite hygiene found on the way (no product change).** (1) The device-memory-returns-to-start checks in
`PrefillRegionHandlerParityTest` and `GpuAttentionHandlerParityTest` failed intermittently by 53 to 165 MiB.
The display server, desktop and two browser processes share this card (about 400 to 630 MiB between them,
moving during a run); the unchanged HEAD build failed `GpuAttentionHandlerParityTest` 1 of 3 runs alone, and this
tree passed `PrefillRegionHandlerParityTest` 3 of 3 with VRAM traced before and after each half (back to within
17 MiB, in both directions). The checks are left as they are; a failure of that size on this host should be
re-run before it is read as a leak. (2) `DeviceKvWindowAppendTest.windowGrowthOutOfMemory_isAVramOom` failed 3 of
3 alone on HEAD as well as here: its device-fill helper backed off in 16 MiB steps when the single allocation of
"free minus 4 MiB" was refused, leaving more than the 8 MiB the test expects to fail. The helper now fills in
halving chunks down to 1 MiB (as the new region test does); 3 of 3 pass.

**Verification (2026-10-02).** `mvn test` on the eleven unit-test modules: **1,968 tests, 0 failures, 0 errors,
49 skipped** (step 6: 1,936), 27:39, GPU-tagged tests included. `mvn install -DskipTests`, then `mvn verify -pl
juno-master`: 20 ITs, 0 failures. `-Pgpu`: `PrefillRegionGreedyIT` 2 of 2 (6 of 6 prompts identical over 64
tokens on TinyLlama and Mistral 7B), `GpuAttentionDivergenceIT` 3 of 3 with the same first-divergence steps as
step 6 (TinyLlama 8, 13, 23; Phi-3.5-mini 37, 40; Qwen3-1.7B 20, 23, 32), `GpuForwardPassIT` 5 of 5 on TinyLlama
against the CPU oracle (top-1 equal, top-5 overlap 5, logits relative L2 0.009; it needs `-Djuno.gpu.test=true
-Dit.model.path=...`, without which `-Pgpu` reports 0 tests, not a failure). `check-plan-thresholds.sh`,
`compare-llama-cpp.sh --selftest`, `prefill-breakdown.sh --selftest` (16 of 16) and `selftest-engine-stdin.sh`
pass. Not run, owed to the closing matrix (step 8) as before: `compare-lora.sh` (the LoRA handlers are separate
classes and do not use the region or `sgemmLayerInto`), `compare-vision.sh` (moondream2's text half is Phi-2,
CPU-only; the vision encoder still calls the allocating `sgemm`, which on CUDA now delegates to `sgemmInto`:
same kernels and bits, covered by `CudaSgemmIntoTest` at the vision width), `compare-prefill-batch.sh`.

**Measurement (agent, unpinned; the pinned gate is owed to the owner).**

*Breakdown*, published as [`20261002T050741Z`](../perf-compare/20261002T050741Z/INDEX.md) (`--gpu --device-spans
--juno-reps 3 --no-tuned-lane`, `n_prompt=512`, jar `caaf5810752a1ecb`; `prefill-breakdown.sh` exit 0, residue
1.3% to 2.9%; every prefill 512 of 512, every row scorable). Per window, all three repetitions identical:

| Model | Prefill ms (step 6 → now) | H2D + D2H MB (step 2 → step 6 → now) | Bytes against step 2 | Residual D2D MB | Matmul and region staging share |
|---|---|---|---|---|---|
| tinyllama-1.1b | 556 → 501 | 1,405 → 207 → **31** | **-97.8%** | 87 | 3.8% → 0.9% |
| qwen2.5-3b | 1,260 → 1,172 | 3,288 → 339 → **45** | **-98.6%** | 146 | 2.7% → 0.6% |
| Phi-3.5-mini | 3,121 → 3,012 | 2,645 → 1,809 → **1,419** | **-46.4%** | 194 | 3.7% → 2.6% |
| mistral-7b | 2,741 → 2,583 | 4,689 → 670 → **150** | **-96.8%** | 259 | 2.4% → 0.7% |

What remains crossing the bus on the region models is each layer's K and V rows for the host KV cache, plus
one upload and one download of the residual per window. The host side of the region
(`projection_and_region_host`) fell from 7.3% to 2.6% on TinyLlama and from 4.1% to 1.4% on Mistral 7B: the
per-layer residual copy-out and its host-array writes are gone. On Phi-3.5-mini, Q, K and V still come back and
the attention output still goes up every layer, because its LongRoPE and attention run on the host.

*Informational A/B* (not a gate: unpinned), alternating baseline and candidate three times each
(`PIN=0`, the script then in `target/tier01b-item2-ab/`, jars `d287e929f78c11ff` and `caaf5810752a1ecb`),
`n_prompt=512`, median (min to max), Juno t/s, logs under `target/perf-compare/item2-ab-unpinned-*` (since
deleted by `mvn clean`; the readings are reproduced here):

| Model | Prefill baseline | Prefill candidate | B/A | Generation B/A |
|---|---|---|---|---|
| TinyLlama | 916.23 (856.14 to 917.45) | 957.64 (935.55 to 1,014.33) | 1.045x | 1.007x |
| Mistral 7B | 190.32 (190.26 to 196.45) | 199.23 (198.25 to 200.67) | 1.047x | 1.001x |

Read against step 5's estimate for item 2 alone (1.16x and 1.15x, most of which the first change of step 6
already took by moving the attention input and output onto the device): the remainder was 2% to 4% of the
window in the post-region breakdown, and it moved prefill by about that.

**Item 2's threshold, scored as far as an unpinned agent can.** Residency across a layer's projections, with
the materialization boundary documented (`PrefillWindowRegion` and `ResidentActivation` javadoc,
`docs/agent-arch.txt`, `docs/howto.md`): **met**. Bytes per 512-token window `>= 70%` below step 2: **met on
tinyllama, qwen2.5-3b and mistral-7b (97% to 99%), missed on Phi-3.5-mini (46.4%, was 31.6%)**. `sgemmLayerInto`'s
per-matmul allocate-and-copy removed through the non-allocating form whose contract Tier 10 item 4 adopts:
**met** (Tier 10 item 4 now names this spelling and what it leaves for that item).

**Raised with the owner (2026-10-02): Phi-3.5-mini's bytes. Decided the same day: option 1.** Item 2's bytes
threshold is scored on tinyllama, qwen2.5-3b and mistral-7b, the three sweep models whose attention runs in the
region, and Phi-3.5-mini's prefill bytes moved to [Tier 02](TIER-02-attention-long-context.md) item 8 with its own
exit criterion (the same `>= 70%` against 2,645 MB). The criterion is ticked. The analysis as raised: Reaching 70% on Phi-3.5-mini needs its RoPE and
attention inside the region: LongRoPE on the device (its short factors folded into the inverse-frequency table
`CudaRope` takes) and the fused Q/K/V split on the device. No item in this tier owns either. Tier 02 item 8 already
owns the same LongRoPE folding for the decode region. Options: (1) score item 2's bytes threshold on the three
models whose handler runs attention in the region, and hand Phi-3.5-mini's prefill bytes to Tier 02 item 8
alongside its decode work (recommended: one LongRoPE kernel serves both, and Phi-3.5-mini's remaining host RoPE,
attention copies and attention host work are about 11% of its window, which the tier's milestones do not need:
the unpinned sweep reads its pp ratio at 0.142x against the 0.08x row); (2) pull the device LongRoPE and the
Phi-3 in-region attention into this tier as a new step; (3) keep the criterion open and close the tier
partial-complete on it.

**Owed, for the owner (pinned, needs prompt-free sudo):** this change's no-regression gate, same-hour, alternating
A B A B A B, generation `>= 0.95x` on TinyLlama and Mistral 7B, prefill recorded:
`bash dist/tier01b-item2-ab/run-gate.sh` (baseline `d287e929f78c11ff`, candidate `ba4ee78b6a77541e`; the six
runs land in `dist/tier01b-item2-ab/runs/`; it publishes nothing, so copy them under
`docs/perf-compare/<timestamp>-tier01b-step6-item2-ab/` as the step 6 gate was). It scores the decode
criterion below, not item 2's.

**Gate taken by the owner 2026-10-02 (pinned, 13:52 to 14:00 UTC): met.** Published:
[`20261002T135308Z-tier01b-step6-item2-ab`](../perf-compare/20261002T135308Z-tier01b-step6-item2-ab/INDEX.md).
All six runs pinned (governor performance, turbo off, GPU locked at 1860 MHz), jar hashes `d287e929f78c11ff` and
`ba4ee78b6a77541e` as expected, every prefill 512 of 512, every row scorable, no repetition withheld, GC pause at
most 16 ms (both sides alike). Median (min / max) of three:

| Model | Prefill A | Prefill B | B/A | Generation A | Generation B | B/A |
|---|---|---|---|---|---|---|
| TinyLlama | 959.54 (941.69 / 965.94) | 1029.42 (983.10 / 1030.55) | **1.073** | 67.91 (67.67 / 68.62) | 67.73 (66.98 / 68.22) | **0.997** |
| Mistral 7B | 194.63 (191.17 / 196.39) | 204.18 (200.47 / 207.49) | **1.049** | 22.89 (22.47 / 22.94) | 22.73 (22.69 / 22.88) | **0.993** |

Generation `>= 0.95x` met on both. Prefill moved by what the breakdowns predict: matmul and region staging plus
the region's host side were 11.1% and 6.5% of the window before and 3.5% and 2.1% after, worth about 1.08x and
1.05x. The decode criterion below stays open: it asks for every sweep model against the step-2 build, and for
vision, which the closing matrix (step 8) takes.

### 2026-10-02 — implementation step 6, item 3: the chunk-sizing review per surface

**Plan-versus-code check before starting.** The tier's recorded state matched the code (items 0, 1, 2, 6 to 9
done; HEAD `dec2b3e`; tree dirty only in `.gitignore` and the untracked `.github/`). This session's starting
prompt again expected Tier 00, which has been complete since 2026-09-24; the plan, not the prompt, was right.
Item 3's description held: adaptive sizing ran only in `ConsoleMain.runLocalRepl` on GPU plus `static`, and
every other `GenerationLoop` construction took the fixed 32. Four things it did not say:
- **The `JunoPlayer` facade was one of the fixed-32 surfaces**, although it builds the same in-process GPU
  pipeline as the local REPL (`LocalInferencePipeline`, three shards by default, shared `GpuContext`). It is
  a surface where the VRAM query is meaningful and the default was inherited, which is exactly the case
  the step 4 decision says turns item 3 from a review into an implementation item.
- **`docs/howto.md` said "local single-shard"**; the code applies the adaptive default at any `--nodes`.
- **`./juno cluster --prefill-batch N` failed with "Unknown cluster flag"** (`run.sh`, and `run.bat` likewise
  had no case for it), while `docs/howto.md` lists the flag for cluster. Not a silent no-op, but the
  documented surface did not exist; only `JUNO_PREFILL_BATCH` reached the cluster REPL.
- **Cross-surface row 7's premise is wrong: cluster prefill is not chunked at all.** Neither
  `ProcessPipelineClient` nor `TensorParallelPipelineClient` overrides `InferencePipeline.prefillBatch`, so a
  cluster prefill runs the interface default, one `forward` gRPC round trip per prompt token. Row 7 asks
  that the per-chunk cost "not simply be relocated into serialization"; there is no per-chunk cost on that
  path to relocate, and the batched prefill and the device region never reach a cluster node. Making
  cluster prefill batched is a gRPC contract change and is not in item 3's scope; **raised with the owner
  for a home** (Tier 09, tensor parallelism, or Tier 13, the cluster surface).

**Measurement** (agent, unpinned: these readings choose defaults and gate nothing), published as
[`20261002T194000Z-tier01b-item3-chunk-review`](../perf-compare/20261002T194000Z-tier01b-item3-chunk-review/INDEX.md).
TinyLlama Q4_K_M, a raw prompt calibrated to 512 prompt tokens, two discarded warm-ups then three measured
requests (`max_tokens=1`), median request wall time; jar `2267dbd7f9855ae5` (HEAD). Every spread under 5%.

| Surface | 32 | Wider | Wider over 32 | Default decided |
|---|---|---|---|---|
| GPU local, static, 1 shard | 1,417 ms | adaptive 549 ms | 2.58x | adaptive (unchanged) |
| GPU local, static, 3 in-process shards (the facade's shape) | 1,433 ms | adaptive 575 ms | 2.49x | **adaptive for the facade (was 32)** |
| CPU local, static | 90,433 ms | 128: 90,608; 512: 91,169 ms | 1.00x; 0.99x | 32: window width does not move CPU prefill |
| GPU `--lora-play` (the LoRA handler `juno lora` also runs) | 19,135 ms | adaptive 18,858 ms | 1.01x | 32 in `juno lora`: no gain, and no `GpuContext` there to size from |
| GPU cluster, pipeline | 20,291 ms | 512: 21,292 ms | 0.95x | 32: inert (per-token gRPC prefill, above) |
| GPU cluster, tensor | 28,792 ms | 512: 29,729 ms | 0.97x | 32: inert; the standalone coordinator builds the same two clients |
| GPU local, continuous, request alone | 2,235 ms | 128: 875; 512: 600 ms | 2.56x; 3.73x | see mixed load below |

Continuous under mixed load (`compare-mixed-prefill.sh --gpu --n-prompt 512`, three short streaming requests
beside the long prompt, two cold runs per value):

| Chunk | Short requests' mean TTFT | Short mean TPOT | Long prompt's TTFT |
|---|---|---|---|
| 32 | 898 / 904 ms | 196 / 197 ms | 3,539 / 3,544 ms |
| 128 | 1,004 / 1,033 ms | 202 / 202 ms | 1,978 / 2,000 ms |
| 512 | 1,336 / 1,383 ms | 124 / 126 ms | 1,478 / 1,533 ms |

32 is the setting that gives short requests their lowest TTFT, which is what the continuous schedule's
mixed stepping exists for, so 32 now has a measured reason there. 128 would shorten the long prompt's TTFT
1.78x for about 13% more short-request TTFT; at 512 interleaving disappears (mixed over admit-time 1.00x and
1.04x). Which trade the continuous schedule should make is a policy decision, **raised with the owner**; the
default stays 32 until then. Only TinyLlama was read; a larger model makes each chunk's step longer, so a
change of this default would be re-read on Mistral 7B first.

**What shipped.**
- `coordinator`: `PrefillChunkDefaults` (new): `Surface` (`LOCAL_REPL`, `EMBEDDED`, `CLUSTER_REPL`,
  `COORDINATOR`, `LORA_REPL`) and one `resolve` every `GenerationLoop` entry point now calls (both
  `ConsoleMain` cluster paths, its local and LoRA REPLs, `JunoPlayer`, `CoordinatorMain`); the javadoc carries
  each fixed default's measured reason. `PrefillBatchOptions.resolveAdaptiveFrom` takes the free-VRAM query
  as a supplier so the resolver is testable without a device.
- `JunoPlayer`: sized from free VRAM on GPU plus `static` (the only behaviour change).
- Launchers: `run.sh` and `run.bat` cluster accept `--prefill-batch` (and `JUNO_PREFILL_BATCH`), with help
  text; the local help's "(default 32)" corrected. Checked live: `./juno cluster --prefill-batch 64` reaches
  the engine's command line.
- Docs: `docs/howto.md` (the flag's row, the facade section), `docs/agent-arch.txt`, `docs/performance.md`,
  `CHANGELOG.md`.

**Tests, written first.** `PrefillChunkDefaultsTest` (`coordinator`, 15 cases): the default per surface on
GPU, CPU, static and continuous, a failed VRAM query, and an explicit value overriding on every surface,
backend and schedule (the plan's "assert the new default per surface, and assert that an explicit
`--prefill-batch N` still overrides on every surface"). Against a skeleton that reproduced the old
resolution, exactly one case failed, for the right reason (`embedded_static_gpu_sizes_from_free_vram`:
expected 16,384, got 32); 15 of 15 pass after. `PrefillBatchOptionsTest` 9 of 9.

**Verification (2026-10-02).** `mvn test` on the eleven unit-test modules: **1,983 tests, 0 failures, 0 errors,
49 skipped** (step 6 second change: 1,968), 27:24, GPU-tagged tests included. `mvn install -DskipTests`, then
`mvn verify -pl juno-master`: 20 ITs, 0 failures. `check-plan-thresholds.sh` passes. No pinned gate: the only
behaviour change is the facade's default, and its reading is the 2.49x above.

**Owner decisions (2026-10-02) on the two questions raised.**
1. *`continuous` default:* kept at 32 now; a load-dependent chunk (the whole remaining prompt when no
   request in the step is decoding, the step's token budget otherwise) is
   [Tier 07](TIER-07-continuous-batching.md) scope item 6, with thresholds (a prompt alone within 1.25x of
   `static`, 4.07x today; under mixed load short-request TTFT within 1.10x of fixed 32 and the long prompt's at
   most 0.75x of it). Rejected: a fixed 128 (one number for every model and load) and free-VRAM sizing for
   `continuous` (interleaving disappears).
2. *Batched cluster prefill:* [Tier 09](TIER-09-tensor-parallelism-multi-gpu.md) scope item 5, widened to both
   cluster modes and the standalone coordinator, with a contract change in `inference.proto` and a gate (512-token
   TinyLlama prefill <= 0.20x the per-token reading in each mode). Cross-surface row 7 above is corrected; rows 7
   and 8 of this tier's closing matrix read "correct, not reached: per-token prefill, Tier 09 item 5".

### Out-of-tier changes (recorded per execution rule 9)

| Change | What it touched | Measurement boundary? |
|---|---|---|
| Working tree, 2026-10-01 (scope item 8, carried here from Tier 02 item 7): Phi-3 LongRoPE factor selection | `Phi3RopeConfig.selectFactors()` returns the short factors when the file has them (it chose the long ones whenever the trained context exceeded the original, which is every Phi-3.5 request); `requirePosition` fails closed at `original_context_length` (4096 on Phi-3.5-mini); called from `Phi3Rope.buildCache`, so every CPU, GPU, batched-prefill and LoRA rotation. Tests: `Phi3RopeFactorPolicyTest`, `Phi3EndOfTurnLiveTest` (P(`<\|end\|>`) 0.5016 before, 0.9924 after). | **For Phi-3.5 output, yes; for throughput, no.** Rotation angles change, work does not, so every throughput reading stands. Phi-3.5 greedy output changes: step 3's `GpuAttentionDivergenceIT` Phi-3.5-mini row (4 of 6 identical, earliest divergence 26) was taken on the long factors; re-taken on this build at 4 of 6, earliest 37 (step 3a record). Any later Phi-3.5 divergence or greedy-parity reading is taken after this change. A Phi-3.5 sequence longer than 4096 tokens now fails with an error; no published run uses one (the sweeps prefill 128 and 512). |
| Working tree, 2026-09-28: architecture checked first at every entry point | `ModelFileGate.requireLoadable` (new, `node`) reads `general.architecture` and refuses an unverified one with `UnsupportedModelException` (new, an `IOException`, now also what `LlamaFamilyArchitectures` throws) before the config or tokenizer is read; called once in `ConsoleMain.main` before the mode dispatch (covers local, cluster, lora and their JFR variants), and in `CoordinatorMain`, `JunoPlayer.build` and `LoraTrainer.open`. The tokenizer's refusal becomes `UnsupportedPreTokenizerException` (new, still an `IllegalArgumentException`). `ConsoleMain` prints either as one `ERROR:` line and exits 1; cluster mode now refuses before forking nodes. Tests first: `ModelFileGateTest` (4 cases) and `BpePreTokenizerTest` tightened to the new type, both failing to compile before the change and passing after; `smoke-tier00-consistency.sh`, unmodified, 54 of 54 checks with the GPU legs (was 33 of 36 without them).; `mvn test` on the eleven unit-test modules passes (25:20 min), and `mvn verify -pl juno-master` passes, including `ThreeNodeClusterIT`, `TensorParallelClusterIT` and the unsupported-architecture IT. | **No.** One metadata read per model load, before any weights; nothing in the forward pass, MatVec, KV, batching or quantization. No published baseline is affected. |
| Working tree, 2026-09-27: `compare-llama-cpp.sh` heap and clock pinning | Juno launched with `-Xms` equal to `-Xmx` (was `-Xms512m`); optional `--pin-clocks`. | **Yes, for Juno readings from this harness**: a fixed-size heap changes when and how often G1 collects, and a pinned run runs at different clocks from an unpinned one (turbo off lowers absolute throughput for both engines). No published reference is invalidated by the code change itself, because none has been taken with it; the **step 2 re-baseline is the first run on this side of it** and every gate in this tier reads against that run, so no gate straddles the boundary. Do not compare a pinned run's absolute t/s with an unpinned run's. |

## Exit criteria

- [x] `--gpu-attention` defaults to on for every architecture whose gain was actually measured on a
      real model — Llama-family, Mistral, Qwen2, **Phi-2, Phi-3, Qwen3 and Qwen3-MoE**, since a file
      exists on disk for each of those last four (see "Models needed"; Qwen3-MoE is measured at
      partial offload and the offloaded layer count is recorded beside its figure). **No architecture
      on this list has a file gap to fall back on**, so none may keep its default resolved off for
      want of a measurement; the Phi-2 exception this criterion used to carry was written against a
      gap that had already closed. The ROCm answer decided and either implemented (`NEEDS-AMD-HARDWARE`) or
      made an explicit announced fallback. No path resolves to scalar silently, whichever way each one
      landed.
      *Amended 2026-09-30 (owner decision): the architectures that get the kernel are Llama-family,
      Mistral, Qwen2, Phi-3 and Qwen3. Phi-2 and Qwen3-MoE run on the CPU on every backend and announce
      it at startup through the capability mechanism; their GPU path is Tier 08 scope item 6. This and
      the next two criteria read "newly covered architecture" as Phi-3 and Qwen3.*
      *Restated 2026-09-30 (owner decision): "defaults to on" is read as behaviour, not as the flag's
      label. The default (`auto`) runs the kernel on every covered architecture on CUDA, and every
      case where it cannot run announces itself. The ROCm answer is decided: announced fallback,
      port in Tier 10 scope item 8. Ticked once the pinned run shows the gains the next criterion
      asks for.*
      *Checked 2026-09-30: the default runs the kernel on the LLaMA family, Phi-3 and Qwen3 on CUDA
      (pinned gains below); Phi-2, Qwen3-MoE, ROCm and an explicit `on` with the CPU backend each
      announce themselves (`GpuAttentionSupportTest`).*
- [x] Each newly covered architecture shows a measured prefill gain against its own pre-item-0
      baseline. "Unmeasured" is only acceptable for an architecture with no file on disk, and the
      claim is checked against `models/` at the time this tier runs rather than against this file's
      table — that table was wrong once already.
      *Checked 2026-09-30, pinned same-build A/B: Phi-3.5-mini 6.55x at 512 and 2.20x at 128, Qwen3-1.7B
      6.10x and 2.16x, spreads under 3%; decode 1.33x and 1.10x.*
- [x] Greedy-decode divergence from the FP16 KV mirror characterised per newly covered architecture
      and documented in `docs/howto.md` and `--help`, with `off` retained as the bit-identical
      CPU-parity baseline.
      *Checked 2026-09-30: `GpuAttentionDivergenceIT`, six prompts, 64 tokens. Phi-3.5-mini 4 of 6
      identical (earliest divergence at 26), Qwen3-1.7B 3 of 6 (earliest at 20), against TinyLlama's
      3 of 6 (earliest at 8); first token equal everywhere. Table in the step 3 record.*
- [x] LoRA training and `--lora-play` exemptions re-verified as still holding and still warning.
      *Checked 2026-09-30: holding, because every LoRA handler is its own class that calls the scalar
      `gqa` and never `GpuAttentionMirror` (all production loads go through
      `LoraTrainingHandlerFactory`). Warning: it was raised only for the LLaMA family, and is now raised
      by the factory for every LoRA architecture (`LoraGpuAttentionNoticeTest`, failing against an empty
      method first).*
- [x] `compare-llama-cpp.sh`'s stale default-lane `--gpu-attention` labelling corrected, and the
      first re-baselined run states which previously published lanes were mislabelled.
      `docs/performance.md`'s GPU-resident-attention section is corrected in the same pass: it states
      the default is **off** while `GpuAttentionOptions.fromEnv()` defaults to `auto`.
      *Checked 2026-09-30: harness labelling and the mislabelled-lanes statement landed with step 2
      (`20260930T135225Z/INDEX.md`); `docs/performance.md` corrected in step 3, as was the same stale
      label in `compare-prefill-batch.sh`.*
- [x] Item 0's own attributable measurement published separately from the rest of the tier's, so the
      default change's effect is visible on its own.
      *Checked 2026-09-30: `20260930T205554Z-tier01b-item0-ab` (flag A/B) and the default sweeps
      `20260930T211637Z` / `20260930T215547Z`, all pinned, before any other forward-pass change of this
      tier.*
- [x] `juno.DeviceStaging` and `juno.WeightDequant` exist, are enabled in
      `scripts/performance-tests/juno-perf.jfc`, are aggregated by `JfrMetricsExtractor`, and have
      `metrics` tests that failed on the pre-tier build for the right reason. Until this is checked,
      the breakdown criterion below cannot be satisfied by anyone.
      *Checked 2026-09-30, with one owner-approved change to its wording: the events are registered in
      `juno-perf.jfc` but **disabled** there, and enabled by the overlay `juno-perf-spans.jfc`
      (`compare-llama-cpp.sh --device-spans`), because counting every copy costs about 6% of TinyLlama
      prefill. Step 1's gate is met on that design (prefill 0.999x and 1.000x, pinned A/B). Every run
      that feeds the breakdown or the item-2 staged-bytes threshold passes `--device-spans`.*
- [x] `juno.DeviceCompute` exists (item 1a-ii), is registered disabled in `juno-perf.jfc` and enabled
      by `juno-perf-spans.jfc`, is extracted with keys on every run, has `metrics` tests that failed on
      the build without it, and passed its same-hour pinned A/B (prefill >= 0.98x with the spans off).
      *Checked 2026-10-01: prefill 0.994x (TinyLlama) and 1.016x (Mistral 7B), pinned,
      [`docs/perf-compare/20261001T172351Z-tier01b-step3a-compute-ab`](../perf-compare/20261001T172351Z-tier01b-step3a-compute-ab/INDEX.md);
      `JfrMetricsExtractorDeviceComputeTest` failed 4 of 5 on the extractor without the keys.*
- [x] Step 4a: `juno.WindowStep` exists, the Phi-3 and Qwen3 window paths emit the per-op events,
      `juno.MatVec` splits by phase, all extracted with keys on every run, with `metrics` and `node`
      tests that failed on the build without them; and its same-hour pinned A/B passed (prefill >= 0.98x
      with the device spans off) before the breakdown below is taken.
      *Amended 2026-10-01 (owner decision): the breakdown was taken first (step 4 record).*
      *Checked 2026-10-02: the gate is met, prefill 0.993x (TinyLlama) and 1.005x (Phi-3.5-mini),
      pinned, [`docs/perf-compare/20261001T235933Z-tier01b-step4a-ab`](../perf-compare/20261001T235933Z-tier01b-step4a-ab/INDEX.md);
      `JfrMetricsExtractorWindowStepTest` failed 5 of 5 and `PrefillWindowSpansTest` 3 of 3 on the build
      without the change (step 4a record).*
- [x] Per-term prefill breakdown published for all four sweep models at `n_prompt` 128 and 512, with
      no unattributed residue — every term named, including host-device staging and dequantization,
      the GEMM compute read from `juno.DeviceCompute`, the host FP16 packing, and the host elementwise
      spans; residue <= 5% of prefill per model. The GEMM's measured share per model is recorded for
      Tier 01C.
      *Checked 2026-10-01: residue 0.4% to 1.3% at 512 and 1.0% to 2.2% at 128,
      [`docs/perf-compare/20261001T224724Z`](../perf-compare/20261001T224724Z/prefill-breakdown.md) and
      [`docs/perf-compare/20261001T225929Z`](../perf-compare/20261001T225929Z/prefill-breakdown.md),
      unpinned by owner decision; GEMM share recorded in the step 4 record. Taken ahead of step 4a's
      pinned gate by owner decision (the step 4 record says why that order does not change the
      breakdown).*
- [x] Item 6: SwiGLU, both norms and the residual adds run on the device for the prefill window (and
      RoPE where the kernel supports the model's pairing, announced otherwise), the KV mirror is appended
      once per layer per window, and the item 6 threshold is met: host elementwise <= 10% of prefill and
      KV H2D copies <= 2 x layers per window on tinyllama, qwen2.5-3b and mistral-7b; tinyllama
      prefill >= 1.25x the pre-item-6 build (same-hour pinned A/B); greedy output identical or
      characterised no worse than item 0's baseline.
      *Checked 2026-10-02: host elementwise 0% and KV mirror copies 0 on tinyllama, qwen2.5-3b and mistral-7b
      ([`docs/perf-compare/20261002T014419Z`](../perf-compare/20261002T014419Z/INDEX.md)); TinyLlama prefill
      3.899x the pre-item-6 build, pinned same-hour A/B, generation 1.023x and 1.003x
      ([`docs/perf-compare/20261002T030500Z-tier01b-step6-region-ab`](../perf-compare/20261002T030500Z-tier01b-step6-region-ab/INDEX.md));
      greedy output identical over 64 tokens on 6 of 6 prompts on tinyllama and mistral-7b
      (`PrefillRegionGreedyIT`; logits bit-identical, step 6 record). RoPE on the device for the LLaMA family
      and Qwen2 (both pairings); Phi-3 and Qwen3 keep their own rotations on the host, announced in the
      region's startup log line.*
- [x] Item 7: `compare-llama-cpp.sh` runs Juno's hot path at the reference tool's `-t`, records
      `juno_threads`, and a CPU sweep taken with it is published as the new CPU reference, with the
      README's reference column moved in the same change.
      *Checked 2026-10-01: [`docs/perf-compare/20261001T180241Z`](../perf-compare/20261001T180241Z/INDEX.md),
      pinned, 12 threads against `-t 12`; live check at `--threads 4` found matmul samples on exactly four
      threads (step 3a record).*
- [x] Item 8: Phi-3.5 selects its RoPE factors by the owner-chosen policy on every `Phi3Rope` caller;
      the end-of-turn live test reads P(`<|end|>`) >= 0.95; recorded under "Out-of-tier changes", and
      Tier 02 item 7 reduced to the long-factor gap explanation.
      *Checked 2026-10-01: option (a) as a fixed 4096 cap (owner decision), in `Phi3RopeConfig`, reached by
      every rotation through `Phi3Rope.buildCache`.*
      **Evidence (not published):** `Phi3EndOfTurnLiveTest` (CPU, real file) 0.5016 before and 0.9924
      after; `Phi3RopeFactorPolicyTest` 2 of 5 failing before, 5 of 5 after; step 3a record.
- [x] Item 9: `check-plan-thresholds.sh` implements rule 7 check 4, was shown failing on a scratch copy
      with both planted defects, and passes on this tree without the check being weakened.
      *Checked 2026-10-01: both planted defects named, four controls passing; the tree passes with no
      criterion edited (step 3a record).*
- ~~Item 10: `.github/workflows/ci.yml` committed by the owner, and the first green run of both jobs
      recorded here with its URL.~~ **Not an exit criterion** (owner decision, 2026-10-01): see scope
      item 10. Written without a checkbox so it counts neither as met nor as open.
- [x] The threshold decomposition written down before implementation, with each item's expected
      contribution read off the breakdown, and an escalation recorded here if they did not sum.
      *Checked 2026-10-01 (step 5 record): rows 1 and 2 sum on item 6 alone (estimated 1.5x to 2.0x
      against 1.00x to 1.65x needed); row 3 (512 over 128) does not, because attention grows with
      context, so it moved to Tier 02 (owner decision, 2026-10-01).*
- [x] Prefill activations stay device-resident across a layer's projections, with the materialization
      boundary documented; bytes staged per 512-token window down >= 70% against the step-2 baseline;
      `sgemmLayerInto`'s per-matmul allocate-and-copy removed via the non-allocating batched form,
      whose contract matches the one Tier 10 item 4 adds for CPU.
      *2026-10-02, open: residency and the boundary are done in the device region (documented in its class
      and `docs/howto.md`); bytes down 85% to 90% on TinyLlama, Qwen2.5-3B and Mistral 7B but 31.6% on
      Phi-3.5-mini, whose LongRoPE and attention stay on the host; the non-allocating form is not done.*
      *2026-10-02, second change: the residual stays on the device across layers and `MatVec.sgemmInto` is in
      (step 6 second-change record). Bytes down 97.8%, 98.6% and 96.8% on TinyLlama, Qwen2.5-3B and Mistral 7B,
      46.4% on Phi-3.5-mini ([`docs/perf-compare/20261002T050741Z`](../perf-compare/20261002T050741Z/INDEX.md),
      unpinned). Still open on Phi-3.5-mini's bytes, with the owner's decision pending (options in that record);
      the change's pinned no-regression gate is owed.*
      *Checked 2026-10-02 (owner decision: option 1): bytes scored on tinyllama, qwen2.5-3b and mistral-7b,
      down 97.8%, 98.6% and 96.8% against `>= 70%`
      ([`docs/perf-compare/20261002T050741Z`](../perf-compare/20261002T050741Z/INDEX.md); bytes do not depend
      on clock state, identical in all three repetitions); Phi-3.5-mini's prefill bytes moved to Tier 02 item 8
      with the same threshold. Residency across layers and the materialization boundary documented
      (`PrefillWindowRegion`, `docs/agent-arch.txt`, `docs/howto.md`); `MatVec.sgemmInto` in, named in Tier 10
      item 4. The change's pinned no-regression gate belongs to the decode criterion below and is owed.*
- [x] Chunk-sizing defaults reviewed per surface; any surface still pinned at `32` has a measured
      reason, not an inherited one.
      *Checked 2026-10-02 (item 3 record): every `GenerationLoop` entry point resolves through
      `PrefillChunkDefaults`; the `JunoPlayer` facade moved to free-VRAM sizing on GPU plus `static`
      (2.49x on its pipeline shape), and CPU (1.00x), `continuous` (32 lowest short-request TTFT under mixed
      load), cluster and coordinator (per-token gRPC prefill, 0.95x and 0.97x) and `juno lora` (1.01x) keep 32
      on a reading
      ([`docs/perf-compare/20261002T194000Z-tier01b-item3-chunk-review`](../perf-compare/20261002T194000Z-tier01b-item3-chunk-review/INDEX.md),
      unpinned). The two questions it raised are decided (owner, 2026-10-02): the load-dependent `continuous`
      chunk is Tier 07 item 6, batched cluster prefill is Tier 09 item 5.*
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
- ~~Prefill throughput no longer degrades with prompt length: the `n_prompt=512` pp ratio is greater
      than or equal to the `n_prompt=128` pp ratio for every sweep model — **both measured under
      `RAW_PROMPT=1` with the same `--vector` setting**, which no published pair of runs has ever
      been. Step 2 establishes whether the degradation is real before this criterion can be scored.~~
      **Moved to [Tier 02](TIER-02-attention-long-context.md)** (owner decision, 2026-10-01; step 5
      record). Written without a checkbox, so it counts as neither met nor open here. The closing sweeps
      still publish the reading for Tier 02.
- [ ] Threshold above met, or the tier is explicitly marked partial-complete with the dominant term
      named and assigned to a successor tier (not silently marked complete).
- [ ] Decode (tg) and vision both verified not regressed, with published numbers.
- [ ] Cross-surface checklist fully resolved.
- [ ] Docs (`docs/howto.md` for any `--prefill-batch` default change, `docs/performance.md`,
      `docs/agent-arch.txt`) updated, Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
