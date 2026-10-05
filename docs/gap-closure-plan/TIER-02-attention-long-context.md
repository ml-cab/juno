# Tier 02: Attention & long context

Status: **in progress** (2026-10-05). Implementation steps 1 and 2 done: the plan check passes and the
2048-over-512 milestone is decomposed. Owner decisions taken the same day: the milestone restated as a
clock-normalised attention speedup at 2048 (`>= 6.0x`), Mistral 7B's non-attention growth attributed to the
card's thermal clock, and the comparison harness records the in-window GPU clock. Next: implementation step 3
(oracle tests for the scalar attention path). Open with the owner: whether the 512-over-128 row gets the same
restatement.

**Split 2026-10-04 (plan review): read this first.** This tier held nine items, and under execution
rule 1 every later tier, including every remaining GPU throughput lever, waited on all of them. Its
context-policy items, context shifting, sliding-window attention and the Phi-3.5 LongRoPE remainder,
have no throughput target and are prerequisites for [Tier 03](TIER-03-kv-cache-maturity.md)'s KV
layout, not for throughput. They moved to [Tier 02B](TIER-02B-context-policy.md), which runs directly
after this tier and before Tier 03. This tier keeps the attention kernel, the decode residency region,
graph replay, the residency default and the prompt-length milestones. Item numbers 2, 3 and 7 are kept
below as pointers so that references from the closed tiers' records still resolve.

## Objective

Replace the full-materialization GPU attention kernel with a tiled, online-softmax one that does not pay
O(seq²) memory at long context, so that prefill throughput no longer falls off with prompt length. Run
attention and then the whole decode layer inside the device residency region on the LLaMA family, Phi-3
and Qwen3. Decide CUDA graph replay and the `--gpu-residency` default by measurement.

## Why this tier, why now

This tier depends on Tier 01's activation-residency primitive: a tiled/online-softmax attention
kernel is exactly the kind of multi-step, GPU-resident-intermediate-state computation that pays off
once activations don't round-trip to host between steps. Doing this before Tier 01 would repeat the
same "measured regression, shelved as scaffolding" pattern the gap analysis flags in §2.8.

It runs first among the remaining tiers because prompt processing at length is now the largest GPU gap
in the program. At the current reference, GPU pp reads 0.226x to 0.468x at `n_prompt=512` and 0.099x to
0.180x at 2048, against tg at 0.545x to 0.627x; prefill falls to 0.33 to 0.44 of its 512-token ratio by
2048 tokens, and attention is the term that grows (Tier 01C's closing record).

**What Tier 01 handed over (2026-09-27).** Tier 01 shipped the decode residency region through RoPE:
`ResidentQkvPath` runs, per layer at single-sequence decode, the norm, the Q/K/V projections and RoPE on
the device with one download of q, k and v, behind `--gpu-residency` (default off), for +3.6% to +5.4%
decode where it runs. The attention half of that region ("step 3b") is here, not there, by the owner's
decision: after the region the layer still pays a host FP16 conversion and two synchronous copies for
the KV append, and four synchronous uploads plus one download for attention, and removing those is
attention work. `CudaGraphSession` came with it, because a captured graph only pays once a region
issues many launches per wait. Scope items 4 to 6 below carry all three.

The Phi-3.5 LongRoPE defect found on 2026-09-28 and its remainder are recorded in
[Tier 02B](TIER-02B-context-policy.md) item 3.

## Scope

### In scope

1. **Tiled/online-softmax attention kernel** (CUDA), replacing `gqa_attention.cu`'s current
   full-materialization design, built on Tier 01's residency primitive. Must remain numerically
   correct (verified against the existing scalar CPU attention path) while reducing peak memory at
   long sequence lengths. **It inherits whatever architecture coverage Tier 01B actually delivered —
   read that tier's exit table, do not assume it covered everything.** Tier 01B ships the kernel path
   for Phi-2, Phi-3, Qwen3 and Qwen3-MoE, and its exit criteria forbid leaving any of them with the
   default resolved off for want of a measurement: a real file exists for every one
   (`phi-2.Q4_K_M.gguf`, `Phi-3.5-mini-instruct-Q4_K_M.gguf`, `Qwen3-1.7B-Q4_K_M.gguf`, and
   `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` at partial offload; see [`INVENTORY.md`](INVENTORY.md)).
   So the expected starting state is default-on for Llama-family, Mistral, Qwen2, Phi-2, Phi-3, Qwen3
   and Qwen3-MoE. It can still be narrower if Tier 01B measured an architecture and found the kernel
   bought nothing or diverged unacceptably there; that tier's exit table is the record, and an earlier
   draft of this paragraph was wrong twice about which files existed (the two Qwen3 files, then the
   plain Phi-2), so read that table and check `models/` rather than trusting this paragraph.
   Whatever that set turns out to be, this rewrite must not shrink it — no architecture may regress
   to the scalar path as a side effect — and must re-run Tier 01B's per-architecture greedy-decode
   divergence characterisation for every member of it, since a tiled online-softmax accumulates in a
   different order and its divergence profile is not the one Tier 01B measured. The kernel must take a
   per-layer attention-window parameter (no window by default) so that [Tier 02B](TIER-02B-context-policy.md)
   item 2 adds sliding-window masking as a parameter value, not as a second kernel.
2. *Moved 2026-10-04 to [Tier 02B](TIER-02B-context-policy.md) item 1 (context shifting).*
3. *Moved 2026-10-04 to [Tier 02B](TIER-02B-context-policy.md) item 2 (sliding-window attention).*
4. **Attention inside the decode residency region (Tier 01's "step 3b").** Extend `ResidentQkvPath`
   (Tier 01) so that, per layer at single-sequence decode, k and v are appended to the device KV
   mirror (`DeviceKvCache`) without a host round trip and attention (`CudaGqaAttention`) reads q from
   the region, downloading only the attention output. Today, after the region, the layer still pays:
   FP16 conversion on the host plus two synchronous copies for the KV append, four synchronous uploads
   plus one download for attention. Keep the CPU KV tensors written (they are the fallback and the
   source of truth for the mirror); the order and failure handling of `DeviceKvCache.appendToken` and
   the written-prefix watermark (`c91f879`) must hold. Scope and fallbacks as in Tier 01:
   single-sequence decode, CUDA, K-quant device projections, adjacent RoPE, no Q/K/V bias; announce
   (log **and** console, as `GpuResidencyOptions.consoleNotice` does) everything else. Tests:
   bit-identity of the region against the op-at-a-time path; greedy parity on/off on tinyllama,
   mistral-7b, llama-1-30b; per-request device memory flat. Files: `ResidentQkvPath`,
   `GpuResidencyOptions`, `LlamaTransformerHandler`, `DeviceKvCache`, `CudaGqaAttention`; tests
   `ResidentQkvPathTest`, `LlamaTransformerHandlerGpuResidencyTest`; smoke
   `scripts/performance-tests/smoke-gpu-residency.sh`.
5. **`CudaGraphSession` (moved from Tier 01 by the owner on 2026-09-27).** After item 4 lands, measure
   whether capturing one layer's region as a CUDA graph and replaying it (position read from device
   memory, or updated per step via an exec-node parameter update) beats plain launches on the GTX 1080
   at decode width. **Decision rule**: wire it behind the same flag if it saves at least 5% of decode
   forward-pass time on tinyllama and mistral-7b with greedy output unchanged; otherwise delete
   `CudaGraphSession` and `CudaGraphSessionTest` and record the measurement. Either way the class stops
   being dormant in this tier.
6. **Default of `--gpu-residency`.** After item 4 (and 5 if wired), re-measure the region on vs off on
   all four sweep models (full `compare-llama-cpp.sh --gpu` sweeps, both published) and put the default
   to the owner with the numbers; do not change it unasked.
7. *Moved 2026-10-04 to [Tier 02B](TIER-02B-context-policy.md) item 3 (the Phi-3.5 LongRoPE remainder:
   option (b) and the long-factor gap).*
8. **The whole decode layer inside the region, on every handler with a device weight path** (added
   2026-09-30; the README's "Program objective: the layer runs on the device"). After item 4 the region
   covers norm, Q/K/V, RoPE, the KV append and attention, and the layer then leaves the device for the
   output projection, the residual add, the second norm and the FFN, each through today's
   op-at-a-time `MatVec` with its own upload and download. Extend the region through the rest of the
   layer: output projection (packed GEMV, as Q/K/V already are), both residual adds, the FFN norm, gate
   and up, SwiGLU (Tier 01B item 6's kernel at width 1) and down, so one decode layer is one upload of
   the residual row and one download of the layer output. Where the region also spans the next layer
   (the residual stream never leaving the device between layers), take it: the threshold below is the
   minimum, the objective is one crossing per forward pass.

   Then give the region to `Phi3TransformerHandler` and `Qwen3TransformerHandler`, the other two
   handlers with device weights, through the same `ResidentChain` and the same flag. That needs the
   split-half RoPE mode (Tier 01B item 6 builds it) and, for Phi-3, its LongRoPE factors folded into the
   inverse-frequency table the kernel already takes from the host, selected by the policy Tier 01B item
   8 fixed. Q/K/V biases (Qwen2) and Qwen3's per-head Q/K norms are the two operations the region does
   not have today; add them, or keep the model on the op-at-a-time path and announce it (log and
   console, as `GpuResidencyOptions.consoleNotice` does). This is the decode lever the 0.70x GPU tg
   end-of-plan target on Phi-3.5-mini depends on, and before this item no tier owned it.

   Order within the tier: item 4, then this item, then item 5 (graph replay pays more once the region
   issues more launches per wait), then item 6 (the default is put to the owner on the final region).

   *Added 2026-10-02 (owner decision, Tier 01B step 6 second change): Phi-3.5-mini's prefill bytes.* Tier
   01B's prefill-window region runs RoPE and attention inside it on the LLaMA family and Qwen2 only; on
   Phi-3 the region hands Q, K and V back to the host every layer and takes the attention output up again,
   so a 512-token Phi-3.5-mini window still moves 1,419 MB (46.4% below Tier 01B's step 2 baseline of
   2,645 MB, against the 70% that tier held its other three sweep models to; `docs/perf-compare/20261002T050741Z`).
   The LongRoPE folding this item builds for decode serves the prefill window as well. With it, move Phi-3's
   RoPE and attention into the prefill region too: split the fused Q/K/V on the device, rotate with the
   LongRoPE table, cast K and V into the mirror and run the attention kernel there, as the LLaMA family
   does. Host RoPE, attention copies and attention host work were 11.0% of that window. Do the same for
   Qwen3 if its per-head Q/K norm is added to the region (it is not a sweep model, so it has no bytes
   threshold); otherwise it stays announced.

   *Added 2026-10-04 (owner decision, Tier 01C step 5): the KV mirror's run-time growth on a nearly full
   card.* `DeviceScratchBudget` reserves the GPU-attention KV mirror only at its initial 64 positions, by
   design: growth to a long context cannot be reserved up front without pinning memory a short request
   never uses. Since Tier 01C shrank the upload reserve to a prefill window, a partially offloaded model
   reaches the growth with almost nothing free. On llama-1-30b (23 of 60 layers on the GPU, 197 MiB kept
   free) a 508-token prompt's mirror needs about 311 MiB, its growth fails, and attention falls back to
   the CPU for the request (the pre-change build fell back too, in region attention;
   `docs/perf-compare/20261003T235146Z-packed-kquant-reserve`). This tier owns long-context attention and
   its memory: decide how mirror growth is budgeted against weight layers on a card the model does not fit
   (for example, reserving for a stated context, or trading a layer for mirror capacity when a long prompt
   arrives), and either keep a 512-token prompt's attention on the device for llama-1-30b at `auto` or
   state why not with the number.

### Out of scope

- Context shifting, sliding-window attention and the Phi-3.5 LongRoPE remainder — [Tier 02B](TIER-02B-context-policy.md).
- Extending the tiled attention kernel to ROCm (tracked in Tier 10, same reasoning as Tier 01).
- Vision attention (`VisionEncoder`'s own CLIP-style attention) — vision has very different batch
  shapes (~741-wide) and is handled in Tier 11.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | scalar attention stays the correctness oracle; this tier changes no CPU path |
| 2 | CUDA GPU inference | primary target; the tiled kernel is CUDA-only this tier |
| 3 | ROCm GPU inference | FAIL-CLOSED or NEEDS-AMD-HARDWARE for the tiled kernel; the startup notice that ROCm runs attention on the CPU stays |
| 4 | Static schedule | prefill windows and decode both run the new kernel; verify at every `--parallel` width the existing tests cover |
| 5 | Continuous schedule | mixed prefill/decode steps reach the kernel at chunk widths; verify `ContinuousPrefillState` bookkeeping is unaffected and output matches static |
| 6 | Single-node local mode | primary dev surface |
| 7 | Pipeline-parallel cluster | each node runs the kernel on its own layers; greedy output matches local mode |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | confirm training's own attention/backward path (`LoraTrainingMath`) is unaffected — training doesn't currently use the fused GQA kernel |
| 10 | LoRA playback | confirm LoRA-modified attention projections still compose correctly with the new tiled kernel |
| 11 | Vision | N/A — out of scope, must verify vision's separate `VisionEncoder` attention path is untouched |
| 12 | OpenAI REST surface | N/A directly; verified through TTFT and the smoke runs |
| 13 | Native REST surface | same |
| 14 | CLI | `--gpu-residency` default may change on the owner's decision (item 6); `--help` and `docs/howto.md` follow it |
| 15 | JVM embedding facade | N/A: no new embedder-invocable capability (the context-shift opt-in that carried this row is Tier 02B's) |

## Implementation steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh` first.
2. **Decompose the 2048-over-512 milestone before building the kernel** (added 2026-10-04, plan review;
   the same discipline as Tiers 06 and 10). From one `compare-llama-cpp.sh --gpu --device-spans` run at
   `n_prompt=2048` and one at 512 on the pre-tier build, record per sweep model: attention's share of the
   window (kernel, copies and host part, as Tier 01B's step 4 breakdown read them), the non-attention time
   per prefilled token at both lengths, and the attention speedup at 2048 that the `>= 0.90` milestone
   implies with everything else held fixed. If that required speedup exceeds **4x** on any sweep model,
   or the non-attention time per token itself grows from 512 to 2048 by more than 10% (meaning attention
   is not the only term that scales), write the numbers into this file and escalate to the owner before
   building the kernel. A milestone the decomposition says one kernel cannot reach is restated or
   re-owned there, not discovered at the closing sweep.
3. Write correctness tests for the existing scalar attention path as the oracle (if not already
   fully covered) before touching the kernel.
4. Design and implement the tiled/online-softmax kernel using Tier 01's residency primitive;
   validate numerically against the oracle at multiple sequence lengths, including lengths that
   would have overflowed the old kernel's scratch buffer.
5. Scope items 4, 8, 5 and 6, in that order: attention inside the decode residency region, then the
   rest of the layer and the Phi-3 and Qwen3 handlers (item 8), then the `CudaGraphSession`
   measurement and its wire-or-delete decision, then the two on/off sweeps and the flag default put to
   the owner. Item 4 is independent of the tiled kernel (item 1) at decode width
   and may land before it; if it does, item 1 must keep the region's bit-identity test passing.
6. Run the full cross-surface smoke matrix, including `smoke-long-prompt-prefill.sh` at 2048 tokens.

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other
  test or code in this tier (README execution rule 7).
- **New unit tests** for the tiled kernel: exact-match (within float tolerance) against the scalar
  CPU oracle at short, medium, and long (near/at `MAX_SEQ_LEN`) sequence lengths, and with the
  per-layer window parameter set to "no window" bit-identical to the unwindowed call.
- **`ModelLiveRunnerIT`**: the existing 512-token prefill check extended to 2048 tokens on both
  schedules, greedy output identical to the per-token prefill over the first 64 generated tokens.
- **Standing CPU and allocation gate** (README, "Test infrastructure"): run against the pre-tier jar
  and score it before closing this tier.
- **Perf gate (required)**: the tiled kernel is a hot-path change — `compare-lora.sh` plus a
  dedicated long-context latency/memory microbenchmark, plus `compare-llama-cpp.sh` for a
  llama.cpp-relative pp/tg reading on the same models (per README's llama.cpp-relative gate); publish
  under `docs/perf-compare/`.

  **Threshold.** Measure the old full-materialization kernel's peak attention scratch first, then
  hold the tiled kernel to both of these:
  - peak attention scratch at the longest tested sequence length drops by **>= 80%**;
  - peak attention scratch scales sub-quadratically — doubling the sequence length must **less than
    double** it, which is the property the rewrite exists to buy and the one a percentage alone does
    not capture.

  **Milestone (README milestone table; restated 2026-10-05, owner decision 1).** Attention at
  `n_prompt=2048` is **>= 6.0x** faster than on the pre-tier build on every sweep model, clock-normalised:
  attention time per prefill window (kernel, copies and host part, as `prefill-breakdown.sh` reads them) times
  the in-window median SM clock that `compare-llama-cpp.sh` records, pre-tier over candidate. Read from a
  same-session A/B alternating the pre-tier jar and the candidate (A B A B A B), each invocation
  `compare-llama-cpp.sh --gpu --device-spans --n-prompt 2048 --no-tuned-lane --juno-jar <jar> --juno-reps 1
  --juno-warmup 2 --reps 1 --no-publish` (Phi-3.5-mini with `COMPARE_HEAP=8g`), median of three per side, all
  six readings published. Pinning is not required: the threshold is far outside the 15% noise floor, and
  `--pin-clocks` cannot hold this card's GPU clock, which the normalisation accounts for instead. Step 2's
  pre-tier readings: tinyllama 4,459 ms, qwen2.5-3b 7,417 ms, Phi-3.5-mini 19,900 ms, mistral-7b 19,253 ms
  (`docs/perf-compare/20261005T004009Z`, `20261005T005334Z`). The GPU pp ratio at 2048 over the ratio at 512
  is still published from the closing sweeps, as a reading, not a gate.
  *Was, until 2026-10-05:* the GPU pp ratio at `n_prompt=2048` over the ratio at `n_prompt=512` **>= 0.90** on
  every sweep model (reference 0.329, mistral-7b binding). Restated because a ratio of ratios does not score
  the kernel: the kernel also speeds 512, so a 10x kernel would read 0.69 on mistral-7b while lifting its pp
  ratio at 2048 from 0.14x to about 0.57x (step 2 record, decision 1; README "Why the 2048-over-512 row became
  an attention-speedup row").
  *Corrected 2026-10-01:* this paragraph said Tier 01B had established that prefill does not fall off
  between 128 and 512 once attention is on the GPU. Tier 01B's step 4 breakdown showed that it does, on
  three of the four sweep models, and that attention is the term responsible.

  **Milestone moved here from Tier 01B (2026-10-01, owner decision).** The GPU pp ratio at
  `n_prompt=512` over the ratio at `n_prompt=128` is **>= 1.00** on every sweep model, from the same
  closing sweeps at 128 and 512. Reference: 0.606 (tinyllama, binding) to 0.743 (Phi-3.5-mini), from
  Tier 01C's closing sweeps `20261004T113210Z`/`20261004T114812Z`; it fell from 0.845 at Tier 01B's close
  because the packed matmul removed fixed per-token cost and left attention a larger share. The attention
  numbers this tier's kernel is held to come from Tier 01B's step 4 breakdown
  (`docs/perf-compare/20261001T224724Z/` and `20261001T225929Z/`, unpinned, with spans on) and are
  re-read by implementation step 2 on the current build. Item 0 of Tier 01B, which put the kernel on
  Phi-3 and Qwen3, is not credited here.

  **End-of-plan targets.** Record the GPU tg ratio on Phi-3.5-mini and mistral-7b after items 4 to 6
  against the 0.70x end-of-plan targets, and the GPU pp ratios at 512 and 2048 against the per-length
  end-of-plan rows (README, "End-of-plan targets"). Items 4 and 5 are the plan's main decode levers, so
  state per model how far they moved tg and, if 0.70x is out of reach on what is left in the plan, say
  which mechanism is missing and that no tier owns it.

  Throughput must not regress: Juno tg and pp t/s **>= 0.95x** the pre-tier build on every sweep
  model, from a same-hour interleaved A/B with pinned clocks against the pre-tier build (README, "No-regression gates tighter than the floor are Juno-against-Juno"). The llama.cpp-relative ratios are recorded against the program target, not
  gated.
- **Attention inside the residency region (scope item 4)**: extend `ResidentQkvPathTest` with a
  bit-identity case of the region (now through the KV append and attention) against the op-at-a-time
  path at several positions and across concurrent threads, a case that the CPU KV tensors and the
  device mirror's written-prefix watermark agree after every appended token, and the no-retained-device-
  memory and short-lived-thread pooling cases; extend `LlamaTransformerHandlerGpuResidencyTest` for
  greedy parity on/off; re-run `scripts/performance-tests/smoke-gpu-residency.sh` on tinyllama,
  mistral-7b and llama-1-30b (greedy output identical on/off, per-request device memory flat).
  **Threshold**: end-to-end tg with the region on **>= 1.0x** the region-off run on every model where
  the region runs, and **>= 0.95x** everywhere — read from a same-hour A/B alternating the flag
  (region off, on, off, on, off, on) with pinned clocks, per the README's no-regression rule.
- **`CudaGraphSession` (scope item 5)**: a decode-width microbenchmark of one layer's region, graph
  replay against plain launches, plus greedy parity with replay on. **Threshold**: the decision rule in
  scope item 5 (at least 5% of decode forward-pass time on tinyllama and mistral-7b with greedy output
  unchanged, or delete the class and its test).
- **The whole decode layer (scope item 8)**: extend `ResidentQkvPathTest` (or its successor) with
  bit-identity of the full-layer region against the op-at-a-time path at several positions and across
  concurrent threads, per handler; greedy parity on/off on tinyllama, mistral-7b, Phi-3.5-mini and
  Qwen3-1.7B; per-request device memory flat in `smoke-gpu-residency.sh`, which gains the two new
  handlers. A `@Tag("gpu")` test reads the decode-phase copy counts off `juno.DeviceStaging` for one
  generated token and asserts the threshold below, so the property is held by a test and not only by a
  sweep.
  **Threshold** (from a `--device-spans` run, decode phase, per generated token, on every model where
  the region runs): host-to-device copies **<= 1 x layers** and device-to-host copies **<= 1 x layers
  + 1** (the `+ 1` is the logits); the end-to-end tg with the region on **>= 1.0x** region-off from a
  same-hour pinned A/B alternating the flag; greedy output identical on/off. On a prefill window, the
  same run shows **0** activation device-to-host copies inside the window apart from the window's
  final hidden rows and logits, which Tier 01B items 2 and 6 deliver and this tier re-verifies after
  its own kernel changes. Record the GPU tg ratios against the 0.70x end-of-plan targets.
- **Flag default (scope item 6)**: two published `compare-llama-cpp.sh --gpu` sweeps on all four sweep
  models, `--gpu-residency on` and `off`, read side by side.

## Models needed

Existing dense models (`tinyllama`, `mistral-7b`, `qwen2.5-3b`, `Phi-3.5-mini`) cover the kernel and
region work; `Qwen3-1.7B` covers item 8's Qwen3 handler and `llama-1-30b` the partial-offload mirror
budget. No download is needed.

## Exit criteria

- [x] The 2048-over-512 milestone decomposed before the kernel was built (implementation step 2), with
      the escalation recorded if the required attention speedup exceeded 4x on any sweep model.
      *2026-10-05:* decomposed from `docs/perf-compare/20261005T003146Z` (512) and `20261005T004009Z`/
      `20261005T005334Z` (2048); [`milestone-decomposition.md`](../perf-compare/20261005T003146Z/milestone-decomposition.md).
      Escalation recorded below (Mistral 7B: 4.86x required, non-attention +19.2% per token); the owner's
      decision is open.
- [ ] Tiled attention kernel numerically matches the CPU oracle at all tested sequence lengths and
      reduces peak GPU memory at long context vs. the old full-materialization kernel (measured), and
      takes the per-layer window parameter Tier 02B needs.
- [ ] Attention inside the decode residency region (scope item 4): k and v appended to the device
      mirror and attention read from the region, one download of the attention output per layer;
      bit-identical to the op-at-a-time path by test; greedy output identical on/off on tinyllama,
      mistral-7b and llama-1-30b; per-request device memory flat in `smoke-gpu-residency.sh`.
      **Threshold**: end-to-end tg with the region on >= 1.0x the region-off run on every model where
      it runs, and >= 0.95x everywhere.
- [ ] The whole decode layer runs inside the region (scope item 8) on the LLaMA family, Phi-3 and
      Qwen3, or the handler or operation that cannot is announced: decode-phase copies per generated
      token <= 1 x layers host-to-device and <= 1 x layers + 1 device-to-host, read off
      `juno.DeviceStaging` by test and by a published `--device-spans` run; tg region-on >= 1.0x
      region-off (same-hour pinned A/B); greedy output identical; per-request device memory flat; zero
      activation device-to-host copies inside a prefill window re-verified.
- [ ] Phi-3.5-mini's prefill window runs RoPE and attention inside the prefill region (scope item 8,
      added 2026-10-02): **Threshold: H2D + D2H bytes per 512-token window >= 70% below 2,645 MB**
      (Tier 01B's step 2 baseline; 1,419 MB at `docs/perf-compare/20261002T050741Z`), from a published
      `--device-spans` run; logits bit-identical region on against off (`PrefillRegionHandlerParityTest`)
      or characterised no worse than the existing GPU-attention divergence (Phi-3.5-mini earliest 37).
- [ ] KV mirror growth on a partially offloaded model budgeted (scope item 8, 2026-10-04 addition): a
      512-token prompt's attention stays on the device for llama-1-30b at `auto`, or the reason is
      stated with the number.
- [ ] `CudaGraphSession` decided by measurement (scope item 5): wired behind `--gpu-residency` if it
      saves at least 5% of decode forward-pass time on tinyllama and mistral-7b with greedy output
      unchanged, otherwise deleted with `CudaGraphSessionTest` and the measurement recorded. Not
      dormant either way.
- [ ] `--gpu-residency` default put to the owner (scope item 6) with the two published on/off sweeps
      on all four sweep models; changed only on the owner's decision.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published, both memory thresholds above met, Juno t/s >= 0.95x the pre-tier build, and
      the standing CPU and allocation gate met.
- [ ] Milestone (pp ratio at 512 over 128 >= 1.00 on every sweep model, moved here from Tier 01B on
      2026-10-01) reported met or missed with its number, and the attention share of a 512-token window
      per model read against implementation step 2's figures.
- [ ] Milestone (attention at 2048 >= 6.0x faster than the pre-tier build, clock-normalised, on every
      sweep model; restated 2026-10-05 from pp ratio at 2048 over 512 >= 0.90) reported met or missed with
      its number, and 2048 over 512 published as a reading; GPU tg ratios recorded against the 0.70x end-of-plan targets and GPU pp ratios against
      the per-length end-of-plan rows, with the missing mechanism named if they are out of reach.
- [ ] Docs (`docs/howto.md`, `docs/agent-arch.txt`, `docs/performance.md`) updated, Juno-native
      language only.
- [ ] `CHANGELOG.md` entry added.

## Execution record

### 2026-10-05: implementation steps 1 and 2 (plan check; milestone decomposition), and an escalation

**Step 1.** `scripts/performance-tests/check-plan-thresholds.sh` passes (21 tier files, milestone and
end-of-plan tables checked).

**Step 2: pre-tier build and method.** HEAD `05a17de`, clean tree; its engine code is identical to `127c9a0`
(only plan files and the plan check changed since), so it is the pre-tier build. Jar `3306ea4261f84a49`,
built with `mvn -q package -DskipTests`. Three `compare-llama-cpp.sh --gpu --device-spans --no-tuned-lane`
runs, published, unpinned (a breakdown reports shares within one run, as Tier 01B's step 4 did):
`docs/perf-compare/20261005T003146Z` (512, four models), `20261005T004009Z` (2048, TinyLlama, Qwen2.5-3B,
Mistral 7B at their fixed heaps) and `20261005T005334Z` (2048, Phi-3.5-mini, `COMPARE_HEAP=8g` as in Tier
01C's reference). `prefill-breakdown.sh` on each: residue 1.9% to 4.0% at 512, 0.5% to 1.0% at 2048, inside
its 5% bound. Every row scorable, prompt tokens 512/512 and 2048/2048. Attention is kernel plus copies plus
host part (`region_attention_compute`, `attention_compute`, `attention_copies`, `attention_host`); on the
LLaMA-family models it runs inside the prefill region, whose host dispatch stays in the region's host term
(0.4% to 3.4%). The arithmetic and how to read each column are in
[`milestone-decomposition.md`](../perf-compare/20261005T003146Z/milestone-decomposition.md).

| Model | Prefill ms 512 / 2048 | Attention ms 512 / 2048 | Attention share 512 / 2048 | Non-attention ms per token 512 / 2048 (growth) | pp ratio 512 / 2048 (2048 over 512) | Attention speedup at 2048 needed, 512 held | Speedup needed at both lengths | 2048 over 512 with attention at zero |
|---|---|---|---|---|---|---|---|---|
| Phi-3.5-mini-instruct-Q4_K_M | 2072 / 24334 | 943 / 19900 | 45.5% / 81.8% | 2.206 / 2.165 (-1.9%) | 0.200x / 0.091x (0.452) | 2.56x | 6.44x | 1.352 |
| mistral-7b-instruct-v0.1-q4_k_m | 1591 / 23226 | 757 / 19253 | 47.6% / 82.9% | 1.628 / 1.940 (+19.2%) | 0.468x / 0.144x (0.307) | 4.86x | 84.35x | 0.942 |
| qwen2.5-3b-instruct-q4_k_m | 765 / 8933 | 408 / 7417 | 53.3% / 83.0% | 0.697 / 0.740 (+6.1%) | 0.415x / 0.169x (0.407) | 2.94x | 14.24x | 1.119 |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 400 / 5060 | 248 / 4459 | 62.1% / 88.1% | 0.296 / 0.294 (-0.7%) | 0.330x / 0.125x (0.378) | 2.93x | 15.48x | 1.204 |

The 2048-over-512 readings here (0.307 to 0.452) agree with the pinned reference (0.329 to 0.441) within the
noise floor.

**Findings.**

- **Attention is now most of the prefill window at both lengths.** 45.5% to 62.1% of a 512-token window
  (11% to 17% at Tier 01B's step 4, before the packed GEMM removed most of everything else) and 81.8% to
  88.1% of a 2048-token one. Per prefilled token it grows 4.5x (TinyLlama) to 6.4x (Mistral 7B) from 512 to
  2048, so the kernel's cost per call grows 18x to 25x for 4x the length.
- **Trigger 1 fired: the required attention speedup exceeds 4x on Mistral 7B** (4.86x with everything else,
  the 512 reading included, held fixed; 2.56x to 2.94x on the other three).
- **Trigger 2 fired: Mistral 7B's non-attention time per token grows 19.2%** from 512 to 2048 (limit 10%;
  Qwen2.5-3B +6.1%, TinyLlama -0.7%, Phi-3.5-mini -1.9%). Both lengths run one window with the same launch
  counts (224 packed GEMMs, 32 attention calls), yet GEMM compute grows 4.98x for 4x the rows and each
  elementwise kernel 4x to 6x. Not attributed: a GPU clock that sags over a 23-second window (unpinned; the
  card refuses clock locking) and the GEMM's behaviour at 2048 rows both fit. *Attributed 2026-10-05 to the card's
  thermal clock; see the decision 2 record below.* With attention at zero at both
  lengths Mistral 7B's milestone ratio could reach only 0.942.
- **"Everything else held fixed" understates the ask.** The faster kernel also runs at 512 and raises the
  512 ratio the milestone divides by. With the same attention speedup at both lengths, 0.90 needs 6.4x
  (Phi-3.5-mini), 14.2x (Qwen2.5-3B), 15.5x (TinyLlama) and 84x (Mistral 7B). The reference tool keeps 75%
  to 89% of its prefill t/s from 512 to 2048 and Juno 27% to 34%; the milestone asks for at least 0.90 of the
  reference tool's retention, which an exact-attention kernel can approach only by also matching the
  reference tool's attention cost per token, not just its scaling.
- **Not decomposed by this step, flagged:** the 512-over-128 row (>= 1.00) depends on the same term. At
  45.5% to 62.1% attention at 512 it carries the same both-lengths effect between 128 and 512.

**Raised with the owner (2026-10-05). Decided the same day: decision 1 (b), with the 6.0x clock-normalised
threshold, and decision 2 (a); see the records below.**

- *Decision 1: the 2048-over-512 milestone.* (a) Keep `>= 0.90` and build the kernel, reporting a probable
  miss on Mistral 7B. (b) Restate the row as something the kernel owns: attention milliseconds per
  prefilled token at 2048 reduced by a stated factor against this decomposition on every sweep model
  (Juno-against-Juno), with 2048 over 512 reported but not gated. (c) Keep the number but re-own it to Tier 14
  as a reported distance. Recommended: (b). The decomposition says no one kernel delivers the ratio as
  written, and the end-of-plan row (pp at 2048 `>= 0.25x`) already scores the absolute outcome.
- *Decision 2: Mistral 7B's non-attention growth.* (a) Attribute it before the kernel: re-run Mistral 7B at
  512 and 2048 with `--device-spans` while sampling `nvidia-smi` clocks during the window (no harness
  change, no sudo). (b) Leave it and let the closing sweep show it. Recommended: (a). It caps Mistral 7B's
  milestone at 0.942 whatever attention does, and it is cheap to read.

**Out-of-tier changes.** None.

### 2026-10-05: decision 2 taken (a); Mistral 7B's non-attention growth is the GPU's thermal clock

Owner decision: attribute it before the kernel. Mistral 7B re-run at 512 and 2048 with `--device-spans`
(`docs/perf-compare/20261005T012110Z`, `20261005T012348Z`; jar `3306ea4261f84a49`, unpinned), with
`nvidia-smi` sampling the SM clock, temperature, power and clock event reasons every 100 ms. Each measured
prefill window (its `juno.PrefillBatch` event) joined to the samples inside it
([`gpu-clocks.md`](../perf-compare/20261005T012348Z/gpu-clocks.md)):

| Length | SM clock in window (median) | Temperature | Power | Clock event reasons | GEMM ms per row (median) |
|---|---|---|---|---|---|
| 512 | about 1845 MHz (1809 to 1860) | 52 to 67 C | about 160 W | none, or software power cap (`0x4`) | 1.199 |
| 2048 | 1607 MHz (low 1354) | 79 to 93 C | about 110 W | software thermal slowdown (`0x20`) | 1.361 |

GEMM time per row rises 13.3% against a 14.8% clock drop: clock-normalised it is flat (within 1.3%), so the
growth is the 21-second window heating the card, not the packed GEMM at 2048 rows. On this pair the
non-attention time per token grows 9.3% (under the 10% trigger; about -5% clock-normalised). The attention
speedup the milestone needs with everything else held fixed reads 4.98x, so trigger 1 still fires; decision 1
stays open.

Consequences recorded for later steps:
- The thermal clock inflates every 2048 reading of Juno, attention included (about 15% on Mistral 7B). The
  reference tool prefills 2048 tokens in about 3.3 s and does not heat the card the same way, so part of the
  2048 gap is a duration effect that a faster kernel removes on its own.
- `--pin-clocks` cannot hold this card's clock (the driver refuses locking), and the harness records GPU
  clocks only at idle, before the run. A 2048 gate reading on this host therefore carries an unrecorded clock.
  Raised with the owner as decision 3: (a) have `compare-llama-cpp.sh` sample the GPU clock during each
  measured window and publish the in-window median and the throttle reasons beside every result (a harness
  change, recorded here as this tier's measurement infrastructure); (b) leave it and read 2048 gates with this
  caveat. Recommended: (a).

### 2026-10-05: decision 1 taken (b), decision 3 taken (a)

**Decision 1 (b).** The 2048-over-512 milestone is restated as attention at `n_prompt=2048` **>= 6.0x**
faster than the pre-tier build on every sweep model, clock-normalised, from a same-session A/B (the milestone
paragraph under "Tests to write/upgrade before implementation", the exit criterion, and the README milestone
table and its "Why the 2048-over-512 row became an attention-speedup row" paragraph). The 2048-over-512 ratio
stays published as a reading. Not decided: whether the 512-over-128 row (`>= 1.00`), which has the same
ratio-of-ratios form, gets the same treatment; it stays as written until the owner decides.

**Decision 3 (a), landed the same day.** `compare-llama-cpp.sh` runs `nvidia-smi` every 100 ms while each
measured request is in flight (GPU runs only) and records `gpu_clock_in_window` in every per-repetition
result: median, lowest and highest SM clock over samples the driver does not flag idle, hottest sample,
median power and the clock event reasons seen. The aggregate carries the median of the per-rep medians, the
lowest minimum, the hottest sample and every reason, per lane (`lanes.prefill`, `lanes.generate`), and
`INDEX.md` lists them per model; the raw samples are kept as `<rep>-gpu-clocks.csv`. Measurement
infrastructure for this tier's gates (the attention-speedup milestone is clock-normalised from these
figures), so not an out-of-tier change. Docs: `docs/howto.md` (the device-span section), `CHANGELOG.md`.

Tests, written first: 18 new `compare-llama-cpp.sh --selftest` checks (summary of a sample file, idle and
missing files, rep aggregation, lane merge). Seen failing first for the right reason (the summary function
did not exist, then the aggregate did not carry the clock). One design change came from a real run: busy
was first "utilization >= 50%", and an end-to-end TinyLlama run read no clock for the generate lane,
because utilization is a trailing average (1% to 37% through a one-second decode at a full 1885 MHz). Busy
is now "not flagged idle by the driver" (`gpu_idle` in the event reason mask), with a check for it, seen
failing first. Full selftest: 114 checks, all pass.

| Check | Result |
|---|---|
| `compare-llama-cpp.sh --selftest` | pass, 114 checks (96 before) |
| `check-plan-thresholds.sh` | pass |
| End to end, unpublished: `--gpu --n-prompt 512 --no-tuned-lane --models tinyllama --juno-reps 1 --juno-warmup 1 --reps 1 --no-publish` | pass: prefill 1873 MHz (low 1860, 58 C, no event reasons); generate 1860 MHz (low 1860, 54 C); sampler process gone after the run |

The published note `docs/perf-compare/20261005T012348Z/gpu-clocks.md` and the decision 2 table above first
called clock event reason `0x4` the applications-clock setting; it is the software power cap (`0x2` is the
applications-clock setting). Both corrected 2026-10-05. That note's medians use the earlier
utilization-over-50% filter, which its method paragraph states; its 2048 windows run at near-full
utilization, so the reading does not change.
