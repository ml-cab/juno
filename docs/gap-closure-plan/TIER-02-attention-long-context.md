# Tier 02: Attention & long context

Status: **in progress** (2026-10-05). Implementation steps 1 to 4 done: the plan check passes, the
2048-over-512 milestone is decomposed, the scalar attention oracle is held to an independent reference
(`GqaMathOracleTest`), and the tiled online-softmax kernel replaces the full-materialization one on every
CUDA attention path (validated against the oracle from one key to `MAX_SEQ_LEN`; no score scratch; per-layer
window parameter; indicative 2048-token attention 10x to 17x faster, `docs/perf-compare/20261005T043507Z-tiled-attention-2048`).
Decisions 4 and 4b taken (a) the same day: the attention checks restated as kernel properties and
`GpuForwardPassIT`'s greedy check as a teacher-forced decode; every `-Pgpu` IT passes (10 of 10). Step 5's first
item (scope item 4, attention inside the decode residency region) is done: pinned A/B met, tg region on/off
1.145 and 1.096 (`docs/perf-compare/20261006T033707Z-gpu-residency-attention-ab`). Decision 5 taken (b): the llama-1-30b
region-off memory creep moves to item 8's mirror budget. Decision 6 taken (a): the 512-over-128 row stays
`>= 1.00` as written, scored by the closing sweeps (read 2026-10-06: 1.015 to 1.171, unpinned). Scope item 8's
first part is done (2026-10-06): the whole decode layer runs in the region on the LLaMA family, the residual row
stays on the device between layers (TinyLlama: 2 uploads and 23 downloads per token on 22 layers), bit-identical to
the op-at-a-time GPU path by test; its pinned A/B is met (owner run, tg on/off 2.208 and 1.716 where the region runs,
`docs/perf-compare/20261006T080204Z-gpu-residency-whole-layer-ab`). Item 8's second part is done (2026-10-06): the
whole decode layer runs in the region on Phi-3 and dense Qwen3 too (Phi-3.5-mini 2 H2D and 33 D2H per token on 32
layers, Qwen3-1.7B 2 and 29 on 28; greedy identical on against off), with a device LongRoPE kernel bit-identical to
the host rotation; its pinned A/B is met (owner run, tg on/off 1.545 and 1.699 where new,
`docs/perf-compare/20261006T200344Z-gpu-residency-phi3-qwen3-ab`). Item 8's third part is done
(2026-10-06): Phi-3's prefill window splits its fused Q/K/V on the device, rotates and attends inside the prefill region;
a 512-token Phi-3.5-mini window moves 414.8 MB (84.3% below 2,645 MB, threshold 70%,
`docs/perf-compare/20261006T184154Z-phi3-prefill-region-attention`), logits bit-identical region on against off;
pp 1.897x and tg 0.990x the pre-change build on the owner's pinned A/B
(`docs/perf-compare/20261006T192743Z-phi3-prefill-region-ab`). Item 8's fourth part is done (2026-10-06, owner
decision: a fixed 512-token reserve): `--gpu-layers auto` keeps one request's KV mirror free at 512 positions on the
device layers, and llama-1-30b's 508-token prompt keeps its attention on the device at one GPU layer fewer (22 to 21,
`docs/perf-compare/20261006T212419Z-kv-mirror-reserve`); the region-off memory creep it carried remains (64 MiB over
requests 4 to 8, limit 32). Decision 7 taken (a) (2026-10-07): the creep was a `DeviceKvCache` leak when a request's
mirror allocation ran out part-way; fixed, smoke flat in both modes, the mirror-budget criterion ticked. Decision 8
taken (a) (2026-10-07): a failed scratch growth no longer leaves a freed pointer behind (`DeviceScratchSlot`, three
classes) and `sgemv` no longer leaks on a partial allocation (CUDA and ROCm). Next: item 8's closing `--device-spans`
run with the prefill-window re-verification, and the pinned A/B on the final region (owner).

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

   *Corrected 2026-10-06 (item 8, second part): LongRoPE cannot be folded into the inverse-frequency
   table.* Phi-3.5-mini declares `rope.scaling.attn_factor = 1.1902381`, and `Phi3Rope.ropeExt` scales
   both cos and sin by it; a table of frequencies cannot carry a magnitude. It also builds the angle in
   float (`theta *= thetaScale` once per pair, then `/ factor`, then `* freqScale`), where `CudaRope`
   computes `pos * invFreq` in double, so even a folded table would change the angle's bits and lose
   bit-identity with the host rotation. The device rotation is instead a dedicated kernel entry
   (`rope_ext_split_half`, `CudaPhi3Rope`) that repeats the host arithmetic step for step and takes the
   selected factors and the scale as arguments. Scope and order unchanged; only the mechanism differs
   from the sentence above.

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
   *Corrected 2026-10-05 (step 3): the old kernel has no fixed scratch buffer to overflow.*
   `CudaGqaAttention.attendBatched` sizes its scores scratch per call to `B x numHeads x maxSeqLen`
   floats, so at long context it fails on the device allocation instead: a 2048-row window at 2048 tokens
   on a 32-head model needs 512 MiB, and a 512-row window at `MAX_SEQ_LEN` (32768) needs 2 GiB. Read
   "would have overflowed" as "whose scores scratch would not fit next to the model on this host's 8 GiB
   card"; the peak-scratch thresholds under "Tests to write/upgrade" measure exactly that.
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
- [x] Tiled attention kernel numerically matches the CPU oracle at all tested sequence lengths and
      reduces peak GPU memory at long context vs. the old full-materialization kernel (measured), and
      takes the per-layer window parameter Tier 02B needs.
      *2026-10-05 (step 4):* **Evidence (not published):** `GqaAttentionTiledTest` (8 cases, `@Tag("gpu")`)
      and `GqaAttentionKernelParityTest`, against `GqaMath` from 1 key to 32768, head widths 64/80/96/128/256,
      2048-row windows; peak scratch measured on both kernels (old: exactly `rows x heads x seqLen` floats
      plus the rows, 257 MiB at 64 rows x 32768; new: 1.0 MiB, flat from 1024 to 32768); `window` argument
      on the kernel, `GqaAttentionKernel.launch` and `CudaGqaAttention.attendBatched`, with window 0
      bit-identical to an unbounding window. The greedy-divergence re-characterisation scope item 1 asks for
      was run; its result is decision 4 in the step 4 record.
- [x] Attention inside the decode residency region (scope item 4): k and v appended to the device
      mirror and attention read from the region, one download of the attention output per layer;
      bit-identical to the op-at-a-time path by test; greedy output identical on/off on tinyllama,
      mistral-7b and llama-1-30b; per-request device memory flat in `smoke-gpu-residency.sh`.
      **Threshold**: end-to-end tg with the region on >= 1.0x the region-off run on every model where
      it runs, and >= 0.95x everywhere.
      *2026-10-05: implemented; bit-identity by test (`ResidentQkvPathTest`, 13 cases), greedy identical on
      against off on all three models, per-request device memory flat with the region on on all three
      (`smoke-gpu-residency.sh`; llama-1-30b's region-off check fails on the pre-change build too, decision 5).
      Owed by the owner: the pinned A/B threshold, `bash dist/gpu-residency-attention-ab/run-gate.sh`
      (indicative unpinned reading 1.122 and 1.097 on the two models where the region runs).*
      *2026-10-06: threshold met on the owner's pinned A/B,
      [`20261006T033707Z-gpu-residency-attention-ab`](../perf-compare/20261006T033707Z-gpu-residency-attention-ab/INDEX.md):
      tg on/off 1.145 (TinyLlama) and 1.096 (Mistral 7B) where the region runs (>= 1.00), 1.022 (Qwen2.5-3B)
      and 1.001 (Phi-3.5-mini) where it is declined (>= 0.95).*
- [ ] The whole decode layer runs inside the region (scope item 8) on the LLaMA family, Phi-3 and
      Qwen3, or the handler or operation that cannot is announced: decode-phase copies per generated
      token <= 1 x layers host-to-device and <= 1 x layers + 1 device-to-host, read off
      `juno.DeviceStaging` by test and by a published `--device-spans` run; tg region-on >= 1.0x
      region-off (same-hour pinned A/B); greedy output identical; per-request device memory flat; zero
      activation device-to-host copies inside a prefill window re-verified.
      *2026-10-06 (first part): LLaMA family done and held by test (`decodeRunsTheWholeLayerInTheRegion`: 2 H2D,
      23 D2H per token on TinyLlama's 22 layers; bit-identity in `ResidentQkvPathTest`); greedy identical and memory
      flat on TinyLlama and Mistral 7B (`smoke-gpu-residency.sh`). Still owed: Phi-3 and Qwen3 (next, executor), the
      published `--device-spans` run and the prefill-window re-verification (executor, at the item's end), and the
      pinned A/B on the final region (owner, at the item's end).*
      *2026-10-06: the first part's pinned A/B is met (owner run), tg region on/off 2.208 (TinyLlama) and 1.716
      (Mistral 7B), 0.959 and 0.991 where declined,
      [`20261006T080204Z-gpu-residency-whole-layer-ab`](../perf-compare/20261006T080204Z-gpu-residency-whole-layer-ab/INDEX.md).
      Not ticked: the criterion covers Phi-3 and Qwen3 too.*
      *2026-10-06 (second part): Phi-3 and Qwen3 done and held by test (`Phi3TransformerHandlerGpuResidencyTest`: 2 H2D,
      33 D2H per token on Phi-3.5-mini's 32 layers; `Qwen3TransformerHandlerGpuResidencyTest`: 2 and 29 on Qwen3-1.7B's
      28; bit-identity in `ResidentQkvPathPhi3Test` and `ResidentQkvPathQwen3Test`); greedy identical and memory flat on
      both (`smoke-gpu-residency.sh`). Still owed: the published `--device-spans` run and the prefill-window
      re-verification (executor, at the item's end), and the pinned A/B (owner; prepared,
      `bash dist/gpu-residency-phi3-qwen3-ab/run-gate.sh`). Not ticked.*
      *2026-10-06 (third part): Phi-3's prefill window rotates and attends inside the prefill region; on
      Phi-3.5-mini the window's only downloads are the K and V rows and the final residual
      ([`20261006T184154Z-phi3-prefill-region-attention`](../perf-compare/20261006T184154Z-phi3-prefill-region-attention/INDEX.md)).
      Still owed: the KV mirror budget part (executor, next), the `--device-spans` run and prefill-window
      re-verification on the other sweep models (executor, at the item's end), the pinned A/B on the final region
      (owner, at the item's end). Not ticked.*
      *2026-10-06: the second part's pinned A/B is met (owner run), tg region on/off 1.545 (Phi-3.5-mini), 1.696
      (Qwen3-1.7B), 2.054 (TinyLlama), 1.649 (Mistral 7B); 1.006 where declined (Qwen2.5-3B),
      [`20261006T200344Z-gpu-residency-phi3-qwen3-ab`](../perf-compare/20261006T200344Z-gpu-residency-phi3-qwen3-ab/INDEX.md).
      The copy-count, greedy and memory parts hold by test and smoke for every handler; not ticked until the
      fourth part, the published `--device-spans` run and the final region's pinned A/B.*
- [x] Phi-3.5-mini's prefill window runs RoPE and attention inside the prefill region (scope item 8,
      added 2026-10-02): **Threshold: H2D + D2H bytes per 512-token window >= 70% below 2,645 MB**
      (Tier 01B's step 2 baseline; 1,419 MB at `docs/perf-compare/20261002T050741Z`), from a published
      `--device-spans` run; logits bit-identical region on against off (`PrefillRegionHandlerParityTest`)
      or characterised no worse than the existing GPU-attention divergence (Phi-3.5-mini earliest 37).
      *2026-10-06 (item 8, third part): met.* 414.8 MB per 512-token window (6.6 H2D, 408.1 D2H; identical
      on three repetitions), 84.3% below 2,645 MB, [`20261006T184154Z-phi3-prefill-region-attention`](../perf-compare/20261006T184154Z-phi3-prefill-region-attention/INDEX.md);
      held by test as well (`Phi3PrefillRegionAttentionTest`: <= 793.5 MB, no host RoPE or attention, no
      fused Q/K/V download or attention upload). Logits bit-identical region on against off on every site of
      `PrefillRegionHandlerParityTest` (Phi-3.5-mini relative L2 0.000000 at all six).
- [x] KV mirror growth on a partially offloaded model budgeted (scope item 8, 2026-10-04 addition): a
      512-token prompt's attention stays on the device for llama-1-30b at `auto`, or the reason is
      stated with the number.
      *2026-10-05 (decision 5 (b)): also carries the region-off device-memory creep on llama-1-30b
      (`smoke-gpu-residency.sh`: 92 MiB over requests 4 to 8 on the pre-change build, 40 MiB after scope
      item 4; limit 32). Closing this criterion re-runs that smoke at `--requests 8` and reads both modes.*
      *2026-10-06 (item 8, fourth part): the 512-token part is met. `--gpu-layers auto` keeps one request's
      mirror free at 512 positions on the device layers; llama-1-30b, one node: 22 to 21 GPU layers, and a
      508-token prompt runs with no device-memory fallback where the pre-change build moved region prefill
      attention to the CPU,
      [`20261006T212419Z-kv-mirror-reserve`](../perf-compare/20261006T212419Z-kv-mirror-reserve/INDEX.md).
      Not ticked: the region-off creep this criterion carries is not removed (smoke at `--requests 8`: off 64
      MiB over requests 4 to 8, fail; on 14 MiB, pass). Owed: the owner's decision 7, then its outcome.*
      *2026-10-07 (decision 7 (a)): the creep was a device-memory leak in `DeviceKvCache` when a request's mirror
      allocation ran out part-way; fixed, tests first. The smoke at `--requests 8` now reads 0 MiB over requests 4
      to 8 in both modes (`20261007T023945Z`, 0 failures). Ticked.*
- [ ] `CudaGraphSession` decided by measurement (scope item 5): wired behind `--gpu-residency` if it
      saves at least 5% of decode forward-pass time on tinyllama and mistral-7b with greedy output
      unchanged, otherwise deleted with `CudaGraphSessionTest` and the measurement recorded. Not
      dormant either way.
- [ ] `--gpu-residency` default put to the owner (scope item 6) with the two published on/off sweeps
      on all four sweep models; changed only on the owner's decision.
- [ ] Cross-surface checklist fully resolved.
- [ ] Perf gate published, both memory thresholds above met, Juno t/s >= 0.95x the pre-tier build, and
      the standing CPU and allocation gate met.
      *2026-10-05 (step 4): the two memory thresholds are met by test (peak scratch -99.6% at 64 rows x
      32768 and -94.1% at a 2048-row window; the context-dependent part does not grow at all), see the step 4
      record; the 0.95x A/B, the standing CPU and allocation gate and the long-context microbenchmark's
      publication are owed at the tier's close.*
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

### 2026-10-05: implementation step 3 (the scalar attention oracle)

**What the oracle is.** `GqaMath.attend` is the method every GPU attention parity test compares the kernel
against (`GqaAttentionKernelParityTest`, `GqaAttentionReproducibilityTest`,
`LlamaTransformerHandlerGpuAttentionLiveTest`), and `LlamaTransformerHandler`'s CPU path calls it. The other
handlers keep private copies (`Phi2TransformerHandler.gqa`, `Phi3TransformerHandler.gqaInto`/`gqa`,
`Qwen3TransformerHandler.gqaInto`/`gqa`, `Qwen3MoeTransformerHandler.gqaInto`, and the two LoRA-trainable
handlers); read on 2026-10-05, each has the same loop order and calls the same
`LlamaTransformerHandler.softmax`, so the oracle stands for all of them by identical arithmetic, not by a
shared call.

**Coverage before this step.** None on the CPU. The oracle ran only inside `@Tag("gpu")` tests, as the
expected side of a comparison with a kernel written to the same algorithm, so a defect shared by both (a
wrong head mapping, a wrong scale) passed them. Nothing checked it at long context except as the GPU
parity test's reference at 8192 tokens.

**Test added: `node/src/test/java/cab/ml/juno/node/GqaMathOracleTest.java`** (CPU only, untagged, runs in
the normal `mvn test -pl node`). Seven cases:

| Case | What it holds |
|---|---|
| `matchesReferenceAcrossShapesAndLengths` | against an attention written from its definition in double, sharing no code with `GqaMath`: five head shapes (TinyLlama 32/4/64, Mistral 7B 32/8/128, Qwen2.5-3B 16/2/128, Phi-3.5-mini plain multi-head 32/32/96, Qwen3-1.7B 16/8/128) at 1, 2, 17, 128, 513 and 2048 keys, each with a flat and a peaked query (scale 1 and 6) |
| `matchesReferenceAtMaxSeqLen` | the same at `DenseKvTensor.MAX_SEQ_LEN` (32768 keys), TinyLlama shape, flat and peaked |
| `largeLogitsAreStable` | logits in the thousands: output finite and within tolerance (the max subtraction) |
| `singleKeyReturnsValueRow` | one key: every head returns its KV head's value row bit for bit |
| `zeroQueryAveragesValues` | equal scores: the output is the mean value row |
| `groupedHeadsReadOnlyTheirKvHead` | changing one KV head changes exactly the query heads `h / gqaRatio` maps to it and leaves the others bit-identical |
| `rowsPastSeqLenAreIgnored` | NaN in the K/V rows past `seqLen`, in `out` and in the scores scratch: output bit-identical to the unpadded call |

Tolerance is absolute on value rows in [-1, 1]: `2e-6 + 2e-6 x sqrt(seqLen)` (about 3.6e-4 at 32768),
covering float rounding in the dot products and in the two length-long float sums. The tiled kernel's
oracle test in step 4 compares against `GqaMath` at the same lengths, so the oracle's own error is bounded
at every length the kernel is tested at.

**Seen failing?** No: the code under test already exists, so this is a regression test, and it passed on
its first run. To show it can fail for the right reason, four defects were put into the oracle one at a
time, the class re-run, and the file restored (`git diff` empty afterwards):

| Injected defect | Cases failing (of 7) |
|---|---|
| grouped-query mapping `h % numKvHeads` instead of `h / gqaRatio` | 6 |
| scale `1 / headDim` instead of `1 / sqrt(headDim)` | 3 |
| last key dropped from the weighted value sum | 4 |
| softmax without the max subtraction | 1 (`largeLogitsAreStable`) |

| Command | Result |
|---|---|
| `scripts/performance-tests/check-plan-thresholds.sh` | pass (21 tier files) |
| `mvn -q test -pl node -Dtest=GqaMathOracleTest` | pass, 7 tests, 4.2 s |
| the same, once per injected defect | fail, as tabled above |

**Plan correction.** Implementation step 4 said the tiled kernel is validated "including lengths that would
have overflowed the old kernel's scratch buffer". The old kernel has no fixed buffer; its scratch is sized
per call and fails on the device allocation at long context. Corrected in place in step 4, dated.

**Not done here, by design.** No windowed oracle: the per-layer window parameter's test in this tier is
the kernel with "no window" against the kernel unwindowed, which needs no oracle; the windowed case belongs
to Tier 02B item 2, where the oracle is `GqaMath.attend` over the last `window` rows. No CHANGELOG entry:
this step adds a test only and ships nothing.

**Out-of-tier changes.** None.

### 2026-10-05: implementation step 4 (the tiled online-softmax kernel)

**Plan versus code, re-verified first.** `gqa_attention.cu` was the one-block-per-(row, head) kernel with a
`B x numHeads x seqLen` float scores scratch; both call sites launched it with that scratch,
`CudaGqaAttention.attendBatched` (decode, `--parallel` multi-decode, Phi-3 and Qwen3 prefill) and
`PrefillWindowRegion.attendInside` (the LLaMA-family prefill region), and `PrefillWindowFootprint` budgeted the
region's scores into the upload reserve and the adaptive prefill chunk. Every claim step 4 depends on held.

**Design.** `gqa_attention.cu` is rewritten in place (same file, same `GqaAttentionKernel`/`CudaGqaAttention`
entry points, so every caller keeps its call). A block of 128 threads is 32 slots of 4 lanes; it owns one
query head and a tile of up to 32 consecutive query rows that share one KV cache. K and V are staged through
shared memory in 32-key tiles (16 for heads over 128), converted from FP16 to FP32 once per block, and each
slot keeps a running maximum, running sum and rescaled output accumulator in registers (online softmax), so
no score row exists anywhere. At decode width (one row, or `--parallel` streams over different caches) the
32 slots split one row's keys and are merged in shared memory at the end. `rowsPerBlock` is chosen by the
caller (`GqaAttentionKernel.rowsPerBlock(oneCache, B)`). A `window` argument (0 = none) restricts row `b` to
its last `window` keys: the parameter Tier 02B item 2 needs. Entries `gqa_attention_d64`, `_d128` and `_d256`
bound the head width held in registers (no register spills on sm_61); a head width that is not a multiple of
4 or exceeds 256 keeps the caller on `GqaMath` (`supportsHeadDim`; no supported model has one). The region's
scores buffer and `PrefillWindowFootprint.scoresBytes` are removed, so a window's footprint no longer depends
on its context.

**Tests, written first** (README rule 3).

| Test | Seen failing first? | Now |
|---|---|---|
| `GqaAttentionTiledTest` (new, `@Tag("gpu")`, 8 cases): 2048-row windows from position 0 (TinyLlama and Mistral 7B shapes); windows off the key-tile boundary at head widths 64, 80, 96, 128, 256; one decode row at 32768 keys; a 64-row window at 8192; five `--parallel` streams of 1 to 4097 keys; window 0 bit-identical to an unbounding window; a window shorter than the context against the oracle over the last rows; device scratch | yes, on the old kernel with only the `window` overload added: the window case (window ignored) and the scratch case (context-dependent scratch doubles with the context). The six oracle-parity cases passed on the old kernel too, so for them this is a regression test of both | pass, 8 of 8 |
| `PrefillWindowFootprintTest` (changed): no term growing with rows x context | yes (the scores term) | pass, 7 of 7 |
| `GqaAttentionKernelParityTest`, `GqaAttentionReproducibilityTest` (existing) | n/a | pass |

The scratch case first asserted "doubling the context less than doubles the scratch", and that passed on the
old kernel, because the fixed query and output rows dilute the ratio. Tightened before the implementation to
"the scratch beyond the query, output and table rows does not grow with the context", which is stricter than
the plan's threshold and failed on the old kernel.

**Peak attention scratch, measured on both kernels** (TinyLlama shape, 32 heads; the perf gate's memory
thresholds):

| Rows x context | Old kernel | New kernel | Change |
|---|---|---|---|
| 64 x 1024 | 9,438,464 B | 1,049,856 B | -88.9% |
| 64 x 32768 | 269,485,312 B | 1,049,856 B | -99.6% |
| 2048 x 2048 | 570,466,304 B (formula; the old kernel's reading matched it exactly at every measured length) | 33,595,392 B | -94.1% |

Both thresholds are met: >= 80% lower at the longest tested length, and doubling the context leaves the
context-dependent part unchanged (it is zero).

**Accuracy against exact attention.** A probe (not kept) compared both kernels and `GqaMath` with a
double-precision attention over the same FP16 K/V (TinyLlama shape, Gaussian data, 10 trials per cell):

| Scores | Keys | New kernel | Old kernel | `GqaMath` |
|---|---|---|---|---|
| flat | 25 / 600 / 4000 | 1.6e-7 / 2.2e-7 / 2.9e-7 | 2.2e-7 / 5.3e-7 / 1.2e-6 | 2.3e-7 / 6.5e-7 / 1.5e-6 |
| peaked (query x 8) | 25 / 600 / 4000 | 2.9e-7 / 4.8e-7 / 6.3e-7 | 5.2e-7 / 1.1e-6 / 1.4e-6 | 5.3e-7 / 1.2e-6 / 2.3e-6 |

(relative L2). The new kernel is 1.4x to 5x closer to exact attention than the old one and than the CPU
oracle. The old kernel was close to `GqaMath` because it summed in the same order (sequential dot, sequential
weighted-V sum), not because it was more accurate.

**Indicative speed at 2048 tokens** (unpinned, one repetition, candidate only; published as
[`20261005T043507Z-tiled-attention-2048`](../perf-compare/20261005T043507Z-tiled-attention-2048/INDEX.md); not
the milestone reading):

| Model | Attention ms per window, pre-tier / now | Raw | Clock-normalised | Prefill ms, pre-tier / now | pp ratio at 2048, pre-tier / now |
|---|---|---|---|---|---|
| tinyllama-1.1b | 4459 / 291 | 15.3x | no pre-tier clock | 5060 / 855 | 0.125x / 0.672x |
| mistral-7b | 19253 / 1141 | 16.9x | 15.5x | 23226 / 4375 | 0.144x / 0.729x |
| Phi-3.5-mini | 19900 / 1973 (kernel 1036, copies 250, host 687) | 10.1x | no pre-tier clock | 24334 / 6325 | 0.091x / 0.304x |

Qwen2.5-3B was not read. Two design iterations are behind these numbers: the first version kept the tiles in
FP16 and converted per row (TinyLlama 3.2x, `target/` only, not published), and the 64-wide variant moved
TinyLlama from 6.8x to 15.3x (the 128-wide variant needs 211 registers, so 2 blocks per SM). Every model reads
well above the `>= 6.0x` milestone here; it is scored only by the same-session A/B at the tier's close.

**Regression runs on the final kernel.**

| Command | Result |
|---|---|
| `check-plan-thresholds.sh` | pass |
| `mvn test -pl node` | 909 run, 0 failures, 44 skipped |
| `mvn test -pl node -Dgroups=gpu` | 301 run, 2 failures, 7 skipped. The two failures, `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` (71 MiB reported free on a full device against a 64 MiB bound) and `ResidentActivationTest.openChain_isVisibleToMemGetInfo_andCloseReturnsIt` (free bytes after close 448 KiB lower than before), are device-wide free-VRAM readings; both classes passed alone twice in a row afterwards, and both passed in the first full run on this change. Environment, not this change |
| `mvn verify -pl juno-master -Pgpu` (after `mvn install -DskipTests`) | 10 run, 1 failure: `GpuForwardPassIT.greedy_decode_agrees` (decision 4). `PrefillRegionGreedyIT` 2 of 2, region greedy identical on 6 of 6 prompts over 64 tokens on TinyLlama and Mistral 7B. `GpuAttentionDivergenceIT` 3 of 3 |

Not run in this step: the 11-module unit reactor (no other module changed), the real-model `ModelLiveRunnerIT`,
and the smoke scripts. Those belong to the tier's closing matrix (implementation step 6).

**Greedy divergence, re-characterised (scope item 1)** (`GpuAttentionDivergenceIT`, six prompts, 64 greedy
tokens, kernel on against off; first token equal on every prompt for every model):

| Model | Identical over 64, before / now | First divergent step, before / now |
|---|---|---|
| TinyLlama-1.1B | 3 / 4 of 6 | 8, 13, 23 / 8, 23 |
| Phi-3.5-mini | 4 / 4 of 6 | 37, 40 / 13, 19 |
| Qwen3-1.7B | 3 / 3 of 6 | 20, 23, 32 / 40, 44, 53 |

**Raised with the owner (2026-10-05): decision 4, three checks the new summation order moves.**

1. `GpuForwardPassIT.greedy_decode_agrees` (TinyLlama, "The capital of France is", 16 greedy tokens, CPU
   backend against CUDA with GPU attention at its default) now parts at step 13 of 16; the pre-change tree
   passes it (run on a `git archive` of HEAD, in-reactor).
2. Phi-3.5-mini's earliest greedy divergence moved from step 37 to 13 (still 4 of 6 identical); Tier 01B's
   rule for that architecture was "never earlier than the item-0 baseline's earliest divergence" (26).
   *Corrected 2026-10-05: that rule governed Tier 01B item 6's own change, on TinyLlama and Mistral 7B; it
   is not a standing rule for Phi-3.5-mini. The earlier divergence is still a change to report.*
3. `GpuAttentionHandlerParityTest` (logits relative L2 kernel on against off, bound 0.025) passes on the final
   kernel, but its Qwen3-1.7B multi-decode B site reads 0.0149 (old kernel), 0.0317 (first version of this
   kernel), 0.0368 (the same with the scale applied after the dot), 0.0223 (final): a site that moves 2.5x
   between kernels whose own error is about 1e-6 is measuring the FP16 mirror's rounding amplified by the
   model, so it will pass or fail by draw. The planted-fault reference reads 0.090.

All three compare against the scalar path's rounding (float KV, `GqaMath`'s order), which the old kernel
reproduced and an online-softmax kernel cannot; the accuracy probe above shows the new kernel is the closer of
the two to exact attention. Options: (a) restate the three as kernel properties, not order agreement: hold the
kernel to exact attention over the FP16 mirror (as `GqaAttentionTiledTest` does against `GqaMath`), take this
step's divergence table as the new characterisation baseline for scope item 1, and have `GpuForwardPassIT`
compare greedy tokens with GPU attention off (its purpose is the CUDA matmul path; GPU attention has its own
divergence IT); (b) reproduce the scalar order at decode width (one slot sums every key sequentially), giving up
the decode split and some accuracy, and accept that prefill windows still differ; (c) keep the checks as they
are and widen bounds. Recommended: (a).

**Out-of-tier changes.** None. The removal of the scores term from `PrefillWindowFootprint` is this tier's
kernel change, but it is a measurement boundary for one thing outside attention: the adaptive prefill chunk
and the upload reserve are sized from the footprint, so on a card the window does not fit easily a wider
window can now be chosen (the reserved 64-row window shrinks by 0.5 MiB on Mistral 7B). The 512 and 2048
sweeps in this tier's decomposition ran one window per prompt and are unaffected.

### 2026-10-05: decision 4 taken (a); the attention checks restated; decision 4b raised

Owner decision: restate the three checks as properties of the kernel rather than agreement with the scalar
path's rounding order.

**`GpuAttentionHandlerParityTest`, restated.** `GpuAttentionMirror` gains a package-private
`DispatchObserver` (tests only; `null` in production, one volatile read per launch) that sees each successful
launch's mirrors, queries, lengths and output. The test now holds the kernel, at every launch on the real
Phi-3.5-mini and Qwen3-1.7B runs, to `GqaMath.attend` over the same FP16 rows read back from the mirror
(relative L2 per row <= 1e-3), and requires every call-site kind to be seen (prefill window, single decode,
multi-stream decode). The logits bound (0.025) and top-1 check stay at the two prefill windows, where on
against off reads 0.0001 to 0.0006. The three decode sites' logits are printed, not bounded.

| Reading | Phi-3.5-mini | Qwen3-1.7B |
|---|---|---|
| Kernel against `GqaMath` over the mirror, worst row: window / decode / streams | 1.7e-6 / 8.3e-7 / 1.0e-6 (64, 32, 32 launches) | 3.5e-6 / 1.4e-6 / 1.8e-6 (56, 28, 28 launches) |
| Logits, prefill windows A / B (bounded) | 0.000088 / 0.000226 | 0.000304 / 0.000411 |
| Logits, decode sites (reported) | 0.0052, 0.0011, 0.0036 | 0.0045, 0.0057, 0.0223 |
| Planted fault (one head zeroed after every launch), kernel check alone | 0.91, fails | 0.71, fails |

The planted fault was put into `CudaGqaAttention` temporarily, the class run, the file restored. On the first
planted run the logits check failed first (0.118 and 0.090 at window A), so the kernel check was moved ahead
of it and the fault planted again to show the kernel check catches it by itself.

One environment note: in 2 of 5 runs of the restated class the Qwen3-1.7B case failed its pre-existing
free-VRAM check ("device memory back to where the test started", 16 MiB slack), 76 and 90 MiB short, while
the desktop's own GPU use moved between 736 and 874 MiB across samples. The observer allocates nothing on the
device (its reads stage through host memory). The same class of device-wide check as
`PrefillRegionHandlerParityTest`'s; not loosened.

**Divergence baseline.** This step's `GpuAttentionDivergenceIT` table (re-run on the final kernel, same
figures) is the characterisation for scope item 1 and now stands in `docs/performance.md` and the
`--gpu-attention` row of `docs/howto.md`, with the earlier kernel's figures beside it.

**Decision 4b, raised: `GpuForwardPassIT.greedy_decode_agrees`.** The plan for this check was to run its GPU
leg with GPU attention off, so it would compare the CUDA matmul path alone. Done, and it fails earlier: the
GPU leg parts from the CPU at step 8 of 16 (CPU `29907`, GPU `315`). The pre-change tree with the same edit
fails identically (same step, same tokens; scratch copy, in-reactor), so the CUDA path without GPU attention
never matched the CPU token for token on this prompt. The check passed before only because the old kernel's
rounding happened to land the GPU-on run on the CPU's choice at that near-tie (the same prompt diverges at
step 8 in `GpuAttentionDivergenceIT`, on against off). A free-running greedy comparison over 16 steps measures
where near-ties fall, whichever path is compared. Options:
- (a) Teacher-forced decode: feed the CPU's greedy tokens to both runs and hold every step's logits to the
  bounds the test already applies to one forward pass (relative L2 <= 0.025, top-5 overlap >= 4), with top-1
  required only where the CPU's top-two margin is clear of a near-tie. This tests 16 decode steps over a
  growing KV cache without compounding.
- (b) Keep free-running greedy, but require only the tokens up to the first near-tie on the CPU's own
  trajectory.
- (c) Drop the greedy case; the single-pass hidden-state and logits cases and `GpuAttentionDivergenceIT`
  remain.
Recommended: (a).

### 2026-10-05: decision 4b taken (a); `GpuForwardPassIT`'s greedy check is a teacher-forced decode

Owner decision: teacher-forced decode. `greedy_decode_agrees` now decodes 16 greedy tokens on the CPU, keeping
the logits each token was chosen from, and feeds those same tokens to the GPU run, which takes the whole
default GPU path again (GPU attention at its default; the attention-off edit of decision 4 is reverted). Every
step is held to the bounds the test already applies to one forward pass: logits relative L2 <= 0.025 and top-5
overlap >= 4. Top-1 must agree wherever the CPU's top-two logit gap is at least `NEAR_TIE_MARGIN` = 1.0. A flip
needs the errors on the two candidates to add up to the gap, and the largest logit error measured at any step is
0.41 (TinyLlama) and 0.12 (Mistral 7B), so 1.0 is beyond any measured error; the margin was read from a
calibration run with it at 0, which printed every step's gap and error.

| Model | Relative L2 per step | Largest logit error per step | Top-1 held (gap >= 1.0) | Top-1 flips |
|---|---|---|---|---|
| tinyllama-1.1b | 0.0072 to 0.0220 | 0.17 to 0.41 | 6 of 16 steps, all agree | step 13 only, CPU gap 0.123 (a near-tie; the step at which the free-running check used to part) |
| mistral-7b | 0.0040 to 0.0075 | 0.08 to 0.12 | 9 of 16 steps, all agree | none |

**Seen failing for the right reason.** One head's attention output zeroed after every launch (planted in
`CudaGqaAttention`, installed, run, file restored and reinstalled): the teacher-forced check fails at step 0,
relative L2 0.086 (bound 0.025), as do the two single-pass cases (hidden state, logits).

| Command | Result |
|---|---|
| `mvn verify -pl juno-master -Pgpu -Dit.model.path=<tinyllama> -Dit.test=GpuForwardPassIT`, margin 0 (calibration) | tinyllama fails at step 13 (the near-tie), mistral-7b passes all 16 steps |
| the same, margin 1.0, tinyllama | 5 of 5 |
| the same with the planted fault | 3 of 5 fail, the teacher-forced case at step 0 |
| `mvn verify -pl juno-master -Pgpu -Dit.model.path=<tinyllama>` (restored tree, after `mvn install -pl node -DskipTests`) | 10 of 10: `GpuForwardPassIT` 5, `GpuAttentionDivergenceIT` 3 (same figures as the step 4 record), `PrefillRegionGreedyIT` 2 |

### 2026-10-05: implementation step 5, scope item 4 (attention inside the decode residency region)

**Plan versus code, re-verified first.** After the region (`ResidentQkvPath.run`, which downloaded q, k and v
with one wait), `LlamaTransformerHandler.transformerLayer` packed K and V to FP16 on the host and copied them in
two synchronous copies (`DeviceKvCache.appendToken`), then `CudaGqaAttention.attendBatched` made four
synchronous uploads (query, K and V pointer tables, lengths) and one download. GPU attention defaults to `auto`
(on under CUDA), so the item's path is the default whenever `--gpu-residency` is on. Every claim the item depends
on held. One wording point, recorded rather than escalated: the item says "downloading only the attention output"
and also "keep the CPU KV tensors written". The host tensors need k and v, so the one per-layer download carries
k, v and the attention output, packed into one activation row so it is one copy.

**Design.** `ResidentQkvPath.run(li, x, pos, mirror)` goes on after RoPE when it is given a mirror it can attend
through (`attendsOnDevice()`: the attention kernel runs the head width and `PrefillWindowKernels` loaded; the
mirror is live, `readableThrough(pos)` and already has room for `pos`). It casts k and v into the mirror at
`pos` (`DeviceKvCache.writeWindowOnDevice`, the cast the prefill region uses), copies a 20-byte table (K pointer,
V pointer, `pos + 1`) up from a pinned buffer, and launches the tiled kernel at one row on the region's stream.
K, V and the attention output share one activation (`decode region k, v, attention`); k is rotated in place
there (`CudaRope.applyResidentColumns`, new). The watermark (`c91f879`) is not moved by the region: the handler
writes k and v into the host KV tensors first and then calls `markWritten(pos, 1)`. The handler grows the mirror
before the call (`ensureCapacity`, with the existing out-of-memory retire-and-warn), so growth, and running out
of device memory for it, never happens inside the region. Any mirror the region cannot read (null, closed, short
of its history) gives the old exit: q, k and v back, the handler attends as before. `--gpu-attention off` is
announced once, in the log and on the console (`GpuResidencyOptions.consoleNotice`), as attention staying outside
the region. Per layer and token, with the region attending: two uploads (residual row, table) and one download,
one host wait, where the path before made one upload and three downloads in the region and seven synchronous
copies after it. No allocation per call: the region's buffers are pooled; the per-thread `Output` is reused.

**Tests, written first** (README rule 3). Against a stub of the new API (`attendsOnDevice` false, never attends):

| Test | Seen failing first? | Now |
|---|---|---|
| `ResidentQkvPathTest.attentionInsideTheRegionMatchesBitForBit`: k, v and attention output against host FP16 pack plus `attendBatched` at one row, positions 0, 1, 63, 64, 517, 4096; watermark not moved by the region; mirrors bit-identical after `markWritten` | yes (not attended) | pass |
| `decodeRunMatchesStepForStep`: 100-token decode from 0, growth past 64, every step bit-identical, mirrors identical at the end | yes | pass |
| `concurrentDecodersEachMatch`: three threads, own mirrors, 30 steps each | yes | pass |
| `attentionAllocatesNothingPerCall`: 200 attended calls, region bytes unchanged, device free memory not lower | yes | pass (see below) |
| `aMirrorItCannotReadIsNotAttended`: a short, a closed and a null mirror give q, k, v as without one and leave the mirror unwritten | no: the stub's behaviour, so a regression test of the fallback | pass |
| pooling and retention cases extended: half the short-lived threads attend; every other of 300 create-run-close cycles attends | no (stub never attends) | pass |
| `LlamaTransformerHandlerGpuResidencyTest.decodeAttendsInsideTheRegion` (real TinyLlama, JFR): zero decode copies at `memcpy(K row H2D)`, `memcpy(V row H2D)`, `memcpy(gqa qBatch H2D)`, `memcpy(gqa kPtrs H2D)`, `memcpy(gqa outBatch D2H)`, `materializeRows(resident activation)`; exactly layers x tokens at the region's input upload, table upload and packed download | yes (at its first assertion, attention not active; its copy-count assertions were not run against the old code) | pass, 3 of 3 in the class; greedy token equal at all 24 positions, largest logit difference on against off 0.2256 (bound 0.5) |
| `GpuResidencyOptionsTest`: console notice for `--gpu-attention off` | yes (null) | pass, 7 of 7 |

Planted fault: the table's length set to `pos` instead of `pos + 1` (last key dropped) fails 3 of 13 cases
(bit-identity, 100-token decode, concurrent decoders); file restored.

`attentionAllocatesNothingPerCall` first asserted the device-wide free-memory reading equal before and after,
copying the existing `noDeviceAllocationPerCall`. It passed alone and failed once in the full GPU group with free
memory 320 KiB *higher* at the end (7,694,385,152 to 7,694,712,832 bytes): something else released memory
during the run. An allocation per call can only lower that reading, so this new test asserts "not lower". It
passed alone twice before and after the change. The existing test keeps its strict equality.

**Regression runs.**

| Command | Result |
|---|---|
| `check-plan-thresholds.sh` | pass |
| `mvn test -pl node` | 916 run, 0 failures, 44 skipped |
| `mvn test -pl node -Dgroups=gpu` | 307 run, 1 failure (`attentionAllocatesNothingPerCall`, above), 7 skipped; that class then 13 of 13, three times |
| `mvn verify -pl juno-master -Pgpu -Dit.model.path=<tinyllama>` (after `mvn install -DskipTests`) | 10 of 10: `GpuForwardPassIT` 5, `GpuAttentionDivergenceIT` 3, `PrefillRegionGreedyIT` 2 |
| `smoke-gpu-residency.sh --models tinyllama...,mistral...` (unmodified) | 0 failures: region active with the KV append and attention on 8 of 8 and 11 of 11 layers per node, greedy identical on against off (32 tokens), GPU memory flat (0 MiB growth) both modes. Cluster pipeline and tensor answer and leave no node JVM; their text differs from local mode's from word 21, as recorded at Tier 01C's close (cluster prefill runs a different kernel route) |
| `smoke-gpu-residency.sh --models llama-1-30b --no-cluster` | region active on 20 of 20 GPU layers, greedy identical on against off. **1 failure, in the region-off mode**: off grows 18 MiB over requests 2 to 4 (7470 to 7488; limit 16), on 2 MiB |
| the same on the pre-change build (`ffb9583`, `git archive`, its own script) | 0 failures: off grows 10 MiB (7444 to 7454), on 2 MiB |
| candidate, `--requests 8` | off 7436 7484 7492 7508 7534 7544 7548 7548 (40 MiB over requests 4 to 8, limit 32: fail); on 7514 7526 7538 7538 7550 7550 7554 7564 (26 MiB: pass); greedy identical |
| pre-change build, `--requests 8` | off 7294 7328 7404 7422 7422 7470 7510 7514 (**92 MiB** over requests 4 to 8: fail); on 7514 7518 7522 7524 7530 7532 7536 7540 (16 MiB: pass); greedy identical |

**Reading of the llama-1-30b memory check.** The failure is in the region-off mode, whose code this change only
restructures around the JFR attention event, and it reproduces on the pre-change build, more strongly (92 MiB
against 40 over requests 4 to 8). So it is not introduced here: the default decode path on this partially
offloaded model, with about 200 MiB free on the card, keeps gaining device memory across requests. With the
region on, both builds stay inside the limit (16 and 26 MiB over requests 4 to 8). The item's own criterion,
"per-request device memory flat" for the region, holds on all three models; the off-mode creep is raised with
the owner below rather than fixed here, since it is outside this item.

Not run in this step: the 11-module unit reactor (only `node` changed), the real-model `ModelLiveRunnerIT`, the
other smoke scripts, and the vision and LoRA gates. Those belong to the tier's closing matrix (implementation
step 6). `compare-llama-cpp.sh`'s `--gpu-residency` help text now names the KV append and attention.

**Owed to the owner: the item's threshold.** End-to-end tg with the region on >= 1.0x region-off on every model
where it runs, >= 0.95x everywhere, is a pinned same-hour A/B (tighter than 0.90x). Prepared at
`dist/gpu-residency-attention-ab/`: `candidate-shaded.jar` (sha256 `ace9e5a64b1192eb`, this working tree) and
`run-gate.sh`, which alternates `--gpu-residency off` and `on` (A B A B A B) over the four sweep models at
`n_prompt=128` with `--pin-clocks`, and scores tg medians: >= 1.00 on TinyLlama and Mistral 7B (region runs),
>= 0.95 on Qwen2.5-3B and Phi-3.5-mini (region declined). Command: `bash dist/gpu-residency-attention-ab/run-gate.sh`.
llama-1-30b is not a sweep model and the harness carries no reference reading for it; its greedy parity is held
by the smoke above.

**Indicative reading, not scorable** (the same script with `PIN=0`, clocks not pinned, published as
[`20261006T025445Z-gpu-residency-attention-ab-unpinned`](../perf-compare/20261006T025445Z-gpu-residency-attention-ab-unpinned/INDEX.md)):
tg on/off 1.122 (TinyLlama), 1.097 (Mistral 7B), 0.994 (Qwen2.5-3B) and 1.004 (Phi-3.5-mini), medians of three.
TinyLlama pp on/off reads 0.903, with one low run (`on-3`, low on both lanes of both models) and prefill not
running the region; the pinned run records pp beside tg.

**Raised with the owner (2026-10-05).**
- *Decision 5: the region-off device-memory creep on llama-1-30b.* The default decode path (region off) gains
  device memory across requests on this partially offloaded model: 92 MiB over requests 4 to 8 on the pre-change
  build, 40 MiB on this one, so `smoke-gpu-residency.sh --models llama-1-30b` fails its region-off check on
  both. (a) Attribute it now, as an out-of-tier item recorded here (per-request reading of `DeviceKvCache`
  allocations and the allocator's free bytes across 16 requests). (b) Leave it to scope item 8's mirror-budget
  work, which already owns KV mirror growth on this model, and carry the smoke's region-off failure on
  llama-1-30b as known until then. Recommended: (b). The growth is bounded by the mirror's growth on a
  near-full card, item 8 changes exactly that budget, and the region-on mode this item adds passes.
- *Still open from decision 1:* whether the 512-over-128 milestone row gets the 2048 row's restatement.

**Decision 5 taken (b) the same day (owner).** The region-off device-memory creep on llama-1-30b goes to scope
item 8's KV-mirror budget work. Until that lands, `smoke-gpu-residency.sh --models llama-1-30b`'s region-off
memory check is a known failure, reproduced on the pre-change build; its region-on check and greedy parity
still hold. The mirror-budget exit criterion now carries it.

**Out-of-tier changes.** None.

### 2026-10-06: scope item 4's pinned gate (owner run): met

`bash dist/gpu-residency-attention-ab/run-gate.sh`, clocks pinned on all six runs (03:37Z to 03:50Z), jar
`ace9e5a64b1192eb` on both sides, the flag alternated off, on, off, on, off, on; published as
[`20261006T033707Z-gpu-residency-attention-ab`](../perf-compare/20261006T033707Z-gpu-residency-attention-ab/INDEX.md).

| Model | Region | tg off median | tg on median | tg on/off | Threshold | pp on/off (not gated) |
|---|---|---|---|---|---|---|
| tinyllama-1.1b | runs | 61.31 | 70.23 | **1.145** | >= 1.00, met | 0.970 |
| mistral-7b | runs | 20.96 | 22.98 | **1.096** | >= 1.00, met | 0.989 |
| qwen2.5-3b | declined | 26.84 | 27.43 | 1.022 | >= 0.95, met | 0.989 |
| Phi-3.5-mini | declined | 29.55 | 29.60 | 1.001 | >= 0.95, met | 0.991 |

On both region models every on repetition is above every off repetition. Allocation per generated token falls
about 3% with the region on (TinyLlama 47.4M to 46.5M bytes, Mistral 7B 224.3M to 218.3M, medians); GC pause in
the token span unchanged (7 and 14 ms). The unpinned TinyLlama pp reading of 0.903 was noise: pinned, pp on/off
is 0.970 to 0.991 on every model, inside the off side's own spread.

Scope item 4's exit criterion is ticked. Against the 0.70x end-of-plan tg targets, the 1.096x on Mistral 7B is a
Juno-against-Juno figure; the llama.cpp-relative tg ratios are read after items 8, 5 and 6 from the closing
sweeps, as the tier's milestone paragraph says.

### 2026-10-06: the 512-over-128 milestone read before deciding its form (owner request)

Owner decision on the question left open by decision 1: read the current number first. Two unpinned GPU sweeps
on the current build (jar `ace9e5a64b1192eb`), `n_prompt=128` then 512, four sweep models, every row scorable;
published as [`20261006T045710Z-pp-length-scaling`](../perf-compare/20261006T045710Z-pp-length-scaling/INDEX.md).

| Model | pp ratio 128 | pp ratio 512 | 512 over 128 | Before the kernel (Tier 01C close, pinned) |
|---|---|---|---|---|
| tinyllama-1.1b | 0.623x | 0.730x | 1.171 | 0.606 |
| qwen2.5-3b | 0.734x | 0.842x | 1.147 | 0.697 |
| Phi-3.5-mini | 0.303x | 0.307x | 1.015 | 0.743 |
| mistral-7b | 0.775x | 0.826x | 1.067 | 0.642 |

Every model reads above `>= 1.00`. Unlike the 2048 row, this one is met by the kernel it was moved here to
score, so the ratio-of-ratios objection does not bite in practice: the kernel speeds 128 as well, and the row
still rose 0.27 to 0.57. Phi-3.5-mini is the exception to watch. It clears by 1.5%, inside the 15% noise floor,
and its prefill attention still runs outside the region with host copies (scope item 8's Phi-3 prefill
addition), so its pinned closing reading could land either side of 1.00.

**Raised with the owner (decision 6): the row's form.** (a) Keep `>= 1.00` as written; the tier's pinned
closing sweeps at 128 and 512 score it, with item 8's Phi-3 prefill work expected to widen Phi-3.5-mini's
margin. (b) Restate it as the 2048 row was (an attention speedup at 512). (c) Re-own it to Tier 14 as a
reported distance. Recommended: (a). The row measures what it was meant to and is met on three models by a
clear margin; restating a row that its kernel met would only remove the check on Phi-3.5-mini.

Also read here, not scored: the end-of-plan GPU pp target at 512 (`>= 0.40x`) reads 0.730x to 0.842x on three
models and 0.307x on Phi-3.5-mini, which binds.

**Decision 6 taken (a) the same day (owner).** The 512-over-128 milestone row keeps `>= 1.00` as written, in this
file and in the README milestone table, and the tier's pinned closing sweeps at 128 and 512 score it. Decision 1's
open question is closed. Phi-3.5-mini's margin (1.015 unpinned) is the one to watch; scope item 8's Phi-3 prefill
work is expected to widen it.

### 2026-10-06: implementation step 5, scope item 8, first part (the whole decode layer in the region, LLaMA family)

**Plan versus code, re-verified first** (HEAD `d3ad6ee`, tree clean apart from the untracked `.github/`).
After the region (`ResidentQkvPath.run`, attending), `LlamaTransformerHandler.transformerLayer` ran the output
projection (`matVecProjection`, `CudaMatVec.sgemv` over the K-quant matrix: its own upload and download), the
residual add on the host, the FFN norm (`rmsNormGpuOrCpu`, on the CPU in production since `rmsNormGpu` is never
constructed), gate and up (`sgemvSameX`: one upload, two downloads), SwiGLU on the host and the down projection
(`sgemv`), each with a host wait. Every device kernel the rest of the layer needs existed (MMQ quantize and GEMV,
`rms_norm.cu`, `add_inplace`) except one. Two drifts from the item's wording, neither changing scope:
- **SwiGLU.** The item says "Tier 01B item 6's kernel at width 1". That kernel (`swiglu_half`) writes FP16 for the
  prefill GEMM; the decode down projection quantizes an FP32 row to Q8_1. Added `swiglu`, the same arithmetic
  without the cast.
- **The copy threshold needs the residual to stay on the device between layers.** Per layer the region made two
  uploads (the residual row and the 20-byte attention table) and, per token, the LM head adds one upload. One upload
  per layer plus the LM head's is `layers + 1`, over the `<= 1 x layers` threshold. So the item's optional "where the
  region also spans the next layer, take it" is what the threshold requires, and the table is now written by a kernel
  from its launch arguments (`decode_attention_table`) instead of copied up.

Item 8 has four parts (the LLaMA-family whole layer; the Phi-3 and Qwen3 handlers; Phi-3's prefill RoPE and
attention; the KV mirror budget on a partially offloaded model). This step is the first, per the item's own order
(the rest of the layer, then the other handlers). Its exit criteria stay open until the other parts land.

**Design.** New class `ResidentLayerTail` issues the rest of the layer on the region's stream: output projection,
`add_inplace`, FFN norm (`CudaRmsNorm.normalizeResident`, the tree-reduction kernel the attention norm already uses
in the region), gate and up sharing one Q8_1 quantization, `swiglu`, down projection, and `residual_add_both`, which
writes the layer output into the residual activation (the next layer's input) and into the packed download row in
one launch. `ResidentQkvPath.create(..., ResidentLayerTail.Weights)` builds it; a layer runs whole only when its four
matrices are K-quant on the device with the model's shape (`runsWholeLayer`), otherwise it leaves the region after
attention (announced once, with the count). The packed row widens to k, v, attention, layer output; one download per
layer. `ResidentQkvPath.Lease` holds one region for a token's layer loop (`LlamaTransformerHandler.runLayers`,
try-with-resources); a layer run whole records (next layer, position), and the next call in the lease with exactly
that layer and position skips its upload. Any other call (a layer the region declines, one run partly or not at all)
breaks the chain and the next layer uploads. `runLayers` copies the last layer's output, since the thread-owned row is
overwritten by the next call. Allocation: the handler path that ran per layer (several fresh arrays per op) now
returns the region's row; one `hidden`-wide copy per token.

**Tests, written first** (README rule 3), against stubs of the new API (`Weights` accepted, the tail never run):

| Test | Seen failing first? | Now |
|---|---|---|
| `ResidentQkvPathTest.wholeLayerMatchesBitForBit`: one layer at positions 0, 1, 63, 64, 517, 4096; k, v, attention and layer output against the op-at-a-time GPU path (MMQ GEMVs through `CudaMatVec`, round-trip GPU norm, host adds and SwiGLU) | yes (`layerDone` false) | pass |
| `layersInALeaseMatchStepForStep`: 40 tokens through four layers, one lease per token, layer 2 without a device down projection (leaves the region after attention, its tail on the host) | yes | pass |
| `concurrentLeasesEachMatch`: three threads, 15 tokens each | yes | pass |
| `aLayerOutsideTheRegionBreaksTheChain`: layer 1 run entirely on the host between two region layers in one lease | written after a planted fault went uncaught (below); passed on the implementation, then failed with the fault | pass |
| `wholeLayerNeedsItsWeightsAndAMirror`, `wholeLayerAllocatesNothingPerCall` | no: the stub already answered `runsWholeLayer` and ran the head only, so regression tests | pass |
| `LlamaTransformerHandlerGpuResidencyTest.decodeRunsTheWholeLayerInTheRegion` (real TinyLlama, JFR; replaces `decodeAttendsInsideTheRegion`): zero decode copies at every op-at-a-time site of the layer and at the old table copy; the residual row uploaded once per token; one packed download per layer and token; the single-row K-quant sites carry only the LM head; per token H2D `<= layers` and D2H `<= layers + 1` | yes (whole layer not active) | pass: **2 H2D and 23 D2H per token on 22 layers** |

Planted faults, each in its own run, file restored after (`cmp` against a saved copy):
- FFN norm reading another layer's weight: 3 of the whole-layer cases fail.
- The lease skipping the upload on position alone (layer index not checked): **not caught** by the first four
  cases, because every region call already resets the lease, so the index check only matters when a layer never
  reaches the region. `aLayerOutsideTheRegionBreaksTheChain` was added for it and fails with the fault.
- The residual uploaded at every layer: the handler test fails at the once-per-token upload count.

`wholeLayerAllocatesNothingPerCall` failed once in a group run on its first version, with the device-wide free
memory lower after the loop; the loop then included the test's own reference calls on layer 2 (`CudaMatVec`, round-trip
norm). It passed six times alone. Rewritten so only region work (layers 0 and 1) runs between the two readings; three
group runs pass. The region's own bytes are held exactly by `deviceBytes()` either way.

**Regression runs.**

| Command | Result |
|---|---|
| `check-plan-thresholds.sh` | pass |
| `mvn test -pl node` | 922 run, 0 failures, 44 skipped |
| `mvn test -pl node -Dgroups=gpu` | 313 run, 0 failures, 7 skipped |
| `mvn verify -pl juno-master -Pgpu -Dit.model.path=<tinyllama>` (after `mvn install -DskipTests`) | 10 of 10 |
| `smoke-gpu-residency.sh --models tinyllama...,mistral...` (unmodified) | 0 failures: every layer of every node runs whole (8 of 8, 7 of 7, 22 of 22 in cluster, 11 of 11 and 10 of 10 on Mistral 7B); greedy identical on against off (32 tokens); GPU memory flat both modes; cluster pipeline and tensor answer, equal local mode's output, no node JVM left |
| `smoke-gpu-residency.sh --models llama-1-30b --no-cluster --requests 8` | 0 failures: region active and whole on 20 of 20 GPU layers; greedy identical on against off; GPU MiB per request off 7434 7482 7502 7508 7522 7524 7524 7534 (26 MiB over requests 4 to 8), on 7528 7528 7530 7530 7530 7534 7538 7538 (8 MiB); limit 32. The region-off creep read 40 MiB at scope item 4 and 92 on the pre-change build; it passing here is the reading moving, not a fix, and it stays with the mirror-budget criterion (decision 5) |

Not run in this step: the 11-module unit reactor (only `node`, and help text in `juno-player`, changed; `juno-player`
compiled in `mvn install`), the real-model `ModelLiveRunnerIT`, the other smoke scripts, and the vision and LoRA gates
(LoRA playback announces the region as unsupported and keeps its path). Those belong to the tier's closing matrix.

**Indicative throughput, not scorable** (`PIN=0 bash dist/gpu-residency-whole-layer-ab/run-gate.sh`, clocks not
pinned, one repetition per run, published as
[`20261006T073749Z-gpu-residency-whole-layer-ab-unpinned`](../perf-compare/20261006T073749Z-gpu-residency-whole-layer-ab-unpinned/INDEX.md)),
medians of three:

| Model | Region | tg off | tg on | tg on/off | alloc/token off / on | GC pause in span, ms, off / on | GPU tg ratio off / on |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | whole layer | 60.33 | 116.91 | 1.938 | 47.7M / 38.1M | 8.0 / 0.0 | 0.311x / 0.649x |
| mistral-7b | whole layer | 21.90 | 36.36 | 1.660 | 223.7M / 193.6M | 14.1 / 0.0 | 0.587x / 0.990x |
| qwen2.5-3b | declined | 27.42 | 26.30 | 0.959 | 143.3M / 143.6M | 0.0 / 0.0 | 0.404x / 0.378x |
| Phi-3.5-mini | declined | 29.78 | 29.52 | 0.991 | 211.6M / 212.2M | 12.2 / 11.6 | 0.494x / 0.494x |

Every on repetition is far above every off repetition on both region models. Against the attention-only region's
unpinned reading (scope item 4: 73.83 and 24.34 t/s on) generation is about 1.6x (TinyLlama) and 1.5x (Mistral 7B)
faster. **Read against the 0.70x end-of-plan tg target, not scored** (unpinned, one repetition; the closing sweeps
score it): Mistral 7B reads 0.990x with the region on, above the target, but only with `--gpu-residency on`, which is
off by default (item 6 puts the default to the owner). Phi-3.5-mini stays at 0.494x until item 8's second part
gives its handler the region. Qwen2.5-3B (0.959, bound 0.95) is the closest declined row and the one to watch in the
pinned run; TinyLlama's pp on/off reads 0.923, as the attention-only region's unpinned reading did (0.903, then 0.970
pinned), although prefill does not run the region. Greedy text parts on against off after about 20 to 30 tokens on
this harness's prompt on both region models, as it already did with the attention-only region (the region's norms sum
in a different order than the default path's CPU norms); each mode is identical across its three runs.

**Owed to the owner: the pinned A/B.** The item's tg threshold (region on `>= 1.0x` region off, same-hour, pinned) is
scored when the item's other parts have landed; reading it now tells whether this part holds before Phi-3 and Qwen3
are built on it. Prepared at `dist/gpu-residency-whole-layer-ab/`: `candidate-shaded.jar` (sha256 `ade0bb660c5e2065`,
this working tree) and `run-gate.sh`, the item 4 gate with its paths renamed (A B A B A B, `--gpu-residency off`/`on`,
four sweep models at `n_prompt=128`, `--pin-clocks`; tg >= 1.00 on TinyLlama and Mistral 7B, >= 0.95 on Qwen2.5-3B and
Phi-3.5-mini). Command: `bash dist/gpu-residency-whole-layer-ab/run-gate.sh`.

**Docs.** `docs/howto.md` (the `--gpu-residency` row), `docs/agent-arch.txt` (`ResidentQkvPath`, new `ResidentLayerTail`,
`PrefillWindowKernels`, `GpuResidencyOptions`), the flag's help in `ConsoleMain`, `scripts/run.sh` and
`compare-llama-cpp.sh`, the `--gpu-attention off` console notice, `CHANGELOG.md`.

**Out-of-tier changes.** None. The table kernel changes the attention-only region (the `--gpu-attention on` decode
path with some layers' tails off the device) as well: one upload fewer per layer.

**Next.** Scope item 8's second part: the region for `Phi3TransformerHandler` and `Qwen3TransformerHandler` (split-half
RoPE through `rope_split_half`, Phi-3's LongRoPE factors folded into the inverse-frequency table, Qwen3's per-head Q/K
norms added or announced), then Phi-3's prefill RoPE and attention in the prefill region, then the KV mirror budget.

### 2026-10-06: scope item 8, first part: pinned A/B (owner run): met

`bash dist/gpu-residency-whole-layer-ab/run-gate.sh`, clocks pinned on all six runs (08:02Z to 08:19Z), jar
`ade0bb660c5e2065` on both sides, the flag alternated off, on, off, on, off, on; published as
[`20261006T080204Z-gpu-residency-whole-layer-ab`](../perf-compare/20261006T080204Z-gpu-residency-whole-layer-ab/INDEX.md).
Re-scored from the run files with the script's method (medians of three), since its console output was not kept.

| Model | Region | tg off | tg on | tg on/off | Threshold | pp on/off (not gated) | alloc/token off / on | GPU tg ratio off / on |
|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | whole layer | 55.24 | 121.95 | **2.208** | >= 1.00, met | 0.901 | 48.1M / 38.5M | 0.323x / 0.677x |
| mistral-7b | whole layer | 20.20 | 34.66 | **1.716** | >= 1.00, met | 0.958 | 223.0M / 193.1M | 0.546x / 1.008x |
| qwen2.5-3b | declined | 26.58 | 25.48 | **0.959** | >= 0.95, met | 0.978 | 143.2M / 143.1M | 0.392x / 0.377x |
| Phi-3.5-mini | declined | 28.71 | 28.46 | **0.991** | >= 0.95, met | 0.993 | 210.3M / 209.3M | 0.515x / 0.498x |

On both region models every on repetition is far above every off repetition. GC pause inside the token span falls
from 8.3 and 15.8 ms to 0. The gain over region off was +14.5% (TinyLlama) and +9.6% (Mistral 7B) with the
attention-only region (scope item 4's pinned gate) and is +120.8% and +71.6% with the whole layer.

**Read against the 0.70x end-of-plan GPU tg targets, not scored** (a gate A/B at `n_prompt=128`, one repetition per
run; the closing sweeps score them): Mistral 7B reads 1.008x with `--gpu-residency on`, above its target, but the flag
is off by default and the default path reads 0.546x; item 6 puts the default to the owner. Phi-3.5-mini reads 0.498x
and moves only when item 8's second part gives its handler the region.

**Two readings noted, not attributed, not gated.**
- Qwen2.5-3B reads 0.959 against its 0.95 bound, the same as unpinned. It declines the region (the handler never
  builds it), so on and off run the same code; the attention-only region's pinned gate read 1.022 on this model. Its
  lowest repetition is the last (`on-3` 24.30). If item 8's later gates read it below 0.95, the first thing to check
  is whether anything runs for a declined model under the flag.
- TinyLlama prefill on/off reads 0.901 (Mistral 7B 0.958, the declined models 0.978 and 0.993), although the
  prefill window does not run the decode region. In `on-1` against `off-1` the measured prefill forward pass reads
  59.4 against 51.6 ms, while the whole request reads 79 against 89 ms. The off side spreads 13% by itself. The
  `--device-spans` run owed at the end of item 8 reads the prefill window and should attribute it.

**Out-of-tier changes.** None.

### 2026-10-06: implementation step 5, scope item 8, second part (the region for the Phi-3 and Qwen3 handlers)

**Plan versus code, re-verified first** (HEAD `3761abc`, tree clean apart from the untracked `.github/`; no gate or
sweep running; `check-plan-thresholds.sh` passes). `Phi3TransformerHandler` and `Qwen3TransformerHandler` had no
decode region: every decode layer round-tripped each projection, rotated on the host and went through
`GpuAttentionMirror` for the KV append and attention. The split-half device RoPE (`rope_split_half`) existed, as the
item says. One claim was false, corrected in scope item 8's text: Phi-3's LongRoPE cannot be folded into the
inverse-frequency table (Phi-3.5-mini's `rope.scaling.attn_factor = 1.1902381` scales cos and sin, and the host
builds the angle in float). It changes the mechanism, not the scope or the order, so it was corrected rather than
escalated. Qwen3-1.7B declares no YaRN scaling and every projection is K-quant (`Q4_K`, `Q6_K` for V and down);
Phi-3.5-mini's fused Q/K/V is `Q5_K`, gate/up `Q4_K`, down `Q6_K`, all on the packed path.

**Design.**
- `ResidentRope` (new interface): what the region rotates q and k with. `CudaRope` implements it; new `CudaPhi3Rope`
  is Phi-3's rotation, a new kernel entry `rope_ext_split_half` (`RopeKernel.launchExtSplitHalf`) that repeats
  `Phi3Rope.ropeExt`'s float arithmetic step for step, with the selected factors uploaded once and the magnitude
  scale as an argument. It refuses the positions `Phi3RopeConfig.requirePosition` refuses; the handler also checks
  before any device work. The two existing PTX entries compile to the same code (diffed).
- `DeviceQ4KMatrix.rowSlice` (non-owning row view): the region multiplies Phi-3's fused Q/K/V and gate/up through
  views, so the region and `ResidentLayerTail` needed no Phi-3 code at all.
- `ResidentQkvPath.HeadNorms` (Qwen3): q and k are projected into region scratch and each head is normalized into
  place by the existing norm kernel (`RmsNormKernel.launchResident`, one row per head) before RoPE; a layer without
  both norms is not eligible. `create(..., rope)` and `create(..., rope, headNorms)` overloads; the LLaMA handler's
  path is unchanged. `ResidentQkvPath.activate` is the startup notice all three handlers share (moved out of
  `LlamaTransformerHandler`, text unchanged); `unsupportedReason(ctx, cfg)` is the shape-only check.
- Handlers: each builds the region under `--gpu-residency`, leases one region per token, grows the mirror before the
  call (`GpuAttentionMirror.reserve`, new, the mirrors' retire-on-OOM contract), writes host KV first and marks the
  mirror, and returns the region's layer output when the layer ran whole. Qwen3's shared static attention helper is
  split (`attendAndProject`) so the region's q, k and v can feed it when the region does not attend; the MoE handler
  calls it unchanged. Qwen3 YaRN files, a query width different from the hidden size, and non-K-quant weights
  decline the region with a log notice.
- `GpuResidencyOptions.unsupportedArchitectureReason` no longer lists `phi3` and `qwen3`; `ForwardPassHandlerLoader`
  announced "has no device-resident decode region" for every dedicated handler and now does so only for those
  without one (`phi2`, `qwen3moe`); before this fix it would have printed a false notice for Phi-3 and Qwen3.

**Defect found by the smoke and fixed in this step.** `smoke-gpu-residency.sh`'s tensor-parallel cluster on
Phi-3.5-mini failed: `cudaMalloc failed: rc=2` out of the forward pass. On this host each tensor-parallel node loads
the whole Phi-3 model, three copies fill the card, and the region opens its per-call buffers on first use. The same
launch with the region off answers (a scratch copy of the smoke with the cluster's flag switched; the shipped script
was not edited). Fix: `ResidentQkvPath.run` returns `null` (every handler's "keep the existing path") when opening a
region runs out of device memory, logs once and stops opening regions. Test first:
`ResidentQkvPathDeviceFullTest` fills the real card and was seen failing with the same `rc=2`; its fill needed two
corrections before it was reliable (collect and top up, since earlier tests' cleaners free memory; fill down to
256-byte requests, since the runtime places small requests in pages it already holds), and a planted fault (the
out-of-memory error rethrown) fails it again.

**Tests, written first** (README rule 3):

| Test | Seen failing first? | Now |
|---|---|---|
| `CudaPhi3RopeTest` (5): one row at positions 0 to 4095 with factors and the 1.19 scale, a model without factors and with a linear scale, a 37-row window, a column range, positions needing the long factors refused | yes, 4 of 5 against a stub that does not launch (the refusal case passes on the stub) | pass, bit-identical to `Phi3Rope.ropeExt` |
| `DeviceQ4KMatrixRowSliceTest` (3): each part's product equals its rows of the fused product for Q4_K, Q5_K, Q6_K; a view owns nothing; ranges outside refused | no: `rowSlice` was written first, so a regression test that passed on its first run | pass |
| `ResidentQkvPathPhi3Test` (5): Phi-3 layout (fused Q5_K Q/K/V and Q4_K gate/up through views, Q6_K down, 96-wide heads, GQA 2, extended RoPE): q/k/v and the whole layer bit for bit at positions 0 to 4095, a 30-token leased decode through three layers, concurrent leases, refusal at 4096 | no: composes pieces tested above; passed on its first run | pass |
| `ResidentQkvPathQwen3Test` (5): Qwen3 layout (per-head norms, split-half, 128-wide heads, Q6_K V and down): q/k/v and the whole layer bit for bit at positions 0 to 30000, leased decode, concurrent leases, a layer without both head norms declined | yes, 5 of 5 against a stub that ignored the head norms | pass |
| `Phi3TransformerHandlerGpuResidencyTest` (2, real Phi-3.5-mini): greedy token equal on against off at all 24 positions, logits within 0.5 (measured 0.301); whole layer: zero op-at-a-time decode copies, residual uploaded once per token, one packed download per layer | yes (region not active) | pass: **2 H2D and 33 D2H per token on 32 layers** |
| `Qwen3TransformerHandlerGpuResidencyTest` (2, real Qwen3-1.7B): the same | yes (region not active) | pass: **2 H2D and 29 D2H per token on 28 layers** |
| `ResidentQkvPathDeviceFullTest` (1): above | yes (`rc=2`) | pass |
| `GpuResidencyOptionsTest`: `phi3` and `qwen3` run the region, `phi2` and `qwen3moe` decline | updated with the code | pass, 7 of 7 |

**Two Qwen3 handler-test decisions, recorded.** (1) Its first input reused the TinyLlama token ids the LLaMA and
Phi-3 tests use, which are gibberish in Qwen3's vocabulary; greedy tokens parted at position 13, where the flag-off
run's own top two logits were 0.0078 apart and the on/off difference (0.196) sat in the same 0.12 to 0.31 band as
every other position. The input was changed, not the check: the test now decodes the flag-off handler's own greedy
continuation of five opening tokens ("The capital of France is Paris. The capital of Germany is ...") and requires
the region to reproduce every token, which it does. (2) On that text the largest logit difference is 0.644 (logits of
magnitude 15 to 32, three CPU-against-GPU norms per layer); the bound is 1.3, about twice the measurement, the
LLaMA-family test's convention. Both are in the test's comments.

Planted faults, each in its own run, file restored after (`cmp` against a saved copy): the RoPE magnitude scale
dropped (`CudaPhi3RopeTest` 2 of 5, `ResidentQkvPathPhi3Test` 4 of 5 fail); K read from the V rows of the fused tensor
(Phi-3 handler test fails at position 1); K heads normalized with Q's weights (Qwen3 handler test fails at position 1);
Qwen3 rotated with base 1e4 instead of 1e6 (fails at position 9); the region's out-of-memory error rethrown
(`ResidentQkvPathDeviceFullTest` fails).

**Regression runs.**

| Command | Result |
|---|---|
| `check-plan-thresholds.sh` | pass |
| `mvn test -pl node` (before the device-full fix) | 944 run, 0 failures, 44 skipped |
| `mvn test -pl node -Dgroups=gpu` (before the device-full fix) | 335 run, 0 failures, 7 skipped |
| after the fix: the eight residency classes above plus `ResidentQkvPathTest` and `LlamaTransformerHandlerGpuResidencyTest` | 0 failures |
| `mvn test -pl node` (after the fix) | 945 run, 0 failures, 44 skipped |
| `mvn test -pl node -Dgroups=gpu` (after the fix) | 336 run, 0 failures, 7 skipped |
| `smoke-gpu-residency.sh --models Phi-3.5-mini-instruct-Q4_K_M,Qwen3-1.7B-Q4_K_M` (unmodified, after the fix, `20261006T152413Z`) | 0 failures: region active and whole on every layer of every node (Phi-3.5-mini 11 of 11, Qwen3-1.7B 10 of 10); greedy identical on against off (32 tokens); GPU memory flat in both modes; cluster pipeline and tensor answer with local mode's output, no node JVM left. Before the fix (`20261006T150246Z`) the tensor cluster failed as described |
| `smoke-gpu-residency.sh --models tinyllama...,mistral...` (unmodified, `20261006T152632Z`) | 0 failures |

Not run in this step: the eleven-module unit reactor (only `node` and help text in `juno-player` changed;
`juno-player` compiled in `mvn install`), the real-model `ModelLiveRunnerIT` and `-Pgpu` ITs, `smoke-gpu-residency.sh`
on llama-1-30b, the other smoke scripts, and the vision and LoRA gates. They belong to the tier's closing matrix.

**Indicative throughput, not scorable** (`PIN=0 bash dist/gpu-residency-phi3-qwen3-ab/run-gate.sh`, clocks not
pinned, published as
[`20261006T152821Z-gpu-residency-phi3-qwen3-ab-unpinned`](../perf-compare/20261006T152821Z-gpu-residency-phi3-qwen3-ab-unpinned/INDEX.md)),
medians of three:

| Model | Region | tg off | tg on | tg on/off | alloc/token off / on | GC pause in span, ms, off / on | GPU tg ratio off / on |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | whole layer | 66.03 | 130.62 | 1.978 | 48.0M / 38.8M | 7.1 / 0.0 | 0.340x / 0.676x |
| mistral-7b | whole layer | 22.28 | 36.36 | 1.632 | 223.9M / 193.6M | 14.2 / 0.0 | 0.605x / 0.985x |
| Phi-3.5-mini | whole layer (new) | 31.34 | 48.20 | 1.538 | 207.5M / 174.1M | 11.5 / 13.2 | 0.524x / 0.805x |
| Qwen3-1.7B | whole layer (new) | 37.62 | 63.93 | 1.699 | 111.9M / 97.5M | 0.0 / 16.4 | 0.327x / 0.557x |
| qwen2.5-3b | declined | 29.21 | 29.35 | 1.005 | 143.5M / 143.3M | 0.0 / 0.0 | 0.412x / 0.413x |

**Read against the 0.70x end-of-plan GPU tg targets, not scored** (unpinned, one repetition; the closing sweeps score
them): Phi-3.5-mini reads 0.805x and Mistral 7B 0.985x with `--gpu-residency on`, both above target, but the flag is off
by default and the default path reads 0.524x and 0.605x; scope item 6 puts the default to the owner. **One reading
to carry to the standing allocation gate:** the GC pause in the token span rises on Qwen3-1.7B on every repetition
(one young collection now inside the 64-token span, 1 against 0, in a 4 GiB heap while allocation per token falls by
13%), and slightly on Phi-3.5-mini; a threshold effect, not gated here.

**Owed to the owner: the pinned A/B for this part.** Prepared at `dist/gpu-residency-phi3-qwen3-ab/`:
`candidate-shaded.jar` (sha256 `9659ff62d527578b`, this working tree) and `run-gate.sh` (A B A B A B,
`--gpu-residency off`/`on`, the four sweep models plus Qwen3-1.7B at `n_prompt=128`, `--pin-clocks`; tg >= 1.00 on
TinyLlama, Mistral 7B, Phi-3.5-mini and Qwen3-1.7B, >= 0.95 on Qwen2.5-3B). Command:
`bash dist/gpu-residency-phi3-qwen3-ab/run-gate.sh`.

**Docs.** `docs/howto.md` (the `--gpu-residency` row: the Phi-3 and Qwen3 handlers, what still declines, and that
file-dependent declines are log-only), `docs/agent-arch.txt` (`ResidentQkvPath`, new `ResidentRope` and
`CudaPhi3Rope`, `RopeKernel`, `CudaRope`, `DeviceQ4KMatrix.rowSlice`, `GpuAttentionMirror.reserve`,
`GpuResidencyOptions`), `ConsoleMain` help, `CHANGELOG.md`.

**Known limitation, recorded.** A decline that depends on the file rather than the architecture (non-K-quant weights,
a Qwen3 file with YaRN RoPE, a query width different from the hidden size) is announced in the log only, which the
console shows with `--verbose`; the console's startup notice reads the architecture. This is the existing behaviour
for LLaMA-family files without K-quant weights; no YaRN Qwen3 file is on disk.

**Out-of-tier changes.** None. The device-full fallback also covers the LLaMA-family region (the same `run`), where a
full card would have failed the request in the same way.

**Next.** Scope item 8's third part: Phi-3's RoPE and attention inside the prefill-window region (the 70% bytes
threshold), then the KV mirror budget on a partially offloaded model, then the published `--device-spans` run, the
prefill-window re-verification and the pinned A/B on the final region.

### 2026-10-06: implementation step 5, scope item 8, third part (Phi-3's prefill RoPE and attention in the prefill region)

**Plan versus code, re-verified first** (HEAD `5e4c913`, tree clean apart from the untracked `.github/`; no gate,
sweep or mvn run in flight, the matching `pgrep` lines were idle wait loops of earlier sessions; `check-plan-thresholds.sh`
passes). `Phi3TransformerHandler.openPrefillRegion` built the region with no RoPE and no attention
(`create(..., null, 0f, false)`); `PrefillWindowRegion.create` refused a fused Q/K/V projection with device RoPE, and
`runLayer` kept attention out of the region for a fused layer (`layers[li].qkv() == null`). The window therefore
downloaded the fused Q/K/V rows, rotated with `Phi3Rope.ropeExt` and attended on the host side, and uploaded the
attention output, every layer. The device LongRoPE kernel the item calls for exists since the second part
(`CudaPhi3Rope`, `rope_ext_split_half`, bit-identical to the host rotation on a 37-row window by `CudaPhi3RopeTest`).
GQA attention takes Phi-3.5-mini's 96-wide heads (`GqaAttentionTiledTest`). No drift that changes scope; the item
said the LongRoPE "folding" serves the prefill window, and per the 2026-10-06 correction it is the dedicated kernel
that does.

**Design.**
- `PrefillWindowRegion`'s RoPE is a `ResidentRope` (was `CudaRope`), built by a factory: the existing
  `create(..., RopePairing, theta, attentionKernel)` builds `CudaRope` as before; a new
  `create(..., Phi3RopeConfig, attentionKernel)` builds `CudaPhi3Rope`. A fused projection with device RoPE is
  now accepted (a fused projection with biases is still refused).
- A fused layer keeps its one GEMM (as on the host path; a GEMM per row range could round differently) and,
  with device RoPE, is split on the device by a new kernel, `split_qkv` in `prefill_window.cu` (exact copies,
  timed as the `juno.DeviceCompute` site `split_qkv`), into the region's q, k and v buffers; from there the
  layer runs as on the LLaMA family: RoPE, K and V cast into the mirror, attention, the rest of the layer, one
  download of k and v. Without device RoPE the fused rows still come back to the host as before.
- `Phi3TransformerHandler`: the region gets `ropeCfg` and `attentionKernel = gpuAttention != null`; the
  per-layer path is the LLaMA handler's (host KV written from the region's rows, then `markWritten`; a layer the
  region does not attend falls back to rotated q/k/v and `writeKvAndAttend`, with a mirror the region retired
  passed as `null`). `ropeCfg.requirePosition(startPos + W - 1)` runs before any device work, as on decode.
- The region's activations are named (`prefill q`, `prefill k`, `prefill v`, `prefill qkv`, `prefill attention`;
  were all `resident activation`), so their copies are separate `juno.DeviceStaging` sites.
- `split_qkv` added to the metrics extractor's always-written compute sites (`DeviceSpanBucket`) and to
  `prefill-breakdown.sh`'s device-elementwise term, so the breakdown does not count it as GEMM time.
- The regenerated `prefill_window.ptx` differs from the committed one only by the new entry and renumbered block
  labels (`diff` with labels normalised: no other line); the committed PTX was first reproduced byte for byte by
  the local `nvcc` 12.0.

**Tests, written first** (README rule 3):

| Test | Seen failing first? | Now |
|---|---|---|
| `Phi3PrefillRegionAttentionTest` (new, `@Tag("gpu")`, real Phi-3.5-mini, one 512-token window under JFR): no `juno.Rope` or `juno.Attention` event, no `materialize(prefill qkv)` or `upload(prefill attention)` copy, H2D + D2H <= 793.5 MB | yes: a host RoPE event, 1,422.3 MB (the published 1,419 MB) | pass, 415.7 MB |
| `PrefillWindowKernelsParityTest.splitQkv_copiesEachPart`: Phi-3.5-mini's and a GQA shape at 1, 8, 9, 32 and 512 rows, every q, k, v row bit for bit | yes, against a stub that launched nothing | pass |
| `JfrMetricsExtractorDeviceComputeTest` (`split_qkv` added to its site list) | yes, the key missing | pass, 5 of 5 |
| `PrefillRegionHandlerParityTest` (unchanged; Phi-3.5-mini now attends inside the region) | regression test | pass: logits bit-identical region on against off at all six sites on all four models |

Planted fault, file restored after (`cmp` against a saved copy): K and V swapped in the split. First run
inconclusive: three cases failed on the test's device-wide free-memory check, which runs before the logits
comparison. Second run: only Phi-3.5-mini fails, at the logits (first window, relative L2 1.137, top-1 267 against 18041).

**One existing test changed, recorded.** `GpuAttentionHandlerParityTest` (Phi-3.5-mini and Qwen3-1.7B, GPU attention
on against off) failed on Phi-3.5-mini with "WINDOW: launches seen 0": the region now launches the kernel on its own
stream, past the dispatch observer the test holds each launch to the oracle through. The test now runs both sides
with the prefill region off, so its window check again sees every launch; the region's attention is held to that
region-off path bit for bit by `PrefillRegionHandlerParityTest`, so the kernel check still covers it. Not loosened:
same bounds, and the window check now reads 64 launches, worst relative L2 1.7e-6 (Qwen3-1.7B: 56, 3.5e-6).

**Regression runs.**

| Command | Result |
|---|---|
| `check-plan-thresholds.sh` | pass |
| `prefill-breakdown.sh --selftest`, `compare-llama-cpp.sh --selftest` | 16 of 16; all checks passed |
| `mvn test -pl metrics,node` | metrics pass; node 947 run, 5 failures, 44 skipped, before the test change above. One was `GpuAttentionHandlerParityTest` (above); four were device-wide free-memory checks: `PrefillRegionHandlerParityTest` (Phi-3.5-mini case, at the free-VRAM check, before its logits), `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`, `ResidentQkvPathTest.attentionAllocatesNothingPerCall` and `noDeviceMemoryRetained` |
| re-runs of the four classes, `-Dgroups=gpu`, desktop GPU use moving between 684 and 1,020 MiB | `GpuAttentionHandlerParityTest` 2 of 2; `ResidentQkvPathTest` 19 of 19 and `PrefillRegionHandlerParityTest` 4 of 4 on the third run (each failed one free-memory check on an earlier one: `noDeviceAllocationPerCall`, then two parity cases); `PrefillReserveDeviceTest` fails on every run (92.5 MB and 93.7 MB reported free with the card full, bound 64 MiB) |
| `PrefillReserveDeviceTest` on the pre-change build (`git archive 5e4c913`, built in a scratch tree) | fails the same way: 94.3 MB against 64 MiB. The test fills the card and reads the driver's free figure, with no Juno code between; it moves with the desktop's own GPU use and is not this change's |
| `smoke-long-prompt-prefill.sh --baseline-jar <pre-change jar>` (unmodified) | it accepts only `tinyllama` and `mistral`, so it cannot take Phi-3.5-mini (`--models Phi-3.5-mini` exits on an unknown key); on its defaults, TinyLlama and Mistral 7B, static and continuous, 128/512/2048: 48 checks pass, greedy text identical to the pre-change jar everywhere (the region's RoPE factory refactor touches their path) |
| greedy text, Phi-3.5-mini, pre-change jar against candidate | identical on all six A/B runs: the 512-token prefill request and the 64-token generation request |

Not run in this step: the eleven-module unit reactor (only `node`, `metrics` and a harness script changed), the
real-model `ModelLiveRunnerIT` and `-Pgpu` ITs, `smoke-gpu-residency.sh` (decode-only), the vision and LoRA gates.
They belong to the tier's closing matrix.

**Measurements.**

*Published `--device-spans` run*, [`20261006T184154Z-phi3-prefill-region-attention`](../perf-compare/20261006T184154Z-phi3-prefill-region-attention/INDEX.md)
(unpinned; jar `326cbb9ff2bd86e2`, this tree): per 512-token window 6.6 MB H2D and 408.1 MB D2H, **414.8 MB**,
identical on three repetitions; **-84.3% against 2,645 MB** (threshold -70%), from 1,419 MB. What still crosses: K and
V rows for the host KV cache (200.9 MB each), the residual once each way (6.3 MB each), the attention tables
(0.3 MB). Breakdown: prefill 763.7 ms; host RoPE, host attention, attention copies, KV mirror copies 0; region
attention 73.7 ms (9.7%). `prefill-breakdown.sh` exits 1 on its residue flag (10.9% against 5%): 83.6 ms, against
88.7 ms (2.9% of 3,012 ms) at `20261002T050741Z`, so the absolute residue did not grow and its share rose with the
faster window. Reported, flag not relaxed. The prefill-window re-verification of scope item 8's main criterion
("zero activation device-to-host copies inside a prefill window") holds on Phi-3.5-mini in this run: the only
downloads are the K and V rows and the final residual. The other three sweep models are still owed it, with the
item's closing `--device-spans` run.

*Indicative A/B* (`PIN=0 bash dist/phi3-prefill-region-ab/run-gate.sh`, unpinned, Phi-3.5-mini, `n_prompt=512`,
pre-change jar `a4f334f20440e3e1` against candidate `326cbb9ff2bd86e2`, A B A B A B), published as
[`20261006T184535Z-phi3-prefill-region-ab-unpinned`](../perf-compare/20261006T184535Z-phi3-prefill-region-ab-unpinned/INDEX.md):

| Metric | Pre-change | Candidate | B/A |
|---|---|---|---|
| pp t/s, median (all three) | 375.67 (373.23, 375.67, 381.62) | 705.19 (712.31, 680.26, 705.19) | **1.877** |
| tg t/s, median | 31.03 | 31.26 | 1.007 |
| GPU pp ratio at 512, median | 0.306x | 0.578x | |
| GC max pause, ms | 8 to 9 | 8 | |
| Allocated bytes per generated token | 207.5M | 207.6M | |

**Read against the end-of-plan rows, not scored** (unpinned; the closing sweeps score them): Phi-3.5-mini, the binding
model of the GPU pp row at 512 (>= 0.40x; reference 0.226x), reads 0.578x on the candidate here against 0.306x on the
pre-change build in the same session.

**Owed.**
- The owner: the pinned form of the A/B above, `bash dist/phi3-prefill-region-ab/run-gate.sh` (prepared: both jars
  and the script; bound pp and tg >= 0.95). Not a criterion of this part (its threshold is the bytes figure); it
  is read with the tier's closing no-regression gate. Also still owed from the second part:
  `bash dist/gpu-residency-phi3-qwen3-ab/run-gate.sh`.
- The executor, at item 8's end: the published `--device-spans` run and the prefill-window re-verification on the
  other sweep models.

**Docs.** `docs/howto.md` (the prefill-region table's Phi-3 row; `split_qkv` among the `device_layer` sites),
`docs/agent-arch.txt` (`PrefillWindowRegion`, `PrefillWindowKernels`, the `juno.DeviceCompute` site lists),
`prefill-breakdown.sh` comment, `CHANGELOG.md`.

**Out-of-tier changes.** None. The region's activation names change the site names its copies are recorded under
(`materialize(resident activation)` becomes `materialize(prefill k)` and so on) for every model; the totals and the
breakdown's terms do not move.

**Next.** Scope item 8's fourth part: the KV mirror budget on a partially offloaded model (llama-1-30b at `auto`), then
the published `--device-spans` run, the prefill-window re-verification and the pinned A/B on the final region.

### 2026-10-06: scope item 8, third part: pinned A/B (owner run): bound met

`bash dist/phi3-prefill-region-ab/run-gate.sh`, clocks pinned on all six runs (19:27Z to 19:31Z), pre-change jar
`a4f334f20440e3e1` against candidate `326cbb9ff2bd86e2`, alternated A B A B A B, Phi-3.5-mini at `n_prompt=512`;
published as [`20261006T192743Z-phi3-prefill-region-ab`](../perf-compare/20261006T192743Z-phi3-prefill-region-ab/INDEX.md).
Re-scored from the run files with the script's method (medians of three), since its console output was not kept.
Every row scorable, every prefill 512 of 512.

| Metric | Pre-change | Candidate | B/A | Bound |
|---|---|---|---|---|
| pp t/s | 378.93 (378.93, 375.88, 380.39) | 718.79 (718.79, 723.40, 706.62) | **1.897** | >= 0.95, met |
| tg t/s | 31.21 (31.00, 31.23, 31.21) | 30.89 (30.67, 31.17, 30.89) | **0.990** | >= 0.95, met |
| GPU pp ratio at 512 | 0.303x | 0.577x | | reading |
| GC max pause, ms | 8 to 10 | 8 | | reading |
| Allocated bytes per generated token | 206.7M | 207.6M | | reading |

Every candidate prefill reading is above every pre-change one; the unpinned reading (1.877, 1.007) agrees.
Generation does not run through the prefill region; 0.990 sits inside either side's own spread (30.67 to 31.23).

**Read against the end-of-plan rows, not scored** (a gate A/B, one repetition per run; the closing sweeps score
them): Phi-3.5-mini, the binding model of the GPU pp row at `n_prompt=512` (>= 0.40x; reference 0.226x), reads
0.577x on the candidate against 0.303x on the pre-change build, pinned, same session. This is not the tier's closing
0.95x gate (that is read on every sweep model against the pre-tier jar at the tier's close); it confirms this part
holds before the next is built on it.

**Out-of-tier changes.** None.

**Still owed to the owner:** nothing from item 8's first three parts (the second part's pinned A/B is recorded below).

### 2026-10-06: scope item 8, second part: pinned A/B (owner run): met

`bash dist/gpu-residency-phi3-qwen3-ab/run-gate.sh`, clocks pinned on all six runs (20:03Z to 20:20Z), jar
`9659ff62d527578b` on both sides (the second part's tree, committed as `5e4c913`), the flag alternated off, on, off,
on, off, on; published as
[`20261006T200344Z-gpu-residency-phi3-qwen3-ab`](../perf-compare/20261006T200344Z-gpu-residency-phi3-qwen3-ab/INDEX.md).
Re-scored from the run files with the script's method (medians of three), since its console output was not kept.
Every row scorable on all six runs.

| Model | Region | tg off | tg on | tg on/off | Threshold | pp on/off (not gated) | alloc/token off / on | GPU tg ratio off / on |
|---|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | whole layer | 64.00 | 131.47 | **2.054** | >= 1.00, met | 0.913 | 47.9M / 38.8M | 0.329x / 0.672x |
| mistral-7b | whole layer | 22.16 | 36.54 | **1.649** | >= 1.00, met | 0.990 | 223.8M / 191.7M | 0.596x / 0.982x |
| Phi-3.5-mini | whole layer (new) | 31.22 | 48.22 | **1.545** | >= 1.00, met | 1.001 | 207.5M / 180.7M | 0.515x / 0.799x |
| Qwen3-1.7B | whole layer (new) | 37.25 | 63.18 | **1.696** | >= 1.00, met | 0.972 | 111.8M / 97.7M | 0.320x / 0.546x |
| qwen2.5-3b | declined | 28.78 | 28.95 | **1.006** | >= 0.95, met | 0.989 | 143.3M / 143.2M | 0.404x / 0.407x |

On every region model every on repetition is above every off repetition. The pinned figures agree with the unpinned
ones (1.538 and 1.699 on the two new handlers).

**Read against the 0.70x end-of-plan GPU tg targets, not scored** (a gate A/B at `n_prompt=128`, one repetition per
run; the closing sweeps score them): with `--gpu-residency on`, Phi-3.5-mini reads 0.799x and Mistral 7B 0.982x, both
above 0.70x; the default path (flag off) reads 0.515x and 0.596x. The flag is off by default; scope item 6 puts the
default to the owner on the final region.

**Readings carried to the standing allocation gate, not gated here.**
- Qwen3-1.7B's GC max pause rises from 0 to 15-19 ms with the region on, on every repetition, while allocation per
  generated token falls 13% (111.8M to 97.7M): one young collection now lands inside the 64-token span. The unpinned
  run showed the same. The standing gate bounds GC pause in the token span at <= 5 ms where the pre-tier reading is
  under 5 ms, so this is the reading to check first at the tier's close.
- TinyLlama's pp on/off reads 0.913 although prefill does not run the decode region (0.901 on the whole-layer gate);
  the item's closing `--device-spans` run is to attribute it, as recorded at the first part's gate.

**Out-of-tier changes.** None.

### 2026-10-06: implementation step 5, scope item 8, fourth part (the KV mirror budget on a partially offloaded model)

**Plan versus code, re-verified first** (HEAD `3b27aa3`, tree clean apart from the untracked `.github/`; no gate,
sweep or mvn run in flight, the matching `pgrep` lines were idle wait loops of earlier sessions;
`check-plan-thresholds.sh` passes). `LlamaTransformerHandler.inferenceReserveBytes` kept the GPU-attention KV
mirror free at `DeviceKvCache.INITIAL_SEQ_CAPACITY` (64 positions) on every layer of the shard and nothing for
its growth; growth that ran out of device memory retired the mirror and attention fell back to the CPU, as the
item says. Mirrors are allocated for all of a shard's layers at 64 positions but only GPU-resident layers write
to them, so only those grow. The adaptive prefill chunk (`PrefillChunkDefaults`) took half of whatever was free
after the upload, so any reserve kept for the mirror would have gone to a wider window. No context-size flag
exists anywhere, and no later tier plans one. No drift that changes scope.

**Owner decision (2026-10-06, before implementation), as recommended:** a fixed 512-token reserve. The other
options put to the owner were a new context flag, trading a weight layer at run time when a long prompt
arrives, and stating the number without a code change.

**Design.**
- `DeviceScratchBudget.RESERVED_MIRROR_TOKENS = 512` and `kvMirrorReserveBytes(deviceLayers, hostLayers, kvDim)`:
  the device layers' mirrors at 512 positions, the host layers' at their initial 64, and the old buffer one
  layer holds while it grows (at most 256 positions).
- `LlamaTransformerHandler`: both `auto` upload loops evaluate the reserve per layer with the layer about to be
  uploaded counted (`scratchReserveBytes` + `mirrorReserveBytes(uploaded + 1, L)`), since the device-layer
  count is only known there. 0 under `--gpu-attention off`, as before.
- `ForwardPassHandler.kvMirrorReserveDeviceBytes()` (default 0; `LlamaTransformerHandler` reports its resolved
  device layers' reserve, `VisionAwareForwardPassHandler` delegates), and `PrefillChunkDefaults` sizes the window
  from free memory less the shards' summed reserve. A failed free-memory query stays a failed query.
- Phi-3 and Qwen3 handlers have no `auto` stop rule (`DeviceScratchBudget` is the LLaMA handler's only), so
  they report 0 and are unchanged; their partially offloaded case (Qwen3-30B-A3B) is not covered by this part.

**Tests, written first** (README rule 3), against stubs of the new API that kept today's behaviour (64-token
constant, today's mirror term, chunk sizer ignoring the reserve):

| Test | Seen failing first? | Now |
|---|---|---|
| `DeviceScratchBudgetTest.theMirrorIsReservedForAFiveHundredTwelveTokenPrompt` | yes (64) | pass |
| `onlyDeviceLayersAreReservedTheirGrownMirror`: 23 device and 37 host layers of the 30B, exact bytes | yes | pass |
| `theReservedMirrorHoldsTheThirtyBsFiveHundredTwelveTokenPrompt`: 299 MiB grown mirror covered | yes | pass |
| `aLayerThatWouldLeaveNoRoomForTheGrownMirrorIsNotUploaded`: a 310 MiB layer the old reserve uploads is refused | yes | pass |
| `negativeLayerCountsAreRejected` | yes | pass |
| `aModelThatFitsWholeReservesLittleForItsMirror` (Mistral 7B: 65 MiB), `noDeviceLayersReserveOnlyTheInitialMirror` | no: the stub's arithmetic agrees for these, so regression tests | pass. The Mistral case first asserted < 33 MiB, which was the executor's arithmetic error (32 x 512 x 2 x 1024 x 2 B = 64 MiB, plus 1 MiB growing); corrected to the exact 65 MiB |
| `PrefillChunkDefaultsTest.the_window_is_sized_from_what_the_reserved_kv_mirror_leaves`, `a_mirror_reserve_larger_than_free_memory_gives_the_smallest_window`, `mirrorReserveOf_sums_every_shard` | yes, 3 of 3 | pass |

The handler's per-layer stop rule has no unit test of its own (it needs a card filled to the reserve); it is
held by the real-model measurement below.

**Measurement**, [`20261006T212419Z-kv-mirror-reserve`](../perf-compare/20261006T212419Z-kv-mirror-reserve/INDEX.md)
(unpinned; layer counts and fallbacks do not depend on clocks): Tier 01C's `measure-reserve.sh`, llama-1-30b, one
node, `auto`, 30 GiB heap, a short request then a 508-token prompt, pre-change jar `1a959d18cab40409` (`3b27aa3`)
and candidate `fe69862b55a1a935`, one engine at a time.

| Build | GPU layers | Kept free | Free after load | Chunk | Fallbacks on the 508-token prompt | Prefill / 8 tokens, wall (single, unpinned) |
|---|---|---|---|---|---|---|
| pre-change | 22 | 196 MiB | 382 MiB | 448 | region prefill attention to the CPU | 2030 s / 160 s |
| candidate | **21** | 453 MiB | 697 MiB | 448 | **none** | 2127 s / 171 s |

The 512-token part of the criterion is met: the candidate logs no device-memory fallback of any kind on the
prompt. The cost is one weight layer, as forecast (a 512-token mirror on 22 layers is 286 MiB, one 30B layer is
about 310 MiB), and on this model it does not pay in wall time: both readings are about 5% slower, because the
39th CPU layer costs more than CPU attention over 508 positions. Single unpinned readings, recorded, not scored.
A first candidate attempt at a 22 GiB heap (the residency smoke's heap rule, below the launcher's 30 GiB for this
file) ran out of Java heap in the prompt's first window and was killed; it is published as a superseded attempt.
The baseline was not run at 22 GiB, so whether the extra CPU layer is what crossed that heap is not known.

**Regression runs.**

| Command | Result |
|---|---|
| `check-plan-thresholds.sh` | pass |
| unit reactor, `mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player` | 2,134 run, 0 failures, 49 skipped, BUILD SUCCESS (node 954) |
| `mvn test -pl node -Dgroups=gpu` | 338 run, 0 failures, 7 skipped (the known free-VRAM checks passed this time) |
| `smoke-gpu-residency.sh --models llama-1-30b --no-cluster --requests 8` (unmodified; decision 5's closing read) | region on: active on 20 of 20 GPU layers, GPU MiB 7424 7456 7456 7456 7456 7466 7466 7470 (14 MiB over requests 4 to 8: pass); greedy identical on against off. **Region off: 7416 7450 7450 7482 7502 7516 7538 7546, 64 MiB over requests 4 to 8, limit 32: fail** |

**Reading of the region-off creep (decision 5).** It is not fixed by this part. Its readings over requests 4 to 8
are now 92 (pre-scope-item-4 build), 40, 26 and 64 MiB: it moves between runs by more than the limit and has never
been attributed. The smoke runs three in-process nodes (20 GPU layers), not the one-node configuration measured
above, and its requests are short, so the mirror's growth reserve is not what it reads. With the region on, every
reading since scope item 4 passes.

Not run in this step: the real-model `ModelLiveRunnerIT` and `-Pgpu` ITs, the other smoke scripts, the vision and
LoRA gates, and a re-read of the sweep models' resolved prefill chunk. By arithmetic the reserve lowers the
adaptive chunk only slightly on the LLaMA-handler sweep models (Mistral 7B: 65 MiB less free out of several GiB,
about 50 rows off the 2601 Tier 01C read; Qwen2.5-3B 18 MiB), so every 2048-token prompt stays one window; the
tier's closing sweeps and `smoke-long-prompt-prefill.sh` re-read it.

**Docs.** `docs/howto.md` (device memory with `--gpu-layers auto`, the `--prefill-batch` row),
`docs/agent-arch.txt` (`DeviceScratchBudget`, `PrefillChunkDefaults`), `CHANGELOG.md` (Session 116).

**Out-of-tier changes.** None outside scope item 8's text ("decide how mirror growth is budgeted against weight
layers"). Recorded because it touches the launcher-facing chunk sizer in `coordinator` and the
`ForwardPassHandler` interface: a measurement boundary for partially offloaded models (one GPU layer fewer on
llama-1-30b at `auto`), none for models that fit whole (layer count unchanged; adaptive chunk a few percent
narrower, still above every published prompt length). No published baseline is invalidated: no sweep or gate uses
a partially offloaded model.

**Raised with the owner (decision 7): the region-off device-memory creep on llama-1-30b.** Decision 5 (b) moved it
here; the mirror budget does not remove it (64 MiB this run). (a) Attribute it now as its own item before the
tier's closing matrix: a per-request reading of `DeviceKvCache.allocatedBytes()`, the allocator's free bytes and
the region-off path's per-call device buffers across 16 requests, on three nodes. (b) Record it as a known
limitation of the region-off path on a model the card does not fit, carried to the tier's close, where item 6
(the `--gpu-residency` default) may make the region-on path the default and the off path the fallback.
Recommended: (a). An unattributed per-request device-memory growth on the default path is a leak until shown
otherwise, its readings straddle the limit, and item 6's default question should not be answered with it open.

**Next.** Decision 7. Then scope item 8's close: the published `--device-spans` run with the prefill-window
re-verification on the other sweep models, and the pinned A/B on the final region (owner), then scope item 5.

**Decision 7 taken (a) the same day (owner).** The region-off device-memory creep on llama-1-30b is attributed now,
as its own item, before the tier's closing matrix.

### 2026-10-07: decision 7 (a): the region-off device-memory creep attributed and fixed

**Cause.** On llama-1-30b at the card's capacity, the request's KV mirror allocation fails request after request
(the smoke's region-off log: "out of device memory for the attention KV mirror - running attention on the CPU for
this request" on all three shards; region on, one). `DeviceKvCache` leaked on exactly that path, in two places:
`newLayers` allocated one mirror per layer in a loop and, when layer *k* ran out, never closed layers 0 to *k*-1; and
the constructor allocated K, then V, and never freed K when V ran out. Every failed request therefore kept part of a
mirror set, and the next request started from a fuller card, so the leak also made the next failure more likely. The
region-off mode creeps more because its shards hit the failure more often. This matches every reading on record: a
creep only on the one model at the card's capacity, varying between runs (92, 40, 26, 64 MiB) with how many layers'
allocations happened to succeed before each failure.

**Fix** (`DeviceKvCache`): `newLayers` is all or nothing (closes the layers it allocated before rethrowing); the
constructor frees K when V fails. `grow` already freed its first tensor on failure.

**Tests, written first** (`DeviceKvCacheLifecycleTest`, `@Tag("gpu")`, the real card; sizes taken from the device's free
memory so the failure lands at a chosen point without filling the card; tolerance 256 MiB on the free reading against
leaks of gigabytes):

| Test | Seen failing first? | Now |
|---|---|---|
| `a_partly_allocated_set_of_layers_is_released_when_one_runs_out`: 12 layers each about a sixth of free memory | yes: mirror bytes still counted after the failure | pass |
| `a_mirror_whose_second_tensor_does_not_fit_releases_the_first`: one tensor at 60% of free memory | yes: device memory not given back | pass |

**Attribution run.** `smoke-gpu-residency.sh --models llama-1-30b --no-cluster --requests 8` (unmodified, jar
`3b8b2f53d1bd0a04`, `20261007T023945Z`): **0 failures**. Region off, GPU MiB 7348 7382 7412 7412 7412 7412 7412 7412
(**0 MiB** over requests 4 to 8, was 64 on the run before the fix); region on 7422 throughout (0 MiB); region active
on 20 of 20 GPU layers; greedy identical on against off. The same mirror-allocation fallbacks are logged as before
(off: three shards; on: one), so the run exercises the failing path and it no longer leaks. One run; the readings
before the fix never read flat on this model.

| Command | Result |
|---|---|
| `mvn test -pl node -Dgroups=gpu` | 340 run, 0 failures, 7 skipped |
| `mvn test -pl node` | 956 run, 0 failures, 44 skipped |

**Out-of-tier changes.** None: the mirror's memory on a partially offloaded model is scope item 8's (decision 5). A
measurement boundary only for a run that hits the mirror-allocation fallback (a model at the card's capacity); no
published baseline is such a run.

**Found while attributing, not changed: decision 8 raised with the owner.** Three scratch-growth sites free the old
device buffer and then allocate the new one without clearing the field: `CudaGqaAttention` (query, output, pointer
tables, lengths), `CudaRmsNorm` and `DeviceActivationBatch`. If that allocation runs out of memory, the field keeps the
freed pointer and the recorded size stays at the old one, so a later call that fits the old size writes through a
pointer the allocator may have handed to something else (a KV mirror, a weight), and `close` frees it a second time.
`CudaMatVec`'s scratch already clears its field first, which is the fix. Separately, `CudaMatVec.sgemv(float[] A, ...)`
allocates its three buffers before its `try`, so a failure on the second or third leaks the first. All four sit on
fallback paths a card at its capacity takes (the attention one is this tier's kernel path); none is known to have
produced a wrong output. (a) Fix all four in this tier now, tests first on the real card (fail a growth at a chosen
size, then assert the field is cleared and a following call reallocates); (b) fix only `CudaGqaAttention` here, as
attention is this tier's, and record the other three for Tier 14's audit; (c) record all four for later. Recommended:
(a). A use-after-free on a fallback is a silent-corruption risk on exactly the near-full card this tier has been
budgeting, the fix is one pattern already in the codebase, and the three non-attention sites share the same tests.

**Decision 8 taken (a) the same day (owner):** fix all four in this tier, tests first on the real card.

### 2026-10-07: decision 8 (a): no freed pointer kept after a failed scratch growth; one leak per backend

**Design.** New class `DeviceScratchSlot`: one grow-on-demand buffer, on the device or in pinned host memory.
`ensure(bytes)` returns the buffer when it is large enough; otherwise it empties the slot (pointer `null`, size 0),
frees the old buffer and allocates the new one, so an allocation that runs out of memory leaves nothing freed behind.
`free()` empties the slot before freeing. The scratch entries of `CudaGqaAttention` (query, output, both pointer
tables, lengths), `CudaRmsNorm` (three device and three pinned host buffers) and `DeviceActivationBatch` (input,
output) are built from slots; their fast path is the same single size comparison as before. `CudaMatVec.sgemv(float[]
A, ...)` and its ROCm twin `RocmMatVec.sgemv` (same pattern, found while fixing the CUDA one) now allocate their three
buffers inside the `try` whose `finally` frees them. Every other growth site in `node` already cleared its field
before freeing (`CudaMatVec`'s and `RocmMatVec`'s scratch, `Q4KDequantScratch`, `Q8WindowScratch`), checked by grep.

**Tests, written first** (`DeviceScratchSlotTest`, `@Tag("gpu")`, the real card; a request larger than the whole
device fails in the allocator, which is what a full card does at a smaller size):

| Test | Seen failing first? | Now |
|---|---|---|
| `aGrowthThatRunsOutLeavesTheSlotEmptyNotDangling`: grow, then fail; no pointer, no size; the next small request allocates afresh | yes, against a stub with today's free-then-allocate growth: the old size kept for a freed buffer | pass |
| `aPinnedHostGrowthThatRunsOutLeavesTheSlotEmpty` | yes, same | pass |
| `aRequestThatFitsKeepsTheBuffer` | no: both versions reuse a large enough buffer, so a regression test | pass |

Not tested by a dedicated case: the three call sites' conversion (held by their existing tests below, which exercise
every slot on every call) and the two `sgemv` fixes, whose failing case needs a host matrix about the size of the card;
the ROCm one is compiled only (NEEDS-AMD-HARDWARE).

**Regression runs** (jar `57cb08a0fa77e9a5`).

| Command | Result |
|---|---|
| `mvn test -pl node -Dgroups=gpu` | 343 run, 1 failure, 7 skipped: `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` (130.9 MB reported free with the card full, bound 64 MiB) |
| unit reactor, eleven modules | 2,139 run, 1 failure (the same test, node 959), 49 skipped |
| `PrefillReserveDeviceTest` re-run right after, pre-change tree (`3b27aa3`, scratch build) and this tree | 3 of 3 pass on both. It fills the card through the bindings with none of the changed classes in between and reads the driver's free figure; the known desktop-dependent check (third part's record), not loosened |
| `mvn verify -pl juno-master -Pgpu -Dit.model.path=<tinyllama>` (after `mvn install -DskipTests`) | 10 of 10 |
| `smoke-gpu-residency.sh --models tinyllama...,mistral...` (unmodified) | 0 failures: region active on every layer, greedy identical on against off, memory flat both modes, cluster pipeline and tensor answer with local mode's output, no node JVM left |

**Out-of-tier changes.** `CudaRmsNorm`, `DeviceActivationBatch`, `CudaMatVec.sgemv` and `RocmMatVec.sgemv` are outside
this tier's attention scope, changed by owner decision 8 (a) because they share the defect and the test. Not a
measurement boundary: the allocation pattern on every successful call is unchanged; only a failed allocation behaves
differently. No published baseline is invalidated.

**Next.** Scope item 8's close: the published `--device-spans` run with the prefill-window re-verification on the other
sweep models (executor), and the pinned A/B on the final region (owner, prepared then), then scope item 5.
