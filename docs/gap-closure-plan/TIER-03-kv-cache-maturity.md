# Tier 03: KV cache maturity

Status: not started
Gap analysis refs: §1.3

## Objective

Turn Juno's "paged" KV allocator into a real block-table-aware attention design (no mandatory
gather-to-contiguous-buffer step before every attention call), add cross-session prefix sharing,
add KV cache defragmentation, add at least one additional KV quantization tier (Q4_0 KV), carry the
quantized KV types to the GPU's copy of the cache, and add disk-backed session/prompt-cache persistence.

## Why this tier, why now

This follows Tiers 02 and 02B because the shape of KV storage (dense vs. paged, windowed vs. not,
shift-capable or not, the last two settled by Tier 02B) needs to be settled before investing in a block-table-aware attention kernel
that reads pages directly — building the fused kernel against a KV layout that's about to change
again would be wasted work. It precedes Tier 04 only because the *architecture* decision (does
attention read pages directly, or still gather) affects every later tier that touches KV; it has no
dependency on Tier 04 in either direction (item 4). *(Corrected 2026-10-11, plan review: this
paragraph said the Q4_0 KV tier "shares code paths with general quantization work and benefits from
Tier 04's expanded quant-format support landing afterward", which item 4 below and Tier 04's own
ordering note both withdraw.)*

## Scope

### In scope

0. **Real-model sliding-window validation before the block-table kernel is built on it** *(added
   2026-10-11, plan review)*. Tier 02B validated its windowed attention against synthetic metadata and
   against real models with the window injected (`SlidingWindowLiveTest`), and handed validation on a
   model *trained* with a binding window to [Tier 08](TIER-08-model-architecture-breadth.md), because the
   only such file on disk (`gemma-4-E4B`) has no handler. Item 1 extends Tier 02's tiled kernel, window
   parameter included, to read pages directly, so a windowing defect found in Tier 08 would land under two
   tiers' worth of work. No on-disk file on a supported handler declares a binding window: checked
   2026-10-11 with `./juno gguf-info`, Mistral 7B v0.1 (trained with a 4096-token window), Qwen2.5-3B and
   Qwen3-1.7B declare none, and Phi-3.5-mini's 262144 never binds. A Phi-3-mini-4k GGUF is the likely
   candidate: its training configuration uses a 2047-token window on the `phi3` handler Juno already runs.
   *Unverified:* whether a given GGUF of it carries `phi3.attention.sliding_window`; confirm with `./juno
   gguf-info` before relying on it. At the start of this tier, ask the owner for the file (README rule 3,
   [`INVENTORY.md`](INVENTORY.md)). If it declares the window, run `SlidingWindowLiveCheck` on it at a
   3,000-token prompt on CPU and GPU, both schedules, and add a teacher-forced NLL comparison over a
   3,000-token text with the window on against off: the windowed run must read **lower** NLL past token
   2047, which is the evidence the window is applied where the model expects it. If it does not declare
   the window, or the owner declines the download, record that here and Tier 08 keeps the obligation.

1. **Block-table-aware attention**: extend the Tier 02 tiled attention kernel to read
   `KvPageTable`'s pages directly, removing `PagedKvTensor.needsAttentionScratch()`'s unconditional
   `true` and the "gather tax" it causes. (An earlier draft offered "or the existing GQA kernel if
   Tier 02's tiling isn't ready" — under execution rule 1 that state cannot exist, since Tier 02's
   exit criteria require the tiled kernel before this tier starts. If Tier 02 ended partial-complete,
   that is an escalation, not a fallback path to take quietly.)
2. **Cross-session prefix sharing**: allow `PrefixCache`'s trie to be consulted and populated across
   sessions when the leading tokens genuinely match, with correct reference counting so that KV
   pages backing a shared prefix are not freed while any session still references them (this is the
   piece Tier 00's `generateBatch()` fix explicitly does *not* attempt — cross-session sharing needs
   proper page reference counting, which today's single-owner `KvBlockPool` doesn't have).
3. **KV cache defragmentation**: compact `KvBlockPool`'s free/live pages when fragmentation crosses
   a threshold, to reclaim memory without requiring a full session evict.
4. **Q4_0 KV quantization tier**, alongside the existing F16/Q8_0, for memory-constrained
   long-context serving. This is **standalone work with no dependency on Tier 04**, which runs
   later: Q4_0 weight decode already exists (gap analysis §1.1), and a KV codec is a different
   layout against different call sites than a weight dequant path. Nothing here waits on, or shares
   a design pass with, Tier 04 — the two tiers cannot share one under execution rule 1, and the
   earlier "check dependency order at implementation time" wording implied an option that does not
   exist.
5. **Disk-backed session persistence**: a session-save/session-restore mechanism (KV state to disk,
   reloadable in a later process), analogous in purpose (not implementation) to what was removed in
   the earlier three-tier-to-two-tier simplification (`CHANGELOG.md:1758`) — this time scoped
   narrowly (explicit save/restore commands, not always-on disk IO) so it doesn't reintroduce the
   complexity that motivated removing the old design.
6. **Host KV footprint: exact growth and real FP16 storage** *(added 2026-10-04 from Tier 01C step 6)*.
   `DenseKvTensor` doubles its capacity, and its `f16` element type stores 32-bit floats. A 2,048-token
   prompt plus one generated token doubles every layer's K and V to 4,096 positions. On Phi-3.5-mini (32
   KV heads) that is 3.2 GB of host KV where 1.6 GB holds the request, and 0.8 GB with real FP16. With
   about 2.2 GB of retained host weights, it exhausted the comparison harness's fixed 6 GiB heap at 2048
   tokens (Tier 01C record; the sweep used an explicit 8 GiB for that model). Grow toward the known
   request length (prompt plus `max_tokens`) instead of doubling past it, store `f16` as FP16 or rename
   the type for what it is, and measure Phi-3.5-mini's live heap at 2048 tokens before and after. Real
   FP16 storage changes KV rounding, so it carries a greedy-parity check.
   *2026-10-11 (owner decisions, Tier 02D decision 2 and its revision): Tier 02D found the prefill window's cost is the
   per-request allocation of the host KV tensors and the GC it triggers, not the growth, and removed it by reusing a
   finished request's dense tensors for the next one ([Tier 02D](TIER-02D-gpu-host-overhead.md) item 3). Growing
   toward the known request length, real FP16 storage and the live-heap measurement stay here, measured against
   the build Tier 02D leaves. Note for that measurement: an idle handler now keeps one request's dense host KV.*

7. **Quantized KV on the device** *(added 2026-10-11, plan review)*. `--cache-type-k`/`--cache-type-v`
   (`f16|q8_0`, and item 4's `q4_0`) choose the **host** KV storage only. On a GPU run every running
   request also keeps a device copy of its KV (`DeviceKvCache`), always FP16, and that copy is what bounds
   how many sessions run at once on the card: per 4,096-token session about 92 MB on TinyLlama, 150 MB on
   Qwen2.5-3B, 540 MB on Mistral 7B and 1.6 GB on Phi-3.5-mini (32 KV heads, no grouping), against the
   roughly 2.5 to 6 GB the 8 GB GTX 1080 has left after weights. Items 4 and 6 shrink the host copy and
   leave the device copy unchanged, so without this item the plan does not move GPU concurrency at all.
   Store the device copy in the type the flags select: Q8_0 blocks (34 bytes per 32 values, about 0.53x
   FP16) and Q4_0 blocks (18 bytes per 32, about 0.28x), written by the residency region's K/V write and
   read by the GPU attention kernels (decode width, Tier 02's tiled prefill kernel, and item 1's
   block-table kernel), dequantized in registers inside the kernel rather than to a scratch buffer. The
   upload stop rule and the forward-pass reserve (`DeviceScratchBudget`) count the device KV at its real
   size, and Tier 07 item 2's admission budget reads the same figure. A handler or backend whose attention
   kernel does not read the quantized layout keeps FP16 and says so at startup, never silently. ROCm has no
   GPU attention kernel until Tier 10, so it is N/A there.

### Out of scope

- Any change to the `static` vs `continuous` scheduling decision itself (Tier 07).
- Extending cross-session prefix sharing across cluster nodes (single-process only this tier;
  cluster-aware prefix sharing is a `continuous`+cluster question, deferred alongside Tier 07's
  cluster-fallback work if it's ever revisited).

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | block-table attention and defrag must work without a GPU; only the *kernel* reading pages directly is GPU-specific, the page-table/refcounting logic is shared |
| 2 | CUDA GPU inference | primary target for the block-table-aware kernel |
| 3 | ROCm GPU inference | FAIL-CLOSED/NEEDS-AMD-HARDWARE for the kernel path; page-table refcounting/defrag/disk persistence are backend-agnostic and must work on ROCm too; item 7 is N/A (no ROCm attention kernel until Tier 10), and a quantized `--cache-type` on ROCm keeps the host-side meaning only, stated at startup |
| 4 | Static schedule | cross-session sharing must respect `generateBatch()`'s existing (Tier-00-fixed) session gating — sharing is now allowed *safely*, not by removing the gate |
| 5 | Continuous schedule | `ContinuousBatchEngine`'s slot admission/retirement must correctly increment/decrement shared-page refcounts |
| 6 | Single-node local mode | primary dev/test surface, including session save/restore round-trip in a single process |
| 7 | Pipeline-parallel cluster | disk-persisted sessions must be resumable on any node holding that shard (or explicitly documented as node-pinned if not) |
| 8 | Tensor-parallel cluster | same |
| 9 | LoRA training | N/A — training doesn't use the inference KV cache |
| 10 | LoRA playback | confirm a LoRA-playback session's KV correctly invalidates cross-turn hits when the adapter changes the tokenized prefix (existing behavior, must not regress) |
| 11 | Vision | vision sessions' KV (image-token-spliced activations) must be excluded from cross-session prefix sharing unless proven safe — images are per-request; default to excluding vision sessions from sharing and document why |
| 12 | OpenAI REST surface | session save/restore needs an explicit API surface (new endpoint or `x_juno_*` field) |
| 13 | Native REST surface | same |
| 14 | CLI | session save/restore reachable from `./juno local`/`cluster` if there's a REPL-level use case; otherwise API-only, documented either way |

## Implementation steps

0. Run `scripts/performance-tests/check-plan-thresholds.sh`; then item 0 (ask for the windowed file,
   check its metadata, validate or record), before any kernel work.
1. Write correctness tests for the current gather-based attention as the oracle.
2. Add page refcounting to `KvBlockPool`/`KvPageTable`, unit-tested in isolation before touching
   the attention kernel.
3. Build the block-table-aware kernel read path; validate against the oracle.
4. Wire cross-session prefix sharing through `PrefixCache`, using the new refcounting so a page is
   only freed when its last referencing session evicts.
5. Add defragmentation as a background/on-demand compaction pass over `KvBlockPool`.
6. Add the Q4_0 KV codec, then item 7: the quantized device KV layout, the attention kernels' read
   path for it, and its accounting in `DeviceScratchBudget`.
7. Add disk-backed session save/restore, scoped to explicit save/restore calls only.
8. Run the full smoke matrix, with particular attention to the static-batch cross-session sharing
   case that Tier 00 deliberately did *not* fix by sharing (Tier 00 fixed it by gating; this tier
   is what makes safe sharing possible).

## Tests to write/upgrade before implementation

- **Plan check, first**: `scripts/performance-tests/check-plan-thresholds.sh` passes before any other
  test or code in this tier (README execution rule 7).
- **`KvBlockPoolTest`/`KvPageTableTest`**: refcounting correctness — page freed only when last
  reference drops; concurrent access from two sessions sharing a page.
- **New attention-kernel test**: block-table read path matches the gather-based oracle bit-for-bit
  (within tolerance) across single-page and multi-page sequences.
- **`PrefixCacheTest`**: cross-session hit/miss correctness, including the exact scenario Tier 00's
  `StaticBatchPrefixCacheSessionGatingTest` exercises — this tier's test should show that scenario
  now succeeds *by design* (safe sharing) rather than by avoidance (the Tier 00 gate).
- **New defragmentation test**: fragment a pool deliberately (alloc/free pattern), compact, confirm
  no data corruption for any still-live page.
- **New Q4_0 KV codec test**: round-trip encode/decode accuracy vs. F16.
- **Device KV parity** (item 7): GPU attention over a Q8_0 / Q4_0 device KV against the CPU attention
  over the same host codec, at decode and prefill width, within the bound the FP16 kernel is held to; and
  a `DeviceScratchBudget` test that the device KV is counted at its quantized size.
- **KV quality live test** (items 4 and 6): greedy agreement over 512 tokens and teacher-forced NLL over
  2,048 tokens, Q4_0 KV and real FP16 storage each against the current storage, on every sweep model.
- **New session persistence test**: save mid-generation, restart process, restore, continue
  generating, confirm output is consistent with an uninterrupted run.
- **`ModelLiveRunnerIT`**: add a cross-session-sharing check and a save/restore check.
- **New bash smoke script**: `scripts/performance-tests/smoke-kv-cache.sh` — drives two
  concurrent clients sharing a system prompt (expect the *shared*-prefill-skip this time, not just
  correctness), a defragmentation stress loop, and a save/kill-process/restore/continue sequence.
- **Standing throughput and allocation gate** (README, "Test infrastructure"): run against the pre-tier jar
  and score it before closing this tier.
- **Perf gate (required)**: block-table attention and defrag are hot-path changes —
  `compare-lora.sh` plus a dedicated "gather tax" re-measurement (reuse
  `scripts/performance-tests/gather-tax-microbench.sh` and the methodology in
  `docs/perf-compare/20260910T213121Z-gather-tax.md`/`20260910T214300Z-gather-tax.md`), plus
  `compare-llama-cpp.sh` for a llama.cpp-relative reading (per README's llama.cpp-relative gate).

  **Threshold.** The existing gate allows gather to be **<= 15%** of gather+attention at batch 8 /
  ctx 8192, and the last published run measured **4.61%**. Reading the block table directly removes
  the step, so its absence is a unit assertion, not a measurement: `PagedKvTensor.needsAttentionScratch()`
  returns `false` on every path, and the microbenchmark's `gather_ms` column reads zero by construction.
  *(2026-10-11, plan review: `gather_ms == 0` was the headline gate; it cannot fail once the code path
  is deleted, so the gate that measures something is the next one.)* This tier's gate is:
  - `paged_vs_dense%` at ctx 8192 / batch 8 is **<= 2%**, so removing the gather did not simply move
    its cost into the kernel's page-walk — the primary throughput gate;
  - Q4_0 KV quality (item 4): greedy decode with Q4_0 KV agrees with F16 KV on **>= 97%** of the first
    512 generated tokens at temperature 0 on every sweep model, and teacher-forced mean NLL over a fixed
    2,048-token text is **<= 1.02x** the F16-KV reading (the `ModelQualityNllLiveTest` pattern);
  - real FP16 dense KV storage (item 6): greedy agreement with today's FP32-backed `f16` storage on
    **>= 99%** of the first 512 tokens on every sweep model, mean NLL **<= 1.005x**, and Phi-3.5-mini's
    live heap at a 2048-token request **<= 0.60x** the pre-tier reading;
  - quantized device KV (item 7): device KV bytes per position **<= 0.55x** the FP16 copy at `q8_0` and
    **<= 0.30x** at `q4_0`; the number of 2,048-token Mistral 7B sessions whose device KV fits beside the
    weights **>= 1.8x** the FP16 figure at `q8_0`; GPU decode tg at a 2,048-token context with `q8_0`
    device KV **>= 0.95x** the FP16 copy (attention reads fewer bytes and pays an in-register dequant;
    same-hour A/B, pinned); greedy agreement with the FP16 device copy on **>= 99%** of 512 tokens at
    `q8_0` and **>= 97%** at `q4_0`, on every sweep model;
  - defragmentation's pause: a compaction pass over a fragmented pool adds **<= 5 ms** to any single
    decode step at ctx 8192, or it runs off the decode path entirely.

## Models needed

Existing models suffice for correctness and defrag testing. No new downloads required for this
tier specifically.

## Exit criteria

- [ ] Sliding-window attention validated on a model trained with a binding window (item 0), or the
      absence of such a file on a supported handler recorded here with the owner's answer, and Tier 08
      keeping the obligation.
- [ ] Block-table attention kernel in place; `needsAttentionScratch()` false on every path, asserted by
      unit test.
- [ ] Cross-session prefix sharing works safely (refcounted) for non-vision, non-LoRA-play sessions
      in both schedules.
- [ ] Defragmentation reclaims fragmented pages without corrupting live sessions.
- [ ] Q4_0 KV tier available alongside F16/Q8_0, meeting its quality threshold (greedy agreement >= 97%,
      NLL <= 1.02x F16 KV on every sweep model).
- [ ] Real FP16 dense KV storage meets its threshold (greedy agreement >= 99%, NLL <= 1.005x, Phi-3.5-mini
      live heap at 2048 tokens <= 0.60x the pre-tier reading).
- [ ] Quantized device KV (item 7): `--cache-type-k`/`-v` reach the GPU's KV copy on every handler with
      a GPU attention path, or the handler keeps FP16 and says so at startup; bytes per position, session
      capacity, decode tg and greedy-agreement thresholds met.
- [ ] Disk-backed session save/restore round-trips correctly across a process restart.
- [ ] Standing throughput and allocation gate met (README, "Test infrastructure"): CPU and GPU tg and pp
      >= 0.95x the pre-tier build, allocation per token <= 1.10x, in-span GC pause total <= 1.25x.
- [ ] Cross-surface checklist fully resolved, vision/LoRA-play exclusions explicit and tested.
- [ ] Perf gate published; `paged_vs_dense%` <= 2% at ctx 8192 / batch 8, defrag pause within budget.
- [ ] The session save/restore surface is in `api/src/main/resources/openapi.yaml` and
      `juno-api.yaml` alongside the code that serves it (README feature-complete rule).
- [ ] Docs updated, Juno-native language only.
- [ ] `CHANGELOG.md` entry added.
