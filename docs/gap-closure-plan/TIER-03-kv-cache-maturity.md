# Tier 03: KV cache maturity

Status: in progress (begun 2026-11-09 by owner decision while Tiers 02F and 02G stay open; README execution rule 1). Step 0
done 2026-11-09: on `Phi-3.1-mini-4k-instruct-Q4_K_M.gguf` (window 2047, every layer) the KV oracle passes at 3,000 tokens on
CPU and CUDA under both schedules, and teacher-forced NLL past position 2047 reads 2.634 with the window against 10.529 without
it (CPU; CUDA 2.634 against 10.524). Exit criterion 1 ticked (1 of 15). Step 1 done 2026-11-10 (tests only): the gather oracle,
kernel level (`PagedKvAttentionOracleTest`) and handler level on all 8 handler classes that read host KV
(`PagedKvHandlerOracleTest`), bit for bit, shown failing on deliberate gather defects. Owner decisions 2026-11-10: item 1's
block-table read is the CPU attention's, the GPU's per-request copy staying dense (paging it is a Tier 07 candidate); step 0's
Phi-3 question, option 2 (a load-time warning, done); Tier 02G's item 3, option (a), landing here before step 6. Step 2 done
2026-11-10: `KvBlockPool` counts holders per page, `KvPageTable.sharePrefix` shares pages with copy-on-write, unit-tested and
shown failing on deliberate defects; no runtime behaviour changed. Step 3 done 2026-11-10: the CPU attention reads host KV in
place on every call site (`KvRows`; paged KV page by page, `Q8_0` decoded per head slice, no gather, no per-token scratch), bit
for bit step 1's oracle; `needsAttentionScratch()` removed; unpinned A/B every CPU lane 0.972x to 1.074x, `paged_vs_dense%`
-6.35% at the gate cell. Exit criterion 2 ticked (2 of 15). Step 4 begun 2026-11-10: a session resumed from another session's
cached prompt over positions it never wrote (defect, both schedules), fixed by matching a prefix entry only for the key that
cached it. Owner decisions 2026-11-11: 1 (b), finished prompts kept under a budget; 2 (a), `static` copies the rows. Step 4 done
2026-11-11: `SharedPrefixPool` keeps finished requests' prompts as KV nothing writes (paged: pages held again, copy-on-write;
dense: copied), new requests start from them on all three admission paths, `--prefix-cache-tokens`; replies equal sharing off
on TinyLlama, CPU and CUDA, both schedules. Exit criterion 3 ticked (3 of 15). Next: step 5, defragmentation.
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
   gguf-info` before relying on it. *(2026-11-09, step 0: one does, `bartowski/Phi-3.1-mini-4k-instruct-GGUF`'s Q4_K_M, window
   2047, read from its header; three other conversions do not. See the execution record. `./juno gguf-info` confirms it once the
   file is on disk.)* *(2026-11-09, later: on disk, `gguf-info` confirms it, and the validation below passes on CPU and CUDA; see
   the execution record.)* *(2026-11-10, owner decision on step 0's question, option 2: a `phi3` file with a 4096-token context
   and no window key is named in a warning at load, and still gets no window; done, see the execution record.)* At the start of
   this tier, ask the owner for the file (README rule 3,
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
   *(2026-11-10, step 1: drift, raised with the owner. Tier 02's tiled kernel reads `DeviceKvCache`, a dense per-request FP16
   device copy, never `KvPageTable`'s pages, which are Java heap arrays; the gather tax is the host copy before CPU attention.)*
   *(Corrected 2026-11-10 by owner decision, option (d) of that entry's decision 1: the block-table read is the **CPU**
   attention's. Every CPU attention call site (single decode, decode batch, prefill window and its fallback, in every handler
   that reads host KV, the LoRA handlers included) reads K and V rows through `KvPageTable` in place, `F16` rows from the page
   bytes and `Q8_0` rows decoded inside the attention loop, so `needsAttentionScratch()` is `false` for both types and the
   prefill window's per-row gather goes with it. The oracle is step 1's (`PagedKvAttentionOracleTest`,
   `PagedKvHandlerOracleTest`). The GPU keeps its per-request dense device copy, which item 7 quantizes; paging it is out of
   scope here and a candidate in [Tier 07](TIER-07-continuous-batching.md).)*
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
   *(2026-11-10, owner decisions: item 1 no longer has a GPU kernel, so the kernels this item changes are the decode-width,
   batched decode and tiled prefill reads of `DeviceKvCache`, and the kernels that write it. They compute every device KV address
   in one place (request, row, then the element within the row), so that a paged device KV, if
   [Tier 07](TIER-07-continuous-batching.md) takes that candidate, changes that place and the tables the host passes rather than
   each kernel. Before this item, [Tier 02G](TIER-02G-cuda-on-windows.md)'s item 3, option (a), lands here under README rule 9:
   prefill attention and the tiled GEMM split into launches of bounded work, with `gpu.kernel-timing` extended to the longest
   window the device holds.)*

### Out of scope

- Any change to the `static` vs `continuous` scheduling decision itself (Tier 07).
- Extending cross-session prefix sharing across cluster nodes (single-process only this tier;
  cluster-aware prefix sharing is a `continuous`+cluster question, deferred alongside Tier 07's
  cluster-fallback work if it's ever revisited).
- Paging the GPU's KV copy *(added 2026-11-10, owner decision 1 (d))*. `DeviceKvCache` stays one dense copy per request,
  so sessions sharing a prefix through item 2 share host pages and skip that prefill, but each keeps its own device rows. A
  device page pool is a candidate in [Tier 07](TIER-07-continuous-batching.md) item 2, taken only if its measurements need it.

## Cross-surface compatibility checklist

| # | Surface | Notes |
|---|---|---|
| 1 | CPU inference | block-table attention and defrag must work without a GPU. *(2026-11-10, owner decision 1 (d): the block-table read is the CPU attention's, so this is item 1's primary surface; the page-table and reference-counting logic is shared by every backend.)* |
| 2 | CUDA GPU inference | *(Restated 2026-11-10, owner decision 1 (d); it read "primary target for the block-table-aware kernel".)* GPU attention keeps reading its per-request device copy; item 1 reaches CUDA runs on the layers and fallbacks that attend on the CPU (partial `--gpu-layers`, a failed device dispatch). Item 7 is CUDA's KV change. |
| 3 | ROCm GPU inference | page-table refcounting/defrag/disk persistence are backend-agnostic and must work on ROCm too; item 1 applies as on the CPU, since ROCm attends on the CPU path *(2026-11-10, owner decision 1 (d); this read "FAIL-CLOSED/NEEDS-AMD-HARDWARE for the kernel path")*; item 7 is N/A (no ROCm attention kernel until Tier 10), and a quantized `--cache-type` on ROCm keeps the host-side meaning only, stated at startup |
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
3. Build the block-table-aware kernel read path; validate against the oracle. *(2026-11-10, owner decision 1 (d): the CPU
   attention's read through the page table, on every call site; no GPU kernel changes in this step.)*
4. Wire cross-session prefix sharing through `PrefixCache`, using the new refcounting so a page is
   only freed when its last referencing session evicts.
5. Add defragmentation as a background/on-demand compaction pass over `KvBlockPool`.
6. Add the Q4_0 KV codec, then item 7: the quantized device KV layout, the attention kernels' read
   path for it, and its accounting in `DeviceScratchBudget`. *(2026-11-10, owner decisions: first, before the device layout
   changes, [Tier 02G](TIER-02G-cuda-on-windows.md)'s item 3, option (a), bounded launches, recorded under README rule 9.)*
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

## Kit probes

README execution rule 10.

- Keep passing: `schedule.tinyllama`, `gpu.schedule.tinyllama` (session reuse on paged KV),
  `mixed-prefill.*`, `gpu.mixed-prefill.tinyllama`, `long-prompt.*`, `gpu.long-prompt.*`, `context-policy.*`,
  `gpu.context-policy.*`, `consistency.tinyllama`, `gpu.consistency.tinyllama`, `cpu.unit-tests`,
  `gpu.unit-tests`.
- Add:
  - A change, item 2: the consistency probes assert "no prefix-cache hit for stateless traffic". With
    cross-session prefix sharing a shared prefix is meant to hit, so the check becomes: a shared-prefix hit is
    counted, and the reply equals the unshared reply.
  - Q4_0 KV (item 4) and the quantized device KV (item 7): mean NLL against F16 KV per sweep model, and batched
    equal to single on the device at each `--cache-type-k`/`--cache-type-v`.
  - Session save and restore (item 5): a restored session continues with the reply the unbroken session gives.
  - Defragmentation (item 3): replies unchanged under page churn, with the compaction counted.
  - The host KV footprint per session (item 6), as a reading.

## Models needed

Existing models suffice for correctness and defrag testing. Item 0 needs one file that is not on disk: a
GGUF of a model trained with a binding sliding window on a handler Juno runs, likely Phi-3-mini-4k
([`INVENTORY.md`](INVENTORY.md), gaps table). The owner is asked for it at the start of this tier.
*(Corrected 2026-11-09: this section said "no new downloads required for this tier specifically", written before item 0 was
added on 2026-10-11.)* *(2026-11-09, later: the owner downloaded `Phi-3.1-mini-4k-instruct-Q4_K_M.gguf`, SHA-256 checked; it is
in `models/`.)*

## Exit criteria

- [x] Sliding-window attention validated on a model trained with a binding window (item 0), or the
      absence of such a file on a supported handler recorded here with the owner's answer, and Tier 08
      keeping the obligation. *(2026-11-09: validated on Phi-3-mini-4k, window 2047, CPU and CUDA, both
      schedules; see the execution record. Tier 08 keeps Gemma's patterned window.)*
- [x] Block-table attention kernel in place; `needsAttentionScratch()` false on every path, asserted by
      unit test. *(2026-11-10, owner decision 1 (d): the CPU attention's read through the page table on every call site, equal
      to step 1's oracle; the GPU's device copy is out of scope.)* *(2026-11-10, later: met. Every CPU attention call site reads
      `SessionKvTensor.attentionRows`, the stored rows in place; no path needs scratch, so `needsAttentionScratch()` was removed
      with its last caller rather than left returning `false`. Asserted by `KvRowsTest`, `PagedKvAttentionOracleTest` (41),
      `PagedKvHandlerOracleTest` (18, all 8 handler classes) and `GqaWindowAttentionTest` (16), bit for bit; see step 3's
      record.)*
- [x] Cross-session prefix sharing works safely (refcounted) for non-vision, non-LoRA-play sessions
      in both schedules.
      *(2026-11-11, step 4: paged KV shares pages counted by `KvBlockPool`, copy-on-write; the dense `static` KV copies the
      rows instead, by owner decision 2 (a). `KvPrefixShareHandlerTest` bit for bit on every handler family, both schedules,
      both host KV types; `SharedPrefixPoolTest` on all three admission paths; on TinyLlama 9 of 9 replies equal sharing off,
      CPU and CUDA, both schedules ([`20261111T192150Z-shared-prefix`](../perf-compare/20261111T192150Z-shared-prefix/INDEX.md));
      vision and `--lora-play` refused and tested.)*
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
      *(2026-11-10: open; the owner's pinned run at tier close. Step 3's unpinned A/B met the CPU bars on every lane, evidence
      only.)*
- [ ] Kit probes (README execution rule 10): the **Add** list above is in the kit's catalog, each with its
      self-test and its `coverage.tsv` line, and the **Keep passing** list passes on this tier's build.
- [ ] Cross-surface checklist fully resolved, vision/LoRA-play exclusions explicit and tested.
- [ ] Perf gate published; `paged_vs_dense%` <= 2% at ctx 8192 / batch 8, defrag pause within budget.
      *(2026-11-10: the `paged_vs_dense%` half read -6.35%, unpinned,
      [`20261111T051528Z-paged-kv-in-place`](../perf-compare/20261111T051528Z-paged-kv-in-place/INDEX.md); open for the
      defrag pause, owed by the executor at step 5, and the reading is retaken on the tier's final build.)*
- [ ] The session save/restore surface is in `api/src/main/resources/openapi.yaml` and
      `juno-api.yaml` alongside the code that serves it (README feature-complete rule).
- [ ] Docs updated, Juno-native language only.
- [ ] `CHANGELOG.md` entry added.

## Execution record

### 2026-11-09: tier begun (owner decision); step 0, the windowed file located and asked for

**Owner decision 2026-11-09: (1)**, the choice put in [Tier 02F](TIER-02F-hardware-validation-kit.md)'s record of that date. This
tier begins while Tiers 02F and 02G stay open (README execution rule 1). [Tier 02G](TIER-02G-cuda-on-windows.md)'s item 3 choice
is due before step 3.

**Plan against code, verified at HEAD `f234e90c`.** `check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep or
build was running. The code this tier starts from is as the scope describes:
- `PagedKvTensor.needsAttentionScratch()` returns `true` unconditionally.
- `KvBlockPool` has `allocate()` and `free(int)` and no reference count.
- `KvElementType` is `F16` and `Q8_0`, and its Javadoc calls `F16` the CLI name of the float32 path.
- `gather-tax-microbench.sh`, `SlidingWindowLiveCheck`, `StaticBatchPrefixCacheSessionGatingTest`, `ModelQualityNllLiveTest`
  and `DeviceScratchBudget` exist.
- `DenseKvTensor`'s doubling and the FP16-only `DeviceKvCache`, read at `616f577f` (Tier 02F's first 2026-11-09 entry), are
  unchanged: `f234e90c` touched only the kit, `scripts/run.bat` and a launcher test.
- `SlidingWindow` reads `<arch>.attention.sliding_window` (absent or 0: no window) and `<arch>.attention.sliding_window_pattern`.
- `Phi3RopeConfig.from` accepts a file without `rope_factors_short`/`rope_factors_long` tensors (both stay null). Such a file's
  4096-token context holds item 0's 3,000-token prompt.

**Item 0: the windowed file.** Read from Hugging Face without downloading a model:
1. the API's repository search, then each candidate's file tree (size, LFS SHA-256) at a pinned commit;
2. the first 8 MiB of each candidate file, fetched by HTTP range request into a scratch folder;
3. those bytes parsed for their GGUF metadata keys.

| Repository (commit) | File | `phi3.attention.sliding_window` |
|---|---|---|
| `microsoft/Phi-3-mini-4k-instruct-gguf` (`a6411339`) | `Phi-3-mini-4k-instruct-q4.gguf` | absent |
| `bartowski/Phi-3-mini-4k-instruct-GGUF` (`e1447f6d`) | `Phi-3-mini-4k-instruct-Q4_K_M.gguf` | absent |
| `legraphista/Phi-3-mini-4k-instruct-update2024_07_03-IMat-GGUF` (`c2b1652e`) | `Phi-3-mini-4k-instruct.IQ4_NL.gguf` | absent |
| `bartowski/Phi-3.1-mini-4k-instruct-GGUF` (`66a614ac`) | `Phi-3.1-mini-4k-instruct-Q4_K_M.gguf` | **2047** |
| `lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF` (`7b8efa41`) | the same file (same SHA-256) | **2047** |

- **The file:** `Phi-3.1-mini-4k-instruct-Q4_K_M.gguf`, 2,393,232,096 bytes, SHA-256
  `d6d25bf078321bea4a079c727b273cb0b5a2e0b4cf3add0f7a2c8e43075c414f`.
  - Quantized from `microsoft/Phi-3-mini-4k-instruct`: its July 2024 update, which these two repositories call 3.1.
  - MIT licence; the repository is not gated.
- **Its header:** `general.architecture = phi3`; Q4_K_M (`file_type` 15); `context_length` 4096; 32 layers; 32 heads and 32 KV
  heads; `rope.freq_base` 10000; `tokenizer.ggml.pre = default`.
- **The key depends on the conversion, not the model version.** legraphista's conversion of the same July update lacks it.

**Asked of the owner** (README rule 3; this session does not write to `models/`). 91 GB is free on the disk that holds `models/`:

```
curl -L -C - -o models/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF/resolve/66a614ace4d069a12c2f6043f4ea92621c898d4a/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf
echo "d6d25bf078321bea4a079c727b273cb0b5a2e0b4cf3add0f7a2c8e43075c414f  models/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf" | sha256sum -c
```

**The rest of step 0, once the file is on disk:**
1. `./juno gguf-info` on it.
2. `SlidingWindowLiveCheck` at a 3,000-token prompt, on CPU and GPU, both schedules.
3. The teacher-forced NLL comparison over a 3,000-token text, window on against window off. The windowed run must read lower NLL
   past token 2047.

If the owner declines the download, this record says so and Tier 08 keeps the obligation.

**Tests.** No code changed, so none was run. `check-plan-thresholds.sh` passes after these edits.

**Out-of-tier changes:** none.

**Exit criteria:** 0 of 15.

**Owed to the owner:** the download above, or a decline.

**Next for the executor:** the rest of step 0 on the file. Then step 1, the gather-based attention oracle tests.

### 2026-11-09, later: step 0 done; the trained window validated on CPU and CUDA

**Plan against code.** As in the entry above. HEAD is now `e16748ad`, the owner's amend of `f234e90c`, which removes only the tester's
zip ([Tier 02F](TIER-02F-hardware-validation-kit.md)), so no main code changed between the two entries. `check-plan-thresholds.sh` passed at the
end of that entry; nothing in the tree changed before this work except the owner's download. It passes again after these edits. No gate,
sweep or build was running.

**The file.** The owner downloaded `Phi-3.1-mini-4k-instruct-Q4_K_M.gguf`; its SHA-256 matches the one recorded above. `./juno
gguf-info` on it reads:
- `general.architecture = phi3`, `file_type` 15;
- `phi3.attention.sliding_window = 2047`, with no pattern key, so every layer;
- `block_count` 32, 32 heads and 32 KV heads;
- `context_length` 4096, `rope.dimension_count` 96, `rope.freq_base` 10000, `rope.scaling.original_context_length` 4096;
- `tokenizer.ggml.add_bos_token = false`.

**The text.** No new text was needed. The frozen `model-quality-nll.txt` gives 3,066 ids under this file's tokenizer (`GgufTokenizer.encode`
on the raw text, as for the other id files), so its first 3,001 are checked in as `model-quality-nll.phi3-mini-4k.ids`. They match
`model-quality-nll.phi35.ids` id for id over all 2,049 of that file, which checks the tokenization.

**Tests added** (test scope only; no main code changed). All are opt-in with `-Djuno.test.trainedWindow=true` and a 32 GB heap
(`JAVA_TOOL_OPTIONS=-Xmx32g`), and skip without the file. Without the property all 5 report skipped.
- `TeacherForcedNll`: per-position teacher-forced NLL through `forwardVerify`, in windows of 128, as `ModelQualityNllLiveTest` scores.
- `SlidingWindowTrainedLiveTest` (CPU):
  - the window as `SlidingWindow.read` gives it: 2047 on all 32 layers, and 3,000 tokens fit inside the 4096 context;
  - `SlidingWindowLiveCheck` at width 2047 on a 3,000-token prompt, under `static` and under `continuous`;
  - NLL over 3,000 positions with the file's window against `SlidingWindow.NONE`. The two must agree over positions [0, 2047), and
    the windowed run must read lower over [2047, 3000).
- `SlidingWindowTrainedGpuTest` (CUDA, tag `gpu`):
  - the same oracle, with the prompt prefilled on the device and GPU attention asserted active;
  - the same NLL comparison. Each position is a decode-width forward, so every position past 2047 attends on the device under the
    window.

**Results.** Positions are 0-based; position `t` predicts token `t + 1` from a context of `t + 1` keys. The window first excludes a key
at `t = 2047`.

| Reading | CPU | CUDA |
|---|---|---|
| NLL [0, 2047), window / none | 1.31213 / 1.31213 (difference 0) | 1.31328 / 1.31328 (difference 0) |
| NLL [2047, 3000), window / none | **2.63426 / 10.52854**, ratio 0.25020 | **2.63350 / 10.52423**, ratio 0.25023 |
| Perplexity past 2047, window / none | 13.93 / 37,367 | 13.92 / 37,206 |
| NLL [2047, 2560), window / none | 2.56599 / 6.48681 | 2.56612 / 6.48992 |
| NLL [2560, 3000), window / none | 2.71386 / 15.24082 | 2.71205 / 15.22786 |
| Oracle `static`: windowed / global / window against none | 0.000000 / 42.9867 / 9.8494 | 0.000000 / 43.4288 / 9.6124 |
| Oracle `continuous` | 0.000000 / 42.9867 / 9.8494 | 0.000000 / 43.4288 / 9.5701 |
| Wall time | 11,064 s (NLL 2,444 s and 2,673 s) | 528 s (NLL 69 s and 68 s) |

- **The window is applied where the model expects it.** Without it, NLL past 2047 climbs to 15.2 over the last 440 positions. That is
  above the 10.38 of a uniform guess over the 32,064-token vocabulary: the model reads key distances it never saw in training. With the
  window, NLL past 2047 (2.57, then 2.71) stays near the 2.10 of positions 1,024 to 2,047.
- **Before 2047 the window reads every key,** and both backends give identical NLL with it on and off.
- **CUDA agrees with the CPU** to 0.03% past 2047 (2.63350 against 2.63426), with device attention under the window.
- **Oracle.** The rows a windowed layer must not read (positions 0 to 953) were overwritten with values around 50, and that left the
  next logits exactly unchanged, under both schedules on both backends. The same overwrite in a global layer moves them by 43.
- **One small difference, for the record.** On CUDA the window-against-none reading differs by 0.04 between the two schedules; on the CPU
  it does not. Both values are far above the 0.01 bound.

**Finding for the owner: conversions that omit the key.** Three of the five Phi-3-mini-4k conversions checked above omit
`phi3.attention.sliding_window`, including Microsoft's own GGUF. Juno applies no window to them, by design (the how-to: it does not
guess which window a file meant). On the July-update weights that is the no-window run above: NLL 10.5 past position 2047. Microsoft's
file is the April release; whether it degrades the same way is unmeasured and would need a download. Decision below.

**Docs.**
- `docs/howto.md`: a bullet in "Sliding-window attention". It records the validation, and says that a conversion without the key gets no
  window and how to check a file.
- `README.md`: one phrase.
- `CHANGELOG.md`: a Session 137 entry.
- [`INVENTORY.md`](INVENTORY.md): the file's row, and the gap marked closed.

**Out-of-tier changes (README rule 9):** a dated note in [Tier 08](TIER-08-model-architecture-breadth.md)'s handover paragraph. Tier 03
item 0 met the uniform-window half of the obligation; Gemma's patterned window stays there.

**Tests run.**
- `SlidingWindowTrainedLiveTest#declaresItsWindow`: pass.
- `SlidingWindowTrainedLiveTest`: 4 run, 4 pass, `mvn test -pl node`, 32 GB heap.
- `SlidingWindowTrainedGpuTest`: 3 run, 3 pass, GTX 1080, run alone.
- Both classes without the property: 5 skipped, build success.
- `check-plan-thresholds.sh`: ok.

**Exit criteria:** 1 of 15 (criterion 1, item 0).

**Decision for the owner.** What should a `phi3` file with a 4096-token context and no window key get?
1. Documentation only. The how-to bullet is already in.
2. **Recommended:** a one-line startup notice when such a file loads. It names the key and says contexts past 2047 tokens run without
   the window the model was trained with. This is loud, and guesses nothing. It is a few lines in the Phi-3 load path plus a unit test on
   synthetic metadata, as an item 0 follow-up in this tier. Phi-3.5-mini (context 131072) does not match it.
3. Infer a 2047 window for that shape. Not recommended: it contradicts the documented no-guess rule, and would window a file that truly
   has none.

**Owed to the owner:** the decision above. Nothing to run.

**Next for the executor:** step 1, the correctness tests for the current gather-based attention as the oracle, with option 2 first if the
owner chooses it. [Tier 02G](TIER-02G-cuda-on-windows.md)'s item 3 choice is still due before step 3.

### 2026-11-10: step 1, the gather oracle (tests only); item 1's device half raised with the owner

**Plan against code, verified at HEAD `9dda5f7d`** (the owner's commit of step 0, with `.gitattributes`). `check-plan-thresholds.sh`
passed first (26 tier files). No gate, sweep or build was running. `git status`: `report1.qdstrm` deleted and `.github/` untracked,
neither this session's. The owner's answer to step 0's decision (the Phi-3 startup notice) is not recorded anywhere yet; step 1 does
not depend on it.
- `PagedKvTensor.needsAttentionScratch()` returns `true`; `KvBlockPool` has `allocate` and `free` and no reference count;
  `KvElementType` is `F16` and `Q8_0`. Unchanged since the first 2026-11-09 entry.
- **Where the gather happens.** Every CPU attention call site reads host KV through `SessionKvTensor.viewForAttention(seqLen,
  scratch)`. On paged KV that is `KvPageTable.gather`: whole pages copied out of the page bytes for `F16`, one row decoded at a
  time for `Q8_0`. There are 18 K/V call-site pairs in 8 handler classes:
  - `LlamaTransformerHandler` (LLaMA, Mistral, Qwen2): single decode, decode batch, prefill window, and the prefill window's
    fallback when a device attention dispatch fails;
  - `Phi3TransformerHandler`, `Qwen3TransformerHandler`: single decode, decode batch, prefill window (`Qwen3MoeTransformerHandler`
    shares Qwen3's single-decode site and has a decode-batch site of its own);
  - `Phi2TransformerHandler`: the same three;
  - the LoRA handlers the loader returns whenever adapters are given, so `--lora-play` serves on them, on either schedule:
    `LoraTrainableHandler` (decode and prefill window), `Phi3LoraTrainableHandler` and `Qwen3LoraTrainableHandler` (decode).
- **The prefill window gathers once per row**: row `b` calls `viewForAttention(startPos + b + 1)`, so a `W`-row window over `S`
  stored positions copies about `W * S` rows per layer for each of K and V. The gather-tax microbenchmark measures decode-shaped
  calls only (one query per session). Noted for step 3's design; nothing here measures it.
- **GPU attention never reads the pages.** The decode-width kernel and Tier 02's tiled prefill kernel read `DeviceKvCache`, a
  per-request, per-layer, dense FP16 device copy that grows by doubling, written alongside the host KV. `KvPageTable`'s pages are
  Java heap `byte[]` arrays. On a GPU run the host gather happens only where attention falls back to the CPU.

**Drift: item 1 names a kernel that does not read the pages.** Item 1 says "extend the Tier 02 tiled attention kernel to read
`KvPageTable`'s pages directly, removing `PagedKvTensor.needsAttentionScratch()`'s unconditional `true` and the gather tax". The
gather tax is a host cost (the gap analysis §1.3 and the microbenchmark both measure the host copy before CPU attention), and the
kernel item 1 names reads a separate device copy that is not paged. Removing the gather means the CPU attention reads rows through
the page table. A block-table read on the device would need the device KV itself paged, which no item scopes. This changes step 3's
scope, so it is the owner's decision (below); step 1 needs a CPU gather oracle under either reading, and the device side already has
its oracle (`GqaAttentionKernelParityTest`, GPU attention against `GqaMath` over the same rows).

**Tests added** (test scope only; no main code changed). Both are regression tests over code that exists: they passed on their first
run, except for two defects in the tests themselves, found and fixed as described below.
- `PagedKvAttentionOracleTest` (33 cases, kernel level). Over the same rows, a `PagedKvTensor`'s gathered view equals a
  `DenseKvTensor`'s view, and every CPU attention form gives the same output bits over both: `GqaMath.attend` (decode),
  `attendPooledAsSerial` (decode batch) and `attendPooled` with the scalar and the vector head (prefill window).
  - Paged side laid out as a server lays it out: K and V pools shared with a neighbour session written in between, so a session's
    pages are not consecutive blocks; pages recycled from a freed session, so a page's slots past the written length hold large
    stale rows; page sizes 1, 3, 4 and 16, so the last page is often partly filled.
  - Both cache types; shapes 4q/4kv/8 (no grouping), 8q/2kv/16 and 16q/2kv/32; windows 0, 1, 5 (starts mid-page) and 64; every
    context length from 1 to 37 read as a prefix of the stored rows, as the prefill window reads them.
  - The same after a context shift with a rejected draft dropped (and generation continued past another page boundary), and with a
    first write at position 7 (zero rows before it on both sides).
  - One check that the inputs can see a wrong page: a page read twice in place of another changes the output, and so does a swap of
    two pages under a window, while a swap before the window does not.
- `PagedKvHandlerOracleTest` (18 cases, handler level). The continuous schedule's paged KV gives the static schedule's dense-KV logits
  bit for bit, at `f16` and `q8_0`, on 9 families: LLaMA (the test instance, and again with a 5-token window on every layer),
  Phi-3, Phi-2, Qwen3, Qwen3-MoE, and the three LoRA handlers (LLaMA, Phi-3, Qwen3, loaded with an adapter set as `--lora-play`
  loads them). Pages of 4 tokens. Each family runs ten steps: an 11-token prefill window (3 pages, the last partly filled); a 5-token
  one; an 11-token prompt prefilled in two chunks of 6 and 5, so the second chunk attends over pages the first wrote; two single
  decodes crossing the page boundary at 12; a decode batch of three requests at lengths 13, 5 and 11; and an 11-token prompt with
  the first prompt's last token and a different context. The CPU oracle reaches 17 of the 18 call-site pairs; the 18th, LLaMA's
  prefill fallback after a failed device dispatch, runs only on a GPU.
  - Before comparing, it checks that the model's logits depend on the context: the first and last prompts end on the same token
    at the same position, and their logits must differ by more than 1e-4. That check exists because of the second defect below.
- `RandomWeightGgufs`: synthetic LLaMA, Phi-3, Phi-2 and Qwen3-MoE GGUF files with random F32 weights (Gaussian, standard
  deviation 0.05; norm weights 1), in the tensor layouts the zero-weight builders use. Qwen3 uses the existing random builder in
  `Qwen3LoraTrainableHandlerTest`.

**Defects in the tests, found while writing them.**
1. The first version of the wrong-page check swapped the first two pages with no window and expected a different output; it read
   5.96e-8. Attention with no window is a sum over (key, value) pairs and does not see their order, so a swap is invisible there.
   The check now uses a page read twice (visible without a window) and keeps the swap under a window.
2. The first version of the handler oracle used the existing synthetic files for Phi-3, Phi-2, Qwen3 and Qwen3-MoE, whose weights
   are all zero, so their logits are the same whatever attention reads. A deliberate defect (below) passed on those four. They now
   load random-weight files, and the context check above fails any model whose logits ignore attention.

**Shown able to fail.** Two deliberate defects were put into `KvPageTable.gather` one at a time, each test class run against it, and
the line restored (`git diff -- kvcache/` empty after each):

| Defect | Kernel level (33) | Handler level |
|---|---|---|
| `F16` bulk copy reads the partly filled last page from page 0 | 11 fail: every `F16` case with pages of 3, 4 or 16 tokens, except the zero-row case at 16, whose 16 rows fill one page | 6 of 12 fail: every family at `f16` (run before the LoRA families were added; Phi-3, Phi-2, Qwen3 and Qwen3-MoE passed until they had random weights) |
| `Q8_0` row loop reads the partly filled last page from page 0 | 11 fail: the same cases at `q8_0` | 6 of 12 fail: every family at `q8_0` |
| Both at once | not run | 18 of 18 fail, the three LoRA families included |

**Tests run.**
- `PagedKvAttentionOracleTest`: 33 run, 33 pass. `PagedKvHandlerOracleTest`: 18 run, 18 pass (`mvn test -pl node -am -Dtest=...`).
- `mvn test -pl node -am` (every test of `api`, `registry`, `lora`, `kvcache`, `health` and `node`, the GPU groups on the GTX 1080;
  48 min, with the oracle classes as they stood before the LoRA families): `registry` 93, `lora` 116, `kvcache` 105 and `health`
  23 pass (`api` has none); `node` 1,363 run, **1 failure**, 0 errors, 55 skipped.
  - The failure: `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`, 92,995,584 bytes reported free with
    the device full against <= 67,108,864. Re-run alone three times: 91,488,256, 91,488,256 and 91,422,720. This is the device-wide
    reading [Tier 02G](TIER-02G-cuda-on-windows.md)'s 2026-11-07 entry records (80 to 87 MiB on that tree and on a clean build of
    `d8012b11` alike; the driver's holdback moves with the desktop's GPU state). This change adds test classes only and does not
    touch the class or anything it runs. The bound is not loosened.
- After the LoRA families were added, on the restored `KvPageTable`: `PagedKvAttentionOracleTest` 33 of 33,
  `PagedKvHandlerOracleTest` 18 of 18, `LlamaTransformerHandlerScheduleParityTest` 1 of 1, `LlamaTransformerHandlerSlidingWindowTest`
  7 of 7, `KvPageTableTest` 3 of 3, `PagedKvTensorTest` 3 of 3.
- `check-plan-thresholds.sh`: ok.

**Kit probes.** None added: step 1 adds no feature, flag or threshold. The oracle runs in `mvn test`, which the `cpu.unit-tests` probe
runs.

**Out-of-tier changes:** none.

**Exit criteria:** 1 of 15, unchanged. Step 1 is the oracle exit criterion 2 is scored against.

**Decisions for the owner.**
1. **Item 1: which attention reads the block table** (new, above).
   - (a) **Recommended:** the CPU attention reads K and V rows through the page table on all 18 call-site pairs (`F16` rows straight
     from the page bytes, `Q8_0` rows decoded in the attention loop), so `PagedKvTensor.needsAttentionScratch()` is `false` for both
     types, and the prefill window's per-row gather goes with it. The GPU keeps its per-request dense device copy, which item 7
     quantizes. The gate stays as written: `paged_vs_dense%` is a host measurement. Item 1's text is corrected to say so, and the
     device copy staying per request is stated under "Out of scope".
   - (b) The same, and the device KV paged as well: a device page pool with a block table that the decode-width and tiled prefill
     kernels read. That is a new device allocator, both attention kernels changed, and it meets item 7's quantized layout and Tier
     02G's item 3 (bounded attention launches) in the same kernels. It needs its own scope item, thresholds and kit probes.
   - (c) Keep item 1's wording. Not recommended: it reads as (b) but scores as (a).
2. **The Phi-3 startup notice** (step 0's decision, 2026-11-09): still open. Recommendation unchanged, option 2.
3. **[Tier 02G](TIER-02G-cuda-on-windows.md)'s item 3 choice** is still due before step 3.

**Owed to the owner:** the decisions above. Nothing to run.

**Next for the executor:** step 2, page reference counting in `KvBlockPool`/`KvPageTable`, unit-tested in isolation. It depends on none
of the decisions above. Step 3 waits on decisions 1 and 3.

### 2026-11-10, later: owner decisions; the Phi-3 warning; step 2, page reference counting

**Owner decisions 2026-11-10**, on the three put in the entry above:
1. **Item 1: option (d).** The block-table read is the CPU attention's, on every call site; the GPU keeps its per-request
   dense device copy. Paging that copy is a candidate in [Tier 07](TIER-07-continuous-batching.md) item 2, taken only if
   Tier 07's measurements need it, with item 7 computing every device KV address in one place so that change stays local.
   Recorded in item 1, item 7, "Out of scope", cross-surface rows 1 to 3, steps 3 and 6 and exit criterion 2 (each with a
   dated note), and out of tier (README rule 9) in README execution rule 1, [Tier 02H](TIER-02H-batched-gpu-decode.md)'s
   "Out of scope", [Tier 07](TIER-07-continuous-batching.md) item 2 and [Tier 10](TIER-10-gpu-backend-breadth-cpu-simd.md)'s
   "Why this tier", whose text assumed a device block table. Gate unchanged: `paged_vs_dense%` is a host reading.
2. **Step 0's question: option 2**, a warning at load for a `phi3` file with a 4096-token context and no window key. Done
   below.
3. **[Tier 02G](TIER-02G-cuda-on-windows.md)'s item 3: option (a)**, bounded launches on every OS. With decision 1, step 3
   changes no GPU kernel, so the work is due before step 6 (item 7), not step 3; it lands in this tier then, under README
   rule 9, and the Windows timing stays in Tier 02G (its 2026-11-10 entry). The owner's reply named (a) without a timing;
   this follows the recommendation given with it, landing here before step 6.

**Plan against code.** HEAD `9dda5f7d`, unchanged since the entry above. `check-plan-thresholds.sh` passed after the plan
edits, before any code (26 tier files). No gate, sweep or build was running; the runs below ran one at a time.

**The Phi-3 warning (decision 2), tests first.**
- `UndeclaredWindowNotice` (new, node): `forFile(GgufReader)` returns the warning for architecture `phi3`,
  `phi3.context_length` 4096 and no `phi3.attention.sliding_window` key, else `null`. A declared key at any width, 0
  included, gets none, as do Phi-3.5-mini and the 128k files (context 131072) and every other architecture.
- `SlidingWindow.forShard`, which every handler calls once at load (the LoRA handlers and pipeline shards included), logs it
  as a warning when the file declares no window. The file still gets no window.
- The text: "This phi3 file has a 4096-token context and declares no phi3.attention.sliding_window, so attention runs without
  a window. Phi-3-mini-4k was trained with a 2047-token window: on that model, output degrades past 2047 tokens of context. A
  conversion that declares the key applies it; ./juno gguf-info shows the key." It does not say this file was trained with the
  window: the April release's degradation is unmeasured (step 0's entry).
- `UndeclaredWindowNoticeTest` (4 cases, synthetic metadata): the shape gets the warning, one line naming the key, 2047 and
  `gguf-info`; a declared key, other contexts and other architectures get none; `forShard` logs it once at `WARNING` and still
  returns no window, and logs no warning for a declared window.
- Shown able to fail, each defect put in and taken out (`cmp` against a copy afterwards): the warning never produced (2 of 4
  fail), produced and not logged (1 of 4), the key check dropped (1 of 4).
- Docs: `docs/howto.md` ("Sliding-window attention"), `docs/agent-arch.txt` (`SlidingWindow`), `CHANGELOG.md`.

**Step 2: page reference counting, tests first.**
- `KvBlockPool`: a holder count per page replaces the live bit. `allocate` holds once; `retain` adds a holder; `free` drops
  one and returns the page to the free list with its last holder; `liveBlocks` and `allocatedBytes` count distinct pages. A
  page with more than one holder is read-only: `writeToken` and `copyToken` refuse to write into it (`IllegalStateException`).
  `exclusive(id)` returns the page itself to a sole holder, otherwise a byte-for-byte copy with the caller's hold on the
  shared page dropped, atomically under the pool's lock. `retain`, `refCount` and `exclusive` are package-private.
- `KvPageTable.sharePrefix(len)` (public): a new table over the same pool whose positions `[0, len)` are this table's, holding
  its first pages without copying. Every write (an append into a partly filled page, an overwrite below `seqLen`, a context
  shift's destination) goes through one private step that makes the page the table's own first, so a table writing into a
  shared page takes a copy (copy-on-write) and neither table sees the other's later rows. `truncate` and `release` drop holds.
  A table is used by one thread at a time; tables sharing pages may run on different threads.
- Nothing calls `sharePrefix` yet: step 4 wires it through `PrefixCache`. With every page held once, the pool and the table
  behave as before, so no runtime behaviour changed.
- Tests, written first and failing to compile against the old classes (`refCount`, `retain`, `exclusive`, `sharePrefix`
  missing):
  - `KvBlockPoolTest` (4 new cases, 7 runs): a retained page stays live and is not handed out again until its last holder
    frees it, then a further free or retain is refused; a shared page refuses writes and copies into it while reads still work,
    and `exclusive` gives a byte-for-byte copy and leaves the other holder's page untouched (both cache types); 8 holders
    calling `exclusive` at once each end with a page of their own, all with the same bytes; 8 threads retaining and freeing 16
    pages 300 times keep every count.
  - `KvPageTableTest` (6 new cases, 10 runs), every expected content a table written with the same rows and no sharing,
    compared bit for bit, both cache types: sharing a prefix (whole, and ending mid-page) copies no page; after sharing, each
    table's appends are its own, whichever side writes first into the partly shared page, with the page count checked; a page
    returns to the pool only when the last table holding it lets go (release and truncate, across a chain of two shares); a
    context shift and an in-place overwrite on a sharer leave the source's rows intact; an empty share holds no page and a
    share past the written rows is refused; two sessions sharing a prefix whose last page is partly filled append 200 rows each
    on two threads, checking their contents every 10 rows, while a third thread reads the source 2,000 times, and every page
    returns to the pool at the end.
- Shown able to fail, each defect put in and taken out (`cmp` against a copy afterwards):

  | Defect | `KvBlockPoolTest` (10) | `KvPageTableTest` (13) |
  |---|---|---|
  | `free` drops every holder at once | 2 fail | 5 fail |
  | no copy-on-write: the table writes into the shared page, and the pool's refusal removed | 2 fail | 5 fail |
  | `exclusive` hands out a copy without the page's bytes | 3 fail | 5 fail |

- Docs: `docs/agent-arch.txt` (the `KvBlockPool` / `KvPageTable` paragraph), `CHANGELOG.md`. Not user-facing yet, so not in
  `docs/howto.md`.

**Row-write cost** (the one hot-path change: a write into an existing page asks the pool whether the page is shared, one
uncontended lock). `compare-lora.sh` runs the default `static` schedule, whose dense KV never reaches `KvBlockPool`, so it
cannot see this change and was not run; the tier's perf gate stands at its close. Measured directly instead: HEAD's `kvcache`
sources (`git show`) against this tree's, each compiled on its own, appending 2,048 rows into 32 layer tables, 15 warm-up and
25 timed trials, median per row, three rounds alternating the two builds (unpinned, this host, nothing else running):

| Shape | HEAD, ns per row (3 rounds) | This tree | Medians |
|---|---|---|---|
| `f16`, kvDim 1024, page 16 | 475.5 / 469.0 / 478.9 | 482.2 / 480.7 / 477.9 | +1.1% |
| `q8_0`, kvDim 1024, page 16 | 3,767.6 / 3,792.2 / 3,921.5 | 3,978.7 / 3,957.3 / 3,942.5 | +4.4% (HEAD's own spread 4.1%) |
| `f16`, kvDim 256, page 4 | 224.6 / 228.6 / 233.8 | 241.7 / 236.1 / 242.3 | +5.7% |

That is 5 to 13 ns per `f16` row, and at most about 0.2 µs per `q8_0` row, inside HEAD's own spread. A decode token writes 2
rows per layer, 64 on a 32-layer model: under 15 µs per token in the worst reading, against tens of milliseconds per CPU
decode token. Not optimized further (a lock-free path for pages never shared is possible, and not worth its complexity at
this size).

**Tests run.**
- Focused, `mvn test -pl node -am -Dtest=...`: `UndeclaredWindowNoticeTest` 4, `SlidingWindowTest` 9,
  `PagedKvAttentionOracleTest` 33, `PagedKvHandlerOracleTest` 18, `LlamaTransformerHandlerScheduleParityTest` 1,
  `LlamaTransformerHandlerSlidingWindowTest` 7, `KvBlockPoolTest` 10, `KvPageTableTest` 13, `PagedKvTensorTest` 3,
  `PagedKvArenaManagerTest` 2, `SessionKvReuseTest` 5, `KvContextShiftTest` 17: 122 run, 122 pass. Step 1's oracle passes on
  the reference-counted pool.
- The unit suite (`mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player`,
  GPU groups on the GTX 1080, 50 min): `registry` 93, `lora` 116, `kvcache` 120 and `health` 23 pass; `node` 1,373 run,
  **1 failure**, 55 skipped. The failure is `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`
  again: 71,892,992 bytes free with the device full against <= 67,108,864, the device-wide driver holdback recorded in the
  entry above and in Tier 02G's 2026-11-07 entry. The test uses `CudaMatVec` and the device allocator and nothing this change
  touches. Bound not loosened.
- The modules after `node`, which that failure stopped, resumed with the tree in the reactor and `node`'s suite filtered out
  (`mvn test -pl tokenizer,sampler,coordinator,vision,metrics,juno-player -am -Dsurefire.failIfNoSpecifiedTests=false
  -Dtest='**/tokenizer/**,**/sampler/**,**/coordinator/**,**/vision/**,**/metrics/**,**/player/**'`, 19 min): `tokenizer`
  109 (2 skipped), `sampler` 90, `coordinator` 402 (1 skipped), `vision` 97, `metrics` 86, `juno-player` 153 (2 skipped),
  0 failures. `coordinator` runs the continuous engine on the paged KV.
- `check-plan-thresholds.sh`: ok.

**Kit probes.** None added. The warning is a log line; step 2 adds no feature, flag or threshold that a run can see. Their
tests run in `cpu.unit-tests`. `coverage.tsv` needs no line: neither is a probe, script, architecture or cross-surface row.

**Out-of-tier changes (README rule 9):** the plan notes listed under decision 1 and 3 above; no code outside this tier's scope.

**Exit criteria:** 1 of 15, unchanged. Step 2 is the reference counting criterion 3 (cross-session sharing) is built on.

**Owed to the owner:** nothing to run, nothing to decide.

**Next for the executor:** step 3, the CPU attention reading the pages in place on every call site, held to step 1's oracle,
with `needsAttentionScratch()` `false` for both types.

### 2026-11-10, latest: step 3, CPU attention reads host KV in place

**Plan against code, verified at HEAD `74f75d78`.** `check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep or
build was running (a waiter loop left by an earlier session polls for an `mvn` run that no longer exists; it only sleeps). Since the
entry above, HEAD moved by the owner's commit of it (`737ce2cb`) and by `74f75d78`, which touched only the kit, `ClusterHarness`
and `ModelLiveChecks` and is recorded in [Tier 02F](TIER-02F-hardware-validation-kit.md)'s record as that tier's out-of-tier
change; neither touches this tier's code. `git status`: `report1.qdstrm` deleted and `.github/` untracked, neither this session's.
During this session another session added a record to Tier 02F and a dated line to Tier 02H (uncommitted); not this session's.
- Every CPU attention call site still read host KV through `SessionKvTensor.viewForAttention(seqLen, scratch)`: 18 K/V pairs in 8
  handler classes, as step 1's entry lists. Paged: `KvPageTable.gather`. Dense `F16`: the backing array. Dense `Q8_0`: a decode
  of every row into a scratch buffer, per layer per call.
- All of them reach the math through `GqaMath`: `attend` (single decode; Qwen3's `gqa`), `attendPooledAsSerial` (decode batch)
  and `attendPooled` (prefill window, Phi-2, the LoRA handlers), the last through `GqaVectorHead` when the Vector API module is
  present. LoRA training's teacher-forced forward calls the same forms over its own ephemeral `float[]` caches.
- The single decode allocated two scratch arrays of `(pos + 1) * kvDim` floats per token whenever the layout was paged or
  quantized (`SessionKvLayout.needsAttentionScratch()`), in every handler; the batch workspaces held two more.
- `GatherTaxMicrobench` (kvcache) times its own copy of the serial attention, dense against gather-then-attend.

**Design (owner decision 1 (d): the CPU attention only; the GPU's device copy unchanged).**
- `KvRows` (new, kvcache): one layer's K or V rows of one request as attention reads them, the stored rows themselves. Rows sit
  in pages of `rowsPerPage()` rows. A paged tensor's pages are its block table's pages in the pool, in table order; a dense
  tensor is one page (`ONE_PAGE`). An `F16` page is a `float[]` of `kvDim` floats per row; a `Q8_0` page is a `byte[]` of
  packed rows, decoded by the reader one head's slice at a time.
- `SessionKvTensor.attentionRows(seqLen)` replaces `viewForAttention(seqLen, scratch)`, and `needsAttentionScratch()` is gone
  from `SessionKvTensor` and `SessionKvLayout`: no path needs scratch, so the method was removed rather than left returning
  `false`. The tensor keeps its view current as its storage changes (a page added or dropped, a shared page copied before a
  write, a dense array grown), so the handlers take a view per layer per call and nothing is copied.
- `KvBlockPool` stores `F16` pages as `float[]` (they were little-endian bytes behind a `FloatBuffer`), so attention reads them
  directly; `writeToken`, `readToken`, `copyToken` and the copy-on-write copy work on the typed page. `Q8_0` pages stay `byte[]`.
- `Q8_0KvCodec.decodeRange(src, rowOff, from, n, dst, dstOff)` (new): elements `[from, from + n)` of one stored row, bit for
  bit what a whole-row `decode` gives them; a head's slice can start and end anywhere in a 32-element block.
- `GqaMath` and `GqaVectorHead` take `KvRows`. A page's run of rows goes through the loop the whole cache went through when it
  was one array, so the dot products and the key-order sums, and the output bits, do not depend on the page size. The `float[]`
  forms (LoRA training's caches, the GPU kernels' oracle) read their arrays as one page.
- `GqaVectorHead` was split into `scoreKeys`, `addValues` and `scoreRun` and rewritten so that no loop-carried `FloatVector`
  accumulator is passed to a call C2 might not inline (the boxing hazard recorded in Session 129):
  `CpuKernelVectorEscapeTest.attentionHeadCallsNothingOnALoopCarriedVector` (new) reads the class's bytecode and flags 31 such
  calls in HEAD's version and none now.
- Every handler (Llama, Phi-2, Phi-3, Qwen3, Qwen3-MoE, and the three LoRA trainable handlers): the per-token scratch arrays and
  the batch workspaces' scratch are gone, and each K/V pair is read through `attentionRows`. The unused `gqaInto` (Llama, Phi-3,
  Qwen3) and the `BatchWorkspace.scores` array (Llama, Phi-3, Qwen3) went with them.
- `GatherTaxMicrobench` now times attention over dense rows against attention reading the paged rows in place
  (`gather_ms` is zero by construction) and gates `paged_vs_dense%` <= 2% at ctx 8192 / batch 8 / page 16, this tier's perf
  gate (exit code 2 over it); `scripts/performance-tests/gather-tax-microbench.sh`'s header says so.

**Run 1 found a regression, fixed by `GqaWindowAttention`.** The first candidate (`8f58887fd342cbe7`) held every lane but one:
`Q8_0` KV prefill read 0.927x (static) and 0.905x (continuous) the baseline, under the standing gate's 0.95x (run 1's table
below). JFR's top methods on the static `Q8_0` lane: `Q8_0KvCodec.decodeRange` 377 samples and `GqaMath.addRun` 356, against
the baseline's whole-cache `Q8_0KvCodec.decode` 110 and `GqaVectorHead.attend` 63. The prefill window ran `attendPooled` once
per query row, with query heads as the pool's units, and each head decoded its own KV head's slice of every key row: a `Q8_0`
slice was decoded `gqaRatio` times per row (8 on TinyLlama), and the value sum fell back to the scalar `addRun` because the
vector head reads `F16` pages only. A standalone measurement of the decode itself (`decbench`, scratch only): C2 does not
vectorize the scalar byte-to-float loop on this AVX1 host, and a Vector API form, bit for bit equal, was only about 25% faster,
so the fix is fewer decodes, not a faster one.
- `GqaWindowAttention` (new, node): a prefill window's attention in one pool entry, with (row, KV head) pairs as the units. A
  unit runs the `gqaRatio` query heads that share one KV head, and over `Q8_0` rows decodes each key slice and each value slice
  once for all of them. Each head then scores and sums exactly as its own `attendPooled` unit would: the decoded values are the
  same, the dot product is the same per-head form (vector or serial), and each output element sums its products in key order
  from zero; float multiply and add commute, and an accumulator starting at `+0` never holds `-0`. Over `F16` rows a unit runs
  its heads one after another through the per-head forms. Scores live in a per-thread array per head, grown and reused.
- The prefill-window sites (Llama 2, Phi-3, Qwen3 and Phi-2 1 each) call it instead of their per-row loop. Phi-2's single decode
  keeps its own `gqa`. The decode batch (`attendPooledAsSerial`) and the single decode (`attend`) are unchanged: they decode per
  head, and run 1 read their `Q8_0` decode at 0.979x, inside the gate.

**Tests, written first** (they failed to compile against the old classes: `KvRows`, `attentionRows`, `decodeRange` missing).
- `KvRowsTest` (new, 8 cases, 14 runs, both cache types): a paged view reads every written row as `readToken` does, across page
  sizes; a dense view is one page and follows the backing array when it grows; a row written after the view was taken is read
  through it; tables sharing a prefix read the same page until one writes into it, then each its own; after truncate, a context
  shift and release the view reads what the tensor holds; a reused dense tensor reads zero past what the new request wrote; the
  view's length is checked; `KvRows.of` is a one-page view of its array.
- `Q8_0KvCodecTest.decodeRange` (new): every slice of a stored row (widths 1 to 256, every start and length, the row inside a page) equals `decode`,
  bit for bit, and nothing outside the slice is written.
- `PagedKvAttentionOracleTest` (step 1's kernel oracle, 33 runs to 41): every attention form now reads the pages in place and is
  compared, bit for bit, with the same form over the rows gathered into one array; every (K type, V type) pair and page size.
- `PagedKvHandlerOracleTest` (step 1's handler oracle, 18): unchanged, passes on the new read path. It compares paged against
  dense, both now read in place, so it no longer checks the `Q8_0` decode against an independent path; the kernel oracle and
  `Q8_0KvCodecTest` do.
- `GqaWindowAttentionTest` (new, 16 runs): a window in one call against `attendPooled` per row over the gathered rows, bit for
  bit, paged and dense, every (K type, V type) pair and page size, 4 shapes (GQA ratios 1, 4 and 8), windows 0, 1, 5 and 64,
  windows starting at 0, 1, 16 and 30, both head forms; the output is prefilled with NaN so an unwritten head fails.
- `CpuKernelVectorEscapeTest.attentionHeadCallsNothingOnALoopCarriedVector` (new), above.
- Adjusted to the new interface, assertions unchanged in meaning: `DenseKvTensorTest`, `PagedKvTensorTest`,
  `SessionKvLayoutTest`, `SessionKvReuseTest`, `GatherTaxMicrobenchTest` (which now also checks that the in-place read gives the
  dense rows' output bits).

**Shown able to fail**, each defect put in and taken out (`cmp` against a copy afterwards). Kernel oracle 41 runs, handler
oracle 18, `KvRowsTest` 14, `Q8_0KvCodecTest` 6, `GqaWindowAttentionTest` 16:

| Defect | Fails |
|---|---|
| serial `F16` keys always read page 0 | kernel 15, handler 6 |
| vector-head `F16` keys always read page 0 | kernel 15, handler 7 |
| vector-head values read the previous page in one register block | kernel 8, handler 5 |
| `Q8_0` values decode head 0's slice for every head | kernel 16, handler 0 (paged and dense share the defect) |
| `Q8_0` keys ignore the slot within a page | kernel 16, handler 1 |
| the copy-on-write copy not given to the view | `KvRowsTest` 2 (the reactor stopped before node) |
| `decodeRange` takes the first block's scale for every block | `Q8_0KvCodecTest` 1, `KvRowsTest` 5 |
| window: every head of a unit scored with its first head's query (`Q8_0` keys) | window 8 of 16 |
| window: decoded values not zeroed before the sums (`Q8_0` values) | window 8 of 16 |

**Tests run.**
- Focused, `mvn test -pl node -am -Dtest=GqaWindowAttentionTest,PagedKvAttentionOracleTest,PagedKvHandlerOracleTest,`
  `CpuKernelVectorEscapeTest,KvRowsTest,Q8_0KvCodecTest`: 86 run, 86 pass.
- The unit suite on the final tree (`mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,`
  `metrics,juno-player -am -Dmaven.test.failure.ignore=true`, GPU groups on the GTX 1080, 65 min): `registry` 93, `lora` 116,
  `kvcache` 136, `health` 23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator` 401 (1 skipped) and 1, `vision` 97,
  `metrics` 86 and `juno-player` 158 (2 skipped) pass; `node` 1,398 run, 55 skipped, **2 failures**, both device-wide free-memory
  checks on the GPU that nothing this step changes reaches:
  - `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`, the driver holdback recorded in the entries
    above and in Tier 02G's 2026-11-07 entry.
  - `ResidentQkvPathTest.noDeviceMemoryRetained`: 4,587,520 bytes moved across 300 cycles against <= 4,194,304. The class alone
    straight after (`mvn test -pl node -Dtest=ResidentQkvPathTest`): 19 of 19 pass. Tier 02E recorded the same class's
    device-wide equality failing once this way. Bound not loosened.
- An accidental `mvn -pl kvcache install` early in this session put this tree's `kvcache` jar into `~/.m2`; every later run
  built `kvcache` in the reactor (`-am`), so no run read it stale. The next `install` from a clean tree replaces it.

**Measurements (unpinned, this host, one at a time, nothing else measuring;
[`docs/perf-compare/20261111T051528Z-paged-kv-in-place/`](../perf-compare/20261111T051528Z-paged-kv-in-place/INDEX.md)).** Baseline: HEAD `74f75d78` built from `git archive` in
scratch, shaded jar `96ee5784b914097c`. Candidate: this tree, `9af57a8054650216` (run 1: `8f58887fd342cbe7`). CPU lanes through
the candidate tree's compare script with `--juno-jar` per side, `--n-prompt 512` (Phi-3.5-mini 128), 2 warm-ups, A B A B A B,
medians; `compare-lora.sh` and `compare-vision.sh` from each tree with `--skip-build`; the smokes against the baseline jar.

Run 2, the final tree, medians of three (run 1 in parentheses where it differs in kind):

| Lane | pp B/A | tg B/A | Allocation per token B/A | GC pauses A / B | Greedy replies |
|---|---:|---:|---:|---|---|
| TinyLlama, `static`, F16 KV | 1.072 (0.994) | 1.018 | 0.966 | 0 / 0 ms | equal, 6 of 6 |
| TinyLlama, `continuous`, F16 KV | 1.045 (1.039) | 0.982 | 0.760 | 0 / 0 ms | equal, 6 of 6 |
| TinyLlama, `static`, `Q8_0` KV | 0.984 (**0.927**) | 0.975 | 0.748 | 0 / 0 ms | equal, 6 of 6 |
| TinyLlama, `continuous`, `Q8_0` KV | 1.074 (**0.905**) | 0.972 | 0.755 | 0 / 0 ms | equal, 6 of 6 |
| Phi-3.5-mini (128), `static`, F16 KV | 1.021 (1.001) | 0.994 | 0.970 | 0 / 0 ms | equal, 6 of 6 |

- Every CPU lane meets the standing gate's bars, unpinned: pp and tg >= 0.95x, allocation per token <= 1.10x, no GC pause on
  either side. Allocation falls about a quarter on the paged and `Q8_0` lanes: the per-token scratch copies are gone.
- LoRA (bars: train <= 1.25x, playback >= 0.80x): GPU train 1.000x, playback 1.007x; CPU train 1.000x, playback 1.009x; recall 3 of 3 on
  each side. Vision (GPU; latency <= 1.25x, decode >= 0.80x): 0.986x and 1.004x, replies equal 6 of 6. Run 1 met these too.
- Smokes against the baseline jar: long-prompt prefill, 24 checks pass, greedy text identical at 128, 512 and 2024 tokens under
  both schedules; consistency, 54 checks pass. Run 1's single-sample `static` TTFT at 2024 tokens read 83.7 s against 78.3 s;
  run 2 reads 69.6 s against 70.0 s, so that was noise.

**Perf gate (`paged_vs_dense%` <= 2% at ctx 8192 / batch 8): met, unpinned.** `gather-tax-microbench.sh` on the candidate:
-6.35% at the gate cell (page 16), 0 ms gather everywhere. The microbench times `kvcache` code, unchanged since it ran. Not every
cell is under 2%: at ctx 32768 every page size reads +2.5% to +7.8%, and at ctx 2048 page 16 +4.0% to +10.5%. At 8192, -3.4% to
-10.4%. The baseline's microbench measured the old path: the gather was 4.75% of gather plus attention at the gate cell, and gather-then-attend read +6.00% against dense, which this gate would fail. At 32768 tokens and page 16 the in-place read costs more than the gather did (+7.6% against +4.9%); recorded, not addressed, outside the gate's cell. The defrag-pause half of
that criterion waits for step 5.

**Docs.** `docs/howto.md` (`--cache-type-k`, `--kv-page-size`: attention reads the stored rows in place), `docs/agent-arch.txt`
(`KvRows`, the KV paragraph, `GqaMath`, `GqaWindowAttention`), `docs/performance.md` (block KV), `CHANGELOG.md`,
`docs/perf-compare/README.md`, `scripts/performance-tests/gather-tax-microbench.sh`'s header.

**Kit probes.** None added. The step adds no flag, threshold or surface; its tests run in `cpu.unit-tests`, and the A/B and smokes
above are the kit's lanes run here. `coverage.tsv` needs no line.

**Out-of-tier changes (README rule 9):** none.

**Exit criteria:** 2 of 15. Criterion 2 ticked (`needsAttentionScratch()` removed with its last caller; the oracles and
`KvRowsTest` assert the in-place read on every path). The perf gate criterion's `paged_vs_dense%` half is met unpinned; it stays
open for the defrag pause (step 5). The standing gate stays open for the owner's pinned run at tier close; run 2 is evidence,
not that run.

**Owed to the owner:** nothing to run, nothing to decide for this step.

**Next for the executor:** step 4, cross-session prefix sharing through `PrefixCache` on `KvPageTable.sharePrefix`.

### 2026-11-10, latest: step 4 begun; a session resumed from another session's prompt (defect, fixed); sharing raised with the owner

**Plan against code.** Step 3 above left nothing owed, so step 4 was taken up in the same session. Reading `PrefixCache` and its two
callers, the plan's premise that prefix reuse is session-scoped (cross-surface row 4, `StaticBatchPrefixCacheSessionGatingTest`'s
documented contract) did not hold in the code:
- `GenerationLoop.generate` and `ContinuousBatchEngine` looked up `findLongestPrefix(promptIds)`, which returns the longest prefix
  **any** key cached, and skipped prefill to its length without checking whose it was. A session whose prompt starts with
  another session's cached prompt, such as two users opening with the same message, skipped prefill over positions it never
  wrote. Under its own key those positions are empty: a reused dense tensor reads zero there (`static`), and a paged tensor has
  fewer rows than the read (`continuous`), so the reply is wrong or the request fails. The existing gating tests cover stateless
  traffic and one session's own turns, not two sessions.
- The trie held one key per node, so a second session caching the same prompt overwrote the first's entry.
- The lookup reported the trie walk's depth, not the depth of the node holding the key, so a match could run past the cached
  prefix along another entry's branch.
- A session that branched its conversation (an edited or regenerated turn) kept its earlier entry, whose positions past the
  branch the later prefill had overwritten; returning to the earlier branch resumed over them.

**The fix (tests first; no runtime behaviour for a single session changes).**
- `PrefixCache`: a node holds a set of owner keys. `findLongestPrefix(tokens, ownerKey)` (new) returns the longest prefix
  `ownerKey` cached, at the depth of the node that holds it; a `null` key matches nothing. A key holds one entry, its latest
  prompt: `cachePrefix` replaces it, and `invalidate` walks only that entry's path (it walked the whole trie). The lookup
  across keys stays, documented as such, for step 4's sharing.
- `KVCacheManager.findLongestPrefix(tokens, kvKey)` (new); `GenerationLoop.generate` and `ContinuousBatchEngine` look up with
  the request's own KV key. `generateBatch` still never consults the trie.
- Tests, written first:
  - `CrossSessionPrefixHitTest` (new, coordinator, 6 cases; `KvTrackingPipeline` flags a forward over positions not written
    under its key): on the single-request path and the continuous engine, a second session with the same opening prompt, and
    one whose prompt extends the first's, is prefilled in full with the unshared reply; two sessions that cached the same
    prompt each skip their own prefix on their next turn, and the second keeps it after the first's session ends; a session
    that branches back does not resume over positions its branch overwrote. The five written first failed against HEAD's code; the sixth (branch back) was added with the replacement and fails without it (below).
  - `PrefixCacheOwnerTest` (new, kvcache, 5 cases): owner-scoped matching, the same tokens under two keys, invalidation keeping
    another key's shorter prefix, replacement on re-caching, and the counters.
- Shown able to fail, each defect put in and taken out (`cmp` afterwards):

  | Defect | Fails |
  |---|---|
  | `GenerationLoop` looks up across keys | `CrossSessionPrefixHitTest` 3 |
  | `ContinuousBatchEngine` looks up across keys | `CrossSessionPrefixHitTest` 2 |
  | re-caching keeps the earlier entry | `PrefixCacheOwnerTest` 1, `CrossSessionPrefixHitTest` 1 (branch back), `GenerationLoopSessionTest.evict_session_clears_prefix_cache` 1 |
  | the owner lookup ignores the owner | `PrefixCacheOwnerTest` 5 |

**Tests run.**
- Focused (`mvn test -pl coordinator -am -Dtest=CrossSessionPrefixHitTest,PrefixCache*Test,KVCacheManagerTest,`
  `*PrefixCache*GatingTest,GenerationLoopSessionTest`): 44 run, 44 pass (`kvcache` 19, `coordinator` 25).
- The suites the change reaches, on the tree with step 3 (`mvn test -pl kvcache,coordinator,vision,juno-player -am
  -Dtest='**/kvcache/**,**/coordinator/**,**/vision/**,**/player/**'`, `node`'s suite filtered out since nothing in `node`
  changed after step 3's full suite): `kvcache` 141, `coordinator` 408 (1 skipped), `vision` 97, `juno-player` 158 (2
  skipped), 0 failures.
- On a real model ([`docs/perf-compare/20261111T060151Z-cross-session-prefix/`](../perf-compare/20261111T060151Z-cross-session-prefix/INDEX.md)),
  TinyLlama on the CPU, a stateless request and sessions `user-a` and `user-b` sending the same prompt: on HEAD's jar
  (`96ee5784b914097c`) `user-b` replies with garbled text on both schedules; on this tree's (`f0c7eb0186b6fe3a`) all three
  replies are "The capital of France is Paris." `smoke-consistency.sh` on this tree's jar: 54 of 54 pass.
- No performance gate: the change is in the prefix trie and the two lookups, run once per request, not the forward pass,
  MatVec, KV storage or vision. Step 3's A/B read jar `9af57a8054650216`; this jar differs from it by this change only.

**Docs.** `docs/agent-arch.txt` (prefix cache and KV ownership), `docs/howto.md` (`x_juno_session_id`), `CHANGELOG.md`.

**Kit probes.** None added: the consistency probes' "no prefix-cache hit for stateless traffic" still holds, and the new tests run
in `cpu.unit-tests`. The kit change this tier lists for item 2 lands with the sharing itself.

**Out-of-tier changes (README rule 9):** none. The defect predates this tier (the trie and its callers since the session work);
it is fixed here because step 4 builds on it.

**Decisions for the owner: how step 4 shares.** `KvPageTable.sharePrefix` (step 2) gives a paged request another table's prefix
pages, copy-on-write. Two choices the plan leaves open:
1. **What holds a prefix once its request ends.** Today only a session keeps its KV; a stateless request releases its pages when
   it finishes. The case the objective names (many clients sending the same system prompt, mostly stateless) needs prompt pages
   kept past their request.
   - (a) Share only from live sessions: a request whose prompt starts with a live session's cached prompt starts from that
     session's pages. No memory held beyond what sessions hold today; stateless traffic gains only when a session holds the
     prefix.
   - (b) (a), plus a retained prefix pool: when a request retires, its prompt's whole pages stay in the trie, held by the cache
     under a page budget, the least recently used released first. Stateless traffic shares. It needs a budget setting, which
     fails closed on a surface without paged KV.
   - **Recommendation: (b)**, budget default a fixed share of the page pool (say a quarter), settable to 0. The kit change this
     tier lists for item 2 ("a shared-prefix hit is counted" on stateless traffic) assumes it.
2. **The `static` schedule.** Its KV is dense, one array per request, so it has no pages to share; exit criterion 3 says "in both
   schedules".
   - (a) On `static`, copy the matched prefix rows from the holder's dense KV into the new request's (the prefill compute is
     skipped; memory is not shared). Safe by copying, not by counting.
   - (b) Share on `continuous` only. `static` keeps its gate and says so; criterion 3 and cross-surface row 4 are amended.
   - **Recommendation: (a).** `static` is the default schedule and the one cluster mode falls back to, and a copy of a
     500-token prefix costs far less than prefilling it on the CPU.
   Either way, a GPU request keeps its own device rows (owner decision 1 (d)): a shared prefix's rows are uploaded from host KV
   at admission, so the prefill is skipped on the device too.

**Exit criteria:** 2 of 15, unchanged.

**Owed to the owner:** decisions 1 and 2 above.

**Next for the executor:** step 4's sharing, as the owner decides.

### 2026-11-11: owner decisions 1 (b) and 2 (a); step 4, cross-session prefix sharing

**Owner decisions (2026-11-11).** 1 (b): finished requests' prompts are kept under a budget, least recently used released
first, settable to 0. 2 (a): on the `static` schedule a request copies the matched rows.

**Plan against code.** As the step 4 record above left it. Two refinements to the decisions, made while implementing:
- **A request never shares from a live session's KV**, only from kept copies. A live session's next turn writes its KV on
  another thread, possibly below the shared length (a branched turn), so a read could tear. Every finished request's prompt,
  session or stateless, is shared into a pool key that nothing writes, so sessions' prompts are shareable through the pool.
  On paged KV the copy is the same pages held once more, so it costs nothing until written.
- **The budget is in positions, defaulting to a quarter of the maximum heap at the model's host KV bytes per position, at
  most one context.** The decision said "a quarter of the page pool", but `KvBlockPool` has no capacity: it grows on the
  heap. The first default tried, one context, read 32,768 positions (the KV cap), about 8.6 GB on an 8B model, so it was
  replaced before any reading relied on it.

**What changed.**
- kvcache: `SessionKvLayout.sharePrefixK/V` (paged `PagedKvTensor.sharePrefix` over `KvPageTable.sharePrefix`; dense
  `DenseKvTensor.copyPrefixFrom`, stored bytes copied, not re-encoded); `PrefixCache.findLongestCommonPrefix` (the longest
  opening shared with any entry, whether or not the entry ends there: the existing lookup reports only nodes where an entry
  ends, so a request sharing only a system prompt matched nothing; caught by the pool tests); `SharedPrefixOptions`.
- node: `ForwardPassHandler`/`InferencePipeline` `shareKvPrefix`, `supportsKvPrefixSharing`, `hostKvBytesPerToken` (defaults
  false, false, 0); `LocalInferencePipeline` shares on each distinct handler and evicts the target everywhere if one refuses;
  `HandlerKvShare` used by the five inference handlers, which with GPU attention copy the shared rows into the request's
  device mirror on the layers attending there (owner decision 1 (d)'s "uploaded from host KV at admission"). Refused: a
  `q8_0` host KV beside GPU attention (the device copy would hold `q8_0`-rounded rows, not what the request computes), a
  node KV adapter, the LoRA handlers. vision: `VisionAwareForwardPassHandler` refuses any request with an image.
- coordinator: `SharedPrefixPool` (one lock over lookup, share and release); `GenerationLoop.generate`, `generateBatch`
  and `ContinuousBatchEngine` share at admission (when the request has no own-session hit; at least 16 positions, at most
  `promptLen - 1`) and keep the prompt at completion (not after a shift or cut). `generateBatch` still never reads the session
  trie (cross-surface row 4: shared through copies, the gate kept). Health: `sharedPrefixHits`, `sharedPrefixTokens`,
  `sharedPrefixHeldTokens`, `sharedPrefixBudget`, in `openapi.yaml` with the existing prefix counters.
- CLI: `--prefix-cache-tokens N` / `JUNO_PREFIX_CACHE_TOKENS` (`ConsoleMain`, `run.sh`, `run.bat`, cluster and local); a
  startup line states the budget and its MiB, or why sharing is off; an explicit value above 0 fails at startup where sharing
  cannot work (cluster, `--lora-play`, the cases above).
- Readings: `perf-lib.sh` exports `JUNO_PREFIX_CACHE_TOKENS=0` unless set, and the kit's servers get it unless a probe passes
  the flag: readings repeat prompts, and a repeat would time a skipped prefill. A jar without the setting ignores it.

**Tests (written with the code; each shown able to fail).**

| Test | Cases | Defect put in, then taken out (`cmp` after) | Fails |
|---|---|---|---|
| `SessionKvLayoutShareTest` | 12 | dense layout aliases the source | 3 |
| `KvPrefixShareHandlerTest` | 36 | share one row short | 24 (all that share) |
| | | dense layout aliases the source | 12 (the source's next token) |
| `SharedPrefixPoolTest` | 9 | `generate` keeps no stateless prompt | 4 |
| | | no release past the budget | 1 |
| | | exact-end lookup instead of the common prefix (the first version) | 3 |
| `VisionAwareForwardPassHandlerShareTest` | 1 | image check removed | 1 |
| `PrefixCacheOwnerTest` (new case), `SharedPrefixOptionsTest` | 1, 3 | | |

**Runs.**
- Real model ([`20261111T192150Z-shared-prefix`](../perf-compare/20261111T192150Z-shared-prefix/INDEX.md)), TinyLlama,
  sharing on against the same server with it off, nine requests: 9 of 9 replies equal on CPU and CUDA, both schedules.
  After a ~1,000-token system prompt the next request answers in 4.44 s against 47.82 s (CPU `static`), 5.38 against 49.68
  (CPU `continuous`), 0.94 against 1.33 (GTX 1080 `static`), 1.78 against 4.75 (GPU `continuous`), unpinned. Watch item: a
  ~25-position share on GPU `static` reads 0.12 s against 0.09 s (the device copy can cost more than a short prefill).
- `smoke-consistency.sh` on the final jar `a261d4df8de1a18c`: 56 of 56 pass (two new: shared-prefix hits counted, 19 each on
  CPU and GPU, with every repeated reply equal to its first, unshared reply).
- The unit suite on the final tree, the script and kit edits included (`mvn test -pl api,tokenizer,lora,node,coordinator,`
  `sampler,kvcache,health,registry,vision,metrics,juno-player,juno-kit -am -Dmaven.test.failure.ignore=true`, GPU groups on
  the GTX 1080, 72 min): `registry` 93, `lora` 116, `kvcache` 157, `health` 23, `tokenizer` 109 (2 skipped), `sampler` 90,
  `coordinator` 416 (1 skipped) and 1, `vision` 98, `metrics` 86, `juno-player` 158 (2 skipped), `juno-master` 15 and
  `juno-kit` 132 pass; `node` 1,434 run, 55 skipped, **1 failure**: `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThan`
  `TheReservesAllowance`, 99,024,896 bytes reported free with the device full against <= 67,108,864; alone twice after,
  95,289,344 and 96,468,992. The driver holdback recorded in the entries above and in Tier 02G's 2026-11-07 entry; nothing
  this step changes reaches that class. Bound not loosened. `ResidentQkvPathTest` passed this time.
- No performance gate: nothing per token changed (forward pass, MatVec, KV storage and reads are as step 3 left them); the
  share is one copy per admitted request. Throughput readings from here on run with sharing off (above), so the standing
  gate at tier close compares computed prefills.

**Kit probes.** The change this tier lists for item 2 is in: `consistency.tinyllama` and `gpu.consistency.tinyllama` keep "no
prefix-cache hit for stateless traffic" (no request resumes from its own key) and add "repeated prompts start from a kept
prompt", with an explicit `--prefix-cache-tokens 4096`.

**Owed, not done here.** `ModelLiveRunnerIT`'s cross-session-sharing check and `smoke-kv-cache.sh` (two concurrent clients
sharing a system prompt) land with step 8's smoke matrix, which also needs defragmentation and save/restore. The GPU
short-prefix watch item above.

**Exit criteria:** 3 of 15 (criterion 3 ticked above).

**Owed to the owner:** nothing new.

**Next for the executor:** step 5, defragmentation.
