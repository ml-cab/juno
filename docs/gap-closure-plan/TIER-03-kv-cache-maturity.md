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
on TinyLlama, CPU and CUDA, both schedules. Exit criterion 3 ticked (3 of 15). Step 5 done 2026-11-11: defragmentation as
releasing idle pages (item 3 corrected: the pool has no external fragmentation to compact; it kept every freed page's storage
forever): `KvBlockPool.reclaimIdle`, run in the background by `KvPageReclaimer` (`-Djuno.kv.reclaimPeriodMs`, default one
minute); no live page moves; pause met unpinned (0.957 ms added to a decode step, budget 5); health counters; `--lora-play` no
longer keeps every finished request's KV (defect, fixed). Exit criterion 4 ticked (4 of 15). Step 6 begun 2026-11-12: Tier
02G's item 3, option (a), landed first (README rule 9): GPU prefill attention and the tiled K-quant GEMM run in launches of
bounded work (`BoundedLaunches`), bit for bit one launch; at 32,768 tokens the slowest launch reads 57.94 ms on the GTX 1080 (bar
500 ms), and Mistral 7B's greedy replies at 8,192 tokens equal the previous build's on both schedules. Item 4 done 2026-11-13:
`q4_0` KV (`Q4_0KvCodec`, every store and reader, a startup warning that GPU attention keeps an FP16 copy). Its gate is scored on
`q8_0` keys with `q4_0` values by owner decision (a) 2026-11-13, and they meet NLL <= 1.02x and agreement >= 97% on every sweep
model; `q4_0` keys, which meet it on Mistral 7B only (TinyLlama 1.030x, Phi-3.5-mini 1.035x, Qwen2.5-3B 2.02x and 83.6%), stay a
setting with a startup warning naming the loss. Rotated 4-bit keys deferred, revisited before step 7 starts. Exit criterion 5
ticked (5 of 15). Item 7 implemented 2026-11-13: the device KV is stored in the `--cache-type-k/-v` types (the host codecs'
blocks), read in registers by a separate attention module (`GqaAttentionKvKernel`); bytes per position 0.531x / 0.281x, Mistral
7B sessions 1.83x to 2.00x, GPU greedy agreement 99.41% to 100% (`q8_0`) and 98.63% to 99.61% (`q8_0`/`q4_0`), all met; its
decode-rate bar (>= 0.95x, pinned), owner run 2026-11-13: TinyLlama 1.085x, Phi-3.5-mini 1.004x, Mistral 7B 0.999x, Qwen2.5-3B
**0.918x**. Owner decision 2 (a), 2026-11-14: the KV write and the decode attention reworked (unpinned, `q8_0` over FP16:
Qwen2.5-3B 1.02x to 1.04x, all four sweep models above 1.0x); the FFN GEMVs' share traced to allocation placement, not the
quantized path (decision 4). Owner re-run 2026-11-15: TinyLlama 1.235x, Qwen2.5-3B 1.030x, Phi-3.5-mini 1.056x, Mistral
7B 1.045x, every sweep model above the bar; exit criterion 7 ticked (6 of 15). A 2,048-token prefill with `q8_0` device KV
takes 1.21x to 1.31x the FP16 time, outside every bar. Owner decisions 2026-11-15: 1 (a), item 6 before step 7; 3 (a),
rotated keys deferred; 4 (a) and 5 (a), the FFN GEMV and the FP16 wide decode block as out-of-tier follow-ups; 6 (a),
item 7's prefill with a quantized device KV held to <= 1.10x FP16. Item 7's kit probes in 2026-11-15
(`gpu.kv-quality.<model>`, the GPU batch of eight at each KV type; first run 16 of 16 and 44 of 44). Decision 6 (a) begun
2026-11-15: traced, the prefill's extra time over FP16 was 9% to 45% the attention launch and the rest the host KV encoded on
the CPU after each layer (about 9 ns a value, one thread; 3.6 s a prefill on Phi-3.5-mini); the attention's quantized tile
staging reworked, now 0.90x to 0.95x the FP16 kernel with outputs unchanged bit for bit; prefill `q8_0` over FP16, unpinned,
1.09x to 1.17x in one window (Phi-3.5-mini 1.84x), so the 1.10x bar waits on decision 7 (the host encode) and decision 8 (the
widths it scores). Owner decisions 7 (a) and 8 (a), 2026-11-15. Decision 7 (a) in 2026-11-16: a whole-layer prefill window
writes the host KV from the device copy's stored rows (no CPU encode; a quarter of the bytes over the bus), host bytes
unchanged; prefill `q8_0` over FP16, unpinned, 0.90x to 0.99x in one window and 0.90x to 0.97x in 128-row windows, every
sweep model; kit probe `gpu.kv-prefill.<model>` in. The owner's pinned prefill gate (decision 8 (a)), 2026-11-16: 0.877x to
0.961x at both widths, 8 of 8 under 1.10x. Decision 5 (a) in 2026-11-16 (out of tier): the FP16 decode attention on the wide decode block, the launch 0.30x to 0.68x its old time at 2,048 keys, FP16 decode tg 1.07x to 1.30x unpinned; `q8_0` over the faster FP16 decode 0.955x to 0.977x; the owner's gate, 2026-11-16, pinned: FP16 decode after over before 1.067x to 1.281x, 4 of 4 over 1.00, prefill unchanged; `q8_0` over the new FP16 decode 0.963x, 0.975x, 0.977x and Qwen2.5-3B **0.938x**, so exit criterion 7 is open again (5 of 15). Owner decision 9 (a), 2026-11-16: the quantized decode loop takes up-front loads, before decision 4. Decision 9 (a) in 2026-11-17: K and V loaded up front, quant bytes unpacked without conversion instructions, no branch per block; the launch 0.74x to 0.99x its old time at 2,048 keys and under the FP16 launch on every sweep model; `q8_0` decode 0.971x to 0.983x FP16 unpinned. The owner's gate, 2026-11-17, pinned: `q8_0` decode 0.958x (Qwen2.5-3B) to 0.985x FP16, 4 of 4 at or above 0.95x; exit criterion 7 ticked again (6 of 15). Decision 4 (a) in 2026-11-17 (out of tier): gate and up in one launch (`q4k_gemv_pair`), bit for bit the two launches, 0.63x to 0.97x their device time at nine placements of the input vector, unpinned; LoRA gate and suites pass. The owner's gate, 2026-11-18, pinned: decode after over before, FP16 1.014x (Mistral 7B) to 1.075x (TinyLlama) and `q8_0` 1.015x (Phi-3.5-mini) to 1.045x (Qwen2.5-3B), 8 of 8 over 1.00, no cycle under; `q8_0` over FP16 read 0.951x to 0.978x and bounded at 0.956x or above, so criterion 7 stays met (6 of 15). Next: item 6, on a commit that holds decision 4 (a); then step 7.
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
   *(Corrected 2026-11-11, step 5: the premise does not hold for this pool, and the step is done in the form that does. Compaction
   moves live data so that free space becomes one reclaimable region, which a contiguous KV cache needs. `KvBlockPool` keeps every
   page as its own Java array behind a block table: any free page serves any request, attention reads each page where it is, and
   the heap compacts itself, so moving a live page reclaims nothing and risks exactly the corruption the exit criterion names. The
   memory the pool does fail to reclaim is different: a freed page keeps its array on the free list, so the pool holds its peak
   for the life of the process, after every session has ended too (a "full session evict" did not reclaim it either). Step 5
   therefore releases the storage of pages that stay idle for a whole period (`KvBlockPool.reclaimIdle`, run in the background by
   `KvPageReclaimer`), touching no live or shared page. The objective, exit criterion, tests and pause threshold are unchanged;
   see the execution record.)*
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
5. Add defragmentation as a background/on-demand compaction pass over `KvBlockPool`. *(2026-11-11: done as a background pass that
   releases idle pages' storage; item 3's correction says why nothing live is moved.)*
6. Add the Q4_0 KV codec, then item 7: the quantized device KV layout, the attention kernels' read
   path for it, and its accounting in `DeviceScratchBudget`. *(2026-11-10, owner decisions: first, before the device layout
   changes, [Tier 02G](TIER-02G-cuda-on-windows.md)'s item 3, option (a), bounded launches, recorded under README rule 9.)*
   *(2026-11-13: no step names item 6, real FP16 host storage, added to the scope on 2026-10-04 after these steps were written.
   Raised with the owner in the item 7 entry; the recommendation is to run it next, before step 7 fixes a stored format.
   2026-11-15, owner decision 1 (a): item 6 is step 6's last item, after item 7 closes and before step 7.)*
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
  no data corruption for any still-live page. *(2026-11-11: `KvBlockPoolReclaimTest`, `PagedKvReclaimHandlerTest`.)*
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
    *(2026-11-12, step 6: "agrees on 97% of the first 512 generated tokens" is read as the share of positions, not a
    comparison of two free-running replies, which part at their first differing token and then share nothing, so one
    early difference would read as a few percent. `KvQuantQualityLiveTest`: the F16-KV handler continues the quality
    text's first 1,024 tokens greedily for 512 tokens; that sequence is teacher-forced through both handlers in the
    same windows of 128, and the gate is the share of the 512 generated positions where their argmax agree. The first
    differing position, where a free-running Q4_0 reply would leave the F16 one, is reported beside it, as is the
    argmax agreement over the text's first 512 positions (the reading Tiers 02C and 02E called greedy agreement). The
    same reading serves item 6's and item 7's agreement thresholds.)*
    *(2026-11-13, owner decision (a): scored on `--cache-type-k q8_0 --cache-type-v q4_0`, 8-bit keys with 4-bit values. Measured,
    4-bit keys missed this gate on three of four sweep models and 4-bit values beside 8-bit keys met it on all four. `q4_0` keys
    stay available with a startup warning (`KvKeyPrecisionNotice`); rotated 4-bit keys are deferred (the 2026-11-13 decision
    entry).)*
  - real FP16 dense KV storage (item 6): greedy agreement with today's FP32-backed `f16` storage on
    **>= 99%** of the first 512 tokens on every sweep model, mean NLL **<= 1.005x**, and Phi-3.5-mini's
    live heap at a 2048-token request **<= 0.60x** the pre-tier reading;
  - quantized device KV (item 7): device KV bytes per position **<= 0.55x** the FP16 copy at `q8_0` and
    **<= 0.30x** at `q4_0`; the number of 2,048-token Mistral 7B sessions whose device KV fits beside the
    weights **>= 1.8x** the FP16 figure at `q8_0`; GPU decode tg at a 2,048-token context with `q8_0`
    device KV **>= 0.95x** the FP16 copy (attention reads fewer bytes and pays an in-register dequant;
    same-hour A/B, pinned); greedy agreement with the FP16 device copy on **>= 99%** of 512 tokens at
    `q8_0` and **>= 97%** at `q4_0`, on every sweep model;
    *(2026-11-13, owner decision (a): at `q4_0` the bytes bar is read on `q4_0` keys and values, a property of the format (18 bytes
    per 32 values), and the agreement bar on `q8_0` keys with `q4_0` values, the setting item 4's gate scores. That setting's device
    bytes, about 0.41x the FP16 copy (52 bytes per 32 key and 32 value elements against 128), are reported beside them.)*
  - defragmentation's pause: a compaction pass over a fragmented pool adds **<= 5 ms** to any single
    decode step at ctx 8192, or it runs off the decode path entirely. *(2026-11-11: the pass runs off the decode path, on its own
    thread, and `KvReclaimPauseMicrobench` measures what it still adds through the pools' lock: the longest decode step's KV writes
    overlapping a pass, less the median step alone, at ctx 8192 / batch 8 / page 16 on each sweep model's page count.)*

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
    equal to single on the device at each `--cache-type-k`/`--cache-type-v`. *(2026-11-12: item 4's half is in:
    `kv-quality.<model>` on the four sweep models, CPU suite, `QualityRunMain` twice with `f16` and `q4_0` host KV, held
    to 1.02x; its self-test runs on the synthetic model; `coverage.tsv` line `feature:kv-q4-0`. Item 7's half lands with
    item 7. 2026-11-13, owner decision (a): the quantized run is `q8_0` keys with `q4_0` values; any pair runs on request.
    2026-11-15: item 7's half is in: `gpu.kv-quality.<model>` on the four sweep models, CUDA, `QualityRunMain` with the
    handler on the GPU at `f16`, `q8_0` and `q8_0`/`q4_0`, each held to 1.02x and to the handler's notice naming the selected
    device formats; `gpu.consistency.tinyllama` runs its decode batch of eight at each of those types; `coverage.tsv` line
    `feature:device-kv-quant`.)*
  - Session save and restore (item 5): a restored session continues with the reply the unbroken session gives.
  - Defragmentation (item 3): replies unchanged under page churn, with the compaction counted. *(2026-11-11: in
    `schedule.tinyllama` and `gpu.schedule.tinyllama`, the continuous leg; `coverage.tsv` line `feature:kv-page-reclaim`.)*
  - The host KV footprint per session (item 6), as a reading.
  - Item 7's prefill bar (decision 6 (a)): the 2,048-token prefill time with `q8_0` device KV over FP16, as a reading, on the
    sweep models. *(2026-11-15: owed with decision 7, which settles what the bar scores; `DeviceKvPrefillTimeLiveTest` is
    the reading it would run. 2026-11-16: in: `gpu.kv-prefill.<model>` on the four sweep models, CUDA, `KvPrefillRunMain`
    with the handler on the GPU at `f16` and `q8_0`, the 2,048-token quality text in one window and in 128-token windows
    (decision 8 (a)'s widths); per width both sides' times and the ratio, a reading; a run whose GPU attention, device
    prefill region or selected device formats did not open fails the probe; `coverage.tsv` line
    `feature:device-kv-prefill`. First run 0.874x to 1.031x, every probe NOISY on the shared desktop,
    [`20261116T150000Z-kit-device-kv-prefill`](../perf-compare/20261116T150000Z-kit-device-kv-prefill/INDEX.md).)*
  - A change, the bounded launches landed before item 7 ([Tier 02G](TIER-02G-cuda-on-windows.md)'s item 3, option (a)):
    `gpu.kernel-timing` reads the heaviest of the launches the prefill paths split a window into, and adds a check at the
    longest window the device KV holds (32,768 tokens) against the same 500 ms bar. *(2026-11-12: in; its self-test expects
    both checks; `coverage.tsv` keeps its existing lines, `feature:gpu-kernel-watchdog` among them.)*

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
- [x] Defragmentation reclaims fragmented pages without corrupting live sessions.
      *(2026-11-11, step 5: idle pages' storage released after a whole period unused, no live or shared page moved or touched
      (item 3's correction says why). `KvBlockPoolReclaimTest` (fragmented pools, live pages' bytes and views unchanged, concurrent
      writers), `PagedKvReclaimHandlerTest` (every handler family, both host KV types, bit for bit across a reclaim); on TinyLlama,
      CPU and CUDA, 9,504 idle pages released, replies unchanged, the live session resumed from its own pages
      ([`20261112T040139Z-kv-page-reclaim`](../perf-compare/20261112T040139Z-kv-page-reclaim/INDEX.md)).)*
- [x] Q4_0 KV tier available alongside F16/Q8_0, meeting its quality threshold (greedy agreement >= 97%,
      NLL <= 1.02x F16 KV on every sweep model).
      *(Restated 2026-11-13, owner decision (a): the threshold is scored on `q8_0` keys with `q4_0` values; `q4_0` keys are a
      setting with a startup warning naming the loss. Ticked 2026-11-13: NLL 1.00019x, 1.00607x, 1.00202x and 0.99728x, agreement
      98.83%, 99.80%, 99.61% and 99.61% on TinyLlama, Qwen2.5-3B, Phi-3.5-mini and Mistral 7B
      ([`20261113T051804Z-kv-q4-0`](../perf-compare/20261113T051804Z-kv-q4-0/INDEX.md), section 2).)*
      *(2026-11-13: available; the threshold is not met, so this stays open. `q4_0` K and V: Mistral 7B 1.006x and 99.0%, TinyLlama
      1.030x and 97.1%, Phi-3.5-mini 1.035x and 98.1%, Qwen2.5-3B 2.02x and 83.6%; `q8_0` K with `q4_0` V meets both bars on all
      four ([`20261113T051804Z-kv-q4-0`](../perf-compare/20261113T051804Z-kv-q4-0/INDEX.md)). How it closes is the owner's
      decision, put in the 2026-11-13 entry.)*
- [ ] Real FP16 dense KV storage meets its threshold (greedy agreement >= 99%, NLL <= 1.005x, Phi-3.5-mini
      live heap at 2048 tokens <= 0.60x the pre-tier reading).
- [x] Quantized device KV (item 7): `--cache-type-k`/`-v` reach the GPU's KV copy on every handler with
      a GPU attention path, or the handler keeps FP16 and says so at startup; bytes per position, session
      capacity, decode tg and greedy-agreement thresholds met.
      *(2026-11-13: implemented on the three handlers with a GPU attention path (Llama family, Phi-3, Qwen3); a head width that
      is not a multiple of 32 keeps FP16 with a startup warning. Met, unpinned
      ([`20261113T194534Z-device-kv-quant`](../perf-compare/20261113T194534Z-device-kv-quant/INDEX.md)): bytes per position
      0.531x at `q8_0` and 0.281x at `q4_0`; Mistral 7B 2,048-token sessions 22 against 11 to 12 (1.83x to 2.00x); greedy
      agreement 99.41% to 100% at `q8_0` and 98.63% to 99.61% at `q8_0` keys with `q4_0` values, every sweep model. Open for
      decode tg (>= 0.95x, pinned): the owner's run
      ([`20261113T215326Z-device-kv-tg-gate`](../perf-compare/20261113T215326Z-device-kv-tg-gate/INDEX.md)) read TinyLlama
      1.085x, Phi-3.5-mini 1.004x, Mistral 7B 0.999x and Qwen2.5-3B 0.918x, under the bar; see the item 7 entry.
      2026-11-14: the KV write and the decode attention reworked
      ([`20261115T045828Z-device-kv-decode-path`](../perf-compare/20261115T045828Z-device-kv-decode-path/INDEX.md)):
      unpinned 1.015x to 1.225x, Qwen2.5-3B 1.023x and 1.044x; the owner's re-run owed.
      2026-11-15, met: the owner's re-run
      ([`20261115T200324Z-device-kv-tg-gate-rerun`](../perf-compare/20261115T200324Z-device-kv-tg-gate-rerun/INDEX.md)),
      CPU pinned, GPU clock recorded and matched per model, read TinyLlama 1.235x, Qwen2.5-3B 1.030x, Phi-3.5-mini 1.056x
      and Mistral 7B 1.045x. Not a bar: a 2,048-token prefill takes 1.21x to 1.31x the FP16 time (decision 6 of that
      entry). Decisions 6 (a) and 8 (a) added a prefill bar: a 2,048-token prefill with `q8_0` <= 1.10x the FP16 time, one
      window and 128-row windows, every sweep model, pinned. 2026-11-16, met: the owner's gate
      ([`20261116T163811Z-device-kv-prefill-gate`](../perf-compare/20261116T163811Z-device-kv-prefill-gate/INDEX.md)), CPU
      pinned, GPU clock recorded and matched within 25 MHz per model, read TinyLlama 0.961x / 0.961x, Qwen2.5-3B 0.944x /
      0.912x, Phi-3.5-mini 0.877x / 0.877x and Mistral 7B 0.953x / 0.952x (one window / 128-row windows), 8 of 8.
      2026-11-16, the decode bar's denominator moved: decision 5 (a) gave the FP16 decode attention the wide decode block,
      and against the faster FP16 decode `q8_0` reads 0.955x to 0.977x, unpinned
      ([`20261116T204541Z-fp16-wide-decode`](../perf-compare/20261116T204541Z-fp16-wide-decode/INDEX.md)). Ticked on the
      pinned evidence above until the owner's gate (`dist/fp16-wide-decode`, lane C, bar 0.95) re-scores it; a miss there
      reopens it. 2026-11-16, reopened: the owner's gate
      ([`20261116T224500Z-fp16-wide-decode-gate`](../perf-compare/20261116T224500Z-fp16-wide-decode-gate/INDEX.md)), CPU
      pinned, GPU clock recorded and matched per model, read `q8_0` over the new FP16 decode at TinyLlama 0.963x, Qwen2.5-3B
      **0.938x** (0.938x in each of three cycles), Phi-3.5-mini 0.975x and Mistral 7B 0.977x. Open for decode tg on
      Qwen2.5-3B; decision 9 (a). 2026-11-17: the quantized decode loop reworked (decision 9 (a),
      [`20261117T024602Z-quant-decode-loads`](../perf-compare/20261117T024602Z-quant-decode-loads/INDEX.md)): `q8_0` over
      FP16 0.971x to 0.983x unpinned (Qwen2.5-3B 0.971x); the owner's gate (`dist/quant-decode-loads`) scores it.
      2026-11-17, met again: the owner's gate
      ([`20261117T053104Z-quant-decode-loads-gate`](../perf-compare/20261117T053104Z-quant-decode-loads-gate/INDEX.md)), CPU
      pinned, GPU clock recorded and matched per model, read `q8_0` over FP16 at TinyLlama 0.985x, Qwen2.5-3B 0.958x,
      Phi-3.5-mini 0.974x and Mistral 7B 0.972x, 4 of 4 at or above 0.95x, on the tree committed as `9c33ab93` (tree hash
      checked). Every bar of this criterion is met.)*
- [ ] Disk-backed session save/restore round-trips correctly across a process restart.
- [ ] Standing throughput and allocation gate met (README, "Test infrastructure"): CPU and GPU tg and pp
      >= 0.95x the pre-tier build, allocation per token <= 1.10x, in-span GC pause total <= 1.25x.
      *(2026-11-10: open; the owner's pinned run at tier close. Step 3's unpinned A/B met the CPU bars on every lane, evidence
      only.)*
- [ ] Kit probes (README execution rule 10): the **Add** list above is in the kit's catalog, each with its
      self-test and its `coverage.tsv` line, and the **Keep passing** list passes on this tier's build.
      *(2026-11-13: item 4's probe is in, `kv-quality.<model>`, with its self-test and `feature:kv-q4-0`, scoring `q8_0` keys with
      `q4_0` values since owner decision (a); item 7's half, item 5's and item 6's are owed with those items.)* *(2026-11-13,
      later: item 7's half is still owed, by the executor: the GPU quality probe and batched equal to single on the device at
      each cache type. The tests it would run in the kit exist (`DeviceKvQuantQualityLiveTest`,
      `ResidentBatchDecodeTest`'s quantized cases).)*
- [ ] Cross-surface checklist fully resolved, vision/LoRA-play exclusions explicit and tested.
- [ ] Perf gate published; `paged_vs_dense%` <= 2% at ctx 8192 / batch 8, defrag pause within budget.
      *(2026-11-10: the `paged_vs_dense%` half read -6.35%, unpinned,
      [`20261111T051528Z-paged-kv-in-place`](../perf-compare/20261111T051528Z-paged-kv-in-place/INDEX.md); open for the
      defrag pause, owed by the executor at step 5, and the reading is retaken on the tier's final build.)* *(2026-11-11, step 5:
      the defrag pause met, unpinned: a reclaim pass on its own thread adds at most 0.957 ms to an overlapping decode step at
      ctx 8192 / batch 8 / page 16, every sweep model's page count
      ([`20261112T040139Z-kv-page-reclaim`](../perf-compare/20261112T040139Z-kv-page-reclaim/INDEX.md)). Open only for both
      readings retaken on the tier's final build.)*
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

### 2026-11-11, later: step 5, idle KV pages released (defragmentation); `--lora-play` kept every request's KV (defect, fixed)

**Plan against code, verified at HEAD `6ad5c728`.** `check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep or
build was running. Since the entry above, HEAD moved by the owner's commit of it (`1b90ca0c`) and by `6ad5c728`, which touched the
kit, the JFR clock check and one GPU test (recorded in [Tier 02F](TIER-02F-hardware-validation-kit.md)); neither touches this tier's
code. `git status`: `.github/` untracked, not this session's.
- Item 3 reads "compact `KvBlockPool`'s free/live pages when fragmentation crosses a threshold, to reclaim memory without requiring
  a full session evict", from the gap analysis's "paged KV's block pool only tracks free/live pages with no compaction". In the
  code every page is its own Java array (`List<float[]>`/`List<byte[]>`) behind a block table, attention reads each page in place
  (`KvRows`, step 3), any free page serves any request, and the heap compacts itself. So the pool has no external fragmentation
  to compact, and moving a live page would reclaim nothing while risking the corruption criterion 4 names.
- What the pool does not reclaim: `free` returned a page to the free list with its array, and nothing ever dropped it, so the pool
  held its peak for the life of the process, after every session had ended too (a full session evict did not reclaim it either).
- Decision gate: the premise was false, the objective, criterion, tests and pause threshold still apply, and the form that meets
  them is narrower (no live page moves). Recorded as a dated correction under item 3 and carried out; raised below for the owner
  to overrule if moving pages was meant.
- Found while testing the step (below): the three LoRA handlers (`LoraTrainableHandler`, `Phi3LoraTrainableHandler`,
  `Qwen3LoraTrainableHandler`) did not override `ForwardPassHandler.evict`, whose default does nothing, so under `--lora-play` every
  finished request's KV stayed in the handler's maps for the life of the process, on both schedules (dense layers on `static`,
  pages on `continuous`). The reclaim cannot release pages nobody frees, and cross-surface row 10 covers `--lora-play`, so it is
  fixed here, as step 4 fixed the defect it built on.

**What changed.**
- kvcache: `KvBlockPool` keeps idle pages (free, storage kept, reused first) and released ids (storage dropped, new zeroed storage
  when next allocated) in two `IntRing`s (new: an `int` ring buffer, first in first out as the `ArrayDeque<Integer>` it replaces,
  without boxing an id per allocate and free). `reclaimIdle()` releases the storage of the idle pages that stayed unused since the
  previous call, counted as the fewest idle pages seen in between (a page taken and returned meanwhile was needed and keeps its
  storage), in lock holds of at most 256 pages with a yield between them. `idlePages`, `storedPages`, `storedBytes`,
  `reclaimedPages`. `KvPageReclaimer` (new): one shared daemon thread runs `reclaimIdle` on every pool `PagedKvArena.fromOptions`
  makes (the serving arenas), every `-Djuno.kv.reclaimPeriodMs` (default 60000, `0` off, any other value that is not a whole number
  of milliseconds stops startup with an error naming it; read only where an arena is made, i.e. under `continuous`); pools are
  held weakly. `KvPageCounts` (new) sums the figures over arenas.
- node: `ForwardPassHandler.pagedKvArena()` (default null) on the eight handlers that hold a `SessionKvLayout` (through
  `HandlerKvShare.pagedArena`) and the vision decorator; `InferencePipeline.pagedKvArenas()`, each distinct arena once in
  `LocalInferencePipeline`. The three LoRA handlers' `evict` removes the request's layers and releases them (not recycled: they do
  not count running passes, which the inference handlers' recycling relies on).
- coordinator: `RequestScheduler.kvPages()`; `/v1/cluster/health` adds `kvPagesLive`, `kvPagesIdle`, `kvPagesReclaimed`,
  `kvPageBytes` (all 0 with dense KV), in `openapi.yaml` (`juno-api.yaml` has no health schema).
- `KvReclaimPauseMicrobench` and `scripts/performance-tests/kv-reclaim-microbench.sh` (new; not a smoke or compare script, so no
  `coverage.tsv` line), the pause gate below.

**Tests, written first** (they failed to compile on the missing API: `reclaimIdle`, `KvPageReclaimer`, `KvPageCounts`,
`pagedKvArena`). Each shown able to fail, the defect put in and taken out, `cmp` against a copy afterwards:

| Test | Cases | Defect | Fails |
|---|---|---|---|
| `KvBlockPoolReclaimTest` | 10 | reclaim ignores the period (releases every idle page) | 6 |
| | | allocate does not lower the fewest-idle count | 1 |
| | | released ids never reused (the pool always grows) | 2 |
| | | a released id given a live page's storage | 4 |
| `PagedKvReclaimHandlerTest` (every handler family of step 1's oracle, F16 and `q8_0`: a request live across a reclaim, and one started after it on released ids, read the logits they read without it, bit for bit) | 18 | a released id given a live page's storage | 18 |
| | | (first run, before the fix) the LoRA handlers keep finished requests' pages | 6 (nothing to reclaim) |
| `HandlerEvictReleasesKvTest` (every handler family, both schedules: a pass after evict reads none of the evicted rows; paged pages go back to the pool) | 18 | HEAD's LoRA handlers (no `evict`) | 6, exactly the three LoRA handlers on both schedules |
| `KvPageReclaimerTest` | 4 | `register` schedules nothing | 1 |
| `KvPageCountsTest`, `IntRingTest`, `LocalInferencePipelineKvArenaTest`, `KvReclaimPauseMicrobenchTest`, health case in `InferenceApiServerTest` | 1, 2, 1, 1, 1 | | |

`IntRingTest` checks 200,000 random operations against `ArrayDeque`. The kit's schedule-probe self-test runs the new check
(below).

**Pause gate (`<= 5 ms` added to any decode step at ctx 8192): met, unpinned.** `kv-reclaim-microbench.sh`, each sweep model's
page count at ctx 8192 / batch 8 / page 16 (eight live and eight finished requests built interleaved, so the idle pages sit among
the live ones), 180,224 to 294,912 idle pages per pass: a pass on its own thread adds at most **0.957 ms** to the KV writes of a
decode step it overlaps (Mistral 7B; TinyLlama 0.709, Qwen2.5-3B 0.720, Phi-3.5-mini 0.930; 69 to 81 overlapping steps per model).
The pass timed on its own, what a step would wait if it ran on the decode thread (the server does not), reads 2.99 to 4.50 ms
median, 4.87 to 5.41 ms max, and 13.3 ms on TinyLlama's first, interpreted pass. Run 1, on the first form of the code (boxed ids,
1,024-page holds, no yield), read 4.6 to 5.6 ms median on its own, and a writer contending with a reclaimer reached 3.7 to 5.9 ms
per step against 0.4 to 0.6 alone, because the pool's lock is not fair; the `int` ring, the 256-page holds and the yield are what
fixed it. [`20261112T040139Z-kv-page-reclaim`](../perf-compare/20261112T040139Z-kv-page-reclaim/INDEX.md).

**Real model** (same directory; TinyLlama, `continuous`, `--parallel 8`, sharing off, period 1 s for the reading): after a burst of
eight requests the pool held 9,504 idle pages, 175.2 MB; two periods later all were released and it held the one live session's
1,188 pages, 19.5 MB, on CPU and on the GTX 1080. The greedy reply after the reclaim equals the one before, and the session's next
turn resumed from its own pages (`prefixHits` 1). Under `--lora-play` (CPU) live pages stayed at the session's 1,188 through nine
stateless requests (before the fix each kept about 1,176 live) and 4,752 idle pages were released after two periods; the run hit
the script's 25-minute limit before its two closing requests (a decode batch runs one request at a time under `--lora-play`), and
`PagedKvReclaimHandlerTest` checks replies across a reclaim on the three LoRA handlers.

**Same-session A/B, unpinned** (same directory). A: HEAD `6ad5c728`, built from `git archive` in scratch, jar `03820803bf46dce6`.
B: this tree, `f9b39bf898e734e2`. A B A B A B, medians of three:

| Lane | A | B | B/A | Bar |
|---|---|---|---:|---|
| TinyLlama `continuous` F16, `n_prompt` 512: pp t/s | 31.49 (31.92, 31.49, 31.33) | 31.52 (31.52, 31.41, 31.73) | 1.001 | >= 0.95 |
| same: tg t/s | 10.28 (10.35, 10.28, 10.14) | 10.29 (10.29, 10.46, 10.16) | 1.001 | >= 0.95 |
| same: allocation per token in the recording | 305,795 B | 312,530 B | 1.022 | <= 1.10 |
| same: in-span GC pause total | 0 ms (all three) | 0 ms (all three) | | <= 5 ms |
| same: greedy replies | | | equal, 6 of 6 | |
| LoRA CPU, train ms per pass | 58,133 (57,733, 58,133, 58,200) | 58,133 (58,133, 58,133, 58,000) | 1.000 | <= 1.25 |
| LoRA CPU, playback t/s | 0.757 (0.759, 0.757, 0.756) | 0.763 (0.770, 0.762, 0.763) | 1.008 | >= 0.80 |
| LoRA GPU, train ms per pass | 3,000 (all three) | 3,000 (all three) | 1.000 | <= 1.25 |
| LoRA GPU, playback t/s | 11.44 (11.44, 11.48, 11.35) | 11.29 (11.29, 11.29, 10.85) | 0.987 | >= 0.80 |
| LoRA recall | 3 of 3, CPU and GPU | 3 of 3, CPU and GPU | | |

- The continuous lanes' pp is `prompt_tokens * 1000 / forward_pass_prefill_total_ms` and tg is `token_gen_tps_decode_derived`, as in
  step 3's record: the harness withholds the lane JSON's own `prompt_eval_tps`/`token_gen_tps` there, on both builds ("JFR
  timestamps disagree with the engine clock": 0.79 s of a 17 s chunked prefill and 3.27 s of a 10 s generation fall outside the
  forward-pass spans). Step 3's published continuous lanes carry the same nulls. Recorded as a harness finding for the owner (below);
  the standing gate's lanes are `static`.
- Every bar is met. Allocation per token on the continuous lane reads 1.022x: two of B's three repetitions read 312.6 kB and one
  305.4 kB, A's all 305.7 to 306.2 kB.
- The `q8_0` continuous lane was not run: the pool change is the same code for both page types, and the first attempt's
  `--models tinyllama` matched the Q2_K file too and doubled every invocation (stopped and restarted with the exact basename).

**Tests run.**
- Focused, during the work: `kvcache` 175 of 175; `PagedKvReclaimHandlerTest` 18, `HandlerEvictReleasesKvTest` 18,
  `LocalInferencePipelineKvArenaTest` 1, `PagedKvHandlerOracleTest` 18 and `InferenceApiServerTest` (with the new case) pass.
- The unit suite on the final tree (`mvn test -pl api,tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,`
  `metrics,juno-player,juno-kit -am -Dmaven.test.failure.ignore=true`, GPU groups on the GTX 1080, 79 min): `registry` 93, `lora`
  116, `kvcache` 175, `health` 23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator` 418 (1 skipped), `vision` 98, `metrics`
  86, `juno-player` 158 (2 skipped) and `juno-master` 15 pass; `node` 1,475 run, 55 skipped, **3 failures**; `juno-kit` 140 run,
  **1 failure**. None is in code this step changes:
  - `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`: 85,000,192 bytes reported free with the device
    full against <= 67,108,864; alone three times after, 88,866,816 each. The driver holdback recorded in the entries above and in
    Tier 02G's 2026-11-07 entry. Bound not loosened.
  - `DeviceUploadOrderingTest.theFirstGemvAfterAnUploadReadsTheWholeMatrix`, 2 of 4 (Q5_K 512x2048, round 2, row 269: 48.13 right
    after the upload against 19.85 once the device was idle; Q6_K, round 1, row 230: 35.31 against 25.51). This is the test HEAD's
    commit `6ad5c728` added to isolate a laptop GPU's `KQuantGemvRowsTest` failure (a copy from pageable memory returning before
    its transfer lands, the GEMV on a stream not ordered after it), recorded there as passing on this GTX 1080. Alone three times
    after: 4 of 4 each. So the race shows on this host too, intermittently, inside the suite. Raised with the owner below; this
    step changes no GPU code.
  - `ProbeRunnerTest.aProbeRunsInItsOwnJvmAndLeavesItsReportAndLog` (`juno-kit`): `NOISY` instead of `DONE`, other processes using
    10% to 21% of the CPU (the desktop: a browser, the system monitor). It fails the same way alone on this tree and on HEAD's tree
    (built in scratch), so it is the host's load, not the change.
  - The schedule probe's self-test, with the new check, and `CoverageTest` pass.
- `smoke-consistency.sh` (unmodified) on the candidate jar: 56 of 56 pass, including the shared-prefix and session checks on both
  schedules, CPU and GPU (`smoke-consistency.log` in the directory above).

**Cross-surface rows for this step.** 1 CPU: PASS (handler tests, real model). 2 CUDA: PASS (real model on the GTX 1080; the device
KV copy is per request and dense, owner decision 1 (d), and not paged). 3 ROCm: the reclaim is backend-agnostic host code;
NEEDS-AMD-HARDWARE for a run. 4 Static: N/A, dense KV has no page pool (health counters read 0, tested). 5 Continuous: PASS. 6
Local: PASS. 7, 8 Clusters: N/A, cluster launchers fall back to `static`. 9 LoRA training: N/A, ephemeral float caches. 10 LoRA
playback: PASS, and the evict defect fixed (tests, real model). 11 Vision: the decorator reports its text handler's arena; the
pages it holds are reclaimed like any other (handler and pool tests), not run live with an image. 12, 13 REST: the four health
counters (test, real model). 14 CLI: `-Djuno.kv.reclaimPeriodMs` through `JUNO_JVM_OPTS` or `java -D`; invalid values stop
startup (checked on the jar). 15 Embedding facade: reclaim runs for any embedder on `continuous` with no call; the counters are on the REST health endpoint,
which `JunoHttpClient` does not wrap (it has no health call), so an embedder reads them over HTTP. Not a new invocable capability.

**Docs.** `docs/howto.md` (`--kv-page-size` row: reclaim, the period property, the health counters), `docs/agent-arch.txt` (idle
pages and reclaim, the LoRA handlers' evict; the stale "nothing shares pages across sessions yet" sentence replaced),
`docs/performance.md` (block KV), `CHANGELOG.md`, `docs/perf-compare/README.md`.

**Kit probes.** Item 3's line is in: `schedule.tinyllama` and `gpu.schedule.tinyllama` start their continuous server with a
500 ms period, record a greedy reply before the workloads, wait until the health endpoint counts pages reclaimed, and check the
same request gets the same reply ("continuous: idle KV pages reclaimed, replies unchanged"); `coverage.tsv`
`feature:kv-page-reclaim`; the probe's self-test expects the check and a positive reclaimed count.

**Out-of-tier changes (README rule 9):** none. The `--lora-play` evict defect predates this tier; it is fixed here because the step
depends on it.

**Exit criteria:** 4 of 15. Criterion 4 (defragmentation reclaims fragmented pages without corrupting live sessions) ticked. The
perf-gate criterion's two halves are now both met unpinned (`paged_vs_dense%` at step 3, the pause here); it stays open because its
readings are retaken on the tier's final build.

**Raised with the owner** (no decision needed to continue):
1. Item 3 was carried out as releasing idle pages rather than moving live ones (correction above). If compaction that moves pages
   was meant, for example to prepare a device page pool for [Tier 07](TIER-07-continuous-batching.md), say so; the recommendation is
   to leave it, since nothing in the host pool needs it.
2. The static schedule's `DenseKvRecycler` keeps one finished request's dense layers (the larger) per handler for the life of the
   process, the dense counterpart of what this step releases. Item 6 (grow toward the known request length) bounds its size;
   releasing it after an idle period like the pages is a small follow-up the owner may want in item 6.
3. Harness finding: on the `continuous` schedule `compare-llama-cpp.sh` withholds the lane's `prompt_eval_tps` and `token_gen_tps`
   on every build (above), so continuous lanes are read from the forward-pass totals by hand.
4. GPU finding: `DeviceUploadOrderingTest` failed 2 of 4 inside the unit suite on this GTX 1080 (above) and passed alone, so the
   upload-ordering race that commit isolated is not confined to the laptop GPU. It belongs with that investigation in
   [Tier 02F](TIER-02F-hardware-validation-kit.md)/[Tier 02G](TIER-02G-cuda-on-windows.md); ordering the K-quant GEMV after the
   upload (or uploading from pinned memory) is the fix the test's own description points to.

**Owed to the owner:** nothing to run for this step.

**Next for the executor:** step 6, Tier 02G's item 3 option (a) (bounded attention launches, under README rule 9) first, then the
Q4_0 KV codec and the quantized device KV (item 7).

### 2026-11-12: step 6 begun: bounded GPU prefill launches (Tier 02G item 3, option (a); README rule 9)

**Plan against code, verified with the step 5 work uncommitted in the tree** (HEAD `6ad5c728`; built on, not redone).
`check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep, build or test run was in progress. `.github/` untracked,
not this session's. Step 6 opens with Tier 02G's item 3, option (a), by the 2026-11-10 owner decision (item 7's note, step 6's
note, README rule 1), before the Q4_0 KV codec and item 7. Every claim it depends on holds:
- Prefill attention launches in two places, both on rows of one cache: the device region's `PrefillWindowRegion.attendInside`
  (one `GqaAttentionKernel.launch` per layer over the whole window) and the host-staged window through
  `CudaGqaAttention.attendBatched` (also the decode entry for the handlers' non-resident decode). Decode launches elsewhere
  (`ResidentQkvPath`, one row; `ResidentBatchDecode`, one row per request) attend few rows and are left as they are.
- The tiled K-quant GEMM is reached by both prefill paths through `CudaMatVec.gemmPackedOnStream` ->
  `KQuantGemmKernel.multiplyHalf`, its only caller. The FP16 cuBLAS GEMM (`gemmHalfOnStream`, FP16-resident weights and the
  tiled kernel's fallback) is not the tiled GEMM the decision names; cuBLAS chooses its own kernels per shape, so a row split
  there would not keep the bits. Left as it is and raised below.
- A window is at most `DeviceKvCache.MAX_SEQ_LEN` (32,768) positions: the device KV and `RopeTable` stop there. That is "the
  longest window the device holds" for the kit probe.
- Bit-identity holds by construction. An attention block's output depends only on its own rows (their Q, cache pointers and
  sequence lengths) and `rowsPerBlock` (`gqa_attention.cu`: `b0 = blockIdx.x * rowsPerBlock`, the key range from the block's own
  rows); a GEMM output's accumulation order over super-blocks does not depend on the column tile, and rows are independent.
- Calibration: the MX150 tester run (`20261112T001051Z-remote-linux-amd64-core-i7-8550u-geforce-mx150`) reads Mistral 7B's
  attention at 2048 tokens 268.75 ms (7.2x this GTX 1080) and the down-matrix GEMM 209.69 ms (8.3x), the slowest GPU any kit run
  has measured.

**What changed.**
- node: `BoundedLaunches` (new): attention work = keys attended (a sliding window counts its own) x heads x the kernel's
  register head width, at most `ATTENTION_WORK_PER_LAUNCH` 1.2e10 a launch, split between whole blocks; GEMM work = rows x weight
  rows x weight columns, at most `GEMM_WORK_PER_LAUNCH` 2e11, in runs of whole 64-row tiles. Scaled from the MX150's readings the
  heaviest launch is 375 ms (attention) and 349 ms (GEMM), under 400; a 2048-token window on every sweep model is one launch of
  each, so no sweep or gate shape changes. `GqaAttentionKernel.launchBounded` (every launch keeps the whole window's
  `rowsPerBlock`; offsets Q, the pointer tables, the lengths and the output) and `registerWidth`; `KQuantGemmKernel.launchBounded`
  (every run keeps the whole batch's column tile), called by `multiplyHalf`. `PrefillWindowRegion` keeps the window's sequence
  lengths in an `int[]` sized to its capacity (allocated with the window); `CudaGqaAttention` passes its own.
- `GpuKernelTimings`: each reading is the slowest launch of the split, made as the prefill paths make it (attention: the slower
  of the launch with the most work and the last; K and V for the whole window, Q only for the launch's rows), named with the
  launch count; `measureWindow` (no LM head), `LONGEST_WINDOW`.
- juno-kit: `gpu.kernel-timing` adds a second check at `longestTokens` (default 32,768) against the same 500 ms bar, with its
  readings (13 in all); catalog description and peak device memory (152 -> 514 MiB) updated.

**Tests, written first.** `BoundedLaunchesTest`, `BoundedLaunchParityTest` and the new `GpuKernelTimingsTest` case failed to
compile on the missing API (`BoundedLaunches`, `launchBounded`, `measureWindow`, `LONGEST_WINDOW`); the probe's self-test failed
on the missing second check. Then:

| Test | Cases | Result | Defect put in, then taken out (`cmp` after) |
|---|---|---|---|
| `BoundedLaunchesTest` | 10 | pass | |
| `BoundedLaunchParityTest` (GPU; split vs one launch, raw bits) | 9: attention over one cache (heads 64, 96, 160 wide; a sliding window), over separate caches, the tiled GEMM (Q4_K, Q5_K, Q6_K; a padded row stride) | pass | attention output not offset per launch, and GEMM activation not offset per run: 8 of 9 fail (the ninth is the one-launch GEMM case, which takes no split) |
| `GpuKernelTimingsTest` | 4 (1 skipped: the CUDA-absent refusal) | pass | |
| `ProbeSelfTests` (kernel timing), `CoverageTest`, `ProbeCatalogTest` | 1, 7, 5 | pass | |

**Readings** ([`20261112T181947Z-bounded-launches`](../perf-compare/20261112T181947Z-bounded-launches/INDEX.md), unpinned):
- `gpu.kernel-timing` on the final tree: at 2048 tokens one launch each, slowest 37.75 ms (Mistral 7B attention); at 32,768
  tokens 98 to 212 attention launches per window, slowest 57.94 ms (Qwen2.5-3B's heads; Mistral 7B 54.95), the GEMM 10 runs of
  3,392 rows, slowest 42.26 ms. Both checks met. One launch over Mistral 7B's 8,192-token window read 596.51 ms before
  (`20261106T045718Z-gpu-kernel-timing`), about 9.5 s extrapolated to 32,768.
- `smoke-long-prompt-prefill.sh --models mistral --lengths "2048 8192" --baseline-jar <the tree before the change>`, GPU, both
  schedules: 16 of 16 pass; greedy text identical to the baseline at 2044 and 8472 prompt tokens on both schedules (at 8472 both
  attention and the GEMM split). TTFT, single readings: static 8472 347,788 ms against 347,935; continuous 8472 74,932 against
  76,329. The smoke's default 8 GiB heap failed the 8,192-token calibration with `OutOfMemoryError` before either build ran (the
  dense host KV doubling and FP32-backed `f16`, item 6); run at 16 GiB.
- No standing-gate A/B was run: below the budgets the launch sequence is the one it was (one launch, the same arguments), and
  the gate's lanes (`n_prompt` 128 and 512) sit below them. The owner's pinned standing gate at tier close covers it.

**Tests run.** The unit suite on the final tree (`mvn -o test -pl api,tokenizer,lora,node,coordinator,sampler,kvcache,health,`
`registry,vision,metrics,juno-player,juno-kit -am -Dmaven.test.failure.ignore=true`, GPU groups on the GTX 1080, 80 min):
`registry` 93, `lora` 116, `kvcache` 175, `health` 23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator` 418 (417 plus
the integer-kernel execution's 1; 1 skipped), `vision` 98, `metrics` 86, `juno-player` 158 (2 skipped), `juno-master` 15 and
`juno-kit` 140 pass; `node` 1,495 run (the step 5 suite's 1,475 plus this change's 20), 55 skipped, **3 failures**, the same
three as step 5's suite, in no code this change touches:
- `DeviceUploadOrderingTest.theFirstGemvAfterAnUploadReadsTheWholeMatrix`, 2 of 4 (Q5_K 512x2048, round 1, row 219: -15.84 right
  after the upload against -10.02 once the device was idle; Q6_K 512x2048, round 1, row 190: 4.52 against 44.44). Alone three
  times after: 4 of 4 each. The upload-ordering race raised in step 5's entry (finding 4); still open, and still not this tier's code.
- `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`: 82,247,680 bytes reported free with the device
  full against <= 67,108,864; alone three times after, 75,956,224, 76,021,760 and 76,021,760. The driver holdback recorded in
  step 5's entry and Tier 02G's 2026-11-07 entry. Bound not loosened.
- `ProbeRunnerTest`, which failed `NOISY` in step 5's suite under desktop load, passed this time.

**Cross-surface rows for this change.** 1 CPU: N/A, GPU launches only. 2 CUDA: PASS (parity tests, real model both schedules,
kit probe). 3 ROCm: N/A, the attention and tiled GEMM kernels are CUDA-only. 4 Static, 5 Continuous: PASS (smoke). 6 Local:
PASS. 7, 8 Clusters: the nodes run the same prefill paths; the split is inside the kernel launch sites, not run on a cluster
here. 9 LoRA training: N/A, no device prefill attention. 10 LoRA playback: N/A, `--lora-play` keeps the host path. 11 Vision:
the text handler's prefill takes the same launch sites; not run live. 12 to 15: N/A, no surface change.

**Docs.** `docs/howto.md` (device prefill windows: bounded launches), `docs/agent-arch.txt` (`BoundedLaunches`, the two
kernels, `GpuKernelTimings`, the probe), `CHANGELOG.md`, `docs/perf-compare/README.md`.

**Out-of-tier changes (README rule 9).** This change is Tier 02G's item 3, option (a), landed here by owner decision. It
touches the GPU prefill attention and GEMM launch sites. It is not a measurement boundary for the sweep: at `n_prompt` 128, 512
and 2048 every sweep model takes the single launch it took, and no published baseline is invalidated. Windows longer than the
budget (above about 2,900 tokens on Mistral 7B's heads) now launch in pieces; their results are bit for bit as before.

**Exit criteria:** 4 of 15, none ticked by this change. The kit-probe criterion's line for this change is in (the **Kit probes**
section above).

**Raised with the owner** (no decision needed to continue):
1. The FP16 cuBLAS GEMM (`CudaMatVec.gemmHalfOnStream`: FP16-resident weights, and the tiled kernel's fallback when its module
   does not load) is one cuBLAS call per matmul and is not split. A sweep model's Q4_K_M weights never take it. On this card a
   Mistral-sized FP16 matmul over a 32,768-row window is estimated at about 0.45 s (3.85e12 FLOP at FP32 rate; not timed). Splitting
   it would change bits (cuBLAS picks its kernels per shape). Recommendation: leave it unless a Windows run times it near the
   watchdog.
2. Finding, both builds: on the `static` schedule an 8,472-token Mistral 7B prompt reads 348 s to first token on the GTX 1080
   against 75 s on `continuous`. Not caused by this change (the baseline reads the same). Worth a decomposition before Tier 07's
   long-prompt readings; it is not this tier's scope.
3. The Windows half of Tier 02G's item 3 (the longest kernel timed against the watchdog on Windows) can now read the 32,768-token
   check on a tester's Windows GPU through `gpu.kernel-timing`.

**Owed to the owner:** nothing to run for this change.

**Next for the executor:** the rest of step 6: the Q4_0 KV codec (item 4; `Q4_0KvCodec` beside `Q8_0KvCodec`, round-trip test
first, the quality live test against F16 KV), then item 7 (quantized device KV).

### 2026-11-13: step 6, the Q4_0 KV codec (item 4): in on every surface `q8_0` reaches; its quality gate missed on three of four sweep models

**Plan against code, verified with the step 5 and bounded-launch work uncommitted in the tree** (HEAD `6ad5c728`; built on, not
redone). `check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep, build or test run was in progress. `.github/`
untracked, not this session's. Every claim the step depends on held:
- `KvElementType` was `F16` and `Q8_0`; a serialized KV block (`PagedKvCodec`, `NodeKVCacheAdapter`) names each side's type by
  its ordinal, so `Q4_0` is appended (ordinal 2) and existing blocks read as written.
- The quantized slice decode was called as `Q8_0KvCodec.decodeRange` at five attention sites (`GqaMath` keys and values,
  `GqaWindowAttention` keys and values, `GqaVectorHead` keys); dense, paged and serialized storage each switched on the type.
- `HandlerKvShare.supported` refuses any non-`f16` host type beside GPU attention, so `q4_0` is refused there as `q8_0` is.
- Nothing said at startup that a quantized `--cache-type` on a GPU attention launch sets the host rows only (item 7: "keeps FP16
  and says so at startup, never silently"); it was silent for `q8_0`.
- The four sweep models are on disk.
- One wording needed a definition, recorded under the perf gate (dated): greedy agreement is the share of 512 generated positions
  where the two caches' argmax agree along the `f16` cache's own greedy continuation, both sides teacher-forced the same way.

**What changed.**
- kvcache: `Q4_0KvCodec` (new): the GGUF block, 18 bytes per 32 values, FP16 `d = max / -8` from the element of largest
  magnitude, element `j` in the low and `j + 16` in the high nibble of byte `j`, decoded `d * (q - 8)`; `decodeRange` is the
  loop `decode` runs, so a slice reads the whole-row decode's bits. `KvRowCodec` (new): row size, encode and decode for every
  type in one place; `DenseKvTensor` (its packed array now serves both block types), `KvBlockPool`, `PagedKvCodec` and
  `NodeKVCacheAdapter` go through it. `KvRows.decodeRange`: the one entry point attention calls, a `final boolean` choosing the
  codec. `KvElementType.Q4_0` (`q4_0`, `q4`); the parse error lists `f16|q8_0|q4_0`. `CacheTypeOptions.usesQuantized` is any
  non-`f16` side (it read `Q8_0` only), so LoRA training's ephemeral-KV warning covers `q4_0`.
- node: the five attention sites call `KvRows.decodeRange`. `DeviceKvTypeNotice` (new): the startup warning for a CUDA launch
  with GPU attention on a handler that runs the kernel and a quantized type, not LoRA; `consoleNotice` printed by `ConsoleMain`,
  `announce` logged once where `GpuAttentionMirror.open` and `LlamaTransformerHandler` open the kernel.
- CLI help: `ConsoleMain`, `scripts/run.sh`, `scripts/run.bat` (three subcommands each), `compare-llama-cpp.sh`.
- juno-kit: `KvQualityProbe` (new), `kv-quality.<model>` on the four sweep models in the CPU suite; `QualityProbe.score` with
  JVM properties (extracted, the quality probe's behaviour unchanged); `QualityRunMain` reports its run's cache types.

**Tests, written first** (they failed to compile on the missing `KvElementType.Q4_0` and `Q4_0KvCodec`):

| Test | Runs | Notes |
|---|---|---|
| `Q4_0KvCodecTest` (new) | 8 | the published byte layout on a hand-worked block; the extreme exact and its opposite clamped to 7 steps; every value within one step of its block's scale and the error energy within 0.8x to 1.25x of the uniform-quantization model (`step^2 / 12`), coarser than `Q8_0`; zeros and a padded last block; every slice equal to the whole-row decode, bit for bit; bad ranges |
| `CacheTypeOptionsTest` | 7 (was 4) | `q4_0`/`q4` parse, CLI names round trip, ordinals stable, `usesQuantized` with `q4_0` on one side. Its `parse_rejects_unknown` asserted that `q4_0` is refused: that expectation is the one this item reverses, so it now refuses `q4_1` and checks the message lists `q4_0` |
| `DeviceKvTypeNoticeTest` (new) | 3 | the warning on CUDA with GPU attention (`auto`, `on`) for `q4_0` and `q8_0`; none for `f16`, CPU, ROCm, LoRA, Phi-2, Qwen3-MoE, `--gpu-attention off`; the handler log once per process |
| By enumeration of `KvElementType.values()` | | `PagedKvAttentionOracleTest` 41 to 73, `GqaWindowAttentionTest` 16 to 36, `KvRowsTest` 14 to 20, `KvBlockPoolTest`, `KvPageTableTest`, `KvBlockPoolReclaimTest` |
| `q4_0` added by hand | | `PagedKvHandlerOracleTest` 18 to 27 (all 9 families, LoRA handlers included), `PagedKvReclaimHandlerTest` 18 to 27, `KvPrefixShareHandlerTest` 36 to 54, `KvContextShiftTest` 25 (dense and paged `q4_0`), `SessionKvLayoutShareTest` 16, `KvWriteRowsTest` 5, `DenseKvTensorTest` 4, `PagedKvArenaManagerTest` 3 and `NodeKVCacheAdapterTest` 15 (a flushed `q4_0` side restores the rows the tensor read, bit for bit) |
| `KvQuantQualityLiveTest` (new, opt-in `-Djuno.test.kvQuality=true`) | 4 | below |
| kit `ProbeSelfTests` (`kv-quality` on the synthetic model), `CoverageTest`, `ProbeCatalogTest` | 1, 7, 5 | pass |

Defects in the tests, found while running them: `KvRowsTest.readRow` and a `KvBlockPoolReclaimTest` helper decoded every
non-`f16` page as `Q8_0` (now through `KvRows.decodeRange`); `KvRowsTest`'s reused-dense check compared exact `0.0f` bits, and
zeroed `q4_0` storage decodes as `-0.0` (an FP16 `+0` scale times step `-8`), which is zero and which no attention sum can see
(every sum starts at `+0`), so it compares values; the first `Q4_0KvCodecTest` stepped past an 80-value row's last block, and
its first error check was a guessed RMS range (0.04 to 0.10) that mixed-scale rows read at 0.101, replaced by the
uniform-quantization model before anything else changed.

**Shown able to fail**, each defect put in and taken out (`cmp` against a copy afterwards):

| Defect | Fails |
|---|---|
| `KvRows.decodeRange` decodes `q4_0` pages as `q8_0` | `KvRowsTest` 5, `PagedKvAttentionOracleTest` 28, `PagedKvHandlerOracleTest` 7, `GqaWindowAttentionTest` 20 |
| `Q4_0KvCodec` reads the high-nibble half from the low nibble | `Q4_0KvCodecTest` 2, `DenseKvTensorTest` 1, `PagedKvArenaManagerTest` 1 (the in-place and gathered paths share the codec, so the oracles cannot see it; the codec tests do) |

**Quality readings** ([`20261113T051804Z-kv-q4-0`](../perf-compare/20261113T051804Z-kv-q4-0/INDEX.md), the candidate tree,
CPU, deterministic): NLL ratio against `f16` (bar <= 1.02), greedy agreement (bar >= 97%), first differing generated position.

| Model | `q4_0` K and V | `q8_0` K and V | `q8_0` K, `q4_0` V |
|---|---|---|---|
| TinyLlama 1.1B | **1.03006**, 97.07%, 1 | 0.99977, 99.80%, 20 | 1.00019, 98.83%, 20 |
| Qwen2.5-3B | **2.01752**, **83.59%**, 1 | 1.00073, 99.22%, 1 | 1.00607, 99.80%, 17 |
| Phi-3.5-mini | **1.03538**, 98.05%, 4 | 0.99875, 99.80%, 58 | 1.00202, 99.61%, 26 |
| Mistral 7B | 1.00612, 99.02%, 0 | 0.99909, 99.80%, 35 | 0.99728, 99.61%, 3 |

- The `f16` side reproduces between runs and equals the integer-kernel quality reading on TinyLlama (2.45609), and its own
  teacher-forced argmax equals the tokens it generated at decode width on 512 of 512 positions on every model.
- The loss is the format's and the keys': the same read path is lossless at 8 bits, the codec matches the published block and
  the uniform-quantization model, and 4-bit values cost at most 0.6% beside 8-bit keys. Qwen2.5-3B collapses with 4-bit keys
  (perplexity 30.7 against 5.46); the likely cause, unmeasured here, is a few key channels far larger than the rest setting
  each block's one scale.
- So `q4_0` as the plan scoped it (both K and V) meets the gate on Mistral 7B only, and exit criterion 5 stays open.

**Throughput A/B (unpinned, same directory).** A: the tree before this change, jar `2e7ab50ca9c2cdf4` (published as
`20261112T181947Z-bounded-launches`'s kit-run jar); B: `9fc4975d8a843f6e`. TinyLlama CPU `static` `n_prompt` 512, A B A B A B,
medians: `f16` pp 1.004, tg 0.997, allocation per token 1.000; `q8_0` pp 1.001, tg 0.995, allocation 0.974; no GC pause; greedy
replies equal 3 of 3 on both lanes. `q4_0` reads 0.954x pp and 0.975x tg of `q8_0` at this length. Evidence for the standing
gate, not its pinned run.

**Launch checks** (same directory): TinyLlama `q4_0` on the GTX 1080 prints the warning and replies "The capital of France is
Paris."; on the CPU, `static` and `continuous`, no warning and the same reply.

**Tests run.**
- Focused: `kvcache` 216 of 216; `node` 266 of 266 (`PagedKvAttentionOracleTest`, `GqaWindowAttentionTest`,
  `PagedKvHandlerOracleTest`, `KvPrefixShareHandlerTest`, `PagedKvReclaimHandlerTest`, `NodeKVCacheAdapterTest`,
  `DeviceKvTypeNoticeTest`, `LlamaTransformerHandlerCacheTypeParityTest`, `CpuKernelVectorEscapeTest`,
  `HandlerEvictReleasesKvTest`); kit 14 of 14.
- The unit suite on the final code (`mvn -o test -pl api,tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,`
  `metrics,juno-player,juno-kit -am -Dmaven.test.failure.ignore=true`, GPU groups on the GTX 1080, 73 min): `registry` 93, `lora`
  116, `kvcache` 216, `health` 23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator` 418 (1 skipped), `vision` 98,
  `metrics` 86, `juno-player` 158 (2 skipped; `LauncherParityTest` and `WindowsLauncherScriptTest` with the help edits) and
  `juno-master` 15 pass; `node` 1,588 run, 56 skipped, **3 failures**; `juno-kit` 141 run, **1 failure**. None is in code this
  change reaches, and none was seen differently before it:
  - `DeviceUploadOrderingTest.theFirstGemvAfterAnUploadReadsTheWholeMatrix`, 2 of 4 (Q5_K row 256, Q6_K row 191); alone three
    times after, 4 of 4 each. The upload-ordering race of step 5's finding 4, still open.
  - `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`: 84,475,904 bytes free with the device full
    against <= 67,108,864; alone three times, 79,888,384, 80,609,280 and 81,657,856. The driver holdback. Bound not loosened.
  - `ProbeRunnerTest.aProbeRunsInItsOwnJvmAndLeavesItsReportAndLog` (kit): `NOISY`, other processes at 7% (suite) and 11% to
    14% (alone) of the CPU; step 5 saw it fail the same way on HEAD's tree.
- `KvQuantQualityLiveTest`: run 1 (`q4_0`, four models, 175 min) 4 run, 3 failures, the three NLL assertions above; run 2
  (`q8_0`, `q8_0/q4_0`, three models, 95 min) pass; run 3 (Phi-3.5-mini) (`q8_0`, `q8_0/q4_0`, 132 min) pass.
- `check-plan-thresholds.sh`: ok.

**Cross-surface rows for this item.** 1 CPU: PASS (oracles, quality runs, launches). 2 CUDA: the type sets the host rows (CPU
attention fallback, flush, sharing refused beside GPU attention as for `q8_0`); GPU attention keeps its FP16 copy, said at
startup (tested; launch). Item 7 changes the copy. 3 ROCm: attention reads the host rows through the CPU path tested here;
NEEDS-AMD-HARDWARE for a run. 4 Static, 5 Continuous: PASS (dense and paged in every oracle; launches). 6 Local: PASS. 7, 8
Clusters: `ClusterHarness` forwards `JUNO_CACHE_TYPE_K/V` to the nodes, whose handlers build the same storage; not run on a
cluster here. 9 LoRA training: the ephemeral-KV warning now fires for `q4_0` (`usesQuantized`, tested). 10 LoRA playback: PASS
(the three LoRA handlers in the handler oracle, reclaim and prefix tests with `q4_0`). 11 Vision: the decorator's text handler
reads the same storage; not run live. 12, 13 REST: N/A, no field. 14 CLI: PASS (help in both launchers and `ConsoleMain`; an
unknown type stops startup naming the three). 15 Embedding facade: as for `q8_0`, through the `JUNO_CACHE_TYPE_K/V` properties
the handler reads at load; `JunoPlayer` has no cache-type method for either.

**Docs.** `docs/howto.md` (the two flag rows: `q4_0`, the per-model quality, the GPU copy, the context-shift note),
`docs/agent-arch.txt` (codecs, `KvRowCodec`, `KvRows.decodeRange`, `DeviceKvTypeNotice`, the kit probe), `README.md` (block KV),
`docs/performance.md` (the quality table), `CHANGELOG.md` (Session 145), `docs/perf-compare/README.md`.

**Kit probes.** Item 4's line is in: `kv-quality.<model>`, four sweep models, CPU suite, with its self-test and
`feature:kv-q4-0`.

**Out-of-tier changes (README rule 9):** none. `compare-llama-cpp.sh`'s help lists `q4_0` (it passes the value through).

**Exit criteria:** 4 of 15, unchanged. Criterion 5 stays open on the readings above.

**Decision for the owner: how item 4 closes.** The plan scoped `q4_0` for K and V; measured, 4-bit keys miss the gate on three
of four sweep models and 4-bit values do not.
1. **(a) Recommended:** score item 4 on `--cache-type-k q8_0 --cache-type-v q4_0`, which meets both bars on every sweep model
   and stores 0.76x of `q8_0`'s bytes (0.20x of float32). `q4_0` keys stay available, with a startup warning naming the measured
   loss and the `q8_0` keys alternative (a few lines in `DeviceKvTypeNotice`'s neighbourhood plus a test); the gate text and
   criterion 5 are restated to say so. Item 7's `q4_0` device threshold (greedy >= 97%) is read the same way.
2. (b) Keep `q4_0` keys in the gate and make them good enough: a fixed orthogonal transform (a 32-point Walsh-Hadamard) of each
   key block before it is quantized, with the query transformed the same way at attention, so a few large channels no longer
   set a block's scale. New scope: the CPU read path, context shift (moved keys transformed back before re-rotation) and item
   7's device kernels all change, with its own tests and readings; a session or two, and whether it reaches 1.02x on Qwen2.5-3B
   is unmeasured.
3. (c) Close criterion 5 on the miss by owner decision, `q4_0` documented with its per-model quality.
4. (d) Refuse `--cache-type-k q4_0` at startup (values only). Not recommended: it removes a working setting that is lossless
   enough on Mistral 7B.

**Owed to the owner:** the decision above. Nothing to run for this item; the standing gate stays the pinned run at tier close.
*(Decided 2026-11-13: (a); the next entry.)*

**Next for the executor:** item 7, the quantized device KV, starting with its `q8_0` layout, which no option above changes; its
`q4_0` half follows the decision.

### 2026-11-13, later: owner decision (a): item 4 is scored on `q8_0` keys with `q4_0` values; `q4_0` keys warned at startup

**Owner decision (2026-11-13): (a).** Item 4's gate (NLL <= 1.02x, greedy agreement >= 97%, every sweep model) is scored on
`--cache-type-k q8_0 --cache-type-v q4_0`. The bars are unchanged; the setting they are read on changed, on the measurement that
the loss is the keys'. `q4_0` keys stay a working setting (Mistral 7B 1.006x), with a startup warning. The readings that close
it are the previous entry's run 2, on the candidate tree whose codec and read path this change leaves untouched.

**What changed.**
- `KvKeyPrecisionNotice` (kvcache, new): for `--cache-type-k q4_0`, whatever the value type, the line naming the measured loss
  and `--cache-type-k q8_0 --cache-type-v q4_0`. `SessionKvLayout.fromEnv`, which every handler (LoRA included) builds its
  layout through, logs it once per process at WARNING, so server and cluster node logs carry it; `ConsoleMain` prints it under
  the KV policy line before branching into local, cluster or LoRA mode, since console logging is off without `--verbose`.
- `KvQuantQualityLiveTest` gates `q8_0/q4_0` (default `-Djuno.test.kvQuality.types`); any pair, `q4_0` included, is read and
  printed on request. `KvQualityProbe` (kit) scores `q8_0` keys with `q4_0` values by default (its `type` argument takes one type
  or a `k/v` pair); the catalog title, its self-test and `feature:kv-q4-0` follow.
- Gate text, item 7's `q4_0` bars and criterion 5 restated above (dated). Criterion 5 ticked.
- Docs: how-to (`--cache-type-k` row: use `q8_0` keys with `q4_0` values for a 4-bit cache; `q4_0` keys warned), README, the
  performance doc's 2026-11-13 paragraph, agent-arch, CHANGELOG (Session 145), the help text of `ConsoleMain`, `run.sh` and
  `run.bat`.

**Tests.**

| Test | Run | Result | What it shows |
|---|---|---|---|
| `KvKeyPrecisionNoticeTest` (kvcache, new) | 3 | pass | warned for `q4_0` keys with every value type and names both settings; silent for `f16` and `q8_0` keys with every value type, `q4_0` values included; logs one WARNING per process. Shown able to fail: keyed on the value type instead, all 3 fail |
| `SessionKvLayoutTest`, `CacheTypeOptionsTest` (kvcache) | 2, 7 | pass | unchanged behaviour with the notice in `fromEnv` |
| `ProbeSelfTests#kvQuality*` (juno-kit) | 1 | pass | the probe on the synthetic model with `q8_0` keys and `q4_0` values: its readings, check name, `quantizedRunKv = cache-type-k=q8_0 cache-type-v=q4_0` |
| `CoverageTest` (juno-kit) | 7 | pass | `coverage.tsv` and the catalog in step |
| `KvQuantQualityLiveTest` (node) | 1 | skipped | opt-in; compiles with the new gate |

Launch, TinyLlama, CPU, greedy, "What is the capital of France?" through `./juno local`: `--cache-type-k q4_0 --cache-type-v
q4_0` prints the warning under the policy line; `--cache-type-k q8_0 --cache-type-v q4_0` prints nothing; both reply "The capital
of France is Paris." No hot path changed (one check when a layout is built), so no throughput gate applies.

**Deferred: rotated 4-bit keys (option (b) of the previous entry).** Not owed work and not an exit criterion. Revisit before
step 7 (item 5, disk-backed session save and restore) starts, on item 7's readings: if the GPU's KV copy at `q8_0` keys with
`q4_0` values still bounds how many sessions fit on the card, or a long-context target needs the full `q4_0` size (0.28x of FP16
against 0.41x), take (b) then. Before disk persistence, the stored meaning of `q4_0` keys can change freely; after it, saved
sessions hold unrotated keys and rotated ones need a type of their own. Add to (b)'s scope: a shared prefix's device rows are
uploaded from host KV, which `HandlerKvShare` allows only with an `f16` host KV today; if item 7 lifts that for quantized types,
the upload decodes rotated keys back.

**Exit criteria:** 5 of 15. Criterion 5 ticked.

**Owed to the owner:** nothing for this change. The standing gate stays the pinned run at tier close.

**Next for the executor:** item 7, the quantized device KV, starting with its `q8_0` layout.

### 2026-11-13, latest: step 6, item 7: the device KV stored in the `--cache-type-k/-v` types; its decode-rate bar missed on Qwen2.5-3B

**Plan against code, verified with HEAD `79fa87b0` (step 5, bounded launches and the `q4_0` codec committed), tree clean but
for the untracked `.github/` (not this session's).** `check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep,
build or test run was in progress. Every claim item 7 depends on held: `DeviceKvCache` stored FP16 only; its writes were the
host-staged `appendToken`/`appendWindow`/`replacePrefix`, `writeWindowOnDevice` (the resident decode region and the prefill
region, `PrefillWindowKernels.toHalf`) and `kv_append_rows` (the decode batch); its readers were the one attention kernel
(`gqa_attention.cu`, decode rows and prefill tiles) launched from four sites (`CudaGqaAttention`, `ResidentQkvPath`,
`ResidentBatchDecode`, `PrefillWindowRegion`); the shift kernel rotated FP16 keys; `DeviceScratchBudget.kvMirrorBytes` counted
FP16; `DeviceKvTypeNotice` warned that the copy stays FP16; `HandlerKvShare` refused sharing with a quantized host KV beside GPU
attention. The four sweep models are on disk. All four kernel modules rebuilt from source byte for byte with the local
nvcc 12.0 before any edit, so every PTX change below is measured against what was committed.

**What changed.**
- CUDA: `kv_format.cuh` (new), the one place a device KV row's address and an element's read or write are defined: rows of
  `F16`, `Q8_0` or `Q4_0` (the host codecs' 34- and 18-byte blocks, byte for byte); reads as templates on the format (values
  equal to the host decode, bit for bit, `__fmul_rn`); a decode lane's whole-block reads (`kv_dot_chunk`, `kv_axpy_chunk`:
  one scale per block, realigned 4-byte words, the scale applied once per block, every read through the read-only cache);
  the host encoders, one thread per block, every rounding spelled out. `gqa_attention.cuh` (new): the attention kernel body,
  moved out of `gqa_attention.cu` and given a `QUANT` flag; the FP16 entries instantiate it unchanged, their PTX equal to
  the committed one but for internal symbol names (diffed after normalising them). `gqa_attention_kv.cu` (new, its own
  module): the quantized entries, one switch per launch selecting decode and staging loops compiled for the format pair, a
  decode row's lanes holding the head block-major. `prefill_window.cu` gains `kv_encode_rows`, `batched_decode.cu`
  `kv_append_rows_fmt`; their 22 existing entries' PTX is unchanged.
- node: `DeviceKvFormat`, `DeviceKvFormats` (new; select once per handler from the cache types, a head width that is a
  multiple of 32, and the quantized module loading, which is asked only for a quantized type); `GqaAttentionKvKernel` (new);
  `DeviceKvCache` stores each side in its format (`formats()`, `rawK`/`rawV` for tests; `shiftInPlace` declines a quantized
  copy, the caller rewrites it from the shifted host rows); `GqaAttentionKernel.launch`/`launchBounded`,
  `BatchedDecodeKernels.kvAppendRows` and `PrefillWindowKernels.encodeKvRows` take the formats; the four launch sites pass
  the cache's; `CudaGqaAttention` refuses a launch mixing formats; `DeviceScratchBudget` counts the copy at its size;
  `LlamaTransformerHandler` and `GpuAttentionMirror` (Phi-3, Qwen3) choose the formats when they open GPU attention;
  `HandlerKvShare.supported` allows sharing whenever the device copy holds the host types' values
  (`DeviceKvFormats.holds`: re-encoding a decoded block gives back its values); `DeviceKvTypeNotice` now warns only when a
  quantized type leaves the copy FP16 (head width, or the module not loading), and logs the device formats otherwise.
- juno-player: `ConsoleMain` reads the head width from the GGUF header for the notice.

**Tests, written first** (they failed to compile on the missing `DeviceKvFormat`, `DeviceKvFormats`, `GqaAttentionKvKernel`,
`rawK`):

| Test | Runs | Result |
|---|---|---|
| `DeviceKvFormatTest` (new, no GPU) | 9 | pass: row sizes, the bytes-per-position bars as assertions on every sweep model's width, selection (an `f16` launch never asks for the module), `holds`, host packing equal to the host codecs' bytes, unpacking equal to their decode bit for bit, capacity arithmetic |
| `DeviceKvQuantParityTest` (new, GPU) | 10 | pass: the window write and the batch append store the host codec's bytes (rounding ties, a maximum appearing with both signs, zero blocks, a scale that rounds to zero, outliers), across a grow; host appends round-trip; allocation at the quantized size; the shift declined; attention against `GqaMath` over the host codec's values at decode width (separate caches, 8,192 keys) and prefill width (tiled), head widths 64, 96, 128, 256, a sliding window, five format pairs, within the FP16 kernel's bound (3e-3, 8e-3 long); the quantized kernel over FP16 rows: tiled path bit for bit the FP16 kernel's, decode within 1e-5 |
| `ResidentBatchDecodeTest` | 22 (was 15) | pass: seven quantized cases, every row of a decode batch bit for bit its own single decode, K bytes equal |
| `DeviceKvTypeNoticeTest`, `DeviceScratchBudgetTest` | 3, 27 | pass: the notice's new meaning; the budget at the formats' size |
| `DeviceKvQuantQualityLiveTest`, `DeviceKvDecodeTgLiveTest` (new, opt-in, GPU) | | the readings below |

**Shown able to fail**, each defect put in and taken out (`cmp` of source and PTX afterwards):

| Defect | Fails |
|---|---|
| none put in: the device encoder stored the FP16 scale's low byte as the scale's integer value (1.0 gave byte 1) | `deviceWindowWrite_isTheHostEncoding`, `batchAppend_isTheHostEncoding`: a real defect, found by the first run. Reproduced in a standalone CUDA program at `-O0` too (nvcc 12.0, sm_61: a byte store straight from the converted FP16 value); fixed with one aligned 16-bit store |
| the first reader's 4-wide `q8_0` read took its second pair from the first pair's bytes | decode, long-context and sliding-window cases |
| the first reader's 2-wide `q4_0` read took the high nibble for the low half | prefill and sliding-window cases |
| the final decode reader never realigns its words | decode, long-context and sliding-window cases |
| the final decode reader swaps `q4_0`'s nibble halves | decode, long-context and sliding-window cases |
| the batch append writes one row past the request's position | all 7 quantized `ResidentBatchDecodeTest` cases, no FP16 case |

**A slow first kernel, and the fix.** The first quantized kernel switched on the format inside every per-quad read; its
decode launch took 2.9x to 8.2x the FP16 one at 2,048 keys, and Mistral 7B decoded at 0.62x the FP16 rate (unpinned). Three
changes, each timed with `GqaAttentionDecodeBench` (gained quantized lanes): compile-time format pairs (1.3x to 3.5x),
block-major decode lanes (0.64x to 1.30x), reads through the read-only cache (0.58x to 1.02x). No spills at any step
(`ptxas -v`).

**Readings** ([`20261113T194534Z-device-kv-quant`](../perf-compare/20261113T194534Z-device-kv-quant/INDEX.md), unpinned,
the final kernel):

| Bar | Reading | |
|---|---|---|
| bytes per position `q8_0` <= 0.55x, `q4_0` <= 0.30x | 0.531x, 0.281x (`q8_0` K + `q4_0` V 0.406x, reported) | met |
| Mistral 7B 2,048-token sessions >= 1.8x at `q8_0` | 22 against 11 to 12: 1.83x to 2.00x | met |
| greedy agreement >= 99% `q8_0`, >= 97% `q8_0` K + `q4_0` V | TinyLlama 99.41% / 98.83%, Qwen2.5-3B 99.61% / 98.63%, Phi-3.5-mini 99.61% / 99.41%, Mistral 7B 100% / 99.61%; NLL 0.997x to 1.004x | met |
| decode tg at 2,048 tokens >= 0.95x, pinned | owner run ([`20261113T215326Z-device-kv-tg-gate`](../perf-compare/20261113T215326Z-device-kv-tg-gate/INDEX.md)): TinyLlama 1.085, Phi-3.5-mini 1.004, Mistral 7B 0.999, Qwen2.5-3B 0.918 (unpinned preview: 1.123, 0.9496, 0.982, 0.9497) | **missed on Qwen2.5-3B** |

**Tests run.** The unit suite on the final tree (`mvn -o test -pl api,tokenizer,lora,node,coordinator,sampler,kvcache,health,`
`registry,vision,metrics,juno-player,juno-kit -am -Dgroups='gpu | !gpu' -Dmaven.test.failure.ignore=true`, GPU groups on the GTX
1080, 1 h 20 min): `registry` 93, `lora` 116, `kvcache` 219, `health` 23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator`
417 (1 skipped) and its integer-kernel execution 1, `vision` 98, `metrics` 86, `juno-player` 158 (2 skipped) and `juno-master` 15
pass; `node` 1,617 run, 58 skipped, **4 failures**; `juno-kit` 141 run, **1 failure**. Each failing class re-run alone after:
  - `ResidentQkvPathTest.attentionAllocatesNothingPerCall`: the device-wide free figure read 7,366,049,792 bytes after 199 calls
    against 7,376,076,800 after the first (9.6 MiB lower; the desktop shares the card). The test's own figure,
    `path.deviceBytes()`, was unchanged, asserted just before. The test's cache is FP16, which this change routes to the
    unchanged FP16 kernel without loading the quantized module. Alone: 19 of 19 pass. Bound not loosened.
  - `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`: 97,386,496 bytes reported free with the
    device full against <= 67,108,864; alone 94,044,160. The driver holdback recorded in the entries above. Bound not loosened.
  - `DeviceUploadOrderingTest.theFirstGemvAfterAnUploadReadsTheWholeMatrix`, 2 of 4 (Q5_K 512x2048 round 1, row 236: -54.13 right
    after the upload against -13.15 once the device was idle; Q6_K round 1, row 187: 33.61 against 69.89). Alone, the first run
    failed the same 2 of 4 (Q5_K round 6, Q6_K round 1), the next six passed 4 of 4; HEAD's tree, built in the scratchpad
    (`git archive`), passed 4 of 4 seven times, four of them interleaved with this tree's runs. No class on its path
    (`DeviceQ4KMatrix`, `CudaMatVec`, `Q4KMmqKernel`) is in this change. The upload race raised with the owner in the previous
    entry, now seen once outside the suite as well.
  - `ProbeRunnerTest.aProbeRunsInItsOwnJvmAndLeavesItsReportAndLog` (`juno-kit`): `NOISY` instead of `DONE`, other processes at
    12% to 21% of the CPU, alone 19% to 22% (a browser, the editor, the desktop). `juno-kit` is not in this change.

**Live CLI.** The packaged tree (`mvn -o package -DskipTests`), `./juno local --verbose --gpu-attention on` on TinyLlama Q4_K_M, the
  prompt "Name three colors.": `--cache-type-k/-v q8_0` logs "GPU attention's KV copy stored as K q8_0, V q8_0, 544 bytes per
  position against 1024 for FP16" and answers "1. Red / 2. Blue / 3. Green"; `q8_0` K with `q4_0` V logs 416 bytes per position
  and answers "Certainly! Here are three colors: 1. Red ..."; `f16` loads only the FP16 attention module and prints no notice. No
  warning in any launch.

**Cross-surface rows for this item.** 1 CPU: N/A, the device copy only; a CPU launch selects nothing (tested). 2 CUDA: PASS
(parity, batch, live quality and capacity on the four sweep models, decode preview). 3 ROCm: N/A, no ROCm attention kernel; the
formats stay FP16 there and the cache types apply to the host rows in full. 4 Static: PASS (the live readings). 5 Continuous:
its running set decodes through the decode batch, whose quantized path is tested bit for bit against single decodes; not run
live. 6 Local: PASS. 7, 8 Clusters: nodes build their handlers from the same `JUNO_CACHE_TYPE_K/V`; not run on a cluster. 9 LoRA
training, 10 LoRA playback: N/A, attention on the CPU. 11 Vision: the text handler's GPU attention, unchanged in kind; not run
live. 12, 13 REST: N/A, no field. 14 CLI: PASS (the startup notice tested; launched live, above). 15 Embedding facade: as before,
through the properties the handlers read.

**Docs.** `docs/howto.md` (`--gpu-attention`, `--cache-type-k`, the context-shift GPU note), `docs/agent-arch.txt`
(`DeviceKvFormat`, `kv_format.cuh`, `GqaAttentionKvKernel`, `DeviceKvCache`, the notice, the sharing rule, the batch append),
`README.md`, `docs/performance.md` (2026-11-13 paragraph), `CHANGELOG.md` (Session 146), `docs/perf-compare/README.md`.

**Kit probes.** Item 7's half is owed: a GPU `kv-quality` probe and batched equal to single at each cache type.

**Out-of-tier changes (README rule 9):** none. Test-only: `KvQuantQualityLiveTest`'s helpers made package-visible for the GPU
reading; `GqaAttentionDecodeBench` gained quantized lanes.

**Exit criteria:** 5 of 15, unchanged. Criterion 7 stays open on the decode-rate bar, missed on Qwen2.5-3B.

**The pinned gate, owner run 2026-11-13T21:53Z** (`dist/device-kv-tg/run-gate.sh`, the tree `READY` named;
[`20261113T215326Z-device-kv-tg-gate`](../perf-compare/20261113T215326Z-device-kv-tg-gate/INDEX.md)). Per model f16 and `q8_0`
alternated three times, each JVM one warm-up and three measured repetitions:

| Model | FP16 tg (median of three JVMs) | `q8_0` tg | Ratio |
|---|---|---|---|
| TinyLlama 1.1B | 125.225 | 135.910 | 1.0853, pass |
| Qwen2.5-3B | 57.047 | 52.394 | **0.9184, fail** (per cycle 0.936, 0.918, 0.945) |
| Phi-3.5-mini | 39.779 | 39.920 | 1.0035, pass |
| Mistral 7B | 33.225 | 33.203 | 0.9993, pass |

- **The GPU clock was locked at 1,911 MHz and not held** (63 to 94 C; loaded medians 1,404 to 1,860 MHz by model), but each
  model's two sides ran at the same loaded clock (Qwen2.5-3B 1,809 against 1,809 MHz), so the ratios compare like with like.
  The gate script now locks at 1,300 MHz (below what the card held loaded at 94 C), cools the card to 65 C before every JVM
  and prints a CLOCK line per JVM; a lower core clock weighs the quantized side's added work more, not less.
- **Where Qwen2.5-3B's time goes** (traced with `nsys`, both orders, matched clocks; per token of 17.3 ms): no host-side cost
  (the GPU idles less with `q8_0`); the decode attention launch 1.08x (+250 us: 16 query heads over 2 KV heads at width 128,
  a 4 x 16 grid bound by latency; 0.955x in isolation with the KV warm in L2); the KV write 2.7x a launch (+215 us: one
  thread per 32-value block, 8 threads a launch); and the FFN gate and up GEMVs, unchanged code with identical launches, 5.6%
  slower in all 36 layers (+340 us). The last is not the clock and not the weights' placement: an FP16 JVM that loads the
  quantized module before its weight upload (a scratch build) reads 0.994x to 0.997x on them. Its cause is not found.
- Kit probes, item 6 and the rest of the tier wait on decision 2.

**Decisions for the owner.**
1. **Item 6 has no step.** Real FP16 host storage was added to the scope on 2026-10-04, after the steps were written, and no
   step names it. (a) Recommended: run it next, after the pinned gate and before step 7, since disk persistence fixes the
   stored dense format that item 6 changes (the argument the rotated-keys deferral makes). (b) Run it after step 7, with saved
   sessions naming the storage type.
2. **The decode bar on Qwen2.5-3B** (the case this decision anticipated: Qwen2.5-3B at 0.918x pinned while Mistral 7B
   passes). Its option (a), the host-side cost first, is done: there is none; the cost is on the GPU, above. Now: (a)
   recommended: work the quantized decode path and re-run the gate on all four models: the KV write as one launch for K and
   V with a warp per block (the same bytes as today's encoder, so the parity tests hold unchanged), the FFN GEMV interaction
   found with DRAM and L2 counters (`nvprof` reads them on this card), and the decode attention at few KV heads; (b) score the
   bar on Mistral 7B only, as the capacity bar beside it is, and record Qwen2.5-3B's 0.918x as the known cost of `q8_0` on a
   model with few KV heads.
3. **Rotated 4-bit keys** (deferred 2026-11-13, revisited on item 7's readings): `q8_0` keys with `q4_0` values leave 0.406x of
   the FP16 copy, and 22 of Mistral 7B's 2,048-token sessions fit at `q8_0` already. Recommended: keep them deferred; no
   target asks for the full `q4_0` size.

**Next for the executor:** decision 2: under (a), the quantized decode path's three costs, then the gate again; under (b),
item 7's kit probes. Then the step decision 1 settles.

### 2026-11-14: owner decision 2 (a); step 6, item 7: the quantized decode path reworked; the gate re-run owed

**Owner decision 2 (a)** (2026-11-14): work the quantized decode path's three costs on Qwen2.5-3B (the KV write, the decode
attention at few KV heads, the FFN GEMV interaction) and re-run the gate on all four models.

**Done** ([`20261115T045828Z-device-kv-decode-path`](../perf-compare/20261115T045828Z-device-kv-decode-path/INDEX.md)):

| Cost (Qwen2.5-3B, per token, traced) | Before | Change | After |
|---|---|---|---|
| KV write | 340 us, two launches a layer, one thread per block (FP16 cast 125 us) | `kv_encode_kv_rows`: K and V in one launch, one warp per 32-value block, the largest magnitude reduced across the warp with the scalar scan's comparisons; `kv_append_rows_fmt` (the decode batch) encodes the same way | 139 us |
| Decode attention | 3,256 us: one 128-thread block per head and row, 16 blocks (FP16 3,018 us) | `gqa_decode_kv_d64/_d128` (`GqaAttentionKvKernel` routes decode rows at head widths up to 128 to them): 256 threads, 64 slots of 4 lanes, the same per-slot loop at a 64-key stride, the same merge over 64 slots, 120 registers | 1,958 us |
| FFN gate and up GEMVs | +5.6% on FP16's | not changed: traced to where their input vector lands (below) | +4.4% to +4.7% |

GPU busy time per token: FP16 17,330 us; `q8_0` as gated 18,173 us; `q8_0` now 16,537 us. Tried and not kept: each key's K
and V loaded one iteration ahead in the decode loop (3,295 us against 3,256; the loop is short of warps, not waiting on its
loads).

**The FFN GEMVs: placement, not the quantized path.** Scratch builds of this tree, one variable at a time (`gemv-placement.txt`):
the quantized encoder and attention run beside an FP16 decode on a shadow cache, 0.990x to 0.993x; the quantized module
loaded before an FP16 weight upload, 0.994x to 0.997x; a `q8_0` JVM with its mirrors allocated at the FP16 size, 0.989x to
0.992x (as allocated, 1.039x to 1.042x). The smaller mirrors move the decode region's buffers allocated after them, and the
one that matters is the q8_1 input vector the gate and up GEMVs read (2,304 bytes, read whole by each of 11,008 blocks): its
offset from a 64 KiB boundary moves the two GEMVs by up to 6% (0x800 0.997x; 0x200 1.062x; 0x1000 1.028x; FP16's vector
sat at 0x4800, `q8_0`'s at 0xbe00), with no alignment rule that covers the fast offsets. L2-only weight loads in
`q4k_gemv` make FP16's GEMVs 3.7% faster and leave the placement cost. Decision 4.

**Kept properties.** The encoder writes the host codec's bytes; a decoded row depends on its own cache and length only, so a
row in a decode batch stays bit for bit its single decode; the FP16 modules' PTX is byte for byte the committed one; in
`prefill_window.ptx` and `batched_decode.ptx` only the two encoder entries changed (every other entry byte for byte HEAD's);
`gqa_attention_kv.ptx` gained the two decode entries and its three existing entries are unchanged. No spills (`ptxas -v`).

**Tests.** `DeviceKvQuantParityTest` (10), `ResidentBatchDecodeTest` (22) and `DeviceKvFormatTest` (9) pass on the final
kernels; the edge rows' tie case now puts each sign first in turn. Shown able to fail, each defect put in and taken out
(`cmp` of source and PTX afterwards):

| Defect | Fails |
|---|---|
| the warp encoder keeps the last index among equal magnitudes (`q4_0`) | `deviceWindowWrite_isTheHostEncoding`, `batchAppend_isTheHostEncoding` |
| the warp encoder takes the high nibble from the wrong lane (`q4_0`) | the same two |
| the wide decode's merge drops slot 63 | `decodeStreams_matchTheCpu`, `slidingWindow_matchesTheCpu`, `quantizedKernelOverFp16_isTheFp16Kernel` |

**Tests run.** The unit suite on the final tree (`mvn -o test -pl api,tokenizer,lora,node,coordinator,sampler,kvcache,health,`
`registry,vision,metrics,juno-player,juno-kit -am -Dgroups='gpu | !gpu' -Dmaven.test.failure.ignore=true`, GPU groups on the GTX
1080, 1 h 13 min): `registry` 93, `lora` 116, `kvcache` 219, `health` 23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator`
417 (1 skipped) and its integer-kernel execution 1, `vision` 98, `metrics` 86, `juno-player` 158 (2 skipped), `juno-master` 15 and
`juno-kit` 141 pass; `node` 1,617 run, 58 skipped, **3 failures**, in two classes, each re-run alone after:
  - `DeviceUploadOrderingTest.theFirstGemvAfterAnUploadReadsTheWholeMatrix`, 2 of 4 (Q5_K 512x2048 round 1, row 227: -25.40 right
    after the upload against -4.18 once the device was idle; Q6_K round 1, row 181: 21.84 against -6.26). Alone: 4 of 4 pass. No
    class on its path is in this change. The upload race recorded in the entries above.
  - `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`: 91,815,936 bytes reported free with the device
    full against <= 67,108,864; alone 91,750,400. The test and `DeviceScratchBudget.ALLOCATOR_HOLDBACK_BYTES` are not in the diff;
    the test only allocates until the driver refuses. The driver holdback recorded in the entries above. Bound not loosened.

**Readings, unpinned** (`decode-tg-unpinned.txt`, `decode-kernel-bench.txt`, `quality-recheck.txt`): decode tg `q8_0` over FP16,
Qwen2.5-3B 1.044 and 1.023, TinyLlama 1.225, Mistral 7B 1.022, Phi-3.5-mini 1.015; the decode launch alone at 2,048 keys
0.337x (TinyLlama), 0.568x (Qwen2.5-3B), 0.453x (Phi-3.5-mini), 0.504x (Mistral 7B) of the FP16 kernel (before: 0.576x,
0.955x, 0.604x, 0.825x); at 64 keys the 64-slot merge costs more than it saves on Phi-3.5-mini (1.19x of 7.9 us). Greedy
agreement and NLL unchanged on TinyLlama and Qwen2.5-3B (prefill windows run the unchanged tiled kernel, the encoder writes
the same bytes); the free-running reply, decoded through the wide kernel, shares 28 and 33 tokens with FP16's (1 and 1).

**Cross-surface rows**, against the 2026-11-13 entry: 2 CUDA PASS (parity, batch, live on the four sweep models); 5
Continuous: its running set decodes through the decode batch, whose append and attention take the new kernels, tested bit
for bit against single decodes, not run live; the rest unchanged.

**Docs.** `docs/agent-arch.txt` (`kv_format.cuh`'s warp encoder, `DeviceKvCache`'s window write, the batch append,
`GqaAttentionKvKernel`'s decode entries), `docs/howto.md` (`--cache-type-k`'s decode figures), `docs/performance.md`,
`CHANGELOG.md` (Session 146), `docs/perf-compare/README.md`.

**Out-of-tier changes (README rule 9):** none. The GEMV and placement experiments ran on scratch copies of the tree.

**Exit criteria:** 5 of 15, unchanged. Criterion 7 stays open until the owner's re-run.

**Owed to the owner: the gate, re-run.** In a terminal left open: `bash dist/device-kv-tg/wait-and-run.sh`. The script
changed after the first run: it locks the graphics clock at 1,300 MHz (the first run's 1,911 MHz lock did not hold: 63 to
94 C, loaded medians 1,404 to 1,860 MHz), cools the card to 65 C before every JVM and prints a CLOCK line per JVM (loaded
samples, how many fell below the lock, the hottest sample, the throttle reasons). Per model f16 and `q8_0` alternated three
times; `runs/<UTC>/results.txt` holds one RESULT line per model against 0.95. About an hour with the cool-downs.

**Decisions for the owner.**
1. **Item 6 has no step** (2026-11-13 entry, decision 1): open; the recommendation stands (after this gate, before step 7).
3. **Rotated 4-bit keys:** still deferred.
4. **The FFN GEMVs' placement cost** (up to 6% on the gate and up GEMVs, for either format, from where the q8_1 input
   vector lands). (a) Recommended: a GEMV step outside item 7 that reads the input vector once for two rows per block
   (halving the reads the placement penalizes, and using the two warps a 2,048-wide row leaves idle), measured against
   today's kernel at several placements, with the GEMV gates of its own; item 7 is scored without it. (b) Place the vector
   at an offset measured fast on this card (works here, says nothing about another card). (c) Leave it recorded here only.
5. **The FP16 decode kernel and the wide block.** FP16 decode launches have the same one-block-per-head shortage (Qwen2.5-3B:
   16 blocks), and part of `q8_0`'s margin is the wider block. (a) Recommended: give the FP16 decode the same block as a
   follow-up after this gate, with its own A/B against today's FP16 kernel (it changes the baseline every FP16 launch
   reads, so not inside item 7). (b) Leave FP16 as it is.

**Next for the executor:** after the owner's re-run, item 7's kit probes; then the step decision 1 settles.

### 2026-11-15: step 6, item 7: the decode gate re-run passes on every sweep model; criterion 7 met

**Owner run** ([`20261115T200324Z-device-kv-tg-gate-rerun`](../perf-compare/20261115T200324Z-device-kv-tg-gate-rerun/INDEX.md),
`dist/device-kv-tg/run-gate.sh` through `wait-and-run.sh`, 2026-11-15T20:03Z; the tree READY named, `head 79fa87b0`,
`diff e458312807298dc0`, matched before and after): GPU decode tg at a 2,048-token context, `q8_0` keys and values over the
FP16 device copy, A B A B A B per model, median over median:

| Model | FP16 tg (median of 3 JVMs) | `q8_0` tg (median of 3 JVMs) | Ratio | First run |
|---|---|---|---|---|
| TinyLlama 1.1B | 124.074 | 153.177 | **1.2346** | 1.0853 |
| Qwen2.5-3B | 56.759 | 58.483 | **1.0304** | 0.9184 |
| Phi-3.5-mini | 43.369 | 45.810 | **1.0563** | 1.0035 |
| Mistral 7B | 35.011 | 36.576 | **1.0447** | 0.9993 |

Bar >= 0.95x: met on all four. Every `q8_0` JVM is faster than every FP16 JVM on every model (slowest `q8_0` over fastest
FP16: 1.212, 1.028, 1.053, 1.044). Capacity lines: `q8_0` holds 1.87x to 2.00x the FP16 sessions.

- **The clock.** CPU pinned (performance governor, turbo off). `nvidia-smi -lgc 1300,1300` returned success and was not
  applied: loaded samples read 1,480 to 1,885 MHz. The README's rule for consumer Pascal parts applies (the GPU clock is
  recorded, not fixed), and per model both sides ran at the same median loaded clock (TinyLlama 1,860, Qwen2.5-3B 1,835,
  Phi-3.5-mini 1,607 under software thermal slowdown, Mistral 7B 1,809 MHz) and memory clock (4,513 MHz, P2). The first
  run's 1,911 MHz lock most likely never applied either; its INDEX carries the correction. The CLOCK line counted only
  samples below the requested clock; `run-gate.sh` (git-ignored) now also counts samples above it and prints "lock NOT
  applied" when most loaded samples are, and its pin line says the lock is requested, not held (the previous script kept as
  `run-gate.second-run.sh`).
- **Qwen2.5-3B's A3** read 51.054 against 56.759 and 56.799; its prefill was slower by the same share, at the same clock,
  temperature and power, with no warning in its log. Cause not found (the desktop shares the card and the host). The median
  rule leaves the ratio unchanged; the other two FP16 JVMs give 1.028 to 1.031.
- **Prefill, not a bar.** From the JVM logs' per-layer prefill records, a 2,048-token prefill with `q8_0` device KV takes
  1.31x (TinyLlama), 1.21x (Qwen2.5-3B) and 1.31x (Mistral 7B) the FP16 time; the first run's logs read the same (1.32x,
  1.24x, 1.29x), so it comes from item 7's quantized prefill path (the tiled attention kernel over `q8_0` K and V in
  128-row windows), not from the decode rework. Phi-3.5-mini's handler writes no per-layer prefill records. Decision 6.

**Exit criteria:** 6 of 15. Criterion 7 ticked: bytes per position, session capacity and greedy agreement met on
2026-11-13, decode tg met pinned today, every handler with a GPU attention path carries the quantized device KV or keeps
FP16 with a startup warning.

**Tests run.** None this entry: no code changed since the 2026-11-14 entry's suite (the owner's run measured that tree).
`check-plan-thresholds.sh` after the record: passes.

**Out-of-tier changes (README rule 9):** none.

**Owed to the owner:** nothing to run for this entry.

**Decisions for the owner.**
1. **Item 6 has no step** (2026-11-13 entry, decision 1): open. Recommended (a): run it after item 7's kit probes and before
   step 7, so disk persistence saves the final dense format.
3. **Rotated 4-bit keys:** still deferred; recommended to keep them deferred (no target asks for the full `q4_0` size; after
   step 7 they would need a type of their own).
4. **The FFN GEMVs' placement cost** (2026-11-14 entry): open; recommended (a), a GEMV step outside item 7, now that the
   gate is scored.
5. **The FP16 decode kernel and the wide block** (2026-11-14 entry): open; recommended (a), a follow-up with its own A/B
   against today's FP16 kernel, now that the gate is scored.
6. **Prefill with a quantized device KV** (1.21x to 1.31x the FP16 time at 2,048 tokens; no threshold covers it). (a)
   Recommended: a follow-up inside item 7, after its kit probes, on the tiled attention kernel's quantized read at prefill
   width, measured per launch with `nsys` as the decode was and scored with an A/B of the prefill time (a bar of 1.10x the
   FP16 prefill proposed), its parity and quality tests unchanged. (b) Record the cost in the user docs beside
   `--cache-type-k` and leave it. (c) Add a prefill bar to item 7's thresholds now and reopen criterion 7 until it is met.

*(Decided 2026-11-15 by the owner: 1 (a), 3 (a), 4 (a), 5 (a), 6 (a).)*

**Owner decisions 2026-11-15.**
- **1 (a):** item 6 (exact host KV growth and real FP16 host storage) runs before step 7, so disk persistence saves the
  final dense format; it is step 6's last item.
- **3 (a):** rotated 4-bit keys stay deferred; after step 7 they would need a cache type of their own.
- **4 (a):** a GEMV change outside item 7 (README rule 9, recorded under Out-of-tier changes when it lands): the gate and
  up projections read the q8_1 input vector once for two rows per block, measured against today's kernel at several
  placements of the vector, with GEMV gates of its own (FP16 and `q8_0` device KV both).
- **5 (a):** the FP16 decode attention takes the wide decode block as a follow-up outside item 7 (rule 9), with its own
  A/B against today's FP16 kernel and the hot-path gate, since it changes the default path.
- **6 (a):** inside item 7, after its kit probes: the tiled attention kernel's quantized read at prefill width, traced per
  launch, scored by an A/B of the 2,048-token prefill time with `q8_0` device KV against FP16 (bar **<= 1.10x**, sweep
  models, pinned), parity and quality tests unchanged. Criterion 7 stays ticked: its thresholds are met; the prefill bar
  is item 7's own follow-up.

Order of work (the executor's, within the decisions): item 7's kit probes; decision 6 (item 7's prefill); decision 5;
decision 4; item 6 (decision 1); step 7.

**Next for the executor:** item 7's kit probes (mean NLL against FP16 KV per sweep model, and batched equal to single on the
device, at each `--cache-type-k`/`--cache-type-v`). *(Done 2026-11-15; the next entry.)*

### 2026-11-15: step 6, item 7: the device KV's kit probes

**Done** (README execution rule 10; [`20261115T232030Z-kit-device-kv-probes`](../perf-compare/20261115T232030Z-kit-device-kv-probes/INDEX.md)):

- **`gpu.kv-quality.<model>`** (new, `GpuKvQualityProbe`; TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B; suite `gpu`):
  `QualityRunMain` three times with the handler on CUDA (`-Djuno.kit.quality.gpu=true`, new): `f16`, `q8_0`, `q8_0` keys with
  `q4_0` values, teacher-forced over the 2,048-token quality text. Per type two checks: NLL at most 1.02x the `f16` run's,
  and the handler's device KV notice (captured from the child's log) naming the selected formats; the FP16 fallback warning
  or no notice fails it. The argmax agreement along the text's first 512 positions is recorded, not checked.
- **`gpu.consistency.tinyllama`**: its decode batch of eight runs at `f16`, `q8_0` and `q8_0`/`q4_0`
  (`ConsistencyProbe.DEVICE_KV_TYPES`), each batched reply equal to its single-request reply at its type.
- `coverage.tsv`: `feature:device-kv-quant`. The catalog carries the probes' measured costs (26, 46, 148, 60 s; the
  consistency probe 15 to 31 s with its two extra servers) and device peaks (992, 2,488, 3,384, 4,796 MiB).

**First run, reference host, unpinned:**

| Model | `q8_0` NLL ratio | `q8_0`/`q4_0` NLL ratio | Device copy announced | Agreement along the text (reading) |
|---|---|---|---|---|
| TinyLlama 1.1B | 1.00047 | 1.00077 | K q8_0, V q8_0 / K q8_0, V q4_0 | 98.63%, 96.48% |
| Qwen2.5-3B | 1.00112 | 1.00398 | the same | 99.41%, 99.02% |
| Phi-3.5-mini | 1.00099 | 1.00358 | the same | 99.80%, 99.22% |
| Mistral 7B | 0.99914 | 0.99735 | the same | 99.22%, 99.02% |

16 of 16 `gpu.kv-quality` checks pass; `gpu.consistency.tinyllama` 44 of 44. The agreement column reads the human text,
where the CPU probe reads the same; item 7's agreement bar is the greedy continuation's (`DeviceKvQuantQualityLiveTest`,
met 2026-11-13), and TinyLlama's argmax flips more often on the text.

**Tests** (the unit test and self-test written with the probe, before its first live run):

| Test | Result | What it shows |
|---|---|---|
| `GpuKvQualityProbeTest` (juno-kit, new) | 5 of 5 pass | the stored check passes on the formats' INFO line, fails on another type's line, on the handler's own FP16 warning text (`DeviceKvTypeNotice.consoleNotice`) and on no notice; the notice is picked from the run's messages |
| `ProbeSelfTests#gpuKvQualityProbeFailsTheStoredCheckWhereTheHandlerKeepsAnFp16DeviceCopy` (new, CUDA) | pass, ran on this host | on the synthetic model (head width 16) the real handler keeps FP16 and warns: the stored check fails with that warning, the NLL check passes; the check is shown able to fail |
| `ProbeSelfTests#consistencyProbeBatchOfEightFillsABatchFromThePrompts` (changed) | pass | the batch of eight's arguments carry `--cache-type-k/-v` per type, three types |
| juno-kit suite (`mvn -o test -pl juno-kit -am -Dtest='cab.ml.juno.kit.**'`) | 147 run, 1 failure | `ProbeRunnerTest.aProbeRunsInItsOwnJvmAndLeavesItsReportAndLog`: `NOISY`, other processes at 6% to 16% of the CPU (a browser, the editor, Steam); alone the same; the environmental failure the entries above record, not in this change |
| `ProbeCatalogTest`, `CoverageTest`, `PlanTest`, `SelectionTest`, `EstimatorTest` after the measured costs and peaks | pass | timeouts at least 3x the costs; every catalog probe covered; the peaks name GPU probes |

**Out-of-tier changes (README rule 9):** none (kit code only).

**Exit criteria:** 6 of 15, unchanged (the kit probes are rule 10's, not a criterion).

**Owed to the owner:** nothing to run.

**Next for the executor:** decision 6 (a): item 7's prefill with a quantized device KV, held to <= 1.10x the FP16 prefill
time at 2,048 tokens on the sweep models (pinned, the owner's run), parity and quality tests unchanged.

### 2026-11-15, later: decision 6 (a): item 7's prefill traced; the attention's quantized tile staging reworked (now 0.90x to 0.95x the FP16 kernel); the host KV encode found, raised with the owner

**Plan against code, verified with HEAD `a4a91182`, tree clean but for the untracked `.github/` (not this session's).**
`check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep, build or test was running. Every claim this step
depends on held: prefill windows of more than 8 rows run in the device region on the Llama, Phi-3 and Qwen3 handlers
(`PrefillWindowRegion.MAX_HOST_WINDOW`); a window's attention is `gqa_attention_kv_d64/_d128/_d256` at `rowsPerBlock` 32 over
`q8_0`/`q4_0` K and V, staged into FP32 shared tiles by `gqa_stage_kv` (one `kv_load2_t` per pair: a scale read and two bytes);
the window's device KV write is `kv_encode_kv_rows`. The four sweep models are on disk. Both attention modules rebuilt from
source byte for byte with the local nvcc 12.0 before any edit.

**The trace** ([`20261116T004641Z-device-kv-prefill-staging`](../perf-compare/20261116T004641Z-device-kv-prefill-staging/INDEX.md),
section 2; `nvprof`, since this host's `nsys` cannot import its own traces). A new live test,
`DeviceKvPrefillTimeLiveTest`, times a 2,048-token prompt through `forwardBatch`, the serving prefill path, one model and type
per JVM, in one window (the local GPU surface's default chunk when the window fits) and in 128-row windows. One window, under
the profiler:

| Model | Prefill FP16 -> `q8_0` | Attention launch a layer, FP16 -> `q8_0` | Off the device |
|---|---|---|---|
| TinyLlama | 1,002 -> 1,339 ms (1.34x) | 14.17 -> 21.08 ms (1.49x), +152 ms a prefill | +185 ms |
| Qwen2.5-3B | 2,267 -> 2,766 ms (1.22x) | 19.76 -> 25.35 ms (1.28x), +201 ms | +310 ms |
| Phi-3.5-mini | 3,419 -> 6,985 ms (**2.04x**) | 36.13 -> 45.70 ms (1.26x), +306 ms | +3,274 ms |
| Mistral 7B | 4,768 -> 6,234 ms (1.31x) | 39.87 -> 51.01 ms (1.28x), +356 ms | +1,101 ms |

**Drift against decision 6's premise.** The 2026-11-15 entry put the prefill cost in "item 7's quantized prefill path (the
tiled attention kernel over `q8_0` K and V in 128-row windows)". The kernel is 9% to 45% of it. The rest is off the device,
about 8 ns per host KV value on every model: after each layer the region downloads the window's K and V as FP32
(`ResidentActivation.materializeToKv`, after `chain.sync()`) and `DenseKvTensor.writeRows` encodes every row to `q8_0` on the
CPU, one thread (`Q8_0KvCodec.encodeBlock`), while the device waits. Read alone (`host-kv-encode.txt`): 9.0 to 9.3 ns a value
for `q8_0` against 0.27 to 0.58 ns for the `f16` host type (FP32-backed, a copy); Phi-3.5-mini (KV width 3,072, no grouping)
pays 57 ms per tensor per layer, about 3.6 s a prefill. The host encode is the host type's cost, not the device copy's, and
predates item 7; it is visible on the GPU because nothing overlaps it. The kernel work below cannot remove it, so the 1.10x
bar is not reachable within decision 6's scope. Decision 7.

**What changed (the kernel half of decision 6 (a)).**
- `kv_format.cuh`: `kv_load8_t`, eight values of one block as one scale and two realigned quant words (`kv_words`), the values
  `kv_load2_t` gives, bit for bit (the same scale times the same integer, `__fmul_rn`).
- `gqa_attention.cuh`: `gqa_stage_kv8` stages a tile eight values of a key per thread (index stepped without a division);
  every format pair with a quantized side takes it, `(F16, F16)` keeps `gqa_stage_kv`'s pairs. `gqa_attention_tiled` gains
  `QDECODE` (default `QUANT`): `gqa_attention_kv_d64/_d128` are compiled without the quantized decode loop, whose rows
  `GqaAttentionKvKernel` already sends to the wide decode entries; a row of one would still be served, by the 32-key tiles.
  `_d256` keeps it (16-key tiles). Registers: `_d64` 196 -> 127 (the FP16 entry's count: four blocks per multiprocessor
  instead of two), `_d128` 226 -> 217, `_d256` 245 -> 247; no spills.
- `GqaAttentionKvKernel.launch` refuses a quantized format at a head width that is not a multiple of 32 (`DeviceKvFormat`'s
  rule, which the handlers already apply when they select formats); the eight-value read needs it, as the decode lanes do.
- `gqa_attention.ptx` (FP16): code unchanged; it differs from the committed PTX only in the mangled names of its shared arrays
  (the template gained an argument), byte for byte equal once they are normalised.

**Tests** (written before the kernel change):

| Test | Result | What it shows |
|---|---|---|
| `DeviceKvQuantParityTest#prefillTileOverExactValues_isTheFp16Kernel` (new, GPU) | passed on HEAD's kernel (a regression test, not seen failing there) and on the new one | blocks whose values FP16 stores exactly (power-of-two scales, integer steps the encoders give back): a prefill window over the quantized cache equals the FP16 kernel's bit for bit, five format pairs, head widths 64, 96, 128, 256 (all three entries), windows of 40, 33 (sliding, 64) and 5 rows |
| `DeviceKvQuantParityTest#quantizedLaunch_refusesAHeadWidthNotAMultipleOf32` (new, GPU) | failed first ("Expecting code to raise a throwable"), passes with the refusal | `rowsPerBlock` 1 and 4 at head width 36 |
| `DeviceKvQuantParityTest` (12), `ResidentBatchDecodeTest` (22), `GqaAttentionKernelParityTest` (2), `GqaAttentionTiledTest` (9) | pass on the new kernel | the existing parity bounds, decode batches bit for bit their single decodes |
| `DeviceKvQuantQualityLiveTest`, TinyLlama and Qwen2.5-3B, `f16`, `q8_0`, `q8_0`/`q4_0` | every printed figure equals the readings on the previous kernels | outputs unchanged (`quality-recheck.txt`) |

Shown able to fail, each defect put in and taken out (`cmp` of source and PTX afterwards):

| Defect | Fails |
|---|---|
| `kv_load8_t` takes `q4_0`'s other nibble half | `prefillTileOverExactValues_isTheFp16Kernel`, `prefillWindow_matchesTheCpu`, `slidingWindow_matchesTheCpu` |
| `gqa_stage_kv8` drops the key step when the value index wraps | `prefillTileOverExactValues_isTheFp16Kernel` and `prefillWindow_matchesTheCpu`, on Phi-3.5-mini's head width 96 only, the one shape where the index wraps |

**Readings, unpinned** (section 3 of the published entry). Attention launch a layer at a 2,048-row window, `q8_0` over FP16:
TinyLlama 0.95x, Qwen2.5-3B 0.90x, Phi-3.5-mini 0.92x, Mistral 7B 0.91x (before 1.49x, 1.28x, 1.26x, 1.28x). Prefill time
`q8_0` over FP16, median of five, one JVM per side, same session:

| Model | One 2,048-row window | 128-row windows |
|---|---|---|
| TinyLlama | **1.094** (before 1.259) | **1.046** (before 1.210) |
| Qwen2.5-3B | 1.150 | 1.076 |
| Phi-3.5-mini | 1.835 | 1.650 |
| Mistral 7B | 1.172 | 1.103 |

Against the 1.10x bar: TinyLlama under it in both widths; the others over it by roughly what the host encode costs.

**Cross-surface rows for this change.** 1 CPU: N/A (device kernel). 2 CUDA: PASS (parity, quality unchanged, the four sweep
models timed). 3 ROCm: N/A, no ROCm attention kernel. 4 Static: PASS (the live prefill readings). 5 Continuous: its prefill
chunks run the same region and kernel; the parity tests cover the kernel, not run live. 6 Local: PASS. 7, 8 Clusters: nodes
run the same handlers; not run on a cluster. 9 LoRA training, 10 LoRA playback: N/A, attention on the CPU. 11 Vision: the
text handler's GPU attention, unchanged in kind; not run live. 12, 13 REST: N/A. 14 CLI: N/A, no flag changed. 15 Embedding
facade: unchanged.

**Docs.** `docs/howto.md` (`--cache-type-k`'s prefill figures and their cause), `docs/performance.md` (the 2026-11-13
paragraph's prefill sentence), `docs/agent-arch.txt` (`kv_load8_t`, the eight-value staging, the entries without the decode
loop, the launcher's refusal), `CHANGELOG.md` (Session 147), `docs/perf-compare/README.md`.

**Out-of-tier changes (README rule 9):** none. The kernel change is item 7's own follow-up (decision 6 (a)); the FP16 module
was regenerated with unchanged code.

**Tests run.** `DeviceKvQuantParityTest` 12 of 12 (GPU group) before the suite. The unit suite (`mvn -o test -pl
api,tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player,juno-kit -am -Dgroups='gpu | !gpu'
-Dmaven.test.failure.ignore=true`, GPU groups on the GTX 1080, 1 h 14 min): `registry` 93, `lora` 116, `kvcache` 219, `health`
23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator` 417 (1 skipped) and its integer-kernel execution 1, `vision` 98,
`metrics` 86, `juno-player` 158 (2 skipped), `juno-master` 15 and `juno-kit` 147 pass; `node` 1,620 run (the two new parity
tests; the new prefill-time live test skipped without its property), 59 skipped, **3 failures**, in two classes, each re-run
alone after:
  - `DeviceUploadOrderingTest.theFirstGemvAfterAnUploadReadsTheWholeMatrix`, 2 of 4 (Q5_K 512x2048 round 1, row 231: -38.88 right
    after the upload against 44.98 once the device was idle; Q6_K round 1, row 177: -5.19 against -13.95). Alone: 4 of 4 pass,
    three runs. No class on its path is in this change. The upload race recorded in the entries above.
  - `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`: 88,735,744 bytes reported free with the device
    full against <= 67,108,864; alone 84,148,224. The test and `DeviceScratchBudget.ALLOCATOR_HOLDBACK_BYTES` are not in the diff;
    the test only allocates until the driver refuses. The driver holdback recorded in the entries above. Bound not loosened.

**Exit criteria:** 6 of 15, unchanged. Criterion 7 stays ticked (decision 6 (a)); the prefill bar stays open.

**Owed to the owner:** no run yet. The pinned prefill A/B is prepared once decision 7 is settled: today it would fail on
three of four models for a cause outside the kernel.

**Decisions for the owner.**
7. **The host KV encode on the prefill path** (about 9 ns a value on one thread; with the quantized attention now faster than
   FP16's, it is all of what is left of `q8_0`'s prefill cost over FP16; 3.6 s a prefill on Phi-3.5-mini). (a) Recommended: the host KV takes the device's encoded rows.
   When a whole layer runs in the region, the device copy already holds the window's rows in the host codec's bytes
   (`kv_encode_kv_rows`, held equal to `Q8_0KvCodec`/`Q4_0KvCodec` byte for byte by `DeviceKvQuantParityTest`), so the region
   downloads those bytes into the host tensor's packed storage (a packed-row write on `SessionKvTensor`, dense and paged)
   instead of FP32 rows to encode: no CPU encode, and 34 (`q8_0`) or 18 (`q4_0`) bytes per 32 values crossing the bus instead
   of 128. Projected, not measured: `q8_0` prefill at or under the FP16 time on all four models, since the quantized attention
   is already faster than FP16's. Item 6 can then do the same for real FP16 host storage (the device's FP16 rows). Tests: the
   host tensor's bytes after a region prefill equal the host codec's, every handler family, both schedules; the pinned A/B
   after. (b) Encode on the CPU worker pool, rows across threads (CPU prefill gains too; the FP32 download stays): about
   6x less host time at 6 threads, which leaves Phi-3.5-mini near 1.15x by the readings above, likely still over the bar.
   (c) Score decision 6's bar on the attention launch (met, 0.90x to 0.95x), keep the prefill ratio as a reading, and leave
   the host encode documented beside `--cache-type-k` (the docs already say it).
8. **Which prefill the bar scores** (with 7 (a) or (b)). (a) Recommended: both widths, one 2,048-row window (the local GPU
   default when the prompt fits) and 128-row windows (a typical `--prefill-batch`), each `q8_0` over FP16 <= 1.10x, sweep
   models, pinned, A B A B A B per model as the decode gate ran. (b) The one-window prefill only.

**Next for the executor:** decision 7; then the prefill gate (decision 8) for the owner; then decisions 5 and 4; item 6;
step 7.

**Owner decisions 2026-11-15, later.**
- **7 (a):** a whole-layer prefill window writes the host KV from the device copy's encoded rows: where a side's device
  format is that side's host type (`q8_0` or `q4_0`), the region downloads the window's stored blocks into the host
  tensor's packed rows (dense and paged) instead of FP32 rows the CPU encodes; an `f16` side, and a side the device holds
  as FP16 (a head width not a multiple of 32), keep the FP32 download. Tests: the host bytes after a region window equal
  the host codec's encoding of the window's FP32 rows, every format pair, dense and paged.
- **8 (a):** the prefill bar scores both widths, one 2,048-row window and 128-row windows, each `q8_0` over FP16
  **<= 1.10x** on every sweep model, pinned, A B A B A B per model (the decode gate's protocol).

**Next for the executor:** decision 7 (a); the prefill kit probe (owed with it); then the pinned prefill gate (decision 8
(a)) for the owner; then decisions 5 and 4; item 6; step 7.

### 2026-11-16: decision 7 (a): the host KV takes the device copy's encoded rows; `q8_0` prefill under the FP16 time on every sweep model (unpinned); the pinned prefill gate owed

**Plan against code, verified with HEAD `a4a91182` plus the previous entry's uncommitted kernel change.**
`check-plan-thresholds.sh` passed first (26 tier files). No gate, sweep, build or test was running. Every claim decision 7 (a)
rests on held: a window of more than 8 rows runs whole layers in `PrefillWindowRegion.Window.runLayer`, which writes the
window's K and V into the device copy (`DeviceKvCache.writeWindowOnDevice`, `kv_encode_kv_rows`) before its attention, then
downloads the same K and V as FP32 (`ResidentActivation.materializeToKv`) for `writeRows` to encode on the CPU; the device
encoder is the host codecs' arithmetic byte for byte, rounding ties included
(`DeviceKvQuantParityTest#deviceWindowWrite_isTheHostEncoding`). No drift.

**What changed.**
- `SessionKvTensor.writeEncodedRows(startPos, rows, src, srcOffset)`: rows already in the tensor type's blocks, written as
  they are. `DenseKvTensor`: one copy into the packed rows, a gap before `startPos` exposed as zeros as `writeRows` does.
  `PagedKvTensor` -> `KvPageTable.writeEncodedRows`: one copy per page run; a page boundary at the end appends a new page, an
  existing page is taken through `writablePage` (a page shared with another session is copied first); `KvBlockPool` copies
  under its lock and refuses a shared block. An `f16` tensor refuses (its rows are floats).
- `DeviceKvCache.encodedRowsK/V(hostType, startPos, count)`: the device rows as they are stored, where that side's device
  format is quantized and is the host type; null otherwise (an `f16` side, or a side kept as FP16 at a head width that is not
  a multiple of 32).
- `ResidentActivation.materializeToKv` queues each side's download, the stored rows (JFR `juno.DeviceStaging` site
  `materialize(<name> encoded)`, `count x rowBytes`: 34 bytes per 32 values at `q8_0`, 18 at `q4_0`) or the FP32 rows (128
  bytes per 32 values) as before, syncs once, then writes with `writeEncodedRows` or `writeRows`. It refuses stored rows for
  an `f16` host tensor and a download larger than the staging buffer.
- `PrefillWindowRegion.Window.runLayer` passes the two slices. Per window and layer: two segment slices, as the window's device
  write already takes; nothing per row.

**Tests:**

| Test | Result | What it shows |
|---|---|---|
| `PrefillRegionEncodedKvTest` (new, GPU; dense and paged) | failed first on HEAD's region (the download counts; the stored bytes already equal), passes after | a synthetic region (8 heads, 4 KV heads, head width 64) prefilled in two windows (40 rows, then 24): the host tensor's bytes equal those the FP32 rows written through `writeRows` give, seven device and host format pairs (`q8_0`/`q8_0`, `q8_0`/`q4_0`, `q4_0`/`q4_0`, `f16`/`q8_0`, `q4_0`/`f16`, an FP16 device copy over `q8_0`/`q4_0` hosts, `f16`/`f16`); a quantized side downloads stored rows only (two copies, 64 x row bytes), every other side FP32 rows only |
| `KvWriteEncodedRowsTest` (new, 9) | passes; written with the method, shown able to fail below | dense and paged, `q8_0` and `q4_0`, starting inside a page, at 0 and on a page boundary; a reused dense tensor's gap reads zeros; a page shared with another session is copied before the write and the other session's rows stay; `f16` refused |
| `GpuKvPrefillProbeTest` (new, 5), `ProbeCatalogTest` 5, `CoverageTest` 7, `GpuKvQualityProbeTest` 5 | pass | the kit probe below: its parse, readings and ratio, a missing width, and the checks that refuse to read another path's time |
| `ProbeSelfTests#gpuKvPrefillProbeFailsWhereTheDeviceCopyIsNotInTheSelectedFormats` (new, CUDA) | passes | on the synthetic model (head width 16, an FP16 device copy) the probe fails with the handler's warning instead of reading |
| 17 neighbouring `node` classes, GPU group (139 tests); `kvcache` 228 of 228 | pass | |
| `DeviceKvQuantQualityLiveTest`, TinyLlama, Qwen2.5-3B, Phi-3.5-mini, `f16`, `q8_0`, `q8_0`/`q4_0` | every printed figure equals the previous readings | outputs unchanged (`quality-recheck.txt`) |

Shown able to fail, each defect put in and taken out (`cmp` of the source afterwards):

| Defect | Fails |
|---|---|
| `DenseKvTensor.writeEncodedRows` without the gap's zeros | `KvWriteEncodedRowsTest.denseReused`, `q8_0` and `q4_0` |
| `KvPageTable.writeEncodedRows` writes the page in place (`blockIds.get`, not `writablePage`) | `pagedSharedPages`, both types (the pool refuses: "block 2 is shared by 2 holders and cannot be written") |
| a page run's length ignores the slot it starts at | `paged` and `pagedSharedPages`, both types (the pool refuses: "slots [5, 21) outside a page of 16") |
| the region passes K's stored rows for the V side | `PrefillRegionEncodedKvTest.dense` and `.paged` (the V bytes) |
| `DeviceKvCache.encodedRows` slices from row 0, not from the window's start | `PrefillRegionEncodedKvTest.dense` and `.paged` (the second window's K bytes) |

**Readings, unpinned**
([`20261116T032708Z-device-kv-prefill-encoded-rows`](../perf-compare/20261116T032708Z-device-kv-prefill-encoded-rows/INDEX.md)).
Prefill time of the 2,048-token quality text through `forwardBatch`, median of five, one JVM per model and type, same
session, `q8_0` over FP16:

| Model | One 2,048-row window | 128-row windows | `q8_0`/`q4_0`, one window / 128 |
|---|---|---|---|
| TinyLlama | **0.938** (before 1.094) | **0.926** (before 1.046) | 0.956 / 0.927 |
| Qwen2.5-3B | **0.989** (1.150) | **0.971** (1.076) | 1.018 / 1.002 |
| Phi-3.5-mini | **0.895** (1.835) | **0.895** (1.650) | 0.891 / 0.895 |
| Mistral 7B | **0.938** (1.172) | **0.948** (1.103) | 0.933 / 0.949 |

Under the 1.10x bar on every sweep model at both widths, unpinned; the owner's pinned gate scores it.

**The kit probe** ([`20261116T150000Z-kit-device-kv-prefill`](../perf-compare/20261116T150000Z-kit-device-kv-prefill/INDEX.md)):
`gpu.kv-prefill.<model>` (new; `GpuKvPrefillProbe`, child `KvPrefillRunMain`; `coverage.tsv` row
`feature:device-kv-prefill`) prefills the 2,048-token text with the handler on CUDA at `f16` then `q8_0`, both widths, and
reads the ratio of medians. It refuses to read unless both runs opened GPU attention and the device prefill region and the
`q8_0` handler logged its copy in the selected formats. First run, reference host, every probe NOISY (the desktop used 5% to
12% of the CPU, three attempts each): all 24 ratios 0.874x to 1.031x; the device copy logged as `q8_0` on every run. The
catalog takes its costs (28, 52, 73, 100 s at the reference speed) and device peaks (1,172 to 5,266 MiB) from this run.

**Cross-surface rows for this change.** 1 CPU: N/A, the CPU path writes rows as before (`writeRows`). 2 CUDA: PASS (the
region test, the four sweep models timed, quality unchanged on three). 3 ROCm: N/A, no ROCm device KV attention, so no
region download changes. 4 Static: PASS (dense tensor, the live readings). 5 Continuous: the paged write is covered by
`PrefillRegionEncodedKvTest.paged` and `KvWriteEncodedRowsTest`; not run live (no live test runs the continuous schedule on
the GPU with a quantized KV). 6 Local: PASS. 7, 8 Clusters: nodes run the same handlers; not run on a cluster. 9 LoRA
training, 10 LoRA playback: N/A, attention on the CPU. 11 Vision: the text handler's region, unchanged in kind; not run
live. 12, 13 REST: N/A. 14 CLI: N/A, no flag changed. 15 Embedding facade: unchanged.

**Docs.** `docs/howto.md` (`--cache-type-k`: the type costs no time to first token, with the figures), `docs/performance.md`
(the readings and their cite), `docs/agent-arch.txt` (`writeEncodedRows`, the region's stored-row download, the kit probe),
`CHANGELOG.md` (Session 148), `docs/perf-compare/README.md`; the kit's `coverage.tsv` (`feature:device-kv-prefill`).

**Out-of-tier changes (README rule 9):** none.

**Tests run.** The unit suite (`mvn -o test -pl
api,tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player,juno-kit -am -Dgroups='gpu | !gpu'
-Dmaven.test.failure.ignore=true`, GPU groups on the GTX 1080, 1 h 19 min): `registry` 93, `lora` 116, `kvcache` 228 (the
nine new), `health` 23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator` 417 (1 skipped) and its integer-kernel
execution 1, `vision` 98, `metrics` 86, `juno-player` 158 (2 skipped) and `juno-master` 15 pass; `node` 1,622 run (the two
new region tests), 59 skipped, **3 failures**; `juno-kit` 153 run (the six new), **1 failure**. Each failing class re-run
alone after:
  - `DeviceUploadOrderingTest.theFirstGemvAfterAnUploadReadsTheWholeMatrix`, 2 of 4 (Q5_K 512x2048 round 1, row 259: 5.51 right
    after the upload against -19.85 once the device was idle; Q6_K round 1, row 182: -32.79 against -44.76). Alone: 4 of 4
    pass, two runs. The upload race recorded in the entries above; no class on its path is in this change.
  - `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`: 84,541,440 bytes reported free with the
    device full against <= 67,108,864, in the suite and alone (two runs). The driver holdback recorded in the entries above;
    neither the test nor `DeviceScratchBudget` is in the diff. Bound not loosened.
  - `ProbeRunnerTest.aProbeRunsInItsOwnJvmAndLeavesItsReportAndLog`: the runner re-ran `EnvProbe` and reported NOISY because
    other processes used 12% of the CPU (desktop programs: an editor, a browser, a game client); alone 3 of 3 the same, at
    15% to 51%. The test asserts one attempt and a DONE status, which a busy desktop prevents; `EnvProbe` and the runner are
    not in this change (147 of 147 passed in the previous entry's suite on a quieter desktop). Not changed.

**Exit criteria:** 6 of 15, unchanged. The prefill bar (decisions 6 (a) and 8 (a)) stays open until the owner's gate.

**Owed to the owner:** the pinned prefill gate, `bash dist/device-kv-prefill/wait-and-run.sh`, started in a terminal left
open (it asks for sudo once, waits for `dist/device-kv-prefill/READY` and for no Maven or test process, checks the tree is
the one READY names, then runs `run-gate.sh`): per sweep model six JVMs, A B A B A B (A `f16`, B `q8_0`), each the
2,048-token prefill in one window and in 128-row windows, one warm-up and five measured repetitions; CPU governor
performance, turbo off, the GPU clock lock requested at 1,300 MHz and recorded per JVM (this host's driver accepts it without
applying it), the card cooled to 65 C before each JVM. About 1 to 1.5 h. The desktop's programs (a browser, an editor, a game
client) used 5% to 51% of the CPU through this entry's runs; closing them before the gate keeps both sides alike. Results: `dist/device-kv-prefill/runs/<UTC>/results.txt`,
eight `RESULT` lines (four models, two widths), each PASS at a ratio <= 1.10.

**Decisions for the owner:** none.

**Next for the executor:** the owner's prefill gate: publish it and score the prefill bar; then decisions 5 and 4 (the
out-of-tier follow-ups); item 6; step 7.

### 2026-11-16, later: the owner's prefill gate passes on every sweep model at both widths; item 7's prefill bar met

**The run** ([`20261116T163811Z-device-kv-prefill-gate`](../perf-compare/20261116T163811Z-device-kv-prefill-gate/INDEX.md)):
`dist/device-kv-prefill/run-gate.sh`, owner run 2026-11-16T16:38Z, started by the waiter on `READY`. The tree hash it
checked (`head a4a91182...`, `diff 023fd7eb880c6512`) still matched after the run, so the gate measured the tree the entry
above tested. CPU pinned (performance, turbo off); the GPU lock was requested and not applied, as on every run on this card,
so the clock was recorded: each model's two sides ran within 25 MHz of each other's median loaded clock, the `q8_0` side the
lower and warmer, at the same memory clock. All 24 JVMs passed their test and logged no Juno warning.

| Model | One 2,048-row window | 128-row windows |
|---|---|---|
| TinyLlama | **0.9608** | **0.9609** |
| Qwen2.5-3B | **0.9443** | **0.9122** |
| Phi-3.5-mini | **0.8772** | **0.8769** |
| Mistral 7B | **0.9525** | **0.9515** |

`q8_0` over FP16, median of three JVMs per side, A B A B A B; bar <= 1.10x: 8 of 8 pass. Every `q8_0` JVM is faster than
every FP16 JVM on every model at both widths (the slowest `q8_0` over the fastest FP16 at most 0.982).

**Docs.** `docs/howto.md` (`--cache-type-k`: the pinned figures), `docs/performance.md` (the gate and its cite),
`CHANGELOG.md` (Session 148's figures), `docs/perf-compare/README.md`; exit criterion 7's note.

**Exit criteria:** 6 of 15. Criterion 7 stays ticked, now with its prefill bar met (decisions 6 (a) and 8 (a)).

**Owed to the owner:** nothing.

**Decisions for the owner:** none.

**Next for the executor:** decisions 5 (a) and 4 (a), the out-of-tier follow-ups under README rule 9 (the FP16 decode
attention taking the wide decode block; the FFN gate and up GEMVs reading the q8_1 input once for two rows), each with its
own A/B and gates; then item 6 (exact host KV growth and real FP16 host storage, exit criterion 6); then step 7.

### 2026-11-16, latest: decision 5 (a): the FP16 decode attention on the wide decode block; FP16 decode 1.07x to 1.30x, unpinned; the owner's gate owed

**Plan against code.** No drift. Every FP16 decode row (the decode region's `ResidentQkvPath.attend`, the decode batch's
`ResidentBatchDecode`, `CudaGqaAttention` for `--parallel` streams) reaches `GqaAttentionKernel.launch` at
`rowsPerBlock` 1, which ran the 128-thread tiled entries' decode loop: 32 slots, one block per head and row (Qwen2.5-3B:
16 blocks on the GTX 1080's 20 multiprocessors). The quantized module's wide decode (`gqa_decode_kv_d64/_d128`, 2026-11-14
entry) was the template.

**What changed** ([`20261116T204541Z-fp16-wide-decode`](../perf-compare/20261116T204541Z-fp16-wide-decode/INDEX.md)):
- `gqa_decode_wide.cuh` (new): the wide decode body moved out of `gqa_attention_kv.cu`, with an FP16 variant: 256
  threads, 64 slots of 4 lanes, each slot the tiled kernel's FP16 decode loop at a 64-key stride, merged over 64 slots.
  The FP16 variant keeps the interleaved lanes (every lane of a 96- or 80-wide head has work; the quantized variant's
  contiguous chunks leave a quarter of a 96-wide head's lanes idle), and loads each key's K and V quads up front, one
  8-byte read-only load a quad with no branch between them (a slot outside the window reads row `lo` and discards it).
- `gqa_attention.cu`: entries `gqa_decode_d64/_d128`. `GqaAttentionKernel.wideDecode(rowsPerBlock, headDim)` (rows of
  one at head widths up to 128) routes both launchers; the FP16 decode launch goes through a reused `KernelParams` block.
  256-wide heads and prefill tiles keep the tiled entries.
- PTX: the tiled entries of `gqa_attention.ptx` byte for byte the committed ones (the diff only adds lines);
  `gqa_attention_kv.ptx` byte for byte the committed one apart from the shared-variable names (the new template argument).
  New entries 128 and 78 registers, no spills.
- The first version read each quad as two 4-byte generic loads, each behind its own branch: 1.04x to 1.05x the tiled
  kernel at 2,048 keys on Phi-3.5-mini and Mistral 7B. The up-front 8-byte loads changed no output bit (SHA-256 of 80
  decode outputs over five shapes and two windows, equal before and after; scratch test, removed).

**Tests first.** `GqaAttentionWideDecodeTest` (GPU, 3): rows on either side of a 64-key stride (contexts 1, 2, 63, 64,
65, 127, 128, 129, 191, 1,000 over their own caches, one launch) at head widths 64, 80, 96, 128 and 256 against the CPU
oracle; sliding windows 1, 63, 64, 65, 100 and 128 whose first key is off the stride; a row among four others bit for bit
the row alone. `GqaAttentionDecodeRouteTest` (1): which launches take the wide block. Shown able to fail, each defect put
in, rebuilt, run and taken out (checksums of the source, launcher and PTX equal afterwards):

| Defect | Caught by |
|---|---|
| D1: the merge drops slot 63 | `GqaAttentionWideDecodeTest` (2), `GqaAttentionTiledTest` (2), `GqaAttentionKernelParityTest` (2) |
| D2: the loop strides 32 keys over 64 slots | the same six |
| D3: the merge stores the FP16 accumulator block-major | the same, plus `GqaAttentionTiledTest.decodeAtMaxSeqLen`, `unalignedWindows` |
| D4: the FP16 decode ignores the window | `GqaAttentionWideDecodeTest.slidingWindowsMatchTheOracle` only |
| D5: the route takes 256-wide heads | `GqaAttentionDecodeRouteTest`, `GqaAttentionWideDecodeTest.rowsAroundTheKeyStride`, `GqaAttentionTiledTest.decodeRow...` |

**Readings, unpinned** (one JVM at a time; baseline = `git archive` of `637751b0` in scratch, run from its test classes):
- Kernel (`GqaAttentionDecodeBench`, now with a `tiled` lane: the unchanged tiled entry at one row a block): at 2,048 keys
  0.295x (TinyLlama), 0.434x (Qwen2.5-3B), 0.684x (Phi-3.5-mini), 0.499x (Mistral 7B) the old decode time; under 1.0 at
  every context from 64 keys (64-key launches of 6 to 10 us vary by over 10% run to run).
- Decode tg at a 2,048-token context (`DeviceKvDecodeTgLiveTest`, A1 B1 C1 A2 B2): FP16 after over before **1.302**
  (TinyLlama), **1.091** (Qwen2.5-3B), **1.065** (Phi-3.5-mini), **1.078** (Mistral 7B); every new JVM faster than every
  old one.
- Item 7's decode ratio moves with its denominator: `q8_0` over the new FP16 decode 0.968, 0.955, 0.977, 0.976 (the
  2026-11-15 gate read 1.03x to 1.23x against the old one). The quantized decode launch is now 1.22x, 1.37x, 0.68x and
  1.05x the FP16 one at 2,048 keys (its loop still loads block words behind per-block branches). Exit criterion 7's note
  records it; decision 9 below.
- Hot-path gate, `compare-lora.sh --gpu --skip-build --reps 3` from each tree: train 1.000x, playback 0.998x, recall equal
  (bars 1.25x and 0.80x). LoRA attends on the CPU, so the gate holds that nothing else moved.

**Cross-surface rows**, against the 2026-11-14 entry: 2 CUDA, single decode (the decode region), the decode batch
(continuous's running set, `ResidentBatchDecodeTest` 22 of 22 bit for bit against single decodes) and `--parallel`
streams (`CudaGqaAttention`, `BoundedLaunchParityTest`) all reach the routed launch; vision's text backbone runs the same
handler. ROCm, `--gpu-attention off`, the CPU backend, LoRA training and `--lora-play` attend on the CPU, unchanged. No
flag added.

**Docs.** `docs/agent-arch.txt` (`GqaAttentionKernel`'s decode entries, `gqa_decode_wide.cuh`, the new tests),
`docs/howto.md` (`--gpu-attention`: the wide block's figures; `--cache-type-k`: the decode ratio against the faster FP16),
`docs/performance.md`, `CHANGELOG.md` (Session 149), `docs/perf-compare/README.md`. The howto's greedy divergence counts
against `--gpu-attention off` were measured on the old kernel; the wide block sums the same values in another order, held
to the same tolerance of the CPU oracle; not re-measured.

**Kit probes.** None added: no flag, threshold or surface. The `gpu.kv-quality`, `gpu.consistency` and `gpu.kv-prefill`
probes run this decode path; their catalog costs were measured on the old kernel and can only fall.

**Out-of-tier changes (README rule 9):** this change, uncommitted on HEAD `637751b0`: the FP16 GPU decode attention
(`gqa_attention.cu`, `gqa_decode_wide.cuh`, `GqaAttentionKernel`, `GqaAttentionKvKernel`'s route). A measurement boundary
for GPU decode: every published GPU decode tg reading before it (FP16 and the `q8_0`-over-FP16 ratios, including
`20261115T200324Z-device-kv-tg-gate-rerun`) was taken on the old FP16 decode kernel. GPU prefill is unchanged (tiled
PTX byte for byte).

**Tests run.** The suites on the final tree (`mvn -o test -pl node,coordinator -am -Dgroups='gpu | !gpu'
-Dmaven.test.failure.ignore=true`, GPU groups on the GTX 1080, 51 min): `api` (no tests), `registry` 93, `lora` 116, `kvcache` 228,
`health` 23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator` 417 (1 skipped) and its integer-kernel execution 1
pass; `node` 1,626 run, 59 skipped, **2 failures**, both `DeviceUploadOrderingTest.theFirstGemvAfterAnUploadReadsTheWholeMatrix`
(Q5_K 512x2048 round 1, row 252: -13.71 right after the upload against -82.77 once the device was idle; and Q6_K), the
upload race recorded in the entries above; alone straight after: 4 of 4 pass. No class on its path is in this change.
The attention and resident-path classes (141 tests) also pass alone on the final kernel. `check-plan-thresholds.sh`:
passes.

**Exit criteria:** 6 of 15, unchanged. Criterion 7 stays ticked on its pinned evidence; the owner's gate re-scores its
decode bar against the faster FP16 decode (lane C), and a miss reopens it.

**Owed to the owner: the gate.** In a terminal left open: `bash dist/fp16-wide-decode/wait-and-run.sh` (it waits for
`READY`, which is written). Per model nine JVMs, A B C three times (A f16 from `dist/fp16-wide-decode/baseline-tree`, the
`git archive` of `637751b0`, compiled before `READY`; B f16 and C `q8_0` from this tree), `DeviceKvDecodeTgLiveTest` at
2,048 tokens, cooled to 65 C before each JVM, a CLOCK line per JVM. Two RESULT lines per model: FP16 after over before
(bar >= 1.00, decision 5's A/B) and `q8_0` over FP16 after (bar >= 0.95, criterion 7's decode bar). About 90 minutes.

**Decisions for the owner.**
9. **The quantized decode loop's loads.** The FP16 decode now loads each key's quads up front; the quantized loop
   (`gqa_decode_kv`) still loads block words behind per-block branches, and its launch reads 1.05x to 1.37x the new FP16
   one at 2,048 keys on three sweep models (Phi-3.5-mini 0.68x). (a) Recommended: a follow-up inside item 7 giving it
   up-front loads the same way, measured by the bench and the decode gate's `q8_0`-over-FP16 ratio, parity and quality
   tests unchanged; run it before decision 4 if the gate's lane C misses on any model, else after decision 4. (b) Leave
   it: `q8_0` keeps its memory savings and decodes at about 0.96x to 0.98x the FP16 rate.

The gate's bars are the executor's reading of decision 5's "own A/B": FP16 decode no slower on any sweep model (>= 1.00),
and criterion 7's existing 0.95. Say so if you want others; the script reads them from `BAR_AB` and `BAR_CB`.

**Next for the executor:** after the owner's gate, record it (and decision 9); then decision 4 (a) (the FFN gate and up
GEMVs reading the q8_1 input once for two rows), or decision 9 first if lane C misses; then item 6; then step 7.

### 2026-11-16, latest: the owner's gate: decision 5 (a) passes on every sweep model; `q8_0` misses item 7's decode bar on Qwen2.5-3B (criterion 7 reopened); decision 9 (a)

**The gate** ([`20261116T224500Z-fp16-wide-decode-gate`](../perf-compare/20261116T224500Z-fp16-wide-decode-gate/INDEX.md);
owner run 2026-11-16T22:45Z to 2026-11-17T00:02Z, `dist/fp16-wide-decode/run-gate.sh`): CPU pinned, GPU clock recorded and
not fixed (the lock accepted and not applied, every CLOCK line), cooled to 65 C before each JVM, three sides at the same
median loaded clock per model (Qwen2.5-3B's two candidate sides 13 MHz below the baseline side). The tree hash after the
run still matched `READY` (`diff 6c54ee645662eb41`).

| Model | FP16 after over before (bar 1.00) | `q8_0` over FP16 after (bar 0.95) |
|---|---|---|
| TinyLlama 1.1B | 1.2813 pass | 0.9633 pass |
| Qwen2.5-3B | 1.0895 pass | **0.9382 fail** (0.938 in each cycle) |
| Phi-3.5-mini | 1.0859 pass | 0.9749 pass |
| Mistral 7B | 1.0669 pass | 0.9765 pass |

Every FP16 JVM of this tree is faster than every one of `637751b0`; a 2,048-token prefill's time is unchanged (0.996x to
1.004x, from the logs). Both sides of lane C agree with the 2026-11-15 re-run within 0.6%: the `q8_0` decode did not move,
the FP16 decode it is scored against did. Qwen2.5-3B needs 1.3% more `q8_0` decode rate (about 6 us a layer).

**Exit criteria:** 5 of 15. Criterion 7 unticked: its decode bar misses on Qwen2.5-3B against the faster FP16 decode. Its
other bars (bytes, capacity, agreement, prefill) stand. Decision 5 (a)'s own A/B is met; the standing gate's GPU lane is a
tier-close gate and is not re-run here.

**Docs.** `docs/howto.md` (`--gpu-attention`, `--cache-type-k`), `docs/performance.md`, `CHANGELOG.md` (Session 149)
carry the pinned figures; `docs/perf-compare/README.md` has the row.

**Owner decision 9 (a)**, 2026-11-16: the quantized decode loop (`gqa_decode_kv`) takes up-front loads the way the FP16
decode does, inside item 7, measured by the bench and a decode gate's `q8_0`-over-FP16 ratio, parity and quality tests
unchanged. Lane C missed, so it runs before decision 4.

**Next for the executor:** decision 9 (a); then decision 4 (a); then item 6; then step 7.

### 2026-11-17: decision 9 (a): the quantized decode loop with up-front loads; `q8_0` decode 0.97x to 0.98x FP16, unpinned; the owner's gate owed

**Plan against code.** No drift. Quantized decode rows at head widths up to 128 run `gqa_decode_kv_d64/_d128`
(`GqaAttentionKernel.wideDecode`), whose loop was `gqa_decode_kv` (shared with the 256-wide tiled entry): K chunk loaded
behind per-block branches, V loaded after the softmax update.

**What changed** ([`20261117T024602Z-quant-decode-loads`](../perf-compare/20261117T024602Z-quant-decode-loads/INDEX.md)):
- `gqa_decode_wide.cuh`: `gqa_decode_kv_wide`, the wide block's own quantized loop (`gqa_decode_kv` stays for the
  256-wide tiled entry, whose PTX is unchanged). Each key's K and V chunks are loaded before the dot product
  (`GqaChunk`: scale and quant words, no branch between them; a slot outside the window loads row `lo`, a lane past the
  head the head's first block, both discarded). `gqa_quad_floats` turns quant bytes into floats by a byte permute under
  the exponent of 2^23 and one subtraction, exact, where the old loop shifted and converted each value. No branch per
  block or quad (with a head width a multiple of 32 a lane's chunk is wholly in the head or wholly past it), so the
  old loop's divergence regions are gone. `runSum = fmaf(runSum, corr, w)` written out.
- `gqa_attention_kv.cu`: decode entries `__launch_bounds__(256, 2)`. Uncapped, the up-front loads took 144 registers at
  the 128-wide entry, one block a multiprocessor: Mistral 7B and Phi-3.5-mini read 1.46x and 1.35x slower at 2,048 keys
  (32 heads on 20 multiprocessors in two waves). Capped at 128, no spills.
- PTX: `gqa_attention_kv.ptx`'s tiled entries byte for byte the committed ones; `gqa_attention.ptx` unchanged.
- Outputs (scratch digest test, removed): 45 of 50 decode configurations (5 shapes, 5 format pairs, windows 0 and 100)
  bit for bit the old loop's. K `q4_0` with V `f16` at window 0 differs by at most 2 ulp (119 of 2,048 values of a
  1,000-key row): the old loop's compiler fused the running sum for four pairs and not for that one (shown by forcing
  each form: unfused matched that pair and broke the other four).

**Tests first.** `GqaAttentionWideDecodeKvTest` (GPU, 3, new), passing on the old loop before any change: rows on either
side of a 64-key stride at head widths 32, 64, 96, 128 and 256 for the five format pairs (`q8_0`, `q8_0`/`q4_0`, `q4_0`,
`f16`/`q8_0`, `q4_0`/`f16`), each cache sized to its context; sliding windows 1 to 128 off the stride; a row among four
others bit for bit the row alone. Shown able to fail, each defect put in, rebuilt, run and taken out (source and PTX
checksums equal afterwards):

| Defect | Caught by |
|---|---|
| E1: every slot reads row `lo` | `GqaAttentionWideDecodeKvTest` (2), `DeviceKvQuantParityTest` (4) |
| E2: the window ignored in a slot's validity | `GqaAttentionWideDecodeKvTest.slidingWindows...`, `DeviceKvQuantParityTest.slidingWindow...` only |
| E3: the second half of a block read as the first (64-wide heads) | `GqaAttentionWideDecodeKvTest` (2), `DeviceKvQuantParityTest` (2) |
| E4: a quad's fourth byte read as its third | `GqaAttentionWideDecodeKvTest` (2), `DeviceKvQuantParityTest` (3) |
| E5: a lane past the head adds its dot product | none: that lane's query quads are zero, so it adds +0 either way; the guard stays as a second line |

**Readings, unpinned** (before = this tree without the change, copied to scratch):
- Kernel (`GqaAttentionDecodeBench`, four runs alternating trees): `q8_0`/`q8_0` at 2,048 keys after over before 0.780
  (TinyLlama), 0.735 (Qwen2.5-3B), 0.987 (Phi-3.5-mini), 0.915 (Mistral 7B); over the FP16 launch 0.974, 0.897, 0.651,
  0.900 (before 1.240, 1.081, 0.653, 1.000). Phi-3.5-mini at 1,024 keys read 1.10 (its idle fourth lane now runs).
- Decode tg at a 2,048-token context (A `q8_0` before, B `q8_0` after, F `f16`, x2): B over A 1.045, 1.012, 0.961 (0.994
  over three more pairs; a cool card's first JVM reads high), 1.024; **B over F 0.983, 0.971, 0.975, 0.976**, where A over
  F read 0.941, 0.959, 1.014, 0.953.
- Hot-path gate, `compare-lora.sh --gpu --skip-build --reps 3` from each tree: train 1.000x, playback 0.975x, recall equal
  (bars 1.25x and 0.80x). LoRA attends on the CPU, so the gate holds that nothing else moved.

**Cross-surface rows**, against the 2026-11-16 entry: unchanged. Every quantized decode row on CUDA (the decode region,
the decode batch, `--parallel` streams) reaches the routed launch; ROCm, the CPU backend, LoRA training and `--lora-play`
attend on the CPU. No flag added.

**Docs.** `docs/agent-arch.txt` (the wide block's quantized loop, the new test), `docs/howto.md` (`--cache-type-k`),
`docs/performance.md`, `CHANGELOG.md` (Session 149, a bullet), `docs/perf-compare/README.md`.

**Kit probes.** None added: no flag, threshold or surface; `gpu.kv-quality`, `gpu.consistency` and `gpu.kv-prefill`
run this path.

**Tests run.** The suites on the final tree (`mvn -o test -pl node,coordinator -am -Dgroups='gpu | !gpu'
-Dmaven.test.failure.ignore=true`, GPU groups on the GTX 1080, 56 min): `api` (no tests), `registry` 93, `lora` 116,
`kvcache` 228, `health` 23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator` 417 (1 skipped) and its
integer-kernel execution 1 pass; `node` 1,629 run, 59 skipped, **3 failures**: `DeviceUploadOrderingTest` (2, the upload
race of the entries above; alone straight after, 4 of 4 pass, twice) and
`PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` (free bytes reported with the device
full 106,364,928 against the 67,108,864 allowance). The latter fails alone too, twice, and in the tree before this change
(99,221,504): the driver's free-memory reading with the device full, on the desktop's shared card today; it passed in the
2026-11-16 suite, and no class on its path is in this change. The 62 attention and device-KV GPU tests also pass alone on
the final kernel. `check-plan-thresholds.sh`: passes.

**Exit criteria:** 5 of 15, unchanged; criterion 7's decode bar waits on the owner's gate.

**Owed to the owner: the gate.** In a terminal left open: `bash dist/quant-decode-loads/wait-and-run.sh` (it waits for
`READY`, which is written). Per model six JVMs, A B three times (A `f16`, B `q8_0` keys and values, both this tree),
`DeviceKvDecodeTgLiveTest` at 2,048 tokens, cooled to 65 C before each JVM, a CLOCK line per JVM; one RESULT line per
model, `q8_0` over `f16`, bar >= 0.95 (criterion 7's decode bar). About 60 minutes.

**Next for the executor:** record the gate (a pass on every model ticks criterion 7 again); then decision 4 (a) (the FFN
gate and up GEMVs reading the q8_1 input once for two rows); then item 6; then step 7.

### 2026-11-17, later: the owner's gate: `q8_0` decode 0.958x to 0.985x FP16, pinned, on every sweep model; criterion 7 met again

**The gate** ([`20261117T053104Z-quant-decode-loads-gate`](../perf-compare/20261117T053104Z-quant-decode-loads-gate/INDEX.md);
owner run 2026-11-17T05:31Z to 06:24Z, `dist/quant-decode-loads/run-gate.sh`): CPU pinned, GPU clock recorded and not fixed
(the lock accepted and not applied, every CLOCK line), cooled to 65 C before each JVM, both sides at the same median loaded
clock per model (Qwen2.5-3B's `q8_0` side 13 MHz below its FP16 side). The tree it ran (`head 637751b0`, `diff
c6b66b809fac35e7`, recorded in `READY` and in the run's `tree.txt`) is the commit `9c33ab93` the owner made afterwards: the
hash rebuilt from `git diff 637751b0 9c33ab93` the way `tree-hash.sh` reads a tree is `c6b66b809fac35e7`.

| Model | `f16`, tok/s | `q8_0`, tok/s | `q8_0` over `f16` (bar 0.95) |
|---|---|---|---|
| TinyLlama 1.1B | 160.645 | 158.281 | 0.9853 pass |
| Qwen2.5-3B | 62.688 | 60.049 | 0.9579 pass |
| Phi-3.5-mini | 46.784 | 45.566 | 0.9740 pass |
| Mistral 7B | 37.478 | 36.428 | 0.9720 pass |

- **Per cycle**, two single cycles sit under the bar: Qwen2.5-3B 0.957, 0.936, 0.958 (its second `q8_0` JVM 58.495 against
  60.069 and 60.049), Phi-3.5-mini 0.917, 0.975, 0.976 (its first `q8_0` JVM, the slowest of either side, with the card
  throttling for heat on about 80% of loaded samples). The bar scores medians of three, as every pinned gate of this tier
  has; every row's repetitions agree within 15% of their median (largest spread 8.3%). Qwen2.5-3B's margin is 0.8%.
- **Against the 2026-11-16 gate**, which ran the same FP16 decode code: FP16 within 1.0% on every model; `q8_0` +3.3%
  (TinyLlama), +2.9% (Qwen2.5-3B), -0.3% (Phi-3.5-mini), +0.2% (Mistral 7B), the shape the unpinned bench showed (the loop
  change matters where the old loop's launch was slowest against FP16).
- **Prefill, not a bar:** a 2,048-token prefill with `q8_0` 0.937x to 0.943x the FP16 time, within 1.4% of the 2026-11-16
  gate's `q8_0` readings; the prefill tiles' PTX is the committed one.

**Exit criteria:** 6 of 15. Criterion 7 ticked again: every bar of it is met (bytes per position, session capacity, greedy
agreement and the prefill bar on their earlier evidence, decode tg on this gate).

**Docs.** `docs/howto.md` (`--cache-type-k`: the pinned decode ratio), `docs/performance.md`, `CHANGELOG.md` (Session 149),
`docs/perf-compare/README.md`.

**Tests.** No code changed, so none was run. `check-plan-thresholds.sh` passes after these edits.

**Out-of-tier changes:** none.

**Next for the executor:** decision 4 (a) (the FFN gate and up GEMVs reading the q8_1 input once for two rows, outside item 7,
README rule 9); then item 6; then step 7.

### 2026-11-17, latest: decision 4 (a) begun: the FFN gate and up projections in one launch; thresholds set before implementation

**Plan against code.** No drift. Every sweep model's decode region issues its FFN through `ResidentLayerTail.issue`:
one Q8_1 quantization of the normed row, then two `q4k_gemv` launches over it, gate and up (Phi-3 passes the two halves of
its fused `ffn_up` as row slices). `q4k_gemv` runs one row a block, 4 warps, 16 super-block slots; at hidden width 2,048
(TinyLlama, Qwen2.5-3B: 8 super-blocks a row) warps 2 and 3 have no work, and every block reads the whole 2,304-byte Q8_1
vector. The decode batch (`ResidentBatchDecode`) runs its own batched kernels and is not in this change.

**Design.** One launch for both projections (`q4k_gemv_pair`): a lane loads its slice of the Q8_1 vector once and dots it
with the gate row and the up row; a row's arithmetic is the single-row kernel's, so both outputs equal two `q4k_gemv`
launches bit for bit. Where a row has at most 8 super-blocks, warps 2 and 3 take the next row index, so each block computes
two gate and two up rows. Gate and up of another type than Q4_K, or of unequal shapes, keep the two launches.

**Thresholds** (the executor's reading of decision 4 (a)'s "GEMV gates of its own"; set before any kernel code was written):
- Parity (unit test, GPU): the paired launch's gate and up outputs bit for bit equal to two `q4k_gemv` launches over the
  same Q8_1 input, at odd and even row counts and 1 to 17 super-blocks a row.
- Kernel (bench, unpinned): on each sweep model's gate/up shape (TinyLlama 5,632 x 2,048; Qwen2.5-3B 11,008 x 2,048;
  Phi-3.5-mini 8,192 x 3,072; Mistral 7B 14,336 x 4,096), at each of nine placements of the Q8_1 vector within 64 KiB
  (0x0, 0x200, 0x400, 0x800, 0xe00, 0x1000, 0x1800, 0x4800, 0xbe00), the paired launch's device time **<= 1.00x** the two
  launches' at the same placement; each side's slowest-over-fastest spread across placements reported.
- Decode tg (the owner's gate, pinned, same hour): at a 2,048-token context, FP16 and `q8_0` device KV each **>= 1.00x**
  the build before (`9c33ab93`) on every sweep model.

**What changed** (uncommitted on HEAD `9c33ab93`):
- `q4k_gemv_pair.cu` (new) and `Q4KGemvPairKernel` (new): the paired launch described above, 48 registers, no spills,
  launch parameters reused per thread (`KernelParams`).
- `kquant_gemv.cuh` (new): `q4k_gemv.cu`'s helpers moved out so both kernels share them; `q4k_gemv.ptx` rebuilt from the
  moved source is byte for byte the committed one.
- `ResidentLayerTail.issue`: gate and up through one paired launch where `Q4KGemvPairKernel.pairs` (both Q4_K, one shape),
  else the two launches as before.
- Left to the compiler, the paired kernel's per-block term differed from `q4k_gemv`'s by an ulp: it contracted other
  products into FMAs. The term is now written with explicit roundings in the form `q4k_gemv`'s machine code (SASS, not
  its PTX: `ptxas` fuses `d * (...)` into the subtraction) has them.

**Tests first.** `Q4KGemvPairTest` (GPU, 1): gate and up bit for bit two `q4k_gemv` launches over the same Q8_1 input at
10 shapes (1 to 1,000 rows, odd and even; 1, 3, 5, 8, 9, 12, 16 and 17 super-blocks), and the row past the last never
written. Seen failing for the right reason twice before passing: an ulp on `1 x 256` and then on `2 x 256` until the term
matched `q4k_gemv`'s machine code. Shown able to fail, each defect put in, rebuilt, run and taken out (source and PTX
checksums equal afterwards):

| Defect | Caught by |
|---|---|
| F1: the second row index overlaps the first (`blockIdx.x + group`) | `Q4KGemvPairTest` (gate row 4 of 6 x 1,280) |
| F2: the up row dotted with the low half's input for both halves | `Q4KGemvPairTest` (up row 0 of 1 x 256) |
| F3: the second row index reduces into the first's slots | `Q4KGemvPairTest` (gate row 1 of 2 x 256) |

Existing tests on the routed tail, GPU, all pass: `ResidentQkvPathTest` (19, whole layers bit for bit the op-at-a-time
path, Q4_K gate and up, so the paired route ran), `ResidentQkvPathQwen2Test`, `-Phi3Test`, `-Qwen3Test` (5 each),
`ResidentQkvPathDeviceFullTest` (1), `ResidentBatchDecodeTest` (22), `BatchDecodeRegionHandlerTest` (11),
`SgemvSameXParityTest` (1), `Q4KMmqParityTest` (2), `KQuantMmqParityTest` (3).

**Readings, unpinned** ([`20261117T082501Z-gemv-gate-up-pair`](../perf-compare/20261117T082501Z-gemv-gate-up-pair/INDEX.md)):
the kernel bar is met. Paired over two launches, across two runs and all nine offsets: TinyLlama 0.632x to 0.943x,
Qwen2.5-3B 0.746x to 0.884x, Phi-3.5-mini 0.876x to 0.973x, Mistral 7B 0.935x to 0.969x. Placement still moves the
paired launch (1.009x to 1.209x slowest over fastest); on Phi-3.5-mini its spread is wider than the two launches'.

**Out-of-tier changes (README rule 9):** this change (decision 4 (a), outside item 7): the decode region's FFN gate and up
GEMVs (`q4k_gemv_pair.cu`, `Q4KGemvPairKernel`, `ResidentLayerTail`). A measurement boundary for GPU decode on every
handler with a decode region (Llama family, Qwen2, Phi-3, Qwen3) once its gate passes and it is committed; outputs are
unchanged bit for bit, so no quality reading moves. GPU prefill, the decode batch, the CPU path and LoRA are unchanged.

**Still owed for decision 4 (a)** (the session stopped here for context; `docs/infra-plan/PROMPT-gemv-gate-up-pair.md`
has the commands): an unpinned decode tg A/B against `9c33ab93` (FP16 and `q8_0`, 2,048 tokens, four sweep models);
the hot-path gate `compare-lora.sh --gpu` from each tree; the node and coordinator suites with the GPU groups; the
owner's pinned gate prepared under `dist/gemv-pair/`; docs (`docs/performance.md`, `CHANGELOG.md`).

**Exit criteria:** 6 of 15, unchanged (decision 4 (a) is outside them).

**Next for the executor:** finish decision 4 (a) as listed; then item 6; then step 7.

### 2026-11-17, latest: decision 4 (a): decode 1.005x to 1.077x the build before, unpinned; LoRA gate and suites; the owner's gate prepared

**Plan against code.** No drift. The uncommitted change the previous entry describes is on disk as recorded
(`q4k_gemv_pair.cu`, `kquant_gemv.cuh`, `q4k_gemv_pair.ptx`, `Q4KGemvPairKernel`, `ResidentLayerTail.gateAndUp`,
`Q4KGemvPairTest`, `Q4KGemvPairBench`); no code changed in this session. `check-plan-thresholds.sh` passes.

**Baseline.** `git archive 9c33ab93` in `dist/gemv-pair/baseline-tree` (git-ignored, `.archive-commit` written, `models/`
linked), built there with `mvn -o package -DskipTests`; this tree rebuilt the same way (its shaded jar predated the change).
The same baseline tree serves the owner's gate.

**Readings, unpinned** ([`20261117T082501Z-gemv-gate-up-pair`](../perf-compare/20261117T082501Z-gemv-gate-up-pair/INDEX.md),
sections 2 and 3, added to the kernel bench's entry):
- Decode tg at a 2,048-token context (`DeviceKvDecodeTgLiveTest`, A the baseline tree, B this tree, A B x2, Mistral 7B x4,
  cooled to 65 C before each JVM), B over A, means, `f16` / `q8_0` device KV: TinyLlama **1.060 / 1.077**, Qwen2.5-3B
  **1.045 / 1.041**, Phi-3.5-mini **1.028 / 1.011**, Mistral 7B **1.008 / 1.005** (medians 1.012 / 1.008). Every B JVM
  is faster than every A JVM on TinyLlama, Phi-3.5-mini and Qwen2.5-3B `f16` (Qwen2.5-3B `q8_0`: the slowest B 0.999x the
  fastest A). Mistral 7B's sides overlap: one pair of each type reads under 1.0 (0.969, 0.995), so two more pairs were run
  after the first two read 1.003 and 1.004. The gain follows the kernel bench: largest at width 2,048, where the idle warps
  take a second row, smallest at 4,096, where the gate and up launches save 3% to 6.5% of their own time.
- Hot-path gate, `compare-lora.sh --gpu --skip-build --no-publish --reps 3` from each tree: train 0.976x, playback 0.993x,
  recall equal (bars 1.25x and 0.80x). LoRA does not run the decode region, so the gate holds that nothing else moved.

**Cross-surface rows**, against the 2026-11-17 entries: unchanged. The paired launch runs only inside the CUDA decode
region's layer tail (Llama family, Qwen2, Phi-3, Qwen3, single decode rows); gate and up of another type than Q4_K, or of
unequal shapes, keep two launches, as does a run where the paired module does not load (warning logged). The decode
batch, prefill, ROCm, the CPU backend, LoRA training and `--lora-play` do not reach it. Outputs are bit for bit the two
launches', so no quality, parity or grammar reading moves. No flag added.

**Docs.** `docs/agent-arch.txt` (previous entry), `docs/performance.md` (the pairing, kernel and decode figures),
`CHANGELOG.md` (Session 150), `docs/perf-compare/README.md` (the entry's row extended).

**Kit probes.** None added: no flag, threshold or surface. `gpu.consistency`, `gpu.kv-quality` and the GPU live probes run
the decode region and so this launch.

**Tests run.** The suites on the final tree (`mvn -o test -pl node,coordinator -am -Dgroups='gpu | !gpu'
-Dmaven.test.failure.ignore=true`, GPU groups on the GTX 1080, 53 min): `api` (no tests), `registry` 93, `lora` 116,
`kvcache` 228, `health` 23, `tokenizer` 109 (2 skipped), `sampler` 90, `coordinator` 417 (1 skipped) and its
integer-kernel execution 1 pass; `node` 1,630 run (`Q4KGemvPairTest` among them, pass), 59 skipped, **3 failures**, the
same three as the previous entry's suite: `DeviceUploadOrderingTest.theFirstGemvAfterAnUploadReadsTheWholeMatrix` 2 of 4
(Q4_K row 226 and Q5_K row 186, 512x2048, round 1: the upload race of the entries above; alone straight after, 4 of 4
pass, in this tree and in the tree before this change) and
`PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` (free bytes reported with the device full
88,539,136 against the 67,108,864 allowance; alone straight after, 78,905,344 in this tree and 78,905,344 in the tree
before this change, both failing: the driver's free-memory reading on the desktop's shared card, and no class on either
test's path is in this change). Downstream (`mvn -o test -pl vision,metrics,juno-player -am`, filtered to those modules'
packages, 16 min): `vision` 98, `metrics` 86, `juno-player` 158 (2 skipped), all pass. `check-plan-thresholds.sh`:
passes.

**Out-of-tier changes (README rule 9):** this change, as the previous entry records (decision 4 (a), outside item 7). It
becomes a measurement boundary for GPU decode on every handler with a decode region once the owner's gate passes and it
is committed.

**Exit criteria:** 6 of 15, unchanged (decision 4 (a) is outside them).

**Owed to the owner: the gate.** In a terminal left open: `bash dist/gemv-pair/wait-and-run.sh` (it waits for `READY`,
which is written last, checks the tree hash, and waits for any Maven or test process to end). Per model twelve JVMs, A B C D
three times (A `f16` the baseline tree, B `f16` this tree, C `q8_0` the baseline tree, D `q8_0` this tree),
`DeviceKvDecodeTgLiveTest` at 2,048 tokens, CPU pinned, the card cooled to 65 C before each JVM, a CLOCK line per JVM; two
RESULT lines per model, B over A and D over C, each bar >= 1.00. About 1 hour 45 minutes. Mistral 7B's margin is expected
to be about 1%, inside the unpinned spread; a miss there is the owner's call (keep the change, since outputs are
identical and the other models gain, or drop it).

**Next for the executor:** record the gate; then item 6 (exact host KV growth, real FP16 host storage); then step 7.

### 2026-11-18: the owner's gate: decision 4 (a) passes, decode 1.014x to 1.075x the build before, pinned, FP16 and `q8_0`, every sweep model

**The gate** ([`20261118T054118Z-gemv-pair-gate`](../perf-compare/20261118T054118Z-gemv-pair-gate/INDEX.md); owner run
2026-11-18T05:41Z to 07:26Z, `dist/gemv-pair/run-gate.sh`): CPU pinned, GPU clock recorded and not fixed (the lock accepted
and not applied, every CLOCK line), cooled to 65 C before each JVM. Per model twelve JVMs, A B C D three times (A `f16` before,
B `f16` after, C `q8_0` before, D `q8_0` after), `DeviceKvDecodeTgLiveTest` at 2,048 tokens. After: the tree `READY` recorded
(`head 9c33ab93`, `diff 2213b73c2a755981`, checked by `wait-and-run.sh`); before: `git archive 9c33ab93`. The two sides ran
at the same median loaded clock per model, except TinyLlama's FP16 after side (13 MHz above its before side) and Qwen2.5-3B's
two after sides (13 MHz below theirs). Only this record's documents changed after the gate: no file outside `docs/` is newer
than `READY`.

| Model | B over A (`f16`, bar 1.00) | D over C (`q8_0`, bar 1.00) |
|---|---|---|
| TinyLlama 1.1B | 166.349 / 154.740 = 1.0750 pass | 158.229 / 154.124 = 1.0266 pass |
| Qwen2.5-3B | 63.786 / 62.759 = 1.0164 pass | 60.685 / 58.099 = 1.0445 pass |
| Phi-3.5-mini | 47.846 / 46.483 = 1.0293 pass | 46.091 / 45.397 = 1.0153 pass |
| Mistral 7B | 37.975 / 37.467 = 1.0136 pass | 37.156 / 36.470 = 1.0188 pass |

- **Per cycle** every after JVM reads over the before JVM of its cycle (1.002x to 1.094x). On Phi-3.5-mini and Mistral 7B,
  both types, every after JVM is faster than every before JVM; Mistral 7B, the expected thin margin (1.008x and 1.005x
  unpinned, sides overlapping), reads 1.012x to 1.019x per cycle, its readings within 0.5% per side.
- **Scorable:** a side's three JVM readings agree within 8.6% of their median (TinyLlama D). One JVM's own repetitions spread
  15.3% (TinyLlama C2, a before JVM); its reading is the C side's median, and against the fastest C JVM D over C reads 1.016.
- **The before side** reproduces the 2026-11-17 gate's readings of the same bytes within 0.6% on Qwen2.5-3B `f16`,
  Phi-3.5-mini and Mistral 7B; TinyLlama read 3.7% (`f16`) and 2.6% (`q8_0`) lower and Qwen2.5-3B `q8_0` 3.2% lower.
- **Criterion 7's ratio, read, not scored:** `q8_0` over FP16 on the after side reads 0.951 (TinyLlama), 0.951 (Qwen2.5-3B),
  0.963 (Phi-3.5-mini), 0.978 (Mistral 7B). On the before side, the code the 2026-11-17 gate scored at 0.958 to 0.985, this
  run reads 0.926 on Qwen2.5-3B: in its order the `q8_0` JVMs of TinyLlama and Qwen2.5-3B slowed across the three cycles on
  both sides while the `f16` JVMs did not, so a cross-type ratio from this run moves by up to 3.4% on unchanged code. The
  paired launch reads no KV, so it saves the same time per token with either type; from the 2026-11-17 gate's rates and the
  largest saving measured per model, `q8_0` over FP16 can fall at most to 0.984, 0.956, 0.973 and 0.971. Both readings are at
  or above 0.95 on every model: criterion 7 stays met, its decode margin now about 0.6% on Qwen2.5-3B.
- **Prefill, not a bar:** after over before 0.978x to 1.021x (TinyLlama, Qwen2.5-3B, Mistral 7B, both types); the paired
  launch does not run in prefill.

**Exit criteria:** 6 of 15, unchanged (decision 4 (a) is outside them; criterion 7 stays met, above).

**Docs.** `docs/performance.md` (the pinned figures), `CHANGELOG.md` (Session 150's decode line, now pinned),
`docs/perf-compare/README.md` (the gate's row). `docs/howto.md` unchanged: its `q8_0` over FP16 figures (0.96x to 0.99x,
pinned) were measured on the build before, and the bound above keeps them within 0.2%.

**Tests.** No code changed after the suites of the previous entry, so none was run. `check-plan-thresholds.sh` passes after
these edits.

**Out-of-tier changes (README rule 9):** decision 4 (a), complete with this gate. It is the measurement boundary for GPU
decode on every handler with a decode region once committed.

**Next for the executor:** item 6 (exact host KV growth, real FP16 host storage, exit criterion 6), then step 7. Item 6
changes the KV paths, so its A/B and hot-path gate need a baseline build of the tree before it (`git archive` of a commit):
decision 4 (a) is committed first.
