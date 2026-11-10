# Tier 03: KV cache maturity

Status: in progress (begun 2026-11-09 by owner decision while Tiers 02F and 02G stay open; README execution rule 1). Step 0
done 2026-11-09: on `Phi-3.1-mini-4k-instruct-Q4_K_M.gguf` (window 2047, every layer) the KV oracle passes at 3,000 tokens on
CPU and CUDA under both schedules, and teacher-forced NLL past position 2047 reads 2.634 with the window against 10.529 without
it (CPU; CUDA 2.634 against 10.524). Exit criterion 1 ticked (1 of 15). Next: step 1, the gather-based attention oracle tests.
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
   the execution record.)* At the start of this tier, ask the owner for the file (README rule 3,
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
- [ ] Kit probes (README execution rule 10): the **Add** list above is in the kit's catalog, each with its
      self-test and its `coverage.tsv` line, and the **Keep passing** list passes on this tier's build.
- [ ] Cross-surface checklist fully resolved, vision/LoRA-play exclusions explicit and tested.
- [ ] Perf gate published; `paged_vs_dense%` <= 2% at ctx 8192 / batch 8, defrag pause within budget.
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
