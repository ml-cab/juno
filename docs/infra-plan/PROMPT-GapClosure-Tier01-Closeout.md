# Prompt — Gap-closure plan, Tier 01 close-out (owner decisions of 2026-09-27, three fixes, closure)

## Task

You are a senior performance engineer (JVM, Panama/FFM, CUDA, LLM inference) working in
`/home/medion/Repo/juno`. Tier 01 of the gap-closure plan ("GPU activation-residency redesign") is
implemented and measured; the previous session stopped at five owner decisions. The owner has now
decided all five. Your job is to **carry out those decisions and close Tier 01**, in this order, one
task at a time, each to its own exit condition, reporting after each:

1. **Record the owner's decisions in the plan tree** (docs only): Tier 01's two open exit boxes,
   Tier 01's Scope text, Tier 02's Scope and exit criteria, Tier 01B's hand-off. No code.
2. **Fix the per-request device-memory leak** in `CudaMatVec` (and its ROCm twin) - the one real
   serving bug found, and a blocker for Tier 01's own Threshold block.
3. **Fix the `-Pintegration` Maven profile** so `ModelLiveRunnerIT` actually runs.
4. **Tensor-parallel start failure on a model that does not fit N times:** make the error name the
   real cause, and record the limitation in Tier 09.
5. **Hand the harness's "one fast prefill repetition" question to Tier 01B** (docs only).
6. **Close Tier 01**: final verification, exit boxes, `Status:` line, report.

Stop and ask the owner at every point this prompt says "ask", and whenever a finding changes scope or
sequencing. Do not improvise past a decision point.

## Hard rules (binding; `CLAUDE.md` wins on any conflict)

- Read `CLAUDE.md` first. Git is **read-only** unless the owner's message names the operation (no
  `commit`, `add`, `push`, `reset`, `amend`, `stash`). `status`/`log`/`diff`/`show` are fine. The
  one git write the harness makes on its own - `compare-lora.sh --baseline <ref>` creates a temporary
  detached `git worktree` and removes it - was accepted last session; say in your report that you ran
  it.
- **Never** add Claude/Anthropic attribution anywhere (commits, PR text, code, comments, docs).
- No competitor product names (`llama.cpp`, `vLLM`, `llama-server`, `llama-bench`, `ggml`) in
  production docs, code, comments, CLI help or error messages. Allowed only in
  `docs/gap-closure-plan/`, `docs/infra-plan/`, `docs/perf-compare/` (filenames/run metadata). Say
  "the reference implementation" / "the reference tool" elsewhere.
- No internal tier numbers in code, javadoc, JFR descriptions, CLI help, error messages, script names
  or production docs (`docs/howto.md`, `docs/agent-arch.txt`, `docs/performance.md`, `README.md`,
  `CHANGELOG.md`). No emojis anywhere.
- Tests first: write the test, watch it fail **for the right reason**, then implement. Prefer a new
  class over extending an existing one. Hot path: no allocation or boxing per token.
- New flags must not silently no-op; the console front end turns library logging **off** unless
  `--verbose`, so a log-only notice is silent to a normal user (see `GpuResidencyOptions.consoleNotice`
  for the pattern used last session).
- `docs/gap-closure-plan/README.md` execution rules 1-9 are binding. Rule 9: every change outside Tier
  01's scope (tasks 2, 3, 4 here) is recorded in `TIER-01-gpu-activation-residency.md` under
  **"Out-of-tier changes"** (the table), naming what it touched, whether it is a measurement boundary,
  and which baselines it invalidates.
- **Never run a build (`mvn package/compile/install`) while a `mvn test` run or a harness run is in
  progress.** Read Maven results from `BUILD SUCCESS` / surefire reports, never from a pipeline's exit
  code. Run `mvn -o clean install -DskipTests` before any verification run.
- **A root `mvn clean` deletes `target/perf-compare/`, `target/gpu-residency-smoke/` and every other
  harness output under `target/`.** Copy anything you need to cite (or copy its figures into the
  execution record) before the next clean build. This lost two run directories last session.
- GPU checks are device-wide: run GPU tests and measurements on an idle device
  (`nvidia-smi --query-compute-apps=pid --format=csv,noheader` empty, and no
  `pgrep -f '[c]ab.ml.juno.node.NodeMain'`).
- When waiting on a process with `pgrep -f PATTERN`, bracket one character (`[c]ompare-lora.sh`) or
  the loop matches its own command line and never exits.
- If context runs low mid-task: stop, write a `docs/infra-plan/PROMPT-<short-name>.md` handoff in this
  same structure, and tell the owner.

## Environment

- Branch `67-gap-inference`. HEAD at hand-off: `cc94c53`, **plus a large uncommitted working tree**
  (Tier 01 steps 2, 3a and 4, and four out-of-tier follow-ups). Run `git status` / `git log --oneline
  -5` first; the owner may have committed it since. Never discard working-tree changes you did not
  make.
- GPU: 1x GTX 1080 (8 GiB, Pascal sm_61), CUDA 12.0. PTX compiled by hand and checked in (`nvcc -ptx
  -arch=compute_61 -O3 -o ../resources/cab/ml/juno/node/<name>.ptx <name>.cu` from
  `node/src/main/cuda/`). No ROCm hardware (mark ROCm rows `NEEDS-AMD-HARDWARE`).
- CPU: Xeon E5-1650 v2, 12 threads, AVX2.
- Models (never read into context, never modify) under `models/`: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`,
  `qwen2.5-3b-instruct-q4_k_m.gguf`, `Qwen3-1.7B-Q4_K_M.gguf`, `Phi-3.5-mini-instruct-Q4_K_M.gguf`,
  `mistral-7b-instruct-v0.1-q4_k_m.gguf`, `llama-1-30b.Q4_K_M.gguf`, and others.

## Commands and expected baselines (as of hand-off)

```bash
mvn -o clean install -DskipTests
mvn -o test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,registry,vision,metrics,juno-player
#   at hand-off: 1795 tests, 0 failures, 0 errors, 49 skipped, about 26 min
#   (registry 93, lora 116, kvcache 78, health 23, node 680/44, tokenizer 109/2, sampler 87,
#    coordinator 342/1, vision 95, metrics 61, juno-player 111/2)
mvn -o clean verify -pl juno-master          # 20 tests: InProcess 6, ThreeNode 8, TensorParallel 5, UnsupportedArch 1
mvn -o -pl node test -Dtest='ClassName'      # single class; GPU tests run by default on this host

# Live model IT - AS DOCUMENTED IT RUNS NOTHING (task 3 fixes this). Until then add -Dit.test:
mvn -o clean verify -pl juno-master -Pintegration -Dit.test=ModelLiveRunnerIT \
  -DMODELS=$PWD/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
#   at hand-off: tinyllama passes (pipeline + tensor); mistral-7b errors at tensor-parallel shard
#   loading, with or without --gpu-residency (task 4)

# Quick unpublished parity check (prompt 128/128, generated 64/64 must hold):
./scripts/performance-tests/compare-llama-cpp.sh --gpu --models tinyllama,qwen2.5-3b,Phi-3.5-mini,mistral-7b \
  --n-gen 64 --juno-reps 1 --juno-warmup 1 --reps 1 --no-publish --no-tuned-lane
#   read target/perf-compare/<stamp>/<model>-compare.json: .generation_parity, .prompt_parity

# Perf gates for a MatVec change (task 2):
./scripts/performance-tests/compare-lora.sh --gpu --reps 3 --baseline cc94c53   # train >= 0.95x, playback >= 0.80x
./scripts/performance-tests/compare-llama-cpp.sh --gpu --reps 3 --juno-reps 3 --juno-warmup 2   # publishes
#   GPU reference to read against: docs/perf-compare/20260927T091155Z/ (mistral-7b rows) plus
#   20260927T093054Z/ (tinyllama, qwen2.5-3b, Phi-3.5-mini rows). Re-run any row whose repetitions
#   spread more than 15% of their median (once; if it recurs, record it rather than loop).

# Residency smoke (per-request GPU memory, greedy on/off, cluster):
./scripts/performance-tests/smoke-gpu-residency.sh --requests 4
#   at hand-off: failures=0 on tinyllama, mistral-7b, llama-1-30b; GPU memory grows every request
#   with the region on AND off (task 2), about 23 MiB/request tinyllama, 114 MiB/request mistral-7b
```

## Read first

1. `CLAUDE.md`; `docs/gap-closure-plan/README.md` (execution rules 1-9, tier index, Program target).
2. `docs/gap-closure-plan/TIER-01-gpu-activation-residency.md`:
   - the top `Status:` line;
   - Scope (In scope items 1-4, Out of scope);
   - the **Threshold** block (around the "Tests to write/upgrade" section) - note its last bullet,
     "Device memory returns to its pre-request level after each request";
   - the execution-record sections dated **2026-09-27**, especially **"implementation step 3a
     (follow-up item 5): the decode region wired, and step 4"** and its **"Open for the owner"** list
     (items 1-5 there are the decisions below);
   - **"Out-of-tier changes"** (the table);
   - **Exit criteria** (two boxes still unticked).
3. `docs/gap-closure-plan/TIER-02-attention-long-context.md` (Scope, Implementation steps, Exit
   criteria), `TIER-01B-prefill-throughput.md` (Scope items 1-2, Implementation step 0),
   `TIER-09-tensor-parallelism-multi-gpu.md` (Scope).
4. The code named in each task.

## What exists (do not redo)

- **Residency primitive**: `ResidentChain` (one stream + buffers; `allocateScratch`), `ResidentActivation`
  (`upload`, `materialize`, `materializeRows` - several single-row activations with one wait),
  `KernelParams`, `RopeKernel`/`rope.cu`/`rope.ptx` (adjacent pairs only), `CudaRope`,
  `CudaRmsNorm.normalizeResident`.
- **Step 3a, wired**: `ResidentQkvPath` (per layer: upload, norm, Q8_1 quantize, three K-quant MMQ
  projections, RoPE, one download of q/k/v; device regions pooled, not per thread),
  `GpuResidencyOptions` (`--gpu-residency on|off|auto` / `JUNO_GPU_RESIDENCY`, **default off**;
  `announceUnsupported`, `consoleNotice`), wiring in `LlamaTransformerHandler.transformerLayer`
  (single-sequence decode only), `ConsoleMain`, `scripts/run.sh` (`local`, `cluster`),
  `ClusterHarness` (forwards the property), `compare-llama-cpp.sh --gpu-residency`.
- **Measured (step 4)**: region on vs the 2026-09-27 reference: tinyllama tg 1.054x, mistral-7b
  1.036x, qwen2.5-3b 0.990x and Phi-3.5-mini 0.994x (region declines there). Greedy output identical
  on/off over 32 tokens on tinyllama, mistral-7b, llama-1-30b. LoRA unaffected. Chain microbench:
  decode 0.26x the CPU chain (miss), prefill 1.002x / 0.995x (parity), chaining 0.52-0.58 / 0.55.
- **Out-of-tier follow-ups already landed and recorded**: Qwen RoPE split-half fix, `min_tokens`
  against turn markers, `RopeTable` (the last is the measurement boundary behind the 2026-09-27
  reference sweeps).

## The owner's decisions (2026-09-27)

| # | Question | Decision |
|---|---|---|
| 1 | Step 3b (KV append + attention inside the region) | **Not in Tier 01. Tier 02 owns it.** |
| 2 | Primitive-threshold exit box (decode width missed, prefill at parity, chaining met) | **Tick it**, with the contingency reading recorded: a partial result handed to Tier 01B item 2, not a downgrade. |
| 3 | "No longer dormant scaffolding" box | Owner delegated the choice between (a) "tick, CudaGraphSession explained-not-wired" and (c) "re-scope". **Decided: (c), re-scope to Tier 02** - rationale below. |
| 4 | `--gpu-residency` default | **Keep `off`.** Re-decide after 3b lands, on the larger re-measured gain. |
| 5 | The four pre-existing problems | **As suggested last session**: memory leak fixed now (task 2); `-Pintegration` fixed now (task 3); tensor-parallel start failure recorded in Tier 09 with a clearer error now (task 4); fast prefill repetition handed to Tier 01B (task 5). |

**Why (c) and not (a), from the plan tree's own rules.** The box reads: "either wired live
(preferred...), or the tier is explicitly marked **partial-complete** per the contingency above (not
silently marked complete with the scaffolding still dormant and unexplained)". It offers exactly two
outcomes; "explained, not wired" is neither, so ticking it under (a) would satisfy the box by
reinterpreting it. The owner has decided the tier is **not** a downgrade (decision 2), so
"partial-complete" is not available either. README execution rule 1 says half-wired features are not
acceptable stopping points, and rule 2 says "out of scope for this tier" is acceptable **only when
stated explicitly in the tier's Scope section**. Tier 01's Scope item 2 names `CudaGraphSession`
alongside `CudaRmsNorm`, so the honest fix is to amend the Scope: `CudaRmsNorm` is wired live (the
resident norm inside `ResidentQkvPath`), and `CudaGraphSession` moves to Tier 02 explicitly, with a
decision rule. Tier 02 is the right home on the merits too: a captured graph only pays when a region
issues many launches per wait, which is what attention inside the region (3b) creates; today's region
has five launches per layer, and the position argument that changes every token needs a device-side
read or an exec-node parameter update either way. **Not deletion**: `CudaGraphSession` and
`CudaGraphSessionTest` are tested code that Tier 02 would otherwise have to rebuild to evaluate;
Tier 02's rule below decides wire-or-delete on a measurement. No other tier file mentions
`CudaGraphSession` today, so nothing else needs re-pointing.

---

## Task 1 — Record the decisions in the plan tree (docs only)

Edit only files under `docs/gap-closure-plan/`. No production docs change in this task.

**1a. `TIER-01-gpu-activation-residency.md`, Scope.**
- In "In scope" item 2, keep the text, and add a dated note: `CudaRmsNorm` is wired live as the
  resident norm inside `ResidentQkvPath` (behind `--gpu-residency`); `CudaGraphSession` is **moved to
  Tier 02** by the owner on 2026-09-27 (link Tier 02's new scope item), because graph replay only
  pays once a region spans attention.
- In "Out of scope", add a bullet: `CudaGraphSession` graph capture/replay - deferred to Tier 02 with
  its decision rule; the code and `CudaGraphSessionTest` stay in place until then.
- Add a bullet: step 3b (KV append and attention inside the residency region) - Tier 02, per the
  owner, 2026-09-27.
- Add a bullet: making `--gpu-residency` default-on - re-decided after 3b is measured (Tier 02).

**1b. `TIER-01-gpu-activation-residency.md`, Exit criteria.**
- **Primitive-threshold box** ("RMSNorm + RoPE measured *faster* than CPU scalar..."): tick it. Its
  existing 2026-09-27 italic note already has the numbers; append: "Ticked by the owner on
  2026-09-27 under the contingency: decode width missed, prefill width at parity, chaining met; a
  partial result handed to Tier 01B item 2 (see that tier), not a downgrade. The wired region is
  +3.6% to +5.4% end to end."
- **Dormant-scaffolding box**: tick it with a note: `CudaRmsNorm` wired live (resident norm in
  `ResidentQkvPath`, behind `--gpu-residency`, default off by owner decision); `CudaGraphSession`
  explicitly re-scoped to Tier 02 in this tier's Scope section by the owner (rule 2), with a
  wire-or-delete decision rule there - so neither is dormant and unexplained.
- Do **not** touch the other boxes yet; task 6 re-verifies them.

**1c. `TIER-01-gpu-activation-residency.md`, execution record.** Add a short dated section "2026-09-2X
- owner decisions on the step 3a pass" listing the five decisions verbatim (table above) and the
(c) rationale in two or three sentences. Update the "Open for the owner" list of the step 3a section
by marking each item "decided 2026-09-27: ..." rather than deleting it.

**1d. `TIER-02-attention-long-context.md`.** Add to "In scope" (renumber nothing that already exists;
append):
- **4. Attention inside the decode residency region (Tier 01's "step 3b").** Extend
  `ResidentQkvPath` (Tier 01) so that, per layer at single-sequence decode, k and v are appended to the
  device KV mirror (`DeviceKvCache`) without a host round trip and attention (`CudaGqaAttention`) reads
  q from the region, downloading only the attention output. Today, after the region, the layer still
  pays: FP16 conversion on the host plus two synchronous copies for the KV append, four synchronous
  uploads plus one download for attention. Keep the CPU KV tensors written (they are the fallback and
  the source of truth for the mirror); the order and failure handling of `DeviceKvCache.appendToken`
  and the written-prefix watermark (`c91f879`) must hold. Scope and fallbacks as in Tier 01:
  single-sequence decode, CUDA, K-quant device projections, adjacent RoPE, no Q/K/V bias; announce
  (log **and** console) everything else. Tests: bit-identity of the region against the op-at-a-time
  path; greedy parity on/off on tinyllama, mistral-7b, llama-1-30b; per-request device memory flat.
- **5. `CudaGraphSession` (moved from Tier 01 by the owner on 2026-09-27).** After item 4 lands,
  measure whether capturing one layer's region as a CUDA graph and replaying it (position read from
  device memory, or updated per step via an exec-node parameter update) beats plain launches on the
  GTX 1080 at decode width. **Decision rule**: wire it behind the same flag if it saves at least 5% of
  decode forward-pass time on tinyllama and mistral-7b with greedy output unchanged; otherwise delete
  `CudaGraphSession` and `CudaGraphSessionTest` and record the measurement. Either way the class stops
  being dormant in this tier.
- **6. Default of `--gpu-residency`.** After item 4 (and 5 if wired), re-measure the region on vs off
  on all four sweep models (full `compare-llama-cpp.sh --gpu` sweeps, both published) and put the
  default to the owner with the numbers; do not change it unasked.
Add matching exit-criteria boxes for items 4-6 (with a numeric **Threshold**: item 4 end-to-end tg
>= 1.0x of the region-off run on every model where it runs, and >= 0.95x everywhere; item 5 as its
decision rule), matching "Tests to write/upgrade" bullets, and add `ResidentQkvPath` /
`GpuResidencyOptions` / `LlamaTransformerHandlerGpuResidencyTest` / `ResidentQkvPathTest` /
`smoke-gpu-residency.sh` to the files/tests Tier 02 names. Mention in its "Why this tier, why now" or
Scope preamble that Tier 01 shipped the region through RoPE and why the attention half is here.

**1e. `TIER-01B-prefill-throughput.md`.** Under Scope item 2 ("Stop staging the activation batch to
host between every matmul"), add a dated note: Tier 01 closed with the primitive-threshold reading
"decode width missed (0.26x the CPU chain), prefill width at parity (1.002x / 0.995x), chaining met
(0.55 at prefill)"; per Tier 01's contingency this item **proceeds on the prefill result**; the
evidence is `docs/perf-compare/20260927T115107Z-tier01-resident-chain/`; the device-only lane
(0.157 ms against 2.13 ms on the CPU at batch 512, 13.6x) is the case this item rests on; and
`ResidentQkvPath` is the working decode-width template (chain-owned scratch, pooled regions, one wait
per region).

**Exit condition.** The five plan files read consistently; no production doc touched; no competitor
names or tier numbers leaked outside `docs/gap-closure-plan/`. Report the diff summary.

## Task 2 — Fix the per-request device-memory leak (out-of-tier; blocks Tier 01's Threshold)

**What it is.** `RequestScheduler` starts one new virtual thread per request
(`Thread.ofVirtual().name("gen-" + request.requestId())`, and `"batch-gen-"` for static batches).
`CudaMatVec` keeps its device scratch in `ThreadLocal`s - `FP32_SCRATCH`, `FP16_SCRATCH`,
`Q4K_DEQUANT_SCRATCH` (can be over 100 MB: a whole weight matrix dequantized to FP16) and
`CUDA_STREAM` (a CUDA stream per thread). Each request therefore allocates a fresh set that is never
released, since a `ThreadLocal` on a finished thread is never read again and nothing frees device
memory on GC. `RocmMatVec` has the same pattern (`FP32_SCRATCH`, `FP16_SCRATCH`, `HIP_STREAM`).
Also check `CudaGqaAttention.SCRATCH` and `CudaRmsNorm.SCRATCH` (both `ThreadLocal`, device memory
inside?) - fix every one that holds device memory or a stream. Measured (smoke, local mode, three
in-process nodes, `nvidia-smi` per process): +23 MiB per request on tinyllama, +114 MiB per request
on mistral-7b, identical with `--gpu-residency` on or off. On an 8 GiB card a mistral-7b server runs
out of device memory after a few dozen requests.

**Why it belongs here.** It violates Tier 01's own Threshold bullet "Device memory returns to its
pre-request level after each request... asserted via `GpuBindings.memGetInfo`", which Tier 01 cannot
close without. It is not residency work, so it is recorded as an out-of-tier change (rule 9).

**Fix (design; confirm by reading the code first).** Replace per-thread device state with pooled
state whose size is bounded by concurrent callers, not by threads ever seen - the pattern
`ResidentQkvPath` already uses (a `ConcurrentLinkedQueue` of free entries: poll or create, use, offer
back in `finally`). Constraints:
- A scratch entry and its stream must not be used by two callers at once. Most `CudaMatVec` methods
  already run under `ctx.cublasSerializationLock()`; confirm whether that lock alone makes a single
  shared scratch per `CudaMatVec`/context safe (it serializes everything issued, so one pooled entry
  would be enough), or whether any path issues device work outside the lock. Prefer the simplest
  correct design; state which you chose and why.
- Scratch grows on demand today (`ensureFp32Scratch` etc. reallocate when a bigger size is needed);
  keep that, but on a pooled entry. Respect `DeviceScratchBudget` (landed in `c91f879`) - read it and
  keep its accounting correct.
- Streams: a pooled entry owns one stream for its lifetime; destroy it only when the entry is freed.
- Provide a way to release pooled device memory when a `CudaMatVec` / context is closed (tests need
  the device back to baseline); do not add a per-token cost.
- Hot path: no allocation per call beyond what exists today; no boxing.

**Tests first** (`node`, GPU-tagged like the existing ones; run on an idle device):
- A `CudaMatVec` test that calls `sgemv`/`sgemvSameX`/`sgemm` (Q4_K, FP16 and FP32 variants) from
  50 short-lived threads in sequence and asserts the scratch count or device bytes held stays at one
  entry's worth. It must **fail on current code** (count grows by one per thread). Expose a
  package-private accessor for the pooled bytes, as `ResidentQkvPath.deviceBytes()` does.
- The same for the dequant scratch path used by batched prefill (`Q4KDequantScratch`).
- A device-wide `memGetInfo` check across many short-lived-thread cycles with a bound calibrated the
  way `ResidentQkvPathTest.noDeviceMemoryRetained` was: measure the leak-free drift, plant a leak once
  to show the test fails, then set the bound between the two and write both numbers in the comment.
- Concurrency: 3-4 threads at once still produce bit-identical results to the single-thread path.
- ROCm: same change to `RocmMatVec`; no hardware - unit-level only, mark `NEEDS-AMD-HARDWARE`.

**Verification (MatVec change => perf gates required; README rule 9 + CLAUDE.md).**
- `smoke-gpu-residency.sh --requests 8`: GPU memory after request 2 equals memory after request 8
  (or within a stated few MiB) with the region **off** and **on**, on all three models. Consider
  tightening the script's check from "on grows no more than off" to "neither grows" now that the
  baseline should be flat; if you tighten it, say so in `docs/howto.md`'s smoke section.
- Quick parity check 64/64, 128/128.
- `compare-lora.sh --gpu --reps 3 --baseline cc94c53`: train >= 0.95x, playback >= 0.80x.
- `compare-llama-cpp.sh --gpu --reps 3 --juno-reps 3 --juno-warmup 2`, published: tg and pp within
  0.95x of the 2026-09-27 reference on every sweep model (expect flat or slightly better). Decide from
  the numbers whether this is a measurement boundary (it should not be unless throughput moves
  beyond the spread); say so in the out-of-tier row.
- Full 11-module `mvn -o test`; `mvn -o clean verify -pl juno-master`.

**Docs.** Out-of-tier row in Tier 01; `docs/agent-arch.txt` (`CudaMatVec` / `RocmMatVec` scratch
lifetime); `CHANGELOG.md` new session entry ("device memory no longer grows with every request"
with the before/after MiB figures); `docs/performance.md` only if a published figure moved.

**Exit condition.** The leak test failed first and passes now; smoke shows flat per-request memory on
all three models; gates pass and are published; row recorded.

## Task 3 — Make `-Pintegration` run `ModelLiveRunnerIT` (out-of-tier, small)

**What it is.** `juno-master/pom.xml`: the default `maven-failsafe-plugin` configuration excludes
`ModelLiveRunnerIT` (and `GpuForwardPassIT`); the `integration` profile sets `<excludes />` plus an
`<include>`, but Maven merges configuration elements, so the empty `<excludes />` does not clear the
parent's list and failsafe runs **no test at all** while reporting BUILD SUCCESS. The documented
command in `CLAUDE.md` (`mvn verify -pl juno-master -Pintegration -DMODELS=...`) is therefore a
silent no-op. The `gpu` profile has the same shape (`**/GpuForwardPassIT.java`) - check and fix it
too.

**Fix.** Use `combine.self="override"` on the profile's `<excludes>`/`<includes>` (or restructure the
default config so exclusions live in the default execution only). Verify with `mvn -o help:effective-pom
-pl juno-master -Pintegration` that the effective failsafe config includes `ModelLiveRunnerIT` and
excludes nothing it needs.

**Tests / verification.** No unit test is sensible for a pom; the check is behavioural and must be
shown both ways:
- Before the fix: `mvn -o clean verify -pl juno-master -Pintegration -DMODELS=<tinyllama>` shows no
  `Running cab.ml.juno.master.ModelLiveRunnerIT` line.
- After: the same command runs it (`Tests run: 1`, tinyllama passes).
- The default `mvn -o clean verify -pl juno-master` still runs exactly the 20 stub ITs and not
  `ModelLiveRunnerIT` or `GpuForwardPassIT`.
- If you fix the `gpu` profile: show it runs `GpuForwardPassIT` on this host.

**Docs.** Out-of-tier row (not a measurement boundary); `CLAUDE.md` needs no change if the documented
command now works - confirm it does, verbatim. `CHANGELOG.md` one bullet.

## Task 4 — Tensor-parallel start on a model that does not fit N times (out-of-tier, small; Tier 09 owns the real fix)

**What it is.** Tensor-parallel mode is "geometry only" (`TensorShardContext` javadoc): every node
loads and runs the **full** model, and the coordinator sums full logit vectors. With three nodes on
one 8 GiB card, mistral-7b (about 4 GB) fails at shard loading, and the coordinator reports only
`RuntimeException: Tensor-parallel shard loading failed: UNKNOWN: Application error processing RPC`
(`TensorParallelPipelineClient.awaitAll`, ~293). It fails the same with `--gpu-residency` on or off.

**Do now.**
- Find where the node-side exception is lost (the gRPC server handler in `juno-node` /
  `EmbeddedNodeServer`, and `TensorParallelPipelineClient`), and make the coordinator's error name the
  node, the model and the node's actual cause (out of device memory, out of heap, etc.), using a gRPC
  status with a description rather than `UNKNOWN`. Fail closed, no retry change.
- Test first: a unit/integration test with a stub node whose shard load throws a known exception,
  asserting the coordinator's message contains that cause (use the existing stub-mode cluster test
  infrastructure in `juno-master`'s ITs or `juno-player` tests). It must fail on current code.
- Do **not** add GPU-memory pre-flight estimation or change placement; that is Tier 09's design.

**Record.** In `TIER-09-tensor-parallelism-multi-gpu.md`, add a dated note under Scope item 1 (or a
"Known limitation found before this tier" paragraph): mistral-7b cannot start in tensor mode on a
single 8 GiB card because every node loads the whole model; observed by `ModelLiveRunnerIT` on
2026-09-27 with and without `--gpu-residency`; real weight slicing (this tier's item 1) is what fixes
it, and its parity test should include a model that only fits when sliced. Out-of-tier row in Tier 01
for the error-message change (not a measurement boundary). `CHANGELOG.md` one bullet.

**Exit condition.** The new test failed first and passes; `ModelLiveRunnerIT` on mistral-7b now
fails with a message that names the real cause (show it), which is the expected outcome until Tier 09.

## Task 5 — Hand the fast-prefill-repetition question to Tier 01B (docs only)

**What it is.** In `compare-llama-cpp.sh` sweeps, one of three prefill repetitions sometimes reads
four to five times faster than the others: the tinyllama default lane read 898 then 819 t/s in two
separate runs (others about 168), and the qwen2.5-3b tuned lane 130.6 against about 80. Medians are
unaffected (median of three), but the row breaks the 15% spread rule and forces re-runs. Suspects, not
verified: a prefix-cache or KV reuse between the warm-up and the measured request, or a timing boundary
in how the prefill lane reads `prompt_eval_tps`. Evidence: `docs/perf-compare/20260927T091155Z/` and
`20260927T093054Z/` (INDEX banners; per-rep `*-prefill-rep*-juno.json`).

**Do.** In `TIER-01B-prefill-throughput.md`, Implementation step 0 ("Ship the two script fixes before
the first sweep"), add a third item: investigate and fix the fast prefill repetition before step 1's
re-baseline, because that re-baseline is this tier's before-measurement and must not need re-runs for
a harness artifact; include the evidence paths above and the two suspects. Add a matching bullet to its
"Tests to write/upgrade" list (a harness selftest or a live check that every prefill repetition's
prompt tokens are fully prefilled - e.g. assert the prefill lane's measured request reports no reused
prefix). Do not change the harness in this task.

## Task 6 — Close Tier 01

- `mvn -o clean install -DskipTests`; full 11-module `mvn -o test`; `mvn -o clean verify -pl juno-master`;
  the fixed `-Pintegration` command on tinyllama. Record counts.
- Re-read every Tier 01 exit box against the evidence (the step 3a section and tasks 1-5 of this
  prompt). All should now be ticked; if any is not supportable, **ask** rather than tick.
- The Threshold block's device-memory bullet is now satisfied by task 2 on the default path as well
  as the residency path - cite the smoke numbers.
- Update Tier 01's `Status:` line to **complete** with the date and a one-line summary (region wired
  behind a default-off flag, +3.6% to +5.4% decode where it runs; primitive threshold closed under the
  contingency; 3b, `CudaGraphSession` and the flag default moved to Tier 02). Per README rule 1 the
  next tier in the running order is **Tier 01B**; say so, and do not start it.
- `CHANGELOG.md`: make sure Sessions 95-96 plus this pass's entry cover everything; no tier numbers.
- Final report: tasks done, drift found, files changed, commands and pass/fail with real output,
  boxes ticked, anything left for the owner, exact next step. List uncommitted files. No commit unless
  the owner names it.

## Exit checklist

- [ ] Task 1: Tier 01 Scope amended (CudaGraphSession, 3b, flag default -> Tier 02); both open boxes
      ticked with notes; decisions section written; Tier 02 items 4-6 with thresholds, tests, exit
      boxes; Tier 01B item 2 hand-off note.
- [ ] Task 2: leak test failed first; pooled scratch in `CudaMatVec` and `RocmMatVec` (and any other
      device-holding `ThreadLocal`); smoke flat on three models; parity, LoRA and GPU sweep gates
      published; out-of-tier row; docs and CHANGELOG.
- [ ] Task 3: `-Pintegration` shown running nothing before and `ModelLiveRunnerIT` after; default
      verify still 20 ITs; row; CHANGELOG.
- [ ] Task 4: error names the node's real cause (test failed first); Tier 09 note; row; CHANGELOG.
- [ ] Task 5: Tier 01B step 0 item and test bullet added.
- [ ] Task 6: full suite, juno-master, live IT green; all Tier 01 boxes ticked or escalated; `Status:`
      complete; final report.
