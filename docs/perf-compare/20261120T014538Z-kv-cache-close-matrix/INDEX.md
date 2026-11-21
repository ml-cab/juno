# KV cache tier close: the smoke matrix, the looser gates and the kit

**Purpose.** The KV cache tier's closing evidence on its final build: every earlier smoke script, unmodified, against
the tree; the hot-path gates that may be read unpinned (`compare-lora.sh`, `compare-vision.sh`, median of three
against the pre-tier jar); the tier's two microbench bars, unpinned; the new live check 12 (prompts shared across
requests); and the kit's Keep passing list with the tier's own probes. **Not pinned.** The standing throughput and
allocation gate and the microbench bars, pinned, are the owner's (`dist/kv-cache-close/`).

**Builds.**
- Baseline: the pre-tier jar, `git archive e16748ad` built with `mvn package -DskipTests`;
  `juno-player-0.1.2-shaded.jar` sha256 `21109a3a8fd317c8...`.
- Tree, HEAD `71099a85` (sections 1 to 3): sha256 `a14808b4914e2245...`.
- Tree, HEAD `71099a85` plus live check 12 and one reworded refusal message (sections 4 and 5, and the owner's
  candidate): sha256 `2ceed60ef9139b7a...`. Neither change is on a path sections 1 to 3 measure.

Host: the `docs/perf-compare/README.md` baseline host (Xeon E5-1650 v2, GTX 1080, driver 580.178.04). One JVM and one
GPU workload at a time.

## 1. Smoke matrix, scripts unmodified, tree HEAD `71099a85`

| Script | Lane | Checks | Result |
|---|---|---:|---|
| `smoke-kv-cache.sh` | CPU | 13 | PASS |
| `smoke-kv-cache.sh --gpu` | CUDA | 13 | PASS |
| `smoke-consistency.sh` | CPU and CUDA, static and continuous | 56 | PASS (19 shared-prefix hits on the repeated prompts) |
| `smoke-grammar.sh` | CPU, `--lora-play`, cluster launcher | 19 | PASS |
| `smoke-tools.sh` | CPU, `--lora-play` | 14 | PASS |
| `smoke-long-prompt-prefill.sh --gpu` | CUDA, both schedules, to 2,048 tokens | 36 | PASS |
| `smoke-long-prompt-prefill.sh --cpu --lengths "128 512"` | CPU | 24 | PASS |
| `smoke-packed-kquant-matmul.sh` | CUDA | 48 | PASS |
| `smoke-context-policy.sh --gpu` | CUDA, both schedules | 7 | PASS |
| `smoke-gpu-residency.sh` | CUDA, local and cluster | 13 | **FAIL, 1 of 13**: TinyLlama, residency on against off, greedy parts at the 32nd and last token (section 6); after owner decision 19 (a), 13 of 13 |
| `smoke-cpu-hot-path.sh` | CPU, 4 and 12 threads, vision | 31 | PASS |

## 2. Looser hot-path gates, median of three, A B A B A B against the pre-tier jar

`compare-lora.sh --gpu --skip-build --no-publish`, TinyLlama (bars: train <= 1.25x, playback >= 0.80x, recall):

| Jar | ms per pass (3 runs) | Playback t/s (3 runs) | Recall |
|---|---|---|---|
| baseline | 2,733 / 2,733 / 2,733 | 12.94 / 13.18 / 13.16 | "My name is Juno", 15 passes, loss 1.1797, each run |
| candidate | 2,733 / 2,733 / 2,733 | 11.49 / 11.49 / 11.53 | the same |
| **ratio** | **1.000x** | **0.874x** | equal |

`compare-vision.sh --skip-build --no-publish`, moondream2, `--prefill single` (bars: latency <= 1.25x, decode t/s >= 0.80x):

| Lane | Latency, median (baseline / candidate) | Decode t/s, median | Ratios | Reply |
|---|---|---|---|---|
| GPU | 155,834 / 162,264 ms | 4.887 / 4.756 | latency **1.041x**, t/s **0.973x** | 741 prompt tokens, HTTP 200, the same reply in all six runs |
| CPU | 150,342 / 153,576 ms | 5.024 / 4.906 | latency **1.022x**, t/s **0.976x** | the same |

Both gates pass. LoRA playback is 0.874x the pre-tier jar in every run (seven tokens, so a few milliseconds per token);
it stays above the 0.80x bar.

## 3. The tier's microbench bars, unpinned

| Bar | Reading | Result |
|---|---|---|
| Paged KV read in place: paged_vs_dense <= 2% at ctx 8,192, batch 8, page 16 ([report](gather-tax.md)) | -4.37% | PASS |
| Idle page reclaim: a pass adds <= 5 ms to an overlapping decode step at ctx 8,192, batch 8, page 16 ([report](kv-reclaim.md)) | at most 2.328 ms (TinyLlama); Qwen2.5-3B 0.752, Phi-3.5-mini 0.877, Mistral 7B 0.449 | PASS |

The owner's gate retakes both pinned, CPU governor `performance`, turbo off.

## 4. Live check 12 and the real-model smoke, tree with check 12

| Reading | Result |
|---|---|
| Check 12 alone, TinyLlama, CPU (`ModelLiveChecks.run`, suite `SHARED_PREFIX`) | PASS in 90 s: static and continuous, each 2 shared-prefix hits with sharing on and 0 with it off, 126 positions shared, prompt tokens 87/92/90, every reply equal token for token |
| Check 12 alone, TinyLlama, CUDA (one-position prefill windows) | PASS in 137 s: the same counts, replies equal |
| Check 12 with a planted defect (the pool shares one position fewer than it reports), CPU | FAIL, as it must: static, reply 2 parts from the unshared reply; the source restored, checksum equal |
| `./juno test --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | 12 of 12 in 268 s |
| `mvn -o verify -pl juno-master -am` (stub ITs) | PASS: `ThreeNodeClusterIT` 8, `TensorParallelClusterIT` 5, `InProcessClusterIT` 6, `UnsupportedArchitectureClusterIT` 1 |

## 5. Kit, tree with check 12

`./juno-kit.sh --yes --skip-build --models-dir models --probes <the 50 below>`: the tier's **Keep passing** list, its
**Add** probes, the live checks on the five sweep models on both backends, and the LoRA, vision, grammar, tools and
launcher probes. 8 h 42 min. Files: [`kit/SUMMARY.txt`](kit/SUMMARY.txt), `kit/host.json`, `kit/estimate.json`,
`kit/probes/*.json`.

| Probes | Result |
|---|---|
| Keep passing: `schedule.tinyllama`, `gpu.schedule.tinyllama` | PASS: 8 of 8 blocking, streams and turns on both schedules; idle KV pages reclaimed, replies unchanged; decode batch 8 |
| Keep passing: `mixed-prefill.tinyllama`, `gpu.mixed-prefill.tinyllama` | PASS: 12 prefill chunks in 9 decode steps, every stream answered |
| Keep passing: `long-prompt.*`, `gpu.long-prompt.*` (TinyLlama, Mistral 7B) | PASS: 128, 512 and 2,048 tokens, both schedules, streamed equals unstreamed |
| Keep passing: `context-policy.phi-3.5-mini`, `gpu.context-policy.phi-3.5-mini` | PASS: 4,239 positions against a 4,096 limit with the opt-in; the documented failure without it |
| Keep passing: `consistency.tinyllama`, `gpu.consistency.tinyllama` | PASS: batched equals single; no prefix-cache hit for stateless traffic; 19 shared-prefix hits on repeated prompts, answers equal; GPU decode batch of 8 equal to single at `f16`, `q8_0` and `q8_0`/`q4_0` |
| Keep passing: `cpu.unit-tests` | PASS: 2,888 tests, 0 failures, 33 skipped |
| Keep passing: `gpu.unit-tests` | **FAIL**: 452 tests, 3 failures, 12 skipped; the three below |
| Add: `kv-quality.*` (4 models) | PASS: `q8_0`/`q4_0` NLL 0.997x to 1.006x `f16` (bar <= 1.02x) |
| Add: `gpu.kv-quality.*` (4 models) | PASS: device `q8_0` 0.999x to 1.001x, `q8_0`/`q4_0` 0.997x to 1.004x `f16`; stored as selected, 544 and 416 bytes per position against 1,024 (TinyLlama) |
| Add: `session.save-restore.tinyllama-1.1b`, `gpu.session...` | PASS: `f16` and `q8_0`, 63 tokens restored, the second turn starts from them and equals the uninterrupted one |
| Add: `gpu.kv-footprint.*` (4 models) | readings (1 DONE, 3 NOISY: other processes used 6% to 7% of the CPU): the host KV is 2,112 positions, the request's 2,048 plus 64, on every model |
| Add: `gpu.kv-prefill.*` (4 models) | readings, NOISY (the same CPU guard): `q8_0` over `f16` device KV prefill 0.936x to 1.028x at 2,048-token windows |
| `cpu.live.*`, `gpu.live.*` (5 models each), `gpu.kernel-timing` | PASS: checks 1 to 10 each (check 12 runs outside the kit, section 4); longest kernel well under the 500 ms bound |
| `lora.tinyllama`, `gpu.lora.tinyllama`, `vision.moondream2`, `gpu.vision.moondream2`, `grammar.tinyllama`, `tools.qwen2.5-3b`, `launcher.tinyllama` | PASS |

41 pass, 8 readings, 1 fails. The three `gpu.unit-tests` failures, each re-run after the kit:

- `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance`: 91,029,504 bytes reported free with the
  device full against 67,108,864. Alone on this tree: fails, 89,063,424 bytes, twice. Alone on the pre-tier build
  (`e16748ad`, built from `git archive`): fails, 87,883,776 bytes. The GPU also drives the desktop (about 590 MiB in
  use); the test reads the driver's free figure with no KV code in between.
- `DeviceUploadOrderingTest.theFirstGemvAfterAnUploadReadsTheWholeMatrix`, 2 of 4 (Q5_K row 262, Q6_K row 183): the
  first GEMV after an upload read a different value than a GEMV once the device was idle. Alone on this tree: 4 of 4,
  twice. The test entered the tree during this tier (`6ad5c728`, the kit's upload-ordering finding), so the pre-tier build
  has none; this tier changed no file it exercises (`DeviceQ4KMatrix`, `CudaMatVec`, the bindings, the contexts, the
  GEMV kernel). It has failed 2 of 4 inside every full suite run since it was added.

Neither is loosened. Raised with the owner in the tier record.

## 6. A finding: residency on against off, TinyLlama, since the wide decode block

`smoke-gpu-residency.sh` fails its TinyLlama comparison: with the device-resident decode region off, the reply ends
"... his innovative technology"; with it on, "... technique". Mistral 7B and LLaMA-30B match over their 32 tokens,
the cluster runs equal local mode, and the memory checks pass. Bisected over the tier's commits, each built from
`git archive` ([readings](residency-bisect.txt)):

- the pre-tier jar and every commit up to `637751b0`: on equals off over 32 tokens;
- `9c33ab93` (the FP16 decode attention on the wide decode block) and every later commit: they part at the 32nd and
  last token; over 31 tokens they are equal;
- with one-position prefill windows on both servers the split stays, so it is not the window-split property recorded
  for GPU prefill.

Both paths reach the same wide decode entry. The region's documented difference from the op-at-a-time path is the
RMS norms' summation order (GPU against CPU), so the two were never bit for bit equal. The wide block sums attention in
another order and moved that difference far enough to flip a near tie at the last token. The smoke fails a TinyLlama
difference by design, while its header says the region is measured to keep every greedy token of a 24-position
decode. Raised with the owner in the tier record.

**After owner decision 19 (a)** (the smoke fails a TinyLlama difference only within the first 24 generated tokens,
`--hard-n-gen`, and still reports the 32-token comparison): `smoke-gpu-residency.sh` on the same jar, every model and
the cluster runs, **13 of 13 pass** ([summary](residency-24/summary-hard-n-gen-24.md)); TinyLlama equal over 24 tokens,
parting at the last word over 32, reported. With `--hard-n-gen 32` on TinyLlama alone the check fails, from word 21
([summary](residency-24/summary-hard-n-gen-32.md)).

## 7. Commands

```bash
# baseline jar
git archive e16748ad | tar -x -C <scratch>/e16748ad && (cd <scratch>/e16748ad && mvn -q package -DskipTests)

./scripts/performance-tests/smoke-kv-cache.sh                    # and --gpu
./scripts/performance-tests/smoke-consistency.sh
./scripts/performance-tests/smoke-grammar.sh
./scripts/performance-tests/smoke-tools.sh
./scripts/performance-tests/smoke-long-prompt-prefill.sh --gpu
./scripts/performance-tests/smoke-long-prompt-prefill.sh --cpu --lengths "128 512"
./scripts/performance-tests/smoke-packed-kquant-matmul.sh
./scripts/performance-tests/smoke-context-policy.sh --gpu
./scripts/performance-tests/smoke-gpu-residency.sh
./scripts/performance-tests/smoke-cpu-hot-path.sh

# A B A B A B, baseline from its own tree
./scripts/performance-tests/compare-lora.sh --gpu --skip-build --no-publish --out <dir>
./scripts/performance-tests/compare-vision.sh --gpu --skip-build --no-publish --out <dir>   # and --cpu

./scripts/performance-tests/gather-tax-microbench.sh <dir>
./scripts/performance-tests/kv-reclaim-microbench.sh <dir>

./juno test --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
mvn -o verify -pl juno-master -am -Dtest=NONE -Dsurefire.failIfNoSpecifiedTests=false

# residency bisect, per build
JUNO_JAR=<build>/juno-player/target/juno-player-0.1.2-shaded.jar \
  ./scripts/performance-tests/smoke-gpu-residency.sh --models tinyllama-1.1b-chat-v1.0.Q4_K_M --no-cluster
```
