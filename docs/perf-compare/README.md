# llama.cpp vs Juno — local compare

Baselines on **medion-Precision-T3610** · **Intel Xeon E5-1650 v2** (12 threads) · **62.7 GiB RAM** · **NVIDIA GeForce GTX 1080 (8 GiB)**.

Workload for both backends: `n_prompt=128`, `n_gen=64`, `reps=1`, temperature 0, Juno `--vector 0` (scalar).

Juno metrics use **JFR by default** (`--jfr 30m`): `TokenProduced.tps` for decode tg; pp from `ForwardPass.prefill.total_ms` when present, else `(API latency − decode total_ms)`.

| Run | Backend | Juno metrics | Artifacts |
|-----|---------|--------------|-----------|
| [`20260831T230258Z`](20260831T230258Z/) | CPU (`-ngl 0` / `--cpu`) | JFR pp/tg | [INDEX](20260831T230258Z/INDEX.md) |
| [`20260831T231403Z`](20260831T231403Z/) | GPU (`-ngl 99` / `--gpu`) | JFR pp/tg | [INDEX](20260831T231403Z/INDEX.md) |
| [`20260901T032753Z`](20260901T032753Z/) | GPU + Tier 5 (`JUNO_GPU_LAYERS=auto`) | JFR pp/tg | [INDEX](20260901T032753Z/INDEX.md) |
| [`20260901T154735Z-parallel`](20260901T154735Z-parallel/) | GPU multi-session static batch (`--parallel` 1 vs 8) | aggregate tg | [INDEX](20260901T154735Z-parallel/INDEX.md) |
| [`20260901T155136Z-parallel`](20260901T155136Z-parallel/) | CPU multi-session static batch (`--parallel` 1 vs 8) | aggregate tg | [INDEX](20260901T155136Z-parallel/INDEX.md) |
| [`20260901T173121Z-parallel`](20260901T173121Z-parallel/) | GPU multi-session static batch (`--parallel` 1 vs 8) | aggregate tg | [INDEX](20260901T173121Z-parallel/INDEX.md) |
| [`20260901T234024Z-prefill`](20260901T234024Z-prefill/) | CPU prefill microbatch (`--prefill-batch` 1 vs 32) | JFR pp | [INDEX](20260901T234024Z-prefill/INDEX.md) |
| [`20260902T200210Z-lora`](20260902T200210Z-lora/) | GPU LoRA train-qa + playback (`compare-lora.sh`) | train ms / playback tps | [INDEX](20260902T200210Z-lora/INDEX.md) |
| [`20260905T031520Z-lora`](20260905T031520Z-lora/) | GPU LoRA train-qa + playback (`compare-lora.sh`) | train ms / playback tps | [INDEX](20260905T031520Z-lora/INDEX.md) |
| [`20260904T141315Z-vision`](20260904T141315Z-vision/) | GPU vision chat (`compare-vision.sh`, `47-vision`) | latency / decode tps | [INDEX](20260904T141315Z-vision/INDEX.md) |
| [`20260904T194612Z`](20260904T194612Z/) | CPU Vector SIMD (`--vector 0`) | JFR pp/tg | [INDEX](20260904T194612Z/INDEX.md) |
| [`20260904T195731Z`](20260904T195731Z/) | CPU Vector SIMD (`--vector 1`) | JFR pp/tg | [INDEX](20260904T195731Z/INDEX.md) |
| [`20260910T025804Z`](20260910T025804Z/) | GPU + fused Q4_K MMQ (`JUNO_MMQ=on` / `-DJUNO_MMQ=on`) | JFR pp/tg | [INDEX](20260910T025804Z/INDEX.md) |
| [`20260910T030058Z-lora`](20260910T030058Z-lora/) | GPU LoRA train-qa + playback (`compare-lora.sh`) | train ms / playback tps | [INDEX](20260910T030058Z-lora/INDEX.md) |
| [`20260910T170557Z`](20260910T170557Z/) | GPU default path (post–quantized KV landing, `--cache-type` default `f16`) | JFR pp/tg | [INDEX](20260910T170557Z/INDEX.md) |
| [`20260910T180703Z-lora`](20260910T180703Z-lora/) | GPU LoRA train-qa + playback (`compare-lora.sh`, wall tps gate) | train ms / playback tps | [INDEX](20260910T180703Z-lora/INDEX.md) |
| [`20260910T213121Z-gather-tax.md`](20260910T213121Z-gather-tax.md) | CPU gather-tax (pre page-bulk F16) | gather % of attn | markdown report |
| [`20260910T214300Z-gather-tax.md`](20260910T214300Z-gather-tax.md) | CPU gather-tax (post page-bulk F16; gate PASS) | gather % of attn | markdown report |
| [`20260910T221031Z-lora`](20260910T221031Z-lora/) | GPU LoRA train-qa + playback (post block-KV dual path) | train ms / playback tps | [INDEX](20260910T221031Z-lora/INDEX.md) |
| [`20260910T222026Z`](20260910T222026Z/) | GPU default path (post block-KV dual path, `--schedule` default static) | JFR pp/tg | [INDEX](20260910T222026Z/INDEX.md) |
| [`20260911T194430Z-continuous`](20260911T194430Z-continuous/) | GPU continuous vs static (TPS / SSE TTFT-TPOT / prefix) | agg tg + JFR ContinuousStep | [INDEX](20260911T194430Z-continuous/INDEX.md) |
| [`20260911T195008Z`](20260911T195008Z/) | GPU default path (post continuous landing, `--schedule` default static) | JFR pp/tg | [INDEX](20260911T195008Z/INDEX.md) |
| [`20260911T195711Z-lora`](20260911T195711Z-lora/) | GPU LoRA train-qa + playback (post continuous) | train ms / playback tps | [INDEX](20260911T195711Z-lora/INDEX.md) |
| [`20260911T204721Z-mixed-prefill`](20260911T204721Z-mixed-prefill/) | GPU mixed chunked prefill vs admit-time (long+short SSE) | short TTFT/TPOT + JFR prefill_chunks | [INDEX](20260911T204721Z-mixed-prefill/INDEX.md) |
| [`20260911T204900Z`](20260911T204900Z/) | GPU default path (post mixed-prefill landing) | JFR pp/tg | [INDEX](20260911T204900Z/INDEX.md) |
| [`20260911T205447Z-lora`](20260911T205447Z-lora/) | GPU LoRA train-qa + playback (post mixed-prefill) | train ms / playback tps | [INDEX](20260911T205447Z-lora/INDEX.md) |
| [`20260911T221215Z`](20260911T221215Z/) | CPU default path (post OpenAI field parity; `--vector 0`) | JFR pp/tg | [INDEX](20260911T221215Z/INDEX.md) |
| [`20260911T235203Z`](20260911T235203Z/) | GPU + Q8_1/`dp4a` K-quant GEMV (`--mmq on --gpu-layers auto --vector 0`) | JFR pp/tg | [INDEX](20260911T235203Z/INDEX.md) |
| [`20260911T235353Z`](20260911T235353Z/) | GPU Phi-3.5 `--mmq off` (FP16-resident pair) | JFR pp/tg | [INDEX](20260911T235353Z/INDEX.md) |
| [`20260911T235455Z-lora`](20260911T235455Z-lora/) | GPU LoRA train-qa + playback (post Q8_1/`dp4a` GEMV) | train ms / playback tps | [INDEX](20260911T235455Z-lora/INDEX.md) |
| [`20260912T193402Z`](20260912T193402Z/) | CPU default path (post constrained decoding; `--vector 0`) | JFR pp/tg | [INDEX](20260912T193402Z/INDEX.md) |
| [`20260913T032734Z`](20260913T032734Z/) | CPU default path (post function calling; `--vector 0`) | JFR pp/tg | [INDEX](20260913T032734Z/INDEX.md) |
| [`20260914T220204Z`](20260914T220204Z/) | CPU default path (post embeddings API; `--vector 0`, `--no-jfr`) | wall-clock tg | [INDEX](20260914T220204Z/INDEX.md) |
| [`20260915T041705Z`](20260915T041705Z/) | GPU Tier 17 batched-prefill GEMM, `--mmq off` | JFR pp/tg | [INDEX](20260915T041705Z/INDEX.md) |
| [`20260915T042207Z`](20260915T042207Z/) | GPU Tier 17 batched-prefill GEMM, `--mmq on` | JFR pp/tg | [INDEX](20260915T042207Z/INDEX.md) |
| [`20260915T042421Z`](20260915T042421Z/) | GPU Tier 17, `--raw-prompt --n-prompt 128` (token-count-matched) | JFR pp/tg | [INDEX](20260915T042421Z/INDEX.md) |
| [`20260915T043143Z`](20260915T043143Z/) | GPU Tier 17, `JUNO_PREFILL_BATCH=512 --raw-prompt --n-prompt 512` | JFR pp/tg | [INDEX](20260915T043143Z/INDEX.md) |
| [`20260915T190157Z-lora`](20260915T190157Z-lora/) | GPU LoRA train-qa + playback (post Tier 17, expected flat) | train ms / playback tps | [INDEX](20260915T190157Z-lora/INDEX.md) |
| [`20260915T223032Z`](20260915T223032Z/) | GPU default 4-model sweep + standing Mistral-7B tuned lane (`--mmq on --gpu-layers auto`, auto-added by `compare-llama-cpp.sh` whenever mistral-7b is selected on GPU) | JFR pp/tg | [INDEX](20260915T223032Z/INDEX.md) |

Earlier runs (API wall-clock tg only, no JFR): [`20260831T214609Z`](20260831T214609Z/) (CPU), [`20260831T223850Z`](20260831T223850Z/) (GPU).

## `historyArr` fix measurement — 2026-09-15 (CPU, `compare-schedule.sh`)

`docs/infra-plan/PLAN-Infra-Review-Fixes.md` item 4 replaced
`ContinuousBatchEngine`/`GenerationLoop`'s per-decode-step
`generated.stream().mapToInt(Integer::intValue).toArray()` (boxed `List<Integer>` traversal,
repeated every step for every active slot) with an incrementally-appended `GrowableIntArray`.
Measured a genuine before/after rather than assuming a win: `git stash` isolated exactly the 3
files this fix touches (`ContinuousBatchEngine.java`, `GenerationLoop.java`,
`GrowableIntArray.java`), rebuilt, ran `compare-schedule.sh --cpu --mode tps --sessions 8
--max-tokens 64` against the pre-fix jar, restored the fix, rebuilt, ran the identical command
again.

| Schedule | Before (agg tg t/s) | After (agg tg t/s) | Delta |
|---|---:|---:|---:|
| static | 3.7309 | 3.6607 | -1.9% |
| continuous | 3.6863 | 3.6742 | -0.3% |

**No measurable win at this scale.** Both deltas are smaller than the run-to-run noise this same
investigation's [harness noise section](#harness-noise-investigation--2026-09-15) above found on
this host (unpinned GPU clocks, `schedutil` CPU governor, background load ~4-5) — i.e. this is
consistent with pure measurement noise, not a real regression from the fix. The fix is still
correct and worth keeping (eliminates real per-step boxing/unboxing and stream-pipeline overhead
verified via `mvn test`), but at 64 generated tokens × 8 concurrent sessions on a 1.1B model, the
O(n) history-array rebuild simply isn't large enough to dominate anything measurable. A longer
generation length or higher concurrency would be needed to see whether the reconstruction cost
actually compounds the way the original hypothesis suggested — not tested here, reported honestly
rather than claimed.

## Harness noise investigation — 2026-09-15

`docs/infra-plan/PLAN-Infra-Review-Fixes.md` item 7 flagged a 9x swing in the llama.cpp reference
tg number for the identical tinyllama Q4_K_M CPU config across two sessions ~13 hours apart, on a
run where nothing Juno-side changed — despite `compare-llama-cpp.sh` already averaging 3 reps
internally (`llama-bench -r 3`). Checked this host directly rather than guessing at causes:

- **GPU persistence mode is `Disabled`** (`nvidia-smi -q -d PERFORMANCE`). Without persistence
  mode, the NVIDIA driver can let the GPU drop to a low-power state (observed: `pstate P2`,
  SM clock 1607 MHz vs. a 1911 MHz max) between invocations, so a freshly-started process pays a
  clock ramp-up cost the previous run's warm GPU didn't — a real, well-documented source of
  cross-run GPU variance, and one `nvidia-smi -pm 1` (as root) would eliminate.
- **3614s of accumulated SW power-capping time** (`Clocks Event Reasons Counters` → `SW Power
  Capping`), not active at the moment checked but a sign the card throttles under sustained load
  on this box — plausible additional variance for longer runs.
- **CPU governor is `schedutil`** (dynamic frequency scaling), not a fixed/performance governor —
  affects the CPU-side llama.cpp reference number specifically, which is exactly the number that
  swung 9x.
- **Background load average ~4-5** on a 12-thread host at the time of this check — consistent
  with the review's own characterization of this as "a shared, unisolated dev box," not an
  idle benchmark rig.

None of this proves which factor caused the specific 9x swing in sessions 79/80 (that host state
wasn't captured at the time), but all four are real, verified conditions on this box today that
would independently degrade repeatability. Recommended before trusting tight ratio gates further:
`nvidia-smi -pm 1` to enable persistence mode, pin the CPU governor to `performance` for benchmark
runs, and check `uptime`/`nvidia-smi` for contending load before publishing a bake-off intended as
a baseline. `compare-lora.sh --reps N` (added alongside this investigation, see item 7) now takes
the median across N repeated train+playback cycles for exactly this reason — a single-shot outlier
no longer becomes the published number.

## Standing Mistral-7B tuned lane — `20260915T223032Z`

`compare-llama-cpp.sh --gpu` now automatically adds a second Mistral-7B row (`*-tuned`,
`--mmq on --gpu-layers auto`) whenever mistral-7b is among the selected GPU models — alongside
the existing vanilla-default row, not instead of it (`--no-mistral-tuned-lane` opts out). See
`docs/infra-plan/PLAN-Infra-Review-Fixes.md` item 8: on this 8 GiB card, Mistral-7B's
default-flags lane (`--mmq off`, `--gpu-layers` unset) has consistently measured worse than
Juno's own CPU numbers for smaller models across many prior sessions, while the tuned
configuration nobody was actually exercising in the standing regression gate is ~30-40x faster.
This run (`n_prompt=128`, `n_gen=64`, `reps=3`, full default 4-model set) makes both lanes
visible in the same sweep going forward:

| Lane | Juno tg (JFR) | Juno/llama.cpp tg ratio |
|---|---:|---:|
| default (`--mmq off`, `--gpu-layers` unset) | 0.479 t/s | 0.0136× |
| tuned (`--mmq on --gpu-layers auto`) | 15.65 t/s | 0.444× |

Tuned/default ≈ **32.7×**, consistent with the session-78 finding this lane was added to track.
The tuned ratio clears the P0 mistral-7b gate (≥0.15×) with real margin; the default lane remains
representative of what a user gets who doesn't know `--mmq`/`--gpu-layers` exist, which is the
gap this standing lane exists to keep visible rather than let the sweep quietly measure only the
unrepresentative default going forward.

## GPU batched-prefill GEMM bake-off — Tier 17

`CudaMatVec.sgemm(DeviceHalfMatrix|DeviceQ4KMatrix, float[][])` now uses a real tiled GEMM
(`cublasGemmEx`) for prefill-sized batches (`> HALF_SGEMM_BATCH_MAX = 8`) instead of falling
through to one serial `sgemv` call per batch element. `DeviceQ4KMatrix` additionally dequantizes
once into an FP16 scratch buffer (new `q4k_dequant_to_fp16`/`q5k_.../q6k_...` PTX kernels) before
reusing the same GEMM — no `sgemm` override for `DeviceQ4KMatrix` existed before this tier at any
batch size. Single-token decode kernels are unchanged (out of scope, `PROMPT-P0-Gate.md`'s domain).

`compare-llama-cpp.sh --gpu` (GTX 1080, default 4-model set, `n_gen=64`):

| Model | `--mmq off` pp | `--mmq off` tg | pp/tg ratio | `--mmq on` pp | `--mmq on` tg | pp/tg ratio |
|-------|---------------:|---------------:|------------:|--------------:|--------------:|------------:|
| tinyllama-1.1b Q4_K_M | 64.97 | 28.66 | **2.27x** | 59.38 | 39.20 | **1.51x** |
| qwen2.5-3b Q4_K_M | 34.54 | 13.22 | **2.61x** | 30.99 | 18.75 | **1.65x** |
| Phi-3.5-mini Q4_K_M | 35.48 | 12.74 | **2.78x** | 35.04 | 20.00 | **1.75x** |
| mistral-7b Q4_K_M | 0.93 | 0.54 | 1.72x (CPU fallback, OOM — Tier 5's domain, unaffected) | 21.57 | 16.06 | **1.34x** |

**Qualitative gate — met.** Before this tier, GPU pp and tg sat within ~1x of each other on every
model (the fingerprint of prefill never getting a batched kernel). Every GPU-resident model above
now shows pp materially greater than tg under both `--mmq off` and `--mmq on`.

**Quantitative floor (Tier 17 plan doc, 3x over "today" pp) — not directly checkable as stated.**
The plan doc's "today" baseline column was measured with a token-count-matched raw prompt; the
bake-off numbers above use the compare script's default short API/chat-template prompt (~20-30
tokens) — pp t/s at very different window sizes is not an apples-to-apples ratio. This is the same
prompt-length/methodology gap Tier 8 already named as an open compare-script parity item, not a
new problem introduced here. The pp/tg ratio, not the absolute pp value, is this tier's honest
signal, and it clears the qualitative bar on every model tested.

**Long-window caveat** — [`20260915T043143Z`](20260915T043143Z/) (`--prefill-batch 512`, raw
512-token prompt): pp regresses vs. the 32/128-token-window runs (TinyLlama 64.2 -> 31.8 t/s). JFR
confirms the batched-GEMM path fires correctly at that window size
(`cuda_resident_q4k_gemm.count=308` for 2 prefill calls); `ForwardPass.prefill.total_ms=16650` vs.
`MatVec.duration.total_ms=1873` for the same run shows over 14.7s of the 16.65s prefill wall time
is spent outside MatVec — attention/softmax/RoPE is `O(seq^2)` and untouched by this tier, the more
likely dominant cost at 512-token windows, not the GEMM this tier fixed. No attention-specific JFR
span exists yet to confirm directly — named follow-up before publishing a `--prefill-batch >= 512`
number as a tier result.

**Per-handler coverage:** Llama-family and Phi-3 confirmed live via JFR (`cuda-resident-fp16-gemm`
/ `cuda-resident-q4k-gemm` backend labels firing on real TinyLlama / Phi-3.5-mini forward passes).
Qwen3 shares the identical `backend.sgemm(...)` call site by code inspection
(`Qwen3TransformerHandler.java:683-696`) but has no loadable non-MoE Qwen3 GGUF fixture available
to live-verify (pre-existing Model E2E gap, not a Tier 17 regression). `Qwen3MoeTransformerHandler`
has no GPU residency path at all, batched or serial (pre-existing, unrelated to this tier).
`RocmMatVec` has zero `sgemm` overrides for any residency type — named follow-up, no ROCm hardware
available this session to implement or validate; the serial `MatVec` default keeps it correct, just
unoptimized.

**Correctness:** `CudaSgemmBatchedPrefillParityTest` (`DeviceHalfMatrix`/`DeviceQ4KMatrix`, batches
{1, 8, 9, 16, 32, 128}, non-tile-aligned shapes) and `Q4KDequantParityTest` (isolated dequant
kernels vs. `GgufKQuantCodec.decodeRows`, 5e-3 tolerance) are green.
`CudaSgemmBatchedPrefillConcurrencyTest` (new) runs 4 threads concurrently, each against a
distinctly-shaped weight matrix and independent random data, on the large-batch path for both
matrix types — confirms the per-thread `Fp16Scratch`/`Q4KDequantScratch` buffers do not
cross-contaminate under concurrent prefill.

**LoRA regression gate:** [`20260915T190157Z-lora`](20260915T190157Z-lora/) — train 44,000ms
(2,933ms/pass), playback 12.37 t/s (wall), recall correct. In line with the last pre-Tier-17
snapshot [`20260911T235455Z-lora`](20260911T235455Z-lora/) (train 47,000ms/3,133ms-per-pass,
playback 11.8 t/s) — flat, as expected: `LoraTrainableHandler` routes through
`ResidentWeightMatrix`/`LoraResidentWeights`, not `CudaMatVec.sgemm`, so this tier does not touch
the LoRA path at all.

**Vision gate:** N/A, not run. `VisionAwareForwardPassHandler` delegates its batched window to the
wrapped text handler's `forwardBatch`, so in principle a Phi-3/Llama-backed vision model would
share this tier's fix — but the only vision fixture available (`moondream2-q5_k.llamafile`) is
Phi-2-backed, and `Phi2TransformerHandler`'s batched prefill uses its own CPU
weight-stationary quant kernels (`sgemmQuantBatch` -> `LlamaTransformerHandler.sgemmQ*WeightStationary`),
never `CudaMatVec.sgemm` — confirmed by reading `Phi2TransformerHandler.java`. Separately,
`compare-vision.sh` defaults to `--prefill single` (batch=1, below this tier's `> 8` threshold
regardless), and its own documented `--prefill batched` mode already produces wrong captions for a
pre-existing, unrelated reason. Nothing in this tier changes vision's benchmarked path.

## Inference regression — `20260914T220204Z`

`compare-llama-cpp.sh --models tinyllama --cpu --vector 0 --no-jfr` after the embeddings API landing
(Infra Tier 11 — `POST /v1/embeddings`, off by default). Failures=0. Not a full curated-model
bake-off: this tier adds a new REST route and an `InferencePipeline.embedTokens` default method
but does not touch `MatVec`, `forward`/`forwardMultiDecode`, KV, or vision code paths, so per
Execution rule §2's API-only-tier carve-out this is a regression spot-check on the existing chat
completions path, not a throughput claim; `compare-lora.sh` / `compare-vision.sh` were not run
(same carve-out). `--models tinyllama` matches both TinyLlama GGUFs present under `models/`. See
[INDEX](20260914T220204Z/INDEX.md).

| Model | llama.cpp tg | Juno tg | Juno/llama |
|-------|-------------:|--------:|-----------:|
| tinyllama-1.1b Q2_K | 15.89 | 1.46 | 0.092 |
| tinyllama-1.1b Q4_K_M | 25.28 | 2.49 | 0.099 |

## Inference regression — `20260913T032734Z`

`compare-llama-cpp.sh --cpu --vector 0 --reps 1`. Failures=0. Tools are prompt+parse; CUDA/ROCm cells N/A (no GPU compare). See [INDEX](20260913T032734Z/INDEX.md).

| Model | llama tg | Juno tg | Juno/llama |
|-------|---------:|--------:|-----------:|
| tinyllama-1.1b Q4_K_M | 5.04 | 2.94 | 0.58 |
| qwen2.5-3b Q4_K_M | 2.27 | 1.00 | 0.44 |
| Phi-3.5-mini Q4_K_M | 2.86 | 0.83 | 0.29 |
| mistral-7b Q4_K_M | 1.97 | 0.47 | 0.24 |

Juno absolute tg is in line with the previous CPU bake-off [`20260912T193402Z`](20260912T193402Z/) (TinyLlama 3.33→2.94, Qwen 1.13→1.00, Phi-3.5 0.91→0.83, Mistral 0.48→0.47). llama.cpp tg on this host/bin is far below that earlier INDEX, so **ratios are not comparable** across those two runs.

Cross-feature smoke: [`../../target/tools-smoke/20260913T025903Z/`](../../target/tools-smoke/20260913T025903Z/) (`smoke-tools.sh`, `juno.GrammarConstrained.count=3` on required/named ChatML).

## Inference regression — `20260912T193402Z`

`compare-llama-cpp.sh --cpu --vector 0 --reps 1`. Failures=0. Grammar is sampler-side; CUDA/ROCm cells N/A (no GPU compare). See [INDEX](20260912T193402Z/INDEX.md).

| Model | llama tg | Juno tg | Juno/llama |
|-------|---------:|--------:|-----------:|
| tinyllama-1.1b Q4_K_M | 46.0 | 3.33 | 0.072 |
| qwen2.5-3b Q4_K_M | 15.7 | 1.13 | 0.072 |
| Phi-3.5-mini Q4_K_M | 13.8 | 0.91 | 0.066 |
| mistral-7b Q4_K_M | 5.34 | 0.48 | 0.090 |

Juno absolute tg is in line with the prior CPU bake-off [`20260911T221215Z`](20260911T221215Z/) (TinyLlama 3.09→3.33, Qwen 1.05→1.13, Phi-3.5 0.86→0.91, Mistral 0.50→0.48). llama.cpp tg on this host/bin is ~3–4× that earlier INDEX, so **ratios are not comparable** across those two runs.

Cross-feature smoke: [`../../target/grammar-smoke/20260912T193149Z/`](../../target/grammar-smoke/20260912T193149Z/) (`smoke-grammar.sh`, failures=0, `juno.GrammarConstrained.count` proof).

## P0 MatVec bake-off — `20260911T235203Z` (`--mmq on`)

`compare-llama-cpp.sh --gpu --vector 0 --mmq on --gpu-layers auto`. Failures=0. JFR `cuda_resident_q4k` on all four models (cpu.count=0).

| Model | llama tg | Juno tg | Juno/llama |
|-------|---------:|--------:|-----------:|
| tinyllama-1.1b Q4_K_M | 185.1 | 39.3 | 0.21 |
| qwen2.5-3b Q4_K_M | 68.5 | 18.2 | 0.27 |
| Phi-3.5-mini Q4_K_M | 58.7 | 19.3 | **0.33** |
| mistral-7b Q4_K_M | 35.9 | 15.3 | **0.43** |

P0 Phi-3.5 ≥ **0.5×**: **unmet**. P0 mistral ≥ **0.15×** with `--gpu-layers auto`: **met**.

Paired Phi-3.5 `--mmq off` [`20260911T235353Z`](20260911T235353Z/): Juno tg **12.83**. MMQ/FP16 **1.51×** (tile-kernel ≥1.3× **met**). Phi-3.5 `cuda_resident_q4k.p95` ≈ **0.32 ms** (prior PTX ≈ 2.3 ms).

## LoRA train-qa regression — `20260911T235455Z-lora`

| ref | train total ms | ms/pass | playback tps (wall) | recall |
|-----|---------------:|--------:|--------------------:|:------:|
| release-0.1.2 | 47,000 | 3,133 | 13.5 | ✓ |
| HEAD | 47,000 | 3,133 | 11.8 | ✓ |

**Current vs release-0.1.2:** train wall **1.00×**; playback wall tps **0.88×** (≥0.80 gate). Status **ok**.

## Inference regression — `20260911T221215Z`

CPU (`--vector 0`) after OpenAI `stop` / `seed` / `presence_penalty` landing. Failures=0.
Default curated models + accidental TinyLlama Q2_K row from `--models tinyllama` filter.
GPU re-run deferred (NVIDIA driver unavailable). See [INDEX](20260911T221215Z/INDEX.md).

## Continuous vs static bake-off — `20260911T194430Z-continuous`

TinyLlama Q4_K_M · GPU · 8 sessions · max_tokens=64 · `compare-schedule.sh`.

| Workload | continuous | static | Notes |
|----------|-----------:|-------:|-------|
| Aggregate tg t/s | 25.57 | 29.66 | continuous **0.86×** synchronized blocking |
| SSE mean TTFT ms | 6414 | 5136 | ContinuousStep max_decode_batch=8, shared_steps=64 |
| SSE mean TPOT ms | 208 | 168 | shared-step proof **PASS** |
| Prefix hit rate | 0.875 | 0.875 | multi-turn `x_juno_session_id`; 7/8 hits |

P1 “continuous SSE beats static” gate: **unmet** on this synchronized recipe (honest).

## Mixed chunked prefill — `20260911T204721Z-mixed-prefill`

TinyLlama Q4_K_M · GPU · continuous · n_prompt=256 · 3 shorts · `compare-mixed-prefill.sh`.

| Mode | short mean TTFT ms | short mean TPOT ms | Notes |
|------|-------------------:|-------------------:|-------|
| mixed (default) | 4552 | 929 | `prefill_chunks=12` proof **PASS** |
| admit-time baseline | 13929 | 189 | `-Djuno.continuous.mixedPrefill=false` |

Short TTFT mixed/admit **0.327×**. Short-decode TTFT bound ≤ **5702 ms** (1.25× max). TPOT rises under mix (shared steps) — documented tradeoff.

## Inference regression — `20260911T204900Z`

`compare-llama-cpp.sh --gpu --vector 0`. Failures=0.

| Model | llama tg | Juno tg | Juno/llama |
|-------|---------:|--------:|-----------:|
| tinyllama-1.1b Q4_K_M | 164.7 | 26.2 | 0.16 |
| qwen2.5-3b Q4_K_M | 61.3 | 11.8 | 0.19 |
| Phi-3.5-mini Q4_K_M | 54.2 | 11.3 | 0.21 |
| mistral-7b Q4_K_M | 32.9 | 0.42 | 0.013 |

## LoRA train-qa regression — `20260911T205447Z-lora`

| ref | train total ms | ms/pass | playback tps (wall) | recall |
|-----|---------------:|--------:|--------------------:|:------:|
| release-0.1.2 | 54,000 | 3,600 | 11.7 | ✓ |
| HEAD | 54,000 | 3,600 | 10.1 | ✓ |

**Current vs release-0.1.2:** train wall **1.00×**; playback wall tps **0.86×** (≥0.80 gate). Status **ok**.

## Inference regression — `20260911T195008Z` (continuous landing)

`compare-llama-cpp.sh --gpu --vector 0`. Failures=0. Qwen2.5 re-measured alone after first matrix pass hit CPU MatVec (VRAM contention).

| Model | llama tg | Juno tg | Juno/llama |
|-------|---------:|--------:|-----------:|
| tinyllama-1.1b Q4_K_M | 168.1 | 25.8 | 0.15 |
| qwen2.5-3b Q4_K_M | 62.7 | 12.3 | 0.20 |
| Phi-3.5-mini Q4_K_M | 55.7 | 12.5 | 0.22 |
| mistral-7b Q4_K_M | 34.2 | 0.45 | 0.013 |

## LoRA train-qa regression — `20260911T195711Z-lora`

Scenario: TinyLlama Q4_K_M · `/train-qa` *What is your name?* → *My name is Juno* · loss target 1.2 · playback temperature 0.

| ref | commit | train total ms | ms/pass | passes | playback tps (wall) | recall |
|-----|--------|---------------:|--------:|-------:|--------------------:|:------:|
| release-0.1.2 | 51a3b90 | 51,000 | 3,400 | 15 | 11.9 | ✓ |
| HEAD | 57839dd | 50,000 | 3,333 | 15 | 10.6 | ✓ |

**Current vs release-0.1.2:** train wall **0.98×**; playback wall tps **0.88×** (≥0.80 gate). Status **ok**.

## Quantized KV regression — `20260910T170557Z`

Default `--cache-type-k/v f16` (float32 path) after `DenseKvTensor` landing. `compare-llama-cpp.sh --gpu --vector 0`. Failures=0.

| Model | llama tg | Juno tg | Juno/llama |
|-------|---------:|--------:|-----------:|
| tinyllama-1.1b Q4_K_M | 173.4 | 33.4 | 0.19 |
| qwen2.5-3b Q4_K_M | 63.3 | 12.4 | 0.20 |
| Phi-3.5-mini Q4_K_M | 53.2 | 12.8 | 0.24 |
| mistral-7b Q4_K_M | 32.4 | 0.47 | 0.015 |

No decode regression vs prior default-GPU rows (~0.17–0.22× on small models). Memory claim for `q8_0` is unit-proven (`Q8_0KvCodec.compressionRatioVsF32` ≥2×), not a throughput claim.

## LoRA train-qa regression — `20260910T180703Z-lora`

Scenario: TinyLlama Q4_K_M · `/train-qa` *What is your name?* → *My name is Juno* · loss target 1.2 · playback temperature 0.

| ref | commit | train total ms | ms/pass | passes | playback tps (wall) | recall |
|-----|--------|---------------:|--------:|-------:|--------------------:|:------:|
| release-0.1.2 | 51a3b90 | 49,000 | 3,267 | 15 | 12.5 | ✓ |
| HEAD | e137c15 | 49,000 | 3,267 | 15 | 11.0 | ✓ |

**Current vs release-0.1.2:** train wall **1.00×**; playback wall tps **0.88×** (≥0.80 gate). Status **ok**. Gate uses REPL wall-clock tps (`compare-lora.sh`); JFR `TokenProduced.tps` is informational (`tps_jfr` ≈ 0.98× on this run).

Earlier ok snapshot: [`20260910T030058Z-lora`](20260910T030058Z-lora/). Earlier failing snapshot: [`20260902T200210Z-lora`](20260902T200210Z-lora/).

## LoRA train-qa regression — `20260910T030058Z-lora` (prior)

Scenario: TinyLlama Q4_K_M · `/train-qa` *What is your name?* → *My name is Juno* · loss target 1.2 · playback temperature 0.

| ref | commit | train total ms | ms/pass | passes | playback tps | recall |
|-----|--------|---------------:|--------:|-------:|-------------:|:------:|
| release-0.1.2 | 51a3b90 | 45,000 | 3,000 | 15 | 38.3 | ✓ |
| HEAD | 8b78382 | 44,000 | 2,933 | 15 | 38.3 | ✓ |

**Current vs release-0.1.2:** train wall **0.98×**; playback tps **1.00×** (≥0.80 gate). Status **ok**. Run: `./scripts/performance-tests/compare-lora.sh --gpu --baseline release-0.1.2`.

Earlier ok snapshot: [`20260905T031520Z-lora`](20260905T031520Z-lora/). Earlier failing snapshot: [`20260902T200210Z-lora`](20260902T200210Z-lora/).

## GPU fused Q4_K MMQ bake-off — `20260910T025804Z`

`compare-llama-cpp.sh --gpu --vector 0` with `-DJUNO_MMQ=on`. JFR proves `cuda_resident_q4k` on all four models (including Phi-3 fused path).

| Model | llama tg | Juno tg (MMQ on) | Juno/llama | q4k MatVec count |
|-------|---------:|-----------------:|-----------:|-----------------:|
| tinyllama-1.1b Q4_K_M | 195.4 | 27.3 | 0.14 | 12462 |
| qwen2.5-3b Q4_K_M | 71.8 | 10.7 | 0.15 | 12744 |
| Phi-3.5-mini Q4_K_M | 60.6 | 7.39 | 0.12 | 6720 |
| mistral-7b Q4_K_M | 37.1 | 5.34 | 0.14 | 15936 |

**Gates:** P0 Phi-3.5 ≥ **0.5×** llama — **unmet** (0.12×). Original Tier 13B ≥ **1.3×** vs FP16-resident — **amended / deferred** (MMQ ships as VRAM-fit; this run is slower than prior FP16 GPU baseline Phi-3.5 tg **12.6**). Mistral packed-Q4 ≈ **0.14×** llama supports the fit claim (near P0 **0.15×**).

## Vision chat regression — `compare-vision.sh`

Scenario: `moondream2-q5_k.llamafile` (embedded vision, no mmproj) · `POST /v1/vision/chat` · *What is in this image?* · max_tokens 32 · temperature 0 · `./juno local --jfr` · **`--prefill single`** (default in the script).

Fixed prefill window: **~741 tokens** (729 image patches + ~11 text). Local mode only — cluster does not register vision routes.

Default `--prefill single` matches the known-good sequential Phi2 path on `47-vision`. Batched Q5_K prefill on current inference branches can finish after the hang fix but still yields wrong captions; use `--prefill batched` only when intentionally measuring that path.

| Check | Threshold |
|-------|-----------|
| Quality | HTTP 200, non-empty reply |
| Latency | `current.latency_ms / baseline ≤ 1.25` |
| Decode tps | `current.tps / baseline ≥ 0.80` (JFR `TokenProduced.tps` when present) |

```bash
./scripts/performance-tests/compare-vision.sh --gpu --baseline 47-vision
./scripts/performance-tests/compare-vision.sh --gpu --no-publish   # single ref only
./scripts/performance-tests/compare-vision.sh --gpu --prefill batched --no-publish  # batched path only
```

Test image: `scripts/performance-tests/fixtures/vision-bench.jpg`. Override with `--image` or `VISION_TEST_IMAGE`.

### Known-good snapshot — `20260904T141315Z-vision` (`47-vision`)

| Field | Value |
|-------|-------|
| Status | success |
| Prompt tokens | 741 |
| Latency | ~503 s |
| Decode tps (JFR) | ~1.41 |
| Reply | non-empty (color squares) |

## Vector SIMD CPU bake-off — `--vector 0` vs `--vector 1`

Paired CPU runs on the default model set (`n_prompt=128`, `n_gen=64`, `reps=1`, JFR). Policy: Q4_K/Q5_K weight-stationary accumulate stays scalar; `--vector` only toggles `--add-modules jdk.incubator.vector` (Q8_0 dequant when probe passes). See [`../performance.md`](../performance.md) and [`../infra-plan/PLAN-Infra-Vector-SIMD.md`](../infra-plan/PLAN-Infra-Vector-SIMD.md).

| Model | Juno tg `--vector 0` | Juno tg `--vector 1` | v1/v0 |
|-------|---------------------:|---------------------:|------:|
| tinyllama-1.1b Q4_K_M | 2.89 | 2.98 | 1.03 |
| qwen2.5-3b Q4_K_M | 0.957 | 0.965 | 1.01 |
| Phi-3.5-mini Q4_K_M | 0.818 | 0.812 | 0.99 |
| mistral-7b Q4_K_M | 0.453 | 0.463 | 1.02 |

**Verdict:** near-parity (±3%) as expected under the scalar accumulate policy. Artifacts: [`20260904T194612Z`](20260904T194612Z/) / [`20260904T195731Z`](20260904T195731Z/).

## CPU summary (JFR) — `20260831T230258Z`

| Model | llama.cpp pp | llama.cpp tg | Juno pp | Juno tg | Juno/llama tg |
|-------|-------------:|-------------:|--------:|--------:|--------------:|
| tinyllama-1.1b Q4_K_M | 32.6 | 0.61* | 5.2 | 3.12 | 5.14* |
| qwen2.5-3b Q4_K_M | 23.6 | 3.80 | 1.78 | 1.01 | 0.27 |
| Phi-3.5-mini Q4_K_M | 17.5 | 3.54 | 0.86 | 0.84 | 0.24 |
| mistral-7b Q4_K_M | 10.7 | 2.18 | 0.81 | 0.48 | 0.22 |
| Qwen3.5-0.8B Q4_K_M | 68.1 | 8.01 | — | — | Juno load failed |

\* TinyLlama llama.cpp tg (0.61 t/s) looks like a single-rep outlier — prior CPU baseline was ~6.8 t/s on the same host. Juno JFR tg (3.12 t/s) is in line with expectations.

## GPU summary (JFR) — `20260831T231403Z` · GTX 1080

| Model | llama.cpp pp | llama.cpp tg | Juno pp | Juno tg | Juno/llama tg |
|-------|-------------:|-------------:|--------:|--------:|--------------:|
| tinyllama-1.1b Q4_K_M | 3583 | 186 | 19.5 | 31.4 | 0.17 |
| qwen2.5-3b Q4_K_M | 1356 | 68.0 | 8.1 | 13.3 | 0.19 |
| Phi-3.5-mini Q4_K_M | 1096 | 57.8 | 11.7 | 12.6 | 0.22 |
| mistral-7b Q4_K_M | 610 | 35.2 | 0.82 | 0.48 | 0.01 |

JFR tg is **~1.6–1.7×** API wall-clock tg on GPU for models that fit in VRAM. Mistral-7B Juno GPU still matches CPU (~0.48 t/s JFR), indicating VRAM/residency fallback on 8 GiB.

## Tier 5 GPU offload (`JUNO_GPU_LAYERS=auto`) — `20260901T032753Z` · mistral-7b only

| Model | llama.cpp tg | Juno tg (JFR) | Juno/llama tg | Notes |
|-------|-------------:|--------------:|--------------:|-------|
| mistral-7b Q4_K_M | 35.5 | 0.94 | **0.026** | Hybrid MatVec: ~10.4k GPU fp16 + ~6.6k CPU quant ops |

Prior GPU baseline (`20260831T231403Z`): mistral Juno tg **0.48** t/s (**0.01×**). Tier 5 auto offload is **~2×** faster but still below the P0 gate (**≥0.15×** ≈ 5.3 t/s).

## Multi-session static batch (`--parallel`) — TinyLlama Q4_K_M

Workload: 8 concurrent blocking `POST /v1/chat/completions`, `max_tokens=64`, temperature 0, `--nodes 1`, `--batch-window-ms 50` when `parallel>1`.

| Run | Backend | parallel=1 agg tg | parallel=8 agg tg | Speedup 8/1 | Notes |
|-----|---------|------------------:|------------------:|------------:|-------|
| [`20260901T154735Z-parallel`](20260901T154735Z-parallel/) | GPU | **28.8** t/s | 24.9 t/s | **0.87×** | Before multi-request decode batching |
| [`20260901T173121Z-parallel`](20260901T173121Z-parallel/) | GPU | 28.8 t/s | **32.1** t/s | **1.11×** | `forwardMultiDecode` + batched CUDA GEMV |
| [`20260901T155136Z-parallel`](20260901T155136Z-parallel/) | CPU | 1.46 t/s | **2.22** t/s | **1.52×** | Clear aggregate uplift on CPU |

**GPU (post-fix):** `LocalInferencePipeline.forwardBatch` routes N decode steps through `ForwardPassHandler.forwardMultiDecode`. All supported handler families implement batched decode: **Llama**, **Phi-3**, **Phi-2**, **Qwen3 dense**, and **Qwen3 MoE** (attention batched; MoE FFN routed per stream). Linear projections and LM head use `cublasHSSgemvStridedBatched` / `GpuBlasOps` where GPU weights are resident (batch ≤ 8). Prefill windows stay serial on GPU.

| Handler family | `forwardMultiDecode` | Parity test |
|----------------|---------------------|-------------|
| Llama | Yes | `LlamaTransformerHandlerMultiDecodeTest` |
| Phi-3 | Yes | `Phi3TransformerHandlerMultiDecodeTest` |
| Phi-2 | Yes (CPU quant batched GEMV) | `Phi2TransformerHandlerMultiDecodeTest` |
| Qwen3 | Yes (+ `forwardBatch` prefill) | `Qwen3TransformerHandlerMultiDecodeTest` |
| Qwen3 MoE | Yes (MoE FFN per stream) | `Qwen3MoeTransformerHandlerMultiDecodeTest` |

**Phi-3 / Qwen3 GPU multi-session:** re-run `./scripts/performance-tests/compare-parallel.sh --gpu --sessions 8` with the target model when validating non-Llama speedup; Llama baseline is [`20260901T173121Z-parallel`](20260901T173121Z-parallel/) (1.11× aggregate tg).

**GPU regression (0.87×, pre-fix):** static batching did not fuse multi-request decode on GPU.

1. **`LocalInferencePipeline` had no `forwardBatch` override** — N serial `forward()` per decode step.
2. **Prefill in `generateBatch()` is serial** — eight `prefillBatch()` calls in a loop before decode starts.
3. **Unfair baseline:** `--parallel 1` still launches each HTTP request on its own virtual thread (`dispatchSingle`), so eight clients overlap on the GPU lock. `--parallel 8` runs all eight in **one** `generateBatch()` on a single thread — fully serialized GPU work without batched kernels.

Handler `forwardBatch(BatchForwardRequest)` only batches **one request's prefill window** (W prompt tokens), not N concurrent decode streams.

CPU uplift (1.52×) likely comes from fewer contending threads and better cache locality despite the same serial decode path.

Re-run:

```bash
./scripts/performance-tests/compare-parallel.sh --gpu   # or --cpu
```

## Prefill microbatch (`--prefill-batch`) — TinyLlama Q4_K_M

Workload: long raw prompt (`n_prompt=256`), single blocking chat completion, JFR on. Script: `compare-prefill-batch.sh`.

| Run | Backend | batch=1 pp | batch=32 pp | Speedup 32/1 | `prefill.count` (1 / 32) |
|-----|---------|----------:|------------:|-------------:|-------------------------|
| [`20260901T234024Z-prefill`](20260901T234024Z-prefill/) | CPU | **2.30** t/s | **5.39** t/s | **2.35×** | 246 / 9 |

Default `--prefill-batch` is **32**. `--prefill-batch 1` matches per-token batched prefill (many small `PrefillBatch` JFR events). GPU re-run: see "GPU batched-prefill GEMM bake-off — Tier 17" below.

```bash
./scripts/performance-tests/compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32
```

## Single-stream compare re-run

```bash
# CPU (5 models incl. Qwen3.5 — Juno expected to fail on Qwen3.5)
./scripts/performance-tests/compare-llama-cpp.sh --cpu --vector 0 --reps 1 \
  --models tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf,Qwen3.5-0.8B.Q4_K_M.gguf,qwen2.5-3b-instruct-q4_k_m.gguf,Phi-3.5-mini-instruct-Q4_K_M.gguf,mistral-7b-instruct-v0.1-q4_k_m.gguf

# GPU (default 4-model set)
./scripts/performance-tests/compare-llama-cpp.sh --gpu --vector 0 --reps 1

# Tier 5 mistral bake-off (partial GPU residency)
JUNO_GPU_LAYERS=auto ./scripts/performance-tests/compare-llama-cpp.sh --gpu --vector 0 --reps 1 \
  --models mistral-7b-instruct-v0.1-q4_k_m.gguf
```

Use `--no-jfr` to revert to API latency tg only. Per-model artifacts: `*-llama-cpp.json`, `*-juno.json`, `*-juno-jfr.json`, `*-compare.json`.

Build CUDA llama-bench once:

```bash
cmake -S ../llama.cpp -B ../llama.cpp/build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build-cuda --target llama-bench -j"$(nproc)"
```
