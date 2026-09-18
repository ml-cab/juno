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
| [`20260916T003101Z-prefill`](20260916T003101Z-prefill/) | GPU `--n-prompt 512`, new `juno.Attention` JFR span (prefill-batch=1 vs 32) | JFR pp + `Attention`/`MatVec` breakdown | [INDEX](20260916T003101Z-prefill/INDEX.md) |
| [`20260916T034621Z`](20260916T034621Z/) | GPU default 4-model sweep, `--gpu-attention off` (default) regression gate | JFR pp/tg | [INDEX](20260916T034621Z/INDEX.md) |
| [`20260916T035124Z`](20260916T035124Z/) | GPU default 4-model sweep, `--gpu-attention on` regression gate | JFR pp/tg | [INDEX](20260916T035124Z/INDEX.md) |
| [`20260916T035640Z-lora`](20260916T035640Z-lora/) | GPU LoRA train-qa + playback (post GPU-resident attention, expected flat — explicit no-op) | train ms / playback tps | [INDEX](20260916T035640Z-lora/INDEX.md) |
| [`20260916T035952Z-prefill`](20260916T035952Z-prefill/) | GPU `--n-prompt 512`, `--gpu-attention off` (default) | JFR pp + `Attention` share of prefill | [INDEX](20260916T035952Z-prefill/INDEX.md) |
| [`20260916T040113Z-prefill`](20260916T040113Z-prefill/) | GPU `--n-prompt 512`, `--gpu-attention on` | JFR pp + `Attention` share of prefill | [INDEX](20260916T040113Z-prefill/INDEX.md) |
| [`20260916T043335Z`](20260916T043335Z/) | GPU default 4-model sweep + Mistral-7B tuned lane, HEAD `8e73d5e` | JFR pp/tg | [INDEX](20260916T043335Z/INDEX.md) |
| [`20260916T043833Z-lora`](20260916T043833Z-lora/) | GPU LoRA train-qa + playback vs release-0.1.2, HEAD `8e73d5e` | train ms / playback tps | [INDEX](20260916T043833Z-lora/INDEX.md) |
| [`20260918T024641Z`](20260918T024641Z/) | GPU default 4-model sweep + generalized per-model tuned lane (`--mmq auto --gpu-attention auto --gpu-layers auto`, Tier 18) | JFR pp/tg | [INDEX](20260918T024641Z/INDEX.md) |
| [`20260918T030920Z`](20260918T030920Z/) | GPU default 4-model sweep, bare default flags (`--vector 0`, no `--mmq`/`--gpu-layers`), P0 gate re-verification post-Tier-17/18 | JFR pp/tg | [INDEX](20260918T030920Z/INDEX.md) |
| [`20260918T031702Z`](20260918T031702Z/) | CPU Vector SIMD refresh (`--vector 0`) | JFR pp/tg | [INDEX](20260918T031702Z/INDEX.md) |
| [`20260918T032455Z`](20260918T032455Z/) | CPU Vector SIMD refresh (`--vector 1`) | JFR pp/tg | [INDEX](20260918T032455Z/INDEX.md) |
| [`20260918T044915Z-lora`](20260918T044915Z-lora/) | GPU LoRA train-qa + playback vs release-0.1.2, Tier 10 (multi-adapter `--lora-play` + `lora-import`) regression gate | train ms / playback tps | [INDEX](20260918T044915Z-lora/INDEX.md) |
| [`20260918T063656Z`](20260918T063656Z/) | GPU TinyLlama (Q2_K + Q4_K_M) default path, ngram speculative decoding (`--spec-type` still `none` here) regression gate | JFR pp/tg | [INDEX](20260918T063656Z/INDEX.md) |
| [`20260918T063739Z-lora`](20260918T063739Z-lora/) | GPU LoRA train-qa + playback vs release-0.1.2, ngram speculative decoding regression gate (flat as expected — LoRA doesn't route through `forwardVerify`) | train ms / playback tps | [INDEX](20260918T063739Z-lora/INDEX.md) |
| [`20260918T152002Z`](20260918T152002Z/) | GPU Mistral-7B, draft-model speculative decoding (`--spec-type` still `none` here) regression gate | JFR pp/tg | [INDEX](20260918T152002Z/INDEX.md) |
| [`20260918T152100Z-lora`](20260918T152100Z-lora/) | GPU LoRA train-qa + playback vs release-0.1.2, draft-model speculative decoding regression gate (flat as expected — LoRA doesn't route through `forwardVerify`) | train ms / playback tps | [INDEX](20260918T152100Z-lora/INDEX.md) |

Earlier runs (API wall-clock tg only, no JFR): [`20260831T214609Z`](20260831T214609Z/) (CPU), [`20260831T223850Z`](20260831T223850Z/) (GPU).

## LoRA train-qa regression — `20260918T044915Z-lora` (Tier 10: multi-adapter `--lora-play` + `lora-import`)

Regression gate for [`PLAN-Infra-Tier10.md`](../infra-plan/PLAN-Infra-Tier10.md) (multi-adapter
scaled `--lora-play`, GGUF LoRA import). The single-file, scale-1.0 `--lora-play` path this
benchmark exercises returns the original `LoraAdapterSet` unchanged (`LoraPlaybackMerge`'s identity
shortcut), so no forward-pass or training math changed for this scenario — flat/near-flat ratios
here are the expected outcome, not just a passing number.

| ref | train total ms | ms/pass | playback tps (wall) | recall |
|-----|---------------:|--------:|--------------------:|:------:|
| release-0.1.2 | 57,000 | 3,800 | 10.09 | ✓ |
| HEAD | 57,000 | 3,800 | 9.07 | ✓ |

**Current vs release-0.1.2:** train wall **1.00×**; playback wall tps **0.90×** (≥0.80 gate). Status **ok**.
Also ran a CPU regression spot-check (`compare-llama-cpp.sh --cpu --vector 0 --models tinyllama`,
`--no-publish`, base text inference only): failures=0 on both TinyLlama Q2_K and Q4_K_M — Tier 10's
changes are additive to LoRA loading/CLI only and do not touch the base forward-pass/MatVec path.

## Ngram speculative decoding — regression gate + live smoke test — `20260918T063656Z`

Regression gate for ngram speculative decoding (`--spec-type none|ngram-simple`,
`docs/infra-plan/PLAN-Infra-Tier9.md`): `compare-llama-cpp.sh --gpu --models tinyllama` with
`--spec-type` left at its default (`none`) — failures=0, TinyLlama Q4_K_M tg **46.4** t/s, Q2_K tg
**31.1** t/s (JFR), both in line with prior TinyLlama baselines above. LoRA regression
(`20260918T063739Z-lora`) flat as expected (**1.00×** train wall, **0.90×** playback wall tps —
LoRA's `LoraTrainableHandler` never overrides `forwardVerify`, so it always uses
`ForwardPassHandler`'s correctness-preserving serial default regardless of `--spec-type`).

**This is not yet the standardized script bake-off** — `compare-llama-cpp.sh` has no built-in
workload for measuring draft-acceptance rate, so the numbers below are from a manual `./juno local`
REPL smoke test (TinyLlama Q4_K_M, GTX 1080, `--gpu-layers all`, `--temperature 0`, `--jfr 30s`,
`--spec-ngram-n 3 --spec-ngram-m 8`), prompting the model to literally repeat a 3-word cycle
("apple banana cherry") 10 times — a maximally repetitive workload, deliberately chosen per the
tier's exit gate ("measurable TPS gain on at least one repetitive workload"), not representative of
natural text.

| Metric (300 generated tokens) | `--spec-type none` | `--spec-type ngram-simple` |
|---|---:|---:|
| Wall-clock tg (`TokenProduced.tps`) | 59.1 t/s | **63.3 t/s** (**1.07×**) |
| Wall-clock elapsed | 5.074 s | 4.737 s |
| `juno.ForwardPass.count` (single-token decode calls) | 903 | 57 |
| `juno.Attention.count` | 6,666 | 1,276 |
| `juno.Attention.duration.total_ms` | 777.2 ms | 316.4 ms |
| `juno.MatVec.count` | 27,165 | 47,792 |
| `juno.MatVec.duration.total_ms` | 3,321.3 ms | 3,699.0 ms |
| `juno.Speculation.acceptanceRate` | n/a | **0.949** (280/295 drafted tokens accepted) |

**Reading this honestly:** draft acceptance is very high (94.9%) on this workload and the number of
decode *rounds* drops by roughly 16× (903 single-token forwards -> 57 forward/verify calls,
consistent with `--spec-ngram-m 8`'s max draft length), but wall-clock tg only improves **~7%**, not
proportionally. `juno.Attention` count and time both drop by roughly half (batching multiple query
positions into one attention dispatch per round, same mechanism `--gpu-attention`/Tier 13C already
uses for prefill) — this is where the real wall-time saving comes from. `juno.MatVec` time is flat
to slightly *higher* under speculation (a batched-window GEMM over up to 8 rows costs more per call
than a single-row GEMV, even though there are far fewer calls) — i.e. on this GPU, at this scale,
verify-window GEMM cost roughly cancels out the per-launch overhead saved by not issuing one GEMV per
token, which is consistent with (not a contradiction of) the P0 finding elsewhere in this doc set that
per-launch host/FFI overhead, not kernel throughput, is the current decode ceiling — fewer, larger
launches trade one kind of overhead for a bigger per-launch payload rather than eliminating overhead
outright. `forwardVerify`'s own GEMM/attention work is not yet wrapped in a `ForwardPassEvent`-style
JFR span (only single-token `forward()` calls are), so `juno.ForwardPass.count`/`.decode.total_ms`
undercount total decode work under speculation — noted here as a known instrumentation gap, not
double-counted or hidden.

**A real bug was found and fixed via this live smoke test, not by the unit suite**: the first
implementation fed the drafted tokens directly as the verify window's input row-for-row (row *b* =
`draft[b]`), which silently overwrote the KV entry for the *already-confirmed* token at the window's
first position with an unverified draft token's embedding — corrupted context that unit tests (driven
by a scripted test double with no real causal/KV semantics) could not catch, but immediately produced
garbled output against a real model. Fixed by shifting the verify window by one position (row 0 =
the already-confirmed last token, rows 1..M-1 = the first M-1 drafted tokens; see
`GenerationLoop.generate()`'s comment and `InferencePipeline.verifyDraft`'s javadoc for the exact
contract). Recorded here so a future session extending this path starts from a known-correct
baseline instead of re-discovering the same off-by-one.

**Not done this session**: a multi-model, multi-workload standardized bake-off (natural-text
neutrality per the tier's exit gate is plausible but unmeasured — only the maximally-repetitive
case was tested live); wiring `forwardVerify` into `Phi2TransformerHandler`/`Phi3TransformerHandler`/
`Qwen3TransformerHandler`/`Qwen3MoeTransformerHandler` (they fall back to the correctness-preserving
serial default — no speed benefit, named follow-up); a JFR span around `forwardVerify` itself.

## Draft-model speculative decoding — regression gate + live smoke test — `20260918T152002Z`

Regression gate for draft-model speculative decoding (`--spec-type none|ngram-simple|draft-simple`,
`docs/infra-plan/PLAN-Infra-Tier12.md`): `compare-llama-cpp.sh --gpu --models mistral` with
`--spec-type` left at its default (`none`) — failures=0, Mistral-7B Q4_K_M tg **18.76** t/s (JFR),
in line with prior Mistral baselines above (default-lane, no `--gpu-layers`/`--mmq` override). LoRA
regression (`20260918T152100Z-lora`) flat as expected (**0.98×** train total ms, **0.88×** playback
wall tps, both within the ±25%/≥80% gate — `LoraTrainableHandler` never overrides `forwardVerify` or
touches `--model-draft`, so `draft-simple` never reaches the LoRA path at all).

**Not yet the standardized script bake-off** — same caveat as the ngram entry above:
`compare-llama-cpp.sh` has no built-in draft-acceptance workload, so the numbers below are a manual
`./juno local` REPL smoke test: TinyLlama Q4_K_M as `--model-draft` (draft), Mistral-7B Q4_K_M as the
target (both share the same 32000-token Llama-family vocabulary — the pairing the vocab-size fail-closed
check in `GenerationLoop`'s constructor is designed to accept), GTX 1080, `--gpu-layers auto`,
`--temperature 0`, `--jfr 30s`, `--spec-ngram-m 8`, same "repeat apple banana cherry 10 times" maximally
repetitive prompt as the ngram entry for direct comparability.

| Metric (59 generated tokens) | `--spec-type none` | `--spec-type draft-simple` |
|---|---:|---:|
| Output text | `apple banana cherry` × 10 | **byte-identical** — same 10 lines |
| Wall-clock tg (`TokenProduced.tps`) | 19.78 t/s | **10.35 t/s** (**0.52×** — a regression, not a speedup) |
| `juno.MatVec.count` | 7,965 | 31,058 (**3.9×**) |
| `juno.MatVec.duration.total_ms` | 2,763.3 ms | 5,070.5 ms |
| `juno.Speculation.acceptanceRate` | n/a | **0.552** (53/96 drafted tokens accepted) |

**Reading this honestly:** token identity holds exactly (exit gate #1 met) and acceptance is decent
(55.2%) for a draft model that was never fine-tuned to match the target's distribution — but
wall-clock tg **regresses to 0.52×**, the opposite of a speedup. Unlike `ngram-simple`'s free
lookup-table proposals, `draft-simple`'s proposals cost real GPU work: `DraftModelSession.propose()`
drives TinyLlama through its own full transformer forward pass once per drafted token, and that cost
is *additional* to Mistral's own verify pass, not a replacement for it. `juno.MatVec.count` nearly
quadruples (7,965 -> 31,058) because every one of TinyLlama's own decode/prefill/resync forward calls
routes through the same global `juno.MatVec` span the target uses — this is the clearest evidence for
*why* the wall-clock result is negative: it directly compounds the P0 gap analysis's finding that
per-launch host/FFI overhead, not raw kernel throughput, is the current decode ceiling on this
hardware (`PLAN-Infra-PERF-ANALYSIS.md` → "Post-MMQ GPU idle-time finding"). A ~7x-smaller draft model
still issues its own thousands of tiny per-projection launches, and on this GPU those launches are not
free even though the FLOPs they represent are small. This is an honest, informative negative result,
not a bug — the exit gate's own wording anticipates it ("TPS uplift documented when draft is small and
acceptance is high; **failure cases documented**"). A draft model whose own decode cost is
proportionally smaller relative to the target (e.g. a much larger target, or a future lower-launch-overhead
decode path per the P0 lever) would be expected to change this ratio; not measured this session.

**Not done this session**: a multi-model, multi-workload standardized bake-off (only one draft/target
pair, one maximally-repetitive workload, was measured live); tuning `--spec-ngram-m` or trying a
smaller/larger draft model to see whether the regression narrows; wiring `--model-draft` into vision,
ROCm, or cluster/tensor-parallel launches (all explicit follow-ups per the tier doc, several fail
closed at CLI-parse time rather than silently no-op).

## HEAD bake-off + vision chat-template regression fix — 2026-09-16 (`8e73d5e`)

Full CPU + GPU compare-llama-cpp.sh sweep against `llama.cpp` requested for HEAD (`8e73d5e`,
tip of this branch). This host had **nine other Claude Code sessions** concurrently working the
same repo at the time, several of which had left long-running `compare-llama-cpp.sh --gpu` and
`compare-vision.sh` processes running well past their normal completion time (up to ~59 minutes,
vs. a normal ~5-20 minute run) — real contention for the single GTX 1080 and 12 CPU threads on
this shared dev box, not simulated. Reported honestly rather than pushed through with noisy
numbers:

- **GPU sweep** [`20260916T043335Z`](20260916T043335Z/) (default 4-model set + Mistral-7B tuned
  lane) completed cleanly (failures=0) — numbers are in line with prior GPU baselines on this host
  (e.g. TinyLlama tg 28.5 t/s vs. the `20260916T035124Z` regression-gate run's 29.1 t/s). No GPU
  contention was present during this specific run's ~5-minute window.
- **LoRA regression gate** [`20260916T043833Z-lora`](20260916T043833Z-lora/) vs. release-0.1.2:
  train ratio **1.00x**, playback tps ratio **0.906x** (≥0.80 gate) — **ok**.
- **CPU sweep**: attempted twice; both runs overlapped with the other sessions' stray GPU/vision
  processes competing for the same 12 CPU threads, making the numbers untrustworthy. Rather than
  publish contaminated CPU throughput numbers, this run was abandoned per explicit user direction
  after the contention did not clear within a bounded wait — **no CPU sweep published this
  session**. Re-run `compare-llama-cpp.sh --cpu --vector 0` alone on a quiet host for a trustworthy
  CPU baseline at this commit.
- **Vision regression found and fixed.** `compare-vision.sh --gpu --baseline release-0.1.2` caught
  a real bug, not noise: HEAD returned `completion_tokens=0` / empty caption /
  `finish_reason=stop` for the standard moondream2 vision-chat scenario (baseline release-0.1.2
  produced a normal 32-token caption). Root cause: `ConsoleMain.registerEmbeddedChatTemplate`
  (landed in commit `ebe4301`, the GGUF chat-template feature) registered the *named fallback*
  chat template into `EmbeddedChatTemplateRegistry` under the model's raw filename even when the
  GGUF carries no `tokenizer.chat_template` metadata at all. For `moondream2-q5_k.llamafile` that
  fallback resolved to generic ChatML (via `ChatModelType.fromPath`, which has no moondream/phi2
  case) and got published under the filename key, which `ChatTemplateFormatter.forModelType`
  checks *before* falling through to the filename-substring match that used to correctly select
  the moondream Q&A template (`"...\n\nAnswer:"`) that `VisionChatHandler` depends on. Result: the
  prompt ended in `<|im_start|>assistant\n` instead of `\n\nAnswer:`, so the first sampled token
  was immediately `<|endoftext|>`. The 7.7-minute wall time / 2301 "decode" JFR events were not a
  hang — `--prefill single` mode classifies every prompt position past index 0 as a decode-shaped
  forward pass, so 768 prompt tokens × 3 shard events ≈ 2304, all correctly attributed and
  producing nothing useful once the wrong template was selected.
  - **Fix**: only register into `EmbeddedChatTemplateRegistry` when
    `GgufChatTemplateResolver.hasEmbeddedTemplate()` is true (`ConsoleMain.java`,
    `registerEmbeddedChatTemplate`) — when no embedded template exists, downstream
    `ChatTemplateFormatter.forModelType` already resolves the correct named/substring template on
    its own, so registering the fallback was actively harmful, never useful.
  - **Regression test**: `ConsoleMainEmbeddedChatTemplateTest` (new) — verified to fail against the
    pre-fix code (reproduces the exact bug via a synthetic no-template GGUF named
    `moondream2-q5_k.llamafile`) and pass with the fix; a second case confirms models that *do*
    carry a real embedded template still register correctly under both keys.
  - **Suite-wide check**: full `mvn test -pl tokenizer,lora,node,coordinator,sampler,kvcache,health,
    registry,juno-player` passed clean after the fix — no other surface depends on the buggy
    registration.
  - **Not yet re-verified**: the live `compare-vision.sh` quality gate was not re-run against the
    fix in this session (host contention, per above) — the fix is unit-test-verified but the
    end-to-end perf-compare vision number at this commit is still outstanding.

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

## Standing per-model tuned lane — `20260918T024641Z` (Tier 18: generalized)

`compare-llama-cpp.sh --gpu` originally added a second row only for Mistral-7B
(`--mmq on --gpu-layers auto`, see history below). [`PLAN-Infra-Tier18.md`](../infra-plan/PLAN-Infra-Tier18.md)
generalized `run_mistral_tuned_lane` into `run_tuned_lane`: every model in the default GPU set now
gets a second, clearly-labeled `*-tuned` row using all three shipped `auto` flags together
(`--mmq auto --gpu-attention auto --gpu-layers auto`), alongside the existing vanilla-default row,
not instead of it (`--no-tuned-lane`, renamed from `--no-mistral-tuned-lane`, opts out). Rationale
unchanged from the original mistral-only lane (`docs/infra-plan/PLAN-Infra-Review-Fixes.md` item 8):
the published *default*-flags bake-off understates what Juno already does with its own shipped
`auto` modes, so the standing regression gate now measures both configurations for every model, not
only Mistral-7B.

Run (`n_prompt=128`, `n_gen=64`, `reps=3`, full default 4-model set, HEAD after the Tier 18 script
change): [`20260918T024641Z`](20260918T024641Z/):

| Model | default tg (JFR) | tuned tg (JFR) | tuned/default tg | default Juno/llama tg | tuned Juno/llama tg |
|---|---:|---:|---:|---:|---:|
| TinyLlama-1.1B Q4_K_M | 27.17 t/s | 44.16 t/s | 1.63× | 0.152× | 0.247× |
| Qwen2.5-3B Q4_K_M | 13.73 t/s | 20.26 t/s | 1.48× | 0.198× | 0.293× |
| Phi-3.5-mini Q4_K_M | 12.45 t/s | 19.95 t/s | 1.60× | 0.203× | 0.326× |
| Mistral-7B Q4_K_M | 0.529 t/s | 18.86 t/s | 35.6× | 0.0142× | 0.505× |

pp (`n_prompt=128`, single-window prefill, not the long `--prefill-batch` shape
`--gpu-attention` was measured for): flat-to-slightly-down for the three already GPU-resident
models (TinyLlama 66.2 → 61.2 t/s, Qwen2.5-3B 33.3 → 29.7 t/s, Phi-3.5-mini 38.1 → 34.6 t/s — a few
percent, within the noise band of the auto-resolved kernel path switching from dequant-to-FP16 to
packed-Q4-GEMM), and a large win for Mistral-7B (0.92 → 20.65 t/s, 22.6×) purely from
`--gpu-layers auto` giving it GPU residency it does not otherwise get. This is expected, not a
regression: `--gpu-attention`'s own prefill win (3.85×, see below) only shows up at
`--prefill-batch ≥ 32`; at the default single-window `n_prompt=128` shape used here, attention's
share of prefill cost is small on these models regardless of the flag (documented in the
`--gpu-attention` section below), so pp tracks the dominant GEMM path, not attention.

**JFR confirms exactly which auto resolution ran, per model, per flag** (`*-tuned-juno.log`,
`juno.MatVec.backend.*` / `juno.Attention.*` counts):

- `--mmq auto` resolved **on** (packed `cuda_resident_q4k` + `cuda_resident_q4k_gemm` MatVec
  events replace `cuda_resident_fp16`) for **all four models**, including Phi-3.5-mini.
- `--gpu-attention auto` resolved **on** for TinyLlama, Qwen2.5-3B, and Mistral-7B — attention p95
  dropped (TinyLlama decode p95 0.426ms → 0.174ms; Qwen2.5-3B 0.311ms → 0.119ms; total attention
  wall time roughly halved on each). It correctly resolved to **off** for **Phi-3.5-mini**: that
  model's `juno.Attention` event count is **0** in both the default and tuned runs, because
  `Phi3TransformerHandler` owns a separate attention implementation the GPU-resident kernel is not
  wired to yet — a named follow-up in Tier 13's own interaction matrix, not a silent gap introduced
  here. The tuned row's tg win for Phi-3.5-mini (1.60×) therefore comes entirely from `--mmq auto`,
  not `--gpu-attention`.
- `--gpu-layers auto` gave Mistral-7B full GPU residency it does not get by default on this 8 GiB
  card (near-zero GPU MatVec events in the default log vs. a full `cuda_resident_q4k` path in the
  tuned log), explaining its outsized tg/pp jump relative to the other three models, which are
  already fully GPU-resident by default.

Tuned/default ratios for TinyLlama, Qwen2.5-3B, and Phi-3.5-mini (1.48–1.63×) directionally confirm
the same effect already proven individually for Mistral-7B (`--mmq`/`--gpu-layers`, prior sessions)
and for `--gpu-attention` (3.85× pp at long `--prefill-batch`, below) — now visible together, per
model, in the one standing artifact. This is a reporting change only; no flag's own default value
changed (all three keep default off/none), and `auto`'s per-architecture fallback behavior is
unchanged from what Tier 5 and Tier 13 already shipped and tested.

### History: Mistral-7B-only tuned lane — `20260915T223032Z` (superseded by the above)

`compare-llama-cpp.sh --gpu` originally added a second Mistral-7B row (`*-tuned`,
`--mmq on --gpu-layers auto`) only when mistral-7b was among the selected GPU models. See
`docs/infra-plan/PLAN-Infra-Review-Fixes.md` item 8: on this 8 GiB card, Mistral-7B's
default-flags lane (`--mmq off`, `--gpu-layers` unset) has consistently measured worse than
Juno's own CPU numbers for smaller models across many prior sessions, while the tuned
configuration nobody was actually exercising in the standing regression gate is ~30-40x faster.

| Lane | Juno tg (JFR) | Juno/llama.cpp tg ratio |
|---|---:|---:|
| default (`--mmq off`, `--gpu-layers` unset) | 0.479 t/s | 0.0136× |
| tuned (`--mmq on --gpu-layers auto`) | 15.65 t/s | 0.444× |

Tuned/default ≈ **32.7×**, consistent with the session-78 finding this lane was added to track.

## GPU-resident attention bake-off — Tier 13 Phase C

`--gpu-attention on|off|auto` (`JUNO_GPU_ATTENTION`, default off, CUDA only) moves attention
(QK^T + softmax + weighted-V-sum) onto a device-resident FP16 KV mirror (`DeviceKvCache` +
`CudaGqaAttention` + `gqa_attention.ptx`, one block per batch-row/head, 3-pass QK^T+max / softmax /
weighted-V-sum) instead of scalar CPU Java (`GqaMath`, extracted unchanged from the prior
`gqaInto`/`gqa`). Direct follow-on to the Tier 17 long-window finding above that attention, not the
GEMM path, dominates prefill/decode wall time at realistic context length.

`compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32` (TinyLlama Q4_K_M, GTX 1080):
[`20260916T035952Z-prefill`](20260916T035952Z-prefill/) (off) vs.
[`20260916T040113Z-prefill`](20260916T040113Z-prefill/) (on):

| `--gpu-attention` | prefill-batch | pp t/s (JFR) | prefill ms | attention share of prefill |
|---|---:|---:|---:|---:|
| off | 1 | 20.85 | 25,414.9 | 0.0% |
| off | 32 | 31.04 | 17,076.8 | **78.7%** |
| on | 1 | 40.74 | 13,008.1 | 0.2% |
| on | 32 | **119.56** | **4,432.8** | **11.0%** |

**Honest speed claim — met.** At `prefill-batch=32` (the window size where `juno.Attention` events
are actually classified as `prefill`, not folded into decode accounting at window=1): pp throughput
**31.04 -> 119.56 t/s (3.85x)**, attention's share of prefill wall time **78.7% -> 11.0%**. This is
a measured decode/prefill throughput lever on the exact workload (long raw prompt, GPU-resident
model) that motivated the feature — not a peer-latency claim.

**Regression gate — flat/improved, never regressed.**
[`20260916T034621Z`](20260916T034621Z/) (off) vs. [`20260916T035124Z`](20260916T035124Z/) (on),
`compare-llama-cpp.sh --gpu --vector 0` default short prompt (~20-30 tokens), 4-model set: TinyLlama
tg 28.85 -> 29.10 t/s, Qwen2.5-3B tg 13.39 -> 14.55 t/s, Phi-3.5-mini tg 12.87 -> 13.23 t/s,
Mistral-7B tuned lane (`--mmq on --gpu-layers auto`) tg 16.30 -> 19.52 t/s (+20%); pp flat within
run-to-run noise on every model. Expected: attention's share of cost is naturally small at this
short a context, so the regression gate is a flatness check, not this feature's bake-off signal —
`compare-prefill-batch.sh` above is.

**LoRA regression gate — flat, as expected (explicit no-op).**
[`20260916T035640Z-lora`](20260916T035640Z-lora/) vs. `release-0.1.2` baseline: train_total_ms ratio
0.95x, ms/pass ratio 0.95x, playback tps ratio 0.89x (all within the 1.25x train / 0.80x playback
gate), recall correct. `LoraTrainableHandler` never reads `GpuAttentionOptions` — attention stays
scalar CPU there regardless of the flag, with a one-time startup warning if `--gpu-attention` is set.

**Per-handler coverage:** `LlamaTransformerHandler` only (Llama-family, Mistral, Qwen2); vision is
wired automatically since `VisionAwareForwardPassHandler` delegates to the same handler. Phi-2,
Phi-3, Qwen3, and Qwen3-MoE keep their own scalar attention implementations and KV maps — named
follow-up, not silently unimplemented. ROCm: `CudaGqaAttention.tryCreate` returns `null` on
non-CUDA backends, falling back to `GqaMath` scalar CPU — never silent, logged at handler
construction.

**Correctness:** `GqaAttentionKernelParityTest` (kernel vs. `GqaMath` CPU oracle),
`DeviceKvCacheLifecycleTest` (device-byte allocate/free, grow-and-preserve),
`LlamaTransformerHandlerGpuAttentionLiveTest` (real GGUF + CUDA, confirms
`gpuAttentionActive()` and greedy-token parity), `LlamaTransformerHandlerAttentionJfrTest`,
`GpuAttentionOptionsTest`. Full `node` suite 522/522, GPU-tagged suite 91/91 (`mvn test
-Dgroups=gpu -pl node`), both unchanged in count from the pre-feature baseline.

**Known limitation:** multi-token greedy-decode can occasionally diverge between `on`/`off` after
15+ tokens on some prompts (FP16 KV rounding flipping a close greedy choice) — same class of
tradeoff already accepted for `--mmq` and other reduced-precision paths; single-step logits match
tightly via the parity test. Not treated as a defect; bit-identical multi-step generation is not the
bar for reduced-precision paths anywhere in this codebase.

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
is spent outside MatVec — attention was the suspected dominant cost, not the GEMM this tier fixed.

**Confirmed** with a new `juno.Attention` JFR span (wraps `gqaInto`/`gqa` — QK^T + softmax +
weighted-V-sum, scalar CPU regardless of GPU layer offload) added specifically to check this:
re-run at `--n-prompt 512 --prefill-batch 32`
([`20260916T003101Z-prefill`](20260916T003101Z-prefill/), TinyLlama Q4_K_M) shows
`Attention.prefill.total_ms=13088.7` of `ForwardPass.prefill.total_ms=16708.5` — **78.3%** of
prefill wall time is attention, versus an estimated ~9.9% GEMM (`MatVec`) and ~11.7% other
(RoPE/RMSNorm/residual/KV-cache-write loops). `Attention.prefill.count=374` = 17 windows x 22
layers, `p95_ms=66.0` per per-layer call at this context length. Decode attention share at the same
(grown) context is **64.2%** (`Attention.decode.total_ms=432.6` of `ForwardPass.decode.total_ms=
674.3`) — materially higher than the ~4-7% "non-MatVec" decode share measured on the original
short-prompt bake-off, because that bake-off's decode context length stayed small. Tier 17's
batched-GEMM fix is not the remaining lever at long context; attention is — evidence for the P5
FlashAttention-style work (still gated on Tier 8 baselines per the ROADMAP), not a scope change on
its own.

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

### Refresh — 2026-09-18 (post Tier 13B/13C, Tier 17, Tier 18)

Re-verification per [`PROMPT-Vector-SIMD-Refresh.md`](../infra-plan/PROMPT-Vector-SIMD-Refresh.md):
every GPU bake-off since 2026-09-04 had published with `--vector 0`, so the CPU SIMD path had gone
unverified across several sessions of GPU-side kernel work. `VectorQuantKernels`/`SimdThreadPool` had
no commits in that window; `LlamaTransformerHandler`/`Phi2TransformerHandler` gained GPU-only code but
no diff touched their `*WeightStationary` methods, so this is a like-for-like re-run, not a
methodology change.

| Model | Juno tg `--vector 0` | Juno tg `--vector 1` | v1/v0 |
|-------|---------------------:|---------------------:|------:|
| tinyllama-1.1b Q4_K_M | 3.271 | 3.300 | 1.009 |
| qwen2.5-3b Q4_K_M | 1.096 | 1.099 | 1.003 |
| Phi-3.5-mini Q4_K_M | 0.919 | 0.919 | 0.999 |
| mistral-7b Q4_K_M | 0.521 | 0.520 | 0.998 |

**Verdict:** still near-parity (0.998–1.009×), tighter than the original ±3% band and consistent with
"no shared-path change" from the diff check above — not a coincidence. No regression found;
`VectorQuantKernels` policy unchanged. `compare-vision.sh` skipped intentionally (no change to the
shared CPU MatVec dispatch path this refresh would need to catch). Artifacts:
[`20260918T031702Z`](20260918T031702Z/) / [`20260918T032455Z`](20260918T032455Z/).

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
