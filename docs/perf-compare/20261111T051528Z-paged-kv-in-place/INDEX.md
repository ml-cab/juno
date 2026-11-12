# CPU attention reading host KV in place: same-session A/B

**Purpose.** Evidence for the change that makes CPU attention read each request's K and V rows where they are stored: paged
KV page by page through its block table, `q8_0` KV decoded one head's slice at a time inside the loop, with no gather into a
contiguous buffer and no whole-cache decode per layer. It checks the standing throughput and allocation bars (CPU tg and pp
>= 0.95x, allocation per token <= 1.10x, in-span GC <= 1.25x), the LoRA and vision bars, greedy replies against the
pre-change build, and the paged-KV perf gate (`paged_vs_dense%` <= 2% at ctx 8192 / batch 8 / page 16).

**Pinned: no.** Unpinned, same host, same session, one run at a time, nothing else measuring (CPU governor `schedutil`, turbo on,
GPU clocks not pinned; `host.json` in each lane). This is not the pinned standing-gate run, which is the owner's at the close of
the KV work.

**Host.** Intel Xeon E5-1650 v2 (6 cores, 12 threads, AVX, F16C), 62.7 GiB; NVIDIA GTX 1080 for the GPU lanes (LoRA, vision).

**Builds.**

| Side | Tree | Shaded jar (`juno-player-0.1.2-shaded.jar`, SHA-256, first 16) |
|---|---|---|
| A, baseline | `74f75d78`, built from `git archive` in scratch (`mvn -q package -DskipTests`) | `96ee5784b914097c` |
| B, run 1 | `74f75d78` plus the change, first form | `8f58887fd342cbe7` |
| B, run 2 | `74f75d78` plus the change, final form (prefill window attended in one pooled call) | `9af57a8054650216` |

## Commands

CPU lanes, A B A B A B, each side its own jar through the candidate tree's compare script:

```bash
scripts/performance-tests/compare-llama-cpp.sh --cpu --models <model> --n-prompt 512 [--schedule continuous] \
  [--cache-type-k q8_0 --cache-type-v q8_0] --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish \
  --juno-jar <side's jar> --out <lane>-<side>-<i>
```

Lanes: TinyLlama 1.1B Q4_K_M at `n_prompt` 512 under `static` and `continuous`, each with F16 and `q8_0` KV; Phi-3.5-mini Q4_K_M
at `n_prompt` 128, `static`, F16 KV.

LoRA and vision, A B A B A B, each tree's own script on its own build:

```bash
scripts/performance-tests/compare-lora.sh --gpu --skip-build --no-publish --out <dir>
scripts/performance-tests/compare-lora.sh --cpu --skip-build --no-publish --out <dir>
scripts/performance-tests/compare-vision.sh --gpu --skip-build --no-publish --out <dir>
```

Smokes, candidate jar against the baseline jar:

```bash
JUNO_JAR=<B jar> scripts/performance-tests/smoke-long-prompt-prefill.sh --cpu --models tinyllama --baseline-jar <A jar>
JUNO_JAR=<B jar> scripts/performance-tests/smoke-consistency.sh
```

Microbench, each tree on its own build (run 1; the `kvcache` code it times is unchanged in run 2's build):

```bash
scripts/performance-tests/gather-tax-microbench.sh <out>
```

## Run 2 (final form), medians of three

| lane | pp A (t/s) | pp B (t/s) | pp B/A | tg A (t/s) | tg B (t/s) | tg B/A | alloc/token B/A | GC pauses A / B | greedy reply |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TinyLlama, static, F16 KV | 32.44 | 34.78 | 1.072 | 11.91 | 12.12 | 1.018 | 0.966 | 0 / 0 ms | equal, all 6 |
| TinyLlama, continuous, F16 KV | 32.22 | 33.66 | 1.045 | 11.88 | 11.67 | 0.982 | 0.760 | 0 / 0 ms | equal, all 6 |
| TinyLlama, static, Q8_0 KV | 31.32 | 30.82 | 0.984 | 11.98 | 11.68 | 0.975 | 0.748 | 0 / 0 ms | equal, all 6 |
| TinyLlama, continuous, Q8_0 KV | 30.39 | 32.65 | 1.074 | 11.84 | 11.52 | 0.972 | 0.755 | 0 / 0 ms | equal, all 6 |
| Phi-3.5-mini, static, F16 KV | 10.03 | 10.24 | 1.021 | 3.04 | 3.03 | 0.994 | 0.970 | 0 / 0 ms | equal, all 6 |

Per repetition, pp / tg (A, then B):
- TinyLlama, static, F16 KV: A 32.01/11.69, 32.44/11.91, 33.62/11.93; B 34.78/12.16, 34.78/11.91, 33.99/12.12
- TinyLlama, continuous, F16 KV: A 31.82/11.88, 32.22/11.90, 32.34/11.19; B 33.66/11.43, 33.10/11.67, 34.40/12.02
- TinyLlama, static, Q8_0 KV: A 31.32/11.98, 29.70/12.03, 31.35/11.96; B 30.54/11.71, 30.82/11.68, 31.78/11.25
- TinyLlama, continuous, Q8_0 KV: A 30.76/11.84, 30.39/11.64, 29.34/11.96; B 32.39/11.52, 32.65/11.52, 32.70/11.30
- Phi-3.5-mini, static, F16 KV: A 10.12/3.03, 9.67/3.04, 10.03/3.07; B 9.83/3.03, 10.24/3.05, 10.32/3.00

| Surface | A | B | Median B/A | Bar | Result |
|---|---|---|---:|---|---|
| LoRA GPU, train total ms | 41000, 41000, 41000 | 41000, 41000, 41000 | 1.000 | <= 1.25 | met |
| LoRA GPU, playback t/s | 13.011, 12.658, 13.035 | 13.109, 13.109, 12.456 | 1.007 | >= 0.80 | met; recall 3 of 3 each side |
| LoRA CPU, train total ms | 778000, 774000, 774000 | 773000, 775000, 774000 | 1.000 | <= 1.25 | met |
| LoRA CPU, playback t/s | 0.854, 0.852, 0.849 | 0.862, 0.860, 0.857 | 1.009 | >= 0.80 | met; recall 3 of 3 each side |
| Vision GPU, latency ms | 153552, 147155, 151706 | 149159, 149552, 150269 | 0.986 | <= 1.25 | met |
| Vision GPU, decode t/s | 4.724, 5.202, 5.046 | 5.121, 5.052, 5.068 | 1.004 | >= 0.80 | met; replies equal, 6 of 6 |

Smokes (`run2/smoke-*.log`): long-prompt prefill, all 24 checks pass, greedy text identical to the baseline build at 128, 512 and
2024 tokens under both schedules; TTFT at 2024 tokens, `static` 69.6 s (B) against 70.0 s (A), `continuous` 72.3 s against
88.2 s (one sample each). Consistency smoke, all 54 checks pass.

## Run 1 (first form), medians of three

| lane | pp A (t/s) | pp B (t/s) | pp B/A | tg A (t/s) | tg B (t/s) | tg B/A | alloc/token B/A | GC pauses A / B | greedy reply |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TinyLlama, static, F16 KV | 33.42 | 33.24 | 0.994 | 12.03 | 12.15 | 1.010 | 1.001 | 0 / 0 ms | equal, all 6 |
| TinyLlama, continuous, F16 KV | 31.78 | 33.02 | 1.039 | 11.86 | 11.89 | 1.002 | 0.747 | 0 / 0 ms | equal, all 6 |
| TinyLlama, static, Q8_0 KV | 30.22 | 28.01 | **0.927** | 11.77 | 11.52 | 0.979 | 0.728 | 0 / 0 ms | equal, all 6 |
| TinyLlama, continuous, Q8_0 KV | 30.73 | 27.83 | **0.905** | 11.74 | 11.49 | 0.979 | 0.738 | 0 / 0 ms | equal, all 6 |
| Phi-3.5-mini, static, F16 KV | 10.03 | 10.04 | 1.001 | 3.00 | 3.00 | 0.999 | 1.026 | 0 / 0 ms | equal, all 6 |

Run 1's `q8_0` prefill fell under the 0.95x bar. In its first form, a prefill window ran attention once per query row with query
heads as the pool's units, so each `q8_0` key and value slice was decoded once per query head (8 per KV head on TinyLlama), and
the value sum took the scalar path. JFR's top methods on the static `q8_0` lane: B `Q8_0KvCodec.decodeRange` 377 samples and the
scalar value sum 356, against A's whole-cache decode 110 and vector head 63. The final form attends a window in one pooled call
with (row, KV head) units, decoding each slice once for the query heads that share it, to the same output bits. LoRA, vision
and smokes in run 1 also met their bars (`run1/`).

## Microbench (run 1 builds)

`microbench-candidate.md`: attention over dense rows against attention reading the paged rows in place, TinyLlama-like GQA
(32 query heads, 4 KV heads, head width 64), F16 rows; `gather_ms` is zero by construction.
`microbench-baseline.md`: the pre-change microbench, which timed the gather into a contiguous buffer.

| ctx | batch | page | B: paged_vs_dense% (in place) | A: paged_vs_dense% (gather, then attention) | A: gather % |
|---:|---:|---:|---:|---:|---:|
| 8192 | 8 | 16 | **-6.35** (gate <= 2: met) | +6.00 | 4.75 |
| 8192 | 1 to 32 | 16 to 128 | -3.35 to -10.42 | +4.79 to +6.55 | 4.45 to 4.86 |
| 2048 | 1 to 32 | 16 | +4.04 to +10.46 | +5.85 to +9.96 | see file |
| 2048 | 1 to 32 | 64, 128 | -0.47 to +4.04 | +3.64 to +7.63 | see file |
| 32768 | 1 to 32 | 16 to 128 | +2.53 to +7.76 | +3.41 to +4.97 | see file |

A's `paged_vs_dense%` is (gather + attention over the gathered copy) against attention over dense rows; B's is attention
reading the pages in place against the same dense attention. At 32768 tokens B reads above A at page 16 (+7.6% against +4.9%):
reading a 32k context page by page costs more there than the gather did. Recorded, not addressed; outside the gate's cell.

## How to read it

- pp is prompt tokens per second of the measured request's prefill; tg is generated tokens per second over the decode span.
  The `continuous` lanes chunk the prompt, so their pp is `prompt_tokens * 1000 / forward_pass_prefill_total_ms` and their tg is
  `token_gen_tps_decode_derived` (the lane JSON's own `prompt_eval_tps` is null there).
- alloc/token is `jfr.allocated_bytes_per_token`, B over A. It falls about a quarter on the paged and `q8_0` lanes because the
  per-token scratch copies of the whole cache are gone.
- "greedy reply equal" compares B's reply text with A's in the same lane and repetition (6 pairs per lane).
- The microbench's paged-faster-than-dense cells at 8192 are not a claim that paging speeds attention: both columns run the
  same loop, one over one array and one page by page, and the difference is placement and noise. The cells over 2% (32k, and 2k
  at page 16) are outside the gate's cell and recorded as they read.
- Unpinned: a 2% to 4% swing between repetitions of one side is normal on this host (see the per-repetition lines).
- Paths in the copied JSON and logs are scrubbed to `<repo>`, `<scratch>`, `<baseline-tree>` and `<home>`.
