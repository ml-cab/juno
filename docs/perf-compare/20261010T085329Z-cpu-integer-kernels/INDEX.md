# CPU integer-dot kernels: decode and prefill (2026-10-10)

**Purpose.** Reading of the CPU integer kernels (activation quantized to 8 bits once per matmul, integer dot
products over the packed Q4_K/Q5_K/Q6_K/Q8_0 weights) on the four sweep models, against the reference tool on
the same host, with the kernel-variant measurements that chose them and the model-quality comparison against
the float-activation kernels.

**Pinned: no.** CPU governor `schedutil`, turbo on. This is an interim reading of the first CPU fix. It is not
the CPU closing gate: that is a pinned sweep, the owner's run. Compare the Juno t/s below with the pinned pre-fix
reference only as an indication; the ratios against the reference tool are taken in one session.

**Build.** Juno `aabc42cb` plus the uncommitted CPU kernel work (`CpuKernelSelection`, `CpuIntegerKernels`,
`CpuIntegerGemm`, and their wiring into every CPU matmul dispatch point), jar sha256 `91ca04a9caa4f403`, OpenJDK
25.0.3. Reference tool build `379ac6673`, CPU, `-t 12`. Juno's kernels ran on 12 threads (common pool
parallelism 11 plus the caller).

## Commands

```bash
scripts/performance-tests/compare-llama-cpp.sh --cpu \
    --models tinyllama-1.1b-chat-v1.0.Q4_K_M,qwen2.5-3b,Phi-3.5-mini,mistral-7b --n-prompt 128 --n-gen 64 \
    --threads 12 --reps 3 --juno-reps 3 --juno-warmup 2 --jfr 30m --no-publish --out <run>
java scripts/performance-tests/JfrSpanBreakdown.java --top 10 <run>/*-measured.jfr > jfr-span-breakdown.jsonl
scripts/performance-tests/cpu-decode-kernel-variants.sh --skip-build --out <dir>     # kernel-variants/decode-*
scripts/performance-tests/cpu-prefill-kernel-variants.sh --skip-build --out <dir>    # kernel-variants/prefill-*
mvn test -pl node -Dtest=ModelQualityNllLiveTest -Djuno.test.modelQuality=true        # model quality
```

## 1. Against the reference tool (median of three, min/max in `sweep/INDEX-sweep.md`)

| Model | Juno pp t/s | Juno tg t/s | Reference pp / tg t/s | pp ratio at 128 | tg ratio | Pre-fix pinned Juno pp / tg (`20261001T180241Z`) |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 16.81 | 7.75 | 71.26 / 25.19 | **0.236x** | **0.308x** | 6.11 / 3.14 |
| Qwen2.5-3B | 6.24 | 2.58 | 26.90 / 10.89 | **0.232x** | **0.236x** | 1.99 / 1.08 |
| Phi-3.5-mini | 4.86 | 2.22 | 19.04 / 8.71 | **0.255x** | **0.255x** | 0.91 / 0.90 |
| Mistral 7B | 2.65 | 1.30 | 11.60 / 5.15 | **0.228x** | **0.252x** | 0.86 / 0.51 |

Pre-fix ratios at the pinned CPU reference were 0.049x to 0.103x (pp) and 0.092x to 0.121x (tg). Every row here
reads above the CPU milestones (tg >= 0.20x, pp >= 0.10x at 128) on this unpinned reading; the milestones are
scored on the pinned closing sweep.

## 2. Attained weight bandwidth (file size as bytes per token; host 42.4 GB/s, `../20261009T220314Z-host-read-bandwidth/`)

| Model | Juno attained | Juno share | Reference tool share (this session) | Pre-fix Juno share |
|---|---|---|---|---|
| TinyLlama 1.1B (0.669 GB) | 5.19 GB/s | 12.2% | 39.7% | 5.0% |
| Qwen2.5-3B (2.105 GB) | 5.42 GB/s | 12.8% | 54.1% | 5.4% |
| Phi-3.5-mini (2.393 GB) | 5.31 GB/s | 12.5% | 49.2% | 5.1% |
| Mistral 7B (4.368 GB) | 5.66 GB/s | **13.4%** | 53.1% | 5.3% |

The tier's attained-bandwidth gate (>= 25% on Mistral 7B, about 2.43 t/s) is not met by this kernel alone.

## 3. Where the time goes now (JFR, `jfr-span-breakdown.jsonl`)

| Model, lane | ms per token | Top frames (share of samples) | Allocated per token | GC pauses in span |
|---|---|---|---|---|
| Mistral 7B, decode | 783 | Q4_K integer kernel 56.7%, Q6_K integer kernel 36.4%, `f16ToF32` 4.8% | 20.2 MB | 3, 17 ms |
| Mistral 7B, prefill | 381 | `CpuIntegerGemm` tile 88.4%, weight unpack 7.7%, attention 1.6% | 107.4 MB | 0 |
| Phi-3.5-mini, decode | 459 | integer kernels (Q6_K, Q5_K, Q4_K) with `f16ToF32` about 93% | 35.2 MB | 3, 10 ms |
| Phi-3.5-mini, prefill | 207 | `CpuIntegerGemm` tile 87.0%, unpack 6.4% | 109.9 MB | 0 |

Before the fix (`../20261010T032108Z-cpu-cost-breakdown/`): Mistral 7B decode 1,972 ms and prefill 1,130 ms per
token, prefill allocating 261.5 MB per token (the per-row dequantization scratch, now gone). Decode allocation is
unchanged (20.8 MB per token before); removing it is the allocation item, not this one.

## 4. Kernel variants that chose the kernels (`kernel-variants/`)

Decode width (one activation row), synthetic weights at Mistral 7B's FFN shapes, median ms per call, ratio
against the shipped float kernel (`a-single`); final run, `production` is now the integer kernel:

| Kernel | Threads | a: float, one accumulator | b: float, split 4 / 8 | c: integer, scalar | d3: integer, Vector API | production (shipped) |
|---|---|---|---|---|---|---|
| Q4_K 14336x4096 | 12 | 13.97 ms | 1.00 / 1.11 to 1.15 (earlier runs) | 0.680 | 0.300 | **0.328** |
| Q4_K 14336x4096 | 1 | 83.47 ms | 1.02 to 1.03 / 1.03 | 0.653 | 0.279 | **0.297** |
| Q6_K 4096x14336 | 12 | 23.55 ms | 0.95 to 1.02 / 0.78 to 0.84 | 0.631 | 1.512 | **0.652** |
| Q6_K 4096x14336 | 1 | 126.70 ms | 0.99 to 1.01 / 0.81 to 0.82 | 0.681 | 0.850 | **0.670** |

Prefill width, 4096x4096, median ms per call, ratio against the shipped float path:

| Format, B | a: float (weight-stationary; Q6_K per row) | b: float split 4 | c: integer scalar | d: `CpuIntegerGemm` |
|---|---|---|---|---|
| Q4_K, 128 | 254.7 ms | 0.762 | 1.060 | **0.329** (first run 0.258) |
| Q4_K, 741 | 1,617.5 ms | 0.901 | 0.939 | **0.235** |
| Q5_K, 128 | 250.7 ms | - | - | **0.292** |
| Q5_K, 741 | 1,723.2 ms | - | - | **0.219** |
| Q6_K, 128 | 905.5 ms | - | - | **0.146** |
| Q6_K, 741 | 5,172.6 ms | - | - | **0.141** |

How the shipped kernels were chosen, and why `production` differs from `d3`:

- **Q4_K and Q5_K decode** take the Vector API over 16-bit lanes (variant d3: packed bytes reinterpreted as
  16-bit lanes, nibbles split with shifts and masks, 8-bit activation split the same way). The shipped form
  reduces each sub-block's lanes as integers (exact, so order-independent) and scales in scalar float. `d3`
  accumulated in float vectors; inside the production row loop that form exceeded C2's inlining node budget,
  vector objects were boxed (Q5_K allocated 184 MB per call and ran 3x slower) and float lane reduction order
  varied with the compilation tier. The shipped form allocates nothing per call and is bitwise repeatable, at
  about 9% over `d3` on Q4_K. `host-specific`: AVX without AVX2 here; re-measure on AVX2/FMA and AVX-512/VNNI.
- **Q6_K and Q8_0 decode** take scalar integer loops: the Vector API forms measured slower at 12 threads
  (register pressure; AVX has 16 vector registers). `host-specific`.
- **Prefill** takes `CpuIntegerGemm` at every format: 3x to 7x faster than the float path.
- `decode-explore-int*` are the exploration runs (variants c, d, d2, d3), taken before the integer kernels were
  wired in, so their `production` row is still the float kernel.

## 5. Model quality (teacher-forced NLL over 2,048 tokens, integer against float-activation kernels)

| Model | NLL integer | NLL float | Ratio (threshold <= 1.01) | Argmax agreement, first 512 positions |
|---|---|---|---|---|
| TinyLlama 1.1B | 2.45603 | 2.45566 | **1.00015** | 506 / 512 (98.8%) |
| Qwen2.5-3B | 1.69713 | 1.69766 | **0.99969** | 511 / 512 (99.8%) |
| Phi-3.5-mini | 1.35909 | 1.35826 | **1.00061** | 511 / 512 (99.8%) |
| Mistral 7B | 1.52792 | 1.52891 | **0.99935** | 511 / 512 (99.8%) |

Threshold met on every sweep model. The text is `node/src/test/resources/cab/ml/juno/node/model-quality-nll.txt`
(token ids checked in beside it); the LLaMA-family handler scores through its batched verify path (the
integer prefill matmul), Phi-3 through one decode-width forward per position (the integer decode kernels).
Run on the final kernels; 13,111 s for all four models, both settings.

## How to read it

Unpinned, one session, median of three per row; a difference inside a row's own min/max spread is not a result.
The ratios move with the reference tool's own run-to-run variation as well as Juno's. The pinned closing sweep
scores the milestones.
