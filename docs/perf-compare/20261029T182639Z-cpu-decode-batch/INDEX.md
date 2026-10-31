# CPU decode batch: batched equals single, throughput against the prefill gemm (2026-10-29)

**Purpose.** A request in a CPU decode batch now gets, bit for bit, the logits its single-request decode gives. The
five handlers' multi-decode paths take `CpuIntegerDecodeBatch`, which runs each activation row through the single-row
kernel's own row function, instead of the prefill gemm `CpuIntegerGemm`. Batched decode attention takes the
serial-head form (`GqaMath.attendPooledAsSerial`). Cause: `../20261029T045221Z-cpu-batch-consistency/`. Readings here:

1. what the change costs, or buys, in CPU `--parallel` aggregate throughput, against the owner's threshold
   (aggregate t/s at `--parallel` 2, 4 and 8 at least 0.90x the previous build, on TinyLlama and Qwen2.5-3B);
2. whether a server's batched replies equal its single reply;
3. the kernel microbenchmark that came before the A/B;
4. the LoRA no-regression gate.

**Pinned: no.** The 0.90x threshold is a no-regression gate at 0.90x. The plan scores such a gate only from a pinned
same-hour A/B. These readings are same-hour and interleaved but unpinned (governor `schedutil`, turbo on), so they are
an indication, not the score. The owner's pinned run of the same A/B scored it the same day: **met**, B/A 1.009 to
1.214 (`../20261029T203047Z-cpu-decode-batch-gate/`). The LoRA gate (training time at most 1.25x, playback at least 0.80x) is looser than 0.90x,
so it may be read unpinned.

**Builds.**

- Baseline (A): `06d6d22d`, built from `git archive` with `mvn package -DskipTests` (player shaded jar, sha256 first
  16 `758520d7ad418890`).
- Candidate (B): this tree, `06d6d22d` plus the uncommitted decode-batch work (`1c181de29e52d249`).

OpenJDK 25.0.3. Host: 12 CPU threads, 62 GB RAM. The CPU integer kernels run at 128-bit lanes (the server log's SIMD
line), with the default `--threads`.

## Throughput: CPU `--parallel`, aggregate token generation

**Method.** `ab-parallel.sh` (copied here, run with `JUNO_ROOT` set) runs this tree's `compare-parallel.sh` against
each jar (`--juno-jar`):

- `--cpu --sessions P --parallel-values P`, 64 tokens, the script's greedy prompt, batch window 50 ms;
- two untimed concurrent warmup rounds per server (`--warmup-rounds 2`);
- heap 4 GB (TinyLlama) and 6 GB (Qwen2.5-3B);
- three repetitions per side, all four P values per repetition, A B on odd repetitions and B A on even;
- 2026-10-29, 16:30 to 18:22 UTC.

The reading is `aggregate_token_gen_tps`: all sessions' completion tokens over the round's wall time, prefill
included. Every run reports success, no failed session, and 64 tokens per session.

| Model | P | A runs | A median | B runs | B median | B/A | Threshold |
|---|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 1 | 10.30 / 10.68 / 10.60 | 10.60 | 10.38 / 10.61 / 10.44 | 10.44 | 0.985 | control |
| TinyLlama 1.1B | 2 | 10.13 / 10.48 / 10.38 | 10.38 | 10.14 / 10.62 / 10.56 | 10.56 | **1.017** | >= 0.90 |
| TinyLlama 1.1B | 4 | 8.94 / 8.85 / 8.57 | 8.85 | 11.20 / 10.72 / 11.04 | 11.04 | **1.247** | >= 0.90 |
| TinyLlama 1.1B | 8 | 10.72 / 8.25 / 9.76 | 9.76 | 10.88 / 10.77 / 11.44 | 10.88 | **1.114** | >= 0.90 |
| Qwen2.5-3B | 1 | 3.97 / 3.97 / 3.96 | 3.97 | 3.54 / 4.00 / 4.01 | 4.00 | 1.006 | control |
| Qwen2.5-3B | 2 | 3.93 / 4.01 / 3.88 | 3.93 | 4.01 / 4.27 / 4.18 | 4.18 | **1.064** | >= 0.90 |
| Qwen2.5-3B | 4 | 3.32 / 3.24 / 3.27 | 3.27 | 4.10 / 4.09 / 4.00 | 4.09 | **1.250** | >= 0.90 |
| Qwen2.5-3B | 8 | 3.59 / 3.92 / 3.59 | 3.59 | 4.07 / 4.08 / 3.91 | 4.07 | **1.134** | >= 0.90 |

- **The threshold holds at every point in this unpinned reading.** The smallest margin is at P=2 (1.017 TinyLlama,
  1.064 Qwen2.5-3B). At P=4 and P=8 the candidate is faster, 1.11x to 1.25x.
- **P=1 is the control.** At P=1 both builds run the single-request path (`forwardBatch` sends a one-row batch through
  `forward`), so 0.985 and 1.006 are the run-to-run spread.
- **Batching buys no aggregate throughput on this host's CPU, in either build.**
  - The candidate at P=8 reads 1.04x its own P=1 on TinyLlama and 1.02x on Qwen2.5-3B. CPU decode here is bound by
    instructions per weight, and the decode-batch matmul repeats the single-row kernel's work for each row by design,
    so its aggregate stays at one stream's.
  - The baseline at P=8 reads 0.92x and 0.90x its own P=1, and at P=4 0.83x and 0.82x.
- **The candidate beats the gemm end to end, although the gemm wins in isolation. Not investigated.**
  - In the microbenchmark below (run 1's Q4_K lines and run 2), the gemm takes 0.55x to 0.91x the decode-batch
    matmul's time at batch 4 and 8.
  - End to end, the baseline's batched decode is slower than its own single-request decode.
  - One possible cause is the gemm's sensitivity to what ran before it. In run 1, the Q6_K gemm took 15 ms per call
    at batch 4 after the Q4_K shapes; in run 2, with Q6_K first, it took 3.3 ms. In the server, the same gemm also
    runs every prefill.
  - The A/B is the measurement that counts; the microbenchmark only motivated it.

## Replies: batched against single

Every session sends the same prompt, greedy, so each reply should equal its build's single reply (P=1, repetition
1). The two builds' single replies are identical. Replies that differ from the single reply:

| Model | Build | P=2 | P=4 | P=8 |
|---|---|---|---|---|
| TinyLlama 1.1B | baseline | 0 of 6 | 6 of 12 | 8 of 24 |
| TinyLlama 1.1B | candidate | 0 of 6 | **0 of 12** | **0 of 24** |
| Qwen2.5-3B | baseline | 0 of 6 | 6 of 12 | 7 of 24 |
| Qwen2.5-3B | candidate | 0 of 6 | **0 of 12** | **0 of 24** |

The baseline's batch dependence is not limited to the kit probe's near tie. On this ordinary prompt, half of its
batched replies at P=4 differ from the single reply, and a third at P=8. That fits the earlier finding: a reply
changes when its decode steps switch between the two kernels, and which steps run batched depends on when each
session arrives. The candidate's 84 batched replies all equal the single reply.

## Kernel microbenchmark (before the A/B)

`CpuDecodeBatchMicrobench`, through `scripts/performance-tests/cpu-decode-batch-kernels.sh`: 12 threads, 128-bit
lanes, three repetitions of about 600 ms each, unpinned. Variants per shape and batch size:

- `gemm`: `CpuIntegerGemm`, the baseline's batched decode;
- `decode-batch`: `CpuIntegerDecodeBatch`;
- `single-calls`: one `CpuIntegerKernels` call per row.

Every report has a column that says whether each row is bit-identical to the single-row kernel: `yes` for
decode-batch and single-calls on every line, `no` for the gemm.

- `kernel/run1-threads12.md`: TinyLlama's and Qwen2.5-3B's decode shapes at batches 1, 2, 4 and 8. Its Q6_K gemm
  rows (15 to 60 ms) are the call-history effect described above, not the kernel's speed.
- `kernel/run2-threads12.md`: Q6_K and Q4_K at 2048x5632, batches 4, 16 and 64, Q6_K first. The gemm leads by 1.2x
  to 1.5x at batch 4 and 2.3x to 3.0x at batch 64.

Raw lines: `kernel/run*-threads12.jsonl`.

## LoRA gate

The change touches batching, so `compare-lora.sh` runs as `CLAUDE.md` requires. LoRA training's frozen forward uses
the float-activation kernels, and playback decodes one request, so neither reaches the changed code: both are
expected unchanged.

**Method.** `ab-lora.sh` (copied here) runs each tree's own `compare-lora.sh --skip-build --no-publish`:

- three runs per side, A B on odd runs and B A on even;
- the GPU lane first, then the CPU lane;
- 2026-10-29, 18:31 to 19:59 UTC.

The baseline tree is the `git archive` of `06d6d22d` the baseline jar was built from, with `models/` linked in. Its
first attempt failed with no model found and is not counted.

TinyLlama Q4_K_M, rank 8, q and v, to loss 1.2:

| Lane | Train ms per pass, A | B | Time B/A (gate <= 1.25) | Playback tps, A | B | B/A (gate >= 0.80) |
|---|---|---|---|---|---|---|
| GPU | 2733 / 2667 / 2733 | 2800 / 2733 / 2733 | **1.000** | 12.50 / 13.08 / 11.97 | 12.64 / 13.06 / 12.96 | **1.037** |
| CPU | 49267 / 49067 / 54200 | 49267 / 49733 / 55933 | **1.009** | 0.881 / 0.877 / 0.819 | 0.874 / 0.871 / 0.739 | **0.993** |

Every run trains in 15 passes to the target loss and recalls "My name is Juno". All four gates are met. Each side's
third CPU run is the slow one: 54.2 and 55.9 s per pass. Medians of three, unrounded ratios; per-run `current.json`
and `lora-perf.json` are in `lora/`.

## Tests

Each test failed before the change and passes after it.

- `CpuIntegerDecodeBatchTest`, 31 cases. Every row is checked bit for bit against `CpuIntegerKernels` and
  `CpuIntegerKernels256`: per format (Q4_K, Q5_K, Q6_K, Q8_0), at batch sizes 1 to 9 and both lane widths. It also
  checks row ranges, scratch reuse after a larger call, and concurrent callers.
- The handlers' multi-decode tests.
  - Llama, on random Q4_K, Q5_K, Q6_K and Q8_0 weights: serial forward against batched decode, logits bit for bit.
  - Phi-2, Phi-3, Qwen3 and Qwen3-MoE: call counters show the decode-batch matmul ran and the gemm did not.
- `CpuDecodeBatchLiveTest`, on real weights: three requests at different positions, three steps.
  - Models: TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Qwen3-1.7B, and moondream2's Phi-2 backbone (Q5_K).
  - Before the change, every case failed at its first step, with logits differing by up to 0.12 to 0.24.
  - After the change, every case is bit for bit equal.
  - The Qwen3-Coder-30B-A3B case is opt-in and was not run.
- `TinyLlamaStaticBatchLiveTest` (`coordinator`), now also run in its own execution with the Vector API module. In
  its [colour, spider] round, the pre-change `node` jar gave " eight" (9475) at token 4 where the single request
  gives " six" (4832); the changed `node` gives the single reply.

**Full unit suite: 0 failures.**

| Module | Tests |
|---|---|
| tokenizer | 93 |
| lora | 116 |
| kvcache | 105 |
| health | 23 |
| node | 1242 (48 skipped) |
| registry | 109 (2 skipped) |
| sampler | 89 |
| coordinator | 398 (1 skipped), plus 1 in the integer-kernel execution |
| vision | 97 |
| metrics | 86 |
| juno-player | 127 |

## Files

- `ab-parallel.sh`: the throughput A/B driver.
- `parallel/progress.txt`: every reading, in run order.
- `parallel/<model>/p<P>/<A|B>-rep<N>/parallel-<P>.json` and `replies/resp-*.json`: per-run results and replies,
  with local paths replaced by `<repo>`, `<run-dir>` and `<scratch>`.
- `kernel/`: the microbenchmark reports and raw lines.
- `ab-lora.sh`, `lora/progress.txt`, `lora/lora-<lane>-<A|B>-<run>/`: the LoRA gate driver, run order, and per-run
  results.
