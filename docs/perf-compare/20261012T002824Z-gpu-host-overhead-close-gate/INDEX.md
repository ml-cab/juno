# GPU host overhead: closing gate (2026-10-12)

**Purpose.** The closing no-regression gates for the GPU host-overhead work: Qwen2 and Qwen2.5 in the device-resident
decode region, host KV tensors reused across requests, and the GPU prefill window writing K and V into the host KV cache
straight from its download. Part A is the GPU lane against the pre-work build, and Part C is the standing CPU and allocation
gate. The closing sweeps against the reference tool (Part B) are published by the harness as `20261012T013642Z` (128
tokens), `20261012T015002Z` (512), `20261012T020410Z` (2048) and `20261012T021800Z` (2048, Phi-3.5-mini at an 8 GiB heap).

**Pinned: yes.** Owner run (prompt-free sudo), `compare-llama-cpp.sh --pin-clocks` on every invocation: performance
governor, turbo off. `clock_pinned` is true on all 12 A/B runs and all four sweeps. Started 2026-10-12 00:28 UTC.

**Builds.**
- Baseline: `198467ad`, the last build before this work, jar sha256 `6ae135b0e21839cf`.
- Candidate: `198467ad` plus the uncommitted work, jar sha256 `921ab742ba3f362a`, byte-identical to the tree's own shaded
  jar.
- OpenJDK 25.0.3; reference tool build `ac4cdde`.

## Command

```bash
bash dist/gpu-host-overhead-close/wait-and-run.sh      # runs run-gate.sh, PART=ACB; log: gate.log
```

Each A/B invocation: `compare-llama-cpp.sh <lane> --pin-clocks --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane
--no-publish --juno-jar <jar>`, A B A B A B, and the gate reads medians of three.

## Part A: GPU no-regression (four sweep models, `--n-prompt 512`)

| Model | pp t/s A | pp t/s B | pp B/A | tg t/s A | tg t/s B | tg B/A | Allocation per token B/A | GC pause in span (A, B) | Greedy output |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 3015.9 | 3107.8 | 1.030 | 148.88 | 148.76 | 0.999 | 0.992 | 0.0, 0.0 ms | equal in 3 of 3 |
| Qwen2.5-3B | 1214.1 | 1238.3 | 1.020 | 31.23 | 57.55 | **1.843** | 0.903 | 0.0, 0.0 ms | differs in 3 of 3 (below) |
| Phi-3.5-mini | 682.7 | 815.5 | **1.195** | 56.22 | 56.25 | 1.001 | 0.955 | 0.0, 0.0 ms | equal in 3 of 3 |
| Mistral 7B | 537.8 | 546.9 | 1.017 | 37.98 | 38.25 | 1.007 | 0.991 | 0.0, 0.0 ms | equal in 3 of 3 |

Gate: tg and pp at least 0.95x, allocation at most 1.10x, and GC pause at most 5 ms where the baseline is under 5 ms.
**Met on every model.**

- Qwen2.5-3B generation is 1.84x: it now decodes in the device region.
- Phi-3.5-mini prefill is 1.20x: the host KV reuse and the direct K/V write.
- Qwen2.5-3B's candidate reply is identical across its three runs and agrees with the baseline's for its first 230
  characters, about 48 tokens. It then takes the other of two near-equal continuations: the region's RMS norms sum in a
  different order (GPU against CPU). The baseline's own three runs agree with each other.

## Part C: standing CPU and allocation gate (TinyLlama and Mistral 7B, `--cpu --n-prompt 128`)

| Model | pp t/s A | pp t/s B | pp B/A | tg t/s A | tg t/s B | tg B/A | Allocation per token B/A | GC pause in span (A, B) | Greedy output |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 16.60 | 16.31 | 0.983 | 7.67 | 7.66 | 0.999 | 0.966 | 0.0, 0.0 ms | equal in 3 of 3 |
| Mistral 7B | 2.52 | 2.59 | 1.030 | 1.26 | 1.27 | 1.005 | 0.920 | 0.0, 0.0 ms | equal in 3 of 3 |

Gate (README standing CPU gate: tg and pp at least 0.95x; allocation and GC as in Part A): **met**.

The harness's span check withheld Mistral 7B's generation on one baseline run. The tg median is then over two baseline
and three candidate runs. The request-timed `api_token_gen_tps` medians of three, 1.1565 and 1.1645 t/s (1.007x), agree.

## Hot methods (`jdk.ExecutionSample`, run 1 of each jar; reported, not gated)

Full top-10 lists for both lanes, both jars, every model, are in `gate.log`.

- **GPU lane:** the frames shifted where the work moved. Qwen2.5-3B's baseline was led by host
  `LlamaTransformerHandler.ffn` (35 samples). The candidate has none of it and is led by `SoftmaxStep.apply`, the
  full-vocabulary greedy softmax left to the sampling work. On the other three models, the GPU lane's few host samples
  look alike on both jars.
- **CPU lane:** the same kernels in the same order on both jars: `CpuIntegerKernels.q6kRow` and `q4kRow`, the worker
  pool, `CpuIntegerGemm.tile`.

## Closing sweeps (Part B), summary

Ratios against the reference tool, default lane, every row scorable, prompt parity exact:

| Model | tg, 128 | pp, 128 | pp, 512 | pp, 2048 | pp 512 over 128 |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 0.776x | 0.681x | 0.782x | 0.713x | 1.149 |
| Qwen2.5-3B | 0.828x | 0.760x | 0.833x | 0.680x | 1.097 |
| Phi-3.5-mini | 0.973x | 0.695x | 0.726x | 0.602x | 1.044 |
| Mistral 7B | 1.066x | 0.785x | 0.813x | 0.730x | 1.036 |

## How to read it

- A is the baseline and B the candidate; B/A over 1 is faster, and an allocation ratio under 1 allocates less.
- Per-run `host.json`, `*-juno.json`, `*-compare.json` and the generated reply are under `gpu-ab/` and `cpu-ab/`, as
  `<side>-<run>/`.
- Local paths are replaced by `<repo>`, `<home>` and `<scratch>`.
