# CPU decode allocation after the decode scratch (2026-10-10)

**Purpose.** The allocation rate of steady-state CPU decode after the per-thread decode scratch, `MatVec.sgemvInto`,
and the removal of a per-call KV allocation, against the pre-fix baseline (20.8 MB per token on Mistral 7B,
`../20261010T032108Z-cpu-cost-breakdown/`), and a CPU sweep confirming throughput is unchanged.

**Pinned: no.** CPU governor `schedutil`, turbo on.

**Build.** Juno `aabc42cb` plus the uncommitted CPU work through the allocation fix, jar sha256
`c009f616d94d2523`, OpenJDK 25.0.3. Reference tool build `379ac6673`, CPU, `-t 12`.

## Commands

```bash
scripts/performance-tests/compare-llama-cpp.sh --cpu \
    --models tinyllama-1.1b-chat-v1.0.Q4_K_M,qwen2.5-3b,Phi-3.5-mini,mistral-7b --n-prompt 128 --n-gen 64 \
    --threads 12 --reps 3 --juno-reps 3 --juno-warmup 2 --jfr 30m --no-publish --out <run>
java scripts/performance-tests/JfrSpanBreakdown.java --top 8 <run>/*-measured.jfr > jfr-span-breakdown.jsonl
```

## Steady-state decode allocation (JFR `jdk.ThreadAllocationStatistics`, decode span only, median of three)

| Model | Before (step 3 or item 1 reading) | Now | Change | GC pauses in the decode span |
|---|---|---|---|---|
| Mistral 7B | 20.8 MB per token | **0.87 MB** (0.862 to 0.867) | **-95.8%** | 0 |
| Phi-3.5-mini | 35.2 MB | 1.92 MB | -94.5% | 0 (was 3) |
| Qwen2.5-3B | 12.1 MB (item 1 reading) | 1.40 MB | -88% | 0 |
| TinyLlama 1.1B | 5.6 MB (item 1 reading) | 0.38 MB | -93% | 0 |

The tier's allocation threshold (a drop of at least 50% on Mistral 7B steady-state decode) is met. Measured without
JFR, through the JVM's thread allocation counters, decode allocates 0.17 to 0.21 MB per token on TinyLlama, Mistral 7B
and Phi-3.5-mini and 0.65 MB on Qwen2.5-3B, whose 151,936-entry logits array is most of it; the rest of the JFR figure
is the recording itself and the HTTP and sampling path.

What was removed: per-layer arrays in the single-token CPU decode path of the LLaMA-family and Phi-3 handlers (now
per-thread buffers, `CpuDecodeScratch`), and a full set of KV tensors that every handler built and discarded on each
call for a request that already had one (about 8.4 MB per token on Mistral 7B).

## Throughput (Juno t/s, median of three; unchanged within noise)

| Model | pp | tg | pp ratio | tg ratio |
|---|---|---|---|---|
| TinyLlama 1.1B | 16.62 | 7.88 | 0.240x | 0.362x |
| Qwen2.5-3B | 6.18 | 2.62 | 0.233x | 0.244x |
| Phi-3.5-mini | 4.93 | 2.26 | 0.254x | 0.251x |
| Mistral 7B | 2.40 | 1.21 | 0.221x | 0.360x |

Against the previous reading (`../20261010T145223Z-cpu-threads/`) Juno moves -7% to +4% per row; the reference
tool moved as much in the same direction this session (Mistral 7B: 10.89 against 11.49 pp, 3.38 against 3.04 tg), which is
within this host's unpinned noise. Prefill allocation (about 100 MB per token on Mistral 7B, from per-window
workspaces and activation buffers) is not part of this threshold and is unchanged.
