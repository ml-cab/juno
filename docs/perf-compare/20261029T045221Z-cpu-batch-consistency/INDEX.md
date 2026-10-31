# CPU static batching: greedy output against single requests (2026-10-28)

**Purpose.** The cause of the intermittent failure of the kit's `consistency.tinyllama` probe: with `--parallel 2`
on the CPU, TinyLlama's greedy answer to "How many legs does a spider have?" was "A spider has six legs." alone and
sometimes "A spider has eight legs." in a batch (2 of 4 probe runs). The question was whether a batched request reads
KV it did not write, or whether batch composition changes its arithmetic.

**Pinned: no.** No reading here is a throughput figure; greedy tokens and logits do not depend on clocks.

**Build.** The kit's shaded jar (`juno-kit/target/juno-kit-0.1.2-shaded.jar`, sha256 first 16 `aec5cfe9571fea81`),
built from `06d6d22d` plus the uncommitted kit and `sampler` changes of 2026-10-28; no engine source differs from
`06d6d22d` except `GbnfGrammar` (not on this path: no grammar is used). Host: 12 CPU threads, JDK 25.0.3, the
default integer-kernel lane width. Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`.

## Method

Two single-file programs run against the jar in one JVM, through the same `GenerationLoop`, `LocalInferencePipeline`
and `LlamaTransformerHandler` the server uses (system prompt and prompts as the probe sends them, greedy, 8 tokens):

- `BatchPairs.java`: each prompt's single-request answer (`generate`), then every ordered pair of the three prompts as
  one static batch (`generateBatch`, run one after another), then pairs of batches run at the same time on two
  threads, as the scheduler runs them (3 repetitions of 3 pair sets).
- `KernelSwitch.java`: the spider prompt alone, with a pipeline wrapper that sends chosen decode steps through the
  batched decode path (`forwardMultiDecode` with one row, which runs `CpuIntegerGemm`) instead of the single-request
  path (`forward`, which runs `CpuIntegerKernels`); the top two logits at every step.

```
java -Xmx4g --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp juno-kit/target/juno-kit-0.1.2-shaded.jar BatchPairs.java models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 3
java -Xmx4g --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp juno-kit/target/juno-kit-0.1.2-shaded.jar KernelSwitch.java models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 2
```

Each was run again with `-Djuno.cpu.integerKernels=false` (the float-activation kernels; `BatchPairs` with 1
repetition). About 2 minutes per run. Result lines: `batch-pairs-integer.txt`, `batch-pairs-float.txt`,
`kernel-switch-integer.txt`, `kernel-switch-float.txt`.

## Batch composition (`BatchPairs`)

Single answers: "The capital of France is Paris." (7 tokens), "Red." (3 tokens), "A spider has six legs." (7 tokens).

| Kernels | Sequential pairs that differ from single (of 9) | Concurrent batches that differ (of 18 rows) |
|---|---|---|
| integer (default) | 2: [colour, spider] and [spider, colour], the spider row, at token 4: "eight" | 0 |
| float-activation | 0 | 0 (6 rows) |

The spider row matches its single answer when its partner is the France prompt or another spider request. Only the
colour partner changes it, and that partner ends after 3 tokens and EOS: from decode step 4 the batch has one active
row, and `LocalInferencePipeline.forwardBatch` sends a one-row batch through `forward`, the single-request path.

## Kernel switch (`KernelSwitch`), the spider prompt

Logits of " six" and " eight" at token 4 (the only position where the answers differ; at every other step, in every
row, the top token leads the second by at least 0.255, the closest being the final "." at token 6):

| Decode steps through the batched path | Integer kernels: answer, " six" / " eight" | Float-activation kernels: answer, " six" / " eight" |
|---|---|---|
| none (single request) | six, 13.85606 / 13.84674 (gap 0.00931) | six, 13.85313 / 13.82737 |
| all | six, 13.88527 / 13.85629 | six, 13.85313 / 13.82737 |
| first 1 | **eight**, 13.83835 / 13.86614 | six, 13.85313 / 13.82736 |
| first 2 | **eight**, 13.84709 / 13.88993 | six, 13.85312 / 13.82738 |
| first 3 | **eight**, 13.86103 / 13.87625 | six, 13.85314 / 13.82737 |
| first 4 (the [colour, spider] batch) | **eight**, 13.84346 / 13.87179 | six, 13.85314 / 13.82739 |
| first 5, first 6 | six | six |

"First 4" reproduces the batched answer exactly; "all" reproduces the [France, spider] batch.

## Finding

Not a KV leak. A batched row's arithmetic does not depend on the other rows (`CpuIntegerGemmTest` already pins that
bit for bit, padding included), and the batched answers are reproduced by a single request whose decode steps switch
between the two kernels at the same step. The two integer kernels agree to about 1e-5 of the largest product per
matmul (the bound `CpuIntegerGemmTest` checks), not bit for bit; because every layer re-quantizes its input to 8 bits,
a last-bit difference sometimes lands on the other side of a rounding step, and by the logits the two kernels differ
by about 0.03 to 0.05. That flips the 0.009 tie between " six" and " eight". With the float-activation kernels the
two paths differ by about 1e-5 at the logits and the tie holds.

Which decode steps of a request run batched depends on how many requests are active at each step, so on the CPU's
default kernels a greedy answer at a near tie depends on what else the server is serving. The probe's failure was
intermittent because which prompts share a batch depends on arrival order.

Not covered: the GPU (its consistency check passed), the continuous schedule (its decode batch also changes size as
requests finish), other models.
