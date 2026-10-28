# CPU kernels: model quality and greedy output against the pre-tier build (2026-10-26)

**Purpose.** The quality half of the CPU decode and prefill kernels work, read on the final build before the closing
gate's run 2: mean negative log-likelihood of the integer kernels against the float-activation kernels on the four
sweep models (threshold: at most 1.01x), and every greedy reply of the CPU smoke compared with the build before the
work.

**Pinned: no.** Neither reading is a throughput figure: NLL and greedy text do not depend on clocks.

**Builds.**

| Role | Jar sha256 (first 16) | Build |
|---|---|---|
| pre-tier (smoke baseline) | `b9a0440b3a1b597a` | HEAD `01027ee9`, before the CPU kernel work |
| candidate | `4e480ab9131887eb` | `c2633a68` plus the in-recording allocation field (the padded prefill tile of 2026-10-14 included) |

The NLL test runs from the tree's compiled classes (`node` module), the same source as the candidate jar.

## Model quality: `ModelQualityNllLiveTest`

Command: `mvn test -pl node -Dtest=ModelQualityNllLiveTest -Djuno.test.modelQuality=true` (4 tests, 0 failures,
148 min). 2,048 tokens per model; the same text scored by the default integer kernels and by the float-activation
kernels. Raw lines: `nll.txt`.

| Model | NLL, integer | NLL, float | Ratio (bound <= 1.01) | Argmax agreement, first 512 |
|---|---|---|---|---|
| TinyLlama 1.1B | 2.45609 | 2.45566 | **1.00017** | 507 (99.0%) |
| Qwen2.5-3B | 1.69656 | 1.69766 | **0.99935** | 512 (100%) |
| Phi-3.5-mini | 1.36110 | 1.35826 | **1.00209** | 509 (99.4%) |
| Mistral 7B | 1.53013 | 1.52891 | **1.00080** | 510 (99.6%) |

## Greedy output: `smoke-cpu-kernels.sh`

Command: `scripts/performance-tests/smoke-cpu-kernels.sh --baseline-jar <pre-tier jar>` (78 min). It runs
`smoke-cpu-hot-path.sh` with each jar: the four sweep models at `--threads` 4 and 12, a short decode and a prompt of
about 512 tokens each, and the moondream2 CPU vision case (the prefill width of about 741). Both runs pass every check,
and **all 18 greedy replies are identical between the builds** (0 different). Per-reply texts:
`smoke-cpu-kernels-compare.json` (local paths removed).

## Context shift on the CPU: check 10 (added 2026-10-27)

Check 10 of `ModelLiveChecks` (the check `ContextShiftCheckTest` runs) on TinyLlama 1.1B on the CPU backend: the
handler's context limit is lowered to 192 positions, and a 191-token prompt asks for 200 tokens with context shifting
on, on the static schedule and then the continuous one. Run from `juno-master/target/juno-master.jar`, built in the same
`mvn package` as the candidate jar (its `CpuIntegerKernels` and `CpuIntegerGemm` classes are byte-identical to the
candidate's), with CUDA hidden so the check takes the CPU backend, and the Vector API module on so the integer kernels
run (the log's `CPU quantized matmul: integer-dot kernels for Q4_K, Q5_K, Q6_K, Q8_0` line, 12 threads). About 65 s.
Log: `context-shift-cpu.log`.

Command, with a driver whose `main` calls `ModelLiveChecks.run(args[0], EnumSet.of(Suite.CONTEXT_SHIFT), System.out)`
and exits 0 when every result passed:

```
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= java --enable-preview --source 25 --enable-native-access=ALL-UNNAMED \
  --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.nio=ALL-UNNAMED \
  --add-modules jdk.incubator.vector -XX:+UseG1GC -Xmx8g -cp juno-master/target/juno-master.jar ShiftOnly.java \
  models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

| Schedule | Tokens generated (of 200) | Shifts | Highest position (limit 192) |
|---|---|---|---|
| static | 200 | 3 | 191 |
| continuous | 200 | 3 | 191 |

**PASS**: every requested token came back, the request shifted, and no position reached the limit.

`smoke-context-policy.sh --cpu` is not the CPU check here: its limit is Phi-3.5-mini's 4,096 positions (the one model
whose handler refuses positions below the KV cap), and an earlier CPU run of it at that length was stopped after 76
minutes. On TinyLlama the script reaches no limit, since TinyLlama's limit is the 32,768-position KV cap.

## How to read it

- Every NLL ratio is inside the bound. Phi-3.5-mini moved most (1.00209; 0.99935 to 1.00061 across the four models
  after the earlier integer-kernel work): this work scales sub-block sums in float lanes and computes the prefill
  attention dot in vector lanes, both of which change float rounding by design. `host-specific` magnitudes.
- Identical greedy replies on every case are a stronger result than the smoke requires (it reports a different reply
  rather than failing it, since rounding may move a greedy token); they hold for these prompts, not in general.
