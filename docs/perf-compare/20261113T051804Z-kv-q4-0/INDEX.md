# q4_0 host KV: quality against f16 KV, and the read path's cost (unpinned)

**Purpose.** Score the `q4_0` KV cache type (`--cache-type-k q4_0 --cache-type-v q4_0`) against the plan's Q4_0 KV
quality gate on the four sweep models, and check that routing the quantized slice decode through one dispatch
(`KvRows.decodeRange`) costs the existing `f16` and `q8_0` paths nothing.

**Pinned:** no. Nothing here is scored against a gate at 0.90x or tighter: the quality readings are deterministic (CPU
backend, fixed text, greedy), and the throughput A/B is evidence for the standing gate, whose pinned run is the owner's
at tier close.

**Host:** Xeon E5-1650 v2 (6 cores, 12 threads, AVX, no AVX2), 64 GiB, GTX 1080 (the GPU launch below only).

**Builds.**

| Side | Tree | Shaded jar (SHA-256, first 16) |
|---|---|---|
| A (baseline) | HEAD `6ad5c728` plus the uncommitted paged-KV reclaim and bounded GPU launches (the tree published in `20261112T181947Z-bounded-launches` as its kit-run jar) | `2e7ab50ca9c2cdf4` |
| B (candidate) | A plus the `q4_0` KV type (`Q4_0KvCodec`, `KvRowCodec`, `KvRows.decodeRange`, `DeviceKvTypeNotice`) | `9fc4975d8a843f6e` |

## 1. Quality: `q4_0` K and V against `f16` KV (gate: NLL <= 1.02x, greedy agreement >= 97%)

`KvQuantQualityLiveTest` (node, `-Djuno.test.kvQuality=true`, CPU backend so attention reads the host rows, 24 GiB heap),
the candidate tree:

```bash
JAVA_TOOL_OPTIONS=-Xmx24g mvn -o test -pl node -am -Dsurefire.failIfNoSpecifiedTests=false \
  -Dtest=KvQuantQualityLiveTest -Djuno.test.kvQuality=true
```

Per model, one handler per KV type (the type is read when the handler is built):
- **NLL**: mean teacher-forced NLL over the fixed 2,048-token quality text (`model-quality-nll.txt`, the integer-kernel
  quality test's text and ids), windows of 128 through `forwardVerify`.
- **Greedy agreement**: the `f16` handler continues the text's first 1,024 tokens greedily for 512 tokens; that sequence
  is teacher-forced through both handlers in the same windows, and the agreement is the share of the 512 generated
  positions where their argmax agree. `self` is the `f16` handler's teacher-forced argmax against the tokens it generated
  at decode width (the noise floor of reading both sides teacher-forced): 512 of 512 on every model.
- **First disagreement**: the first generated position where the two disagree, where a free-running `q4_0` reply would
  first leave the `f16` one. **Text agreement**: argmax agreement over the text's first 512 positions.

| Model | NLL f16 | NLL q4_0 | Ratio (<= 1.02) | Greedy agreement (>= 97%) | First disagreement | Text agreement | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| TinyLlama 1.1B | 2.45609 | 2.52993 | **1.03006** | 497/512 = 97.07% | 1 | 89.8% | NLL missed |
| Qwen2.5-3B | 1.69656 | 3.42284 | **2.01752** | **428/512 = 83.59%** | 1 | 63.5% | both missed |
| Phi-3.5-mini | 1.36110 | 1.40926 | **1.03538** | 502/512 = 98.05% | 4 | 96.1% | NLL missed |
| Mistral 7B | 1.53013 | 1.53949 | 1.00612 | 507/512 = 99.02% | 0 | 97.3% | met |

The `f16` NLL on TinyLlama (2.45609) is the integer-kernel quality reading published in
`20261027T025856Z-cpu-kernels-quality`, so the path scores what that test scores. Wall time: 7.5, 21.9, 102.8 and 42.6
minutes. Log: `quality/kv-quality-q4_0.log` (the result lines, the cache-type policy line of each handler, the
assertions).

**How to read it.** The gate is met on Mistral 7B only. Qwen2.5-3B collapses (perplexity 30.7 against 5.46); TinyLlama
and Phi-3.5-mini lose 3.0% and 3.5% NLL (perplexity +7.7% and +4.9%). This is the format, not the read path: the same
code reads `q8_0` and `f16` rows (the A/B below: replies equal), the in-place read equals attention over the gathered
rows bit for bit (`PagedKvAttentionOracleTest`, `GqaWindowAttentionTest`, `PagedKvHandlerOracleTest` with `q4_0`), the
codec gives the published Q4_0 block byte for byte on a hand-worked block, and its error energy over Gaussian rows sits
within 0.8x to 1.25x of the uniform-quantization model for 16 steps (`Q4_0KvCodecTest`). The likely cause, not
measured here: a block's one scale is set by its largest value, so four-bit steps lose most where a few key channels
are much larger than the rest; Qwen2's K projection carries a bias, and its keys are where the loss is largest.

## 2. Follow-up configurations

Same test, the candidate tree, two more configurations per model (the `f16` side recomputed in each run, and equal to
run 1's on every model):

```bash
JAVA_TOOL_OPTIONS=-Xmx24g mvn -o test -pl node -am -Dsurefire.failIfNoSpecifiedTests=false \
  -Dtest=KvQuantQualityLiveTest -Djuno.test.kvQuality=true -Djuno.test.kvQuality.types=q8_0,q8_0/q4_0 \
  -Djuno.test.kvQuality.models=tinyllama,qwen25,mistral        # then -Djuno.test.kvQuality.models=phi35
```

| Model | `q8_0` K and V: ratio, greedy, text | `q8_0` K with `q4_0` V: ratio, greedy, text |
|---|---|---|
| TinyLlama 1.1B | 0.99977, 511/512 = 99.80%, 98.8% | 1.00019, 506/512 = 98.83%, 97.9% |
| Qwen2.5-3B | 1.00073, 508/512 = 99.22%, 99.2% | 1.00607, 511/512 = 99.80%, 98.2% |
| Phi-3.5-mini | 0.99875, 511/512 = 99.80%, 99.4% | 1.00202, 510/512 = 99.61%, 99.0% |
| Mistral 7B | 0.99909, 511/512 = 99.80%, 99.6% | 0.99728, 510/512 = 99.61%, 98.4% |

**How to read it.** With 8-bit keys every model is within 0.7% of the `f16` NLL and agrees on at least 98.8% of the
greedy tokens, whatever the values are stored as; `q8_0` K with `q4_0` V meets the `q4_0` gate's bars (NLL <= 1.02x,
agreement >= 97%) on all four sweep models. The loss in section 1 is the keys'. Per position, `q8_0` K with `q4_0` V
stores 52 bytes for every 32 elements of a key row plus the matching 32 of its value row, against 68 for `q8_0` and 256
for float32 (0.76x and 0.20x); `q4_0` K and V stores 36 (0.14x of float32). Logs:
`quality/kv-quality-q8_0-and-q8_0-q4_0.log`, `quality/kv-quality-phi35-q8_0-and-q8_0-q4_0.log`.

## 3. Throughput A/B (unpinned): the read path through one dispatch

TinyLlama 1.1B Q4_K_M, CPU, `static`, `n_prompt` 512, A B A B A B per lane (B only for `q4_0`, which A rejects), each
invocation through the candidate tree's compare script with its side's jar:

```bash
scripts/performance-tests/compare-llama-cpp.sh --cpu --models tinyllama-1.1b-chat-v1.0.Q4_K_M --n-prompt 512 \
  [--cache-type-k q8_0 --cache-type-v q8_0 | --cache-type-k q4_0 --cache-type-v q4_0] \
  --juno-warmup 2 --juno-reps 1 --reps 1 --no-tuned-lane --no-publish --juno-jar <side's jar> --out <lane>-<side>-<i>
```

| Lane | pp t/s A (3) | pp t/s B (3) | pp B/A | tg t/s A | tg t/s B | tg B/A | Alloc/token in recording B/A | GC in span | Replies |
|---|---|---|---:|---|---|---:|---:|---|---|
| f16 | 34.08 (32.67, 34.12, 34.08) | 34.21 (34.40, 34.21, 32.35) | 1.004 | 12.00 (11.28, 12.00, 12.01) | 11.97 (11.97, 11.84, 12.00) | 0.997 | 1.000 | 0 / 0 ms | equal, 3 of 3 |
| q8_0 | 32.33 (32.30, 32.52, 32.33) | 32.36 (32.36, 31.25, 32.39) | 1.001 | 11.44 (11.50, 11.44, 11.37) | 11.38 (11.39, 11.38, 11.35) | 0.995 | 0.974 | 0 / 0 ms | equal, 3 of 3 |
| q4_0 (B) | | 30.86 (32.10, 30.34, 30.86) | 0.954 of B's q8_0 | | 11.10 (11.12, 11.10, 10.93) | 0.975 of B's q8_0 | 1.044 of B's q8_0 | 0 ms | differs from f16 from the first token |

Every A/B ratio sits inside the run's own spread; the standing gate's bars (0.95x pp and tg, 1.10x allocation) hold on
both lanes. `q4_0` decodes two nibbles per byte and reads about 5% slower at prefill than `q8_0` here, where the KV is
small (512 positions); its saving is memory. The generate lane's prompt is a calibrated run of raw words, so the
`q4_0` reply leaving the `f16` one there is the quality result above, not a separate finding. Raw per-run JSON:
`ab/<lane>-<side>-<i>/`, summary `ab/summary.txt`, driver `ab/run-ab.sh`.

## 4. Launch checks

`./juno local` on TinyLlama with `--cache-type-k q4_0 --cache-type-v q4_0`, greedy, "What is the capital of France?":
- GTX 1080: the startup warning that the type sets the host KV only (GPU attention reads its FP16 copy); reply "The
  capital of France is Paris." (`launch/launch-gpu-q4.log`).
- CPU, `static` (dense) and `continuous` (paged): no warning (attention reads the host rows); the same reply on both
  (`launch/launch-cpu-q4-*.log`).
