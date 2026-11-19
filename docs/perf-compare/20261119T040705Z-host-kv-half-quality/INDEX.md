# f16 host KV (IEEE halves): quality against f32 host KV (deterministic, not a timing)

**Purpose.** Score real FP16 host KV storage against the quality bars of the plan's FP16 dense KV criterion (greedy
agreement >= 99%, NLL <= 1.005x) on the four sweep models. The `f16` cache type now stores each value as its IEEE half
(two bytes); the float32 storage it used before is kept as the `f32` type. Both run in one build, so the comparison
isolates the storage format.

**Pinned:** no, and not needed: the readings are deterministic (CPU backend, fixed text, greedy argmax), not timings.

**Host:** Xeon E5-1650 v2 (6 cores, 12 threads, AVX, no AVX2), 64 GiB. CPU backend only, 12 matmul threads, integer-dot
kernels for Q4_K, Q5_K, Q6_K, Q8_0.

**Build.** One tree: HEAD `c607e926` plus the uncommitted real FP16 storage and growth toward the request's length,
compiled by the run itself (`mvn test -am`). Its shaded player jar at the run's start: `10c5f4533d58e47d` (SHA-256,
first 16). Its production sources are those of the candidate in
[`20261118T232702Z-host-kv-half-ab`](../20261118T232702Z-host-kv-half-ab/INDEX.md) (jar `244271ec4dd6b154`); the jars
differ only by rebuild, since jar entries carry their build time.

## Command

```bash
JAVA_TOOL_OPTIONS=-Xmx24g mvn -o test -pl node -am -Dsurefire.failIfNoSpecifiedTests=false \
  -Dtest=HostKvHalfQualityLiveTest -Djuno.test.hostKvHalfQuality=true
```

`HostKvHalfQualityLiveTest` (node). Per model, one handler per KV type, `f32` first, then `f16` (the type is read when
the handler is built; each handler's policy line is in the log). The readings are those of the KV quality test used for
`q8_0` and `q4_0` ([`20261113T051804Z-kv-q4-0`](../20261113T051804Z-kv-q4-0/INDEX.md)), with `f32` as the reference:
- **NLL**: mean teacher-forced NLL over the fixed 2,048-token quality text (`model-quality-nll.txt`, its per-model
  ids), windows of 128.
- **Greedy agreement**: the `f32` handler continues the text's first 1,024 tokens greedily for 512 tokens; that
  sequence is teacher-forced through both handlers in the same windows, and the agreement is the share of the 512
  generated positions where their argmax agree.
- **First disagreement**: the first generated position (counted from 0) where the two disagree, where a free-running
  `f16` reply would first leave the `f32` one. **Text agreement**: argmax agreement over the text's first 512
  positions.

## Results

| Model | NLL f32 | NLL f16 | Ratio (<= 1.005) | Greedy agreement (>= 99%) | First disagreement | Text agreement | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| TinyLlama 1.1B | 2.45609 | 2.45529 | 0.99967 | 510/512 = 99.61% | 19 | 509/512 = 99.4% | met |
| Qwen2.5-3B | 1.69656 | 1.69746 | 1.00053 | 507/512 = 99.02% | 1 | 511/512 = 99.8% | met |
| Phi-3.5-mini | 1.36110 | 1.36087 | 0.99983 | 510/512 = 99.61% | 4 | 511/512 = 99.8% | met |
| Mistral 7B | 1.53013 | 1.52983 | 0.99980 | 511/512 = 99.80% | 35 | 512/512 = 100.0% | met |

Surefire: 4 tests, 0 failures (the test asserts both bars per model). Wall time 2 h 47 min. Log:
`quality/host-kv-half-quality.log` (each handler's load and cache-type policy line, the result lines, the surefire
summary). The `s_f32` and `s_f16` fields are not comparable: `s_f32` includes the greedy generation pass, `s_f16` does
not.

## How to read it

- **Both bars are met on all four models.** The NLL moves by at most 0.053% (Qwen2.5-3B, up) and goes down on the other
  three; the bar allows 0.5%.
- **Qwen2.5-3B meets the agreement bar with no margin.** 507 of 512 is the smallest count at or above 99%; one more
  disagreement would read 98.83%. Its text agreement (511 of 512) and NLL ratio are as close as the other models'. The
  same model with `q8_0` KV read 508 of 512, 99.2% text agreement and NLL 1.00073x against float32 KV in the 2026-11-13
  run; halves move the NLL less than `q8_0` did and agree on more of the text, so the 507 against 508 reads as the
  count's noise, not a weaker format. Its first disagreement is at generated position 1: on this text a free-running
  `f16` reply would leave the `f32` one at its second token (the margins of these argmax are not recorded).
- **The `f32` side is the storage `f16` had before.** Its NLL on every model equals, to the fifth decimal, the `f16`
  reference NLL published in `20261113T051804Z-kv-q4-0`, measured when `f16` still stored float32 rows.
- **What this does not cover:** throughput and allocation (in `20261118T232702Z-host-kv-half-ab`), the memory footprint
  (in `20261118T220603Z-host-kv-footprint`), and the GPU's attention, which reads the device's FP16 copy under both
  types and so is unchanged by this storage.
