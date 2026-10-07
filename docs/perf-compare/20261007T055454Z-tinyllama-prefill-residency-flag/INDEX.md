# TinyLlama prefill with the decode residency flag off and on - 20261007T055454Z (gpu)

**Purpose.** Two pinned A/B gates of the decode residency region read TinyLlama's prefill on/off at 0.901
(`20261006T080204Z-gpu-residency-whole-layer-ab`) and 0.913 (`20261006T200344Z-gpu-residency-phi3-qwen3-ab`),
although the decode region does not run in a prefill window. This pair reads what the 128-token prefill
window does under each setting, with device spans, to attribute it.

**Pinned: no.** Bytes and the breakdown's terms are the question; the t/s figures are unpinned, with span
recording on, one session, a reading only.

**Command** (one jar, `off` then `on`):

```
scripts/performance-tests/compare-llama-cpp.sh --gpu --device-spans --juno-reps 3 --no-tuned-lane \
  --n-prompt 128 --gpu-residency off|on --models tinyllama-1.1b-chat-v1.0.Q4_K_M \
  --juno-jar dist/gpu-residency-final-region-ab/candidate-shaded.jar
scripts/performance-tests/prefill-breakdown.sh <run dir> --json prefill-breakdown.json > prefill-breakdown.md
```

Jar: HEAD `d05d1fe` built on a clean tree, sha256 `652bf47e12609973` (the harness notes "plus uncommitted
changes" because published result files had been added to the working tree by then; no source had changed).
Copied by hand from the harness output, JSON and Markdown only, local paths made repository-relative.
`off/` and `on/` hold each run; `RUN-INDEX.md` in each is the harness's own index (`off` started 05:54:54Z,
`on` 05:56:18Z).

## Reading

| Setting | pp t/s (3 reps) | Median | Prefill H2D / D2H MB | Prefill forward pass ms (3 reps) |
|---|---|---|---|---|
| off | 2220.7, 2097.9, 2077.3 | 2097.9 | 1.10 / 6.76 | 57.6, 61.0, 61.6 |
| on | 2206.4, 2334.7, 2232.8 | 2232.8 | 1.10 / 6.76 | 58.0, 54.8, 57.3 |

pp on/off **1.064**. Bytes identical on all six repetitions. Breakdown (median ms, off / on): GEMM compute
33.0 / 29.5, device elementwise 8.9 / 8.7, region attention 4.2 / 4.0, projection and region host 4.8 / 5.2,
KV host write 0.8 / 0.8, LM head 1.0 / 1.2, residue 7.1 / 5.7 (both runs trip the breakdown's 5% residue flag,
11.7% and 10.0%; reported, not relaxed).

**How to read it.** The prefill window does the same work under either setting: the same copies, the same
terms, within each side's own spread. The 0.90 and 0.91 readings are not reproduced here (1.064 the other
way) and are read as TinyLlama's prefill spread at `n_prompt=128`, where a window takes about 60 ms and the
gates' off side spread 13% by itself. The tier's closing pinned no-regression gate reads it again.
