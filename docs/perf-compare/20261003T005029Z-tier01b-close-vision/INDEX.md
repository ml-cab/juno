# Vision gate after the prefill-throughput work (2026-10-03)

Purpose: the closing vision regression gate for the prefill-throughput work, which changed batched
dispatch, the GPU prefill path and `MatVec` (`sgemmInto`). Gate: `latency_ms` <= 1.25x and decode tps
>= 0.80x the baseline, median of three. **Unpinned**, agent-run; a gate this loose may be read across
unpinned runs, and both builds were alternated in one session.

Command, each run: `compare-vision.sh --gpu --skip-build --no-publish`, moondream2 Q5_K llamafile, the
standard image and prompt (*What is in this image?*), `max_tokens=32`, single-token prefill (the script's
default). Run in the order HEAD, baseline, HEAD, baseline, HEAD, baseline (2026-10-02 19:50 to 20:47
-0500), each from its own tree.

Builds: HEAD `1ac490a` (shaded jar sha256 `66f02ee7c2908f78`; the working tree carried only test and
script changes). Baseline `ffd0ca7`, the build the 2026-09-30 reference sweeps measured, rebuilt from an
export of that commit (no git metadata there, so its JSON reads `git_commit: unknown`; jar
`bd19a9306ea7901f`). Directories `head-N` and `base-N` hold each run's `current.json`, JFR summary,
response and INDEX.

| Run | HEAD latency ms | HEAD decode t/s | Baseline latency ms | Baseline decode t/s |
|---|---|---|---|---|
| 1 | 478,593 | 1.478 | 479,501 | 1.467 |
| 2 | 473,575 | 1.494 | 475,779 | 1.441 |
| 3 | 473,238 | 1.486 | 473,631 | 1.489 |
| **Median** | **473,575** | **1.486** | **475,779** | **1.467** |

**Gate met:** latency 0.995x (limit <= 1.25x), decode 1.013x (limit >= 0.80x). All six captions are
byte-identical.

Why nothing moved: moondream2's text model is Phi-2, whose handler runs its matmuls and attention on the
CPU on every backend (announced at startup), and its prefill here is single-token, so neither the
prefill-window device region nor the batched GEMM path is reached; `juno.MatVec.backend.cuda.count` is 0
in every run. The gate confirms that the shared changes (`sgemmInto`, the dispatch and handler wiring)
left this path's output and speed unchanged.
