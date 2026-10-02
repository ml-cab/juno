# Prefill chunk size per surface: review readings (2026-10-02)

Purpose: decide the default `--prefill-batch` on every surface that still used the fixed 32, from a
measurement on that surface rather than an inherited value. **Unpinned**, agent-run; for a default
decision, not a gate and not a ratio reference.

Build: HEAD `dec2b3e` (tree clean apart from `.gitignore` and the untracked `.github/`), shaded jar sha256
`2267dbd7f9855ae5`. Host: the `docs/perf-compare/README.md` baseline host (Xeon E5-1650 v2, GTX 1080).
Model: TinyLlama 1.1B Q4_K_M.

## Latency of one 512-token prefill (`readings.txt`)

`driver/surface.sh` launches the engine on the named surface with the REST API, and `driver/ttft.py`
calibrates a raw prompt (`x x x ...`) to 512 prompt tokens against the engine's own `usage.prompt_tokens`,
issues two discarded warm-up requests and then three measured ones, each `max_tokens=1`, temperature 0,
no session id (no prefix reuse). The figure is request wall time, median (min to max) of three. Every
request prefilled 512 of 512 tokens. The cluster runs set the chunk through `JUNO_PREFILL_BATCH`, because
the cluster launcher did not accept the flag when they were taken.

| Surface | Chunk 32 | Wider chunk | Wider over 32 |
|---|---|---|---|
| GPU, `local --nodes 1`, static | 1,417 ms (1,413 to 1,426) | adaptive (whole prompt): 549 ms (536 to 556) | 2.58x |
| GPU, `local --nodes 3` (in-process shards, as the embedding facade builds), static | 1,433 ms (1,418 to 1,458) | adaptive: 575 ms (567 to 588) | 2.49x |
| GPU, `local --nodes 1`, continuous, request alone | 2,235 ms (2,210 to 2,252) | 128: 875 ms; 512: 600 ms | 2.56x; 3.73x |
| CPU, `local --nodes 1`, static | 90,433 ms (90,252 to 90,615) | 128: 90,608 ms; 512: 91,169 ms | 1.00x; 0.99x |
| GPU, `local --lora-play` (LoRA handler), static | 19,135 ms (19,097 to 19,154) | adaptive: 18,858 ms | 1.01x |
| GPU, `cluster --pType pipeline` (three forked nodes) | 20,291 ms (19,907 to 20,587) | 512: 21,292 ms | 0.95x |
| GPU, `cluster --pType tensor` (three forked nodes) | 28,792 ms (28,789 to 28,830) | 512: 29,729 ms | 0.97x |

Why the cluster rows do not move: neither gRPC pipeline client (`ProcessPipelineClient`,
`TensorParallelPipelineClient`) overrides `InferencePipeline.prefillBatch`, so a cluster prefill runs the
interface default, one `forward` call per prompt token, whatever the chunk size.

## Continuous schedule under mixed load (`mixed/`)

`compare-mixed-prefill.sh --gpu --n-prompt 512 --prefill-batch {32,128,512} --no-publish`, two runs each
(cold engine per run): one 512-word prompt plus three concurrent short streaming requests, `--parallel 8`.
Each run's INDEX is copied under `mixed/`.

| Chunk | Short requests' mean TTFT | Short mean TPOT | Long prompt's TTFT |
|---|---|---|---|
| 32 | 898 / 904 ms | 196 / 197 ms | 3,539 / 3,544 ms |
| 128 | 1,004 / 1,033 ms | 202 / 202 ms | 1,978 / 2,000 ms |
| 512 | 1,336 / 1,383 ms | 124 / 126 ms | 1,478 / 1,533 ms |

At 512 the whole prompt is one chunk, so the run matches the admit-time baseline (mixed over admit 1.00x
and 1.04x): interleaving no longer happens. 32 gives the short requests the lowest TTFT; 128 shortens the
long prompt's TTFT by 1.78x for about 13% more short-request TTFT.

## Decisions taken from these readings

- Embedding facade on a GPU with the static schedule: sized from free device memory, like the local REPL
  (it was fixed at 32).
- CPU: fixed 32 kept; window width does not change CPU prefill time.
- Continuous schedule: fixed 32 kept for now; it is the fairness setting. Whether to trade it for the long
  prompt's TTFT is a policy question left with the owner.
- Cluster REPL and standalone coordinator: fixed 32 kept; the chunk size is inert there.
- LoRA training REPL: fixed 32 kept; the LoRA handler's prefill does not speed up with window width.
