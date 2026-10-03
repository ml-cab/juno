# Long-prompt prefill smoke: output and time to first token (2026-10-03)

Purpose: the closing correctness check for the prefill-throughput work. Long prompts over
`/v1/chat/completions` on both schedules must answer, report the prompt length they were given, stream
the same text they return unstreamed, and decode greedily to exactly the text the build before that work
produced. Time to first token (TTFT) is recorded beside it. **Unpinned**, agent-run: the TTFT figures are
single readings for orientation, not a gate and not a ratio reference.

Command: `scripts/performance-tests/smoke-long-prompt-prefill.sh --baseline-jar dist/tier01b-close/baseline-shaded.jar`
(defaults: TinyLlama 1.1B and Mistral 7B, both Q4_K_M; `static` and `continuous`; 128, 512 and 2048 prompt
tokens; 16 generated tokens; GPU; `--prefill-batch` left at each surface's default, so the whole prompt in
one window on `static` and 32-token chunks on `continuous`). `console.log` is the run's output,
`results.json` one row per request.

Builds: candidate HEAD `1ac490a` (working tree carried only test and script changes, none in the jar),
shaded jar sha256 `66f02ee7c2908f78`. Baseline: commit `ffd0ca7`, the build the step-2 reference sweeps
(`20260930T135225Z`, `20260930T141026Z`) measured, rebuilt from that commit for this run (jar
`bd19a9306ea7901f`; jar builds are not byte-reproducible, so it does not hash like the published
`dc94bd797c8a3219`). Host: the `docs/perf-compare/README.md` baseline host (Xeon E5-1650 v2, GTX 1080).

## Result: 48 of 48 checks, and greedy output unchanged

- Every request: HTTP 200, a non-empty answer, `usage.prompt_tokens` within 10% of the requested length
  (every one landed exactly), streamed text equal to the unstreamed text.
- **Greedy text identical to the baseline build on all 12 cells** (2 models x 2 schedules x 3 lengths).
- TinyLlama's context is 2,048 tokens, so its long case is capped at 2,024 prompt tokens to leave room for
  the 16 generated ones; the cap is logged. Mistral 7B's long case is 2,044 of 2,048.
- The prompt is a list of numbered notes ending in a question about the first one; whether the answer names
  the right colour is recorded (`recall`), not asserted. It does not depend on the build.

## Time to first token, ms (single reading per cell)

| Model | Schedule | Tokens | Baseline `ffd0ca7` | HEAD | Baseline over HEAD |
|---|---|---|---|---|---|
| TinyLlama | static | 128 | 594 | 130 | 4.57x |
| TinyLlama | static | 512 | 2,072 | 655 | 3.16x |
| TinyLlama | static | 2,024 | 11,684 | 5,963 | 1.96x |
| TinyLlama | continuous | 128 | 1,011 | 536 | 1.89x |
| TinyLlama | continuous | 512 | 3,761 | 2,363 | 1.59x |
| TinyLlama | continuous | 2,024 | 18,640 | 13,359 | 1.40x |
| Mistral 7B | static | 128 | 1,985 | 742 | 2.68x |
| Mistral 7B | static | 512 | 7,584 | 2,839 | 2.67x |
| Mistral 7B | static | 2,044 | 49,125 | 31,192 | 1.57x |
| Mistral 7B | continuous | 128 | 3,429 | 2,242 | 1.53x |
| Mistral 7B | continuous | 512 | 14,183 | 9,555 | 1.48x |
| Mistral 7B | continuous | 2,044 | 68,192 | 51,826 | 1.32x |

How to read it:
- The gain shrinks with prompt length on every row. From 512 to about 2,048 tokens HEAD's static TTFT
  grows 9.1x on TinyLlama and 11.0x on Mistral 7B for 4x the tokens, against 5.6x and 6.5x for the
  baseline: the per-token costs the prefill work removed are a smaller share of a long window, and what
  grows faster than the prompt is attention. This is the first prefill reading at about 2,048 tokens on
  record; the attention work that targets it is planned separately.
- `continuous` is slower than `static` at every length because its 32-token chunks each pay a full
  forward pass; the load-dependent chunk that would let a prompt running alone take one window is planned
  separately.
- TTFT here is end-to-end request time to the first streamed token, one reading per cell, each taken right
  after the unstreamed request for the same prompt. For throughput, read the pinned sweeps.

## CPU leg (`cpu/`)

`smoke-long-prompt-prefill.sh --cpu --models tinyllama --lengths "128 512" --baseline-jar dist/tier01b-close/baseline-shaded.jar`,
run 2026-10-02 21:22 -0500 on the same two builds. 16 of 16 checks; greedy text identical to the baseline
build on all 4 cells. The prefill-window device region is CUDA-only, so CPU TTFT should not move, and it
did not:

| Schedule | Tokens | Baseline `ffd0ca7` ms | HEAD ms |
|---|---|---|---|
| static | 128 | 21,614 | 21,814 |
| static | 512 | 95,887 | 96,959 |
| continuous | 128 | 20,835 | 20,849 |
| continuous | 512 | 92,580 | 93,399 |
