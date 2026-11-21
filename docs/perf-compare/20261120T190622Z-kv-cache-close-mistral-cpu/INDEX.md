# KV cache tier close: the standing gate's Mistral 7B CPU row, re-run alone, owner run, pinned

**Purpose.** The full gate ([`20261120T134937Z-kv-cache-close-gate`](../20261120T134937Z-kv-cache-close-gate/INDEX.md))
met every bar on every row except Mistral 7B's CPU generation rate. That row was not scored, because the harness's span
check withheld `token_gen_tps` on all three baseline runs. By owner decision 21 (b), this run repeats the CPU lane on
Mistral 7B alone, with the same jars, method and bars.

**Builds.** The same two jars as the full gate, each hash checked against `READY` before the run:
- A, baseline: the pre-tier jar (`git archive e16748ad`, `mvn package -DskipTests`), sha256 `21109a3a8fd317c8`.
- B, candidate: the tier's final build, sha256 `2ceed60ef9139b7a`. `run-gate.sh` checks that it equals the tree's own
  shaded jar (`cmp`). `tree.txt` holds the tree's diff hash at the start. It differs from the full gate's, because the tree
  changed between the two runs: the residency smoke script, docs and evidence, none of them in the jar.

**Pinned.** CPU yes: `compare-llama-cpp.sh --pin-clocks` sets the governor to `performance` and turns turbo off for every
run. All six runs' `host.json` read `clock_pinned: true`. No GPU is used on this lane.

**Method.** `run-gate.sh`, started by `wait-and-run.sh`, on 2026-11-20 from 13:06:22 to about 13:35 host time
(UTC-6). It runs A B A B A B (`compare-llama-cpp.sh --cpu --pin-clocks --juno-warmup 2 --juno-reps 1 --reps 1
--no-tuned-lane`, `--n-prompt 128`, 64 generated tokens). The bars are those of the full gate: prefill and generation t/s
B/A >= 0.95; allocation per token in the recording <= 1.10x; GC pause in the token span <= 1.25x (<= 5 ms under a 5 ms
baseline). The script's header, written before the run, sets how the row is scored. The generation rate is the median of
the harness's `token_gen_tps` over the runs whose span check passed; a withheld run is left out, never counted as 0. Of
two values, the scorer takes the lower one. The full terminal output is `gate.log`, which ends `GATES MET`.

## 1. The row

| Mistral 7B, CPU, `--n-prompt 128` | A | B | B/A | Bar | Result |
|---|---:|---:|---:|---|---|
| Prefill t/s, median of 3 | 5.1257 | 5.0495 | **0.985** | >= 0.95 | met |
| Generation t/s, harness `token_gen_tps` (2 of 3 runs each side) | 2.0157 | 2.0234 | **1.004** | >= 0.95 | met |
| Allocation per token in the recording, median of 3 | 343,305 | 347,770 | **1.013** | <= 1.10 | met |
| GC pause in the token span, ms, median of 3 | 0 | 0 | | <= 5 ms | met |

Two other generation readings from the same six runs agree. Neither is withheld on any run:

| Reading, Mistral 7B CPU generation | A, median of 3 | B, median of 3 | B/A |
|---|---:|---:|---:|
| Request-timed `api_token_gen_tps` | 1.8627 | 1.8669 | **1.002** |
| Decode rate from the forward-pass totals (`jfr.token_gen_tps_decode_derived`) | 1.9909 | 1.9913 | **1.000** |

## 2. Per run

| Run | pp t/s | `token_gen_tps` | Request-timed | Decode-derived | Token span residual, ms | Allocation per token |
|---|---:|---:|---:|---:|---:|---:|
| A 1 | 5.1257 | withheld | 1.8627 | 1.9909 | -29.0 | 343,305 |
| B 1 | 5.0495 | 2.0234 | 1.8669 | 1.9903 | 63.4 | 347,770 |
| A 2 | 4.8916 | 2.0157 | 1.8561 | 1.9859 | 10.7 | 343,173 |
| B 2 | 5.1438 | withheld | 1.8614 | 1.9913 | -31.6 | 388,007 |
| A 3 | 5.1900 | 2.0295 | 1.8677 | 2.0009 | -8.6 | 343,402 |
| B 3 | 4.9694 | 2.0350 | 1.8787 | 2.0057 | 0.4 | 342,757 |

The span check allows a token span residual from -25 to 492 ms. A 1 and B 2 fall below the floor, in requests of about
34.4 seconds (0.09%), as in the full gate. That is the harness finding recorded there: the floor is fixed while the
ceiling scales with the generated tokens. The other runs are inside the bounds. B 2's allocation is the highest of the
six, and it is not the median.

## 3. Reported, not gated

- **Greedy output**: the candidate equals the baseline in 3 of 3 runs, and its own run 1 in 3 of 3 (`gate.log`).
- **Against the reference tool**, candidate, same runs: pp 0.471x, tg 0.496x (`gate.log`).
- **Hot methods**: the same integer kernels and worker loop lead on both jars, in the same order (`gate.log`).

## Result

The row meets every bar. Together with the full gate's other rows, every row of the standing gate is met.

Files: `gate.log`, `run-gate.sh`, `wait-and-run.sh`, `tree.txt`, and per run in `ab/`: `<run>-host.json`,
`<run>-mistral-7b-instruct-v0.1-q4_k_m-juno.json` and `...-generate-rep1-juno-response.json`. Local paths are
written `<repo>` and `<repos>`.
