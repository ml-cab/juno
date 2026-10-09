# Context policy: closing no-regression gate, GPU and CPU (owner run)

**Purpose.** Show that context shifting and sliding-window attention change nothing for a request that does not opt in
and a model that declares no binding window: Juno throughput, allocation, GC and greedy output of the candidate against
the last build before this work. Phi-3.5-mini declares a 262,144-token window, wider than its 4,096-token limit, so it
reaches the attention paths as a window that reads the same keys. The shift-step latency bound is a separate
measurement, not in this run.

**Pinned: yes** (`clock_pinned: true` on all twelve runs).

**Builds.**

| Side | Source | Shaded jar sha256 (first 16) |
|---|---|---|
| baseline (A) | HEAD `de98ff6`, the last build before context shifting and sliding windows | `620a5caf606538b6` |
| candidate (B) | HEAD `0f87603` (context shifting) plus the uncommitted sliding-window change | `b12fbfb34a342e70` |

**Command.** `dist/context-policy-close/run-gate.sh` parts A and C, started by `wait-and-run.sh` (2026-10-09 04:35 to
06:56 local): each invocation `compare-llama-cpp.sh --gpu|--cpu --pin-clocks --juno-warmup 2 --juno-reps 1 --reps 1
--no-tuned-lane --no-publish --juno-jar <jar>`, A B A B A B, medians per side. GPU: four sweep models,
`--n-prompt 512`. CPU: TinyLlama and Mistral 7B, `--n-prompt 128`.

## Results (gate: tg and pp B/A >= 0.95; allocation per token <= 1.10x; GC pause in the token span <= 1.25x, or <= 5 ms under a 5 ms baseline; greedy output equal)

| Lane | Model | pp A | pp B | pp B/A | tg A | tg B | tg B/A | alloc/token B/A | GC ms A | GC ms B | greedy equal | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU | tinyllama-1.1b | 3050.27 | 3043.07 | 0.998 | 147.69 | 148.39 | 1.005 | 1.000 | 0.0 | 0.0 | 3 of 3 | met |
| GPU | qwen2.5-3b | 1267.22 | 1266.44 | 0.999 | 30.49 | 30.47 | 0.999 | 1.000 | 0.0 | 0.0 | 3 of 3 | met |
| GPU | Phi-3.5-mini | 713.94 | 702.16 | 0.983 | 51.62 | 51.46 | 0.997 | 1.000 | 11.9 | 11.5 | 3 of 3 | met |
| GPU | mistral-7b | 557.47 | 550.08 | 0.987 | 37.98 | 37.14 | 0.978 | 0.998 | 0.0 | 0.0 | 3 of 3 | met |
| CPU | tinyllama-1.1b | 6.21 | 6.16 | 0.993 | 3.33 | 3.33 | 0.999 | 1.000 | 7.7 | 7.6 | 3 of 3 | met |
| CPU | mistral-7b | 0.87 | 0.87 | 1.001 | 0.5193 | 0.5207 | 1.003 (1 run per side); request-timed median of three **1.002** | 1.000 | 0.0 | 0.0 | 3 of 3 | met (scored on the request-timed rate by owner decision) |

Greedy output: the baseline also equals its own first run in 3 of 3 runs on every model, so the comparison is not masked
by run-to-run variation.

## How to read it

- **Every reading meets its threshold.** The lowest ratio is Mistral 7B GPU generation, 0.978. Allocation per token is
  unchanged to within 0.2%, and GC pause totals are unchanged.
- **Mistral 7B CPU generation: two of three repetitions per side were withheld by the harness's span check** (JFR
  timestamps disagreed with the engine clock by about 89 ms on a 137-second request; baseline runs 1 and 3, candidate
  runs 2 and 3; `token_gen_tps: null`). The engine-timed reading is therefore one run per side (0.5193 to 0.5207,
  1.003x), not the median of three the plan asks for at a 0.95x gate. The request-timed generation rate, which the
  span check does not withhold, is a full median of three: 0.4739 to 0.4748 t/s, **1.002x**. Prefill (median of three)
  reads 1.001x and the hot-method profile is unchanged. By owner decision (2026-10-09) the row is scored on the
  request-timed median of three, 1.002x: met.
- **`gate.log` says `FAIL: mistral generation below 0.95x` and `GATES MISSED`.** That line is a defect in the gate
  script's scorer, not a reading: it counted each withheld repetition as 0 t/s, so the medians were 0 on both sides and
  the ratio 0/0. The scorer was fixed (median over the readings that exist, withheld ones reported) and the saved runs
  rescored without a rerun; the table above is the rescored output.
- Hot methods (CPU, `jdk.ExecutionSample`, top 10 of run 1, reported not gated): the same kernels in the same order on
  both builds; `GqaMath.attend` stays under 0.2% of samples. Full lists in `gate.log`.
- Local paths are replaced by `<repo>/`.
