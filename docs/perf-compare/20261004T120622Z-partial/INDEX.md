# GPU sweep at a 2048-token prompt, interrupted (two of four models)

> **Superseded on 2026-10-04** by the complete 2048 re-run, [`20261004T220015Z`](../20261004T220015Z/INDEX.md) and [`20261004T222758Z`](../20261004T222758Z/INDEX.md). Kept as the record of the interrupted attempt.

**Purpose.** The first parity-corrected GPU reading at `n_prompt=2048`, taken as part B of the packed
K-quant prefill closing gate (`compare-llama-cpp.sh --gpu --pin-clocks --n-prompt 2048 --juno-reps 3
--reps 3`, four sweep models). **It did not finish.** TinyLlama and Qwen2.5-3B completed. Phi-3.5-mini's
prefill requests ran out of Java heap (6 GiB fixed for that model) and returned no response, so each one
waited out the harness's two-hour request timeout. The owner stopped the run during Phi-3.5-mini's second
repetition. Mistral 7B was not reached. The harness writes this directory's INDEX only at the end, so
this one is written by hand from the result files the run left. Those files are copied here, with local
paths replaced by `<repo>/` and `<reference-tool>/`.

**Pinned: yes** (governor performance, turbo off, GPU graphics clock locked at 1911 MHz; `host.json`).
Build: the packed K-quant closing build, jar `5ab4c78c505a4cef`.

| Model | Reference pp t/s | Juno pp t/s | Juno prompt tokens | Juno/reference pp | Juno/reference tg | Scorable |
|---|---|---|---|---|---|---|
| tinyllama-1.1b | 2647.9 | 320.7 | 2048 | **0.121x** | 0.372x | yes (spread 6.0%) |
| qwen2.5-3b | 1088.7 | 184.8 | 2048 | **0.170x** | 0.453x | yes (spread 1.6%) |
| Phi-3.5-mini | - | - | - | not measured: Java heap exhausted in the 2048-row prefill window | - | - |
| mistral-7b | - | - | - | not reached | - | - |

**How to read it.** Against the same build's 512 sweep (`20261004T114812Z`), the pp ratio at 2048 is 0.38x
(TinyLlama, 0.121 over 0.317) and 0.40x (Qwen2.5-3B, 0.170 over 0.429) of the 512 ratio. That is the
fall-off the long-context milestone ("ratio at 2048 over ratio at 512 >= 0.90") measures, from attention
growing with context. Two of four models is not a complete reference for that milestone.
