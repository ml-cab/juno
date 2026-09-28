# llama.cpp vs Juno - 20260927T091155Z (gpu)

> **GPU reference sweep from 2026-09-27** (supersedes [`20260925T172231Z`](../20260925T172231Z/INDEX.md) as reference). Taken after the scalar CPU RoPE began reading a per-position table, a measurement boundary. The `tinyllama-1.1b-chat-v1.0.Q4_K_M`, `qwen2.5-3b-instruct-q4_k_m` and `Phi-3.5-mini-instruct-Q4_K_M` rows spread over 15% of their median and were re-run in [`20260927T093054Z`](../20260927T093054Z/INDEX.md); read those three models there. The `mistral-7b` rows here are the reference.

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3695.616560 | 188.462281 | 170.49586459150237 | 70.0463052373849 | 69.52 / 70.54 | 0.04613461971052061 | 0.3716728082973001 | 128/128 | 64/64 | 6 | 47279204 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3695.616560 | 188.462281 | 172.53036822930045 | 70.91712354460236 | 69.92 / 71.03 | 0.04668513776475243 | 0.3762934586608466 | 128/128 | 64/64 | 636 | 47168012 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1451.926157 | 69.818137 | 80.94614902942503 | 30.833622401755186 | 26.08 / 31.48 | 0.05575087179135724 | 0.44162768768457955 | 128/128 | 64/64 | 7 | 141988117 | NOISY | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1451.926157 | 69.818137 | 86.77023606332159 | 31.046293640225688 | 30.79 / 31.11 | 0.05976215501386658 | 0.444673762065947 | 128/128 | 64/64 | 7 | 141987847 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1180.000473 | 59.009166 | 45.18041884656952 | 24.129295821243872 | 19.86 / 24.57 | 0.03828847520010233 | 0.40890758939456745 | 128/128 | 64/64 | 8 | 212177898 | NOISY | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1180.000473 | 59.009166 | 43.16388904348549 | 24.342166065024294 | 24.24 / 24.62 | 0.0365795523231841 | 0.41251499919562146 | 128/128 | 64/64 | 9 | 212133043 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 652.574203 | 36.226584 | 59.07214005118325 | 23.399259893767297 | 23.2 / 23.58 | 0.0905217211768686 | 0.645914058409904 | 128/128 | 64/64 | 5 | 224088629 | yes | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 652.574203 | 36.226584 | 58.25233077473776 | 23.112974344277063 | 23.03 / 23.21 | 0.08926545135701258 | 0.6380114212335632 | 128/128 | 64/64 | 11 | 223848275 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Prompt tokens column is Juno actual / requested. A prefill ratio is published
  only when the two are within 10%; otherwise it reads `withheld`, because the two
  engines then prefilled measurably different amounts of work.
- Prefill and generation are measured in two separate Juno runs per model, mirroring the
  reference tool, which benchmarks prompt processing and generation separately. The prefill
  run is given the requested prompt length and asked for one token; the generation run is
  given the shortest prompt the chat template allows and asked for the full count, so it
  decodes at a shallow context like the reference does. Measuring both in one request would
  time Juno generation at the prompt length while the reference times it near zero, and
  decode slows as context grows.
  Residual: Juno cannot reach an empty context, because the chat template wraps every
  request; the generation run prefilled 19 tokens against the reference 0.
- Generated-token column is Juno actual / requested. The reference tool generates the
  requested count whatever the model would rather do, while Juno stops at a stop token.
  Where Juno generated nothing there is no reading and the ratio reads `-`; where it
  generated fewer tokens, the ratio is published and the shortfall noted, since it is then
  an average over a shorter and slightly cheaper span of context.
- Juno readings are the median of 3 measured cycle(s), each preceded by
  2 discarded request(s) so the measured request runs on compiled code. The
  min/max column is that median's own spread; a difference smaller than the spread
  is not a result. Each cycle records only its measured request: the recording is
  started after the warmup requests return and stopped before the engine exits.
- Clock state this run: CPU governor schedutil, turbo enabled,
  GPU clocks {"graphics_mhz": 1126, "sm_mhz": 1126, "memory_mhz": 5005, "active_throttle_reasons": "Idle"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
- Rows marked NOISY are not scorable and should be re-run:
  - qwen2.5-3b-instruct-q4_k_m: the 3 generation readings span 17% of their median, over the 15% this host can resolve: re-run rather than score this row
  - Phi-3.5-mini-instruct-Q4_K_M: the 3 generation readings span 19% of their median, over the 15% this host can resolve: re-run rather than score this row
- The Scorable column asks whether a row's own repetitions agree: a generation reading
  whose cycles span more than 15% of their median is not stable at the resolution a
  gate would read it at, and should be re-run rather than scored. Collection pauses and
  lock/park totals are recorded in every result JSON but are not gated on. Neither
  measures lost time reliably here: the park figure sums every thread, so an idle worker
  pool exceeds wall time on a healthy run, and the pause counter has reported ~635 ms on
  rows that produced their tokens in the same span as pause-free repetitions of
  themselves. A pause that does cost time appears in the dispersion anyway.
- GC max ms and Alloc B/tok come from the recording taken alongside each run. A
  result whose GC max is a large fraction of its measurement window should be
  re-run rather than scored: one long pause looks exactly like a regression.
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
