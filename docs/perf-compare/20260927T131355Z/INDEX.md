# llama.cpp vs Juno - 20260927T131355Z (gpu)

> **`--gpu-residency on`** (device-resident decode region), taken to read the region against the 2026-09-27 reference (`20260927T091155Z` + `20260927T093054Z`). Not a reference sweep. The region runs on tinyllama and mistral-7b and declines qwen2.5-3b (split-half RoPE, Q/K/V biases) and Phi-3.5-mini (another handler). The tinyllama tuned and mistral-7b default rows spread over 15% and were re-run in [`20260927T133246Z`](../20260927T133246Z/INDEX.md).

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Scorable | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|----------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3624.853379 | 192.524490 | 191.98443649367897 | 73.04151675475501 | 71.96 / 74.03 | 0.05296336607872463 | 0.3793881846135783 | 128/128 | 64/64 | 6 | 46200567 | yes | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3624.853379 | 192.524490 | 168.7622873118558 | 72.70676396682678 | 41.85 / 73.03 | 0.046556996840079856 | 0.3776494302975522 | 128/128 | 64/64 | 5 | 46177305 | NOISY | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1429.739731 | 70.321577 | 84.49146677969196 | 30.77100030886103 | 30.59 / 35.2 | 0.05909569759287327 | 0.437575515532893 | 128/128 | 64/64 | 8 | 141987459 | yes | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1429.739731 | 70.321577 | 81.73750481685342 | 31.125073014801167 | 30.01 / 31.47 | 0.05716949948623441 | 0.4426105662391667 | 128/128 | 64/64 | 8 | 141987250 | yes | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1168.039868 | 59.363233 | 44.01546397083815 | 24.429524632250942 | 24.24 / 24.51 | 0.037683186316409316 | 0.4115261820772285 | 128/128 | 64/64 | 9 | 211618938 | yes | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1168.039868 | 59.363233 | 44.02059566985585 | 24.442493019454094 | 24.3 / 24.55 | 0.03768757974437209 | 0.4117446403138807 | 128/128 | 64/64 | 10 | 212075344 | yes | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 650.008961 | 36.350603 | 58.056682676031265 | 24.03327263899312 | 19.5 / 24.44 | 0.08931674201339397 | 0.6611519660070873 | 128/128 | 64/64 | 11 | 220728547 | NOISY | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 650.008961 | 36.350603 | 59.46558366372338 | 23.698779062305256 | 22.21 / 23.87 | 0.09148425211283107 | 0.6519500945364031 | 128/128 | 64/64 | 9 | 221608232 | yes | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

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
  GPU clocks {"graphics_mhz": 1809, "sm_mhz": 1809, "memory_mhz": 5005, "active_throttle_reasons": "none"}. A throttled run and a regression look the same
  without this.
- Thread counts are not matched: the reference tool ran with -t 12, while Juno
  dispatches its kernels on the common pool at an effective parallelism of
  11. Juno has no thread-count control reaching the hot path yet, so this
  mismatch is recorded rather than removed.
- Rows marked NOISY are not scorable and should be re-run:
  - tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned: the 3 generation readings span 42% of their median, over the 15% this host can resolve: re-run rather than score this row
  - mistral-7b-instruct-v0.1-q4_k_m: the 3 generation readings span 20% of their median, over the 15% this host can resolve: re-run rather than score this row
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
