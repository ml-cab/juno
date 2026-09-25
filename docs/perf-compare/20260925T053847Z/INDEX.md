# llama.cpp vs Juno - 20260925T053847Z (gpu)

| Model | llama.cpp pp t/s | llama.cpp tg t/s | Juno pp t/s | Juno tg t/s | Juno tg min/max | Juno/llama pp | Juno/llama tg | Juno prompt tok | Juno gen tok | GC max ms | Alloc B/tok | Results |
|-------|------------------|------------------|-------------|-------------|-----------------|---------------|---------------|-----------------|--------------|-----------|-------------|---------|
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 3438.571624 | 192.745331 | 137.69799961124414 | 62.710285162175964 | 60.86 / 63.21 | 0.04004511601566225 | 0.3253530699645117 | 128/128 | 64/64 | 6 | 62210501 | tinyllama-1.1b-chat-v1.0.Q4_K_M-*.json |
| tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned | 3438.571624 | 192.745331 | 148.89361712977887 | 62.454987092832795 | 62.32 / 63 | 0.04330100792159008 | 0.3240285342778695 | 128/128 | 64/64 | 4 | 62514317 | tinyllama-1.1b-chat-v1.0.Q4_K_M-tuned-*.json |
| qwen2.5-3b-instruct-q4_k_m | 1474.813591 | 71.631093 | 67.5259504588384 | 28.29501840196143 | 22.35 / 28.56 | 0.045786091795541634 | 0.3950102841787075 | 128/128 | 64/64 | 6 | 176159088 | qwen2.5-3b-instruct-q4_k_m-*.json |
| qwen2.5-3b-instruct-q4_k_m-tuned | 1474.813591 | 71.631093 | 69.53118871418155 | 28.222403281776547 | 28.14 / 28.38 | 0.04714574719035224 | 0.39399654674788426 | 128/128 | 64/64 | 6 | 175872430 | qwen2.5-3b-instruct-q4_k_m-tuned-*.json |
| Phi-3.5-mini-instruct-Q4_K_M | 1189.941783 | 60.334589 | 43.95904982212965 | 12.37988993069265 | 11.08 / 12.4 | 0.03694218528179009 | 0.20518727542326756 | 128/128 | 64/64 | 8 | 258580274 | Phi-3.5-mini-instruct-Q4_K_M-*.json |
| Phi-3.5-mini-instruct-Q4_K_M-tuned | 1189.941783 | 60.334589 | 44.18363399128965 | 12.304294117207858 | 12.25 / 12.49 | 0.03713092070764747 | 0.2039343322154371 | 128/128 | 64/64 | 8 | 259055452 | Phi-3.5-mini-instruct-Q4_K_M-tuned-*.json |
| mistral-7b-instruct-v0.1-q4_k_m | 660.169889 | 36.972447 | 46.783615471429364 | 20.620357860060118 | 20.47 / 20.72 | 0.07086602441425402 | 0.5577222914150128 | 128/128 | 64/64 | 0 | 275244861 | mistral-7b-instruct-v0.1-q4_k_m-*.json |
| mistral-7b-instruct-v0.1-q4_k_m-tuned | 660.169889 | 36.972447 | 45.88045444634576 | 20.29217861422293 | 20.19 / 20.38 | 0.06949795077119271 | 0.5488459720889701 | 128/128 | 64/64 | 15 | 275176358 | mistral-7b-instruct-v0.1-q4_k_m-tuned-*.json |

Host meta: see any *-llama-cpp.json .host field.

Notes:
- llama.cpp metrics from llama-bench (avg_ts).
- Prompt tokens column is Juno actual / requested. A prefill ratio is published
  only when the two are within 10%; otherwise it reads `withheld`, because the two
  engines then prefilled measurably different amounts of work.
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
- GC max ms and Alloc B/tok come from the recording taken alongside each run. A
  result whose GC max is a large fraction of its measurement window should be
  re-run rather than scored: one long pause looks exactly like a regression.
- Juno pp/tg from JFR (--jfr 30m): TokenProduced.tps + ForwardPass decode total_ms for tg; pp from ForwardPass prefill total_ms when present, else (API latency − decode total_ms).
- Backend: gpu (llama -ngl 99 / juno --gpu), temperature 0, max_tokens=64.
- Juno ran with jdk.incubator.vector. On CPUs without HW FMA this can be pathologically slow; re-run with --vector 0.
