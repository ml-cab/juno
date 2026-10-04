## Status 

**Session 108** — `--gpu-layers auto` keeps a prefill window free, not a weight matrix

- **Smaller reserve.** With prefill multiplying packed weights directly, `auto` no longer keeps a whole
  FP16 weight matrix free after the upload. It keeps the narrowest prefill window (64 rows) with a 40%
  margin, the key/value mirror at its starting size, and 64 MiB for memory the driver reports free but
  will not hand out (44 to 54 MiB measured on a GTX 1080 with the card full). The weight matrix is still
  reserved when the packed prefill kernel cannot load. A 30B Llama on an 8 GiB card keeps 197 MiB free
  instead of 416 MiB and holds 23 layers on the GPU instead of 22; models that fit whole are unchanged.
- **Prefill window sized from what it really costs.** The default `--prefill-batch` on a GPU with the
  `static` schedule (local mode and `JunoPlayer`) is now the widest window whose device footprint (the
  window's buffers, its attention scores and its 8-bit copy, added up over every in-process node) fits
  half of the free memory. It used to divide by a fixed 64 KiB a token, the host-staged figure, while a
  window on the device costs 114 to 412 KB a row before attention scores, so on a nearly full card it
  could pick a window that did not fit. On the sweep models the window is now 2,601 to 5,213 rows, still
  one window for every prompt up to 2,048 tokens.
- Covered by `PrefillWindowFootprintTest`, `DeviceScratchBudgetTest`, `PrefillBatchOptionsTest`,
  `PrefillChunkDefaultsTest` and the GPU test `PrefillReserveDeviceTest`, which holds the footprint equal to
  what a window allocates and runs a wide window in exactly what the reserve guarantees, where the
  FP16-expansion route runs out of memory.

**Session 107** — Prefill multiplies packed K-quant weights directly

- **No FP16 copy of the weights during prefill.** On CUDA with packed weights (`--mmq on|auto`, the
  default), a prefill window wider than 8 rows used to expand each Q4_K, Q5_K or Q6_K matrix to FP16 on the
  device and multiply it with an FP16 GEMM. It now runs a tiled integer kernel over the still-packed weights:
  the window is rounded to 8-bit activations once per matmul, as decode already does for every token, and
  each block of 64 weight rows is unpacked in on-chip memory and multiplied with 4-way integer dot products.
  Both prefill paths take it, the window staged from the host and the device-resident prefill region; the
  JFR compute site is `gemm_kquant`. `--mmq off` keeps FP16 weights and the FP16 GEMM. If the kernel cannot
  load, Juno says so once and falls back to the FP16 expansion.
- **Accuracy.** The kernel's output equals the decode kernel's integer products to within float
  accumulation order, for all three formats at window widths 1 to 741. Against an FP32 reference its mean
  relative error is 0.37%, the decode kernel's own figure; the FP16 route's is 0.026%.

**Session 106** — Long-prompt prefill checked end to end on both schedules

- **Where GPU prefill stands now.** Pinned comparison sweeps against the reference engine put Juno's
  prompt processing at a 512-token prompt at 0.254x (TinyLlama), 0.281x (Qwen2.5-3B), 0.146x (Phi-3.5-mini)
  and 0.285x (Mistral 7B), against 0.011x to 0.098x on 2026-09-30. In a pinned same-hour comparison with the
  build of that date, Juno's own prefill rose 2.96x to 13.91x and generation read 0.994x to 1.361x, the top
  figure on Phi-3.5-mini, whose decode now runs the GPU attention kernel.

- **A smoke test for long prompts.** `scripts/performance-tests/smoke-long-prompt-prefill.sh` sends 128-, 512-
  and 2048-token prompts (calibrated against the server's own `usage.prompt_tokens`, and capped to fit a
  shorter context) to `/v1/chat/completions` on TinyLlama and Mistral 7B, on the `static` and `continuous`
  schedules. Each request must answer, report the prompt length it was given, and stream exactly the text it
  returns unstreamed; time to first token is recorded. With `--baseline-jar` every greedy answer must also
  match an earlier build's. Against the build from before the prefill work began, all 12 greedy answers were
  identical, and time to first token on the `static` schedule fell 2.7x to 4.6x at 128 and 512 tokens
  (TinyLlama, 512 tokens: 2,072 to 655 ms; Mistral 7B: 7,584 to 2,839 ms) and 1.6x to 2.0x at about 2,048.
- **`ModelLiveRunnerIT` prefills a 512-token prompt in-process.** The new check runs the prompt as one
  batched window on `static`, in 32-token chunks on `continuous`, and one token at a time, on the GPU when
  one is present, and requires the same first greedy token from all three. On TinyLlama, Qwen2.5-3B,
  Phi-3.5-mini and Mistral 7B all 16 generated tokens agreed.
- **`./juno test` runs its checks again.** Since the coordinator executable replaced the old
  integration runner, the command had started the coordinator instead and always exited 1 without
  `JUNO_NODE_ADDRESSES`. It now runs `ModelLiveRunner` from `juno-master.jar`: nine checks (pipeline-parallel
  1-6, tensor-parallel 7-8, the in-process long-prompt prefill 9), one `PASS`/`FAIL` line each, exit 0 only
  when all pass. The checks live in one class, `ModelLiveChecks`, which `ModelLiveRunnerIT` runs as well, so
  the command and the integration test cannot drift apart. `--pType pipeline|tensor` limits the cluster
  checks; an unsupported architecture exits 1 with the loader's error.
- **Forked test nodes size their heap from the model.** `ClusterHarness` used a fixed 4 GB node heap when
  no `-Djuno.node.heap` was given, which a tensor-parallel Mistral 7B node outgrew. It now derives the heap
  the way the launchers do (1.5 times the model file plus 2 GB, 4 to 48 GB), so `ModelLiveRunnerIT` runs a
  7B model without extra flags.
- **Smoke scripts are named for what they check.** `smoke-tier00-consistency.sh` is now
  `scripts/performance-tests/smoke-consistency.sh`; its checks are unchanged.
- **Vision and LoRA unchanged.** Against the same earlier build, moondream2 vision latency read 0.995x and
  decode 1.013x with byte-identical captions, and LoRA train-qa read 1.00x train time and 0.978x playback
  with the same final loss.

**Session 105** — The prefill window default reviewed on every surface

- **The `JunoPlayer` facade sizes its prefill window like local mode.** On a GPU with the `static`
  schedule, an embedder that does not set `.prefillBatch(N)` now gets a window covering the whole
  prompt when free VRAM allows, instead of a fixed 32 tokens. A 512-token TinyLlama prefill through
  the same in-process three-shard pipeline took 575 ms against 1,433 ms at 32.
- **One resolver for every entry point.** `PrefillChunkDefaults` decides the default chunk size for
  local mode, the facade, the cluster REPL, the standalone coordinator and `juno lora`. Every surface
  that keeps 32 does so on a measurement: on CPU a wider window does not change prefill time; under
  `--schedule continuous` 32 gives concurrent requests the lowest time to first token; cluster nodes
  prefill one token per call, so the size does not change the work; and the LoRA handler's prefill
  does not speed up with window width.
- **`juno cluster --prefill-batch N` is accepted.** The cluster launchers (`run.sh` and `run.bat`)
  rejected the flag as unknown although the engine reads it; they now pass it through, as
  `JUNO_PREFILL_BATCH` already did. Launcher help no longer states a fixed default of 32 for local mode.

**Session 104** — A prefill window's residual stays on the GPU from layer to layer

- **One upload and one download per window.** On CUDA, the prefill-window device region now keeps the
  window's residual stream on the device across layers instead of uploading it at every layer and
  downloading it after each one. Per layer only the K and V rows the host KV cache keeps cross the bus.
  Bytes moved per 512-token window fell from 207 to 31 MB on TinyLlama, 339 to 45 MB on Qwen2.5-3B and
  670 to 150 MB on Mistral 7B (1,809 to 1,419 MB on Phi-3.5-mini, whose RoPE and attention stay on the
  host). In a pinned same-hour comparison at a 512-token prompt, TinyLlama prefill rose from 959.5 to
  1029.4 t/s (1.073x) and Mistral 7B from 194.6 to 204.2 t/s (1.049x); generation read 0.997x and 0.993x. Logits remain bit-identical to the host window path.
- **Out of device memory inside a layer is still recoverable.** A layer whose input exists only on the
  device copies it aside on the device before updating the residual, so the layer can be redone on the
  host path from the same input, as before.
- **Batched matmul into the caller's buffer.** `MatVec.sgemmInto(A, X, Y)` writes a batched product into
  rows the caller owns, bit-identical to `sgemm`, which now allocates and delegates to it on the CPU and
  CUDA backends. Backends that do not implement it get a correct default. The transformer handlers'
  window projections use it, so a matmul no longer allocates a result batch and copies it out.

**Session 103** — A prefill window's layer runs on the GPU, not between GPU matmuls

- **The prefill window stays on the device across a layer.** On CUDA, a prefill window of more than
  eight positions now runs each transformer layer as one device region: the window is uploaded once
  per layer, and the RMS norms, the FP16 cast of every matmul input, the matmuls, the Q/K/V bias adds,
  SwiGLU and both residual adds run on the GPU with the activations kept there. Before, every matmul
  uploaded its input and downloaded its result, and the norms, SwiGLU (single-threaded) and residual
  adds ran on the CPU in between: host SwiGLU alone was 28% to 45% of a 512-token prefill. On the LLaMA
  family and Qwen2 the region also rotates Q and K (adjacent and split-half pairing) and runs attention,
  casting the window's K and V rows straight into the attention cache on the device; Phi-3 and Qwen3 keep
  RoPE and attention on the host between the region's two halves. In a pinned same-hour comparison
  at a 512-token prompt, TinyLlama prefill rose from 240 to 936 t/s (3.90x) and Mistral 7B from 68.4 to
  191.8 t/s (2.81x); generation read 1.02x and 1.00x.
- **Same logits, bit for bit.** Every operation in the region reproduces the host window path's
  arithmetic: the matmuls see the same FP16 bits, the RMS norm sums in the host loop's order, and SwiGLU
  and the adds round step by step as the host does. On TinyLlama, Qwen2.5-3B, Phi-3.5-mini and
  Qwen3-1.7B the logits after a prefill, a continued prefill and the decode steps that follow equal the
  host path's exactly, and 64 greedy tokens matched on all six test prompts on TinyLlama and Mistral 7B.
- **The attention cache is written once per layer per window.** Every prefill window path now copies
  its K and V rows to the device attention cache as one transfer per tensor instead of one per position
  (22,528 copies per 512-token TinyLlama window before), and the device region writes them without a
  copy at all. A cache growth that ran out of device memory for the second of its two buffers no longer
  leaks the first.
- **Off switch and accounting.** `-DJUNO_PREFILL_REGION=off` keeps every window on the previous path,
  for comparison; cluster launchers forward it. The region's work is recorded as `juno.WindowStep`
  `device_layer` spans with `juno.DeviceCompute` sites per operation, and `prefill-breakdown.sh` splits
  it out.

**Session 102** — A prefill window is now accounted for end to end in a recording

- **Every part of a prefill window has a span.** `juno.WindowStep` covers the window work no per-op
  event covered: the embedding lookup, each projection call (the matmul with the copy of its result
  into the layer workspace), bias adds, the KV write to the host cache and the device mirror, and the
  final norm with the LM head. The Phi-3 and Qwen3 window paths now emit the same `juno.RmsNorm`,
  `juno.Rope`, `juno.Attention`, `juno.ResidualAdd` and `juno.SwiGlu` events as the LLaMA-family path;
  before, they recorded only the forward pass as a whole. `juno.MatVec` records the call's batch width,
  so matmul time splits into prefill and decode.
- **A per-term prefill breakdown from one command.** `scripts/performance-tests/prefill-breakdown.sh`
  reads a `compare-llama-cpp.sh --device-spans` run and splits prefill time into terms that do not
  overlap (the GEMM kernel, dequantization, copies and host packing apart from the rest of each
  projection; the attention kernel; host SwiGLU, norms, RoPE and residual adds; the KV write), and
  reports what no span covers. On the four sweep models at 512 tokens that remainder is under 2% of
  the window, where it was 7% to 45% before.

**Session 101** — Phi-3.5 ends its turns again, GPU kernels can be timed on their own, and the CPU comparison runs both engines on the same thread count

- **Phi-3.5 rotates short sequences with the short RoPE factors.** A Phi-3 model whose file carries
  both LongRoPE factor sets chose between them by its trained context length (131,072 on
  Phi-3.5-mini), so every request was rotated with the long-context factors. The model then rarely
  ended its turn: the probability of `<|end|>` after a finished one-line answer was 0.50, and about
  half of sampled console replies ran on to the token limit. It now uses the short factors, which put
  that probability at 0.99, and a sequence that reaches the original training context (4,096 tokens on
  Phi-3.5-mini) fails with an error naming the limit instead of continuing on factors it did not
  start with. Juno has no per-session context setting, so a Phi-3.5 sequence never starts on the long
  factors. This applies on every Phi-3 path: CPU and GPU, batched prefill, and LoRA training.
- **GPU kernels have their own recording event.** `juno.DeviceCompute` totals the device kernels per
  site and phase: the tiled FP16 GEMM, the batched FP16 GEMV for two- to eight-row windows, the FP32
  BLAS GEMM, the attention kernel, and the packed decode GEMV (counted, untimed). The host FP16 packing
  of each activation window is now its own `juno.DeviceStaging` site under a `HOST` direction, kept out
  of the bytes that cross the bus. With the copy and dequantization totals, every GPU term of a prefill
  window is named rather than left inside the matmul span. Off by default like the other two;
  `compare-llama-cpp.sh --device-spans` turns it on and adds prefill kernel time, per site, to every
  result. With it off, a same-hour pinned comparison against the build without it reads 0.994x and
  1.016x prefill, and 1.011x and 0.996x generation, on TinyLlama and Mistral 7B.
- **The comparison harness matches the reference tool's thread count.** `compare-llama-cpp.sh` sets the
  common fork-join pool's parallelism to `--threads` minus one, since the calling thread joins the work,
  so Juno's CPU kernels run on the same thread count the reference tool is given. `host.json` records
  `juno_threads`, and the index states whether the counts match. At the default `--threads` (the
  processor count) this equals what the JVM already did; the old index note gave Juno one thread too
  few. A pinned CPU sweep taken with it is the new CPU reference: Juno reads 0.092x to 0.121x of the
  reference tool's generation and 0.049x to 0.103x of its prompt processing.

**Session 100** — The GPU attention kernel reaches the Phi-3 and Qwen3 handlers, and every launch that cannot use it says so

- **Phi-3 and Qwen3 run attention on the GPU.** The GPU-resident attention kernel behind
  `--gpu-attention` (default `auto`, which requests it whenever CUDA is present) was wired only into
  the LLaMA-family handler; the Phi-3 and Qwen3 handlers read the default and ran scalar CPU attention
  anyway. Both now run the kernel at all three attention call sites (prefill window, single-token
  decode, multi-stream decode) through a shared `GpuAttentionMirror`, with the same fallbacks as the
  LLaMA-family path: the host KV cache is always written first, and running out of device memory moves
  the request back to CPU attention without losing history. In a pinned same-build comparison,
  prefill rose 6.55x on Phi-3.5-mini (13.7 to 89.7 t/s) and 6.10x on Qwen3-1.7B (28.8 to 175.6 t/s) at
  a 512-token prompt, and 2.20x and 2.16x at 128; generation rose 1.33x and 1.10x. Phi-3.5-mini's
  prefill no longer collapses with prompt length: it reads 98 t/s at 128 tokens and 90 at 512, where it
  read 45 and 14. Logits stay
  within 0.015 relative L2 of `off` at every call site, and the first generated token matched on every
  test prompt; greedy output can part from `off` after 20 or more tokens, as it already could on the
  LLaMA family (from token 8 there). `off` remains the bit-identical CPU-parity baseline.
- **No launch drops the kernel silently.** Every handler reports whether it runs the kernel
  (`ForwardPassHandler.gpuAttentionActive()`). The Phi-2 and Qwen3-MoE handlers compute every matmul
  and attention on the CPU on any backend, so `--gpu-attention` and `--gpu-layers` do not reach them;
  a GPU launch of either now prints a startup warning saying so (this includes moondream2, whose text
  backbone is Phi-2). A backend other than CUDA, and an explicit `--gpu-attention on` with the CPU
  backend, warn the same way. LoRA training and `--lora-play` keep attention on the CPU on every
  architecture; the notice saying so was raised only for the LLaMA family and now covers Phi-3, Qwen2
  and Qwen3 as well.
- **Corrected documentation of the default.** The performance notes gave `--gpu-attention`'s default
  as off; it has been `auto` since 2026-09-17, so every default-flag GPU run of a LLaMA-family model
  since then ran the kernel.

**Session 99** — GPU copies and weight dequantization can be read on their own, and the first benchmark recording no longer slows the request it measures

- **Two new recording events break the matmul span apart.** `juno.DeviceStaging` totals every
  host-device copy (bytes and measured time per direction, per phase and per copy site) and
  `juno.WeightDequant` every weight dequantization (per format, on the device per batched K-quant
  matmul or on the host at load). Asynchronous copies are timed on the device between two stream
  events, because a host clock only sees them being queued; the GPU backends gained the stream-event
  calls for both vendors. Both are off in `juno-perf.jfc`, since counting every copy costs about 6% of
  a TinyLlama prefill window; `compare-llama-cpp.sh --device-spans` turns them on and adds the staged
  bytes to every result. With them off, a same-hour pinned comparison against the build without them
  reads 0.999x to 1.000x prefill and 0.998x to 1.000x generation.
- **The comparison harness warms up under a recording.** Starting the first recording in a process
  deoptimized code compiled during the warmup, inside the measured request: the first two of 22
  TinyLlama layers ran two to three times slower. The last warmup now runs under a discarded
  recording with the measurement settings, which raised TinyLlama prefill readings by about 3% on an
  unchanged build.

**Session 98** — Benchmark readings no longer depend on which CPU core a span started on, and engines launched by the performance scripts no longer leave processes behind

- **Recorded durations are checked against the request that contains them, and read off a clock the
  kernel trusts.** On a host whose kernel has rejected the CPU timestamp counter, the JVM still stamps
  flight-recorder events with it. On the reference host CPU0's counter reads 633 ms ahead of the other
  cores, so a span that crossed CPU0 read 633 ms short or long: single prefill repetitions four to five
  times too fast, prefill spans longer than their request, generation spans misread both ways, and
  633 ms "collection pauses" that cost nothing. The comparison and vision harnesses now make the
  recorder read the operating-system clock whenever the kernel clocksource is not `tsc`
  (`PERF_JFR_OS_CLOCK=auto`, recorded in `host.json` and the index), and withhold any repetition whose
  spans do not account for its own latency or whose prefill did not start at position 0. Same-hour
  check on TinyLlama: 1 of 24 repetitions misread with the default clock, 0 of 24 with the
  operating-system clock; generation measured under the recorder reads 3.9% lower, prefill unchanged.
  Applied to the four comparison sweeps of 2026-09-27, the check withholds 23 of 156 repetitions; no
  reference median moves by more than 2.2%. `docs/howto.md` says how to get correct durations from
  `./juno --jfr` on such a host.
- **Two new prefill metrics.** `juno.PrefillBatch.tokens` and `juno.PrefillBatch.min_start_position`
  (-1 when there was no prefill window) report how much of the prompt the recorded prefill covered and
  where it started, written on every run.
- **Performance scripts reap the engine's stdin keepalive.** Ten scripts kept the console open with a
  shell loop that nothing ever reaped, so every engine launch left a sleeping shell behind, enough to
  make "is a sweep still running" checks return false positives. They now share one named-pipe helper in
  `perf-lib.sh`, checked by `selftest-engine-stdin.sh`; a full run of each leaves no shell, pipe or
  engine behind, and an engine whose script dies without stopping it now exits instead of running on.
  The comparison harness also stops discarding its own warnings after its first engine stop.
- **A refused model file says why in one line, and says the right why.** Files whose architecture
  Juno has no verified handler for (qwen35, gemma4, mistral3, minimax-m2 among those on hand) mostly
  also declare a pre-tokenizer split Juno has not implemented, and local mode had begun refusing them
  for the split, since the tokenizer loaded first, and as a stack trace. Every entry point (local,
  cluster and LoRA modes, the standalone coordinator, the `JunoPlayer` and `LoraTrainer` embedding
  API) now checks the architecture before reading anything else, and the console reports either
  refusal as one `ERROR:` line with exit status 1. Cluster mode now refuses before starting any node.

**Session 97** — GPU device memory no longer grows with every request, GPU prefill is faster, and two failures now say what they are

- **A GPU server's device memory now stays flat across requests.** The server runs every request on
  a new thread, and the GPU matrix-vector backend, the GPU attention path and the round-trip GPU norm
  kept their device scratch (staging buffers, a whole dequantized weight matrix for batched prefill,
  and a CUDA stream) per thread, so every request allocated a fresh set that was never released.
  Measured in local mode with three in-process nodes, GPU memory after each of 8 requests:
  TinyLlama 938 MiB after every request, with and without `--gpu-residency`; Mistral-7B 4604 MiB
  (4608 with the flag); LLaMA-30B, which fills the card, settles within a few requests and then holds
  (7768 MiB without the flag, 7784 with it, over 16 requests). Before, it grew by about 23 MiB per
  request on TinyLlama and 114 MiB per request on Mistral-7B, which on an 8 GiB card ran a Mistral-7B
  server out of device memory after a few dozen requests. The GPU residency smoke test now fails a
  server whose memory grows by more than 8 MiB per request over the second half of its requests, in
  either mode.
- **GPU prefill is 7% to 26% faster.** Moving the scratch exposed a cost the old code had paid
  intermittently: the loop converting a prefill window to FP16 before each batched matmul sat inside a
  large method that the JIT recompiles whenever a rarely taken branch runs, and the rest of that call
  then ran the loop in the interpreter (17 to 38 ms instead of about 1 ms per matmul). The loop is now
  a method of its own. Same-hour comparison against the previous build, prefill medians of three:
  TinyLlama 181 to 229 tokens per second, Qwen2.5-3B 83 to 90, Mistral-7B 61 to 65; generation is
  unchanged within noise, and LoRA training and playback pass their gates (training 0.91 of the
  baseline's time, playback 1.08x). GPU prefill figures from before this change are not comparable.
- **Faster performance checks, same checks.** The GPU residency smoke test generates the full output
  only on its first and last request per mode (which must match); the requests between, which exist
  for the per-request memory reading, generate 8 tokens (`--mem-n-gen`) and must reproduce the first
  request's opening. The comparison harness takes `--juno-jar PATH` to measure another build from the
  same checkout without publishing it, for same-hour comparisons of two builds.
- **Pinned clocks and named builds for performance gates.** The comparison harness takes
  `--pin-clocks` (performance governor, turbo off, GPU clock locked where the driver allows; restored
  on exit) and records in every run the Juno commit, the jar's hash, the JDK build, the JVM flags, the
  GPU driver and the reference tool's build. The Juno heap is now fixed per model (initial size equal
  to the maximum), so absolute figures from before this change are not comparable with those after it.
- **How**: the matrix-vector backend's scratch and stream now belong to the backend instance; every
  use was already serialized by the GPU context's lock and waits for its stream before releasing it,
  so one set serves every caller. The attention path and the round-trip norm run outside that lock, so
  their scratch comes from a small pool sized by how many callers run at once (`DeviceScratchPool`).
  The same change is made to the ROCm backend (not yet exercised on AMD hardware).
- **Tests**: new GPU tests drive every scratch-backed call from 50 short-lived threads and assert the
  scratch stays at one set, check device-wide free memory across 60 such threads, check that
  releasing returns the memory, and check four concurrent callers get bit-identical results. They
  failed on the previous code (device-wide: 42 MiB and 76 MiB taken by 60 threads).

- **The documented live-model integration command runs its test again.** `mvn verify -pl juno-master
  -Pintegration -DMODELS=...` reported success while running no test at all: the profile's empty
  exclusion list was merged with the default one, which excludes the very test the profile exists to
  run. It now runs `ModelLiveRunnerIT` (TinyLlama passes); the default `verify` still runs exactly the
  20 stub cluster tests. The `gpu` profile had the same defect and now runs `GpuForwardPassIT`.
- **The GPU-against-CPU integration test checks agreement the way it can be trusted.** Once it ran
  again, `GpuForwardPassIT` failed 2 of 4 on a fixed tolerance of 0.03 per value, which a correct GPU
  path cannot meet on hidden-state values above 100. It now compares relative error and cosine
  similarity of the hidden state, the top-1 and top-5 tokens and relative error of the logits, and a
  16-token greedy decode over the whole model that must match the CPU exactly. Every bound sits
  between the measured value and a deliberately introduced fault. It passes on TinyLlama and
  Mistral-7B, whose greedy output is identical on GPU and CPU. It reads the model's shapes from the
  file instead of assuming TinyLlama.
- **A tensor-parallel node that cannot load its model now says why.** A node that ran out of JVM heap
  while loading reached the coordinator only as `UNKNOWN: Application error processing RPC`, because the
  error was not an exception the node reported. It is now reported like any failed load, naming the
  node, its layers, the model file and the cause, for example `java.lang.OutOfMemoryError: Java heap
  space` on Mistral-7B with three tensor-parallel nodes, each of which loads the whole model; and a node
  that still answers with a bare error status is named by index and address.

**Session 96** — The decode activation can stay on the GPU from the norm through the Q/K/V projection and RoPE (`--gpu-residency`, off by default)

- **One residency region per layer at decode.** With `--gpu-residency on`, each layer of
  single-sequence decode uploads the residual row once, runs RMS norm, the Q/K/V projection and RoPE
  on one GPU stream, and downloads q, k and v together: one wait, as the default path already pays for
  its projection, with the host norm and the host rotation moved inside it. The projection and
  rotation kernels are the ones the default path runs; the region is bit-identical to the GPU
  operation-at-a-time path in every test, and against the default path only the norm's summation
  order differs (GPU against CPU).
- **Measured: 4% to 5% faster generation where it runs, unchanged where it does not.** TinyLlama
  generation went from 71.5 to 75.4 tokens per second and Mistral-7B from 23.4 to 24.2, medians of
  three. Greedy output matched the flag-off run token for token over 32 tokens on TinyLlama,
  Mistral-7B and LLaMA-30B. LoRA training and playback are unchanged with the flag set.
- **Where it cannot run, it says so once and keeps the existing path**: Qwen2 and Qwen2.5 (the
  split-half rotation layout and the Q/K/V biases have no device counterpart yet), the Phi and Qwen3
  handlers, LoRA training and `--lora-play` (adapters are applied to Q/K/V on the host), a non-CUDA
  backend, and any layer whose projections are not K-quant matrices on the device - so a model only
  partly on the GPU runs the region on the layers that are. Prefill windows and batched decode keep
  the existing path. Cluster nodes receive the setting; pipeline and tensor clusters answer with
  exactly local mode's output.
- **The region keeps no device memory per request.** Its device buffers come from a pool sized by
  concurrent calls, not by threads, because the server runs every request on a new thread: memory
  tied to a finished thread would never be used again.
- **New smoke test: `scripts/performance-tests/smoke-gpu-residency.sh`** - on against off on real
  models (greedy output, activation notice, per-request GPU memory), then pipeline and tensor clusters.

**Session 95** — Qwen models now rotate positions in the layout their weights were converted for, the CPU rotation stops recomputing its angles, and `min_tokens` holds against every way the model can end its own turn

- **The CPU rotary embedding computes each angle once instead of 792 times per token.** The angle of a
  rotated pair depends only on the position and the pair index, but the scalar rotation evaluated a
  power, a cosine and a sine for every pair of every head of every layer on every call. It now reads
  them from a per-position table (`RopeTable`) built with exactly the same expressions, so every
  output is bit-identical, checked against a verbatim copy of the old code across head sizes 64, 96
  and 128, bases 1e4, 5e5 and 1e6, and positions up to 32767. The table fills lazily in blocks,
  publishes each block whole so readers take no lock, allocates nothing per call, and is shared by
  every model with the same head size and base.
- **GPU generation is 12% to 26% faster on the models that use it, prefill 22% to 35%.** With every
  matrix product on the GPU, the CPU rotation had been 8% to 11% of each generated token's forward
  pass and 18% to 23% of a 128-token prefill; it is now under 1% of either. TinyLlama generation went
  from 56.8 to 71.5 tokens per second, Mistral-7B from 20.0 to 23.4, Qwen2.5-3B from 27.8 to 31.1.
  Phi-3.5, which rotates on its own path, is unchanged, and so is CPU inference, where the rotation
  was never a visible share. LoRA training is 7% faster end to end on the benchmark scenario, and
  playback 7.5%.
- **New benchmark references.** The GPU and CPU comparison sweeps were re-taken and are the
  references from here on; the previous ones are kept and marked superseded.

- **Qwen2, Qwen2.5, Qwen3 and Qwen3-MoE were served with the wrong RoPE pair layout.** Rotary
  position embedding rotates pairs of dimensions, either adjacent ones or the two halves of each
  head, and the right choice is fixed by how the file's Q/K weight rows were laid out at
  conversion. LLaMA-family files are converted for adjacent pairs; Qwen files are not permuted and
  need the split-half layout, but Juno applied adjacent pairs to them. The wrong layout does not
  fail: short factual prompts still came out right, which is how it went unnoticed. Measured as
  teacher-forced perplexity over a 700-token English text, Qwen2.5-3B was 169 under the old layout
  and 1.17 under the correct one, Qwen3-1.7B 526 and 4.05. TinyLlama, the control, is 5.27 under
  adjacent pairs and 7618 under split-half, so the method discriminates in both directions.
- **Generated text changes for every Qwen model; nothing else does.** The layout is chosen per
  architecture at load (`RopePairing`), never by a flag, since a switch that selects a known-wrong
  geometry would be a silent-quality hazard. Throughput is unchanged: the arithmetic per pair is the
  same. LLaMA, Mistral, TinyLlama, Phi and the vision backbones are untouched.
- **LoRA training on Qwen models uses the same layout forward and backward.** The Qwen2 training
  handler, the Qwen3 one, and the YaRN path all rotate and un-rotate in the corrected layout, so an
  adapter trains against the geometry it will be played back with. No Qwen adapter was trained
  before this change on the reference host, so none needs retraining; an adapter trained elsewhere
  on a Qwen model with an earlier release was trained against the wrong layout and should be
  retrained.
- **A perplexity ceiling per model now guards the layout.** `RopePairingPerplexityLiveTest` runs
  the production loader over 128 tokens of the same text on TinyLlama, Qwen2.5-3B and Qwen3-1.7B
  whenever the files are present; the full two-layout comparison runs on request.

- **A minimum token count could be cut short by a chat turn marker.** Generation stops on a
  vocabulary's turn markers (by token id) and on role headers in the decoded text, as well as on
  end-of-sequence, but the minimum held back only end-of-sequence. A request asking for 64 tokens
  could end at 3 on `<|im_start|>`, `<|end|>`, `<|endoftext|>`, or a `<|user|>` spelled out in
  text. The benchmark prompt never triggered it, so no published figure moved.
- **Below the minimum the model's own end signals are now all held back.** `MinTokenFloor` masks
  end-of-sequence and the vocabulary's turn-marker ids together, and yields only when nothing else
  is legal, as before. A role header or turn marker that appears in the decoded text below the
  minimum no longer ends the request; it is returned as ordinary text, not dropped, and it is never
  treated as a stop later. The same rule applies on all three generation paths: single request,
  static batch, and the continuous engine.
- **A stop the caller asked for still ends the request below its minimum**, by id or as a stop
  string, as the published contract has always said; a turn marker the caller names explicitly is
  therefore not held back. Both API contract files, the SDK javadoc, `docs/howto.md` and the
  comparison harness help now describe the rule as it is.

**Session 94** — Activations can stay on the GPU between operations, and removing the round trip between two of them halves their cost

- **A device-resident activation path now exists.** `ResidentChain` is a residency region - one GPU
  stream and the activation buffers allocated on it - and `ResidentActivation` is one such buffer.
  Data crosses between host and device in exactly two places: an upload where it enters the region
  and a materialize where it leaves, the only point the host waits for the device and sees its
  results. Operations in between work on device memory only. Closing a region frees every buffer on
  it, so a region cannot leak one; the tests assert that device memory returns exactly to where it
  started after every allocate-and-close cycle, and that the same query does see the allocation,
  so the check can fail.
- **Two operations run on it: RMS norm and a new GPU RoPE.** `CudaRmsNorm.normalizeResident` reads
  one resident buffer and writes another, with the layer's norm weight uploaded once instead of on
  every call. `CudaRope` rotates a resident buffer in place with a new kernel that computes the
  angle, sine and cosine in double and rounds each step of the rotation separately, exactly as the
  CPU path does - it is bit-identical to the CPU rotation in every parity test, including at
  position 30000, where a single-precision angle would already be off by about 2e-3 radians.
- **Measured: keeping the activation on the device between two operations halves their cost.** On a
  GTX 1080 at hidden size 2048, RMS norm then RoPE costs 0.51 of the op-at-a-time path at decode
  width and 0.55 at prefill width, with weights on the device in both, so the difference is the host
  round trip alone. Against the scalar CPU path the resident chain is 3.65 times faster at decode
  and 15.9 times at prefill, but that ratio mostly measures the CPU RoPE rather than residency: the
  CPU norm alone is still about five times cheaper than the resident chain at decode width.
- **The scalar RoPE turns out to be a tenth of GPU decode and nearly a quarter of GPU prefill.** In
  the standing GPU sweep, where every matrix product already runs on the GPU, RoPE on the CPU is
  8% to 11% of forward-pass time during generation and 23% during a 128-token prefill. It recomputes a
  power, a cosine and a sine for every rotated pair of every head of every layer, although the angle
  depends only on the position and the pair. Nothing changes here yet: the new kernel and a
  once-per-position CPU angle cache are both ways to remove it.
- **Kernel launches on the resident path allocate nothing.** A parameter block is built once per
  thread and rewritten in place, and the driver is called with its exact signature instead of through
  argument boxing, which stops being negligible once a launch is most of what a small operation costs.
- **Two uploads in a row can no longer corrupt each other.** Both go through one pinned host buffer,
  so an upload now waits for the previous transfer out of it if that one may still be running. The
  race only shows when device work is queued ahead of the transfer; the test queues milliseconds of
  it, and without the wait the device received the second rows in place of the first on every run.
- **New harness: `scripts/performance-tests/resident-chain-microbench.sh`.** Four lanes per width -
  scalar CPU, op-at-a-time, resident chain, and the two operations on an activation already on the
  device - with the same dispersion rule, correctness check and device-memory reading as the
  round-trip microbench it builds on.
- No performance gate: no handler constructs the new classes, so the forward pass, MatVec, KV and
  batching paths are unchanged. The existing per-call GPU norm path is also unchanged, because it is
  the baseline the resident path is measured against.

**Session 93** — A turn ends where the template says it ends, whether the model says so in text or in a token that decodes to nothing

- **A reply no longer continues into an invented conversation.** Asked "Hello", TinyLlama answered
  "Hey! How are you doing today?", then wrote a `<|user|>` header, invented the user's reply, wrote
  an assistant header and answered that too — all of it streamed to the console as though the
  assistant had said it. Generation only ended a hundred tokens later, when the model finally
  produced an end-of-sequence token. The turn had in fact ended at the first header: every chat
  template marks a speaker change with one, and one can never legitimately appear inside the
  assistant's own content. Those headers are now turn boundaries, so the answer stops at
  "Hey! How are you doing today?" and the fabrication is never streamed.
- **The damage compounded across turns, which is why this is not cosmetic.** The console appends
  each reply to the conversation history and sends it back as context, so a fabricated dialogue
  became the model's own record of what it had said, and every later turn was conditioned on words
  no one wrote. Truncating at the header keeps the history to what the model actually answered.
- **One marker vocabulary, one filter, every surface.** `ChatTurnMarkers` now holds both the
  turn-end markers already recognized (`</s>`, `<|end|>`, `<|eot_id|>`, `<end_of_turn>`,
  `<|im_end|>`, `<|endoftext|>`) and the turn-opening role headers (`<|user|>`, `<|assistant|>`,
  `<|system|>`, `<|im_start|>`, `<|start_header_id|>`, `<start_of_turn>`), and `EosOutputFilter`
  stops on either. Because that filter is wired into single-request generation, static batching and
  the continuous engine alike, the interactive console, the OpenAI-compatible route, the native
  route and vision all truncate identically. A header split across several decoded pieces is held
  back rather than streamed, as turn-end markers already were.
- **Mistral's `[INST]` is deliberately excluded.** Square brackets around a common word appear in
  ordinary prose and in code, and truncating an answer on them would cost more than it saves; that
  template closes the assistant turn with `</s>`, which is caught already.
- **A role header that decodes to nothing now ends the turn too.** Phi-3 stores `<|user|>`,
  `<|assistant|>` and `<|system|>` as control tokens, and a control token decodes to the empty
  string because it is prompt scaffolding rather than content. A model emitting one was therefore
  invisible: nothing reached the text filter and the reply ran to the token limit. Turn markers are
  now recognised by token id as well, from a single list the tokenizer and the filter share, so the
  two halves cannot drift apart again — which is how this survived, the text half knowing the
  turn-end markers while the id half knew only the configured end-of-sequence id.
- No performance gate: the forward pass, MatVec, GPU residency, batching and KV paths are untouched
  and the change is confined to decoded-text filtering and one membership check per sampled token.
  Doubling the marker set would have doubled a scan that ran over the whole answer on every token,
  so the filter now rescans only the window within one marker length of the last emitted character
  — the per-token cost no longer grows with the length of the answer. The marker ids are scanned
  out of the vocabulary once at tokenizer load, not per token.

**Session 92** — The cost of a GPU operation with no device-resident activation is now a measurement, not a remembered number

- **The claim that moving RMS norm to the GPU makes decode slower is reproducible on demand.** That
  finding is why `CudaRmsNorm` is built, parity-tested and then deliberately left unused, and it has
  been carried in prose ever since — roughly eleven times slower on one live comparison, somewhere
  between 1.56 and 2.18 times slower on another at prefill scale. Neither could be re-run. A new
  harness, `scripts/performance-tests/rmsnorm-roundtrip-microbench.sh`, times the scalar CPU norm
  against the GPU norm's per-call upload, kernel launch and download, and both claims hold: on a
  GTX 1080 at hidden size 2048 the GPU path is **10.6 times slower at decode width** (one row) and
  **1.60 times slower at prefill width** (512 rows).
- **The gap narrows with width but does not close, and the two widths fail for different reasons.**
  At one row a fixed per-call cost dominates and the GPU loses by an order of magnitude. At 512 rows
  that cost is spread across the batch and the loss falls to a factor of 1.6, while about four
  megabytes is staged in each direction. Both widths are therefore reported separately and neither
  is at parity, so a result at one says nothing about the other.
- **The harness refuses to report a number it cannot stand behind.** GPU output is checked against
  the scalar path on every run and timings are withheld if they diverge, so a changed kernel stops
  the run rather than quietly producing timings for a different computation. A row whose repetitions
  disagree by more than 15% of their median is marked unscorable rather than averaged into a
  conclusion — which caught a real bad reading immediately: at the harness's first warm-up default
  the decode GPU row's repetitions spanned 37.9% and were correctly refused, so the defaults were
  raised until the same row spanned 2.3%. Device memory is reported either side of the run, and
  collection pauses and allocation are captured through the shared recording configuration.
- **A device-memory figure now says when it is not ours.** The device reports free memory for the
  whole GPU, not per process, so a stray process alongside a run lands in the reading: during
  bring-up a leftover test JVM holding 7.2 GB made the harness report 4.1 GB unreturned, which is
  memory it never allocated. The report now carries device capacity next to free bytes and says
  plainly when retention is larger than anything the harness could be holding, so a contaminated
  reading announces itself rather than being published as a leak.
- No performance gate: the harness adds a standalone class that no handler references, so the
  forward pass, MatVec, GPU residency, batching, KV and quantization paths are unchanged and there is
  nothing for a regression gate to detect. This is not a measurement boundary.

**Session 91** — A model's declared pre-tokenizer split is read and applied, and a file declaring one Juno has not implemented is refused

- **Text is now cut the way the vocabulary was trained to cut it.** A GPT-2 BPE vocabulary is built
  over text that was first split into pre-tokens by a fixed pattern, and its merges are only ever
  learned inside one of them. Juno merged a whole run of text in one pass, which admits pairs the
  training never produced, and nothing threw when it happened — the model simply received a token
  sequence it had not been trained on. The `tokenizer.ggml.pre` key names the split each file was
  built with, and Juno now reads it, cuts the text accordingly, and merges each piece on its own. Two
  divergence classes disappear as a result: a run of whitespace before a word (`"a  b"` was becoming
  `"a"` + `"  "` + `"b"` rather than `"a"` + `" "` + `" b"`, because the last space of a run belongs
  to the word after it) and groups of digits, where a date or a version string was grouped by
  whatever the merge table happened to allow. Measured against a second engine's tokenization of the
  same 33-line corpus, Qwen2.5-3B went from five divergent lines to none, Qwen3-1.7B from five to
  none, and Llama-3.2-1B from eight to none.
- **A file declaring a split Juno has not implemented is refused by name.** The implemented set is
  `qwen2` and `llama-bpe`, each checked line by line against a second engine's output on real files.
  Three models on disk declare something else, and they are now rejected at load with an error naming
  the declared type and listing what is implemented, rather than tokenized under a split that was
  never verified for them. All three were already refused for their architecture, so no file that
  loaded before this session fails after it; in single-node mode the tokenizer is read before the
  handler, so those three now report the tokenizer refusal rather than the architecture one.
- **A file that declares nothing is untouched, and that was verified rather than assumed.** The key
  governs BPE vocabularies only — a SentencePiece vocabulary takes no pre-tokenizer split, its word
  boundaries coming from the `▁` prefix — so it is not consulted for one, and a declared `default`
  means exactly what an absent key means. The same corpus was encoded before and after the change on
  the five files that declare nothing (TinyLlama, Phi-3.5-mini and Mistral-7B on SentencePiece,
  Phi-2 and moondream2 on GPT-2 BPE): identical token IDs on every line. The load line now names the
  active split, and a BPE vocabulary running without one says so.
- No performance gate: nothing here touches the forward pass, MatVec, GPU residency, batching, KV or
  quantization. This is not a measurement boundary either — the benchmark prompt is single letters
  separated by single spaces, which contains none of the constructs the split moves, and a harness
  run confirms the same prompt-token count at the same calibrated word length.

**Session 90** — Juno is measured warm, repeated and at the prompt length it was asked for

- **A prefill comparison now actually compares prefills.** The previous session made a prefill ratio
  conditional on both engines having prefilled comparable work, and withheld it otherwise. That rule was
  right, but nothing could satisfy it: the raw-prompt mode counted words, while the chat template wraps
  every request in role and control tokens — about 19 of them on TinyLlama — so a 128-word prompt
  prefilled 146 tokens, 14% over the requested count, and a 32-word prompt prefilled 50, 56% over. Every
  prefill ratio in the sweep would have been withheld, which is the one column the comparison exists to
  move. Each measured cycle now sends one short calibration request, reads back the `prompt_tokens` the
  engine actually produced, and corrects the word count by the difference; on TinyLlama at a requested
  128 that lands on 128 exactly. Nothing assumes a particular template, and the withholding rule still
  has the final say.
- **A generation ratio now requires tokens to have been generated.** The reference tool generates the
  requested token count whatever the model would rather do; Juno stops at a stop token. On a real sweep
  at 64 requested tokens, one model generated 64, one 49, one 22, and Qwen2.5-3B generated none at all,
  emitting a stop token immediately on every repetition. The comparison reported that last case as a
  ratio of `0` — which reads as infinitely slower than the reference when it means never measured. A
  run that generated nothing now withholds the generation ratio and states the reason and the finish
  reason instead, and the index carries a generated-token actual/requested column beside the
  prompt-token one. Where a model generated some but not all of the requested tokens the ratio is still
  published, with the shortfall noted: that reading is an average over a shorter and slightly cheaper
  span of context, an effect smaller than this host's measurement floor, so stating it beats discarding
  three models' figures.
- **`min_tokens` closes that gap, on every surface.** A request can now state how many tokens it must
  produce before an end-of-sequence token may end it, which is what makes a generation comparison
  like-for-like instead of merely honest about not being one. The new `MinTokenFloor` suppresses the
  end-of-sequence token below the minimum rather than ignoring it once sampled, so the model falls
  through to its next-best continuation instead of leaking a marker into the output, and it yields in
  the one case where a grammar has left end-of-sequence as the only legal token — masking it there
  would leave a distribution that cannot be sampled from. It is wired into all three generation paths
  (single request, static batch, continuous running set), both REST surfaces (`min_tokens` on chat
  completions, `sampling.minTokens` on the native route), the published contracts, and the embedding
  facade. A minimum above the maximum is rejected rather than clamped, so a caller is never quietly
  given fewer tokens than it asked for. On the model that previously generated nothing at 64 requested
  tokens, generation is now 32 of 32 with a published ratio where there had been no reading at all.
- **The native inference surface answers 400 for sampling it cannot honour.** The sampling parameters
  validate their own ranges and throw, and nothing on that surface caught it, so an out-of-range
  temperature or token count came back as a server error telling the caller Juno had broken rather than
  that the request was wrong. The chat surface already mapped these to 400. Both native routes,
  blocking and streaming, now do too — this was a pre-existing gap across every validated field, not
  only the new one, and the test covers temperature as well as the new minimum.
- **Juno readings are warm, repeated, and taken at a fixed heap.** The reference tool runs its own warmup
  and repetitions, while Juno was measured from a single request against a just-started JVM — so the
  compilation of the entire forward pass sat inside the measurement window, and every Juno figure on
  record before this is a cold one. `compare-llama-cpp.sh` gains `--juno-warmup` (default 2) discarded
  requests and `--juno-reps` (default 3) measured cycles, publishing the median with the min/max spread
  beside it, because a difference smaller than that spread is not a result on this host. A cycle is a
  whole engine start rather than another request in the same process, since model load, page-cache state
  and device residency all sit inside one. The JVM heap is now a fixed value per model rather than derived
  from the file size, so the collector does the same work in a run as in the baseline it is compared
  against; an off-table model keeps the derivation and its result is labelled `derived`. Each run also
  records the CPU governor, the turbo state and the GPU clocks with any active throttle reason, because a
  thermally throttled run and a real regression are otherwise indistinguishable.
- **The measurement window is now scoped to the request being measured.** Warmup forced this: warmup
  requests have to run in the same process as the measured one, since what they buy is compiled code, but
  a recording started with `--jfr` covers the whole process lifetime and would have held the discarded
  requests too. That is not a rounding error — the token-throughput figure is computed across the span
  from the first recorded token to the last, so it would have divided the measured token count by a span
  containing the warmups and the idle gaps between them. The comparison harness now starts the recording
  itself once the warmup requests have returned and stops it before the engine exits, under the same
  settings file every other recording site names, so overhead is unchanged. Verified on a real process:
  with two warmups and an 8-token measured request the recording holds 8 token events, not 24. `--jfr
  DURATION` becomes an upper bound on that window rather than the window itself, and a recording that
  reaches the bound still dumps.
- **Metrics can be extracted from a recording you took yourself.** The existing entry points serve the
  running engine — scan the working directory and map against `models.json`, or extract programmatically
  at shutdown — and both write to a fixed relative path, so a caller could only steer the output by
  changing its working directory. The new `JfrMetricsCli` takes one named recording to one named output
  file, needs no `models.json` entry, and fails on a missing or empty recording instead of writing a
  report of zeroes, which would otherwise read as a run with no collection pauses and no tokens. Covered
  by `JfrMetricsCliTest` (6 cases); the harness arithmetic is covered by a `--selftest` mode on the
  comparison script itself (24 checks, no model required).
- **Prefill and generation are now measured in two separate runs, and that was worth about half the
  generation figure.** The reference tool benchmarks the two separately and measures generation from an
  empty context; Juno measured both in one request, so its generation ran at the prompt length while the
  reference ran near zero, and decode slows as context grows. Harmless while Juno prefilled a short
  sentence, this became the dominant error once prompt-token parity raised the prompt to the requested
  length: on Phi-3.5-mini generation read 12.38 t/s measured after a 128-token prefill against 24.42 t/s
  measured in its own run. The comparison now runs a prefill run and a generation run per model and takes
  each figure from the run that measured it. Juno cannot reach a truly empty context, since the chat
  template wraps every request, so the generation run prefills about ten tokens against the reference's
  zero; that residual is recorded in every result rather than hidden.
- **A measurement now declares itself unscorable when its own repetitions disagree.** Each published
  index carries a `Scorable` column: a generation reading whose cycles span more than 15% of their median
  is not stable at the resolution a gate reads it at, and says so with the reason. This replaces a rule
  based on the longest collection pause, which was built first and then withdrawn on measurement — on
  this host the pause counter does not report time the application was stopped. One model produced 64
  tokens across token spans of 1110, 1120 and 1107 ms while its three pause readings were 633 ms, 5 ms
  and 4 ms; a 633 ms stop-the-world inside a 1110 ms span would imply more than twice the generation rate
  the model can reach. The pause rule rejected three readings that agreed to within 1% and passed one
  that spanned 31%, which the dispersion rule gets the right way round. Collection pauses are still
  reported, now including the share that overlapped the measured token span, as context rather than as a
  gate; a pause that does cost time appears as one slow repetition anyway. The same applies to lock and
  park totals, where the figure sums every thread and an idle pool exceeds wall time on a healthy run.
- **A test that failed one run in three no longer does.** A metrics assertion required the prefill and
  decode duration totals to equal the overall total by exact floating-point comparison. Both sides add up
  the same measured durations in different groupings, and that addition is not associative, so they could
  differ in the last bit. It now permits a difference of one part in a billion, which is still far tighter
  than any real miscount, and passed twelve consecutive runs. It had been halting the documented
  all-module test command before its last two modules.
- **The performance harness no longer leaves a process behind for every engine it starts.** Keeping the
  console session alive needs its input never to reach end of file, and that was arranged with a
  background loop that nothing ever cleaned up — one per launch, each respawning hourly, so they
  accumulated until a check for "is a benchmark still running" answered yes to processes from days
  earlier. The comparison harness now holds a named pipe it owns and releases it with the engine; a
  three-repetition run that previously left six leftovers now leaves none. Eight sibling smoke and
  comparison scripts still carry the same pattern and are unchanged for now, since several cannot be
  exercised without a GPU and real models.
- Unit tests: 1646 across eleven modules, 0 failures, 0 errors (46 skipped, all GPU-, ROCm- or
  missing-model-gated as before), in under 25 minutes.
- No performance gate: nothing here touches the forward pass, MatVec, GPU residency, batching, KV or
  quantization, and the only source change is an extraction entry point no inference path calls. This
  session is a measurement boundary, though, and a larger one than the last: a reading taken after it is
  warm, is a median of three cycles, sits at a fixed heap and prefills the requested token count, and a
  reading taken before it is none of those. Prefill ratios published earlier read as better than a
  like-for-like measurement supports, and generation ratios read as worse, because they were cold.

**Session 89** — `--gpu-layers auto` keeps device memory free for the forward pass, and every device allocation falls back to CPU instead of ending the process

- **A model larger than the card no longer dies at the first prompt.** `--gpu-layers auto` uploaded weight
  layers until the allocator refused, caught that, and kept what fit — so it finished with the card full and
  called it success. But weight upload is not the only consumer: a prompt wider than eight tokens takes the
  batched path, which dequantizes a whole packed weight matrix into a device scratch buffer (227 MiB for a 30B
  Llama), and the GPU-resident attention path allocates a key/value mirror. Both happen after the upload, so a
  model that loaded and decoded single tokens perfectly well aborted on the first real prompt with
  `cudaMalloc failed: rc=2`. `auto` now stops while that memory is still free, sized by the new
  `DeviceScratchBudget` from the widest matmul in the model plus the KV mirror, and learns the per-layer cost
  by measuring free device memory across the first upload. Covered by `DeviceScratchBudgetTest` (14 cases,
  no GPU required).
- **Every device allocation on the inference path now degrades instead of propagating.** The batched matmul
  falls back to the weight-stationary CPU path; the attention KV mirror and the attention kernel fall back to
  CPU attention, which is safe because the CPU key/value tensors are always written and the device copy is
  only ever a mirror. The attention fallback is taken at the dispatch boundary rather than at each allocation
  inside the kernel, so scratch the kernel allocates internally is covered by the same guard. Each path warns
  once rather than per matmul. Verified on an 18.2 GiB model on an 8 GiB card: four prompts in one session,
  coherent output, no failure, where the same command previously aborted on the first.
- **A mirror that has been given up stays given up.** The first cut of the fallback above dropped the device
  key/value mirror by unmapping it from the per-request table, then cleared the flag that had suppressed it.
  The next token found no mapping, so `computeIfAbsent` allocated a replacement — and device memory is never
  zeroed, so every position before the current one held whatever the allocator last left there. Attention was
  handed `seqLen = pos + 1` over that buffer and read it as conversation history. Nothing failed: token
  counts, timings and exit status all looked healthy while the logits were noise, and output became token
  soup from the moment the first fallback fired. A 30B Q4_K_M model on a card that could not hold it answered
  one short prompt and then produced nothing but garbage, while `JUNO_GPU_ATTENTION=off` stayed coherent on
  the same build — so the four-prompt verification above only held because that session never tripped the
  fallback. `DeviceKvCache` now tracks `validTokens`, the contiguous run of positions actually written, and
  `readableThrough(seqLen)` / `live()` gate every attention dispatch and every append. A mirror that runs out
  of memory is retired by closing it in place, never by unmapping it, so no empty replacement can appear
  behind it. Retirement is per mirror rather than per handler, so one layer or one request giving up no
  longer punches holes in the mirrors of the others — the previous flag was handler-wide and was reset after
  a single pass. The same watermark covers a request whose prefix was restored into the host tensors by
  `NodeKVCacheAdapter`, which never had matching device rows to begin with. Covered by
  `DeviceKvMirrorWatermarkTest` (5 cases, GPU-tagged).
- Performance gate: `compare-lora.sh --gpu --baseline 1f90b68 --current HEAD --reps 3`, published under
  `perf-compare/20260924T201548Z-lora`. Train **1.00x**, playback **1.00x**, status ok. The three ratios read
  as exactly 1 because train time is recorded at 1 s granularity (45000 ms on all six repetitions) and the
  median playback window landed on the same millisecond on both sides; the underlying repetitions do differ
  (576/597/575 ms baseline against 576/591/567 ms current), so this is a granularity artifact rather than one
  run reported twice. Recorded GPU dispatch counts are unchanged — 21913 against 21912 resident, 83190
  resident-transpose on both sides, and **0 CPU MatVec dispatches on both**, which is the result that matters
  here: the added guards displace nothing onto the CPU path. Note this gate cannot cover the mirror itself.
  `juno lora` ignores `--gpu-attention` (the run log carries that notice), so no device KV mirror is ever
  built on the LoRA path and every guard short-circuits before touching one; what the run establishes is the
  absence of collateral damage to the forward pass and MatVec. Mirror coverage is
  `DeviceKvMirrorWatermarkTest` plus a real multi-turn run on a model larger than the card.
- **Device scratch buffers stay consistent when an allocation fails.** `Q4KDequantScratch` and the FP16
  staging buffers freed the old buffer before allocating the new one and recorded the new size afterwards, so
  a failed allocation left the field pointing at freed memory and the size stale. That was unreachable while
  an exhausted device ended the process; making the failure survivable exposed it as an unrelated
  `cudaMemcpyAsync ... rc=1` on the next call. They now clear the field and size first, so a failure leaves an
  empty buffer that the next call re-allocates.
- Performance gate: `compare-lora.sh --baseline HEAD~1 --reps 3 --gpu`, published under
  `perf-compare/20260924T185248Z-lora`. Playback **0.99x**, train **1.02x**, both inside the measurement
  noise floor. The stronger result is that the recorded GPU operation counts are identical on both sides
  (21913 resident, 83190 resident-transpose, 0 CPU) with identical allocation, so the reserve displaces no
  layer on a model that fits the card. Three of six repetitions on both sides carried a ~635 ms collection
  pause; they fall in the training phase rather than the measured playback window, which is why that gate
  reads wall-clock time.

**Session 88** — One JFR configuration for every recording, GC and allocation in every metrics report, prompt-token parity for prefill comparisons

- **Every JFR recording is now taken under one settings file.** Juno starts recordings from six places, and
  they disagreed: the launcher `test` command used `profile` on both platforms, forked cluster-node JVMs used
  `default`, and the local, LoRA and cluster-coordinator recordings built their own from `default`
  programmatically. Two runs could therefore differ by their instrumentation overhead alone, which made every
  published throughput comparison weaker than it looked. `scripts/performance-tests/juno-perf.jfc` is now the
  single configuration, resolved by the new `JunoJfrSettings` for the in-process recordings and named directly
  by `scripts/run.sh`, `scripts/run.bat` and `ClusterHarness`. It is packaged into the `juno-player` jar, so a
  run started from a jar resolves the same settings a script-launched one does; override it with
  `JUNO_JFR_SETTINGS` or `-Djuno.jfr.settings`. A settings file that cannot be resolved falls back to the JDK
  stock settings and says so rather than changing the overhead quietly. The AWS deployment script keeps its own
  settings: it launches remote instances, not the benchmark host. Native-thread stack sampling deliberately
  stays at its stock interval — tightening it was behind a JFR sampler crash under `--gpu --mmq` on
  `--lora-play`.
- **The metrics report now covers the JVM, not only Juno.** `JfrMetricsExtractor` consumed `juno.*` events and
  nothing else, so no run could report a garbage-collection pause or an allocation rate — and a short
  measurement window holding one long pause reports a throughput drop that looks exactly like a code
  regression. The new `JdkEventBucket` adds `jdk.GCPhasePause` (count, maximum and total), allocated bytes,
  sampled allocation volume with a bounded per-site breakdown, execution samples with a bounded per-method
  breakdown, and monitor-contention and park time. Allocated bytes is the sum over threads of each thread's own
  running total, not of every sample, since the underlying field is cumulative per thread. Covered by
  `JfrMetricsExtractorJdkEventsTest`, which computes its expected values from the same recording so the
  assertions hold however many events the window caught.
- **A prefill comparison now requires both engines to have prefilled comparable work.**
  `compare-llama-cpp.sh` prefilled a fixed 57-character sentence unless `--raw-prompt` was passed, against a
  reference prefilling `n_prompt` tokens — roughly 20 tokens against 128 in the runs on record. That default is
  reversed (`--no-raw-prompt` restores the old behaviour for a Juno-only measurement), every result JSON records
  the real `prompt_tokens` beside `n_prompt`, and a prefill ratio whose deviation exceeds 10% is withheld with
  its reason stated in the JSON and in the published index rather than reported as a throughput gap. Generation
  ratios are unaffected. The index also gained garbage-collection and allocated-bytes-per-token columns and now
  states the unmatched thread count between the two engines explicitly.
- **The launchers size the JVM heap from the model file.** Juno reads GGUF tensors onto the Java heap, so
  the model has to fit in `-Xmx`, but every command defaulted to a fixed `4g` regardless of the model. A
  model larger than roughly 1.5 GiB of weights therefore died with an `OutOfMemoryError` naming whichever
  tensor happened to cross the limit, with nothing pointing at the heap — an 18 GiB model failed this way on
  a 62 GiB machine. `run.sh` and `run.bat` now derive it for `cluster`, `local`, `lora`, `test`, `merge` and
  `lora-import`: file size x1.5 plus 2 GiB, clamped to 4g..48g, which is the formula the performance
  comparison harness already used, so an interactive run and a benchmark run of the same model now get the
  same heap. An explicit `--heap` or `HEAP` still wins, and the derived value is printed when it is used.
  With no local file to measure — the `--hf` case before a download resolves — it stays at `4g`.
- No performance run: nothing here touches the forward pass, MatVec, GPU residency, batching, KV or
  quantization. The JFR configuration change does break strict comparability with previously published runs, so
  the first run taken under it becomes the new reference.

**Session 87** — Cluster loads fail closed, dead registry/Hazelcast surface removed, REST temperature 0 is greedy

- **A cluster node whose model cannot be loaded no longer serves stub output.** `EmbeddedNodeServer.loadShard`
  used to catch any load failure (unsupported architecture or quantization, missing tensor, missing file, bad
  adapter), install a fixed-logit stub and reply `success = true` with a message beginning `Stub shard`, while
  `ProcessPipelineClient` and `TensorParallelPipelineClient` only logged that message. The result was a
  healthy-looking cluster returning dummy logits. Now the node replies `success = false` with the reason and
  installs `UnloadedShardHandler`, which refuses forward passes (a node started with a real model also refuses
  them before its first successful load); both clients fail `loadShards()` on any node that reports failure; and
  `ClusterHarness.start()` stops the forked node JVMs before rethrowing, so a failed start leaves nothing running.
  Stub mode (no model path, used by tests) is unchanged. Covered by `EmbeddedNodeServerLoadFailureTest`, failure
  cases in `LoadShardsParallelTest` and `TensorParallelPipelineClientTest`, `UnsupportedArchitectureClusterIT`
  (forked JVMs, metadata-only GGUF) and, on real files, `ModelLiveRunnerIT` and the smoke script
  (`./juno cluster` in both modes, and no leftover `NodeMain` processes).
- **`RegistryService` and Hazelcast removed.** No source file used a Hazelcast API; the `RegistryService` RPCs
  (`GetShardMap`, `RegisterNode`, `RecomputeShards`) and their seven messages had no implementation and no
  caller. The proto section, the `hazelcast` dependency in `node`, `health`, `kvcache`, `registry` and
  `coordinator`, the root `hazelcast.version` property and managed dependency, and the unused port 5701 rule in
  `scripts/aws/juno-deploy.sh` are gone, and the comments that described a Hazelcast-backed registry now describe
  the in-memory code. Membership is fixed at cluster launch.
- **A REST `temperature` of 0 now selects greedy decoding.** `/v1/chat/completions`, `/v1/inference` and
  `/v1/vision/chat` only set the temperature, `TemperatureStep` skipped scaling below 1e-6, and `SampleStep` drew
  randomly unless the `greedy` flag was set, so `temperature: 0` sampled at temperature 1 (identical requests could
  return different text) while the CLI, which sets the flag, was deterministic. `SamplingParams.effectivelyGreedy()`
  (flag or temperature below 1e-6) is now the single rule used by the temperature, top-k, top-p and sample steps.
- No performance run: none of the changes touches the forward pass, MatVec, GPU residency, batching or KV code;
  the sampler change is one comparison per sampled token.

**Session 86** — Correctness and consistency audit: prefix-cache ownership on batched paths, fail-closed architecture dispatch, doc/code drift

- **Static micro-batching no longer resumes from KV it did not write.** `GenerationLoop.generateBatch()`
  consulted the prefix trie and registered a `<requestId>:prefix` entry per request, although each batched
  request's pipeline KV is keyed by its own request id and evicted when it finishes. A repeated (or extended)
  prompt in a later batch matched the stale entry, skipped prefill and decoded over zeroed KV; on real TinyLlama
  the greedy output degenerated (`[22443, 23600, 6845, 15945, 13, 13]` instead of `[1576, 7483, 310, 3444, 338,
  3681]`). Adding a session gate would not have been enough, because a session request inside a batch is also
  keyed by its request id, so the batch path now never reads or writes the trie and prefills every prompt in
  full. Reproduced first with a KV-ownership test double (`KvTrackingPipeline`, which records any forward pass
  that continues from a position not written under the same key) and on real weights
  (`TinyLlamaStaticBatchLiveTest`).
- **Same defect in `ContinuousBatchEngine`:** every non-hit slot registered a `<key>:prefix` trie entry, which
  dangles for stateless slots (their KV is evicted at retirement), so a later session request with the same
  prompt skipped prefill against KV it never wrote. Only session slots register trie entries now.
- **Unrecognized architectures no longer fall through to the dense Llama handler.** `ForwardPassHandlerLoader`
  now runs `LlamaTransformerHandler` only for `llama`, `mistral`, `tinyllama`, `qwen2` and `qwen2.5`
  (`LlamaFamilyArchitectures`), and rejects everything else without a dedicated handler with an `IOException`
  that names the architecture, before any tensor is read. Observed on the real files: `qwen35` failed on an
  unrelated missing tensor, `gemma4` failed with a heap error (or, given enough heap, on a missing tensor after
  loading 24 layers with sliding-window attention and logit softcapping ignored), and `mistral3` / `minimax-m2`
  failed on unsupported quantization types; a variant of any of them with the right tensor names would have run
  and produced fluent wrong output. `ForwardPassHandlerLoader.isSupportedArchitecture` exposes the same rule and
  `ModelLiveRunnerIT` asserts the rejection for such files instead of running the suite on them.
- Corrected documentation that described intent rather than wiring: `TensorShardContext`,
  `TensorParallelPipelineClient` and `ClusterHarness` (no node slices its weights; every node runs the full model
  and the coordinator sums N complete logit vectors), `FaultTolerantPipeline` (built and tested, not constructed
  by the production launch path), `CudaRmsNorm` (documented as automatic, actually never constructed), the
  sampler pipeline order (`Sampler` is now the only place it is stated; step javadocs are no longer numbered),
  the GBNF bounded-repetition cap (why it exists and that its value is not measured; the error now states the
  limit), the LoRA optimizer (AdamW with LoRA+ groups, not Adam), the CPU SIMD description, the documented
  `mvn test` module list (now includes `vision` and `metrics`) and the live-IT invocation (`-DMODELS=`), and the
  `docs/agent-arch.txt` sections for architecture dispatch, prefix-cache ownership, tensor parallelism and fault
  tolerance. Internal planning-tier numbers and planning-file pointers were removed from 33 `src/main` files,
  including two exception messages and two CUDA kernel headers.
- Tests: `StaticBatchPrefixCacheSessionGatingTest`, `ContinuousPrefixCacheGatingTest`,
  `TinyLlamaStaticBatchLiveTest`, `ForwardPassHandlerLoaderArchitectureTest`, plus
  `scripts/performance-tests/smoke-consistency.sh` (real-file architecture audit and a batching check over
  both REST surfaces on CPU and GPU). No performance run: the change removes a trie lookup and write from the
  batch path and one redundant write per continuous slot, and touches no MatVec, KV-layout, GPU-residency or
  quantization code.

**Session 85** — Prefill GPU-residency fixes: pinned staging memory + adaptive chunk sizing (Phase A) **feature complete**

- Pinned host-staging memory: new vendor-neutral `GpuBindings.hostMalloc`/`hostFree`
  (`cudaMallocHost`/`cudaFreeHost`; `hipHostMalloc`/`hipHostFree` on ROCm) replaces plain
  `Arena.ofConfined()` staging in `CudaMatVec`'s three batched-GEMM paths
  (`sgemmHalfBatched`, `sgemmHalfBatchedGemm`, `sgemmQ4KBatchedGemm`) and
  `CudaRmsNorm.normalizeBatch`, grown-and-kept-max the same way the existing
  device-side scratch already is.
- Adaptive whole-prompt prefill chunk sizing for `--schedule static` on GPU:
  `PrefillBatchOptions.resolveAdaptive` sizes the chunk to cover the whole
  prompt in one window when there is enough free VRAM, using a new live
  `GpuBindings.memGetInfo`/`GpuContext.freeVramBytes()` query
  (`cudaMemGetInfo`/`hipMemGetInfo`) — the plan's original assumption that
  `--gpu-layers auto`'s existing budget check could be reused turned out to
  be wrong on inspection (it is a reactive per-layer upload-and-catch-OOM
  loop, not a predictive query), so this needed its own minimal binding.
  Floored at the old fixed 32 (can only grow the chunk vs today, never
  shrink it) and capped at 65536 tokens; `--prefill-batch N` still works as
  an explicit override on every surface; `--schedule continuous` and
  CPU-only runs keep the fixed 32-token default unchanged.
- Wired into the local single-shard REPL only (`ConsoleMain.runLocalRepl()`),
  which covers base inference, `--lora-play`, `--parallel`, and vision since
  they share the same `GenerationLoop`; cluster REPL and the LoRA train REPL
  keep the old fixed-32 path, named follow-up.
- Live smoke test (real GTX 1080, `mistral-7b-instruct-v0.1-q4_k_m.gguf`,
  `--gpu-layers auto --mmq auto`, real 488-token chat prompt): adaptive
  default resolved to a 24889-token chunk (1 `PrefillBatch` call vs 16 under
  the old fixed 32), measuring **-30% prefill time, -27% request wall time**,
  3.06x fewer `MatVec` launches. Correctness relies on the pre-existing
  `LlamaTransformerHandlerPrefillChunkParityTest` (chunk-boundary numeric
  identity already proven generically across chunk sizes, not just 32).
- Phase B checkpoint (re-measure GPU-resident Rope/SwiGlu under pinned
  memory, go/no-go): **no-go** — still 1.56-2.18x slower than CPU scalar,
  confirming the bottleneck is per-launch overhead with no
  activation-residency chain, not memcpy speed. `RopeKernel`/`SwiGluKernel`
  not built.
- This session's `nsys` install could not reproduce the original
  `cudaMemcpyAsync`-collapse timeline verification (fails on every
  invocation, including a bare argument-free `nsys profile -- echo hi` — an
  environment defect); end-to-end wall-clock evidence and green
  parity/regression tests are relied on instead, honestly flagged as an open
  re-verification.
- §2 regression: `compare-llama-cpp.sh --gpu` (default 4-model set),
  failures=0; `compare-lora.sh --gpu --baseline release-0.1.2`, **ok** (train
  1.00x, playback tps 0.87x ≥ 0.80 gate).

**Session 84** — Draft-model speculative decoding (`--spec-type draft-simple`) **feature complete**

- `--spec-type none|ngram-simple|draft-simple` + `--model-draft PATH`
  (`JUNO_MODEL_DRAFT`; local REPL only). `draft-simple` drafts tokens from a
  second, independently-loaded GGUF model instead of an in-request ngram
  cache, reusing the same `GenerationLoop.generate()` draft/verify loop and
  batched `forwardVerify` path the ngram mode already established. Both
  strategies now implement a shared `DraftProposer` interface
  (`propose`/`observe`/`close`) so the loop itself does not care which one is
  active.
- New `DraftModelSession` (coordinator module): drives the draft model
  through its own persistent KV session with ordinary greedy `forward()`
  calls (one per drafted token, feeding its own prediction back in — the same
  shape a plain non-speculative decode step already takes), then reconciles
  that tentative continuation against ground truth after every round by
  walking forward from the last agreed position and issuing at most one
  corrective `forward()` call on the first disagreement — no bulk resend of
  the drafted window, and no explicit KV-truncate API, since KV storage is
  indexed by absolute position and a later real write simply overwrites a
  stale speculative one.
- `GenerationLoop`'s constructor fails closed when `--spec-type draft-simple`
  is given without a loaded draft pipeline, or when the draft and target
  models' vocabulary sizes differ — draft-proposed token ids are compared
  directly against the target's own sampled ids, so a vocab mismatch would
  otherwise silently compare incompatible id spaces.
- Wired for the local single-shard REPL only (`ConsoleMain.loadDraftPipeline()`
  loads the draft model sharing the target's `MatVec`/`GpuContext`, never
  carrying LoRA adapters or vision wrapping); `--lora-play`, LoRA train, and
  cluster/tensor-parallel launches fail closed at CLI-parse time with an
  explicit error rather than silently ignoring `--model-draft`.
- Live smoke test (TinyLlama Q4_K_M as `--model-draft`, Mistral-7B Q4_K_M as
  the target, both sharing the same 32000-token vocabulary, GTX 1080, greedy
  decode, same maximally-repetitive prompt as the ngram smoke test): output
  byte-identical to `--spec-type none` (token identity holds), draft
  acceptance **55.2%** — but wall-clock tg **regressed to 0.52×** (19.78 →
  10.35 t/s), not an improvement. `juno.MatVec.count` nearly quadrupled
  (7,965 → 31,058) because the draft model's own decode/prefill/resync calls
  route through the same global `MatVec` span the target uses — unlike the
  ngram cache's free lookups, a draft *model*'s proposals cost a real forward
  pass per token, and on this GPU that cost is not offset by the verify-side
  savings. Honestly reported as a negative result, not hidden — see
  `docs/performance.md` / `docs/perf-compare/README.md`.
- §2 regression: `compare-llama-cpp.sh --gpu --models mistral` (default
  `--spec-type none`), failures=0; `compare-lora.sh --gpu --baseline
  release-0.1.2`, flat as expected (LoRA never routes through `forwardVerify`
  or touches `--model-draft`).
- Vision, ROCm, and the remaining handler families' `forwardVerify` overrides
  remain the same named follow-ups the ngram mode already carried — not
  reintroduced or re-scoped here.

**Session 83** — Ngram speculative decoding (`--spec-type`) **feature complete**

- `--spec-type none|ngram-simple` (`--spec-ngram-n`, `--spec-ngram-m`; default
  `none`, byte-for-byte identical to the pre-speculation decode path).
  `NgramDraftCache` (coordinator module) indexes every `n`-gram seen so far
  (prompt + generated tokens) against the token that followed it — no second
  model, no static corpus — and `GenerationLoop.generate()` drafts up to `m`
  tokens per round, verifies the whole window against this model in one
  batched pass, and emits the target model's own prediction at the first
  mismatch.
- New node-module primitive: `ForwardPassHandler.forwardVerify` /
  `InferencePipeline.verifyDraft` / `VerifyBatchResult` — sibling to the
  existing `forwardBatch`/`BatchForwardResult` prefill path, but keeps every
  window position's logits instead of discarding all but the last.
  `LlamaTransformerHandler.forwardVerify` reuses the same
  `runLayersBatch`/`outputProjectionBatch` machinery `forwardBatch` already
  has (one GEMM per weight per layer over the whole window), so Llama /
  Mistral / Qwen2 get real batched-GEMM verification; Phi-2/Phi-3/Qwen3/
  Qwen3-MoE and cluster/tensor-parallel pipelines fall back to the
  correctness-preserving serial default (no speed benefit there yet, named
  follow-up).
- Sampling runs exactly once per verify position, in order, and always emits
  its result whether or not it matches the draft — this keeps rng/grammar
  state, and therefore the emitted token, byte-identical to plain decoding
  regardless of draft accuracy (verified both by unit tests and a live
  TinyLlama smoke test at temperature 0).
- Wired for `GenerationLoop.generate()` (single-request decoding) only;
  `generateBatch` (static multi-request batching) does not draft/verify yet —
  the local REPL/API launcher warns at startup when both `--spec-type` and a
  `--parallel > 1` batch config are configured together, rather than silently
  dropping speculation under concurrent load.
- **A real correctness bug was found and fixed via a live smoke test, not the
  unit suite**: the first cut fed drafted tokens directly as the verify
  window's input row-for-row, silently overwriting the KV entry for the
  already-confirmed token at the window's first position with an unverified
  draft token's embedding — garbled output against a real model, even though
  a scripted (non-causal) test double couldn't catch it. Fixed by shifting
  the verify window by one position (row 0 = the already-confirmed last
  token, rows 1..M-1 = the first M-1 drafted tokens). See
  `docs/perf-compare/README.md`'s writeup for the full story.
- Live smoke test (TinyLlama Q4_K_M, GTX 1080, maximally-repetitive prompt,
  greedy decode): **94.9%** draft acceptance, decode rounds down ~16× (903
  single-token forwards → 57 forward/verify calls), but wall-clock tg
  improved only **~7%** (59.1 → 63.3 t/s) — `Attention` JFR count/time
  roughly halved (the real saving, from batching multiple query positions
  into one attention dispatch per round) while `MatVec` time was flat to
  slightly higher (a batched-window GEMM over several rows costs more per
  call than a single-row GEMV). Honestly reported, not overclaimed — see
  `docs/performance.md` / `docs/perf-compare/README.md`.
- JFR: new `juno.Speculation` event (`draftTokens`/`acceptedTokens`),
  aggregated by `JfrMetricsExtractor` into `juno.Speculation.{count,
  draftTokens.sum, acceptedTokens.sum, acceptanceRate}`.
- §2 regression: `compare-llama-cpp.sh --gpu --models tinyllama` (default
  `--spec-type none`), failures=0; `compare-lora.sh --gpu --baseline
  release-0.1.2`, flat as expected (LoRA never routes through
  `forwardVerify`).
- Vision, ROCm, and the remaining handler families' `forwardVerify` overrides
  are named follow-ups, not claimed as working.

**Session 82** — Embeddings API (`POST /v1/embeddings`) **feature complete**

- OpenAI wire-compatible `POST /v1/embeddings`, opt-in via `--embeddings` (off
  by default — a server started without it returns HTTP 400
  `embeddings_disabled` rather than 404 or a silent no-op). `--pooling
  mean|cls|last` (default `mean`) reduces the RMS/LayerNorm-normalized hidden
  state at every prompt position to one vector via the new
  `EmbeddingPooling`/`PoolingMode` (node module). Batch string `input`,
  deterministic output, OpenAI-shaped `usage`/`data` response.
- `InferencePipeline.embedTokens` is a new default method that throws
  `UnsupportedOperationException` — a fail-closed contract only
  `LocalInferencePipeline` overrides. Distributed pipelines (gRPC node
  clients, tensor-/pipeline-parallel `cluster` launchers) inherit the
  default and return HTTP 400 `embeddings_unsupported` on every request
  instead of a 500 or a silently wrong vector, matching `--schedule
  continuous`'s existing local/single-shard-only scope.
  `LocalInferencePipeline.embedTokens` generalizes the pre-existing
  `embedLastToken` loop to keep every position's hidden vector instead of
  discarding all but the last (no extra compute — that work was already
  happening); both now call `evict(requestId)` before returning, fixing a
  pre-existing per-request KV-cache leak that had gone unnoticed because
  `embedLastToken`'s only prior caller (`JunoPlayer.embed`) is rarely used at
  volume.
- `EmbeddingsHandler` (coordinator module) runs directly on the Javalin
  request's own thread, bypassing `RequestScheduler`/`GenerationLoop`'s
  sampler-driven decode loop and its queue-depth/429 semantics entirely — a
  deliberate v1 scope choice, documented rather than silently assumed.
- Live-smoked against a real TinyLlama GGUF (not just unit-test doubles):
  correct hiddenDim-length vectors, batch input, determinism, invalid-input
  400s, chat completions unaffected, and a `--lora-play` combination (the
  embedding path runs through the LoRA-wrapped handler chain, so embeddings
  reflect the loaded adapter rather than bypassing it). LoRA train mode and
  `cluster` mode both print a startup warning when `--embeddings` is passed
  but cannot honor it, rather than ignoring the flag silently.
- §2 regression: `compare-llama-cpp.sh --models tinyllama --cpu --vector 0
  --no-jfr`, failures=0 — a regression spot-check per the API-only-tier
  carve-out (no MatVec/forward/KV/vision path touched); not a published
  multi-model bake-off.
- Vision + `--embeddings` combination is an unverified follow-up (documented,
  not claimed as working).

**Session 81** — GGUF chat template + Hugging Face Hub download **feature complete**

- Prefer a GGUF's own embedded `tokenizer.chat_template` metadata, rendered
  through a restricted Jinja subset (`MiniJinjaTemplate`), over the named
  template registry; falls back automatically when the metadata is absent,
  unparseable, or fails a smoke render. `GgufChatTemplateResolverTest` — 9
  tests, including real-GGUF spot checks (Phi-3.5, TinyLlama).
- Manual multi-turn testing against the real TinyLlama GGUF surfaced a
  whitespace bug the automated substring/ordering checks missed: block tags
  (`{% %}`) without explicit `{%-`/`-%}` dashes left stray blank lines that
  compounded every turn, degrading generation by turn 3-4. Fixed by
  implementing Jinja2's own `trim_blocks`/`lstrip_blocks` **defaults** (the
  behavior HF's `apply_chat_template` and llama.cpp's renderer both assume) —
  most real-world templates rely on this default rather than writing dashes
  themselves. New exact-match regression test
  (`default_trim_blocks_and_lstrip_blocks_apply_without_explicit_dashes`)
  locks in the fix.
- `--hf repo[:quant]` resolves and downloads a GGUF from the Hugging Face
  Hub into the effective model path (`Q4_K_M` preferred quant, resume +
  ETag cache under `~/.cache/juno/models`, no Python dependency); fails
  closed when `--hf` and `--model-path` disagree. `HfGgufFetcherTest` — 16
  tests against a local mock HTTP server.
- A live run against the real Hub (`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`,
  668 MB) surfaced two bugs the mock-only tests missed: the fetcher's
  `HttpClient` didn't follow redirects, so every real download failed
  against the Hub's CDN-redirecting `/resolve/main/...` endpoint; and a
  nonexistent or private repo surfaced as a bare "HTTP 401" instead of a
  clear "not found" message (the Hub API conflates the two cases
  deliberately, to avoid disclosing private-repo existence). Both fixed,
  with regression tests (redirect-following, 401-message) added.
- LoRA train / `--lora-play` keep the named template only (explicit no-op,
  startup notice) so train-time and inference-time formatting stay
  identical for adapter recall; vision + embedded template is an unverified
  follow-up.
- §2 CPU regression spot-check `target/perf-compare/20260914T181535Z/`
  (`--cpu --vector 0 --models tinyllama`, failures=0) — not a published
  multi-model bake-off; this tier touches no MatVec/forward/KV path.

---

## Status 

**Session 80** — Function calling / tools **feature complete**

- OpenAI `tools` / `tool_choice` on chat completions: prompt inject for
  Llama 3 / ChatML / Qwen3; parse `<tool_call>` into `message.tool_calls`.
  `tool_choice=none` never emits tools. `required` / named choice uses GBNF.
  Unsupported templates and grammar conflicts fail closed.
- Bake-off [`docs/perf-compare/20260913T032734Z/`](docs/perf-compare/20260913T032734Z/)
  (`--cpu --vector 0`, failures=0). Cross-feature smoke
  `target/tools-smoke/20260913T025903Z/` (required `tool_calls`, template
  fail-closed, `--parallel 2`).

---

## Status 

**Session 79** — Constrained decoding (GBNF + JSON Schema) **feature complete**

- Sampler GBNF engine + JSON Schema subset compiler; grammar masks logits
  before temperature / top-k / top-p. `GrammarEvalTest`: 20 schemas, ≥95%
  parseable JSON under an adversarial logit prior.
- OpenAI `response_format` `json_object` / `json_schema`; `x_juno_grammar`;
  CLI `--grammar-file` / `--json-schema-file` (local + cluster). LoRA train
  is an explicit no-op. Unsupported schema keywords fail closed.
- Bake-off [`docs/perf-compare/20260912T193402Z/`](docs/perf-compare/20260912T193402Z/)
  (`--cpu --vector 0`, failures=0). Cross-feature smoke
  `target/grammar-smoke/20260912T193149Z/` (JFR `juno.GrammarConstrained`).

---

## Status 

**Session 78** — Packed-Q4 GPU GEMV (Q8_1 + integer-dot) speed follow-on

- Device path quantizes the activation to Q8_1 once, then integer-dots packed
  Q4_K / Q5_K / Q6_K weights (`quantize_q8_1` + fused GEMV). Default `--mmq` stays **off**.
- Bake-off [`docs/perf-compare/20260911T235203Z/`](docs/perf-compare/20260911T235203Z/):
  Phi-3.5 **19.3** tg (**0.33×** peer — program 0.5× still unmet);
  Mistral-7B on 8 GiB `--gpu-layers auto` **15.3** tg (**0.43×** peer — 0.15× **met**).
  JFR `cuda_resident_q4k` on all four models (cpu.count=0).
- Paired [`20260911T235353Z`](docs/perf-compare/20260911T235353Z/): Phi-3.5 `--mmq on`
  is **1.51×** `--mmq off` (12.8 tg). Kernel p95 ≈ **0.32 ms** (was ≈ 2.3 ms).
- LoRA [`20260911T235455Z-lora`](docs/perf-compare/20260911T235455Z-lora/) **ok**
  (train 1.00×, playback 0.88× vs release-0.1.2).
- Docs may state a measured decode win vs `--mmq off` on CUDA — not peer latency.

---

## Status 

**Session 77** — OpenAI field parity (`stop` / `seed` / `presence_penalty`) **feature complete**

- `SamplingParams` gains `presencePenalty`, `stopStrings`, `seed`; `PresencePenaltyStep`
  in the sampler pipeline; seeded `Random` on decode loops.
- `StopSequenceFilter` holds back / truncates OpenAI stop sequences; single-token
  stop strings merge into stop token ids via the tokenizer.
- `response_format` fail-closed (HTTP 400) unless absent or `type=text`.
- Docs / OpenAPI updated; `logit_bias` and `user` remain ignored with honesty.
- §2 CPU regression [`docs/perf-compare/20260911T221215Z/`](docs/perf-compare/20260911T221215Z/)
  failures=0 (GPU driver unavailable this run). Smoke
  [`target/tier2-smoke/20260911T223000Z/`](target/tier2-smoke/20260911T223000Z/):
  seed match, stop truncates, `presence_penalty` 200, `response_format` 400.
- Next Infra: constrained decoding / grammar (Tier 3 plan).

---

## Status 

**Session 76** — Mixed chunked prefill under continuous schedule **feature complete**

- Bake-off [`docs/perf-compare/20260911T204721Z-mixed-prefill/`](docs/perf-compare/20260911T204721Z-mixed-prefill/):
  short mean TTFT **0.327×** admit-time baseline (4552 vs 13929 ms); JFR
  `ContinuousStep.prefill_chunks=12` proof **PASS**. Short TPOT rises under mix
  (shared steps) — documented tradeoff. Short-decode TTFT bound ≤ **5702 ms**
  (1.25× measured max) on this recipe.
- Admit no longer blocks on full prompt eval; `ContinuousPrefillState` +
  `ContinuousMixedStepPolicy` (decode preferred when step slots full).
- §2 [`20260911T204900Z`](docs/perf-compare/20260911T204900Z/) failures=0;
  LoRA [`20260911T205447Z-lora`](docs/perf-compare/20260911T205447Z-lora/) **ok**
  (train 1.00×, playback 0.86×). Smoke [`target/tier16-smoke/20260911T205500Z/`](target/tier16-smoke/20260911T205500Z/).

---

## Status 

**Session 75** — Continuous batching scheduler (`--schedule continuous`) **feature complete**

- Bake-off [`docs/perf-compare/20260911T194430Z-continuous/`](docs/perf-compare/20260911T194430Z-continuous/):
  multi-session TPS continuous **0.86×** static (synchronized 8-way); concurrent SSE
  `juno.ContinuousStep` shared-step proof **PASS** (max_decode_batch=8); prefix session
  hit rate **0.875**. P1 “SSE beats static” gate **unmet** on this synchronized recipe.
- §2 regression [`20260911T195008Z`](docs/perf-compare/20260911T195008Z/) + LoRA
  [`20260911T195711Z-lora`](docs/perf-compare/20260911T195711Z-lora/) **ok** (train 0.98×,
  wall play 0.88×). Health exposes prefix counters; JFR extracts ContinuousStep.
- `compare-schedule.sh`; §6 smoke [`target/tier15-smoke/20260911T200500Z/`](target/tier15-smoke/20260911T200500Z/).
  Next Infra: mixed chunked prefill ([`PLAN-Infra-Tier16.md`](docs/infra-plan/PLAN-Infra-Tier16.md)).

---

## Status 

**Session 74** — Continuous batching scheduler (`--schedule continuous`) **in progress**

- Local / in-process running-set engine: overlapping decode shares `forwardBatch`;
  SSE and publisher streams join the running set (static schedule still isolates SSE).
- Cluster / TP / PP **auto-fallback to static** (`WARN: continuous unsupported on cluster; using static`)
  before handlers load so KV path stays dense.
- Prefix cache: lookup/hit counters; session skip; trie kept after stateless complete.
- Per-request `x_juno_loras` fail-closed under continuous. Bake-off / JFR TTFT still open.

---

## Status 

**Session 73** — Block KV allocator (`--kv-page-size` / dual path) **feature complete**

- Start P1 step 2: `KvBlockPool` + `KvPageTable` with unit tests (allocate/free, gather, concurrent).
- Manager plumbing: `PagedKvTensor` / `PagedKvArena` / `PagedKvCodec`; `KVCacheManager`
  optional arena; `NodeKVCacheAdapter` flush/restore for paged + q8_0 payloads.
- Dual-path handlers: `SessionKvTensor` / `SessionKvLayout`; dense under `static`,
  paged gather under `continuous`; LoRA train stays ephemeral float (warn); Llama
  schedule parity test.
- Gather-tax microbench **PASS** (4.61% at batch 8 / ctx 8k after F16 page-bulk gather);
  `MAX_SEQ_LEN` → 32768; report [`20260910T214300Z-gather-tax.md`](docs/perf-compare/20260910T214300Z-gather-tax.md).
- CLI: `--schedule` / `--kv-page-size` in ConsoleMain + `run.sh` / `run.bat`; cluster `-D` forward;
  howto/README; smoke [`target/tier14-smoke/20260910T220100Z/`](target/tier14-smoke/20260910T220100Z/).
- §2 bake-off [`20260910T222026Z`](docs/perf-compare/20260910T222026Z/) failures=0;
  LoRA [`20260910T221031Z-lora`](docs/perf-compare/20260910T221031Z-lora/) **ok** (wall play 0.88×).
  Next Infra: Tier 15 continuous scheduler.

---

## Status 

**Session 72** — Quantized KV cache (`--cache-type-k/v`) **feature complete**

- Add `Q8_0KvCodec`, `CacheTypeOptions`, `DenseKvTensor`; CLI `--cache-type-k|v f16|q8_0`
  (default `f16` = current float32 path). Wire inference KV in all text + LoRA play handlers.
- Adapter flushes typed payloads; unit + Llama logit parity tests green.
- Cluster: launchers forward flags; `ClusterHarness` passes `JUNO_CACHE_TYPE_*` to forked nodes.
- Bake-off [`20260910T170557Z`](docs/perf-compare/20260910T170557Z/); LoRA [`20260910T180703Z-lora`](docs/perf-compare/20260910T180703Z-lora/) **ok**
  (wall playback 0.88×). `compare-lora.sh` gates on wall tps; JFR `tps_jfr` informational.

---

## Status 

**Session 71** — Tier 13B VRAM-fit ship + shared-activation Phase 1

- Amend Phase B exit: `--mmq` ships as **VRAM-fit** (default off); ≥1.3× tg vs FP16-resident
  deferred to a tile Q4 kernel follow-on (bake-off showed MMQ slower when both fit).
- Add `MatVec.sgemvSameX` (CUDA/ROCm): one activation H2D + coalesced sync for Q/K/V and
  gate/up; wire into `LlamaTransformerHandler` decode; parity `SgemvSameXParityTest`.
- Docs: `PLAN-Infra-Tier13.md`, ROADMAP, `performance.md`, `howto.md`.

---

## Status 

**Session 70** — LoRA playback fused Q4_K MMQ (Phase 1 complete)

- Wire `--mmq` into `--lora-play` via `LoraMmqPolicy` + `ResidentQ4KWeight`
  (playback-only; train stays FP16/FP32 and warns once when `--mmq` is preferred).
- `ConsoleMain` sets `juno.lora.play.path` for in-process local play (cluster already did)
  and prints a REPL line when playback MMQ is active (JUL is off without `--verbose`).
- Play OOM uses `LoraResidentUpload.runPlayback` (close Q4 → CPU; no train FP16 microbatch retry).
- Unit: `LoraMmqPolicyTest`, `LoraResidentUploadTest` playback OOM, `LoraResidentWeightsTest` Q4 close/CPU matVec.
- GPU: `LoraQ4KPlaybackParityTest` (packed GEMV then LoRA delta vs CPU, `1e-2`; close / double-close).
- Smoke (GTX 1080): play JFR `cuda_resident_q4k.count=4020` + recall; train ignore-mmq + finite loss;
  `compare-lora.sh --gpu --baseline release-0.1.2` ok (train 0.96×, play tps 0.89×) → `20260905T031520Z-lora`.

---

## Status 

**Session 69** — Fused Q4_K GPU matmul prototype (`--mmq`)

- Add `--mmq on|off|auto` / `JUNO_MMQ` (default off): keep Q4_K weights packed
  on device and run a PTX fused dequant+GEMV instead of FP16-resident cuBLAS.
- New: `MmqOptions`, `CudaDriverBindings`, `Q4KMmqKernel`, `DeviceQ4KMatrix`,
  classpath `q4k_gemv.ptx` (sm_60+), JFR backend `cuda-resident-q4k`.
- Wire into `LlamaTransformerHandler` for Q4_K projections (non-Q4_K stays FP16).
- Parity: `Q4KMmqParityTest` (GPU). Bake-off vs FP16 path still open.

---

## Status 

**Session 68** — Vector API SIMD for CPU quantized matmul kernels, plus tunable row-parallel pool

- Vectorize dot-product accumulation in sgemmQ4KWeightStationary,
  sgemmQ5KWeightStationary, sgemmQ8_0WeightStationary via new
  VectorQuantKernels (jdk.incubator.vector), with scalar fallback if the
  module is unavailable.
- Vectorize Q8_0 dequantization, gated by a runtime self-test against the
  full signed-byte range.
- Add SimdThreadPool: dedicated ForkJoinPool for the same kernels' row-parallel
  loop, size tunable via -Djuno.simd.pool.size (default unchanged).
- Add JUNO_JVM_OPTS passthrough in scripts/run.sh, scripts/run.bat, and
  ClusterHarness's forked node JVMs, for ad hoc JVM flag overrides.
- Add --add-modules jdk.incubator.vector to compiler/surefire/run scripts/
  AWS deploy script/ClusterHarness.
- Add startup diagnostics (SIMD width, pool parallelism) in ConsoleMain.
- Unit tests: VectorQuantKernelsTest, SimdThreadPoolTest.
- Measured ~30% prefill reduction on moondream2-q5_k CPU inference; gap to
  llama.cpp remains, mainly Q4_K/Q5_K dequant still scalar and activations
  still float32 (llama.cpp int8-quantizes activations, see docs/agent-arch.txt).

---

## Status

**Session 67** — locked down the Session 66 SIMD benchmark against the raw
JFR logs, phase by phase, to separate what actually moved from what's just
run-to-run noise.

Cross-checked every phase sum against both runs' JFR output directly (not
estimates): `qkvProj` 66,313.7ms → 57,426.5ms (-13.4%), `woProj+ffn+residuals`
106,183.5ms → 62,412.8ms (-41.2%), total prefill 178,222.6ms → 124,690.3ms
(-30.0%). `rope+cacheWrite` and `attention` also moved (-8.3%/-17.9%) despite
neither being touched by `VectorQuantKernels` — recorded as the scheduling
noise floor on this box, not a SIMD effect.

Confirmed decode is genuinely unaffected: `matVec` p95 4.488ms vs 4.467ms,
within noise, as expected since decode uses the single-token path, not
`sgemm*WeightStationary`. Flagged that the two runs' raw tok/s and token
counts (22 vs 47 tokens, temperature=0.3) are not a valid before/after
comparison — only the fixed 741-token prefill window is apples-to-apples.

**Conclusion for the record:** a real, measured 30% prefill reduction,
isolated to the two call sites that route through `VectorQuantKernels.dot`;
output stayed coherent both runs; still ~3.4x off the llama.cpp ~30-36s
reference on the same hardware, because only the dot-product accumulate loop
is vectorized so far, not the Q5_K/Q4_K/Q8_0 dequantization bit-unpacking.
That remains the next lever, same as flagged in Session 64/63.

---

## Status

**Session 66** — first real hardware run of the Session 65 SIMD accumulate
kernel against the fixed 741-token vision-chat benchmark from Session 64's
handoff doc.

Total prefill 178.2s → 124.7s (-30%). Confirmed from the log that
`VectorQuantKernels.AVAILABLE` was true (no "jdk.incubator.vector
unavailable" fallback message), so the SIMD path actually ran, not the
scalar fallback. `finish_reason: stop`, coherent output, no numerical
garbage.

Per-phase: `qkvProj` -13.7%, `woProj+ffn+residuals` -41.2%, attention/rope
roughly flat (both untouched code paths — not evidence of anything). The
much bigger win on `woProj+ffn+residuals` is tentatively attributed to
`wUp`/`wDown` being ~4x larger than the QKV projections, giving more
parallel accumulate work per call to amortize thread-dispatch/dequant
overhead against — a plausible read from the shapes, not confirmed by
profiling this run.

Takeaway: this is the win from the lower-risk half of the SIMD work only
(dot-product accumulate). The Q5_K bit-unpacking dequant phase is still
scalar and is now the limiting factor, consistent with landing at ~30%
instead of the 20-40x per-core gap the original FLOP analysis predicted for
a full rewrite. The old unexplained per-layer timing variance (Session 64)
is still present, unchanged by this session — confirmed independent of the
SIMD work, not fixed or worsened.

---

## Status

**Session 65** — implemented the Vector API SIMD kernel spec'd out at the end
of Session 64, plus all the build/run plumbing it needs.

Resolved the open module question first rather than assuming: confirmed JDK
25 (JEP 508) and JDK 26 (JEP 529, current as of Aug 2026) both still ship the
Vector API as `jdk.incubator.vector`, not finalized — so `--add-modules
jdk.incubator.vector` is required at both compile and run time.

**New:** `VectorQuantKernels` (`node` module) — vectorizes only the
dot-product accumulation phase of the Q4_K/Q5_K/Q8_0 weight-stationary
kernels, deliberately leaving the bit-unpacking dequant phase scalar as a
documented follow-up rather than risk an unverified SIMD byte/float shape
conversion in correctness-critical code. All `jdk.incubator.vector`
references are confined to a nested `Simd` class, probed once at class-init
inside `catch (Throwable)`, so a JVM missing the module falls back to the
original scalar loop transparently instead of crashing.

**Modified:** `sgemmQ4KWeightStationary` / `sgemmQ5KWeightStationary` /
`sgemmQ8_0WeightStationary` in `LlamaTransformerHandler` now call
`VectorQuantKernels.dot(...)` in their inner accumulate loop; method
signatures and existing tests untouched. New `VectorQuantKernelsTest`
covering block sizes 32/256, non-lane-aligned tails, offsets, zero-length.

**Build/run plumbing:** root `pom.xml` (compiler args), `node/pom.xml`
(surefire `argLine`), `scripts/run.sh` / `run.bat`, `ClusterHarness.java`
(forked node JVMs), and all three production `java` launch points in
`scripts/aws/juno-deploy.sh` all get `--add-modules jdk.incubator.vector`.
`docs/agent-arch.txt` updated to document the new class.

Could not run `mvn compile`/`mvn test` in this environment — validated
read-only instead: XML well-formedness on both POMs, `bash -n` on the shell
scripts, and manual review of the Vector API call sites
(`FloatVector.fromArray`, `.fma`, `.reduceLanes`, `SPECIES.loopBound`)
against the JDK 25 API shape. **Not yet benchmarked on real hardware** — that
was Session 66.

---

## Status

**Session 64** — closed every architectural/scheduling gap between Juno's
vision-chat prefill and llama.cpp's on the same laptop/model/quant, down to
one remaining, well-quantified problem: the CPU dequant/matmul kernels are
scalar Java, not SIMD. Benchmark: `POST /v1/vision/chat`, fixed 741-token
window (729 image patches + 11 text), `moondream2-q5_k.llamafile`, `./juno
local`, Intel i5-1240P, GPU off both sides. llama.cpp reference on the same
box: ~30s to first token, ~36s total.

**Fixed this session:**
- Merge-conflict compile fixes from the Vision-I2T → release-0.1.2 merge
  (`LoraTrainableHandler` sgemm type mismatch, stale `vision/pom.xml` parent
  version, a stray `run.sh` merge artifact, a dropped `eosFilter.text()`
  call, a deleted-but-still-called `trainOnMasked`).
- **Vision encoder batching** (`VisionEncoder`): every linear layer went
  from one `sgemv` call per patch (~118K calls/image) to one batched `sgemm`
  call per layer. Vision encode dropped to ~33-53s and stayed there — no
  longer the bottleneck.
- **Phi2 batched prefill**: Phi2 had no `forwardBatch` override at all, so
  it silently fell back to 740 sequential single-token `forward()` calls.
  Added real batched prefill mirroring Llama/Phi3's existing pattern.
- **New Q5_K weight-stationary CPU kernel** (`sgemmQ5KWeightStationary`):
  even Llama's own "batched" dispatch only had true batched kernels for
  Q4_K/Q8_0 — Q5_K (this model's actual quant) fell through to a sequential
  per-row loop. Unit-tested against the existing per-row `matVec` oracle.
- **KV cache eviction leak**: `evict(requestId)` existed on every
  transformer handler but was unreachable from the pipeline layer, so every
  stateless request leaked its full per-layer KV arrays — caused an OOM on
  a second request. Fixed by adding the interface methods with
  correctness-preserving no-op defaults, cascaded through
  `LocalInferencePipeline`, `VisionAwareForwardPassHandler`,
  `FaultTolerantPipeline`, `GenerationLoop`. Known remaining gap:
  `ProcessPipelineClient`/`TensorParallelPipelineClient` (forked-JVM cluster
  mode) still no-op on `evict()`.
- **`ShardMap.evenSplit`**: `./juno local --nodes N` routed through
  `ShardPlanner`'s VRAM-aware greedy algorithm with a fabricated per-node
  VRAM figure, so node 0 always got almost the entire model (22/24 layers).
  Added an honest even-split method for local simulation — verified via
  per-node timing: 150s/8s/8s skew → even ~69s/67s/70s.
- **Attention parallelization** (`Phi2TransformerHandler`): the causal
  self-attention loop over 740 window positions was single-threaded.
  Parallelized with `IntStream.range(0, W).parallel()` since each `gqa()`
  call is independent — confirmed by measurement to drop attention from an
  estimated ~20-40s liability to 4.4s / 2.4% of prefill. **Known sibling
  gap, deliberately unfixed:** `LlamaTransformerHandler`/
  `Phi3TransformerHandler` have the same sequential loop, but their
  `gqaInto()` writes into a shared `ws.scores` scratch buffer — naive
  parallelization there would race. Needs its own fix before benchmarking
  non-Phi2 models locally.
- Per-layer timing instrumentation added to Phi2 (`qkvProj` /
  `rope+cacheWrite` / `attention` / `woProj+ffn+residuals`), mirroring what
  already existed for Llama — this is what produced the phase breakdown
  below.

**Measured breakdown** (24-layer sum, `elapsedMs=178,222.6`): `qkvProj`
66.3s (37.2%), `rope+cacheWrite` 1.2s (0.7%), `attention` 4.4s (2.4%),
`woProj+ffn+residuals` 106.2s (59.6%). **96.8% of remaining prefill time is
inside the batched Q5_K dequant/dot-product kernel.** FLOP sanity check on
just QKV: ~447 GFLOP in 66.3s ⇒ ~0.5 GFLOP/s per thread, squarely scalar-Java
territory against llama.cpp's hand-vectorized 10-20+ GFLOP/s per core — the
gap fully explains the observed ~6x wall-clock difference.

Confirmed as *not* a bug: thread pool sizing (`ForkJoinPool.commonPool()` is
correct for local mode; the explicit parallelism flag is only for cluster
mode's forked JVMs).

**Next task, spec'd out for Session 65:** replace the scalar inner loops in
`sgemmQ{4K,5K,8_0}WeightStationary` with explicit Vector API SIMD, inside the
existing weight-stationary batching structure (don't replace it). Priority
order: Q8_0 first (simplest block layout, lowest risk, validates the JDK
module/build plumbing), then Q4_K, then Q5_K; `CpuMatVec`/vision path and
other scalar quant fallbacks are lower priority. First open question to
resolve before writing code: confirm the real target JDK's Vector API status
(`jdk.incubator.vector` vs finalized) — this session did not check it.

---

## Status

**Session 63** — vision encode accuracy fix on branch `47-vision`: SigLIP
vision towers (moondream2) were missing their post-encoder LayerNorm
entirely.

Commit `a5255d3`, titled "fixed some image scaling errors, something is
replying correctly now" — despite the commit message, the actual bug was
numerical, not geometric. `VisionEncoder` never applied `v.post_ln`.
CLIP/LLaVA mmproj files don't declare this tensor at all (CLIP's own
`post_layernorm` only touches the pooled CLS output, which LLaVA-style
callers never use), so its absence from the encoder was easy to miss — but
SigLIP (moondream2's vision tower) applies this LayerNorm to the *entire*
last hidden state before any downstream use, making it structurally
required there, not optional.

Without it, moondream2's patch embeddings were the raw, un-normalized final
transformer-block residual stream, with L2 norm up to ~70000, instead of
properly LayerNorm'd features — a magnitude blowup fully consistent with
the "sees the image but describes something nonsensical" failure mode this
branch had been chasing.

**Fix:** reads `v.post_ln.weight`/`v.post_ln.bias` when present
(`hasPostLn`) and applies the LayerNorm once after the last transformer
block, before the projector. Absence means skip the operation entirely —
deliberately *not* an identity-affine LayerNorm, which would still
normalize mean/variance and change CLIP/LLaVA's existing (correct)
behavior. CLIP/LLaVA mmproj files therefore see zero behavior change from
this fix. New `VisionEncoderTest` cases cover both the present and absent
tensor case.

---

## Status

**Session 62** — `47: Phi2Rope, image placeholder and more formatting for
phi2`: Phi2's own RoPE variant, plus vision-side formatting cleanup, ahead
of the first coherent moondream2 reply landed in Session 63.

**New `Phi2Rope`** (`node` module): Phi2 uses a **partial-rotary** RoPE
variant — only a configured fraction of `headDim` is actually rotated, the
remainder passed through unchanged — different from the full-rotary RoPE
`LlamaTransformerHandler`/`Phi3TransformerHandler` already implement, so it
couldn't reuse either existing implementation. New `Phi2RopeTest` pins down
the exact partial-rotation boundary.

`GgufReader` gained float-array metadata reading (new
`GgufReaderMetaFloatArrayTest`) so Phi2's rotary-fraction config comes
straight from GGUF metadata rather than being hardcoded.

`ChatTemplate` extended for Phi2's chat format; `VisionConfig`,
`ImagePatchEmbedder`, `LlavaHandlerFactory`, and
`VisionAwareForwardPassHandler` all received formatting/wiring adjustments
so the `<image>` placeholder token is handled consistently on the Phi2
path, not just Llama/Phi3. New `VisionConfigNormalizationTest`.

---

## Status

**Session 61** — `adding mm0OutDim to projection math to support phi2
vision`: extended `VisionEncoder`'s CLIP-only assumptions to also cover
SigLIP-family towers (moondream2), ahead of onboarding Phi2 vision proper.

`VisionEncoder`'s tensor-naming contract (javadoc and load-time
expectations) was CLIP/LLaVA-specific: `v.class_embd` always present,
`v.position_embd.weight` sized `numPatches+1` (CLS token included).
SigLIP models have no CLS token at all — `v.class_embd` is absent and
`v.position_embd.weight` is sized exactly `numPatches`. The encoder now
branches on tensor presence rather than assuming CLIP's layout
unconditionally.

`mm0OutDim` (the first projector layer's output width) is now read from
`mm.0.weight`'s own GGUF shape rather than assumed — consistent with this
branch's established policy (first applied to the projector's *second*
layer back in the mm.2-projector work) of trusting the tensor's own shape
over metadata fields that have proven unreliable across mmproj exports.

Two large reference documents were added under `docs/`
(`meta-juno-doc.md`/`.txt`) capturing the full architecture write-up this
branch had been accumulating — bulk documentation, no source-behavior
change.

---

## Status

**Session 60** — `.llamafile as vision archive`: models shipped as
`.llamafile` (Mozilla's self-contained model+runtime bundle) can now be
read directly as a vision model source, not just plain `.gguf`.

**New `LlamafileGgufIndex`** (`node` module): a `.llamafile` is a
self-executing archive with a GGUF payload appended after a Cosmopolitan
binary stub. This parses the container to locate and index the embedded
GGUF's tensors and metadata without needing a separately-extracted `.gguf`
file on disk. New `LlamafileGgufIndexTest` covers the container-parsing
logic directly.

`GgufReader` extended to open through this index transparently, so
downstream code doesn't need to know whether it's reading a bare GGUF or
one embedded in a llamafile. `LlavaHandlerFactory` updated so vision model
resolution accepts a `.llamafile` path wherever a `.gguf` path was
previously required. New `LlavaHandlerFactoryEmbeddedVisionTest` covers
loading vision tensors out of an embedded, llamafile-packaged GGUF
end-to-end.

This is the change that made `moondream2-q5_k.llamafile` — the model used
for every prefill benchmark in Sessions 64-67 — usable at all.

---

## Status

**Session 59** — `gguf-info mode and phi2 support added`: a standalone
GGUF-inspection CLI mode and the first Phi2 architecture support, added
together since getting Phi2 vision working at all needed the inspector
first.

**New `./juno gguf-info` subcommand** (`GgufInfoMain`, `juno-player`):
dumps a GGUF file's metadata keys and tensor list/shapes without loading a
full model or starting inference. Used throughout the rest of this branch
to inspect unfamiliar mmproj/model files before writing code against them
— this is the same tool Session 60 (main narrative numbering aside) later
used to catch the CLIP `use_gelu` metadata bug.

**New `Phi2TransformerHandler`** (`node` module, ~560 lines): first-cut
Phi2 architecture support — distinct attention, FFN, and norm wiring from
Llama/Phi3 — wired into `ForwardPassHandlerLoader`.

`ImagePatchEmbedder` and `VisionEncoder` extended for the patch-embedding
and encoder-config shapes this model family needs; `VisionAwareForwardPassHandler`
updated accordingly. Expanded `ImagePatchEmbedderTest` and
`VisionEncoderTest`.

---

## Status

**Session 58** — `vision replies random scene, missing context of pic`: the
zero-vector text-token bug. Image tokens carried real signal into the
model; every text token carried none.

`VisionAwareForwardPassHandler.buildWindowActivationsWithVision()` (and the
single-token `buildActivationWithVision()`) spliced real CLIP/SigLIP patch
vectors into image-token positions but left every **text**-token position
as an all-zero vector — meaning the entire prompt text (chat template, the
actual question, BOS) was invisible to the model; only the image patches
carried any signal at all. This explains the failure mode precisely:
grammatically-plausible output describing a plausible but unrelated scene,
since the only "real" input reaching the model was the image.

**Fix:** added `ForwardPassHandler.embedToken(int)` (default: throws
`UnsupportedOperationException`) so a decorator can ask the wrapped handler
for a single token's real embedding-table row. Implemented in
`LlamaTransformerHandler.embedToken()`, reusing the existing clamping
logic. Both vision-splicing methods now call
`textHandler.embedToken(tokenId)` for non-image positions instead of
leaving zero.

Updated the two existing `VisionAwareForwardPassHandlerBatchTest` cases
that had asserted the old, buggy zero-vector behavior (they'd now be
actively wrong), added a new single-token-path regression test, and added
direct `embedToken()` tests to
`LlamaTransformerHandlerEmbeddingsNodeActivationsTest`. The shared
`StubForwardPassHandler` test double gained a configurable deterministic
fake embedding.

---

## Status

**Session 57** — `vision finally working, but long and gives gbg reply`,
immediately followed same day by cleanup of an accidentally-committed debug
dump.

First commit where a `/v1/vision/chat` request ran end-to-end without
crashing or hanging and produced *some* reply text — output was still
long-winded and largely garbage at this point (this is the exact state
Session 58's zero-vector bug describes and fixes next). Touched
`GenerationLoop`, `ConsoleMain`, `LlamaTransformerHandler`; new
`LlamaTransformerHandlerEmbeddingsNodeActivationsTest`.

The commit accidentally included an 8700-line `diff.txt` — a leftover
debug artifact, not source — caught and deleted in the very next commit the
same day. No functional change in the cleanup itself; noted here only so
the file's brief appearance in history isn't mistaken for intentional
content later.

---

## Status

**Session 56** — `up`: internal refactor of the batched-prefill code paths
added in Session 55. No externally-visible behavior change.

`LlamaTransformerHandler` and `Phi3TransformerHandler`'s batched-prefill
methods (added in Session 55) were reworked for correctness and clarity
following self-review; `docs/batched-prefil.md` (the design doc from
Session 55) was extended with the implementation notes this pass produced.
No new public API surface; existing batched-prefill tests continued to
apply unchanged.

---

## Status

**Session 55** — `removed bytedeco from the docs and code to not confuse
assistants`, immediately followed by `batched prefill`: the batched-prefill
work that the later SIMD benchmark sessions (64-67) build directly on top
of was designed and implemented here.

Small cleanup first: stray `org.bytedeco` (JavaCPP) references removed
from docs and a couple of `node`/`master` test files. The project had
already moved off JavaCPP for its CUDA bindings well before this branch
started, but leftover mentions were confusing enough to warrant a
dedicated pass.

**Batched prefill**, planned in a new `docs/batched-prefil.md` design
document before any source change, then implemented: `GenerationLoop.generate()`
and `.generateBatch()` previously prefilled a prompt with a sequential
per-position loop, reallocating and copying a growing token-id slice on
every single position. New `PrefillMode` (`SINGLE`/`BATCH`), new
`BatchForwardRequest`/`BatchForwardResult` (`node` module), and real
batched `forwardBatch()` implementations added to
`LlamaTransformerHandler` and `Phi3TransformerHandler`. `CpuMatVec` gained
the batched `sgemm` this all runs on top of. New batched-aware
`LoraTrainableHandler` path. New tests: `PrefillModeTest`,
`ConsoleMainPrefillFlagTest`, `CpuMatVecSgemmTest`,
`VisionAwareForwardPassHandlerBatchTest`.

A `docs/TODO-VectorAPI.md` note was also added here, flagging future SIMD
work as a known follow-up — the same gap Sessions 65-67 eventually closed.

---

## Status

**Session 54** — `#47 model resolver, f16 weights support, stack-trace on
api error, tested on llava-phi-3-mini-f16.gguf llava-v1.5-7b-Q4_K.gguf`:
first real-model validation pass against two actual downloaded LLaVA GGUFs.

**New `ModelIdResolver`** (`registry` module) so `/v1/vision/chat` and the
OpenAI-compatible endpoints can resolve a requested model id against
what's actually loaded, rather than requiring an exact string match.

`LlamaTransformerHandler` gained F16 weight-tensor support (new
`LlamaTransformerHandlerF16MatVecTest`), needed once real F16 mmproj/model
files were tested against, not just quantized ones.

`InferenceApiServer`/`OpenAiChatHandler`/`VisionChatHandler` error paths
now surface a full stack trace on API error instead of swallowing it — this
is what made every subsequent bug in this branch (Sessions 57-63)
traceable to a specific line rather than a bare exception name.

First real-model test pass: `llava-phi-3-mini-f16.gguf` and
`llava-v1.5-7b-Q4_K.gguf`, both downloaded rather than synthetic fixtures.

---

## Status (continued)

**Session 53** — `#47 initial impl added, junits are passing`, preceded by
an unrelated fix, `#43 multiple issues with dots in offending blocks
fixed`: the start of the Vision-I2T branch.

Unrelated small fix first, landed just before this branch started: `#43`
fixed formatting issues with dots inside "offending blocks" in `run.bat`
and docs output; not vision-related, noted here only because it's the last
commit before the branch diverges from release-0.1.0.

**New `vision` Maven module**: `VisionConfig`, `VisionEncoder` (pure-Java
CLIP ViT-L/14 encoder reading GGUF mmproj weights), `ImagePatchEmbedder`,
`LlavaHandlerFactory`, `VisionAwareForwardPassHandler` (wraps a text
`ForwardPassHandler`, splices patch embeddings into image-token
positions), and `StubForwardPassHandler` (test double). New
`docs/Vision-I2T.md` design doc.

New `VisionChatHandler` (`juno-player`) and a `POST /v1/vision/chat` route
wired into `InferenceApiServer`/`ConsoleMain`.

Full unit-test coverage from day one for every new class
(`ImagePatchEmbedderTest`, `VisionAwareForwardPassHandlerTest`,
`VisionConfigTest`, `VisionEncoderTest`) — all passing at this commit, per
the commit message, though this is the scaffolding stage: no real GGUF
file had been tested against yet (that starts in Session 54).

**Session 52** — EU AI ACT Complience User transparency and AI disclosure.

InferenceApiServer.java; ConsoleMain.java and juno-api.yaml was updated with `The replies are generated by an AI system` water-mark.

##Status

**Session 51** — Documentation update: docs/ folder restructured as juno-documentation MyST Jupyter Book.
juno-documentation

The flat docs/ folder has been restructured into juno-documentation/, a standalone MyST-MD

Jupyter Book configured via myst.yml.

Content is organised into 11 parts and 54 chapters, each in its own .md file under part1/ through part11/. Navigation links (<- / ->) and a full Table of Contents in index.md cross-link every chapter.

All Mermaid diagrams are declared with the MyST {mermaid} directive where applicable and render natively in the built book and in any Mermaid-aware viewer.

A references.md back-matter table maps every chapter back to the originating file in docs/.

build.sh provides a one-command build (./build.sh); README.md documents prerequisites and the local preview workflow.

## Status

**Session 50** — `/train-file-qa`: multi-fact Q&A from a JSON file in one training loop; HTTP API.

### `/train-file-qa`

- REPL command loads a `.json` array of `{"Q","A"}` objects via `LoraQaFile`.
- Each pair expands to the same four chat-templated variants as `/train-qa`; all units
  train in one `trainOnUnits` pass with QA loss targets.
- `LoraTrainer.trainQaPairsUntilResult` for the programmatic multi-pair path.
- `LoraApiServer` — with `./juno lora --api-port N`: `POST /v1/lora/train-file-qa`
  (JSON body) and `POST /v1/lora/save` for curl workflows.
- Dropped verbose `[TRACE]` dump of formatted training text / token IDs on `/train-qa`.
- Docs: `docs/LoRA.md`, `docs/howto.md`.

---

## Status

**Session 49** — LoRA Tier 11 (complete): `--lora-microbatch` CLI/env + VRAM OOM auto-fallback.

### LoRA microbatch CLI and VRAM ladder (Tier 11)

- `LoraMicrobatch` — `--lora-microbatch N` / `LORA_MICROBATCH` (default 8, range 1..128);
  applies `juno.lora.microbatch` before resident upload (no `JAVA_TOOL_OPTIONS` required).
- `LoraResidentUpload` — on FP32 microbatch VRAM OOM with half support: close, set
  microbatch=1, retry FP16 once; further OOM uses existing auto→CPU / gpu fail-closed policy.
- Wired through `LoraCliOptions`, `LoraTrainingConfig`, `ConsoleMain`, `LoraTrainer`,
  `scripts/run.sh` / `run.bat`, and all three LoRA training handlers.
- Docs: `docs/LoRA.md`, `docs/howto.md`, `docs/performance.md`, `docs/agent-arch.txt`.

---

## Status

**Session 48** — LoRA Tier 9 (complete): microbatch GEMM + published GPU speed gates.

### LoRA GPU microbatch and product gates (Tier 9)

- `GpuBlasOps` / `DeviceActivationBatch` — FP32 `cublasSgemm_v2` / `rocblas_sgemm` microbatch
  for frozen forward and transpose; CPU oracle `CpuFrozenBatchOps`.
- Default `juno.lora.microbatch=8` uploads FP32 resident weights and batches linears across
  positions in `LoraTrainableHandler.computeGradients` (host adapters / Adam unchanged).
- `LoraTrainableHandlerGpuBackwardTest` — CPU↔GPU loss/grad parity + TinyLlama speed gates
  (GTX 1080: **~14× e2e**, **~11× backward** vs CPU).
- Docs may describe production **GPU LoRA training** as frozen batched GPU + host adapters;
  device-resident adapters / GPU Adam remain deferred (not required after intensity proof).
- `--lora-train-device` and LLaMA/Qwen2 timing subsets remain as in Session 46 (`transferMs` still 0).

---

## Status

**Session 47** — LoRA Tier 10 (complete): multi-arch GPU residency + production gates.

### LoRA multi-arch GPU residency (Tier 10)

- `LoraResidentWeights` — shared upload / close / VRAM-OOM fallback / matVec+transpose routing.
- `LoraTrainableHandler` refactored onto the helper (LLaMA-family / Qwen2 unchanged behavior).
- `Phi3LoraTrainableHandler` / `Qwen3LoraTrainableHandler` upload physical fused (Phi) or dense
  (Qwen3) projections when `--lora-train-device` resolves to a `GpuMatVec`; CPU fallback preserved.
- Gated live LoRA smokes (`LoraLiveSmokeTest`) for TinyLlama / Qwen2.5 / Phi-3.5 / dense Qwen3 fixtures.
- `EosOutputFilter` — hold back / strip turn-end markers (`</s>`, `<|end|>`, `<|im_end|>`, …) so
  `/train-qa` completions never stream into REPL or `GenerationResult` text (all LoRA chat templates).
- DoRA: correctness-complete, **not** production-perf-gated (prefer LoRA/rsLoRA for large all-linear jobs).
- Tier 7 JFR metrics marked **complete** (programmatic `--jfr`, mode identity, extractor, docs).
- Tier 5 held-out research / quality matrix remains **deferred**; exact K-quant QA-LoRA merge unsupported.

---

## Status

**Session 46** — LoRA Tier 9 (start → completed in Session 48): `--lora-train-device` productization.

### LoRA GPU train-device (Tier 9)

- `--lora-train-device auto|gpu|cpu` / `LORA_TRAIN_DEVICE` (default **auto**).
- `LoraTrainDevice` — MatVec selection; `gpu` fails closed without CUDA/ROCm; `cpu` forces `CpuMatVec`.
- `LoraTrainer` / LoRA REPL honor the mode; JFR `trainDevice` is the resolved label (`cpu`/`cuda`/`rocm`).
- `LoraStepTiming` — fills `frozenForwardMs` / `frozenTransposeBackwardMs` / `adapterBackwardMs` /
  `attentionNonlinearMs` on `juno.LoraTrainStep` from LLaMA/Qwen2 handler instrumentation (`transferMs` still 0 until H2D counters).
- Microbatch / parity IT / speed gates: completed in Session 48.

---

## Status

**Session 45** — LoRA Tier 8: train-file scheduling and corpus caps.

### LoRA train-file scheduling (Tier 8)

- `--lora-chunk-tokens` / `LORA_CHUNK_TOKENS` (default **32**; recommend **128** for large `/train-file`).
- `--lora-max-train-tokens` / `LORA_MAX_TRAIN_TOKENS` (`0` = unlimited): seeded whole-chunk subsample of supervised prediction tokens.
- `/train` and `/train-file` use document-level `TrainUnit`s; chunking happens inside `LoraTrainingLoop`.
- `LoraCorpusLimit` helper; docs/help no longer claim a silent 128 default.

---

## Status

**Session 44** — LoRA training progress bar (loss → target).

- `LoraTrainProgressBar` — percent from pass-2 baseline loss toward `--lora-loss-target-*`; max-iters not used.
- ETA from loss-improvement rate since baseline; final frame ETA `0s` when the run ends.

---

## Status

**Session 43** — LoRA Tier 6: multi-architecture training (CPU oracle).

### LoRA multi-architecture (Tier 6)

- `LoraTrainingHandler` / `LoraTrainingHandlerFactory` — explicit allowlist by `general.architecture`.
- `LoraModelLayout` / `LoraProjectionBinding` — logical keys → physical GGUF tensors (Phi fused slices).
- Handlers: LLaMA-family (`LoraTrainableHandler`), `Qwen2LoraTrainableHandler` (frozen QKV biases),
  `Phi3LoraTrainableHandler` (fused QKV/gate-up + NeoX RoPE), `Qwen3LoraTrainableHandler`
  (per-head Q/K RMSNorm, `qDim`).
- `LoraMerge` layout-aware multi-adapter fused-slice F32 patching for Phi-3.
- Rejected for LoRA: `qwen3moe`, `qwen35`, `gemma`, unknown.
- Qwen3 `/train-qa` template parity with empty `<think>` block.

---

## Status

**Session 42** — LoRA REPL UX + WebUI model dropdown.

- `/reset` deletes the `.lora` checkpoint (no overwrite save); memory reset + chat history clear unchanged.
- LoRA banner and chat footer show sampling `temperature` (and top-k / top-p on the banner).
- Default LoRA training log is a compact progress bar; `--verbose` / `-v` restores full `[TRACE]` / per-pass lines.
- WebUI model dropdown parses OpenAI `GET /v1/models` (`data` / `id` / `x_juno_*`) so names appear again.

---

## Status

**Session 41** — LoRA Tier 7 (complete): JFR metrics for all adapter modes and operations.

### LoRA JFR metrics (Tier 7)

- Programmatic LoRA `--jfr` lifecycle matches local mode (`jdk.jfr.Recording` + auto-extract `target/metrics/metrics.json` on exit). Launchers pass `--jfr` as an app arg (no `-XX:StartFlightRecording` for LoRA).
- `LoraMetricsIdentity` — CLI vocabulary tags (`lora` / `rslora` / `dora` / `qa-lora`) on train, validation, merge, norm-refresh, playback, and checkpoint events.
- New events: `juno.LoraNormRefresh`, `juno.LoraMerge`, `juno.LoraPlayback`, `juno.LoraCheckpoint`.
- `JfrMetricsExtractor` aggregates train/validation/merge/DoRA-refresh/playback series with guarded field reads (older recordings still extract).

---

## Status

**Session 40** — LoRA Tier 5 (complete implementation): QA-LoRA + merge policies.

### LoRA QA-LoRA and quantized merge (Tier 5)

- Gate A codecs retained: `QuantizationLayout`, `GgufQuantCodec` / `GgufKQuantCodec` (`juno-kquant-v1`), `QuantizedMergeMetrics`.
- `QaLoraAdapter` — sum-pool grouped A (`rank×groupCount`) + B; dense-expansion oracle and finite-difference tests.
- `AdapterAlgorithm`, `MergeCapability` (`SIDECAR_ONLY` / `F32_PRESERVE` / `SOURCE_TYPE_PROJECTED`; `EXACT_AFFINE` rejected for K-quants).
- Checkpoint v2: QA entries store `groupWidth` before A, Tier-5 extension blob (algorithm, pooling, ggml type, encoder id, merge policy); v1 export rejected for QA-LoRA.
- `QaLoraInitializer` — group width from actual tensor GGML type (Q4_K/Q5_K→32, Q6_K→16); fingerprints verified on load.
- Training/playback: `LoraTrainableHandler`, Adam, gradients, CLI `--lora-mode qa-lora`, `--lora-group-width`, `--lora-merge`.
- `LoraMerge` — F32 preserve (default) and explicit `SOURCE_TYPE_PROJECTED` requantization with per-tensor metrics; zero-delta copies raw bytes; never silent exact→projected fallback.
- Exact K-quant QA-LoRA zero-point merge remains **no-go**. Full held-out experiment matrix / deployment quality gates are research follow-ups; sidecar + F32 stay production-safe.

---

## Status

**Session 39** — LoRA Tier 5 (Gate A start): shared GGUF K-quant codec layer.

### LoRA QA-LoRA / quantized merge foundations (Tier 5 Gate A)

- `QuantizationLayout` — Q4_K / Q5_K / Q6_K geometry (block/sub-block width, affine vs symmetric).
- `GgufKQuantCodec` / `GgufQuantCodec` — versioned encoder id `juno-kquant-v1`; decode matches llama.cpp goldens; encode moved out of `LoraMerge`.
- `QuantizedMergeMetrics` — RMSE, max error, delta-retention helpers for projected merge.
- `GgufReader` and `LlamaTransformerHandler.dequantize` delegate K-quant decode to the shared codec; fused matVec paths unchanged for performance.
- No-op path: `copyRawUnchanged` — decode/re-encode must not be used for byte-identical preservation.
- Non-closure tests: Q6_K additive shift and Q4_K nested-scale offset are not exact (exact K-merge remains no-go).
- Next: grouped QA-LoRA math (Gate B), merge capability policy, then projected merge experiments.

---

## Status

**Session 38** — LoRA Tier 4 (start): resident transpose primitives and baseline instrumentation.

### LoRA GPU training foundations (Tier 4)

- Vendor-neutral `GpuBindings.opNoTranspose()` (CUDA `CUBLAS_OP_N=0`, ROCm `rocblas_operation_none=111`).
- `GpuMatVec.sgemvTranspose` for resident FP32/FP16 `W^T * g` (same row-major buffer as forward `OP_T`).
- `ResidentWeightMatrix` + `LoraTrainableHandler` routes frozen forward and transpose backward through resident GPU weights when uploaded (`supportsHalfResident` FP16 or FP32 fallback).
- JFR backend labels: `*-resident-transpose` / `*-resident-fp16-transpose`.
- `LoraTrainEvent` fields for frozen forward/transpose, attention/nonlinear, adapter backward, and transfer (filled when finer instrumentation lands).
- GPU adjoint tests: `CudaMatVecTransposeTest`, `RocmMatVecTransposeTest` (`GpuMatVecTransposeContractTest`).
- Baseline section in `docs/performance.md` — hybrid path is not yet marketed as production GPU training.
- `--lora-train-device` shipped in Session 46; CPU/GPU gradient parity IT and speed gates remain open.
- Fix: `LoraAdapterSet.resetFrom` (REPL `/reset`) bumps DoRA cache generation so inference drops trained magnitude coefficients.
- Fix: `/reset` also clears REPL chat history and rotates the session id — otherwise multi-turn context still contains the memorized answers.

---

## Status

**Session 37** — LoRA Tier 3 (phase 1–2): rsLoRA, Kaiming, checkpoint v2, DoRA.

### LoRA advanced adapters (Tier 3)

- Explicit adapter metadata: `LoraAdapterConfig` with `LoraScaling`, `LoraInitialization`, `LoraMode`.
- rsLoRA scale `alpha/√rank`; PEFT-compatible Kaiming-uniform A init (legacy-normal retained for compatibility overloads).
- Checkpoint version 2 (length-delimited) with declared alpha, scaling, init, mode, optional DoRA magnitude and base-tensor fingerprints; v1 still loads.
- Canonical detached-norm DoRA (`DoraMagnitude`, `DoraProjection`); magnitude is an AdamW parameter group with decay off.
- `DoraInitializer` builds magnitudes/fingerprints from GGUF dequant; merge applies LoRA/rsLoRA/DoRA formulas to F32.
- CLI/env: `--lora-mode`, `--lora-scaling`, `--lora-init` (`LORA_MODE`, `LORA_SCALING`, `LORA_INIT`).
- DoRA norm-refresh is correctness-complete but **not** production-perf-gated; prefer standard
  LoRA/rsLoRA for large all-linear jobs until a measured refresh budget exists (Tier 10).

---

## Status

**Session 36** — LoRA Tier 2: schedules, AdamW, dropout, validation, and LoRA+.

### LoRA training quality (Tier 2)

- Warmup/cosine and constant learning-rate schedules (`--lora-lr-schedule`, `--lora-warmup-steps`, `--lora-min-lr`).
- True A-only decoupled AdamW (`--lora-weight-decay`); moments see raw gradients only. Numerical trajectories change vs coupled L2; checkpoints remain compatible.
- LoRA+ parameter groups: A uses scheduled LR, B uses `LR * --lora-plus-ratio` (default `1.0` = ordinary behavior).
- Deterministic train-only inverted dropout (`--lora-dropout`, `--lora-seed`); inference and validation stay dropout-free.
- Forward-only `evaluateLoss`; held-out validation split with patience/min-delta and best-weight restore (`--lora-validation-*`).
- Shared `LoraTrainingLoop` orchestration for REPL and `LoraTrainer`; Q&A variants are hold-out units.
- JFR `LoraTrainStep` carries A/B LR, LoRA+ ratio, and dropout; optional `LoraValidation` event.

---

## Status

**Session 35** — LoRA Tier 1: projection coverage, token-weighted accumulation, and clipping.

### LoRA correctness foundation (Tier 1)

- Configurable projection targets: `qv` (default), `all` / `all-linear`, or comma-separated keys (`wq,wk,wv,wo,wgate,wup,wdown`).
- Complete forward/backward for all seven dense linear projections, including current-position K and inverse-RoPE on Q and K.
- `computeGradients` separated from optimizer updates; token-weighted gradient accumulation across chunks.
- Global L2 gradient clipping after prediction-count normalization (`--lora-max-grad-norm`; `0` disables clip).
- Builder-based `LoraTrainingConfig` and `LoraTrainer.open(..., config)`; legacy overload keeps qv, accum=1, clipping off.
- Architecture gate: Phi-3 / Qwen3 / Qwen3-MoE rejected for LoRA (dense LLaMA-family required).
- `/reset` reinitialises A and B from the selected target config (not B-only zeroing).
- Merge maps all seven projections via `LoraProjection`; adapted tensors remain F32.
- Terminology: LoRA on a quantized GGUF base (not QLoRA).

---

## Status

**Session 34** — Windows launcher fixed: `run.bat` and `juno.bat` fully functional on Windows.

### Windows launcher (`scripts/run.bat`, `juno.bat`)

All subcommands (`cluster`, `local`, `lora`, `merge`, `test`) and flags are now working on Windows.

**Root cause fixes:**

- **JAR name mismatch.** `run.bat` referenced `juno-player.jar` and `juno-master.jar` — names that Maven never produces. The actual artifacts are `juno-player-<version>-shaded.jar` and `juno-master-<version>.jar`. Fixed by reading the project version from `pom.xml` at startup using `findstr` and constructing the correct paths dynamically.

- **Java version detection hang.** CMD cannot redirect `stderr` in a pipeline (`2>&1`) reliably inside a `for /f` loop when delayed expansion is active. `java -version` writes to stderr and the output was silently lost, leaving `JAVAVER_RAW` undefined. Fixed by capturing `java -version 2> tmpfile` to a temp file and reading the file with `for /f`.

- **`find_java` nested-if failure.** Nested `if ... (if ... (...))` blocks are not reliable in CMD with `setlocal enabledelayedexpansion`. Replaced with a flat goto-based structure (`find_java_where` label).

- **Infinite loop on empty argument.** In argument-parsing loops, `if exist "%~1"` on an empty `%~1` expands to `if exist ""` which matches the current directory (always true), causing an infinite loop. Fixed by guarding with `if not "%~1"==""` before the `if exist` check in the `cluster`, `local`, `lora`, and `test` parsers.

- **JFR block inside `if not ... (for ...)` silently skipped.** CMD does not support a `for` command inside an `if` parenthesized block when delayed expansion is on. Replaced with a goto-based pattern (`lora_jfr_skip` / `test_jfr_skip` labels).

**Documentation updated:**

- `README.md` — Windows launcher note in section 2.2, Windows requirements paragraph, `juno.bat` references for `merge`.
- `docs/howto.md` — Windows note at top; Windows command-prompt examples added to every subcommand section (`local`, `cluster`, `lora`, `merge`) and Build and Test.

---

## Status

**Session 33** — Model support documentation: Phi-3 supported; Gemma, Qwen 2 / Qwen3 / Qwen3.5 under development.

### Supported model status (docs)

User-facing docs now state a single, consistent model-support policy:

| Family | `general.architecture` | Status |
|--------|------------------------|--------|
| LLaMA, Mistral, TinyLlama, … | `llama`, `mistral`, … | Supported via `LlamaTransformerHandler` |
| Phi-3 / Phi-3.5 | `phi3` | **Supported** via `Phi3TransformerHandler` |
| Gemma | `gemma` | **Under development** (`LlamaTransformerHandler` + `gemma` template) |
| Qwen 2 / 2.5 | `qwen2` | **Under development** (Llama handler + QKV bias groundwork) |
| Qwen3 dense | `qwen3` | **Under development** (`Qwen3TransformerHandler` in progress) |
| Qwen3-MoE | `qwen3moe` | **Under development** (`Qwen3MoeTransformerHandler` in progress) |
| Qwen3.5 | `qwen35` | **Under development** (hybrid DeltaNet; separate handler) |

**Updated files:**

- **`README.md`**, **`RELEASE_NOTES.md`** — Supported models section
- **`docs/arch.md`** — handler routing and tokenizer notes
- **`docs/features.md`**, **`docs/howto.md`**, **`docs/LoRA.md`** — Phi-3 OK for inference; Gemma and Qwen paths not production-ready; LoRA still LLaMA-family (+ Phi-3 template detection)
- **`docs/phi3-inference-handoff.md`** — status set to supported (retains debug handoff notes)
- **`docs/model_support_summary_972ab30f.plan.md`** — roadmap, dispatch table, chat matrix, gaps, decisions log

**Policy:** Phi-3 is production-ready in docs and validation (local + cluster). Gemma and all Qwen families remain under development until dedicated validation lands.

---

## Status

**Session 32** — ROCm/HIP backend for AMD GPU inference via Panama FFI.

### AMD GPU support (ROCm/HIP + rocBLAS)

Full first-class AMD GPU support alongside the existing NVIDIA CUDA backend. The GPU
abstraction layer auto-selects CUDA > ROCm > CPU at startup with no configuration required.
Tested on AMD Radeon RX 7900 XT (gfx1100, ROCm 7.2.x).

**New production classes (`node` module):**

- **`GpuBindings`** — vendor-neutral interface implemented by `CudaBindings` and `RocmBindings`.
  Exposes all device runtime and BLAS handles as `MethodHandle` accessors, shared constants
  (`H2D`, `D2H`, `STREAM_NON_BLOCKING`), and static helpers (`check`, `callInt`, `loadLibrary`,
  `bind`). Static helpers eliminate per-implementation boilerplate.
- **`GpuMatVec`** — sealed interface (`permits CudaMatVec, RocmMatVec`) extending `MatVec`.
  Exposes `upload(float[], int, int)` and `uploadHalf(float[], int, int)` so transformer
  handlers depend on the GPU abstraction rather than a concrete vendor class.
- **`RocmBindings`** — Panama FFI downcall handles for `libamdhip64.so` and `librocblas.so`.
  Pre-binds `hipHostMalloc flags=0` via `MethodHandles.insertArguments` to match the
  `cudaMallocHost` arity visible to all callers. Key ROCm constants: `opTranspose()=112`
  (`rocblas_operation_transpose`), `hipDeviceProp_t` sizeof=1472, name@0, totalGlobalMem@288
  (measured from ROCm 7.2.x headers, Linux x86_64).
- **`RocmAvailability`** — HIP device detection: `isAvailable()`, `deviceCount()`,
  `deviceName(int)`, `vramBytes(int)`. Mirrors `CudaAvailability` in structure.
- **`RocmMatVec`** — `MatVec` / `GpuMatVec` implementation backed by `rocblas_sgemv` (FP32)
  and `rocblas_hssgemv_strided_batched` (FP16). Three compute paths:
  - Host FP32: temporary device buffers per call; synchronous H2D → kernel → D2H.
  - Device-resident FP32 (`DeviceFloatMatrix`): per-thread scratch for x/y; async stream copies.
  - Device-resident FP16 (`DeviceHalfMatrix`): x converted FP16 in off-heap arena; FP32 accumulation.
  Off-heap `Arena.ofConfined()` staging for all H2D/D2H copies — required by Java 25 Panama
  (heap segments rejected by native downcalls).
- **`MatVecBackend`** — enum replacing ad-hoc string literals for the `juno.MatVec.backend` JFR
  dimension. Values: `CPU`, `CUDA`, `CUDA_RESIDENT`, `CUDA_RESIDENT_FP16`, `ROCM`,
  `ROCM_RESIDENT`, `ROCM_RESIDENT_FP16`. Label strings are part of the JFR contract and unchanged.

**Modified production classes:**

- **`GpuContext`** — refactored from CUDA-only to backend-agnostic. Adds `GpuBindings bindings`
  field, `bindings()` accessor, `selectBindings()` (CUDA → ROCm priority order with
  `-Djuno.gpu.backend=cuda|rocm|auto` override), `createMatVec()` factory, `backendLabel()`
  delegate. `close()` uses `bindings.cublasDestroy()` instead of hardcoded CUDA call.
  Private `deviceName()` and `deviceVram()` helpers use `GpuBindings` struct-offset accessors.
- **`CudaBindings`** — adds `implements GpuBindings`; 20 accessor methods expose the existing
  `MethodHandle` fields to vendor-neutral callers. Zero existing fields or constants removed.
- **`CudaAvailability`** — field-access calls updated to use `CudaBindings.instance()` accessor
  methods (`PROP_NAME_OFFSET` → `instance().PROP_NAME_OFFSET`, etc.).
- **`CudaMatVec`** — implements `GpuMatVec` (was `MatVec`); `upload` / `uploadHalf` made public
  with `@Override`; backend labels replaced by `MatVecBackend` enum calls.
- **`DeviceFloatMatrix` / `DeviceHalfMatrix`** — direct `CudaBindings.instance()` field access
  replaced by `GpuContext#bindings()` method calls (`GpuBindings`). Both classes now work
  identically on CUDA and ROCm. `DeviceHalfMatrix` caches `gpu = ctx.bindings()` at construction.
- **`LlamaTransformerHandler`** — `instanceof CudaMatVec` → `instanceof GpuMatVec` for weight
  upload gate; `cudaMalloc` OOM message check extended to also catch `hipMalloc`;
  `matVecQuantBackendLabel(int)` → `matVecQuantBackend(int)` returns `MatVecBackend.CPU`.
- **`Phi3TransformerHandler`** — same `instanceof` fix; OOM check extended to `hipMalloc`.
- **`LoraTrainableHandler`** — same `instanceof` fix.
- **`ForwardPassHandlerLoader`** — `pickMatVec` checks both `CudaAvailability` and
  `RocmAvailability`; device count query reads from the available backend; `GpuContext.shared(dev).createMatVec()` replaces `new CudaMatVec(...)`.
- **`EmbeddedNodeServer`** — uses `gpuContext.createMatVec()` and `gpuContext.backendLabel()`
  for log messages.
- **`ConsoleMain` / `JunoPlayer`** — `new CudaMatVec(gpuCtx)` → `gpuCtx.createMatVec()`.
- **`MatVecEvent`** — adds `backend(MatVecBackend)` setter to avoid hand-written label strings
  at call sites; public `String backend` field kept for JFR contract.

**New tests (55 total, 0 failures on RX 7900 XT):**

- `RocmMatVecTest` (30) — extends `MatVecBackendContractTest` for full API parity; correctness
  vs CPU reference at 2048×2048, 5632×2048, 32000×2048; trivial known-value cases;
  4-thread concurrent safety; throughput sanity.
- `RocmAvailabilityTest` (8) — device detection present/absent; name format; VRAM bounds;
  out-of-range index fallbacks.
- `GpuContextTest` +5 `@Tag(rocm)` — ROCm context lifecycle, backend priority,
  `createMatVec` factory, shared singleton, system-property override.
- `ForwardPassHandlerLoaderSelectBackendTest` +2 `@Tag(rocm)` — `RocmMatVec` routing,
  process-wide `GpuContext.shared(0)` reuse.
- `ForwardPassHandlerLoaderSelectLoraBackendTest` +1 `@Tag(rocm)` — LoRA routing on ROCm.
- `MatVecQuantizedBackendLabelTest` — updated to use `MatVecBackend` enum constants.

Run ROCm-tagged tests:
```bash
mvn test -pl node -Dgroups=rocm
```

**Performance (RX 7900 XT, ROCm 7.2.x):**

| Shape | Path | Time (5 runs) |
|-------|------|--------------|
| 32000×2048 | `rocblas_sgemv` host FP32 | 408 ms |

All existing 194 unit tests pass unchanged.

---

## Status

**Session 31** — Panama FFI for Juno math: JavaCPP / bytedeco removed, CUDA bindings rewritten with `java.lang.foreign`.

### Panama FFI GPU bindings (`node` module)

The entire CUDA bridge has been rewritten using the Java 25 Panama Foreign Function & Memory API
(`java.lang.foreign.Linker`, `SymbolLookup`, `MemorySegment`, `Arena`). The `org.bytedeco:cuda-platform`
dependency has been removed from `node/pom.xml`.

**New production class:**

- **`CudaBindings`** — Panama FFI downcall handles for `libcudart.so.12` and `libcublas.so.12`.
  Resolves all CUDA Runtime and cuBLAS symbols once at class-init time via `Linker` and
  `SymbolLookup`; resulting `MethodHandle` instances are thread-safe with zero per-call Java overhead.
  Exposes: `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaSetDevice`, `cudaMalloc`,
  `cudaFree`, `cudaMallocHost`, `cudaFreeHost`, `cudaMemcpy`, `cudaMemcpyAsync`,
  `cudaStreamCreateWithFlags`, `cudaStreamSynchronize`, `cudaStreamDestroy`,
  `cublasCreate`, `cublasDestroy`, `cublasSetStream`, `cublasSetPointerMode`,
  `cublasSgemv`, `cublasHSSgemvStridedBatched`.
  `cudaDeviceProp` struct-offset constants (`DEVICE_PROP_BYTES=1512`, `PROP_NAME_OFFSET=0`,
  `PROP_TOTAL_MEM_OFFSET=288`) measured from CUDA 12.x headers on Linux x86_64.
  Singleton init: `CudaBindings.instance()` / `CudaBindings.isAvailable()`.

**Modified production classes:**

- **`CudaMatVec`** — all JNI / JavaCPP call sites replaced with `CudaBindings` downcall handles.
  Native memory managed exclusively via `MemorySegment` and `Arena`. Device weight matrices
  (`DeviceFloatMatrix`, `DeviceHalfMatrix`) held resident; `MemorySegment` passed directly to
  cuBLAS as `ADDRESS` — zero H2D copy per token. Per-thread `Fp32Scratch` / `Fp16Scratch`
  scratch on device grown lazily and reused. FP16 x staging packed with `Float.floatToFloat16`
  into a confined off-heap arena in the hot path.
- **`GpuContext`** — cuBLAS handle stored as `MemorySegment` (opaque `cublasHandle_t`); created
  and destroyed via `CudaBindings`. `cublasSerializationLock()` serializes stream-binding and
  kernel submission on the shared handle. `shared(int)` returns a process-wide singleton per
  device index.
- **`DeviceFloatMatrix`** — device memory allocated via `CudaBindings.deviceMalloc`; backing
  `MemorySegment` sized to `rows * cols * 4` bytes; H2D via synchronous `cudaMemcpy`.
- **`DeviceHalfMatrix`** — same pattern; FP16 x staging via confined arena; `MemorySegment.ofArray`
  pins heap array for duration of downcall.
- **`CudaAvailability`** — device detection updated to use `CudaBindings` downcall handles.

**`node/pom.xml`:** `org.bytedeco:cuda-platform` dependency removed.
`maven-surefire-plugin` `argLine` updated: `--enable-native-access=ALL-UNNAMED`,
`--add-opens java.base/java.lang=ALL-UNNAMED`, `--add-opens java.base/java.nio=ALL-UNNAMED`.

**New test: `CudaBindingsTest`** — two scenarios:
- CUDA present (`@Tag("gpu")`): every `MethodHandle` non-null, singleton loads cleanly.
- CUDA absent (CPU-only CI): `isAvailable()` returns false, `instance()` throws `IllegalStateException`.

Run GPU-tagged tests: `mvn test -Dgroups=gpu -pl node`

All existing tests pass unchanged.

---

## Status

**Session 30** — Maven Central publish configuration.

### Maven Central publish (`pom.xml`, all module POMs)

All modules configured for publishing to `central.sonatype.org` via the Central Portal publisher.
Version set to `0.1.0-RC` across root POM and `juno-bom`.

**Changes:**

- **`maven-source-plugin 3.3.1`** — `attach-sources` execution at `verify` phase; produces `-sources.jar`
  required by Maven Central.
- **`maven-javadoc-plugin 3.11.2`** — `attach-javadocs` execution at `verify` phase; `doclint=none`,
  `failOnError=false`; produces `-javadoc.jar` required by Maven Central.
- **`maven-gpg-plugin`** — `sign-release` execution moved from `verify` to `install` phase so
  sources and Javadoc jars are already attached before signing. `--pinentry-mode loopback`
  added to `gpgArguments` to allow `-Dgpg.passphrase=...` without a GUI pinentry agent.
- **`distributionManagement`** — `<repository>` and `<snapshotRepository>` wired to
  `central.sonatype.org` Central Portal publisher endpoint.
- **Developer / SCM metadata** — `<organization>Machine Learning Cabinet</organization>`,
  `<organizationUrl>https://ml.cab/</organizationUrl>`, SCM tag updated to `v0.1.0-RC`.
- **All module POMs** — publish config consolidated into root POM; per-module boilerplate removed.

---

## Status

**Session 29** — OpenAI-compatible REST API (`POST /v1/chat/completions`, `GET /v1/models`).

### OpenAI-compatible API

Any client that speaks the OpenAI Chat Completions wire format — LangChain, LlamaIndex,
LiteLLM, the OpenAI Python/Node SDKs, or any internal tool built against `openai.*` — works
against Juno with a single base-URL change. No prompt reformatting, no adapter library, no
glue code.

**New classes (coordinator module):**

- **`OpenAiAdapter`** — pure static mapping helpers between Juno internals and the OpenAI wire
  format: `repetitionPenaltyFromFrequencyPenalty(float)` (OpenAI −2..2 range → Juno ≥1),
  `validateCompletionsN(Integer)` (rejects n ≠ 1), `toOpenAiFinishReason(StopReason)` (`stop`
  / `length` / `error`), and `chatCompletionId(String)` (`chatcmpl-` + compact UUID).
- **`OpenAiChatHandler`** — Javalin handler class owning three endpoints:
  - `POST /v1/chat/completions` — deserialises `OaiChatCompletionRequest` (Jackson,
    `@JsonIgnoreProperties(ignoreUnknown = true)`), validates `n` and `messages`, builds an
    `InferenceRequest` + `SamplingParams`, then dispatches to either
    `scheduler.submitAndWait()` (blocking, returns `ChatCompletion` JSON) or
    `scheduler.submit()` (streaming, writes `text/event-stream` chunks terminated by
    `data: [DONE]`).
  - `GET /v1/models` — filters `ModelRegistry` to `LOADED` status, wraps each
    `ModelDescriptor` in an OpenAI `Model` object with `x_juno_*` extension fields.
  - `GET /v1/models/{modelId}` — single-model lookup; 404 when absent.

**Modified: `InferenceApiServer`** — constructs `OpenAiChatHandler` in the constructor
(passing the latency callback so `HealthReporter` still records P99). Routes
`POST /v1/chat/completions` and `GET /v1/models[/{modelId}]` to the handler.
The existing `POST /v1/inference` and `POST /v1/inference/stream` endpoints are untouched.

**Modified: `ConsoleMain`** (`juno-player` module) — `--api-port N` flag starts a
`RequestScheduler` + `InferenceApiServer` alongside the existing REPL in both `local` and
cluster modes. A virtual-thread shutdown hook calls `apiServer.stop()` on JVM exit.
`buildLocalModelRegistry()` populates a `ModelRegistry` from the in-process `LlamaConfig` so
`GET /v1/models` returns the loaded model immediately.

**Modified: `scripts/run.sh`** — `--api-port N` flag wired into both `cmd_local()` and
`cmd_cluster()`. Environment override: `API_PORT`.

**New file: `api/src/main/resources/juno-api.yaml`** — OpenAPI 3.0.3 spec for the public
client-facing API. Documents all request fields with their Juno internal mappings, the SSE
chunk event sequence, Juno extension fields (`x_juno_priority`, `x_juno_session_id`,
`x_juno_top_k`, `x_juno_latency_ms`, `x_juno_retry_after_ms`, `x_juno_queue_depth`), and
all error codes.

**New test: `OpenAiAdapterTest`** — unit tests for all four mapping helpers.

**Field mapping summary (request):**

| OpenAI field | Juno internal | Notes |
|---|---|---|
| `model` | `modelId` | First loaded model if omitted |
| `messages[].role` / `.content` | `ChatMessage` | Text only; images not supported |
| `temperature` | `SamplingParams.temperature` | 0.0–2.0; default 0.7 |
| `top_p` | `SamplingParams.topP` | 0.0–1.0; default 0.9 |
| `max_completion_tokens` | `SamplingParams.maxTokens` | 1–32768; default 200 |
| `max_tokens` | `SamplingParams.maxTokens` | Deprecated alias |
| `frequency_penalty` | `SamplingParams.repetitionPenalty` | `1 + max(0, fp/2)` |
| `stream` | route selection | false → blocking JSON; true → SSE |
| `n` | — | Only 1 is accepted; other values → 400 |
| `stop`, `presence_penalty`, `logit_bias`, `user`, `seed` | — | Silently ignored |
| `x_juno_priority` | `RequestPriority` | HIGH / NORMAL / LOW |
| `x_juno_session_id` | `InferenceRequest.sessionId` | Enables KV-cache reuse across turns |
| `x_juno_top_k` | `SamplingParams.topK` | 0 = disabled; default 50 |

All modules compile. All existing tests pass. `OpenAiAdapterTest` (4 assertions) passes.

---

## Status

**Session 28** — Health dashboard: CPU load metric, role-conditional secondary metric, node throughput.

### Health dashboard fixes

**Fix 1 — `temperatureCelsius` → `cpuLoad`.**
`/sys/class/thermal` is unavailable on EC2 VMs; the Temperature row always showed `—`. Replaced with process CPU utilisation read from `OperatingSystemMXBean.getCpuLoad()` (0.0–1.0, available on all JVM platforms, no sysfs). Changes:
- `NodeHealth` record: field `temperatureCelsius` removed, `cpuLoad` added (same sentinel -1.0 convention, clamped to 0.0 on first-sample unavailability).
- `HealthReporter.buildProbeJson()`: `readTemperatureCelsius()` + all sysfs helpers (`findThermalZone`, `findHwmonTemp`, thermalPath/thermalProbed state) removed; replaced by 5-line `readCpuLoad()`.
- `HealthMain.NodeHealthDto`: `temperatureCelsius` field → `cpuLoad`.
- Dashboard HTML (both `HealthMain` and `InferenceApiServer` embedded console): "Temperature" row → "CPU load" formatted as `XX.X %`.

**Fix 2 — Role-conditional secondary metric: coordinator shows Latency P99, nodes show Throughput.**
`Latency P99` was populated by `HealthReporter.recordLatency()`, which is only called from `InferenceApiServer` on the coordinator JVM. Worker nodes always showed `—`. Added a `nodeRole` field (`"coordinator"` | `"node"`) to `NodeHealth` and `NodeHealthDto` so the dashboard can branch:
- **Coordinator card** — Latency P99 (ms): end-to-end generation time, already wired via `InferenceApiServer.setLatencyReporter()`.
- **Worker node cards** — Throughput (MB/s): activation bytes forwarded per second via new `HealthReporter.recordBytes(long n)` + `drainThroughput()` (atomic byte counter drained each probe interval).

Wiring:
- `EmbeddedNodeServer`: retained `NodeServiceImpl` reference as `serviceImpl` field; added `setHealthReporter(HealthReporter)` on outer class delegating to a new package-private setter on the inner class. `forwardPass()` calls `hr.recordBytes(encodedOutput.length)` after each `responseObserver.onNext()`.
- `NodeMain`: constructs reporter with `nodeRole="node"`, calls `server.setHealthReporter(reporter)` after `server.start()`.
- `CoordinatorMain`: constructs reporter with `nodeRole="coordinator"`.
- `HealthReporter` constructors: 2-arg and 3-arg remain backward-compatible (default role `"node"`); new canonical 4-arg constructor `(nodeId, nodeRole, healthBaseUrl, intervalMs)`. Added `startForCoordinator(healthBase)` factory alongside existing `startForNode(nodeId, healthBase)`.
- `buildNodeDetail()` switched from `Map.of()` (10-entry limit) to `Map.ofEntries()` to accommodate 12 fields.

**Investigation 3 — Why 1 of 10 concurrent sessions produced no tokens (no code change).**
Root cause: gRPC `ServerBuilder.forPort(port)` with no custom executor defaults to a thread pool bounded by `~2 × CPU count` (4 threads on `m7i-flex.large`). With 9 sessions concurrently running prefill (26 steps × 9 = up to 234 in-flight blocking stubs), all 4 gRPC threads on each node were saturated. The 10th session's first `pipeline.forward()` call queued behind them for ~8.5 minutes until prefill of the other 9 finished. The fix is `ServerBuilder.forPort(port).executor(Executors.newVirtualThreadPerTaskExecutor())` — virtual threads don't block OS threads on gRPC I/O. JFR evidence: `juno.ForwardPass.decode.p95_ms = 3095 ms` on node-1 (coordinator node running layers 0–8 plus the REST server) vs 914 ms on node-2; coordinator log confirms 10 tokenizer encodes but only 9 near-simultaneous prefills.

All modules compile. All existing tests pass (NodeHealth, HealthEvaluator, HealthReactor constructors updated to 9-arg signature).

---

**Session 27** — GPU lifecycle, multi-device shared contexts, CUDA streams, Llama VRAM fallback, docs.

- **`ForwardPassHandler.releaseGpuResources()`** — default no-op; **`LlamaTransformerHandler`** and **`Phi3TransformerHandler`** close all **`DeviceHalfMatrix`** buffers. **`EmbeddedNodeServer`** invokes it on shard reload, load failure, and **`unloadShard`** (then swaps in **`StubForwardPassHandler`**).
- **`GpuContext.shared(int)`** — one process-wide **`GpuContext`** per CUDA device index (map + lock); **`close()`** remains a no-op for shared instances. **`ForwardPassHandlerLoader.selectBackend()`** and **`EmbeddedNodeServer`** honour **`-Djuno.cuda.device=N`**, validated against **`CudaAvailability.deviceCount()`**.
- **`CudaMatVec`** — per-thread **non-blocking CUDA stream**; **`cublasSetStream_v2`** + **`cudaMemcpyAsync`** for resident FP32/FP16 **`x`/`y`** transfers; **`synchronized(gpuContext.cublasSerializationLock())`** around stream binding and kernels. Host **`sgemv(float[],…)`** also runs under the same lock.
- **Llama GPU OOM** — upload wrapped like Phi-3: on **`cudaMalloc`** failure, partial **`DeviceHalfMatrix`** buffers are **`close()`**d and inference falls back to **CPU quantised** matmul for those projections.
- **Docs/tests:** **`README.md`**, **`docs/arch.md`**, **`GpuContextTest`** (multi-GPU assumption), **`NodeMain`** Javadoc for **`juno.cuda.device`**.

All modules build and all tests pass. Verified end-to-end with:
- TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
- TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
- TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf
- Meta-Llama-3.2-1B-Instruct-Q8_0.llamafile
- phi-3.5-mini-instruct.Q4_K_M.gguf on a 3-node CPU cluster
- Phi-3.5 GPU matmul path: `CudaMatVecBackendTest` FP16 resident matvec + `mvn test -Dgroups=gpu -pl node` on CUDA 12.x

**Session 26** — Phi-3 GPU matmul, FP16 resident weights, CLI and local GPU wiring.

`Phi3TransformerHandler` GPU path uploads dequantized fused QKV / FFN slices and output projection as **`DeviceHalfMatrix`** (IEEE FP16 on device, roughly half the VRAM of `DeviceFloatMatrix`). Forward uses **`CudaMatVec.sgemv(DeviceHalfMatrix, x)`**, implemented with **`cublasHSSgemvStridedBatched`** — same `(CUBLAS_OP_T, m=cols, n=rows, lda=cols)` layout contract as the proven **`cublasSgemv_v2`** path for row-major `A`. Host `float[]` activations are converted to FP16 for the per-call device `x` buffer; accumulation stays FP32. Earlier **`cublasSgemmEx` / `cublasGemmEx`** mixed-dtype attempts returned `NOT_SUPPORTED` / `INVALID_VALUE` on common stacks; the HSS strided-batched GEMV avoids that.

**Session 26** — Native LoRA merge (`juno merge`).

`LoraMerge` (new, `node` module) writes a new GGUF file from a base model and a `.lora` checkpoint without re-quantising the patched tensors. The 44 LoRA-adapted projection weights (wq/wv on every layer) are stored as F32; all other tensors are copied verbatim in their original quantised encoding. F32 is required because the LoRA delta (~6×10⁻⁴) is smaller than Q4_K quantisation noise (~3×10⁻³) — re-quantising would silently erase the training. Verified: merged TinyLlama recalls `/train-qa` facts (name "Dima") correctly under `./juno local` with no `.lora` sidecar.

`GgufReader` gains five new public methods needed by the GGUF writer: `ggufFileOffset()`, `metadataSectionEnd()`, `tensorOrder()`, `tensorNelems(name)`, and keeps the existing `tensorAbsoluteOffset` / `tensorType` / `tensorDims`. Internal storage changed from `HashMap` to `LinkedHashMap` so `tensorOrder()` is stable. A `List<String> tensorOrder` field is added to preserve insertion order.

`LoraMergeMain` (`juno-player` module) — CLI entry point for `juno merge`. Reads `--model-path`, `--lora-path`, `--output`, `--heap`. Derives `<model>.lora` and `<model>-merged.gguf` as defaults.

`run.sh` gains `cmd_merge()` and the `merge)` dispatch case.

`ConsoleMain` `/merge-hint` REPL command updated: now prints the actual `./juno merge` invocation instead of the old "contributions welcome" message.

Three bugs fixed during development of `LoraMerge`:
- **Q4_K**: `d = maxRange/63` → `d = maxRange/(63×15)`. Previous formula collapsed all 4-bit quant values to `{0,1}`.
- **Q5_K**: same bug, factor 31. `d = maxRange/63` → `d = maxRange/(63×31)`.
- **Q3_K scRaw packing**: aux0/aux1 high-nibble extraction used a broken two-pass utmp reconstruction; replaced with a clean direct inverse of `GgufReader.loadQ3_K`.

**Session 25** — Code quality: dead code removed, test helpers moved to test scope, docs fully updated.


`CyclicForwardPassHandler` moved from `node/src/main` to `node/src/test`. It is a deterministic stub with no business value without a model; it belongs exclusively in the test compilation unit. `EmbeddedNodeServer` no longer imports it — the three call sites (pre-load placeholder, model-load-failure fallback, no-model stub mode) are now served by a new private `StubForwardPassHandler` inner class that returns zero-filled arrays of the correct shape with no test machinery. `node/pom.xml` gains a `maven-jar-plugin` `test-jar` execution so other modules can still import `CyclicForwardPassHandler`; `coordinator/pom.xml` and `juno-master/pom.xml` declare the `node:tests` classifier dependency.

**VRAM / OOM:** GPU buffer allocation is wrapped; on failure (including `cudaMalloc` OOM), partial device buffers are closed and the handler falls back to **CPU quantised** `LlamaTransformerHandler.matVec`-style matmul for those projections.

**`ConsoleMain`:** missing **`break`** after **`--cpu`** fixed — parsing no longer fell through into **`--lora`**, which incorrectly set `loraMode` when forcing CPU inference.

**`ConsoleMain.runLocalRepl`:** one shared **`GpuContext`** + **`CudaMatVec`** instance for every in-process shard load (avoids redundant cuBLAS contexts and matches production “one GPU per JVM” usage).

**Tests:** `CudaMatVecBackendTest.device_half_matrix_sgemv_matches_host_path` (512×512) anchors FP16 resident correctness vs `LlamaTransformerHandler.matVec`.

**JFR:** `MatVecEvent.backend` **`cuda-resident-fp16`** labels the Phi FP16 device path. (As of session 27, Llama GPU resident weights also use **`cuda-resident-fp16`**; **`cuda-resident`** remains for **`DeviceFloatMatrix`** / tests.)

---

**Session 26** — LoRA inference overlay (`--lora-play`), Q&A training mode (`/train-qa`), diagnostic tracing, and AWS deploy hardening.

### `--lora-play PATH` — apply trained adapters at inference in any mode

Pre-trained `.lora` checkpoint files can now be applied read-only at inference time without entering the `lora` REPL. Three modes are supported:

**`local` mode:**
```bash
./juno local --model-path model.gguf --lora-play /path/to/model.lora
```
`ConsoleMain.runLocalRepl()` calls `LoraAdapterSet.load(Path.of(loraPlayPath))` before building the shard handlers and passes the result into `ForwardPassHandlerLoader.load(..., playAdapters)`.

**`cluster` mode (forked JVMs):**
```bash
./juno --model-path model.gguf --lora-play /path/to/model.lora
```
`ClusterHarness.withLoraPlay(path)` injects `-Djuno.lora.play.path=PATH` into every forked node JVM command. `EmbeddedNodeServer.NodeServiceImpl` reads this property at construction and loads adapters inside `loadShard()` before the `ForwardPassHandlerLoader` call.

**AWS deployed cluster:**
```bash
./launcher.sh juno-deploy.sh setup --lora-play /absolute/path/to/model.lora
```
See AWS section below.

### `ForwardPassHandlerLoader` — new LoRA overload

```java
// New canonical overload — all others delegate to this
public static ForwardPassHandler load(
    Path modelPath, ShardContext context, MatVec backend,
    LoraAdapterSet adapters) throws IOException
```

When `adapters != null`, the loader routes to `LoraTrainableHandler` (inference-only, no optimizer attached) instead of the architecture-specific handler. When `adapters == null` the existing `phi3` / `llama` dispatch is unchanged. `selectBackend()` promoted from package-private to `public` so juno-player-module callers can reuse it.

### `ClusterHarness` — `withLoraPlay()` fluent method

```java
harness.withLoraPlay("/path/to/model.lora");
```

Stores the path and injects `-Djuno.lora.play.path=PATH` into the `launchNode()` JVM command, after the JFR flags. Without this, forked node JVMs start with `loraPlayPath=null` and run the base model regardless of what the coordinator is told.

### `/train-qa` — conversational Q&A training

New REPL command in `lora` mode for training single-fact associations:

```
you > /train-qa What is my name? A: Dima
  Question: What is my name?
  Answer  : Dima

  Formatted as 4 Q&A pairs  ·  model type: tinyllama
  Training  rank=8 · lr=1.0E-4 · 40 steps ...
  ✔ done  loss=▼ 1.53 (−0.83)
```

The command auto-generates 4 phrasings of the question (exact, `Can you tell me: ...`, `Please answer: ...`, plus one repeat) to improve generalization. The chat template appropriate for the model type (detected from the model path) is applied to each pair. Flags `--lora-steps-qa N` and `--lora-early-stop F` control training depth.

Separator syntax: `Q: <question> A: <answer>` or `<question> A: <answer>`.

### Diagnostic tracing (`--verbose`)

All tracing is prefixed `[TRACE]` for easy grep. Added to:

| Location | What is shown |
|----------|---------------|
| LoRA REPL startup | Model type (chat template key), model path, all LoRA hyperparameters |
| `/train-qa` | Exact formatted training text with `↵` for newlines, token count, token IDs (verbose only) |
| Per training step (verbose) | `step=N loss=F chunk=M/T ms=D` |
| Cluster inference (verbose) | Chat template key used for each inference request |
| `juno-deploy.sh` bootstrap | Per-node params baked into user-data script |
| `juno-deploy.sh` SCP | Local source, remote target, per-node `node.env` patch |
| `juno-deploy.sh` coordinator env | Full `cluster-nodes.env` contents echoed after write |

### AWS deploy hardening (`juno-deploy.sh`)

Multiple bugs fixed during end-to-end AWS validation:

**Double base64 encoding (cloud-init rejected user-data).** `--user-data` was passed as a pre-base64-encoded string. AWS CLI base64-encodes it again; cloud-init received double-encoded garbage and logged `Unhandled non-multipart (text/x-not-multipart) userdata`. Fix: write user-data to a temp file and pass `file:///tmp/juno-userdata-*.sh` — the CLI reads it raw and does single encoding. The `[TRACE]` size line now also prints `first-line: #!/bin/bash` so shebang presence is visible in the setup log.

**TRACE logs contaminating user-data.** `_build_node_userdata` is called as `USER_DATA=$(_build_node_userdata ...)` which captures all stdout. The four `log` / `[TRACE]` calls inside the function were writing to stdout, prepending ANSI escape codes before `#!/bin/bash`. Cloud-init saw no shebang on line 1 and skipped execution. Fix: all `log` calls inside `_build_node_userdata` now redirect to stderr with `>&2`.

**Relative `--lora-play` path not resolved.** When called from `scripts/aws/`, a path like `../models/model.lora` resolves to `scripts/models/model.lora` (which doesn't exist). `_scp_lora_to_nodes` hit the `[[ ! -f ]]` guard and returned silently, leaving `node.env` with empty `JUNO_LORA_PLAY_PATH`. Fix: `--lora-play` is resolved to absolute path at parse time via `realpath`. `setup()` also validates the file exists before any AWS spend.

**Race condition: coordinator started before node restart completed.** `_scp_lora_to_nodes` previously used `systemctl restart --no-block` and polled `systemctl is-active` to detect readiness. The old instance remained `active` during shutdown so the poll returned immediately, `_write_cluster_env_and_start_coordinator` ran, and the coordinator sent `loadShard` to the old (no-LoRA) instance. The restarted instance came up 19 minutes later, too late. Fix: synchronous stop → patch → start per node: `systemctl stop juno-node` (synchronous, waits for JVM exit), `sed` patch of `node.env`, `systemctl start juno-node` (synchronous, returns once gRPC port is bound, ~2s). Coordinator only starts after all three nodes have confirmed `active` status with correct env.

**Local relative path baked verbatim into `cluster-nodes.env`.** Even when SCP succeeded, the coordinator received `JUNO_LORA_PLAY_PATH=../models/...` (the pre-`realpath` value), causing `model load failed: ../models/...` on the nodes. Fix: `_scp_lora_to_nodes` updates the global `LORA_PLAY_PATH` to the remote absolute path (`/opt/juno/models/<basename>`) before returning, so `_write_cluster_env_and_start_coordinator` writes the correct value.

**`_write_cluster_env_and_start_coordinator` missing closing brace.** The `}` was accidentally elided, causing `scan_regions()` to be parsed as part of the function body.

**End-to-end verification:**
```
you> what is my name?
bot> Dima
```
Confirmed working on 3 × m7i-flex.large AWS cluster (eu-north-1) with TinyLlama-1.1B-Chat-v1.0.Q4_K_M and a `.lora` adapter trained locally, SCPed and deployed via `juno-deploy.sh setup --lora-play`.

---

**Session 34** — Windows launcher fixed: `run.bat`/`juno.bat` fully functional; docs updated with Windows examples. *(this session)*

**Session 33** — Model support documentation: Phi-3 supported; Gemma, Qwen 2 / Qwen3 / Qwen3.5 under development. *(unchanged)*

**Session 24** — Configurable activation byte order (`--byteOrder BE|LE`). *(unchanged)*

**Session 22** — Q2_K and Q3_K quantization support. *(unchanged)*

**Session 21** — Two new deployment fat-jar modules and a unified AWS script. *(unchanged)*

**Session 20** — GPU inference actually wired end-to-end. *(unchanged)*

**Session 19** — metrics module, Meta-Llama 3 tokenizer fix, AWS infrastructure scripts. *(unchanged)*

**Session 18** — GPT-2 BPE tokenizer, JFR instrumentation fixes. *(unchanged)*

**Session 17** — AWS infrastructure scripts. *(unchanged)*

**Session 14** — LoRA fine-tuning + JFR profiling. *(unchanged)*

---

## Status

**Session 23** — JFR auto-extraction for local and cluster modes; AWS deploy gains `--jfr` with remote JFR collection. *(unchanged)*

**Session 16** — Naming cleanup: the Session-12 GPU/hardware rename applied consistently across remaining files, tests, and Javadoc. *(unchanged)*

**Session 13** — Tensor parallelism added as a second parallelism mode (`--ptype tensor`, alongside pipeline). Star/coordinator-centric AllReduce. Follow-up fixes: `ClusterHarness.startTensorParallel()` no longer hardcodes TinyLlama constants; fixed an eager `float[]` OOM in tensor-parallel mode; added tensor-parallel checks 7–8 to `ModelLiveRunner`.

**Session 12** — Renamed the GPU/hardware-facing classes for clarity ahead of the ROCm work (`GpuMatVec` → `MatVec`, etc.); removed the now-redundant `GpuForwardPassHandler`; kept `ForwardPassHandler` and `TransformerHandler` as separate interfaces by design (routing vs. math).

**Session 11** — Phi-3 family support: `Phi3TransformerHandler` (Phi-3's attention/FFN shape differs from LLaMA's), `ForwardPassHandlerLoader` routes by GGUF architecture metadata, lazy dequantization via `GgufReader.QuantizedTensor`/`tensorRaw()`, a vocab-size fix in `LlamaConfig`, and a chat-template routing fix so `ChatTemplate.forModelType()` resolves by exact match before falling back to substring match.

**Session 10** — GPU acceleration layer: `MatVec` interface abstracts the matmul backend, with `CpuMatVec` (existing parallel path) and `CudaMatVec` (cuBLAS via `org.bytedeco`) implementations; `DeviceFloatMatrix` uploads weights to VRAM once; `GpuContext` owns the cuBLAS handle lifecycle; `CudaAvailability` does safe, cached CUDA runtime detection so CPU-only machines don't crash on class load. GPU tests gated behind a `-Pgpu` Maven profile and `@Tag("gpu")` to keep default CI GPU-free.

**Session 9** — Multi-turn session KV cache reuse. Previously every REPL turn re-prefilled the entire conversation from scratch (latency grew turn over turn: 23s → 75s on TinyLlama). Added a stable `sessionId`-keyed cache path (`InferenceRequest.ofSession()`, `GenerationLoop.kvCacheKey()`) so only new tokens are prefilled. Turn latency is now flat (~7-8s) regardless of conversation length.

**Session 8** — Cross-platform launcher: unified `juno` / `juno.bat` entry points delegating to `scripts/run.sh` / `scripts/run.bat`, with `console`, `cluster`, and `live` subcommands and shared JVM flags (`--enable-preview`, `--enable-native-access`, ZGC/G1 tuning). Added `logback.xml` to `juno-player` and `integration` to silence default DEBUG-level gRPC/Netty logging. Fixed all 6 `ModelLiveRunner` checks (greeting-vocabulary coverage, raw SentencePiece marker leakage, determinism assertions).

**Session 7** — Fixed broken multi-turn REPL output. Root cause: `GenerationLoop` evicted KV cache after every turn while still hitting the prefix cache, serving stale/freed KV. Interim fix: disable prefix cache for the single-request path and always re-prefill (correct but O(N), later superseded by Session 9's proper fix). Also fixed chat-template selection for TinyLlama via `ChatModelType.fromPath()`.

**Session 6** — Performance and correctness: parallelized `matVec()` in `LlamaTransformerHandler`, parallelized shard loading across nodes, suppressed the EOS token's decoded piece from streamed output, and switched the default activation dtype to FLOAT16 with JVM tuning flags in `scripts/run.sh`.

**Session 5** — Three correctness bugs found during real-model verification with TinyLlama, all fixed: (1) missing `"tinyllama"` chat-template registration caused ChatML tokens to be sent to a Zephyr-trained model, producing garbage output; (2) `decodeToken()` leaked the raw SentencePiece `▁` space-marker character into streamed output (batch `decode()` was already correct); (3) `Q6_K` dequantization used a flat loop that indexed the wrong `qh` byte for block positions ≥32, corrupting the majority of every Q6_K-quantized weight tensor. Added a golden-value regression test (`GgufReaderTest`) built against synthetic in-memory GGUF files.

**Session 4** — Real model inference wired end-to-end for the first time: `GgufReader` (pure-Java GGUF v2/v3 parser, no JNI, dequantizes Q4_0/Q4_K/Q6_K/Q8_0/etc.), `LlamaConfig` (hyperparameters from GGUF metadata), `LlamaTransformerHandler` (LLaMA-family forward pass: RMSNorm, RoPE, GQA attention, SwiGLU FFN, residual connections), and `GgufTokenizer` (SentencePiece BPE reading its vocabulary straight from GGUF metadata, no external `tokenizer.model` file). Also split prefill and decode into two explicit phases in `GenerationLoop`, matching standard LLM inference practice.

---

## 13. Build Status (snapshot, session 15)

All modules built SUCCESS on JDK 25: `api`, `registry` (11 classes), `tokenizer` (9), `sampler` (9), `kvcache` (8), `health` (6), `node` (26, largest module — transformer handlers, GGUF codec, GPU matmul backends, LoRA), `coordinator` (14), `juno-player` (6 main classes), and the `integration` test module (`ModelLiveRunner`, `InProcessClusterIT`, `ThreeNodeClusterIT`, `TensorParallelClusterIT`, `GpuForwardPassIT`). Roughly 475 `@Test` methods total, 0 failures, 0 errors.

## 12. Technology Summary

Java 25, multi-module Maven build. GPU compute via `org.bytedeco` CUDA bindings (cudart + cuBLAS); distributed state and leader election via Hazelcast (`CP FencedLock`); node-to-node data plane over gRPC/Protobuf; RDMA networking via jVerbs; concurrency via Java 25 virtual threads and `CompletableFuture`; REST API served by Javalin (deliberately not Spring Boot), spec generated from OpenAPI 3.0; KV cache is two-tier (GPU VRAM, then Caffeine on JVM heap, no disk tier); circuit breaking via Resilience4j; metrics via Micrometer/Prometheus behind the JDK's built-in `HttpServer`; tokenizer via DJL SentencePiece JNI (with a pure-Java `GgufTokenizer` fallback); weights in GGUF format; sampler is pure Java with zero external dependencies.

## 11. Full Configuration Reference

Original YAML configuration skeleton covering `cluster` (seed nodes, backup count), `coordinator` (ports, queue depth, batch size, preemption strategy), `scheduler` (max wait, priority weights), `node` (gRPC port, VRAM headroom), `kv-cache` (GPU/CPU tier capacity and eviction policy — no disk tier), `health` (probe interval, VRAM warning/critical thresholds, circuit breaker parameters), and `sampling` (default and named profiles: `deterministic`, `creative`). Current authoritative flag/config reference lives in the CLI Reference part of `juno-documentation`.

## 10. Full Token Generation Data Flow

End-to-end trace of a single request: tokenizer encodes the chat-templated prompt; the prefix cache is checked for a reusable KV prefix; the pipeline prefills all prompt positions except the last; the decode loop then calls `pipeline.forward()` once per step, samples the next token, streams it to the client (SSE/gRPC), and repeats until EOS or `maxTokens`; on completion the future resolves and (for sessions) the prefix is cached rather than evicted.

## 9. Activation Compression and Integration Test Infrastructure

**Activation compression:** pipeline-parallel node hops ship full activation tensors over the network; at 70B scale (hidden_dim 8192, seq_len 4096) that's 64MB per hop in FLOAT32. Added an `ActivationDtype` field (FLOAT32/FLOAT16/INT8) negotiated per request so hops can trade precision for bandwidth (FLOAT16 halves the payload, INT8 quarters it), and so heterogeneous nodes (different VRAM budgets) can each request the precision they can afford — this is the quantization-aware sharding mechanism, complementing the GGUF file's own per-layer weight quantization. Implemented in `node/ActivationDtype.java` and `node/ActivationCodec.java` (manual IEEE-754 half-float bit manipulation, no JNI).

**Integration test infrastructure:** `InProcessClusterIT` (zero network, in-JVM stub pipeline, ~250ms) and `ThreeNodeClusterIT` (forks 3 real `NodeMain` JVMs, real gRPC, ~16GB memory budget) exercise the cluster end to end. `ModelLiveRunnerIT` runs 6 real-model checks (greeting response, no raw SentencePiece markers, question answering, greedy determinism, multi-turn conversation, FLOAT16 parity) against an actual GGUF file, disabled by default and activated with `-Pintegration`. `LoadShardsParallelTest` is the timing regression anchor proving shard loading happens in parallel, not serially.

## 8. Actors — Design Decisions

*Status note: this section records the original design, not the current code. The Hazelcast-backed pieces it describes (the distributed `IMap` model registry, the weighted seed-node election, and leader/standby coordinator election with `CP FencedLock`) were never implemented: no source file uses a Hazelcast API, the `RegistryService` RPCs in `inference.proto` had no implementation, and the Hazelcast dependency was declared but unused (the RPCs and the dependency have since been removed). `FaultTolerantPipeline` exists and is unit-tested, but the production cluster launch path does not construct it, so a lost node currently fails the request instead of failing over. The sampler order quoted below is also out of date; the current order is presence penalty, repetition penalty, temperature, top-k, softmax, top-p, sample (see `Sampler`).*

Model registry and shard planning live in a Hazelcast distributed `IMap` (no single point of failure); seed-node election uses an IMQ-inspired weighted score (connectivity, stability, betweenness centrality, VRAM); sharding is greedy and VRAM-aware but capped per node so a single large-VRAM node can't starve later nodes of layers (`ShardPlanner`'s fairness cap). The coordinator uses static micro-batching (configurable window/size, default 8 requests / 50ms), a `PriorityBlockingQueue` (HIGH/NORMAL/LOW), and Java virtual threads throughout; `FaultTolerantPipeline` wraps each node in its own circuit breaker with a configurable retry policy (`none`/`once`/`aggressive`) and reports `CIRCUIT_OPEN` or `RETRIES_EXHAUSTED` as HTTP 503 with a `Retry-After` hint. The tokenizer supports LLaMA/TinyLlama/Mistral/Gemma chat templates by model-id lookup, defaulting to ChatML. The sampler is a pure-Java pipeline: temperature → top-k → top-p → softmax → repetition penalty → sample.

## 7. REST / HTTP — Revised Design

Deliberately not Spring Boot — too heavy for the target footprint. REST is served by Javalin (built on Jetty, ~1MB, explicit routing, no annotation magic, a good fit for virtual threads); the metrics endpoint uses the JDK's built-in `HttpServer` plus Micrometer, avoiding any extra framework dependency for a single `/metrics` scrape endpoint.

## 6. KV Cache — Revised Design

The original three-tier design (GPU + off-heap + disk) was simplified to two tiers, RAM only, no disk IO ever: GPU VRAM (hot, active sequences) and JVM heap via Caffeine (warm sequences, W-TinyLFU eviction). The off-heap and disk-backed candidates (OHC, Ehcache 3, Chronicle Map) were all dropped for the same reason: dead or JAXB-transitive-dependency-poisoned Maven artifacts that don't build cleanly on JDK 25. The prefix cache (a trie, checked before every forward pass) is unchanged — its purpose is to let concurrent clients sharing a system prompt pay for that prefix's compute once.

## 5. System Architecture

Clients reach the cluster via REST (Javalin) or streaming gRPC, behind a load balancer, to one of two coordinators (leader/standby, Hazelcast `CP FencedLock` for leader election). The leader owns tokenization, request scheduling, the autoregressive generation loop, sampling, and the prefix cache, and drives an `InferencePipeline` that fans out over the node cluster: gRPC carries the data plane (activations), Hazelcast carries the control plane (commands, state, health events). Each GPU node owns a contiguous slice of transformer layers; the first node also owns the embedding table, the last node owns the output projection.

## 4. API Module — What Was Built

Client-facing REST surface (`POST /v1/inference`, `/v1/inference/stream` for SSE, model load/list/status/unload, cluster health/nodes/shardmap) generated from an OpenAPI 3.0 spec via `openapi-generator-maven-plugin` (jaxrs-spec mode). Internal node-to-node communication is a separate gRPC surface (`inference.proto`, never exposed to clients) with three services: `InferenceService` (client-facing), `NodeService` (coordinator-to-node forward pass/shard load/unload), and `RegistryService` (internal shard-map queries).

## 3. Maven Project Structure

Multi-module Maven build, JDK 25 throughout, group `cab.ml.juno`, artifact `juno`. Shared libraries: `api` (OpenAPI + gRPC + proto codegen), `registry`, `tokenizer`, `sampler`, `kvcache`, `health`, `lora`. Core engine: `node` (GGUF parsing, quantization codecs, CPU/CUDA/ROCm matmul backends, transformer handlers). Orchestration: `coordinator` (scheduling, generation loop, REST) and `juno-player` (the REPL and cluster harness). Executables: `juno-master` (`CoordinatorMain`) and `juno-node` (`NodeMain`). Key dependency versions locked early and unchanged since: Hazelcast 5.4.0, gRPC 1.63.0, Protobuf 3.25.3, `org.bytedeco` CUDA 12.6-9.5-1.5.11, DJL 0.27.0, Caffeine 3.1.8 (the only cache library — see §6), Resilience4j 2.2.0, Micrometer 1.13.0, Javalin 6.3.0 (explicitly not Spring Boot), JUnit 5.10.2. Spring Boot, OHC, Ehcache, and Chronicle Map were all evaluated and removed early for transitive dependency failures (dead repos, JAXB dependency chains).

## 2. Hardware Stack

Reference cluster: 16 commodity PCs, each with a 4GB-VRAM GPU (64GB total VRAM across the cluster), an 8+ core CPU, 16-32GB RAM for the KV cache JVM heap, and NVMe storage for fast shard loading. Networking: 10GbE to start (25GbE ideal), a managed switch with jumbo frames, RDMA (GPU-to-wire, bypassing the CPU). Total extra networking cost for 16 machines: roughly $800-1000 — far cheaper than a single 64GB GPU.

## 1. Vision

A fully Java-native distributed LLM inference engine that runs large language models across a cluster of commodity GPUs, replacing the need for a single expensive high-VRAM card with a network of affordable machines. Core philosophy: no Python, no GIL, real threads; no Spring Boot, no framework bloat; commodity hardware over premium hardware; Java distributed tooling (Hazelcast, gRPC) over NCCL/MPI; pipeline parallelism (LAN-friendly, no InfiniBand required); open source, Java ecosystem first.

```
  juno — Distributed Java LLM Inference Engine
  Full Architecture Design Document
  JDK 25 · Maven · Java-native · Commodity GPU Cluster
```
