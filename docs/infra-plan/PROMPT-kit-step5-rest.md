# Handoff: hardware validation kit, step 5 (remaining probe groups)

Continue `docs/gap-closure-plan/TIER-02F-hardware-validation-kit.md`, implementation step 5. Read that tier
file's execution record first: "2026-10-28: step 5, part 1" lists what is done, the findings, and what is owed.
Follow CLAUDE.md and the standing operating rules (git read-only, no attribution lines, no competitor names
outside the plan trees, no tier numbers in code or file names, tests first, never build while a test run or
probe run is in progress, the final report standalone).

## State at handoff (2026-10-28)

Done in part 1, each with a self-test in `ProbeSelfTests` or a unit test, all in `juno-kit`:

- `cpu.unit-tests` (`UnitTestsProbe` with `excludedGroups=gpu,rocm` over the documented unit-test modules).
- Per-family live checks, CPU and CUDA: `cpu.live.<model>`, `gpu.live.<model>` for qwen2.5-3b, phi-3.5-mini,
  mistral-7b, qwen3-1.7b (heaps from the step-4 CPU re-verification table); `Requirements.withDeviceMemory`
  exists but no device minimum is set yet (set it from the measured `ownGpuPeakMiB`).
- Consistency, grammar, tools: `ConsistencyProbe`, `GrammarProbe`, `ToolsProbe`, all through
  `JunoProcess` (the launcher, `cab.ml.juno.player.ConsoleMain`, in a child JVM; stopped by typing `exit`).
- Context policy and long prompt: `ContextPolicyProbe` (Phi-3.5-mini only: every other family's limit is
  32,768 positions), `LongPromptProbe`.
- GPU residency: `GpuResidencyProbe` now runs the cluster part (pipeline and tensor with the region on).
- CPU kernels: `CpuHotPathProbe` (text part of the hot-path smoke; the two-build kernel smoke becomes the
  `ingest` comparison of recorded replies).
- Quality: `QualityProbe` + `QualityRunMain` (one JVM per kernel setting; text copied to kit resources).
- Throughput: the remaining sweep-model and length combinations, CPU and GPU (catalog entries only).

Run on real models and passing: the consistency (GPU), grammar, tools, GPU context-policy and long-prompt
(TinyLlama) probes, `gpu.residency.tinyllama` with its cluster part, `cpu.hot-path.tinyllama-1.1b`,
`quality.tinyllama-1.1b`. Failing: `consistency.tinyllama` (explained below; not a KV leak). Probe ids made by the catalog
helpers use the catalog model id (`cpu.hot-path.tinyllama-1.1b`, `quality.qwen2.5-3b`).
Not yet run on real models: the other three `cpu.hot-path.*` and `quality.*`, `cpu.unit-tests`, the per-family `cpu.live.*`/`gpu.live.*`,
`context-policy.phi-3.5-mini` (CPU), `long-prompt.mistral-7b`, `gpu.long-prompt.mistral-7b`, and the new
throughput entries. Their reference costs are guesses; replace each with its measured wall time.

## Progress in part 2 (2026-10-30)

Items 1 to 4 below are coded, each with a self-test in `ProbeSelfTests`; `juno-kit` 91 tests, 0 failures.

- New probes:
  - `ParallelProbe` (`parallel.tinyllama`, `gpu.parallel.tinyllama`)
  - `ScheduleProbe` (`schedule.*`)
  - `MixedPrefillProbe` (`mixed-prefill.*`)
  - `PrefillBatchProbe` (`prefill-batch.*`)
  - `LoraProbe` (`lora.tinyllama`, `gpu.lora.tinyllama`; trains in its own JVM, heap 4g)
  - `VisionProbe` (`vision.moondream2`, `gpu.vision.moondream2`)
  - `LauncherProbe` (`launcher.tinyllama`, quick suite)
- New helpers: `ConcurrentChats`, `StreamTiming`.
- `JunoProcess` gained:
  - `serve` with JVM properties, and `serveCommand` (a launcher script);
  - `postImage` (multipart), `metrics()` (the launcher's `--jfr` summary), `tree()`, and `stop()` (exit code).
- Not seen failing first: the LoRA self-test. It was written before the probe but not run in between.
- The vision self-test runs against a text-only synthetic model (no synthetic vision model exists). Every caption
  check must fail there with HTTP 404.
- The launcher self-test skips, saying why, without juno-player's shaded jar.
- Reference costs of the new entries are estimates until the real-model runs.

Run 1 (2026-10-30, `juno-kit-results/juno-kit-results-linux-amd64-20261030T045842Z.zip`, 1 h 16 min): all 15 probes
PASS or DONE, `gpu.parallel.tinyllama` NOISY (`--parallel 8` spread 17% to 61% over three attempts).
- `consistency.tinyllama` PASS on the fixed engine.
- Wall times went into the catalog's reference costs; the calibration read 0.90 ms against the reference 0.9075.
- Vision fixed after run 1, test first (`VisionProbeTest`).
  - "decode t/s" is now the mean single-token pass. It used to be completion tokens over the decode total, and with
    `--prefill single` that total includes every prompt token: 0.19 t/s read where the true rate is 5.45.
  - "prompt t/s" is dropped. The prompt runs through the same single-token pass.
  - A GPU run whose matmuls all ran on the host gives no readings; fact `gpu.deviceMatVecs`. moondream2's phi2 has no
    device path, and the launcher warns about it: `gpu.vision` read 118 MiB and the CPU's latency.
- The LoRA readings match `compare-lora.sh`: CPU 48.9 s per pass and 0.90 t/s playback, GPU 2.8 s and 12.3 t/s.
- GPU finding: `--parallel 8` read 0.74x to 0.92x of `--parallel 1` (111 t/s) for 8 concurrent clients, and P8 gave
  2 distinct replies to 8 identical prompts. Against it: the README table's GPU row reads 1.11x (Sept 2026), and the
  pending docs wording claims a GPU gain. A like-for-like GPU reading (`concurrency-parallel-gpu.sh`, the CPU script
  with `--gpu`, player jar `8fa44602112f19c4` copied to the scratchpad) runs before run 2.

- Done: the GPU concurrency reading, published as `docs/perf-compare/20261030T062550Z-gpu-parallel-concurrency/`
  (README rows added; the tier file's 2026-10-30 section has a GPU bullet).
  - P8 over P1: 0.90x and 0.87x; mean latency 0.86x and 0.93x.
  - Cause: `RequestScheduler.collectBatch` takes only what is already queued after the first request. It does not
    hold the batch open for `--batch-window-ms`, though the docs and `--help` say "batch collect window".
  - GPU batches of 2 and 3 give a reply that differs from the single reply, also on a fresh server, so no leak.
  - Owner decisions to raise: the docs wording (now on both backends), and whether to make the collector hold the
    batch open. Out of tier.
- `juno-kit` 94 tests, 0 failures, before run 2.
- Run 2 started 06:37 UTC 2026-10-30 (scratchpad `run2.log`, list `run2-probes.txt`: everything else owed, plus
  `consistency.tinyllama` again and `gpu.vision.moondream2` to confirm the fix).

- Run 2 (`juno-kit-results-linux-amd64-20261030T063627Z.zip`, 9 h 42 min): 34 of 40 PASS or DONE.
  - PASS: `cpu.unit-tests` (2098 tests, 0 failures, 25 skipped), the four `cpu.live.*`, `gpu.live` on qwen2.5-3b,
    mistral-7b and qwen3-1.7b, `consistency.tinyllama` (second pass on the fixed engine), the Mistral long-prompt
    probes, the three hot-path probes, and the three quality probes (NLL identical to
    `20261027T025856Z-cpu-kernels-quality`).
  - `gpu.vision.moondream2`: no readings, `gpu.deviceMatVecs` 0, which confirms the fix.
  - Failures, all Phi-3.5-mini, all kit sizing:
    - `gpu.live.phi-3.5-mini` check 9, plus both `*.throughput.phi-3.5-mini.2048`: Java heap OOM in
      `DenseKvTensor.grow`. The f16 KV is stored as `float[]`, and 2048 + 64 positions double to a capacity of 4096,
      about 3 GB (no GQA, 3072-wide).
    - Measured live set after GC: 7.5 GB (GPU), 7.2 GB (CPU), so the heap is now 12g.
    - `context-policy.phi-3.5-mini` (CPU) TIMEOUT at 2 h, in its continuous phase; not a hang (CPU prefill about
      7 to 10 t/s, about 80k prompt tokens in all).
- Catalog after run 2, test first (`ProbeCatalogTest`, 4 tests, 3 failed first):
  - measured costs;
  - every timeout at least 3x its cost: quality and throughput helpers use `max(base, 3 * cost)`; long-prompt Mistral
    11600, unit tests 6600;
  - Phi 2048 entries and `gpu.live.phi-3.5-mini` heap 12g;
  - `MEASURED_DEVICE_PEAK_MIB` with minimums rounded up to 512 MiB;
  - CPU context-policy priority 3, cost 12600 and timeout 37800 provisional until run 3 measures it.
- `juno-kit` 98 tests, 0 failures.
- Run 3 started 2026-10-30 about 17:50 UTC (`run3.log`): `gpu.live.phi-3.5-mini`, both Phi 2048 throughput entries,
  `context-policy.phi-3.5-mini`.

- Run 3 (`20261030T164322Z`, 5 h 10 min):
  - `gpu.live.phi-3.5-mini` PASS at 12g, 7596 MiB.
  - Both Phi 2048 throughput entries complete, NOISY: desktop applications (Discord, a browser) used 7% to 14% of
    the CPU.
  - `context-policy.phi-3.5-mini` ERROR after 3 h 24 min: `HttpTimeoutException`. `JunoProcess.post` had a 20-minute
    limit, and the continuous schedule's near-limit request (about 4,000 tokens plus 200 decode) took longer. The
    limit is now `REQUEST_TIMEOUT` = 2 h, also on `postImage`; not unit-tested (a constant).
  - Costs and device peaks from run 3 are set. `juno-kit` 98 tests, 0 failures.
- The tier file's execution record for part 2 is written, except run 4's row (placeholders `RUN4_RESULT`,
  `RUN4_TIME`) and the status line at the top.
- Run 4 started 2026-10-30 about 21:57 UTC (`run4.log`): `context-policy.phi-3.5-mini` alone.

- Run 4 (`20261030T220016Z`, 5 h 39 min): `context-policy.phi-3.5-mini` PASS, all 7 checks; crossing turn 7 on
  both schedules. Phases: static 1 h 46 min, continuous 3 h 36 min, static with shifting on 17 min. Cost set to
  20,300 s, timeout 60,900 s. Tier file row, finding and status line written. `juno-kit` 98 tests, 0 failures.
- Owner decision pending: shorten the probe's conversation (catalog args `turns=3`, `notesPerTurn=85`: the same
  longest prompt, crossing at turn 2; estimate about 3 h on the CPU instead of 5 h 39 min, to be measured).

Step 5 is done. Next: step 6 (coverage file and its test; `ingest`; the reference run on this host, published).
