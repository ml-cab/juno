# Resume prompt: CPU decode and prefill kernels (gap-closure plan, current tier)

Written 2026-10-12 at a checkpoint. The tier file `docs/gap-closure-plan/TIER-02E-cpu-decode-kernels.md` holds the
full execution record; this file is the short version for the next session. Read CLAUDE.md, the plan README's
execution rules, and the tier file's execution record (all dated 2026-10-11 and 2026-10-12 entries) first.

## Done and verified

- Steps 1 to 3 closed: plan check; pinned CPU sweeps at 512 and 2048 published
  (`docs/perf-compare/20261012T045931Z`, `20261012T071213Z`, `20261012T162344Z`,
  `20261012T205149Z-cpu-prefill-length`); pinned kernel baseline (`20261012T225531Z-cpu-kernel-baseline`: Q4_K
  1.52, Q6_K 5.41 cycles per weight).
- Owner decisions 1 to 3: host prefill attention joins as item 6; kernel bounds scored at 1 and 12 threads, Q6_K
  restated to `<= 2.7`; prefill matmul tile joins as item 7.
- Item 1 (vector Q6_K decode kernel, `CpuIntegerKernels.q6kRow` / `q6kPass`): bit-identical, about 2.3 to 2.4
  cycles per weight unpinned. Tests: `CpuQ6kKernelParityTest`.
- Item 2 (Q4_K): candidate 3 adopted (packed scales, float-lane accumulation), about 1.30 cycles per weight
  unpinned; still above `<= 1.1`. Candidates 1, 2, 2b, 4 rejected with numbers. Tests: `CpuQ4kKernelParityTest`
  (bound 1e-6 of term magnitude).
- Item 6 (pooled prefill attention, `GqaMath.attendPooled`, wired into every handler's window helper):
  bit-identical. Tests: `GqaMathPooledTest`.
- Item 7 (gemm tile): G1, G2, G3 adopted (four activation rows per pass, vectorised offset term).
  `CpuIntegerGemmTest` holds it to the exact reference.
- Unpinned end to end (not published): CPU tg up 36% to 47%; CPU pp at 2048 about 2x; pp ratio at 512 about
  0.27x to 0.28x against `>= 0.30x`.
- `mvn test -pl node`: 1,129 tests pass, but that run predates item 7, so it is owed again.

## Next, in order

1. Done 2026-10-12: `mvn test -pl node` green after G3 (1,125 tests). Full unit suite green after G4
   (2,372 tests, 0 failures).
2. Done 2026-10-12: item 7 G4, vectorised weight `unpack`, adopted (exact; about -8% Q4_K, -7% Q6_K gemm,
   unpinned). Test: `CpuIntegerGemmUnpackParityTest`.
3. Done 2026-10-13: item 2 candidate 5 (activation split once per call into `Q8Activation.xe`/`xo`) adopted,
   bit-identical; Q4_K about 0.91 cycles per weight on one thread, 1.06 per core at 12 (unpinned). Test:
   `CpuQ4kActivationSplitParityTest`. Optional, unmeasured: the same split for `q6kPass`.
4. Done 2026-10-13: item 3. `CpuIntegerWidth` (auto from `ByteVector.SPECIES_PREFERRED`, capped at 256;
   `-Djuno.cpu.integerWidth=128|256|auto`, other values fail closed), `CpuIntegerKernels256`, `CpuIntegerGemm256`;
   128-bit classes unchanged as the oracle. Tests: `CpuIntegerWidthTest`, `CpuIntegerWidthParityTest`. Width row
   `NEEDS-AVX2-HARDWARE`. Owner decision open: whether `auto` may pick the unmeasured 256-bit path on AVX2 hosts
   (recommended: resolve `auto` to 128 until measured).
5. Done 2026-10-13: item 6's second candidate. `GqaVectorHead` (A1: value sums in 128-bit registers across keys;
   A2: dot in vector lanes, held to `GqaMathOracleTest`'s bound) adopted; tests `GqaVectorHeadTest`,
   `GqaMathPooledTest` (serial pooled head bit-identical). Finding recorded: the gemm tile sometimes runs through
   `ShortVector.lanewiseTemplate` (3 of 28 prefills). Optional: the decode-only attention sites
   (`LlamaTransformerHandler` decode step, `Phi3TransformerHandler` decode step, `Qwen3TransformerHandler.gqa`),
   pooled only if measured faster.
6. Smoke script `scripts/performance-tests/smoke-cpu-kernels.sh` written 2026-10-13 (wraps
   `smoke-cpu-hot-path.sh` on both jars, compares greedy replies; not yet run). Then `ModelQualityNllLiveTest`
   on the four sweep models; `ModelLiveRunnerIT`; re-run the earlier smoke scripts unmodified.
7. Closing gate (owner run, pinned): drafted 2026-10-13 in `dist/cpu-kernels-close/` (`wait-and-run.sh`,
   `run-gate.sh`; parts K, S, C, W, A, L; no jars, no baseline tree, no READY yet). Owner decision pending: Part W
   (the 2048-token same-hour A/B, 13 to 14 hours) or read the 2048 host-attention gate from the closing sweeps
   against the step 2 sweeps. Earlier wording, kept: prepare `dist/<name>/` with `wait-and-run.sh`, `run-gate.sh` and READY,
   modelled on `dist/cpu-prefill-length/` and `dist/cpu-kernel-baseline/`. It covers: the kernel microbenchmark
   (pinned), closing sweeps at 128, 512 and 2048, the same-hour A/B against the pre-tier jar `b9a0440b3a1b597a`
   (CPU and GPU lanes, allocation and GC), and the LoRA and vision gates.
8. Docs (`docs/howto.md`, `docs/agent-arch.txt`, `docs/performance.md`), `CHANGELOG.md`, the README CPU rows
   restated (item 5), and the exit criteria.

## Practical notes

- Scratch probes: `<scratchpad>/harn/cab/ml/juno/node/` (`Q6Probe`, `Q4Probe`, `Q4Cand`, `GemmProbe`). Compile them
  with `javac --add-modules jdk.incubator.vector -cp node/target/classes[:node/target/test-classes]` and run with the
  same classpath. They are not part of the repo, so a new session has to rewrite them.
- C2 boxes a Vector API value passed to a helper method it does not inline. Write lane operations in place and
  check allocation with `ThreadMXBean` before trusting a timing.
- A/B jars: `dist/q6k-ab/` (`baseline` = pre-tier `b9a0440b3a1b597a`).
- Git is read-only; nothing is committed.

## State at 2026-10-13 18:03 (resume here)

- The owner's pinned closing gate is RUNNING: `dist/cpu-kernels-close/run-gate.sh`, `PART=KSCAL`, started 18:02:39,
  log `dist/cpu-kernels-close/gate-20261013T180239.log`, results under `dist/cpu-kernels-close/runs/`; Part S publishes
  its sweeps to `docs/perf-compare/` itself. Do not build, test or run anything CPU or GPU heavy until it ends (`pgrep
  -f run-gate.sh` empty). Jars: baseline `b9a0440b3a1b597a`, candidate `e37e16c004f4c213`.
- After it ends: read the log's tables (Parts K, S, C, A) and the LoRA JSONs (Part L: train time per pass B/A >= 0.95,
  playback wall-clock tps B/A >= 0.80); publish the kernel microbenchmark (Part K), the A/B tables (C, A) and the LoRA
  result under `docs/perf-compare/<UTC>-<name>/` with INDEX.md, add rows to `docs/perf-compare/README.md`, scrub
  local paths; restate the README CPU rows (item 5); tick the exit criteria the gate meets, citing the directories.
- Then the deferred checks: `mvn test -pl node -Dtest=ModelQualityNllLiveTest -Djuno.test.modelQuality=true` (about 3 h
  in Tier 02C), `scripts/performance-tests/smoke-cpu-kernels.sh --baseline-jar dist/q6k-ab/baseline-shaded.jar`, the
  earlier smokes unmodified, `compare-vision.sh` and LoRA playback unpinned (median of three, both builds); then docs
  (`docs/howto.md`, `docs/agent-arch.txt`, `docs/performance.md`) and `CHANGELOG.md`.
