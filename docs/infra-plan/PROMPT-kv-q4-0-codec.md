# Resume prompt: Tier 03 step 6, the Q4_0 KV codec (after the bounded GPU prefill launches)

Written 2026-11-12 at a clean boundary inside Tier 03 step 6, so a fresh session can take the next piece without
re-deriving the last one. The governing plan is `docs/gap-closure-plan/` (README execution rules binding); read
`TIER-03-kv-cache-maturity.md` completely, especially its newest execution-record entry
("2026-11-12: step 6 begun: bounded GPU prefill launches").

## Done and verified (this session, uncommitted)

- Tier 02G item 3, option (a), landed in Tier 03 step 6 under README rule 9: `BoundedLaunches` (new),
  `GqaAttentionKernel.launchBounded`/`registerWidth`, `KQuantGemmKernel.launchBounded` (called by `multiplyHalf`),
  `PrefillWindowRegion.attendInside` and `CudaGqaAttention` route through them; `GpuKernelTimings` times the slowest launch
  of the split, `measureWindow`, `LONGEST_WINDOW`; the kit's `gpu.kernel-timing` adds a 32,768-token check.
- Tests: `BoundedLaunchesTest` 10, `BoundedLaunchParityTest` 9 (GPU, bit for bit; shown failing on planted offset defects),
  `GpuKernelTimingsTest` 4, kit `ProbeSelfTests` (kernel timing), `CoverageTest`, `ProbeCatalogTest` pass.
- Readings published: `docs/perf-compare/20261112T181947Z-bounded-launches/` (kit timing at 2048 and 32,768 tokens; long-prompt
  smoke, Mistral 7B, greedy text identical to the pre-change build at 8,192 tokens, both schedules).
- The full unit suite was run at the end of the session; its result is in the Tier 03 entry (replace nothing there).
- Step 5's uncommitted work (idle KV page reclaim) is also still in the tree, built on.
- Jars (git-ignored): `dist/bounded-launches/baseline-shaded.jar` (`8c0edb793628f2e4`, the tree before the change) and
  `candidate-shaded.jar` (`52ff3d6cbf143534`).

## Next: the Q4_0 KV codec (item 4), then item 7

Item 4 is standalone (no Tier 04 dependency). Thresholds (Tier 03 "Perf gate"): greedy agreement with F16 KV >= 97% of the
first 512 tokens at temperature 0 on every sweep model, teacher-forced mean NLL over a fixed 2,048-token text <= 1.02x F16 KV.

Where things are (verify before relying on them):
- Codecs: `kvcache/.../Q8_0KvCodec.java`, `PagedKvCodec.java`, `KvElementType.java`, `CacheTypeOptions.java` (parses
  `--cache-type-k/-v`), `DenseKvTensor.java`, `KvBlockPool.java` (page byte sizes per type).
- In-place reads (step 3): `kvcache/.../KvRows.java`; `node/.../GqaWindowAttention.java`, `GqaVectorHead.java`, `GqaMath.java`
  decode `Q8_0` per head slice inside the attention loop. Q4_0 needs the same, with no gather and no per-token scratch.
- Refusals to keep consistent: `HandlerKvShare` refuses sharing with a quantized host KV beside GPU attention;
  `NodeKVCacheAdapter`.
- CLI: `scripts/run.sh` (`--cache-type-k`, help text near line 329), `scripts/run.bat`, `ConsoleMain`, `docs/howto.md` flag
  table; the OpenAPI/`juno-api.yaml` only if a field is added.
- Quality pattern: `node/src/test/.../ModelQualityNllLiveTest.java`.

Order (tests first, README rule 3):
1. `check-plan-thresholds.sh`; `git status`; `pgrep -af "run-gate|compare-|smoke-|mvn"` (wait if anything runs).
2. `Q4_0KvCodecTest`: round trip against F16 (block of 32, 18 bytes: FP16 scale + 16 bytes of nibbles), see it fail to compile,
   implement `Q4_0KvCodec`; then `KvRows`/attention oracle cases for Q4_0 (extend `PagedKvAttentionOracleTest`,
   `PagedKvHandlerOracleTest`, `KvRowsTest`) on dense and paged layouts.
3. Wire `q4_0` through `CacheTypeOptions`, `KvElementType`, both layouts, the launchers; fail closed where a surface cannot
   take it (GPU attention reads FP16 device KV until item 7: say so at startup, never silently).
4. KV quality live test (greedy 512, NLL 2,048) on TinyLlama, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B; publish under
   `docs/perf-compare/<UTC>-kv-q4-0/` with an INDEX.md and a README row; scrub local paths.
5. Kit probe (Tier 03 "Kit probes"): mean NLL against F16 KV per sweep model; self-test and `coverage.tsv` line.
6. Record in Tier 03 (execution record, Status line), `CHANGELOG.md` (new session entry), `docs/agent-arch.txt`,
   `docs/howto.md`.

Commands:
- Focused tests: `mvn -q -o test -pl node -am -Dtest='<Class>' -Dsurefire.failIfNoSpecifiedTests=false` (always `-am`: without
  it `node` resolves a stale `kvcache` jar from `~/.m2`).
- Build: `mvn -q -o package -DskipTests` (never `install`); never while a test run is in progress.
- GPU tests run in plain `mvn test` (no group exclusion; they skip without CUDA).
- Smoke at long prompts on Mistral 7B needs `SMOKE_HEAP=16g` (the dense host KV doubles; item 6).

Hard rules: git read-only; no Claude/Anthropic attribution anywhere; no competitor names outside the plan and perf-compare
trees; no tier numbers in code, docs or script names; no emojis.
