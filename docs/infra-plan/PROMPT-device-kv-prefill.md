# PROMPT: quantized device KV, prefill path (Tier 03 step 6, item 7, owner decision 6 (a))

Copy everything below the line into a new agent session.

---

You are a senior performance engineer in `/home/medion/Repo/juno` (Java 25, Maven multi-module, CUDA/ROCm via Panama
FFI). Read `CLAUDE.md` first and follow it exactly. The governing plan is `docs/gap-closure-plan/`; the active tier is
`TIER-03-kv-cache-maturity.md`, step 6, item 7 (the GPU's KV copy stored in the `--cache-type-k/-v` types).

## Task

Owner decision 6 (a), 2026-11-15: make a 2,048-token **prefill** with a `q8_0` device KV take **<= 1.10x** the FP16 device
KV's prefill time on the four sweep models (TinyLlama 1.1B, Qwen2.5-3B, Phi-3.5-mini, Mistral 7B), scored pinned by the
owner, with the parity and quality tests unchanged. Then report and stop (README execution rules; the final report format
is the one the previous sessions used: tier and step, plan-versus-code drift, files changed, each test and command with
pass/fail, criteria, decisions with recommendations, anything owed to the owner with the exact command, next step, the
uncommitted files, and a commit message under 200 characters).

## Why this exists

The decode bar is met (owner re-run 2026-11-15, `docs/perf-compare/20261115T200324Z-device-kv-tg-gate-rerun/`: `q8_0` decode
1.030x to 1.235x FP16), criterion 7 is ticked (6 of 15), and the kit probes are in
(`docs/perf-compare/20261115T232030Z-kit-device-kv-probes/`). The same gate logs showed the prefill cost: `q8_0` prefill
**1.31x** (TinyLlama), **1.21x** (Qwen2.5-3B), **1.31x** (Mistral 7B) the FP16 time, the same in the first gate run, so it
comes from item 7's original quantized prefill path, not the decode rework. Phi-3.5-mini's handler writes no per-layer
prefill records, so it has no reading yet. The owner chose to fix it inside item 7 (decision 6 (a)).

## Already done (do not redo)

- Device KV in Q8_0 (34-byte) / Q4_0 (18-byte) blocks, byte-identical to the host codecs. Quantized attention is a
  separate PTX module: `node/src/main/cuda/gqa_attention_kv.cu` -> `node/src/main/resources/cab/ml/juno/node/gqa_attention_kv.ptx`,
  launched by `node/src/main/java/cab/ml/juno/node/GqaAttentionKvKernel.java` (`launch(...)`: decode rows,
  `rowsPerBlock == 1 && width <= 128`, go to the wide decode entries `gqa_decode_kv_d64/_d128`; everything else, prefill
  windows included, runs the tiled entries `gqa_attention_kv_*`).
- Shared templates: `node/src/main/cuda/gqa_attention.cuh` (the tiled kernel, its tile staging - search "Stages one tile" -
  and `gqa_decode_kv<DMAX, KT, VT, SLOTS>` / `gqa_decode_kv_dispatch<DMAX, SLOTS>`); formats in
  `node/src/main/cuda/kv_format.cuh` (`kv_encode_block_warp`, `kv_dot_chunk`, `kv_axpy_chunk`, `kv_store_scale`).
- KV write at prefill width: `kv_encode_kv_rows` in `node/src/main/cuda/prefill_window.cu` (K and V in one launch, one
  warp per 32-value block), called from `DeviceKvCache.writeWindowOnDevice` through `PrefillWindowKernels.encodeKvRows`.
  It was 2.7x a launch slower than the FP16 cast before; now cheap, so the prefill gap is most likely the tiled attention
  kernel's dequantizing read, not the write. Measure before assuming.
- The FP16 kernels are untouched: `gqa_attention.ptx` must stay byte for byte the committed one (decision 5, the FP16
  wide decode block, is a separate later step). In `prefill_window.ptx` and `batched_decode.ptx` only the encoder entries
  differ from HEAD.
- Owner decisions on record (Tier 03, 2026-11-15 entry): 1 (a) item 6 runs before step 7; 3 (a) rotated keys deferred;
  4 (a) FFN GEMV placement change outside item 7 (rule 9); 5 (a) FP16 wide decode block outside item 7 (rule 9);
  6 (a) this task. Order of work: this task; decision 5; decision 4; item 6; step 7.

## Read first

1. `docs/gap-closure-plan/TIER-03-kv-cache-maturity.md`: the status paragraph, item 7 in Scope, criterion 7, and the
   2026-11-13, 2026-11-14 and both 2026-11-15 entries.
2. `docs/perf-compare/20261115T200324Z-device-kv-tg-gate-rerun/INDEX.md` (section 4, prefill) and `prefill-from-logs.txt`.
3. `docs/perf-compare/20261115T045828Z-device-kv-decode-path/INDEX.md`: how the decode costs were found (nsys traces,
   one variable at a time) and fixed.
4. `node/src/main/cuda/gqa_attention.cuh`, `gqa_attention_kv.cu`, `kv_format.cuh`, `GqaAttentionKvKernel.java`.

## Steps

1. Run `scripts/performance-tests/check-plan-thresholds.sh`.
2. Reading the prefill time: `DeviceKvDecodeTgLiveTest` logs per-layer prefill records
   (`[prefill] layer i/n done in X,Yms requestId=dkvtg-<k>`); `dist/device-kv-tg/prefill_table.py <run dir>...` sums them per
   request (median of the measured requests dkvtg-0..2; dkvtg--1 is the warm-up). One JVM per model and type:
   `mvn -o test -pl node -am -Dgroups=gpu -Dtest=DeviceKvDecodeTgLiveTest -Dsurefire.failIfNoSpecifiedTests=false
   -Djuno.test.deviceKvTg=true -Djuno.test.deviceKvTg.models=<tinyllama|qwen25|phi35|mistral> -Djuno.test.deviceKvTg.type=<f16|q8_0>`.
   Phi-3.5-mini needs another reading (its handler has no per-layer records): time the prefill in the test itself, or
   trace it.
3. Find where the time goes with `nsys` (CUDA trace works on this host; `nvprof` counters do not: error 4190:27, admin
   restricted). Compare the tiled kernel's per-launch time `q8_0` against FP16 at each window, and the encoder's.
4. Tests first for any kernel change: the parity tests must hold unchanged and a planted defect must fail them -
   `DeviceKvQuantParityTest` (`decodeStreams_matchTheCpu`, `slidingWindow_matchesTheCpu`,
   `quantizedKernelOverFp16_isTheFp16Kernel`, `deviceWindowWrite_isTheHostEncoding`, `batchAppend_isTheHostEncoding`),
   `DeviceKvFormatTest`, `ResidentBatchDecodeTest` (a decode-batch row bit for bit its single decode),
   `DeviceKvQuantQualityLiveTest` (opt-in: `-Djuno.test.deviceKvQuality=true -Djuno.test.deviceKvQuality.models=<id>
   -Djuno.test.deviceKvQuality.type=<f16|q8_0|q8_0/q4_0>`, the f16 run first; one model per JVM). Quality bars: greedy
   agreement >= 99% at `q8_0`, >= 97% at `q8_0`/`q4_0`.
5. Regenerate PTX: `cd node/src/main/cuda && nvcc -ptx -arch=compute_61 -O3 -o ../resources/cab/ml/juno/node/gqa_attention_kv.ptx gqa_attention_kv.cu`
   (each `.cu` names its own line). Check: no spills (`ptxas -v` on a cubin build), `gqa_attention.ptx` byte for byte
   HEAD's, only the entries you meant to change differ.
6. Unpinned A/B (f16 and `q8_0` JVMs alternated), publish under `docs/perf-compare/<UTC>-device-kv-prefill/` with an
   INDEX.md and a README row; then prepare the owner's pinned gate under `dist/<short-name>/` (copy
   `dist/device-kv-tg/run-gate.sh`, `wait-and-run.sh`, `tree-hash.sh`; score the prefill time instead of tg; READY written
   LAST from `tree-hash.sh` after the full unit suite passes). Never fake a pinned result.
7. Full unit suite with GPU groups, then the record: Tier 03 entry, `docs/agent-arch.txt`, `docs/howto.md`
   (`--cache-type-k` row: the prefill figure), `docs/performance.md`, `CHANGELOG.md`.

## Constraints carried over

- Git is read-only (status/log/diff/show/branch only). No Claude or Anthropic attribution anywhere. No competitor product
  names outside `docs/gap-closure-plan/`, `docs/infra-plan/`, `docs/perf-compare/`. No tier numbers in code, javadoc, CLI
  help, shipped docs or file names. No emojis. Production docs never link into `docs/gap-closure-plan/`.
- Prefer a new class to extending one; KISS; no allocation or boxing on hot paths; new flags fail closed.
- Never run mvn package/compile while a test, IT or gate runs; one GPU workload at a time; earlier builds via `git archive`
  into the scratchpad, never a worktree. Never read model files; never change `models/`.
- `mvn test` on a module after `node` needs `-am` (else a stale `~/.m2` node jar), with a test filter.
- Known environmental failures on this host (record with evidence, never loosen): `PrefillReserveDeviceTest` (driver
  holdback, 87.5 MiB against 64 MiB), `DeviceUploadOrderingTest` (intermittent upload race, passes alone),
  `ProbeRunnerTest` (NOISY when other processes use the CPU), `ResidentQkvPathTest` free-memory read (passes alone).
- The GTX 1080 accepts `nvidia-smi -lgc` with exit 0 and does not apply it: GPU clocks are recorded, not fixed; the gate's
  CLOCK line prints "lock NOT applied" (`dist/device-kv-tg/run-gate.sh`). Compare sides at matched recorded clocks.
- A row in a decode batch must stay bit for bit its single decode; the device KV bytes must stay the host codec's.

## Exit checklist

- [ ] `q8_0` prefill at 2,048 tokens <= 1.10x FP16 on all four sweep models, unpinned evidence published, the owner's
      pinned gate prepared (READY last) or its result recorded.
- [ ] Parity, format, batch and quality tests pass unchanged; a planted defect shown failing them; PTX checked.
- [ ] Decode not regressed (re-read the decode tg A/B on Qwen2.5-3B at least).
- [ ] Full unit suite run, failures re-run alone and explained.
- [ ] Tier 03 record, docs and README row updated; `check-plan-thresholds.sh` passes; scans clean.
- [ ] Final report with the uncommitted files and a commit message under 200 characters; no commit.
