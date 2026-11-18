# Resume: decision 4 (a), FFN gate and up GEMVs in one launch (written 2026-11-17)

Active plan: `docs/gap-closure-plan/TIER-03-kv-cache-maturity.md` (in progress, 6 of 15 exit criteria). Read its last
two execution-record entries first. Everything below is uncommitted on HEAD `9c33ab93`.

## Done and verified this session

1. The owner's gate `dist/quant-decode-loads` (2026-11-17T05:31Z) recorded: `q8_0` decode over FP16 0.985x / 0.958x /
   0.974x / 0.972x (bar 0.95), published `docs/perf-compare/20261117T053104Z-quant-decode-loads-gate/`; the gated tree
   hash equals the commit `9c33ab93`; exit criterion 7 ticked again; howto, performance.md, CHANGELOG, perf-compare
   README updated.
2. Decision 4 (a), thresholds written before code (record entry "decision 4 (a) begun"):
   - `node/src/main/cuda/kquant_gemv.cuh` (helpers moved out of `q4k_gemv.cu`; `q4k_gemv.ptx` rebuilt byte for byte equal).
   - `node/src/main/cuda/q4k_gemv_pair.cu` and `node/src/main/resources/cab/ml/juno/node/q4k_gemv_pair.ptx`
     (`nvcc -ptx -arch=compute_61 -O3`, 48 registers, no spills).
   - `node/src/main/java/cab/ml/juno/node/Q4KGemvPairKernel.java`; `ResidentLayerTail.gateAndUp` routes to it.
   - `Q4KGemvPairTest` passes (bit for bit two `q4k_gemv` launches, 10 shapes); defects F1 to F3 caught; the resident
     path tests (75) pass on the GPU.
   - Kernel bar met, unpinned: `docs/perf-compare/20261117T082501Z-gemv-gate-up-pair/` (`Q4KGemvPairBench`).
   - `docs/agent-arch.txt` updated.

## Next, in order (check `pgrep -af "run-gate|compare-|mvn"` first; one GPU job at a time)

1. Baseline tree (scratch, never `install`):
   ```
   B=<scratchpad>/9c33ab93; mkdir -p $B && git archive 9c33ab93 | tar -x -C $B
   (cd $B && mvn -q package -DskipTests && mvn -o -q test-compile -pl node -am) && ln -s /home/medion/Repo/juno/models $B/models
   ```
2. Unpinned decode A/B at 2,048 tokens, per model (`tinyllama qwen25 phi35 mistral`) and type (`f16`, `q8_0`),
   alternating trees (A before, B after) twice:
   ```
   mvn -o test -pl node -am -Dgroups=gpu -Dtest=DeviceKvDecodeTgLiveTest -Dsurefire.failIfNoSpecifiedTests=false \
     -Djuno.test.deviceKvTg=true -Djuno.test.deviceKvTg.models=<m> -Djuno.test.deviceKvTg.type=<t>
   ```
   Read the `DEVICE-KV-TG ... median_tg=` lines. Publish with the kernel bench or as its own `perf-compare` entry.
3. Hot-path gate from each tree: `scripts/performance-tests/compare-lora.sh --gpu --skip-build --reps 3` (bars train
   <= 1.25x, playback >= 0.80x; LoRA attends on the CPU, so it shows nothing else moved).
4. Suites on the final tree: `mvn -o test -pl node,coordinator -am -Dgroups='gpu | !gpu' -Dmaven.test.failure.ignore=true`
   (about 56 min), then `vision,metrics,juno-player`. Known environment-sensitive: `DeviceUploadOrderingTest` (upload race),
   `PrefillReserveDeviceTest.theAllocatorWithholdsNoMoreThanTheReservesAllowance` (free VRAM on the shared card); re-run alone.
5. The owner's gate under `dist/gemv-pair/` (git-ignored), modelled on `dist/fp16-wide-decode/` (its `run-gate.sh`,
   `wait-and-run.sh`, `tree-hash.sh`, `baseline-tree`): per model A `f16` baseline tree, B `f16` this tree, C `q8_0`
   baseline, D `q8_0` this tree, three cycles, `DeviceKvDecodeTgLiveTest` at 2,048 tokens, cooled to 65 C, a CLOCK line per
   JVM; RESULT lines B/A and D/C, both bars >= 1.00. Write `READY` (tree hash) last. Owner command:
   `bash dist/gemv-pair/wait-and-run.sh`.
6. Docs: `docs/performance.md` (the FFN pairing and its figures), `CHANGELOG.md` (a Session 149 bullet or a new session),
   the Tier 03 record (results, the gate owed) and its status line. `check-plan-thresholds.sh` must pass.

Then item 6 (exact host KV growth, real FP16 host storage), then step 7 (session save/restore).
