# Session save and restore across a process restart

**Purpose.** Evidence that a session saved over `--session-dir` and restored in a later process (a new JVM, a killed
and restarted server, a restarted cluster) continues with the reply the uninterrupted session gives, and starts from
the restored KV (a prefix hit, so only the new turn is prefilled). Correctness readings, not timings: **not pinned**,
and no threshold in them is a speed.

**Build.** The working tree on `d205ab67` with the session save and restore change (uncommitted at the time of the
run). `juno-player-0.1.2-shaded.jar` sha256 `e307dbdc970ad599...`, `juno-kit-0.1.2-shaded.jar` sha256
`196fe2d937e0204d...`. Host: the `docs/perf-compare/README.md` baseline host (Xeon E5-1650 v2, GTX 1080, driver
580.178.04). Model: TinyLlama 1.1B Chat Q4_K_M. Greedy throughout.

## 1. Results

| Reading | What restarts | Backend, KV | Result |
|---|---|---|---|
| Kit probe `session.save-restore.tinyllama-1.1b` ([summary](kit-SUMMARY.txt)) | a new JVM for the restore | CPU, `f16` and `q8_0` | PASS: 63 tokens; 1,421,645 bytes (`f16`), 756,372 bytes (`q8_0`, 0.532x); prefix hit; second turn equal |
| Kit probe `gpu.session.save-restore.tinyllama-1.1b` ([summary](kit-SUMMARY.txt)) | a new JVM for the restore | CUDA (GPU attention, device KV in the cache type), `f16` and `q8_0` | PASS: same counts; prefix hit; second turn equal |
| `ModelLiveRunnerIT` check 11 | a new cluster: three new node processes | pipeline parallel, then tensor parallel; nodes on CUDA | PASS: pipeline 63 tokens / 1,421,644 bytes, tensor 63 tokens / 4,260,283 bytes (every tensor-parallel node saves every layer); second turns equal; 11 of 11 checks |
| `smoke-kv-cache.sh`, part 3 | the server killed (`SIGKILL`), restarted on the `continuous` schedule after a save under `static` | CPU | PASS: restored 63 tokens; prefix hit; second turn equal; 400 without `--session-dir`, 400 for a bad id, 404 for nothing saved |
| `smoke-kv-cache.sh --gpu`, part 3 | as above | CUDA | PASS, every part-3 check |
| `SessionSaveRestoreLiveTest` (`-Djuno.test.sessionLive=true`) | a new server in the same JVM (`JunoPlayer` closed and rebuilt) | CPU `f16`, CPU `q8_0`, CPU `f16` saved `static` restored `continuous`, CUDA `f16`, CUDA `q8_0` | 5 of 5: second turn equal token for token, one prefix hit |
| `KvSessionSnapshotHandlerTest` | a new handler from the same weights | every handler family, every save/restore schedule pair, `f16`/`q8_0`/`f32` | 108 of 108, next decode logits bit for bit |

The `q8_0` file is 0.532x the `f16` one: the rows are stored exactly as the host KV holds them, 34 bytes per 32
values against 64.

## 2. Commands

```bash
./juno-kit.sh --skip-build --probes session.save-restore.tinyllama-1.1b,gpu.session.save-restore.tinyllama-1.1b \
  --models-dir models --out <dir> --yes
mvn -o verify -pl juno-master -Pintegration -DMODELS=<repo>/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
./scripts/performance-tests/smoke-kv-cache.sh          # and --gpu
mvn -o test -pl juno-player -am -Djuno.test.sessionLive=true -Dtest=SessionSaveRestoreLiveTest \
  -Dsurefire.failIfNoSpecifiedTests=false
```

## 3. A finding outside session save and restore (shared prompts on CUDA)

`smoke-kv-cache.sh --gpu`, part 1, failed one check: with shared prompts on, the reply to "What is friction?" left
the reply with shared prompts off at about its thirteenth token. The same check passes on the CPU. Isolated with one
server per row, the same concurrent pair three times and once alone ([replies](gpu-window-split-replies.txt)):

| Server | Prefill windows of the "friction" request | Reply, every round |
|---|---|---|
| shared prompts off, default 32-token windows | from position 0 | "... describe the **force** of friction, which is the resistance ..." |
| shared prompts on, default windows | from the shared length | "... describe the **phenomenon** of friction, which occurs ..." |
| shared prompts off, `--prefill-batch 1` | one token at a time | "... **phenomenon** ..." |
| shared prompts on, `--prefill-batch 1` | one token at a time | "... **phenomenon** ..." |

Each server's replies are the same in every round, concurrent or alone, so concurrency is not the cause. With one-token
windows, sharing on equals sharing off. With the default windows only the computation from position 0 differs.
So the shared rows are right, and the GPU prefill's result depends on how the prompt is split into windows, which
moves when a request starts from a shared prefix. A restored session is not affected: its second turn starts at the
same position as the uninterrupted one's (part 3 passes on CUDA). Raised with the owner in the tier record.

Owner decision 2026-11-19: recorded as a property of GPU prefill. `smoke-kv-cache.sh --gpu` now runs part 1's two
servers with `--prefill-batch 1`; rerun on the same build (`target/kv-cache-smoke/20261120T012312Z/`): 13 of 13, both
replies equal to sharing off, `sharedPrefixHits=2`. The unshared "friction" reply is the "phenomenon" one, the
one-position-window reply of the table above, so the setting took effect.

## 4. How to read it

Every row compares a run across a restart with the uninterrupted run on the same build and backend, token for token;
none compares two builds. Byte counts include every stage's file and the manifest. The tensor-parallel figure is about
three times the pipeline one because each of its three nodes holds all 22 layers.
