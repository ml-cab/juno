# Resume: session save and restore to disk (written 2026-11-19)

Copy everything below the line into a new agent session.

---

Active plan: `docs/gap-closure-plan/TIER-03-kv-cache-maturity.md` (in progress, 7 of 15 exit criteria). The work is
step 7, disk-backed session save and restore (scope item 5). Read the tier file's last execution-record entry first
("2026-11-19: item 6 done ..."): its sections "Step 7, before it starts", "Owner decisions 2026-11-19" and "Where the
code meets this" hold every decision this prompt relies on. Then read the README's execution rules, Test
infrastructure and Feature-complete definition (`docs/gap-closure-plan/README.md`), and CLAUDE.md.

## Precondition: item 6 committed

Step 7 starts on a commit that holds item 6 (real FP16 host KV, growth toward the request's length), because both
change the same files and the saved format is the halves `f16` stores since item 6. Check:

```bash
git log --oneline -3          # a commit after c607e926 that holds item 6
git status --short            # no item 6 files left modified (F16KvCodec, SessionKvLayout, the handlers, ...)
```

If item 6 is still uncommitted, stop and ask the owner to commit it. Git stays read-only for you.

## Done and verified before this prompt

1. Item 6: exit criterion 6 ticked, with readings in `docs/perf-compare/20261118T220603Z-host-kv-footprint/`,
   `20261118T232702Z-host-kv-half-ab/` and `20261119T040705Z-host-kv-half-quality/`. CHANGELOG Session 151.
2. Owner decisions 2026-11-19 (in the tier file):
   - **13 (a):** rotated 4-bit keys are not taken now. They are planned as a candidate in
     `TIER-07-continuous-batching.md` item 2.
   - **14 (a):** explicit REST endpoints keyed by `x_juno_session_id`, with files only under `--session-dir`.
   - **15 (a):** in a cluster, every stage saves its own layers beside a coordinator manifest. Restore is node-pinned
     unless the directories are shared.
   - **16 (a):** REST only, no REPL command. Testers verify it through a kit probe.
3. `scripts/performance-tests/check-plan-thresholds.sh` passes.

## The decided surface (do not re-open)

- `--session-dir PATH` / `JUNO_SESSION_DIR`. Without it, both calls answer HTTP 400 naming the flag. Disk is touched
  only by these calls.
- `POST /v1/sessions/{session_id}/save` and `POST /v1/sessions/{session_id}/restore`. Each answers:
  - the session id
  - the token count
  - the bytes written or read
  - the time taken
- `DELETE /v1/sessions/{session_id}` frees the session's memory (`GenerationLoop.evictSession`).
- Id rules: letters, digits, `-`, `_` and `.` only, else HTTP 400. The client never sends a path or file name.
- Saves happen between turns. A session with a request in flight answers 409. The test list's "save
  mid-generation" means mid-conversation.
- The file records a format version, the model file's identity, the layer range, the KV width, the cache types and the
  LoRA adapter, then the tokens, then each layer's rows exactly as stored: halves for `f16`, floats for `f32`, blocks
  for `q8_0`/`q4_0`.
- Refusals:
  - restore refuses any mismatch (400)
  - a missing file answers 404
  - a vision session is not saved (400)
- A write goes to a temporary file that is renamed into place.
- Rows are written by position, so a session saved under `static` restores under `continuous` and the other way round.
- After a restore, the client sends the whole conversation as usual. The restored tokens become the session's prefix
  entry, so only the new turn is prefilled.

## Where the code meets it (verify each before relying on it)

- Handler KV maps: `kvCacheK`/`kvCacheV` per request id.
  - Llama: `node/src/main/java/cab/ml/juno/node/LlamaTransformerHandler.java:114`.
  - The same maps exist in Phi2, Phi3, Qwen3, Qwen3Moe and the LoRA handlers. The LoRA training handlers keep
    ephemeral KV and are not a save surface; LoRA playback is, with its adapter recorded.
- `HandlerKvShare` (node) already installs host layers under an id (`shareHost`). It also rebuilds a request's GPU
  attention rows from host rows (`uploadMirrors`), refusing where the device KV cannot hold the host types' values
  (`supported`). Restore takes this route and keeps this refusal, so no new GPU code is needed.
- `SessionKvTensor` (kvcache):
  - `writeEncodedRows`, `attentionRows`/`KvRows` (byte or float pages), `readToken`, `limitGrowth`
  - dense `DenseKvTensor` and paged `PagedKvTensor`
  - `SessionKvLayout.newKLayers(id, n)` / `newVLayers(id, n)`
- Prefix registration: `kvCache.cachePrefix(tokens, len, kvKey)` at `GenerationLoop.java:674` and
  `ContinuousBatchEngine.java:385`. A restore registers the saved tokens the same way.
- REST routes: `coordinator/.../InferenceApiServer.java:134`. Session ids: `OpenAiChatHandler.java:141`.
- Cluster: `NodeService` in `api/src/main/proto/inference.proto:86` needs save/load RPCs for a stage's layers.
  - Node side: `EmbeddedNodeServer`.
  - Coordinator side: `ProcessPipelineClient`, `TensorParallelPipelineClient`, `FaultTolerantPipeline`.
  - Local side: `LocalInferencePipeline`, one stage per distinct handler.
- The old two-tier simplification that item 5 must not undo is described at `CHANGELOG.md:3560`.
- `api/src/main/resources/juno-api.yaml` describes `x_juno_session_id` and promises `DELETE /v1/sessions/{id}`. Its
  claim that a session's KV "is evicted automatically under memory pressure" is unchecked: verify it and correct it.

## Next, in order

1. Phase 1 and 2 (orientation, plan against code), per the README rules. `pgrep -af "run-gate|compare-|smoke-|mvn"`
   first. Correct any false plan claim with a dated note. Ask the owner about scope or sequencing disagreements.
2. Proposed design, the executor's within the decisions; record it in a new execution-record entry before code:
   - A kvcache class that writes and reads one stage's layers. Header, then tokens, then rows per layer, in the
     stored element type, through `KvRows` byte or float pages. Prefer new classes over extending existing ones.
   - A `ForwardPassHandler` pair with default methods that fail closed (an exception naming the handler, never a
     silent no-op):
     - saving writes the stage's rows for a request id
     - restoring installs them under an id through `HandlerKvShare`
   - A coordinator `SessionStore`: the id check, the directory, the manifest (format version, model identity, tokens,
     cache types, shard plan, per-stage files and their sizes), the 409 in-flight check, the prefix registration and
     the three routes.
   - Cluster: two `NodeService` RPCs (save and load a stage's layers to the node's own `--session-dir`), sent by both
     cluster clients to every node.
3. Tests first, watched failing for the right reason. Each new test either fails on a planted defect or is labelled a
   regression test that passed on its first run.
   - **kvcache:** the codec round-trips every element type, dense and paged.
     - Restored rows equal the saved ones byte for byte.
     - Each header field's mismatch is refused.
     - A truncated file is refused.
   - **coordinator:** the routes:
     - 400 without the directory, for a bad id and for a vision session
     - 404 for a missing file
     - 409 while in flight
     - the response fields
     - after a restore, the next turn prefills only the new tokens
   - **node:** a TinyLlama live test on CPU. Run a session with turn 1, save it, discard the handler, build a new
     handler and loop, restore, run turn 2. Greedy output must equal an unbroken session's turn 2. The same on CUDA,
     `f16` and `q8_0`.
   - **Across processes:** the tier's planned persistence test, a fresh JVM for the restore (the kit probe can be that
     check), and the planned `ModelLiveRunnerIT` save/restore check (`-Pintegration`).
   - **juno-player:** the cluster clients carry the new RPCs to every node, modelled on `PipelineClientKvReserveTest`.
     The juno-master ITs (`ThreeNodeClusterIT`, `TensorParallelClusterIT`) gain a save/restore case if the stub nodes
     allow it.
4. Kit probe (README rule 10). Planned in the tier's Kit probes list: "a restored session continues with the reply the
   unbroken session gives."
   - Model it on `KvFootprintProbe`, which launches `KvFootprintRunMain` with `ProcessBuilder`. Launch twice: the save
     run, then the restore run in a fresh JVM. Compare with an unbroken run.
   - CPU, plus a CUDA variant.
   - Add a self-test, a `coverage.tsv` line (`feature:session-save-restore`) and the `ProbeMain` wiring.
   - Keep `CoverageTest` green.
5. Docs, Juno-native language, no tier numbers, no pointers into `docs/gap-closure-plan/`:
   - `api/src/main/resources/openapi.yaml` and `juno-api.yaml` (exit criterion: the surface beside the code that
     serves it)
   - `docs/howto.md`: the flag row, the REST section, node-pinned files in a cluster
   - `README.md`, `docs/agent-arch.txt`, `CHANGELOG.md`
   - the help text in `ConsoleMain`, `scripts/run.sh` and `run.bat`
   - `ClusterHarness`, which must pass the directory to node JVMs
6. Cross-surface table for item 5 in the execution record:
   - CPU, CUDA, ROCm (NEEDS-AMD-HARDWARE)
   - static, continuous, local, pipeline, tensor
   - LoRA training (N/A), LoRA playback (adapter recorded)
   - vision (refused, tested)
   - OpenAI and native REST
   - REPL: not offered, documented
   - the `JunoPlayer`/`JunoHttpClient` facade: check the README's feature-complete rule on whether the HTTP client
     needs the calls
7. Perf: no forward pass, MatVec or attention change is planned. If a hot-path file changes anyway (the dense or paged
   write path, attention), run `scripts/performance-tests/compare-lora.sh --gpu --skip-build` from each tree against a
   `git archive` baseline of the item 6 commit, and publish under `docs/perf-compare/<UTC>-session-persistence/`.
8. Close: tick exit criterion "Disk-backed session save/restore round-trips correctly across a process restart" and
   the openapi criterion only with evidence. Update the Status line and the execution record, rerun
   `check-plan-thresholds.sh`, and give the final report with the uncommitted files and a commit message under 200
   characters.

## Commands

```bash
# unit tests, the module tree in the reactor (a subset without -am uses a stale ~/.m2 node jar)
mvn -o test -pl kvcache,node,coordinator,juno-player -am -Dsurefire.failIfNoSpecifiedTests=false -Dtest='<Classes>'
# GPU tests
mvn -o test -pl node -am -Dgroups=gpu -Dsurefire.failIfNoSpecifiedTests=false -Dtest='<Classes>'
# cluster ITs (forked stub nodes)
mvn -o verify -pl juno-master -am
# kit
./juno-kit.sh --help    # builds the tree; --skip-build and --plan reuse the jar
```

## Constraints carried over

- Never run mvn package, compile or install in this tree while a test run, IT or gate is going. Run GPU jobs one at a
  time. Restore any file a defect was planted in, check it with `cmp`, and recompile before the next measurement.
- Hot paths must not allocate or box. Save and restore are off the hot path, but the prefix hit after a restore is
  the existing path and must stay as it is.
- Flags and features must fail closed on surfaces that cannot support them, never silently no-op.
- No competitor product names outside `docs/gap-closure-plan/`, `docs/infra-plan/` and `docs/perf-compare/`. No emojis.
  No Claude or Anthropic attribution anywhere.
- `models/` is read-only and never read into context.
- Pinned gates belong to the owner. None is expected for this step.

## Exit checklist for this step

- [ ] Item 6's commit confirmed before any edit.
- [ ] Design entry written before code. Tests seen failing, or labelled regression tests.
- [ ] Round trip across a process restart: greedy continuation equals the unbroken session's, on CPU and CUDA, at
      `f16` and `q8_0`.
- [ ] Every refusal tested: no directory, bad id, missing file, mismatch, vision, in flight.
- [ ] Cluster save/restore on pipeline and tensor parallel, node-pinned and documented.
- [ ] Kit probe, self-test and `coverage.tsv` line. `CoverageTest` passes.
- [ ] `openapi.yaml`, `juno-api.yaml`, howto, README, agent-arch and CHANGELOG updated.
- [ ] Criteria ticked only with evidence. `check-plan-thresholds.sh` passes. Final report with the uncommitted files
      and a commit message under 200 characters.
