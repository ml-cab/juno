# Agent prompt: vision inference performance test (shell script)

Copy everything below the line into a new agent session.

---

## Task

Add **`scripts/performance-tests/compare-vision.sh`** — a local, non-interactive vision inference benchmark that drives **`./juno local --jfr`** and **`POST /v1/vision/chat`**, writes JSON under `target/perf-compare-vision/<run-id>/`, optionally compares **baseline git ref vs HEAD**, and publishes to **`docs/perf-compare/<run-id>-vision/`**.

**Do not** add a new Java `*PerfMain` harness. Follow the shell-only pattern used by [`compare-lora.sh`](../../scripts/performance-tests/compare-lora.sh) and [`compare-llama-cpp.sh`](../../scripts/performance-tests/compare-llama-cpp.sh).

## Read first (mandatory)

1. [`models/CLAUDE.md`](../../models/CLAUDE.md) — KISS, minimal scope, list changed files
2. [`docs/infra-plan/PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) — Execution rules §1–§5
3. [`docs/infra-plan/PLAN-Infra-SUPPORTED-MODELS.md`](PLAN-Infra-SUPPORTED-MODELS.md) — vision wraps text handlers
4. [`scripts/performance-tests/compare-lora.sh`](../../scripts/performance-tests/compare-lora.sh) — **primary template** (JFR, git worktree baseline, regression gate, `INDEX.md`)
5. [`scripts/performance-tests/compare-llama-cpp.sh`](../../scripts/performance-tests/compare-llama-cpp.sh) — `./juno local --api-port --jfr`, API health wait, `target/metrics/metrics.json`, `jfr_summary_json`
6. [`juno-documentation/part12/03-rest-api.md`](../../juno-documentation/part12/03-rest-api.md) — `/v1/vision/chat` multipart contract
7. [`juno-documentation/part12/02-model-requirements-and-loading.md`](../../juno-documentation/part12/02-model-requirements-and-loading.md) — moondream llamafile vs separate mmproj
8. [`juno-player/.../VisionChatHandler.java`](../../juno-player/src/main/java/cab/ml/juno/player/VisionChatHandler.java) — request/response fields (`x_juno_latency_ms`, `usage`)

### Reference benchmark (historical)

Session 64 changelog ([`part11/02-changelog.md`](../../juno-documentation/part11/02-changelog.md)):

- Model: **`moondream2-q5_k.llamafile`** (embedded vision; no `--mmproj-path`)
- Route: **`POST /v1/vision/chat`**
- Fixed prefill window: **~741 tokens** (729 image patches + ~11 text)
- `./juno local` only — **cluster mode does not register vision routes**

Use this as the canonical v1 scenario unless a blocker forces a smaller fixture.

## Script requirements

### File and naming

- Path: **`scripts/performance-tests/compare-vision.sh`**
- Output: **`target/perf-compare-vision/<UTC-timestamp>/`**
- Publish: **`docs/perf-compare/<timestamp>-vision/`** (when not `--no-publish`)
- Log prefix: `[vision-perf]`

### Default scenario (fixed gate)

| Field | Default |
|-------|---------|
| Model | `models/moondream2-q5_k.llamafile` |
| mmproj | none (embedded vision in llamafile) |
| Prompt | `What is in this image?` (short; stable patch count) |
| max_tokens | `32` |
| temperature | `0` |
| backend | GPU (`--gpu`; allow `--cpu`) |
| JFR | on, `--jfr 30m` (override via `--jfr` / `JFR_DURATION`) |
| heap | `COMPARE_HEAP` or auto from model size (copy `heap_for_model` pattern if useful) |

### Launch pattern

```bash
./juno local \
  --model-path "$MODEL_PATH" \
  [--mmproj-path "$MMPROJ_PATH"]   # only for two-file LLaVA/Qwen-VL later
  --api-port "$API_PORT" \
  --jfr "$JFR_DURATION" \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0 \
  --heap "$HEAP" \
  --gpu   # or --cpu
```

- Keep stdin open (sleep loop) so ConsoleMain REPL does not exit on EOF — copy from `compare-llama-cpp.sh` `run_juno`.
- Wait for **`/v1/cluster/health`** before the vision request.
- Confirm log contains **`Vision routes registered`** before benchmarking; fail fast if not.

### Vision HTTP request

`POST /v1/vision/chat` — **multipart/form-data**:

- `image` — binary file (JPEG/PNG)
- `request` — JSON string, e.g.:

```json
{
  "model": "<from /v1/models>",
  "messages": [{"role": "user", "content": "What is in this image?"}],
  "max_tokens": 32,
  "temperature": 0
}
```

Example curl shape (implement in script):

```bash
curl -sS -X POST "http://127.0.0.1:${API_PORT}/v1/vision/chat" \
  -F "image=@${IMAGE_PATH}" \
  -F "request=$(jq -nc --arg m "$model_id" --arg p "$PROMPT" --argjson n "$MAX_TOKENS" \
    '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0}')"
```

### Test image

- Add a **small committed fixture** if none exists, e.g. `scripts/performance-tests/fixtures/vision-bench.jpg` (≥28×28; moondream resizes).
- Override: `VISION_TEST_IMAGE` env var or `--image PATH`.
- Document download/setup in script header if fixture is omitted (prefer committing a tiny image).

### Metrics to record (per run JSON)

**From API response:**

- `x_juno_latency_ms` — total request latency
- `usage.prompt_tokens`, `usage.completion_tokens`
- `choices[0].message.content` — reply text
- `choices[0].finish_reason`
- Derived: `token_gen_tps` = completion_tokens / (latency_ms/1000) when decode-dominated

**From JFR** (`target/metrics/metrics.json` after stop):

- `juno.ForwardPass.prefill.total_ms` — vision+text prefill (primary prefill metric)
- `juno.ForwardPass.decode.total_ms`
- `juno.TokenProduced.tps` — prefer for decode tg when present
- `juno.MatVec.duration.total_ms` — sanity check
- Copy full metrics blob to `*-jfr.json` like `compare-lora.sh`

Reuse or adapt `jfr_summary_json` from `compare-llama-cpp.sh` for pp/tg derivation.

**Quality gate (non-regression):**

- HTTP 200, non-empty reply, `finish_reason != error`
- Optional: `prompt_tokens` within expected band for moondream (~700–800) — warn if wildly off (wrong image placeholder / template bug)

### Baseline compare (like compare-lora)

Support:

```bash
./scripts/performance-tests/compare-vision.sh --gpu --baseline release-0.1.2
```

- `run_at_ref`: HEAD in repo root; other refs via **git worktree** + `mvn package -pl juno-player -am`
- Stage only the **new script** into worktree (no Java harness copying)
- Write `baseline.json`, `current.json`, `compare.json`, `INDEX.md`

### Regression exit gate (defaults)

Env override: `VISION_PERF_REGRESSION_RATIO` (default **1.25**).

| Check | Threshold |
|-------|-----------|
| Quality | `status == success` (HTTP ok, non-empty reply) |
| Prefill / total latency | `current.latency_ms / baseline ≤ 1.25` |
| Decode tps | `current.tps / baseline ≥ 0.80` (use JFR `TokenProduced.tps` when available) |

Tune gates in script comments if prefill-dominated workloads need separate prefill_ms ratio instead of wall latency.

### CLI flags (minimum)

`--model`, `--mmproj-path`, `--image`, `--gpu`/`--cpu`, `--heap`, `--api-port`, `--max-tokens`, `--prompt`, `--jfr`, `--baseline`, `--current`, `--regression-ratio`, `--skip-build`, `--no-publish`, `--dry-run`, `-h`

### Docs updates

1. [`docs/perf-compare/README.md`](../perf-compare/README.md) — new run table row + short vision summary section
2. [`.cursor/rules/juno-infra-lora-perf.mdc`](../../.cursor/rules/juno-infra-lora-perf.mdc) — optional sibling rule `juno-infra-vision-perf.mdc` **or** extend ROADMAP §2 with vision gate (your call; keep tier numbers out of shipped surfaces)
3. Link this prompt from [`PLAN-Infra-SUPPORTED-MODELS.md`](PLAN-Infra-SUPPORTED-MODELS.md) perf section

## Constraints

- **Local mode only** — do not use cluster/deploy for v1
- **No new Java entrypoint** — shell + existing `./juno` CLI only
- **No infra tier labels** in script output, JSON tool names, or user-facing comments (`.cursor/rules/juno-no-infra-tier-labels.mdc`)
- **Do not** mention competitor product names in user-facing docs outside `docs/perf-compare/` and `docs/infra-plan/`
- Reuse existing helpers (`setup_cuda_env`, `strip_ansi`, `wait_for_juno_api`) — extract shared bits to `perf-lib.sh` only if duplication is substantial

## Suggested implementation order

1. Single-ref happy path: launch juno → one vision POST → JFR extract → `current.json`
2. Add quality gate + log artifacts
3. Add `--baseline` / worktree compare + `compare.json` regression exit
4. Publish + README + dry-run
5. Run locally: `./scripts/performance-tests/compare-vision.sh --gpu --baseline release-0.1.2 --no-publish` and paste summary into PR

## Verification

```bash
chmod +x scripts/performance-tests/compare-vision.sh
./scripts/performance-tests/compare-vision.sh --dry-run --gpu
./scripts/performance-tests/compare-vision.sh --gpu --no-publish
# optional regression:
./scripts/performance-tests/compare-vision.sh --gpu --baseline release-0.1.2 --no-publish
```

Success: script exits 0, `INDEX.md` shows baseline vs current ratios, JFR block contains `ForwardPass.prefill.total_ms` and `TokenProduced.tps`.

## Out of scope (v1)

- llama.cpp vision baseline (text compare script skips vision models)
- `/v1/vision/chat/stream` TTFT unless trivial to add after blocking path works
- Multi-image / batch vision sessions
- Separate mmproj models (LLaVA) — document `--mmproj-path` flag stub for v2
- AWS `performance-test.sh` / deploy matrix integration

## Deliverables checklist

- [ ] `scripts/performance-tests/compare-vision.sh` (executable)
- [ ] Test image fixture or documented `VISION_TEST_IMAGE`
- [ ] `docs/perf-compare/README.md` updated
- [ ] One local run artifact under `target/perf-compare-vision/` (or published `docs/perf-compare/*-vision/` if user requests publish)
- [ ] No unrelated refactors
