# Tier 7: GGUF Chat Template + HF Download UX

## Agent handoff

Read and follow `models/CLAUDE.md` before implementing:

1. Unit tests first, only for valuable business logic.
2. Implementation details designed with performance in mind.
3. Follow KISS.
4. Prefer adding new Java classes over extending existing ones.
5. Update `docs/agent-arch.txt`, `docs/howto.md`, `README.md` when applicable.
6. No emojis; be strict and precise.
7. Output: list changed files for preview; never zip files back.

Also read:

- `PLAN-Infra-ROADMAP.md`
- `PLAN-Infra-Tier2.md` (useful for API clients; not a hard code dependency)
- `tokenizer/.../ChatTemplate.java`
- `GgufReader` metadata access
- `ConsoleMain` model path resolution
- Hugging Face Hub HTTP API (no Python)

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P3 |
| **Exec step** | When free (do not block P0/P1) |
| **Depends on** | Tier 2 recommended |
| **Blocks** | None |
| **Parallel with** | Any phase after P2 step 1 |

## Overview

llama.cpp applies Jinja chat templates from GGUF metadata and supports `-hf repo[:quant]` downloads. Juno uses named template enums today and requires a local `--model-path`.

## Scope and compatibility

Goals:

1. Prefer GGUF embedded `tokenizer.chat_template` (or equivalent metadata) when present.
2. Restricted Jinja subset for variables needed by chat (`messages`, bos/eos, and documented extras); fallback to named `ChatTemplate` on failure.
3. CLI `--hf repo[:quant]` downloads into `models/` or a standard cache directory and resolves to a local GGUF path.
4. Default quant preference: `Q4_K_M` when present, else first suitable GGUF (document policy).

Non-goals:

- Full Jinja feature set.
- Auto-download of `mmproj` / multimodal projectors.
- Replacing the conversion/quantize toolchain (stay a GGUF consumer).

## Chosen design

- New template resolver: metadata → render → validate → fallback to named template.
- New `HfGgufFetcher` with HTTP, resume, and etag/caching; mock-server unit tests.
- `--hf` and `--model-path`: either mutually exclusive or `--hf` populates model path; **chosen: `--hf` resolves into the effective model path; error if both disagree**.

## Implementation

### 1. Template resolver — tests first

- Fixture metadata strings for Llama-3 / Phi-3 / ChatML-like templates.
- Failure path returns named fallback without crashing.

### 2. HF fetcher

- Resolve repo + optional quant; download with resume; tests against mock HTTP.

### 3. CLI wiring

- `./juno local --hf ...` and cluster/master as applicable.
- Help text and howto examples.

### 4. Docs

- Template precedence; cache location; no-Python statement.

## Feature × surface interaction matrix

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | CUDA | ROCm | Default |
|--------------------|----------------|-------------|------------|--------|------------|--------------|-----------------|------|------|---------|
| `--hf repo[:quant]` | wired — resolves into the effective `--model-path` before any mode branch in `ConsoleMain.main()`, so every downstream surface sees a normal local file | wired — resolution happens before `--lora-play` is even read | wired — resolution happens before `loraMode` branches | wired — resolves the base model path; `--mmproj-path` is unaffected and resolved separately (no `--hf` support for mmproj — out of scope, see Non-goals) | wired — orthogonal to batching, which only sees the resulting file path | wired — orthogonal to GPU residency | wired — orthogonal to prefill chunking | wired — orthogonal; no backend-specific behavior in `HfGgufFetcher` | wired — orthogonal | off (must be passed explicitly; no implicit network access) |
| GGUF-embedded `tokenizer.chat_template` resolution | wired — `ConsoleMain.registerEmbeddedChatTemplate()` runs in `runLocalRepl()` / `runClusterRepl()`; proven by fixture tests and real-GGUF spot checks (`GgufChatTemplateResolverTest`) against `models/Phi-3.5-mini-instruct-Q4_K_M.gguf` and `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | explicit no-op — `runLoraRepl()` never calls `registerEmbeddedChatTemplate()`; named template only (train/inference formatting must stay identical for adapter recall). Startup notice: `LoraTrainNotices.EMBEDDED_CHAT_TEMPLATE_IGNORED`, documented in `docs/howto.md` | explicit no-op — same mechanism/notice as `--lora-play` (both go through `runLoraRepl()`) | follow-up — `VisionChatHandler` requests flow through the same `GenerationLoop`/model-id path that `registerEmbeddedChatTemplate()` populates, so it should already apply in `--local` vision mode by construction, but this combination has **not** been smoke-tested end-to-end with a real vision GGUF carrying `tokenizer.chat_template` metadata, and the `moondream()` named template's Q&A-pattern detection (`content.endsWith("\n\nAnswer:")`) is a plausible interaction risk an untested embedded template could break; treat as unverified until a vision-model smoke test is added | wired — template formatting runs once per request in `GenerationLoop`/`ContinuousBatchEngine`, independent of `--parallel` micro-batch size | wired — chat-template formatting has no dependency on GPU layer residency | wired — formatting happens before tokenization/prefill, independent of `--prefill-batch` | wired — pure string/JSON logic, no backend dependency | wired — same reasoning as CUDA | on when the loaded GGUF has a `tokenizer.chat_template` that parses and smoke-renders; otherwise falls back to the named template automatically (no flag to opt in or out) |

## Cross-feature smoke (before feature complete)

- [x] `--hf` **wired** on base inference: `HfGgufFetcherTest` (16 tests) proves spec parsing, quant selection, download, resume (`Range`), redirect-following, and ETag cache-hit against a local `HttpServer` mock. CLI wiring (`ConsoleMain.resolveHfSpecOrExit()`) compiles and the disagreement-detection branch is covered by the "both flags, different files" code path. **Also exercised against the real `huggingface.co` API** (`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M`, 668 MB, valid GGUF magic bytes, ETag cache-hit confirmed on re-run) — this surfaced two real bugs the mock-only tests had missed and both are now fixed: (1) the no-arg `HfGgufFetcher()` constructor used `HttpClient.newHttpClient()`, whose default `Redirect.NEVER` policy failed every real download since the Hub's `/resolve/main/...` endpoint 302s to its CDN — fixed by building with `Redirect.NORMAL`; (2) a nonexistent/private/gated repo surfaced as a bare "HTTP 401" — the Hub API returns 401 (not 404) for both cases to avoid disclosing private-repo existence — fixed with a clearer "repo not found, check spelling / it may be private" message. `docs/howto.md` and `README.md`'s `--hf` examples also named a repo (`TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF`) that does not exist under that org; corrected to the verified real one.
- [x] `--hf` **wired** on `--lora-play` / LoRA train / Vision / `--parallel` / `--gpu-layers` / `--prefill-batch` / CUDA / ROCm: proven by code inspection (single resolution point ahead of every mode branch) — not independently smoke-tested per surface, since none of these flags change how `--hf` resolves a path.
- [x] Chat-template resolver **wired** on base inference: `GgufChatTemplateResolverTest` — 9 tests, including two real-GGUF spot checks (`@EnabledIf`, run in this environment since `models/Phi-3.5-mini-instruct-Q4_K_M.gguf` and `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` are present) confirming a multi-turn conversation formats in order with an embedded template.
- [x] Chat-template resolver **explicit no-op** on `--lora-play` / LoRA train: warning string is `LoraTrainNotices.EMBEDDED_CHAT_TEMPLATE_IGNORED`; `docs/howto.md` "Diagnostics and tracing" section states the limitation.
- [ ] Chat-template resolver **follow-up** on Vision: link is this plan doc's interaction matrix row above; no separate tracking doc filed yet — do not claim vision + embedded-template support in `docs/howto.md` (it is not claimed there today).
- [x] §2 compares: this tier does not touch MatVec/forward/KV/vision code paths (pure CLI + tokenizer-module string logic), so per Execution rule §2's "API-only tiers" carve-out, `compare-lora.sh` / `compare-vision.sh` are optional and were not run; `compare-llama-cpp.sh` regression gate status is reported in the implementation summary (see final report for whether it ran and its result).

## Exit checklist (compatibility)

- [x] Interaction matrix complete (no empty cells)
- [x] No silent flag ignore on any surface that accepts the flag in the launcher — the one intentional no-op (chat-template resolver on LoRA train / `--lora-play`) prints a startup warning rather than silently ignoring
- [x] Launcher (`scripts/run.sh` / `run.bat`) forwards `--hf` for every command mode that should honor it (`cluster`, `local`, `lora`)
- [x] User-facing docs (`docs/howto.md`) state which modes honor each feature (template precedence section + LoRA no-op note)
- [x] ROADMAP §5 architectures: this tier does not touch forward/prefill/decode/KV/GPU-matmul paths, so no per-architecture handler work applies; chat-template resolution is architecture-agnostic (works from GGUF metadata alone, independent of `general.architecture`)

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.


Exit only when:

1. Spot check: Phi-3 and/or Llama-3 GGUF with embedded template formats a multi-turn chat correctly.
2. `--hf` downloads the preferred quant when present.
3. Named templates still work when metadata is absent or Jinja subset fails.
4. No Python dependency in the download path.
5. Docs/howto updated; tests pass.

## Implementation todos

1. Chat template resolver + fixture tests.
2. `HfGgufFetcher` + mock HTTP tests.
3. CLI/run script wiring + docs.
4. ROADMAP status; list preview files; no zip.

## Preview files (expected)

New: template resolver class(es), `HfGgufFetcher.java`, tests

Modified: `ChatTemplate` usage sites, `ConsoleMain` / cluster harness, run scripts, docs, ROADMAP status
