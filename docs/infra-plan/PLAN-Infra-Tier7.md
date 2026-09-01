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
