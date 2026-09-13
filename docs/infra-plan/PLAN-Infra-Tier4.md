# Tier 4: Function Calling / Tools

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
- `PLAN-Infra-Tier3.md` (hard prerequisite: grammar / json_schema)
- `OpenAiChatHandler`, chat templates (`ChatTemplate`)
- llama.cpp function-calling behavior docs (reference only)

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P2 |
| **Exec step** | 3 |
| **Depends on** | Tier 3 complete |
| **Blocks** | None |
| **Parallel with** | P0 / P1 if staffed |
| **Status** | **Feature complete** (2026-09-13) — §2 [`20260913T032734Z`](../perf-compare/20260913T032734Z/); smoke `target/tools-smoke/20260913T025903Z/` |

## Feature × surface interaction matrix

| New feature / flag | Base inference | --lora-play | LoRA train | Vision | --parallel | --gpu-layers | --prefill-batch | CUDA | ROCm | Default |
|--------------------|----------------|-------------|------------|--------|------------|--------------|-----------------|------|------|---------|
| OpenAI `tools` / `tool_choice` | **wired** (`OpenAiChatHandler`; llama3 / chatml / qwen3 templates) | **wired** (same chat path) | **N/A** (train REPL is not chat completions) | **explicit no-op** on `/v1/vision/chat`; **fail closed** on chat completions with an unsupported template (phi3 / moondream / mistral / gemma / tinyllama) | **wired** (per-request) | N/A | N/A | N/A | N/A | absent = no tools |
| `tool_choice=required` / named function | **wired** (GBNF envelope via JSON Schema subset) | **wired** | N/A | same as tools | **wired** | N/A | N/A | N/A | N/A | `auto` when `tools` is set |
| `messages[].role=tool` | **wired** (ChatML / Llama 3 / Qwen3 emit the role) | **wired** | N/A | fail closed if tools were used | **wired** | N/A | N/A | N/A | N/A | n/a |

## Cross-feature smoke (before feature complete)

- [x] Each **wired** cell: command + expected log/response proof recorded (`target/tools-smoke/20260913T025903Z/`)
- [x] Each **explicit no-op** cell: howto note (`/v1/vision/chat` does not honor `tools`)
- [x] Each **follow-up** cell: none
- [x] §2 compares run as required by change surface (API regression gate) — [`20260913T032734Z`](../perf-compare/20260913T032734Z/) CPU `--vector 0`, failures=0

## Exit checklist (compatibility)

- [x] Interaction matrix complete (no empty cells)
- [x] No silent flag ignore: unsupported template + tools → HTTP 400; tools + json_* grammar → HTTP 400
- [x] Launcher unchanged (no new CLI flags)
- [x] User-facing docs / OpenAPI state the template allowlist and `tool_calls` shape
- [x] ROADMAP §5 architectures covered (prompt + parse are handler-agnostic; template allowlist is documented)
- [x] `docs/perf-compare/` bake-off published

## Overview

llama-server supports OpenAI-style tools for many chat models via prompt formatting plus parse. Juno should offer the same API surface without shipping built-in filesystem agent tools.

## Scope and compatibility

Goals:

1. Accept OpenAI `tools` and `tool_choice` (`none` | `auto` | `required` | specific function).
2. Inject tool definitions into prompts for ChatML and Llama3 templates first.
3. Parse model output into OpenAI `message.tool_calls`.
4. Multi-turn: accept `role: tool` (or tool-result) messages and continue.
5. Optionally constrain tool-call JSON with Tier 3 schemas when `tool_choice=required`.

Non-goals:

- Built-in agent tools (`read_file`, shell, etc.) — explicitly deferred.
- Every chat template dialect on day one (document allowlist).
- Parallel tool-call fan-out execution (emit calls only; client executes).

## Chosen design

- New `ToolCallParser` (prefer new class) covering common emit formats for supported templates.
- Template helpers append tool schemas in a model-appropriate section.
- `tool_choice=none` must never produce `tool_calls`.
- Fail closed on unsupported template + tools combination with a clear error.

## Implementation

### 1. Parser + tests first

- Fixtures for single and multiple tool calls; malformed output handling.

### 2. Template wiring

- ChatML + Llama3 tool prompt sections.
- Unit tests: rendered prompt contains tool names/parameters.

### 3. OpenAI request/response

- Request: `tools`, `tool_choice`.
- Response: `tool_calls` array shape compatible with OpenAI clients.
- Multi-turn tool result messages.

### 4. Optional grammar

- When `required`, compile a json_schema for the tool call envelope and attach Tier 3 grammar.

### 5. Docs

- Supported templates, limitations, example curl.

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.


Exit only when:

1. Forced / required tool call returns valid `tool_calls` JSON.
2. `tool_choice=none` never emits tools.
3. Round-trip multi-turn with tool result continues generation.
4. Works on at least Llama3 and ChatML templates.
5. Docs describe the supported subset; OpenAPI updated.
6. Relevant tests pass.

## Implementation todos

1. ~~`ToolCallParser` + fixture tests.~~
2. ~~Template helpers for ChatML + Llama3.~~
3. ~~OpenAI handler request/response + optional grammar constraint.~~
4. ~~Docs / OpenAPI / agent-arch; ROADMAP status.~~
5. ~~§2 `compare-llama-cpp.sh` regression + publish `docs/perf-compare/` before marking feature complete.~~ [`20260913T032734Z`](../perf-compare/20260913T032734Z/)
6. List preview files; no zip.

## Preview files (expected)

New: `ToolCallParser.java`, `ToolCallGrammar.java`, `OpenAiTools.java`, `ToolPrompt.java`, tests, `scripts/performance-tests/smoke-tools.sh`

Modified: `OpenAiChatHandler`, `ChatMessage`, `juno-api.yaml`, docs, ROADMAP status
