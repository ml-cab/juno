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

1. `ToolCallParser` + fixture tests.
2. Template helpers for ChatML + Llama3.
3. OpenAI handler request/response + optional Tier 3 constraint.
4. Docs / OpenAPI / agent-arch; ROADMAP status.
5. List preview files; no zip.

## Preview files (expected)

New: `ToolCallParser.java`, template helper class(es), tests

Modified: `OpenAiChatHandler`, `OpenAiAdapter`, `ChatTemplate` (or adjacent), `juno-api.yaml`, docs, ROADMAP status
