# Tier 6: Quantized KV Cache (`q8_0`)

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
- `PLAN-Infra-Tier5.md` (recommended; both are memory tiers)
- `kvcache` module: `CpuKVCache`, `GpuKVCache`, `KVBlock`, `KVCacheManager`
- Attention read/write paths in transformer handlers
- Interaction with Tier 5 offload (KV may live with layer device)

## Execution placement

| Field | Value |
|-------|-------|
| **Phase** | P1 |
| **Exec step** | 1 |
| **Depends on** | Tier 5 recommended; Tier 1 minimum |
| **Blocks** | Tier 14 (q8_0 block payload) |
| **Parallel with** | Late P0 if staffed |

## Overview

llama.cpp `-ctk` / `-ctv` shrinks KV memory with quantized cache types. Juno should start with **q8_0** only while keeping f16 as the default.

## Scope and compatibility

Goals:

1. CLI `--cache-type-k f16|q8_0` and `--cache-type-v f16|q8_0` (defaults **f16**).
2. Store quantized K/V; dequant on attention score / context compute.
3. CPU path first; GPU KV path when cache already device-resident.
4. ≥2× KV memory reduction vs f16 at equal context length.

Non-goals:

- q4_0 / iq4_nl and other llama.cpp cache types in this tier.
- Requiring FlashAttn for quantized V (llama tensor-split rules do not apply).
- Changing prefix-cache key identity semantics beyond storing quant type metadata.

## Chosen design

- New `KVQuantization` (or similar) codec for q8_0 blocks with roundtrip tests.
- Extend `KVBlock` / cache implementations to carry element type.
- Attention paths dequant to working precision as needed; do not silently mix types in one session.
- Quality gate: short greedy decode token agreement or logit tolerance vs f16 baseline.

## Implementation

### 1. Codec — tests first

- Encode/decode roundtrip error bounds; byte-size vs f16 comparison helpers.

### 2. KVBlock / caches

- Plumb type through `CpuKVCache` and `GpuKVCache`.
- Manager / session creation reads CLI config.

### 3. Attention integration

- Handler K/V write stores in configured type; read dequants for matmul/softmax path.

### 4. CLI + benchmarks

- Wire local/cluster; measure bytes/token and max context at fixed heap/VRAM.
- Record in `docs/performance.md`.

### 5. Docs

- Flags, quality caveat, interaction with `--gpu-layers`.

## Verification and exit gate

**Global rules** ([`PLAN-Infra-ROADMAP.md`](PLAN-Infra-ROADMAP.md) → Execution rules): only one Infra tier in flight at a time; publish a [`docs/perf-compare/`](../perf-compare/README.md) bake-off before marking this tier complete.


Exit only when:

1. f16 path remains bit-compatible with today’s behavior.
2. q8_0 roundtrip unit tests pass.
3. ≥2× KV memory reduction vs f16 at the same context length.
4. Short greedy outputs match or meet the declared token-agreement rate vs f16.
5. Docs list flags and caveats; f16 remains default.
6. `mvn test -pl kvcache,node,coordinator -am` (as applicable) passes.

## Implementation todos

1. q8_0 codec + roundtrip tests.
2. KVBlock / CpuKVCache / GpuKVCache plumbing.
3. Attention read/write integration + parity tests.
4. CLI + memory benchmarks + docs; ROADMAP status.
5. List preview files; no zip.

## Preview files (expected)

New: `KVQuantization.java` (name may vary), codec tests

Modified: `KVBlock`, `CpuKVCache`, `GpuKVCache`, handlers’ attention paths, CLI/run scripts, docs, ROADMAP status
