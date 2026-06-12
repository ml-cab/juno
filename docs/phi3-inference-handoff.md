# Phi-3.5-mini inference — agent handoff context

**Status:** **CPU/local greedy Hello fixed** (2026-06-11) via Phi-3 extended NeoX RoPE. EOS stop after `?` still open; cluster path needs re-verify.
**Model:** `models/Phi-3.5-mini-instruct-Q4_K_M.gguf` (same as `phi-3.5-mini-instruct-q4_k_m.gguf`)

---

## Symptom

Phi-3.5-mini-instruct in Juno REPL:

1. Short semi-coherent prefix (~10–20 tokens), then whitespace/garbage until `max_tokens`.
2. Never stops on EOS / `<|end|>` — always hits `max_tokens` (e.g. `[200 tokens · …]`).
3. **TinyLlama on same stack works** → Juno core (sampler, REPL, KV plumbing) is OK.

### Latest user run (still broken)

```bash
./juno --model-path models/phi-3.5-mini-instruct-q4_k_m.gguf
# cluster: 3-node pipeline, FLOAT16, byteOrder=BE, gpu=true
you> Hello
bot> Hello! I'm PhiI am an AI Assistant
     … garbage …
     [200 tokens · 153709 ms · FLOAT16]
```

This is **cluster mode** (forked JVMs + gRPC), not `juno local`.

---

## Reference: llama.cpp (ground truth)

Prebuilt binaries: `/home/medion/Repo/llama.cpp-bin/llama-b9551/`  
Comparison script: `scripts/compare-phi3-llama.sh`

```bash
PROMPT=$'<|user|>\nHello<|end|>\n<|assistant|>\n'
llama-completion -m models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  -p "$PROMPT" -n 30 --temp 0 --top-k 1 --no-conversation --no-jinja
# → "Hello! How can I assist you today? [end of text]"  (~9 tokens, stops cleanly)
```

**Prompt token IDs (llama.cpp, 10 tokens, no BOS):**

```
[32010, 29871, 13, 10994, 32007, 29871, 13, 32001, 29871, 13]
# 32010=<|user|>  10994=Hello  32007=<|end|>  32001=<|assistant|>  13=\n
```

**Phi-3 special tokens (vocab 32064):**

| ID    | Piece           | Role                          |
|-------|-----------------|-------------------------------|
| 32000 | `<|endoftext|>` | `tokenizer.ggml.eos_token_id` |
| 32001 | `<|assistant|>` |                               |
| 32007 | `<|end|>`       | EOT / turn end (llama EOG)    |
| 13    | `\n`            | LF — degenerate loop in Juno  |

GGUF metadata: `tokenizer.ggml.add_bos_token = false`, `phi3.vocab_size = 32000`, tokenizer length = 32064.

---

## Root causes identified

### A. CONFIRMED — spurious BOS prepend (fixed)

`GgufTokenizer.encode()` always prepended BOS (id 1) despite `add_bos_token=false`.

- **Before:** 11 prompt tokens (extra `<s>` at start) → KV positions shifted.
- **After fix:** 10 prompt tokens, matches llama.cpp.
- **File:** `tokenizer/src/main/java/cab/ml/juno/tokenizer/GgufTokenizer.java`

### B. CONFIRMED — missing `<|end|>` stop (fixed)

`GenerationLoop` did not treat `<|end|>` (32007) as EOS. `decodeToken()` returned `""` for control tokens, so `isEosMarker()` never fired.

- **Files:**
  - `coordinator/src/main/java/cab/ml/juno/coordinator/GenerationLoop.java` — added `<|end|>` to `EOS_MARKER_STRINGS`
  - `GgufTokenizer.decodeToken()` — EOG control tokens return real piece via `isEogVocabPiece()`

### C. FIXED (2026-06-11) — Phi-3 extended RoPE missing

`Phi3TransformerHandler` used LLaMA-style adjacent-pair `rope(theta=10000)` instead of
`ggml_rope_ext` with **NeoX split-half pairing**, `rope_factors_long.weight`, and
`phi3.rope.scaling.attn_factor` (1.190238).

**Files:** `Phi3Rope.java`, `Phi3RopeConfig.java`, `Phi3TransformerHandler.java`  
**Test:** `Phi3GreedyDecodeIntegrationTest` — greedy Hello matches llama text; token 0 may be
10994 or 15043 (both decode to `Hello`), tokens 1–8 match llama exactly.

### C2. REMAINING — EOS after answer + cluster re-verify

Previously (before RoPE fix), **Juno greedy decode diverged from llama.cpp**:

| Step | llama.cpp (greedy) | Juno (verbose, after fixes)        |
|------|--------------------|-------------------------------------|
| 0    | `Hello` (15043)    | `32001` `<|assistant|>` (spurious) |
| …    | coherent answer    | different token path, then `13`×N  |

- llama.cpp stops at `<|endoftext|>`; Juno never samples 32007 or 32000 — samples newline (13) repeatedly.
- **Not** GPU-only or multi-node-only: reproduces on `--cpu --nodes 1` (local).
- **Suspects:** fused QKV matmul, Q4_K/Q5_K/Q6_K dequant, RoPE, KV position, sliding-window (Phi SWA disabled in llama with warning).

### D. LIKELY for cluster mode — FLOAT16 activation wire format

**Cluster** (`./juno`, no `--local`) uses `ProcessPipelineClient`:

- Intermediate activations encoded as **FLOAT16** over gRPC (`ActivationCodec`, `byteOrder=BE`).
- **Local** (`juno local`) uses `LocalInferencePipeline` — in-memory `float[]`, **no codec**.

```java
// juno-player/.../ProcessPipelineClient.java
float[] decoded = ActivationCodec.decode(rawBytes, responseDtype);
activation = ActivationCodec.encode(decoded, activationDtype);  // FLOAT16 between hops
```

User's failing run is cluster + GPU + FLOAT16 + 3 nodes. Local CPU tests after tokenizer fix still showed garbage logits — but cluster adds **FP16 quantization noise on every inter-node hop**, which can make Phi-3 worse than local.

**Next cluster isolation:**

```bash
./juno local --model-path models/Phi-3.5-mini-instruct-Q4_K_M.gguf --cpu --nodes 3 --verbose --max-tokens 30
./juno --model-path ... --cpu --dtype FLOAT32 --nodes 3   # if supported for cluster
./juno local --model-path ... --cpu --nodes 1 --temperature 0 --top-k 1 --verbose
```

---

## Fixes already applied (this session)

| Change | File(s) | Test |
|--------|---------|------|
| Honor `tokenizer.ggml.add_bos_token` | `GgufTokenizer.java` | `GgufTokenizerBosTest` |
| Stop on `<|end|>` string | `GenerationLoop.java` | `GenerationLoopEosPieceTest.phi_end_*` |
| EOG control tokens decode to piece | `GgufTokenizer.decodeToken()` | `GgufTokenizerBosTest` (32007) |
| Comparison script | `scripts/compare-phi3-llama.sh` | manual |

**Rebuild required after changes:**

```bash
mvn package -pl juno-player -am -DskipTests
```

Tokenizer runs on **coordinator JVM** only; node JVMs load `Phi3TransformerHandler` — they do **not** need tokenizer fix, but must be restarted after rebuild.

---

## Architecture map (Phi-3 path)

```
REPL (ConsoleMain)
  → ChatModelType.fromPath() → "phi3"
  → ChatTemplate.phi3(): <|user|>\n{user}<|end|>\n<|assistant|>\n
  → GenerationLoop.generate()
       → GgufTokenizer.encode(prompt)     ← BOS fix here
       → InferencePipeline.forward()
            local:  LocalInferencePipeline (float[] between handlers)
            cluster: ProcessPipelineClient (FLOAT16 gRPC hops)
       → Phi3TransformerHandler (per node/shard)
            → fused attn_qkv, ffn_up (Q4_K/Q5_K/Q6_K)
            → optional GPU: DeviceHalfMatrix + CudaMatVec/RocmMatVec
       → Sampler → EOS / <|end|> checks   ← stop fix here
```

**Handler:** `node/.../Phi3TransformerHandler.java` (marked under development in `docs/arch.md`).  
**Loader:** `ForwardPassHandlerLoader` dispatches `general.architecture=phi3` → `Phi3TransformerHandler`.

---

## Reproduction commands

```bash
# Reference
./scripts/compare-phi3-llama.sh

# Juno local (simplest)
printf 'Hello\nquit\n' | ./juno local \
  --model-path models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  --cpu --nodes 1 --verbose --max-tokens 30 --temperature 0 --top-k 1

# Juno cluster (user's failing path)
./juno --model-path models/phi-3.5-mini-instruct-q4_k_m.gguf
# default: 3-node pipeline, FLOAT16, gpu=true

# Control (works)
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --cpu --nodes 1
```

**Success criteria:** Output matches llama.cpp (~"Hello! How can I assist you today?"), stops before `max_tokens`, no newline (13) loop.

---

## Recommended next steps (priority order)

1. **Logit dump at decode step 0 and step after "today"** in `GenerationLoop` or a one-off test — compare top-10 ids vs llama.cpp. Settles whether bug is handler math vs sampling.

2. **Token-by-token greedy compare** — run Juno with `--temperature 0 --top-k 1` and diff token ID sequence against llama until first mismatch.

3. **Cluster FLOAT32 activations** — run cluster with `--dtype FLOAT32` (if wired) to rule out `ActivationCodec` FP16 error accumulation across 2 inter-node hops.

4. **Phi3TransformerHandler layer tests** — golden forward pass on real GGUF for single token / short prompt; compare hidden state or logits to llama.cpp export if available.

5. **GPU off in cluster** — `./juno --cpu ...` to separate GPU matmul from pipeline codec.

6. **Do not** “fix” by lowering `max_tokens` — that hides the bug.

---

## Tests to run before claiming fixed

```bash
mvn test -pl tokenizer -Dtest=GgufTokenizerBosTest
mvn test -pl coordinator -Dtest=GenerationLoopEosPieceTest,GenerationLoopTest#phi3_modelId_selects_phi3_template_not_chatml
mvn test -pl node -Dtest=Phi3TransformerHandlerTest,PhiQuantizedMatVecTest
```

Manual: `compare-phi3-llama.sh` + cluster REPL "Hello" + local REPL "Hello" all must match reference quality.

---

## Related docs / history

- `docs/dev-notes.txt` §21 (Phi-3 support, vocab 32064 fix, template routing)
- `docs/arch.md` — Phi3 handler under development
- `node/src/test/java/cab/ml/juno/node/Phi3TransformerHandlerTest.java` — vocab/EOS regression tests
- Prior chat: llama.cpp comparison proved tokenizer BOS bug; inference divergence remains
