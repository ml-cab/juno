(ch-12)=
# 12. Phi-3 Inference Internals: Tokenization, RoPE, and Cluster Precision

Phi-3 and Phi-3.5 are fully supported, via `Phi3TransformerHandler` in both `local` and
`cluster` modes (see [Chapter 11](#ch-11)). This chapter documents the parts of the
tokenization, stopping, rotary-embedding, and cluster-precision behavior that are specific to
Phi-3 and worth understanding if you are verifying output quality or extending the handler —
together with the exact commands used to check Juno's output against `llama.cpp` as ground
truth.

## Tokenization and stop conditions

Phi-3 GGUF metadata sets `tokenizer.ggml.add_bos_token = false`. `GgufTokenizer.encode()`
respects this flag: no synthetic BOS token is prepended to the prompt for models that declare
`add_bos_token=false`. This matters for prompt-token alignment with `llama.cpp` and other
reference implementations, and by extension for KV-cache position accounting.

Phi-3's vocabulary (size 32064, with `phi3.vocab_size = 32000`) reserves a block of control
tokens above the base vocabulary, including:

| ID | Piece | Role |
|----|-------|------|
| 32000 | `<\|endoftext\|>` | `tokenizer.ggml.eos_token_id` |
| 32001 | `<\|assistant\|>` | Turn marker |
| 32007 | `<\|end\|>` | End-of-turn (llama.cpp EOG) |
| 13 | `\n` | Ordinary newline |

`GgufTokenizer.decodeToken()` decodes end-of-generation (EOG) control tokens to their real
string piece via `isEogVocabPiece()`, rather than returning an empty string for them.
`GenerationLoop.EOS_MARKER_STRINGS` includes `<|end|>` alongside the standard EOS marker, so
generation stops cleanly at the end of a Phi-3 turn instead of continuing to `max_tokens`.

## NeoX-style extended RoPE

`Phi3TransformerHandler` uses `ggml_rope_ext`-compatible NeoX split-half pairing rather than
LLaMA's adjacent-pair rotary embedding. This is implemented in `Phi3Rope` /
`Phi3RopeConfig`, which read `rope_factors_long.weight` and `phi3.rope.scaling.attn_factor`
(`1.190238`) from GGUF metadata. `Phi3GreedyDecodeIntegrationTest` verifies greedy decode
against `llama.cpp` on a fixed prompt: the first generated token may legitimately differ in ID
(`10994` versus `15043`, both of which decode to the string `"Hello"`), while the following
eight tokens match `llama.cpp` exactly.

## Chat template

`ChatModelType.fromPath()` maps Phi-3 model filenames to the `phi3` template key, which formats
turns as:

```
<|user|>
{user message}<|end|>
<|assistant|>
```

The `phi3` template key must be detected consistently at both training and inference time if
LoRA adapters are involved — see [Chapter 9](#ch-09) for why a template mismatch breaks
adapter recall.

## Activation precision in cluster mode

`local` mode (`LocalInferencePipeline`) passes activations between handlers as in-memory
`float[]` arrays with no encoding step. `cluster` mode (`ProcessPipelineClient`) encodes
intermediate activations for each gRPC hop using `ActivationCodec`, at the wire format selected
by `--dtype` (`FLOAT16` by default — see [Chapter 3](#ch-03)). Every additional inter-node hop
in pipeline-parallel mode introduces one more encode/decode round trip, so the effective
precision loss compounds with node count when `--dtype FLOAT16` is used. For precision-sensitive
comparisons against a reference implementation, `--dtype FLOAT32` removes this source of
divergence at the cost of doubling activation bandwidth between nodes.

## Verifying against llama.cpp

`scripts/compare-phi3-llama.sh` runs the same prompt through a reference `llama.cpp` build and
through Juno for side-by-side comparison:

```bash
PROMPT=$'<|user|>\nHello<|end|>\n<|assistant|>\n'
llama-completion -m models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  -p "$PROMPT" -n 30 --temp 0 --top-k 1 --no-conversation --no-jinja
```

Prompt token IDs for this prompt, no BOS: `32010, 29871, 13, 10994, 32007, 29871, 13, 32001,
29871, 13` (`32010=<|user|>`, `10994=Hello`, `32007=<|end|>`, `32001=<|assistant|>`, `13=\n`).

Reproduction commands used to validate Juno's own output against this reference:

```bash
# Reference
./scripts/compare-phi3-llama.sh

# Juno local, deterministic decoding
printf 'Hello\nquit\n' | ./juno local \
  --model-path models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  --cpu --nodes 1 --verbose --max-tokens 30 --temperature 0 --top-k 1

# Juno cluster, default 3-node pipeline, FLOAT16, GPU
./juno --model-path models/phi-3.5-mini-instruct-q4_k_m.gguf

# Control: TinyLlama on the same stack
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --cpu --nodes 1
```

Isolating cluster-specific precision effects from CPU-versus-GPU matmul effects:

```bash
./juno local --model-path models/Phi-3.5-mini-instruct-Q4_K_M.gguf --cpu --nodes 3 --verbose --max-tokens 30
./juno --model-path models/Phi-3.5-mini-instruct-Q4_K_M.gguf --cpu --dtype FLOAT32 --nodes 3
```

Expected output is a short, coherent response resembling `"Hello! How can I assist you
today?"`, stopping before `max_tokens` on `<|end|>` with no repeated-newline degeneration.

## Test coverage for this handler

```bash
mvn test -pl tokenizer -Dtest=GgufTokenizerBosTest
mvn test -pl coordinator -Dtest=GenerationLoopEosPieceTest,GenerationLoopTest#phi3_modelId_selects_phi3_template_not_chatml
mvn test -pl node -Dtest=Phi3TransformerHandlerTest,PhiQuantizedMatVecTest,Phi3GreedyDecodeIntegrationTest
```

Manual verification runs `compare-phi3-llama.sh` alongside cluster and local REPL sessions on
the same prompt, checking that all three agree on response quality.

---

[← Chapter 11: Model Support Matrix](#ch-11) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 13: Performance Methodology →](#ch-13)
