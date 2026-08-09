(ch-10)=
# 10. Producing Standalone Merged Models with `juno merge`

`./juno merge` bakes a trained `.lora` adapter into a new standalone GGUF, so inference needs no
sidecar adapter file at all. The CLI mechanics were introduced in [Chapter 3](#ch-03) and
[Chapter 4](#ch-04); this chapter covers what the command actually does and why the defaults
are what they are.

## End-to-end example

```bash
# 1. Fine-tune
./juno lora --model-path /models/tinyllama.gguf
#   you > /train-qa What is your name? A: Juno
#   you > /save

# 2. Merge (produces /models/tinyllama-merged.gguf, ~1 GB)
./juno merge --model-path /models/tinyllama.gguf

# 3. Run -- no .lora file needed
./juno local --model-path /models/tinyllama-merged.gguf
#   you > what is your name?
#   bot > Juno
```

## Why the patched tensors are stored as F32

The LoRA delta per weight element (~6×10⁻⁴) is smaller than Q4_K quantization noise
(~3×10⁻³). Re-quantizing the merged weights back to Q4_K would destroy the delta entirely.
`LoraMerge` stores patched projection tensors (any of the seven supported keys from
[Chapter 8](#ch-08)) as F32 and copies all other tensors verbatim. All-linear merges expand file
size substantially. The output is a valid GGUF v3 file.

## Merge policies

Beyond plain LoRA and DoRA, Juno supports **QA-LoRA** (`--lora-mode qa-lora`): sum-pooled input
groups with A shaped `rank × groupCount` (group width auto-selected from tensor type: 32 for
Q4_K/Q5_K, 16 for Q6_K). This is a grouped-adapter algorithm distinct from QLoRA; it does not
use NF4 or double quantization. Shared Q4_K / Q5_K / Q6_K codecs (`GgufQuantCodec`, encoder id
`juno-kquant-v1`) back both QA-LoRA and the projected merge policy below.

Merge policies (`--lora-merge`):

| Policy | Behaviour |
|--------|-----------|
| `f32-preserve` (default) | Adapted tensors written as F32 |
| `source-type-projected` | Decode → add delta → encode with `juno-kquant-v1` (approximate requantization; reports delta retention / RMSE) |
| `sidecar-only` | Forbids bake-in merge; use overlay playback |

Exact QA-LoRA zero-point merge into K-quants is **not** supported: K-quant block formats cannot
generally absorb an arbitrary learned additive shift without an approximation step. Overlay
(sidecar, i.e. `--lora-play`) and F32 preserve remain the safe production paths for merges where
exactness matters; `source-type-projected` is available where a smaller file outweighs a small,
measured quality cost.

## Programmatic API

```java
LoraMerge.Result r = LoraMerge.merge(
    Path.of("TinyLlama.Q4_K_M.gguf"),
    Path.of("TinyLlama.Q4_K_M.lora"),
    Path.of("TinyLlama.Q4_K_M-merged.gguf"));

System.out.println("Patched " + r.adaptersApplied() + " tensors");
// Patched 44 tensors
```

## Legal note on redistribution

Redistributing merged weights may raise questions regarding base-model and adapter licenses.
Juno does not provide a legal determination for this; see [Chapter 17](#ch-17) for the licensing
reference and [Chapter 19](#ch-19) for the EU AI Act angle on distributing fine-tuned models. If
you are unsure, wait until those questions are resolved for your situation, or contact the
project at [dev@ml.cab](mailto:dev@ml.cab?subject=Help%20Request) — pull requests that resolve
open gaps are also welcome.

---

[← Chapter 9: Training and Inference Workflows](#ch-09) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 11: Model Support Matrix →](#ch-11)
