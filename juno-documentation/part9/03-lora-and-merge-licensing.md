(ch-9-3)=
# 9.3. LoRA and Merge Licensing

> This document is not legal advice. Consult a qualified attorney for specific decisions.

## Adapter files (`.lora`)

A `.lora` checkpoint produced by Juno contains delta weights derived from a base model and your
training data. Its legal status as a derivative work is unsettled and jurisdiction-dependent.
Conservative position: treat a `.lora` file as a derivative of the base model and apply the base
model's license to its redistribution.

## Merged GGUFs

`./juno merge` writes a new GGUF combining frozen base weights with adapter deltas. The
resulting file is more likely to be considered a derivative work of the base model than the
`.lora` adapter alone. Before redistributing a merged GGUF:

1. Confirm the base model license permits redistribution of derivative works.
2. Confirm your training data does not introduce additional copyright claims.
3. If the base model requires attribution, include it in any release artifact.

Models on which redistribution of merged outputs is known to be permitted under their standard
license (as of 2026-06): Mistral 7B (Apache 2.0), Phi-3 (MIT).

Models requiring additional review before redistribution: LLaMA 3 (Meta license conditions), and
any model with a non-commercial or prohibited-use clause.

## Training data

Juno does not inspect training data. You are responsible for ensuring that data fed to the LoRA
training pipeline does not infringe third-party copyrights and complies with the terms of any
dataset license. Models trained on proprietary or licensed data may carry obligations that
survive into the resulting adapter and merged weights.

## See also

- [Chapter 9.2 -- Third-Party Model Weights](#ch-9-2)
- [Chapter 4.5 -- Merging Adapters](#ch-4-5)

---

[<- 9.2 Third-Party Model Weights](#ch-9-2) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [9.4 Contributor License Agreement ->](#ch-9-4)
