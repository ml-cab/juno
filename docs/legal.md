# Legal Q&A (informal)

This is **not legal advice**. Juno is licensed under **Apache License 2.0**; see [LICENSE](../LICENSE).

## Source code vs model weights

The Juno **software** license does not grant rights to any third-party **model weights** (GGUF or other). Obtain and comply with each model’s license and Hugging Face or distributor terms separately.

## LoRA merge (`./juno merge`)

Merge writes a **new GGUF** combining a frozen base checkpoint with trained adapter deltas. That output may be a **derivative work** of the base model and training data; whether you may **redistribute** it depends on the base license, adapter provenance, and your jurisdiction.

Open questions to resolve with counsel: stacking of **Llama / Mistral / Phi / community** licenses; **commercial use** flags; obligations when distributing merged binaries vs keeping them internal.

## Operational checklist

- Keep copies of license texts for every base GGUF you ship.
- Document adapter training sources if redistribution is planned.
- Prefer serving merged models **on-prem** until compliance is explicit.

See implementation pointers in [features/lora-and-merge.md](features/lora-and-merge.md) and [LoRA.md](LoRA.md).
