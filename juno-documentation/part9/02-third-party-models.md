(ch-9-2)=
# 9.2. Third-Party Model Weights

> This document is not legal advice. Consult a qualified attorney for specific decisions.

The Juno software license covers the engine code only. It does not grant any rights to
third-party model weights (GGUF files or other formats) that Juno loads at runtime.

Each model family has its own license. Representative examples:

| Model family     | License                | Commercial use        | Key constraints                                     |
|-------------------|-------------------------|-------------------------|--------------------------------------------------------|
| LLaMA 3 (Meta)    | Meta Llama 3 Community | Yes, with conditions  | Attribution required; over 700M MAU requires a separate agreement |
| Mistral 7B        | Apache 2.0             | Yes                    | Standard Apache 2.0 terms                               |
| Phi-3 / Phi-3.5   | MIT                     | Yes                    | Standard MIT terms                                      |
| Gemma 2 (Google)  | Gemma Terms of Use     | Yes, with conditions  | Prohibited use policy applies                           |

**Operator responsibility:** obtain the model, review its license, and comply with its terms
independently of Juno. Juno does not vet model contents, provenance, or compliance status. Keep
copies of license texts for every base GGUF you deploy.

## See also

- [Chapter 9.1 -- License and Patents](#ch-9-1)
- [Chapter 9.3 -- LoRA and Merge Licensing](#ch-9-3)

---

[<- 9.1 License and Patents](#ch-9-1) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [9.3 LoRA and Merge Licensing ->](#ch-9-3)
