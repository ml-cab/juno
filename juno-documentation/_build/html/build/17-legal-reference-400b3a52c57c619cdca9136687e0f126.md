(ch-17)=
# 17. Legal Reference: License, Model Weights, Trademark, Export Control

> This chapter is not legal advice. It consolidates legal context for contributors, operators,
> and commercial users. Consult a qualified attorney for specific decisions.

## Project license

Juno is licensed under the **Apache License 2.0** (SPDX: `Apache-2.0`). The full text is in
`LICENSE`. The SPDX header in every source file reads:

```
// SPDX-License-Identifier: Apache-2.0
```

**Apache 2.0 permits:** use, copy, modify, and distribute the software in source or binary form
for any purpose including commercial production use; sublicensing and selling products built on
or with Juno; private use without disclosure of modifications.

**Apache 2.0 requires:** retaining all existing copyright and license notices; stating
significant changes made to the source files; including a copy of the Apache 2.0 license in any
redistribution; including the `NOTICE` file (or its equivalent contents) in redistributions.

**Apache 2.0 does not require:** contributing modifications back (copyleft is not triggered);
payment of royalties or fees to the Juno project; obtaining a separate commercial license.

There is no "Community Edition vs Enterprise Edition" split — the codebase is one, and the
license is Apache 2.0 in full (see [Chapter 18](#ch-18) for how the project sustains itself
without such a split).

## Third-party model weights

The Juno software license covers the engine code only. It does not grant any rights to
third-party model weights (GGUF files or other formats) that Juno loads at runtime. Each model
family has its own license:

| Model family | License | Commercial use | Key constraints |
|---|---|---|---|
| LLaMA 3 (Meta) | Meta Llama 3 Community | Yes, with conditions | Attribution; >700M MAU requires separate agreement |
| Mistral 7B | Apache 2.0 | Yes | Standard Apache 2.0 terms |
| Phi-3 / Phi-3.5 | MIT | Yes | Standard MIT terms |
| Gemma 2 (Google) | Gemma Terms of Use | Yes, with conditions | Prohibited use policy applies |

**Operator responsibility:** obtain the model, review its license, and comply with its terms
independently of Juno. Juno does not vet model contents, provenance, or compliance status. Keep
copies of the license text for every base GGUF you deploy.

## LoRA fine-tuning and the `merge` command

**Adapter files (`.lora`).** A `.lora` checkpoint produced by Juno contains delta weights
derived from a base model and your training data (mechanics in [Chapter 8](#ch-08)). Its legal
status as a derivative work is unsettled and jurisdiction-dependent. The conservative position
is to treat a `.lora` file as a derivative of the base model and apply the base model's license
to its redistribution.

**Merged GGUFs.** `./juno merge` (mechanics in [Chapter 10](#ch-10)) writes a new GGUF combining
frozen base weights with adapter deltas. The resulting file is more likely to be considered a
derivative work of the base model than the `.lora` adapter alone. Before redistributing a merged
GGUF: confirm the base model license permits redistribution of derivative works; confirm your
training data does not introduce additional copyright claims; if the base model requires
attribution, include it in any release artifact.

As of 2026-06, models on which redistribution of merged outputs is known to be permitted under
their standard license: Mistral 7B (Apache 2.0), Phi-3 (MIT). Models requiring additional review
before redistribution: LLaMA 3 (Meta license conditions), any model with a non-commercial or
prohibited-use clause.

**Training data.** Juno does not inspect training data. You are responsible for ensuring that
data fed to the LoRA training pipeline does not infringe third-party copyrights and complies
with the terms of any dataset license. Models trained on proprietary or licensed data may carry
obligations that survive into the resulting adapter and merged weights.

## Patent grant

The Apache 2.0 license includes an express patent grant from each contributor for patents that
are necessarily infringed by their contributions. This grant is automatically terminated if you
initiate patent litigation alleging that Juno infringes a patent.

Juno does not represent that use of the software is free from third-party patent claims,
particularly in the areas of transformer architectures, GPU matmul, and quantization methods.

## Contributor License Agreement

All contributions to the Juno repository are accepted under the terms described in
[Chapter 16](#ch-16). By opening a pull request you confirm that you have read and agree to
those terms.

In summary: contributors grant the project maintainers a perpetual, irrevocable, royalty-free
license to use and relicense their contributions under Apache 2.0 or any future OSI-approved
license the project adopts. Contributors retain their copyright. A separate Corporate CLA is
available for contributions made on behalf of an employer — contact
[dev@ml.cab](mailto:dev@ml.cab) before submitting substantial employer-owned code.

## Trademark

"Juno" and "Java Unified Neural Orchestration" are project names of the ml-cab collective.
Apache 2.0 does not grant trademark rights.

**Permitted:** truthfully referring to the Juno project or software; stating that your product
is "powered by Juno" or "based on Juno"; using the name in academic publications and neutral
comparisons.

**Prohibited without prior written permission:** implying official affiliation with or
endorsement by the Juno project; using "Juno" as part of the name of a competing inference
product or service; registering a trademark, domain, or service name that includes "Juno" in a
way that could cause confusion with the project.

## Export control

Juno is cryptography-free software. It does not implement or bundle encryption algorithms and
therefore is not subject to EAR or ITAR cryptography controls under US export regulations.

However, LLM technology and GPU compute are subject to evolving US and EU export control rules.
Operators deploying Juno in cross-border or government contexts should independently review
current Commerce Department (BIS) Entity List and EAR Part 744 restrictions. The Juno project
makes no representations about the export status of the software or of third-party model
weights loaded by it.

## EU AI Act

Juno is infrastructure, not an AI system. The regulatory obligations under EU Regulation
2024/1689 fall on the entity that operates Juno in production to serve end users. The engine's
full compliance gap analysis is in [Chapter 19](#ch-19). Summary of operator obligations by
deployment context:

| Deployment context | Minimum obligation |
|---|---|
| Internal developer tooling only | None mandatory |
| Public-facing chat or text generation | Article 50 AI disclosure (trivial) |
| High-risk domains (employment, credit) | Full Chapter III compliance (complex) |
| Distribution of merged GGUF models | Possible GPAI provider obligations |

## Open-source sustainability and commercial services

Juno is and will remain Apache 2.0 open-source. The project sustains itself through paid
support contracts and SLAs (no additional license rights required), paid integration and
consulting engagements, donations via GitHub Sponsors and Open Collective, and grants from
open-source and research funding bodies. None of these arrangements restrict community access
to the source code or create a privileged "commercial edition." See [Chapter 18](#ch-18) for
service terms and `FUNDING.md` for donation channels.

## Disclaimer of warranties

As stated in the Apache 2.0 license, Juno is distributed WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied, including without limitation any warranties of
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, or NON-INFRINGEMENT. The entire risk
as to the quality and performance of the software is with you. The project maintainers are not
liable for any damages arising from use of the software, including but not limited to lost
profits, data loss, or inference errors in production deployments.

*Legal questions: [dev@ml.cab](mailto:dev@ml.cab)*

---

[← Chapter 16: Contributor License Agreement](#ch-16) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 18: Commercial Services →](#ch-18)
