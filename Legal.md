# Juno — Legal Reference

This document is **not legal advice**. It consolidates legal context for contributors,
operators, and commercial users. Consult a qualified attorney for specific decisions.

---

## 1. Project License

Juno is licensed under the **Apache License 2.0** (SPDX: `Apache-2.0`).
The full text is in [LICENSE](../LICENSE). SPDX header in every source file:

```
// SPDX-License-Identifier: Apache-2.0
```

**What Apache 2.0 permits:**

- Use, copy, modify, and distribute the software, in source or binary form, for any
  purpose including commercial production use.
- Sublicense and sell products built on or with Juno.
- Use privately without disclosure of modifications.

**What Apache 2.0 requires:**

- Retain all existing copyright and license notices.
- State significant changes made to the source files.
- Include a copy of the Apache 2.0 license in any redistribution.
- Include the NOTICE file (or its equivalent contents) in redistributions.

**What Apache 2.0 does not require:**

- Contributing modifications back (copyleft is not triggered).
- Payment of royalties or fees to the Juno project.
- Obtaining a separate commercial license.

There is no "Community Edition vs Enterprise Edition" split. The codebase is one,
and the license is Apache 2.0 in full.

---

## 2. Third-Party Model Weights

The Juno software license covers the engine code only. It does not grant any rights
to third-party model weights (GGUF files or other formats) that Juno loads at runtime.

Each model family has its own license. Representative examples:

| Model family        | License                  | Commercial use      | Key constraints                                    |
|---------------------|--------------------------|---------------------|----------------------------------------------------|
| LLaMA 3 (Meta)      | Meta Llama 3 Community   | Yes, with conditions | Attribution; >700M MAU requires separate agreement |
| Mistral 7B          | Apache 2.0               | Yes                 | Standard Apache 2.0 terms                          |
| Phi-3 / Phi-3.5     | MIT                      | Yes                 | Standard MIT terms                                 |
| Gemma 2 (Google)    | Gemma Terms of Use       | Yes, with conditions | Prohibited use policy applies                      |

**Operator responsibility:** Obtain the model, review its license, and comply with
its terms independently of Juno. Juno does not vet model contents, provenance, or
compliance status. Keep copies of license texts for every base GGUF you deploy.

---

## 3. LoRA Fine-Tuning and the `merge` Command

### 3.1 Adapter files (`.lora`)

A `.lora` checkpoint produced by Juno contains delta weights derived from a base model
and your training data. Its legal status as a derivative work is unsettled and
jurisdiction-dependent. Conservative position: treat a `.lora` file as a derivative
of the base model and apply the base model's license to its redistribution.

### 3.2 Merged GGUFs

`./juno merge` writes a new GGUF combining frozen base weights with adapter deltas.
The resulting file is more likely to be considered a derivative work of the base model
than the `.lora` adapter alone. Before redistributing a merged GGUF:

1. Confirm the base model license permits redistribution of derivative works.
2. Confirm your training data does not introduce additional copyright claims.
3. If the base model requires attribution, include it in any release artifact.

Models on which redistribution of merged outputs is known to be permitted under
their standard license (as of 2026-06): Mistral 7B (Apache 2.0), Phi-3 (MIT).

Models requiring additional review before redistribution: LLaMA 3 (Meta license
conditions), any model with a non-commercial or prohibited-use clause.

### 3.3 Training data

Juno does not inspect training data. You are responsible for ensuring that data fed
to the LoRA training pipeline does not infringe third-party copyrights and complies
with the terms of any dataset license. Models trained on proprietary or licensed data
may carry obligations that survive into the resulting adapter and merged weights.

---

## 4. Patent Grant

The Apache 2.0 license includes an express patent grant from each contributor for
patents that are necessarily infringed by their contributions. This grant is
automatically terminated if you initiate patent litigation alleging that Juno
infringes a patent.

Juno does not represent that use of the software is free from third-party patent
claims, particularly in the areas of transformer architectures, GPU matmul, and
quantization methods.

---

## 5. Contributor License Agreement (CLA)

All contributions to the Juno repository are accepted under the terms described in
[docs/CLA.md](CLA.md). By opening a pull request you confirm that you have read
and agree to those terms.

Summary: contributors grant the project maintainers a perpetual, irrevocable,
royalty-free license to use and relicense their contributions under Apache 2.0 or
any future OSI-approved license the project adopts. Contributors retain their
copyright.

A separate Corporate CLA is available for contributions made on behalf of an
employer. Contact dev@ml.cab before submitting substantial employer-owned code.

---

## 6. Trademark

"Juno" and "Java Unified Neural Orchestration" are project names of the ml-cab
collective. Apache 2.0 does not grant trademark rights.

Permitted uses:
- Truthfully referring to the Juno project or software.
- Stating that your product is "powered by Juno" or "based on Juno."
- Using the name in academic publications and neutral comparisons.

Prohibited uses (without prior written permission):
- Implying official affiliation with or endorsement by the Juno project.
- Using "Juno" as part of the name of a competing inference product or service.
- Registering a trademark, domain, or service name that includes "Juno" in a way
  that could cause confusion with the project.

---

## 7. Export Control

Juno is cryptography-free software. It does not implement or bundle encryption
algorithms and therefore is not subject to EAR or ITAR cryptography controls under
US export regulations.

However, LLM technology and GPU compute are subject to evolving US and EU export
control rules. Operators deploying Juno in cross-border or government contexts should
review current Commerce Department (BIS) Entity List and EAR Part 744 restrictions
independently. The Juno project makes no representations about the export status of
the software or of third-party model weights loaded by it.

---

## 8. EU AI Act

Juno is infrastructure, not an AI system. The regulatory obligations under EU
Regulation 2024/1689 fall on the entity that operates Juno in production to serve
end users. The engine's compliance gap analysis is in [EU-AI-Act-compliance.md](EU-AI-Act-compliance.md).

Summary of operator obligations by deployment context:

| Deployment context                     | Minimum obligation                    |
|----------------------------------------|---------------------------------------|
| Internal developer tooling only        | None mandatory                        |
| Public-facing chat or text generation  | Article 50 AI disclosure (trivial)    |
| High-risk domains (employment, credit) | Full Chapter III compliance (complex) |
| Distribution of merged GGUF models     | Possible GPAI provider obligations    |

The project will provide an operator compliance guide template as part of its release
artifacts. See `EU-AI-Act-compliance.md` section 6 for prioritised remediation steps.

---

## 9. Open-Source Sustainability and Commercial Services

Juno is and will remain Apache 2.0 open-source. The project sustains itself through:

- Paid support contracts and SLAs (no additional license rights required).
- Paid integration and consulting engagements.
- Donations via GitHub Sponsors and Open Collective.
- Grants from open-source and research funding bodies.

None of these arrangements restrict community access to the source code or create
a privileged "commercial edition." See [docs/commercial.md](commercial.md) for service
terms and [FUNDING.md](../FUNDING.md) for donation channels.

---

## 10. Disclaimer of Warranties

As stated in the Apache 2.0 license, Juno is distributed WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied, including without limitation
any warranties of MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, or
NON-INFRINGEMENT. The entire risk as to the quality and performance of the software
is with you.

The project maintainers are not liable for any damages arising from use of the
software, including but not limited to lost profits, data loss, or inference
errors in production deployments.

---

*Legal questions: dev@ml.cab*