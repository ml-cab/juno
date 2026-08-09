(ch-19)=
# 19. EU AI Act Compliance Analysis

**Regulation:** EU 2024/1689 (Artificial Intelligence Act), in force since 1 August 2024.
**Subject:** Juno — Java Unified Neural Orchestration, a distributed LLM inference and
fine-tuning engine.

## What Juno is (and is not) under the Act

The EU AI Act regulates **AI systems** and **General-Purpose AI (GPAI) models**. Classifying
Juno correctly is the first and most consequential step.

Juno is an inference and fine-tuning infrastructure engine. It reads third-party GGUF model
files (LLaMA, Mistral, Phi-3, and others — see [Chapter 11](#ch-11) for the full support
matrix), distributes transformer computation across JVM nodes via gRPC, and exposes an
OpenAI-compatible REST API (`POST /v1/chat/completions`, see [Chapter 5](#ch-05)). It does not
contain, produce, or distribute a GPAI model itself.

Under Article 3 of the Act, an **AI system** is a machine-based system that infers outputs such
as predictions, content, or recommendations from inputs. When Juno is running with a loaded
model and a user sends a prompt, the resulting deployment is an AI system. Juno is the runtime
infrastructure that makes that AI system operational.

| Act role | Juno mapping |
|---|---|
| GPAI model provider | Not Juno — applies to whoever releases the base GGUF (Meta, Microsoft, etc.) |
| AI system provider | The entity that deploys the Juno + model combination and makes it available to users |
| Deployer | An organisation using that deployed Juno instance internally or on behalf of users |
| Third-party tool supplier | Juno itself, as infrastructure enabling providers |

Most obligations fall on whoever operates Juno in production, not on the Juno codebase itself.
However, as a third-party tool supplier to AI system providers, Juno has a responsibility to
give providers the capability to meet their obligations (Article 25, Article 53 on downstream
provider cooperation).

## Risk tier of a Juno-served AI system

The Act uses a four-tier risk model. The applicable tier depends entirely on the use case the
deployed system is put to, not on the inference infrastructure.

**Tier 1 — Prohibited practices (Article 5).** None of Juno's built-in capabilities constitute
prohibited practices. Juno has no subliminal manipulation, no social scoring, no biometric
identification, no real-time public-space surveillance. Juno is content-agnostic
infrastructure; whether a deployed model produces prohibited outputs is the operator's
responsibility.

**Tier 2 — High-risk AI systems (Annex III).** If Juno is used to serve a model in any of the
following domains, the deployment is high-risk and the full Chapter III obligations apply:
biometric identification or categorisation; management of critical infrastructure; education
and vocational training (admission, assessment); employment (recruitment, performance
evaluation); access to essential services (credit scoring, insurance, social benefits); law
enforcement; migration and asylum; administration of justice.

**Tier 3 — Limited-risk AI systems (Article 50).** A Juno deployment serving a general-purpose
chat or text-generation function — the primary documented use case — is at minimum a
limited-risk AI system. The sole mandatory obligation at this tier is transparency: users must
be informed they are interacting with an AI system, unless it is obvious from context.

**Tier 4 — Minimal risk.** Pure internal developer tooling with no end-user interaction — for
example, using the `juno local` REPL privately for development, see [Chapter 4](#ch-04) — falls
here with no mandatory obligations.

## Compliance gap analysis

The analysis below evaluates the current Juno codebase against the obligations that the Juno
operator (AI system provider/deployer) must satisfy, and which the Juno engine should ideally
support.

### Article 50 — Transparency to users (limited-risk, mandatory)

**Requirement:** natural persons must be notified that they are interacting with an AI system
when using a conversational AI or a system that generates content.

**Current state:** Juno's OpenAI-compatible REST API (`OpenAiChatHandler`,
`InferenceApiServer`) and the REPL (`ConsoleMain`) return raw model output with no disclosure
header, response field, or banner indicating AI interaction. The OpenAPI spec (`juno-api.yaml`)
documents no disclosure field or mechanism.

**Gap: critical.** This is the minimum mandatory obligation for any public-facing deployment,
and it is missing entirely from the API layer. What is needed: a configurable disclosure field
in the response envelope (e.g. `x_juno_ai_disclosure: true` alongside the `x_juno_*` extensions
described in [Chapter 5](#ch-05)) and/or an operator-configurable system-level banner. For
streaming mode, a first SSE event with disclosure metadata before token emission.

### Article 12 — Automatic logging / record-keeping (high-risk)

**Requirement:** high-risk AI systems must automatically log events throughout their lifecycle
to enable traceability and post-market monitoring.

**Current state:** Juno's JFR instrumentation (see [Chapter 2](#ch-02) and [Chapter 13](#ch-13))
covers `juno.MatVec`, `juno.ForwardPass`, `juno.TokenProduced`, `juno.Tokenizer`,
`juno.TemplateFormat`, and `juno.LoraTrainStep`, plus the wider LoRA event catalog in
[Chapter 8](#ch-08). These are performance and observability events — latency, throughput,
token position — not compliance audit records. There is no logging of who made a request, what
input was provided, what output was returned, which model version was used, or what session
parameters were active.

**Gap: high** for high-risk deployments, partial for limited-risk. The JFR infrastructure is
solid and could be extended, but currently records no information required for a regulatory
audit trail. What is needed: an audit log facility, separate from JFR metrics, capturing at
minimum timestamp, session ID, model ID, input hash or length, output hash or length, sampling
parameters, finish reason, and a per-request unique ID. The existing `x_juno_session_id` and
`chatCompletionId` fields are good foundations.

### Article 9 — Risk management system (high-risk)

**Requirement:** providers of high-risk AI systems must establish, implement, document, and
maintain a risk management system covering the full model lifecycle: identification of known
and reasonably foreseeable risks, evaluation of residual risk, and post-market monitoring.

**Current state:** not present. Juno has a `CircuitBreaker` and `HealthReactor` for operational
fault tolerance and a `FaultTolerantPipeline` for node failure recovery — operational resilience
features, not risk management in the regulatory sense. There is no risk register, no risk
assessment documentation, no process for evaluating misuse scenarios.

**Gap: high** for high-risk deployments; operators must build and maintain this entirely
outside the engine. What is needed is primarily a process/documentation obligation, not a code
obligation: an operator-facing risk documentation template as part of release artifacts, a
configurable allowed-use-case declaration at startup that rejects requests outside declared
scope, and hooks for operator-supplied content filtering before and after generation.

### Article 13 — Transparency and provision of information to deployers (high-risk)

**Requirement:** high-risk AI systems must be designed to enable deployers to understand how the
system works, its capabilities and limitations, under which conditions it may produce unreliable
outputs, and what human oversight measures apply.

**Current state:** Juno's engineering documentation ([Chapter 2](#ch-02), [Chapter 3](#ch-03),
the OpenAPI spec) is comprehensive, but there is no AI Act-oriented documentation covering model
performance characteristics across demographic groups, known failure modes, confidence or
uncertainty indications in outputs, or instructions for deployer human oversight configuration.

**Gap: medium.** Technical documentation is strong; AI Act-specific disclosure content is
absent. What is needed: an operator guide addendum addressing limitations of served models
(accuracy, bias, hallucination rates), conditions under which the system should not be used
autonomously, and how to configure the human oversight hooks described under Article 14 below.

### Article 14 — Human oversight (high-risk)

**Requirement:** high-risk AI systems must include built-in operational constraints enabling
human oversight: the ability to interrupt operation, understand outputs sufficiently to detect
and correct anomalies, and optionally require dual-person confirmation before acting on outputs.

**Current state:** Juno provides no human-in-the-loop mechanism. The API is fire-and-forget — a
request produces a completion. There is no mechanism for flagging low-confidence outputs,
requiring operator confirmation before delivery, or routing uncertain cases to human review.
The `RequestScheduler` priority system (`x_juno_priority`, see [Chapter 5](#ch-05)) is a
throughput mechanism, not an oversight mechanism.

**Gap: high** for high-risk deployments; nothing in the current architecture supports Article
14. What is needed: an optional `x_juno_require_review` flag causing completions to be held in
a review queue, a confidence-threshold configuration below which outputs are flagged, and a
review endpoint (`POST /v1/completions/{id}/approve`) — all new coordinator features.

### Article 10 — Data governance for training (high-risk + LoRA)

**Requirement:** training, validation, and testing datasets for high-risk AI systems must be
relevant, sufficiently representative, and free of errors to the extent possible. Providers must
document data governance practices.

**Current state:** Juno's LoRA fine-tuning facility ([Chapter 8](#ch-08),
[Chapter 9](#ch-09)) allows operators to fine-tune models on arbitrary data via `/train` and
`/train-qa`. There is no validation, filtering, or documentation of training data quality,
provenance, or bias. Training loss is logged per step via JFR, but nothing about the training
corpus itself is recorded.

**Gap: high** for any operator using LoRA fine-tuning in a regulated context — the fine-tuning
pipeline has no data governance hooks. What is needed: data lineage logging for LoRA training
sessions (source, volume, timestamp, hash of training corpus), configurable data validation
hooks before ingestion into the training loop, and a per-adapter documentation artifact
generated at `merge` time (see [Chapter 10](#ch-10)) capturing training data provenance.

### Articles 53–55 — GPAI model obligations (if applicable)

**Requirement:** providers of GPAI models must prepare and keep up-to-date technical
documentation, make available information for downstream providers, and implement a copyright
compliance policy and publish a training data summary.

**Applicability to Juno:** Juno is not a GPAI model provider — it serves third-party GGUF
models. However, if an operator uses Juno's LoRA facility to substantially fine-tune a base
model and then distributes that fine-tuned model (via `./juno merge` producing a new GGUF), the
operator may become a GPAI model provider under the Act if the resulting model has
general-purpose capability. The base models Juno supports (LLaMA 3, Mistral 7B, Phi-3.5) are
themselves GPAI models whose providers (Meta, Mistral AI, Microsoft) already carry these
obligations.

**Gap: low** for Juno itself — the engine correctly positions itself as infrastructure.
**Medium** for operators who fine-tune and redistribute merged models, who may inadvertently
become GPAI providers without realising it. What is needed: clear operator guidance stating
that fine-tuned and merged models may trigger GPAI obligations (see the legal note in
[Chapter 10](#ch-10) and the licensing discussion in [Chapter 17](#ch-17)), and a warning
emitted by `./juno merge` when producing a new GGUF.

### Article 11 — Technical documentation (high-risk, Annex IV)

**Requirement:** high-risk AI system providers must maintain Annex IV technical documentation
covering general description, design specifications, training methodology, performance metrics,
risk management documentation, post-market monitoring plan, and a declaration of conformity.

**Current state:** Juno's technical documentation covers the engineering architecture well (this
book being an example), but Annex IV-required content — system-level accuracy metrics, bias
evaluation, conformity assessment outcomes, post-market monitoring plan — is entirely absent.

**Gap: high** for high-risk deployments. What is needed: an Annex IV documentation template in
the release artifacts, with guidance for operators to populate it for their specific deployment
context and model choice.

### Article 15 — Accuracy, robustness, cybersecurity (high-risk)

**Requirement:** high-risk AI systems must achieve appropriate levels of accuracy and
robustness, and must be resilient against adversarial attacks, including data poisoning and
model manipulation.

**Current state:** Robustness is solid — `FaultTolerantPipeline` handles node failure and
retry, and `HealthReactor`/`CircuitBreaker` handle node health degradation. Accuracy has no
benchmarks, evaluation pipelines, or performance declarations in the codebase beyond the
throughput matrix in [Chapter 13](#ch-13), which measures speed rather than output quality.
Cybersecurity is the weakest area: `InferenceApiServer` (Javalin) exposes an unauthenticated
HTTP API by default, with no authentication, rate limiting beyond the `RequestScheduler` queue,
input sanitisation, or built-in TLS configuration. AWS deployment scripts handle security group
configuration externally (see [Chapter 7](#ch-07)), but the engine itself has no security
layer.

**Gap: medium-high.** Operational resilience is good; security posture is weak for regulated
deployment. What is needed: TLS support and API key authentication as first-class configuration
options in `InferenceApiServer`, input length and content validation hooks, and rate limiting
per API client distinct from the internal scheduler priority.

## Summary table

| EU AI Act requirement | Article | Risk tier | Current status | Gap severity |
|---|---|---|---|---|
| User transparency / AI disclosure | 50 | Limited+ | Absent | Critical |
| Automatic logging / audit trail | 12 | High | Performance metrics only | High |
| Risk management system | 9 | High | Absent | High |
| Data governance (LoRA training) | 10 | High | Absent | High |
| Human oversight mechanisms | 14 | High | Absent | High |
| Technical documentation (Annex IV) | 11 | High | Engineering docs only | High |
| Transparency to deployers | 13 | High | Engineering docs only | Medium |
| Cybersecurity / authentication | 15 | High | Not built in | Medium |
| GPAI obligations (merge/distribute) | 53–55 | GPAI | Operator guidance missing | Medium |
| Operational robustness | 15 | High | Strong (circuit breaker, FTP) | Low |

## What Juno does well

Several existing features align with regulatory intent: the circuit breaker and fault-tolerant
pipeline support the reliability and human-oversight goals of Articles 14 and 15; the JFR
instrumentation is a strong observability foundation that can be extended into compliance
logging without architectural change; the session ID and completion ID in the API layer are
building blocks for an audit trail; the Apache 2.0 license aligns with the Act's encouragement
of open-source approaches and the reduced obligations that apply to open-source infrastructure
tools (Article 25(2)); LoRA adapter isolation — adapters applied read-only, base GGUF never
modified, see [Chapter 8](#ch-08) — reduces the blast radius of fine-tuning and supports
testable, auditable model variants; and the structured OpenAPI spec provides a machine-readable
API contract that facilitates compliance documentation.

## Prioritised remediation recommendations

In order of regulatory urgency for an operator targeting EU deployment:

**1. Implement Article 50 AI disclosure** (immediate, low effort). Add a response field and
configurable startup banner: a non-streaming field such as `"x_juno_system_disclosure": "This
response was generated by an AI system."`, and for streaming, a metadata SSE event before the
first token. Configurable off for API-to-API use without end users.

**2. Add compliance audit logging** (short-term, medium effort). A structured audit log,
distinct from JFR, capturing request ID, session ID, model ID, model version/SHA, input token
count, output token count, finish reason, sampling parameters, timestamp, and client identifier,
written to a configurable sink.

**3. Add authentication and TLS to `InferenceApiServer`** (short-term, medium effort). Bearer
token / API key authentication middleware, plus TLS configuration via JVM keystore — table-stakes
security requirements for any production AI deployment.

**4. Add operator documentation for high-risk contexts** (medium-term, low effort). A compliance
guide covering how to populate Annex IV technical documentation for a Juno deployment,
limitations of served models, how to configure for high-risk use cases, and an explicit warning
that LoRA merge outputs may trigger GPAI obligations.

**5. Add data governance hooks to the LoRA training pipeline** (medium-term, medium effort).
Before ingesting training data, log corpus hash, token count, source label, and timestamp to a
training provenance record; generate a provenance artifact alongside each `.lora` checkpoint;
emit a warning when producing merged GGUFs.

**6. Design human oversight hooks** (longer-term, high effort). A review-queue mode in
`RequestScheduler` for operators who need to intercept completions before delivery, and a
confidence/uncertainty signal hook — even a simple output-length-relative-to-max-tokens
heuristic — to flag potentially truncated or degenerate outputs.

## Conclusion

Juno, as an open-source LLM inference engine, does not itself constitute an AI system or GPAI
model under the EU AI Act. The regulatory obligations fall on the entity that operates Juno in
production to serve end users or downstream systems.

The engine is technically sophisticated and operationally well-designed. Its fault tolerance,
observability, and structured API make it a credible foundation for compliant deployments.
However, the codebase currently provides no compliance-oriented features: no AI disclosure, no
audit logging, no authentication, no human oversight hooks, and no data governance for the LoRA
training pipeline.

An operator deploying Juno in a limited-risk context (general chat assistant) faces one critical
gap — Article 50 AI disclosure — which is trivially fixable. An operator deploying Juno in a
high-risk context (employment screening, credit, healthcare triage) would face the full Chapter
III obligation set and would need to build substantial compliance infrastructure on top of the
current engine.

The most impactful near-term investment for the Juno project is Article 50 disclosure in the API
response, structured audit logging, and API authentication — three changes that collectively
address the most urgent regulatory exposure across all deployment contexts.

---

[← Chapter 18: Commercial Services](#ch-18) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [References](../references.md)
