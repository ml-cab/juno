# Juno — EU AI Act Compliance Analysis

**Regulation:** EU 2024/1689 (Artificial Intelligence Act), in force 1 August 2024
**Subject:** jUno — Java Unified Neural Orchestration (distributed LLM inference and fine-tuning framework)
**Codebase snapshot:** from 2026-05-05

---

## 1. What Juno Is (and Is Not) Under the Act

The EU AI Act regulates **AI systems** and **General-Purpose AI (GPAI) models**. Classifying Juno correctly is the first and most consequential step.

**Juno is an inference and fine-tuning infrastructure framework.** It reads third-party GGUF model files (LLaMA, Phi-3, Mistral, etc.), distributes transformer computation across JVM nodes via gRPC, and exposes an OpenAI-compatible REST API (`POST /v1/chat/completions`). It does not contain, produce, or distribute a GPAI model itself.

Under Article 3 of the Act, an **AI system** is a machine-based system that infers outputs such as predictions, content, or recommendations from inputs. When Juno is running with a loaded model and a user sends a prompt, the resulting deployment is an AI system. Juno is the **runtime infrastructure** that makes that AI system operational.

The entities regulated by the Act are:

| Act role | Juno mapping |
|---|---|
| GPAI model provider | Not Juno — applies to whoever releases the base GGUF (Meta, Microsoft, etc.) |
| AI system provider | The entity that deploys Juno + model combination and makes it available to users |
| Deployer | An organisation using that deployed Juno instance internally or on behalf of users |
| Third-party tool supplier | Juno itself (cab.ml), as infrastructure enabling providers |

This distinction matters: **most obligations fall on whoever operates Juno in production, not on the Juno codebase itself.** However, as a third-party tool supplier to AI system providers, Juno has a responsibility to give providers the capability to meet their obligations (Article 25, Article 53 on downstream provider cooperation).

---

## 2. Risk Tier of a Juno-Served AI System

The Act uses a four-tier risk model. The applicable tier depends entirely on the **use case** the deployed system is put to, not on the inference infrastructure.

**Tier 1 — Prohibited practices (Article 5):** None of Juno's built-in capabilities constitute prohibited practices. Juno has no subliminal manipulation, no social scoring, no biometric identification, no real-time public-space surveillance. Juno is content-agnostic infrastructure; whether a deployed model produces prohibited outputs is the operator's responsibility.

**Tier 2 — High-risk AI systems (Annex III):** If Juno is used to serve a model in any of the following domains, the deployment is high-risk and the full Chapter III obligations apply:

- Biometric identification or categorisation
- Management of critical infrastructure
- Education and vocational training (admission, assessment)
- Employment (recruitment, performance evaluation)
- Access to essential services (credit scoring, insurance, social benefits)
- Law enforcement
- Migration and asylum
- Administration of justice

**Tier 3 — Limited-risk AI systems (Article 50):** A Juno deployment serving a general-purpose chat or text generation function — the primary documented use case — is at minimum a limited-risk AI system. The sole mandatory obligation at this tier is **transparency**: users must be informed they are interacting with an AI system, unless it is obvious from context.

**Tier 4 — Minimal risk:** Pure internal developer tooling with no end-user interaction (e.g. using the `juno local` REPL privately for development) falls here with no mandatory obligations.

---

## 3. Compliance Gap Analysis and steps to improve

The analysis below evaluates the current Juno codebase against the obligations that the Juno **operator** (AI system provider/deployer) must satisfy, and which the Juno **framework** should ideally support.

### 3.1 Article 50 — Transparency to Users (Limited-Risk, MANDATORY)

**Requirement:** Natural persons must be notified that they are interacting with an AI system when using a conversational AI or a system that generates content.

**Current state:** Juno's OpenAI-compatible REST API (`OpenAiChatHandler`, `InferenceApiServer`) and the REPL (`ConsoleMain`) return raw model output with no disclosure header, response field, or banner indicating AI interaction. The OpenAPI spec (`juno-api.yaml`) documents no disclosure field or mechanism.

**Gap: CRITICAL.** This is the minimum mandatory obligation for any public-facing deployment. It is missing entirely from the API layer.

**What is needed:** A configurable disclosure field in the response envelope (e.g. `x_juno_ai_disclosure: true` in `OaiChatCompletionResponse`) and/or an operator-configurable system-level banner. For streaming mode (`SseTokenConsumer`), a first SSE event with disclosure metadata before token emission.

---

### 3.2 Article 12 — Automatic Logging / Record-Keeping (High-Risk)

**Requirement:** High-risk AI systems must automatically log events throughout their lifecycle to enable traceability and post-market monitoring.

**Current state:** Juno has JFR instrumentation (`JfrMetricsExtractor`, five custom event types: `juno.MatVec`, `juno.ForwardPass`, `juno.TokenProduced`, `juno.Tokenizer`, `juno.LoraTrainStep`). These are performance/observability events — latency, throughput, token position — not compliance audit records. There is no logging of: who made a request, what input was provided, what output was returned, which model version was used, or what session parameters were active.

**Gap: HIGH** for high-risk deployments; partial for limited-risk. JFR infrastructure is solid and could be extended, but currently records no information required for regulatory audit trails.

**What is needed:** An audit log facility (separate from JFR metrics) capturing at minimum: timestamp, session ID, model ID, input hash or length, output hash or length, sampling parameters, finish reason, and a per-request unique ID. The existing `x_juno_session_id` and `chatCompletionId` in `OpenAiChatHandler` are good foundations.

---

### 3.3 Article 9 — Risk Management System (High-Risk)

**Requirement:** Providers of high-risk AI systems must establish, implement, document, and maintain a risk management system covering the full model lifecycle: identification of known and reasonably foreseeable risks, evaluation of residual risk, and post-market monitoring.

**Current state:** Not present. Juno has a `CircuitBreaker` and `HealthReactor` for operational fault tolerance and a `FaultTolerantPipeline` for node failure recovery. These are infrastructure resilience features, not risk management in the regulatory sense. There is no risk register, no risk assessment documentation, no process for evaluating misuse scenarios.

**Gap: HIGH** for high-risk deployments. Operators must build and maintain this entirely outside the framework.

**What is needed:** This is primarily a process/documentation obligation, not a code obligation. However, Juno could provide: (a) an operator-facing risk documentation template as part of its release artifacts, (b) a configurable allowed-use-case declaration at startup that rejects requests outside declared scope, (c) hooks for operator-supplied content filtering before and after generation.

---

### 3.4 Article 13 — Transparency and Provision of Information to Deployers (High-Risk)

**Requirement:** High-risk AI systems must be designed to enable deployers to understand how the system works, its capabilities and limitations, under which conditions it may produce unreliable outputs, and what human oversight measures apply.

**Current state:** Juno's `README.md`, `howto.md`, and `arch.md` are comprehensive engineering documentation. The OpenAPI spec (`juno-api.yaml`) is detailed. However, there is no AI Act-oriented documentation covering: model performance characteristics across demographic groups, known failure modes, confidence or uncertainty indications in outputs, or instructions for deployer human oversight configuration.

**Gap: MEDIUM.** Technical documentation is strong; AI Act-specific disclosure content is absent.

**What is needed:** An operator guide addendum addressing: limitations of served models (accuracy, bias, hallucination rates), conditions under which the system should not be used autonomously, and how to configure the human oversight hooks described in Article 14.

---

### 3.5 Article 14 — Human Oversight (High-Risk)

**Requirement:** High-risk AI systems must include built-in operational constraints enabling human oversight: the ability to interrupt operation, understand outputs sufficiently to detect and correct anomalies, and optionally require dual-person confirmation before acting on outputs.

**Current state:** Juno provides no human-in-the-loop mechanism. The API is fire-and-forget: a request produces a completion. There is no mechanism for: flagging low-confidence outputs, requiring operator confirmation before delivery, or routing uncertain cases to human review. The `RequestScheduler` priority system (`HIGH/NORMAL/LOW`) is a throughput mechanism, not an oversight mechanism.

**Gap: HIGH** for high-risk deployments. Nothing in the current architecture supports Article 14.

**What is needed:** An optional `x_juno_require_review` flag causing completions to be held in a review queue, a confidence-threshold configuration below which outputs are flagged, and a review endpoint (`POST /v1/completions/{id}/approve`). These would be new coordinator features.

---

### 3.6 Article 10 — Data Governance for Training (High-Risk + LoRA)

**Requirement:** Training, validation, and testing datasets for high-risk AI systems must be relevant, sufficiently representative, and free of errors to the extent possible. Providers must document data governance practices.

**Current state:** Juno's LoRA fine-tuning facility (`LoraTrainableHandler`, `LoraAdamOptimizer`, `LoraAdapterSet`) allows operators to fine-tune models on arbitrary data. The `/train` and `/train-qa` REPL commands accept free-form training input. There is no validation, filtering, or documentation of training data quality, provenance, or bias. The `LoraAdamOptimizer` logs loss per step via JFR but records nothing about the training corpus.

**Gap: HIGH** for any operator using LoRA fine-tuning in a regulated context. The fine-tuning pipeline has no data governance hooks.

**What is needed:** Data lineage logging for LoRA training sessions (source, volume, timestamp, hash of training corpus), configurable data validation hooks before ingestion into the training loop, and a per-adapter documentation artifact generated at `merge` time capturing training data provenance.

---

### 3.7 Articles 53–55 — GPAI Model Obligations (if applicable)

**Requirement:** Providers of GPAI models must: (a) prepare and keep up-to-date technical documentation, (b) make available information for downstream providers, (c) implement a copyright compliance policy and publish a training data summary.

**Applicability to Juno:** Juno is not a GPAI model provider. It serves third-party GGUF models. However:

- If an operator uses Juno's LoRA facility to substantially fine-tune a base model and then distributes that fine-tuned model (e.g. via the `merge` command producing a new GGUF), the operator may become a GPAI model provider under the Act if the resulting model has general-purpose capability.
- The base models Juno supports (LLaMA 3, Mistral 7B, Phi-3.5) are GPAI models whose providers (Meta, Mistral AI, Microsoft) already carry these obligations. Juno's documentation should clarify this chain.

**Gap: LOW** for Juno itself; the framework correctly positions itself as infrastructure. **MEDIUM** for operators who fine-tune and redistribute merged models — they may inadvertently become GPAI providers without realising it.

**What is needed:** Clear operator guidance in the documentation: fine-tuned and merged models may trigger GPAI obligations. The `merge` command should emit a warning when producing a new GGUF.

---

### 3.8 Article 11 — Technical Documentation (High-Risk, Annex IV)

**Requirement:** High-risk AI system providers must maintain Annex IV technical documentation covering: general description, design specifications, training methodology, performance metrics, risk management documentation, post-market monitoring plan, and a declaration of conformity.

**Current state:** Juno's technical documentation (`arch.md`, `howto.md`, `LoRA.md`) covers the engineering architecture well. Annex IV-required content — system-level accuracy metrics, bias evaluation, conformity assessment outcomes, post-market monitoring plan — is entirely absent.

**Gap: HIGH** for high-risk deployments.

**What is needed:** An Annex IV documentation template in the release artifacts, with guidance for operators to populate it for their specific deployment context and model choice.

---

### 3.9 Article 15 — Accuracy, Robustness, Cybersecurity (High-Risk)

**Requirement:** High-risk AI systems must achieve appropriate levels of accuracy and robustness, and must be resilient against adversarial attacks, including data poisoning and model manipulation.

**Current state:**

- **Robustness:** The `FaultTolerantPipeline` handles node failure and retry. The `HealthReactor`/`CircuitBreaker` handles node health degradation. These are solid operational features.
- **Accuracy:** No accuracy benchmarks, evaluation pipelines, or performance declarations exist in the codebase.
- **Cybersecurity:** Juno exposes an unauthenticated HTTP API by default. `InferenceApiServer` (Javalin) has no authentication, rate limiting beyond the `RequestScheduler` queue, input sanitisation, or TLS configuration built in. The deployment scripts (`juno-deploy.sh`) handle AWS security group configuration externally, but the framework itself has no security layer.

**Gap: MEDIUM–HIGH.** Operational resilience is good; security posture is weak for regulated deployment.

**What is needed:** TLS support and API key authentication as first-class configuration options in `InferenceApiServer`. Input length and content validation hooks. Rate limiting per API client, distinct from the internal scheduler priority.

---

## 4. Summary Table

| EU AI Act requirement | Article | Risk tier | Current status | Gap severity |
|---|---|---|---|---|
| User transparency / AI disclosure | 50 | Limited+ | Absent | CRITICAL |
| Automatic logging / audit trail | 12 | High | Performance metrics only | HIGH |
| Risk management system | 9 | High | Absent | HIGH |
| Data governance (LoRA training) | 10 | High | Absent | HIGH |
| Human oversight mechanisms | 14 | High | Absent | HIGH |
| Technical documentation (Annex IV) | 11 | High | Engineering docs only | HIGH |
| Transparency to deployers | 13 | High | Engineering docs only | MEDIUM |
| Cybersecurity / authentication | 15 | High | Not built in | MEDIUM |
| GPAI obligations (merge/distribute) | 53–55 | GPAI | Operator guidance missing | MEDIUM |
| Operational robustness | 15 | High | Strong (circuit breaker, FTP) | LOW |

---

## 5. What Juno Does Well

Several existing features align with regulatory intent:

- **Circuit breaker and fault-tolerant pipeline** (`CircuitBreaker`, `FaultTolerantPipeline`) support the reliability and human oversight goals of Article 15 and 14.
- **JFR instrumentation** with five custom event types is a strong observability foundation that can be extended into compliance logging without architectural change.
- **Session ID** (`x_juno_session_id`) and completion ID (`chatcmpl-*`) in the API layer are building blocks for an audit trail.
- **Open-source Apache 2.0 licence** aligns with the Act's encouragement of open-source approaches and the reduced obligations that apply to open-source infrastructure tools (Article 25(2)).
- **LoRA adapter isolation** (adapters applied read-only, base GGUF never modified) reduces the blast radius of fine-tuning and supports the principle of testable, auditable model variants.
- **Structured OpenAPI spec** (`juno-api.yaml`) provides a machine-readable API contract that facilitates compliance documentation.

---

## 6. Prioritised Remediation Recommendations

Listed in order of regulatory urgency for a Juno operator targeting EU deployment.

**1. Implement Article 50 AI disclosure (immediate, low effort)**
Add a response field and configurable startup banner. For non-streaming responses, include `"x_juno_system_disclosure": "This response was generated by an AI system."` in `OaiChatCompletionResponse`. For streaming, emit a metadata SSE event before the first token. Configurable off for API-to-API use without end users.

**2. Add compliance audit logging (short-term, medium effort)**
Introduce a structured audit log (distinct from JFR) in `OpenAiChatHandler` capturing: request ID, session ID, model ID, model version/SHA, input token count, output token count, finish reason, sampling parameters, timestamp, and client identifier. Write to a configurable sink (file, stdout JSON). This satisfies Article 12 and provides operators with post-market monitoring data.

**3. Add authentication and TLS to InferenceApiServer (short-term, medium effort)**
Add bearer token / API key authentication middleware to Javalin. Add TLS configuration via JVM keystore. These are table-stakes security requirements for any production AI deployment, and Article 15 compliance for high-risk contexts.

**4. Add operator documentation for high-risk contexts (medium-term, low effort)**
Produce a compliance guide covering: how to populate Annex IV technical documentation for a Juno deployment, limitations of served models, how to configure for high-risk use cases, and a warning that LoRA merge outputs may trigger GPAI obligations.

**5. Add data governance hooks to LoRA training pipeline (medium-term, medium effort)**
Before ingesting training data into `LoraTrainableHandler`, log: corpus hash, token count, source label (operator-supplied), and timestamp to a training provenance record. Generate a provenance artifact alongside each `.lora` checkpoint. Emit a warning in `LoraMergeMain` when producing merged GGUFs.

**6. Design human oversight hooks (longer-term, high effort)**
Expose a review-queue mode in `RequestScheduler` for operators who need to intercept completions before delivery. Add a confidence/uncertainty signal hook (even a simple output-length-relative-to-max-tokens heuristic) to flag potentially truncated or degenerate outputs. This addresses Article 14 for high-risk deployers.

---

## 7. Conclusion

Juno, as an open-source LLM inference framework, does not itself constitute an AI system or GPAI model under the EU AI Act. The regulatory obligations fall on the entity that operates Juno in production to serve end users or downstream systems.

The framework is technically sophisticated and operationally well-designed. Its fault tolerance, observability, and structured API make it a credible foundation for compliant deployments. However, the codebase currently provides no compliance-oriented features: no AI disclosure, no audit logging, no authentication, no human oversight hooks, and no data governance for the LoRA training pipeline.

An operator deploying Juno in a limited-risk context (general chat assistant) faces one critical gap: Article 50 AI disclosure, which is trivially fixable. An operator deploying Juno in a high-risk context (employment screening, credit, healthcare triage, etc.) would face the full Chapter III obligation set and would need to build substantial compliance infrastructure on top of the current framework.

The most impactful near-term investment for the Juno project is: Article 50 disclosure in the API response, structured audit logging, and API authentication — three changes that collectively address the most urgent regulatory exposure across all deployment contexts.
