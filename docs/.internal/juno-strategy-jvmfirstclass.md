# Juno: Strategic Assessment & Roadmap to Best-in-Class JVM-Native Deployment

**Prepared for:** Juno maintainers
**Scope:** Overall project assessment, evaluation against "best model support / best throughput /
biggest ecosystem," and a focused, sequenced plan to win the JVM-native deployment niche first.
**Status:** Draft for review — not yet actioned.

---

## 1. Executive summary

Juno's most interesting bet is architectural, not featural: it runs LLM inference and LoRA
fine-tuning entirely inside the JVM, with no Python subprocess, no GIL, and GPU access via Panama
FFI rather than JNI bindings. That bet is well-executed — the LoRA implementation in particular
has real test discipline (adjointness tests, finite-difference gradient checks, zero-adapter
parity tests), and the JFR-native observability story is a genuine, underused differentiator.

Measured against the broader inference-engine market ("best model support, best throughput,
biggest ecosystem"), Juno is not competitive today, and closing that gap against vLLM/SGLang's
funded teams and llama.cpp's large contributor base is a multi-year undertaking for a two-person
maintainer group.

Measured against a narrower goal — **the obvious choice for JVM-native LLM deployment** — Juno is
much closer to winning outright, because the incumbents have largely conceded this ground by being
Python- or C++-first. This document lays out both assessments and recommends the narrower goal as
the near-term strategy, with the broader roadmap sequenced to follow once that niche is won.

---

## 2. Overall assessment

### 2.1 What's genuinely good

- **Pure-JVM architecture.** GGUF parsing, quantized matmul, and KV caching all run inside the
  JVM. GPU access goes through `java.lang.foreign` (Panama FFI) rather than JavaCPP/bytedeco —
  resolved once at class-init, thread-safe `MethodHandle`s, no per-call boxing overhead. This is a
  legitimately modern approach and depends on JDK 25+, so the project is also betting on recent
  Java rather than targeting the lowest common denominator.
- **LoRA implementation quality.** Real numerical rigor: adjointness tests
  (`dot(A·x, v) == dot(Aᵀ·v, x)`), finite-difference gradient checks, zero-adapter parity tests
  against the non-LoRA handler. An explicit architecture allowlist means an unsupported model
  fails loudly at load time instead of silently training against the wrong tensor layout — a
  detail many ML tooling projects get wrong.
- **JFR-native observability.** Six custom event types cover every hot path — matmul, forward
  pass, token emission, tokenization, template formatting, LoRA training steps — making the whole
  stack visible in JDK Mission Control with zero agents. This is a real, currently under-marketed
  asset (see §4.3).
- **Design discipline elsewhere.** Explicit GPU memory lifecycle
  (`releaseGpuResources()` rather than relying on GC), automatic CPU/GPU/vendor fallback ladders,
  a stub-mode test harness that needs no model file or GPU to run integration tests.

### 2.2 Where it falls short of the broader market

- **Model coverage is thin.** Solid support is really just LLaMA-family, Mistral, TinyLlama, and
  Phi-3. Gemma, Qwen2, Qwen3, Qwen3-MoE, and Qwen3.5 are all "under development." There's no
  DeepSeek MLA attention, no Mixtral-style MoE beyond what Qwen3-MoE work will eventually
  generalize, and nothing multimodal. New architectures land in llama.cpp within days of release;
  Juno is structurally slower to catch up given one bespoke handler class per architecture family.
- **Distributed inference is simpler than modern production serving stacks.** Tensor-parallel
  AllReduce funnels through the coordinator rather than using proper collective communication.
  There's no continuous batching, no paged KV attention, no speculative decoding — the techniques
  that account for most of the throughput gains the field has made over the last two years.
- **Maturity gaps.** Two maintainers, v0.1.x. The project's own EU AI Act compliance chapter is
  candid that the REST API ships with no authentication, no rate limiting, and no TLS by default —
  a real gap for anything beyond local experimentation.

### 2.3 Bottom line

Juno is not yet "best in the world" and shouldn't try to be that in the next 12 months. It is,
however, very plausibly **best-in-class for JVM-native deployment** within reach of a two-person
team in that same timeframe, because that specific niche is currently undefended. §4 below is the
plan for winning it.

---

## 3. Path to "best model support, best throughput, biggest ecosystem"

This is the longer-term, broader campaign. It's included here for completeness and because Phase 3
of the JVM-native plan (§4) explicitly funds and justifies moving into this territory afterward —
but it is **not** the recommended near-term focus.

### 3.1 Model support: change *how* architectures get added, not just add more

- Finish what's already "under development": Gemma, Qwen2, Qwen3, Qwen3-MoE. This is the fastest
  credibility win available — the support matrix currently undersells engineering already done.
- Replace the one-bespoke-handler-per-architecture pattern with a **declarative architecture
  spec** — most differences between LLaMA/Qwen2/Qwen3/Gemma reduce to a small set of knobs (QKV
  bias presence, Q/K norm, RoPE variant, MoE routing). A metadata-driven handler would cut
  time-to-support for the *next* architecture from weeks to days.
- Make **MoE a first-class primitive**, generalized from the Qwen3-MoE work, rather than a one-off
  — Mixtral, DeepSeek-MoE, and most frontier open releases now ship MoE variants.
- Make a deliberate, explicit call on multimodal and SSM/hybrid architectures (Qwen3.5's
  DeltaNet). This changes the forward-pass abstraction significantly and shouldn't be an implicit
  default.
- Publish an **architecture-support SLA** ("new open-weight releases from major labs supported
  within N weeks") — turns coverage from a silent weakness into a visible, trackable commitment
  that itself attracts contributions.

### 3.2 Throughput: adopt the serving-loop techniques the field has already validated

- **Continuous batching** — probably the single highest-leverage change. Current throughput under
  concurrent load depends on however `RequestScheduler` interleaves requests today, rather than
  dynamically batching decode steps across in-flight sequences.
- **Paged KV cache** — a PagedAttention-style block allocator over the existing GPU/CPU
  `KVCacheManager` tiers would stop VRAM fragmentation the way it did for vLLM.
- **Speculative decoding** — a draft-model-plus-verify path; now table stakes for latency-sensitive
  serving, currently absent.
- **Better collective communication for tensor parallel** — the current star AllReduce through the
  coordinator works but won't scale past a handful of nodes the way ring-AllReduce would.
- **Broader quantization support** — beyond the existing GGUF K-quant coverage, consider AWQ/GPTQ
  import or newer low-bit formats gaining traction.
- **Use the existing JFR/performance-matrix discipline as the feedback loop** for all of the above,
  and consider a live public leaderboard against llama.cpp/vLLM on identical hardware as a
  marketing artifact, not just an internal doc.

### 3.3 Ecosystem: highest leverage for a small maintainer team

- **Production-readiness of the REST API is a prerequisite here too** — auth, rate limiting, TLS.
  Without this, ecosystem growth is capped regardless of engine quality.
- **First-class framework integrations**, not just OpenAI-wire compatibility — a maintained SDK
  page, HuggingFace-adjacent model registry integration, and framework-specific quickstarts.
- **Lower the barrier to contributing a new architecture handler** — pair the declarative spec
  (§3.1) with a contributor guide specifically for "add architecture X," a well-scoped task that
  draws first-time contributors.
- **A verified-compatibility model zoo**, maintained the way llama.cpp's README is — today a
  prospective user has to infer support from the architecture matrix rather than seeing a curated
  list of verified checkpoints.
- **Distribution beyond Maven Central** — Docker images, Homebrew formula, prebuilt binaries —
  reducing the "clone and `mvn package`" friction that currently gates even trying it.
- **Governance that scales.** Two maintainers with unanimous-consent decisions works today but
  won't scale to a larger contributor base — worth pre-empting with lightweight approval tiering.

### 3.4 Suggested sequencing (broader campaign)

| Phase | Focus | Why this order |
|---|---|---|
| 1 | Auth/TLS on the API; finish Gemma/Qwen2/Qwen3 | Unblocks serious adoption; ships work already in progress |
| 2 | Continuous batching + paged KV cache | Biggest throughput lever; prerequisite for competitive published benchmarks |
| 3 | Declarative architecture spec + MoE generalization | Turns model support into near-constant effort per new model |
| 4 | Speculative decoding, quantization breadth | Diminishing-but-real gains once the big levers are pulled |
| 5 | SDKs, model zoo, packaging, governance scaling | Compounds everything above into actual adoption |

**Honest caveat:** competing on all three axes simultaneously against funded, large-team
competitors is a multi-year bet. The more defensible near-term goal is §4.

---

## 4. Recommended near-term strategy: win JVM-native deployment outright

### 4.1 The thesis

A JVM shop's actual pain today isn't "which inference engine is fastest" — it's "how do I get an
LLM into my existing Spring/Micronaut/Quarkus stack without also importing a Python runtime, a
FastAPI sidecar, and a second deployment pipeline." Juno already has the right foundations for this
(`JunoPlayer`, the Maven BOM, Panama FFI, JFR). The plan below removes every remaining friction
point in that story and makes the JVM-native properties **legible** — visible and provable, not
just true in the source code.

This is winnable because vLLM, llama.cpp, and TGI have essentially conceded this ground by being
Python- or C++-first; a JVM-native pitch has effectively no direct competitor today.

### 4.2 Phase 1 — Remove the reasons a JVM team would say no (0–2 months)

These are adoption blockers, not features. No further evaluation happens if these are missing.

| Item | Why it's a blocker |
|---|---|
| Auth, TLS, rate limiting on `InferenceApiServer` | Already flagged in the project's own EU AI Act gap analysis; no enterprise JVM team deploys an unauthenticated HTTP server |
| Structured audit logging, separate from JFR's performance events | Enterprise JVM shops require audit trails as a matter of course |
| JPMS module descriptors (`module-info.java`) across the module tree | Juno is already modern-Java-idiomatic (virtual threads, Panama FFI) but doesn't yet claim full modularity explicitly |
| A GraalVM native-image build target | **Highest-leverage single item in this phase** — fast cold start and low memory in containers/serverless is a live JVM-community priority; Panama FFI (no JNI reflection) is well-suited to native-image analysis, unlike JNI-based alternatives |

### 4.3 Phase 2 — Become a first-class citizen of the frameworks JVM teams already use (2–5 months)

This is where "JVM-native" stops being a claim about internals and becomes something a developer
directly experiences.

| Integration | Rationale |
|---|---|
| **Spring Boot starter** (`juno-spring-boot-starter`) | Autoconfiguration for `JunoPlayer` as a bean, property-based config (`juno.model-path`, `juno.gpu.backend`, ...), an `/actuator`-style health indicator wired to the existing `HealthReporter`. Likely the single highest-ROI integration given Spring's market share in the shops Juno should target |
| **Micronaut and Quarkus modules** | Both center GraalVM native-image as a core value prop, pairing directly with Phase 1. Quarkus's LangChain4j ecosystem is a natural plug-in point |
| **Micrometer integration** | Translate existing JFR events (`juno.TokenProduced`, `juno.ForwardPass`, etc.) into Micrometer meters. JVM shops standardize observability around Micrometer → Prometheus/Datadog; this data is currently locked in JFR/Mission Control, a smaller specialist audience |
| **Testcontainers module** (`JunoContainer`) | JVM teams test this way by default — not having it makes Juno feel like an outsider to the standard workflow |
| **LangChain4j integration** | Distinct from generic OpenAI-compatibility — LangChain4j is the JVM ecosystem's answer to LangChain. A native `ChatLanguageModel` backed by `JunoPlayer` puts Juno directly in front of Java developers already building LLM apps |

### 4.4 Phase 3 — Make the JVM-native properties provable (3–6 months, overlapping Phase 2)

Claims are cheap for a JVM-specific audience that will want receipts — and this is where the JFR
investment already made starts paying off as a differentiator rather than a curiosity.

- **Cold-start and memory-footprint benchmark**, framed specifically against a Python sidecar
  deployment (e.g. FastAPI + llama-cpp-python) on identical hardware — container startup time, RSS
  at idle, RSS under load. This is the comparison a JVM platform team actually cares about, and
  very likely a genuine, easy win given no Python interpreter and, with native-image, no JIT
  warmup either.
- **GC-pause / p99 latency study**, showing virtual threads plus the existing careful GPU-memory
  lifecycle design keep tail latency predictable under concurrent load — exactly the property JVM
  teams scrutinize.
- **JDK Mission Control as the headline pitch, not an afterthought.** Most inference engines
  require a bespoke profiling/observability stack; Juno gets JFR for free. Frame this explicitly:
  "debug your LLM serving stack with the same tools you already use to debug the rest of your JVM
  services" — a real, differentiated claim today that isn't currently marketed as one.
- **A reference architecture document**: "Juno inside a Kubernetes-deployed Spring Boot mesh" —
  sidecar-free, one JVM process, one container, one set of health/metrics endpoints. JVM shops are
  often actively migrating away from polyglot sidecar sprawl; give them the diagram that shows Juno
  collapsing it.

### 4.5 Phase 4 — Own the distribution channels JVM developers actually use (4–7 months)

- **Maven Central is already covered** — preserve this. Add a **Gradle plugin** wrapper, since a
  meaningful share of the JVM ecosystem (especially Kotlin/Android-adjacent shops) is Gradle-first
  and currently gets no special treatment.
- **JBang example / one-liner trial** — lets someone try Juno with zero project setup
  (`jbang juno@ml-cab`-style), removing the "clone and Maven-build the whole repo" friction that
  currently gates even a first look.
- **Docker images built from the Phase 1 native-image target** — distroless, minimal, fast-starting
  — showcasing the cold-start advantage directly rather than shipping "a JRE with jars in it."

### 4.6 Sequencing rationale

| Phase | Goal | Why this order |
|---|---|---|
| 1 | Remove blockers (auth, audit log, JPMS, native-image) | Nothing else matters if a security review kills the evaluation on day one |
| 2 | Framework-native integrations (Spring/Micronaut/Quarkus/LangChain4j) | This *is* the niche — meet developers exactly where they already work |
| 3 | Prove the JVM-native advantages with benchmarks and docs | Converts "trust us" into "here's the number," using JFR assets already built |
| 4 | Distribution polish (Gradle, JBang, Docker) | Compounds everything above into actual trial-to-adoption conversion |

### 4.7 Why this is the right first move

"The fastest way to run an LLM inside a JVM stack, full stop" is a claim two maintainers can
credibly defend within about a year. "Fastest LLM engine, period" is not — not against vLLM/SGLang's
dedicated funded teams or llama.cpp's contributor base. Winning the narrower claim first:

- Requires no breakthrough research, only integration and packaging work already well within reach.
- Builds directly on strengths the project already has (Panama FFI, JFR, the BOM/`JunoPlayer`
  facade) rather than starting new capability from zero.
- Produces a credible, defensible position ("the JVM-native inference engine") that can fund and
  justify the broader §3 campaign afterward, once there's a proven niche and a stronger
  contributor base to draw on.

---

## 5. Open questions for maintainer discussion

- Does the team want to commit to GraalVM native-image as a first-class target? It's the highest-
  leverage item in Phase 1 but also a real engineering investment (native-image compatibility
  across all Panama FFI call sites needs verification).
- What's the appetite for a public benchmark campaign (§4.4) before the underlying auth/TLS gaps
  (§4.2) are closed? Recommendation: close the blockers first — publishing performance claims for
  a server nobody can safely deploy would undercut the credibility this plan is trying to build.
- Should Phase 2's framework integrations be maintained in the main `ml-cab/juno` repo or as
  separate community-facing repos (`juno-spring-boot-starter`, etc.)? Separate repos lower the
  contribution barrier for framework specialists who may not want to touch the core engine.
- Is multimodal/SSM support (§3.1) explicitly out of scope for the JVM-native phase, or does it
  need to be addressed earlier if a target JVM audience specifically needs it?
