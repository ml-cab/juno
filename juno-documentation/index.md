# Juno Documentation

*Distributed LLM inference and fine-tuning, written entirely in Java.*

---

**How to read this book**

This is the `docs/` folder of the [Juno](https://github.com/ml-cab/juno) project, structured
as a single, cross-linked reference organized around what a reader is trying to do: get
started, understand a concept, complete a task, or look something up.

11 parts, 54 chapters:

- **Part 1 -- Getting Started.** What Juno requires, two ways to get a first model running, and the full model support matrix.
- **Part 2 -- Architecture.** The distributed architecture end to end: how a request moves through the coordinator and nodes, how a GGUF file picks its handler, the GPU backends, the reasoning behind the major engineering choices, and how the modules fit together.
- **Part 3 -- CLI Reference.** Every command, every flag, and every running mode (local, cluster, LoRA, merge, test), plus how to read verbose diagnostic output.
- **Part 4 -- LoRA Fine-Tuning.** The math behind LoRA, which architectures support it, the full training and inference workflow, merging a trained adapter into a standalone model, the programmatic Java API, and the pitfalls worth knowing before you start.
- **Part 5 -- REST API.** The OpenAI-compatible REST surface, the Juno-native REST surface, the error envelope, and the OpenAPI specification.
- **Part 6 -- Deployment.** Running a cluster on your own hardware, on AWS, and on Windows.
- **Part 7 -- Observability and Performance.** JFR-based observability, the performance-testing methodology, and how to read the published benchmark report.
- **Part 8 -- Testing.** The build, unit, integration, and GPU test suites.
- **Part 9 -- Legal and Compliance.** Licensing and patents, third-party model weights, LoRA/merge licensing, the CLA, trademark policy, export control, and the EU AI Act compliance summary.
- **Part 10 -- Community and Project.** How to contribute, how the project is governed, the security policy, funding, and commercial services.
- **Part 11 -- Releases.** The curated release notes and the detailed engineering changelog.

Where a diagram helps more than a paragraph, it is drawn with Mermaid so it renders natively
in any viewer that supports it, rather than as fixed-width ASCII art.

---

## Table of Contents

**Part 1. Getting Started**

- **1.1.** [Requirements](part1/01-requirements.md)
- **1.2.** [Quickstart: Local Player](part1/02-quickstart-local.md)
- **1.3.** [Quickstart: JVM Embedding](part1/03-quickstart-jvm-embedding.md)
- **1.4.** [Supported Models](part1/04-supported-models.md)

**Part 2. Architecture**

- **2.1.** [Overview](part2/01-overview.md)
- **2.2.** [Distributed Inference](part2/02-distributed-inference.md)
- **2.3.** [Handler Routing](part2/03-handler-routing.md)
- **2.4.** [GPU Acceleration](part2/04-gpu-acceleration.md)
- **2.5.** [Key Design Decisions](part2/05-key-design-decisions.md)
- **2.6.** [Module Map](part2/06-module-map.md)

**Part 3. CLI Reference**

- **3.1.** [Commands](part3/01-commands.md)
- **3.2.** [Flags](part3/02-flags.md)
- **3.3.** [Local Mode](part3/03-local-mode.md)
- **3.4.** [Cluster Mode](part3/04-cluster-mode.md)
- **3.5.** [LoRA Mode](part3/05-lora-mode.md)
- **3.6.** [Merge Mode](part3/06-merge-mode.md)
- **3.7.** [Test Mode](part3/07-test-mode.md)
- **3.8.** [Diagnostics and Tracing](part3/08-diagnostics-and-tracing.md)

**Part 4. LoRA Fine-Tuning**

- **4.1.** [Concepts](part4/01-concepts.md)
- **4.2.** [Architecture Support](part4/02-architecture-support.md)
- **4.3.** [Training Guide](part4/03-training-guide.md)
- **4.4.** [Inference with a Trained Adapter](part4/04-inference-with-adapter.md)
- **4.5.** [Merging Adapters](part4/05-merging-adapters.md)
- **4.6.** [Programmatic API](part4/06-programmatic-api.md)
- **4.7.** [Common Pitfalls](part4/07-common-pitfalls.md)
- **4.8.** [Testing Checklist](part4/08-testing-checklist.md)

**Part 5. REST API**

- **5.1.** [Juno Native API](part5/01-juno-native-api.md)
- **5.2.** [OpenAI-Compatible API](part5/02-openai-compatible-api.md)
- **5.3.** [Error Handling](part5/03-error-handling.md)
- **5.4.** [OpenAPI Spec](part5/04-openapi-spec.md)

**Part 6. Deployment**

- **6.1.** [On-Prem Cluster](part6/01-on-prem-cluster.md)
- **6.2.** [AWS Deployment](part6/02-aws-deployment.md)
- **6.3.** [Windows Notes](part6/03-windows.md)

**Part 7. Observability and Performance**

- **7.1.** [JFR and Metrics](part7/01-jfr-and-metrics.md)
- **7.2.** [Performance Methodology](part7/02-performance-methodology.md)
- **7.3.** [Performance Report](part7/03-performance-report.md)

**Part 8. Testing**

- **8.1.** [Build and Test](part8/01-build-and-test.md)
- **8.2.** [GPU Tests](part8/02-gpu-tests.md)

**Part 9. Legal and Compliance**

- **9.1.** [License and Patents](part9/01-license-and-patents.md)
- **9.2.** [Third-Party Model Weights](part9/02-third-party-models.md)
- **9.3.** [LoRA and Merge Licensing](part9/03-lora-and-merge-licensing.md)
- **9.4.** [Contributor License Agreement](part9/04-contributor-license-agreement.md)
- **9.5.** [Trademark Policy](part9/05-trademark.md)
- **9.6.** [Export Control](part9/06-export-control.md)
- **9.7.** [EU AI Act Compliance](part9/07-eu-ai-act-compliance.md)
- **9.8.** [Disclaimer of Warranties](part9/08-disclaimer.md)

**Part 10. Community and Project**

- **10.1.** [Contributing](part10/01-contributing.md)
- **10.2.** [Governance](part10/02-governance.md)
- **10.3.** [Security Policy](part10/03-security-policy.md)
- **10.4.** [Funding](part10/04-funding.md)
- **10.5.** [Commercial Services](part10/05-commercial-services.md)
- **10.6.** [Contributors](part10/06-contributors.md)

**Part 11. Releases**

- **11.1.** [Release Notes](part11/01-release-notes.md)
- **11.2.** [Changelog](part11/02-changelog.md)

**Back matter**

- [References](references.md) -- which original Juno `docs/` file each chapter is built from