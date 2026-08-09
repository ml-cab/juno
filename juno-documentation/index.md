# Juno Documentation

*Distributed LLM inference and fine-tuning, in pure Java. No Python, no GIL, no Spring.*

---

**How to read this book**

This is the `docs/` folder of the [Juno](https://github.com/ml-cab/juno) project, restructured
as a single, cross-linked reference. It assumes you already know what Juno is for and want the
detail: how the distributed inference engine is put together, how to drive it from the CLI or
from JVM code, how LoRA fine-tuning works end to end, and what governs the project itself.

Four parts, twenty-four chapters:

- **Part I**: Getting started. What Juno is, how it is built internally, and every way to run
  it: CLI flags, local/cluster/lora/merge modes, the OpenAI-compatible REST API, JVM embedding,
  and AWS deployment.
- **Part II**: LoRA fine-tuning. How the adapter math works and which architectures it covers,
  the training and inference REPL, and producing standalone merged models.
- **Part III**: Model support and performance. Which architectures are supported today, a real
  debugging case study for Phi-3 inference, and the methodology behind Juno's performance
  matrix.
- **Part IV**: Governance, legal, and compliance. Project governance, the contribution and
  release workflow, the CLA, the legal reference (licensing, model weights, trademark, export
  control), commercial services, and the EU AI Act compliance gap analysis.

Where a diagram helps more than a paragraph, it is drawn as a Mermaid diagram so it renders
natively in any viewer that supports it.

---

## Table of Contents

**Part I. Getting Started: Running and Integrating Juno**

- **1.** [What Is Juno: Distributed Inference, GPU Acceleration, LoRA, and REST in One Engine](part1/01-what-is-juno.md)
- **2.** [Architecture Reference: Pipeline and Tensor Parallelism, REST Layer, Handler Routing](part1/02-architecture-reference.md)
- **3.** [Commands and Flags: The Complete CLI Reference](part1/03-commands-and-flags.md)
- **4.** [Running Modes: local, cluster, lora, merge, test](part1/04-running-modes.md)
- **5.** [The OpenAI-Compatible REST API](part1/05-openai-compatible-api.md)
- **6.** [JVM Integration: BOM, JunoPlayer, LoraTrainer, and the HTTP Client](part1/06-jvm-integration.md)
- **7.** [AWS Deployment: Cluster Lifecycle and Free-Tier GPU Quotas](part1/07-aws-deployment.md)

**Part II. LoRA Fine-Tuning**

- **8.** [LoRA Fundamentals: The Math, the Architecture Support Matrix](part2/08-lora-fundamentals.md)
- **9.** [Training and Inference Workflows: the REPL, Q&A Facts, Common Pitfalls](part2/09-lora-training-and-inference.md)
- **10.** [Producing Standalone Merged Models with `juno merge`](part2/10-lora-merge.md)

**Part III. Model Support and Performance**

- **11.** [Model Support Matrix: Handlers, Status, and the Qwen/Gemma Roadmap](part3/11-model-support-matrix.md)
- **12.** [Case Study: Debugging Phi-3 Inference End to End](part3/12-phi3-inference-case-study.md)
- **13.** [Performance Methodology: Reproducing and Reading the Test Matrix](part3/13-performance-methodology.md)

**Part IV. Governance, Legal, and Compliance**

- **14.** [Governance: Roles, Decision-Making, Code of Conduct](part4/14-governance.md)
- **15.** [Contributing and the Release Process](part4/15-contributing-and-releases.md)
- **16.** [Contributor License Agreement](part4/16-contributor-license-agreement.md)
- **17.** [Legal Reference: License, Model Weights, Trademark, Export Control](part4/17-legal-reference.md)
- **18.** [Commercial Services](part4/18-commercial-services.md)
- **19.** [EU AI Act Compliance Analysis](part4/19-eu-ai-act-compliance.md)

**Back matter**

- [References](references.md) — the original Juno `docs/` source files each chapter was built from
