(ch-18)=
# 18. Commercial Services

Juno is free and open-source software under Apache 2.0 (see [Chapter 17](#ch-17)). No
commercial license is required to use, deploy, or build on Juno, including in production SaaS
products. The project is sustained by the paid services below, all of which are entirely
optional. None of them restrict community access to the source code or create a separate
"enterprise edition."

## Support contracts

Paid support contracts are available for teams that need guaranteed response times, access to
maintainer engineering time, and private issue triage.

| Tier | Response SLA | Included | Price guide |
|------|--------------|----------|-------------|
| Community | Best effort | GitHub issues and Discord; no SLA | Free |
| Standard | 2 business days | Private issue tracker; email support; patch backports | Contact for pricing |
| Priority | 4 business hours | Dedicated Slack channel; architecture reviews; escalation | Contact for pricing |

Support contracts cover the Juno engine only. They do not include support for third-party model
weights, GPU drivers, or operator infrastructure.

To inquire: [dev@ml.cab](mailto:dev@ml.cab) — subject line "Support Contract Inquiry."

## Integration and consulting

The maintainers offer time-bounded consulting engagements covering:

- Production deployment architecture: on-prem cluster setup, AWS/cloud integration, GPU
  provisioning (see [Chapter 7](#ch-07)).
- JVM integration: embedding `JunoPlayer` or `LocalChat` in your application stack (see
  [Chapter 6](#ch-06)).
- LoRA fine-tuning pipeline design, dataset preparation guidance, and adapter evaluation
  methodology (see [Chapter 8](#ch-08) and [Chapter 9](#ch-09)).
- EU AI Act compliance gap assessment and remediation planning for Juno deployments (see
  [Chapter 19](#ch-19)).
- Performance tuning for specific hardware configurations and model families (see
  [Chapter 13](#ch-13)).

Engagements are scoped, time-boxed, and priced per engagement. All resulting code contributed
back to the project is released under Apache 2.0.

To inquire: [dev@ml.cab](mailto:dev@ml.cab) — subject line "Integration Engagement Inquiry."

## What commercial services do not include

- A separate or proprietary version of the Juno engine.
- Additional license rights beyond Apache 2.0 — Apache 2.0 already permits all commercial use
  without a separate license.
- Endorsement of, or liability for, the operator's production system or compliance posture.
- Support for third-party model weights, GGUF providers, or Hugging Face artifacts.

## Trademark use by commercial partners

Use of the "Juno" project name in marketing materials requires adherence to the trademark
policy in [Chapter 17](#ch-17). Using "Powered by Juno" or "Built on Juno" is permitted without
prior approval when the usage is accurate and does not imply official endorsement beyond that
statement.

*Contact: [dev@ml.cab](mailto:dev@ml.cab)*

---

[← Chapter 17: Legal Reference](#ch-17) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [Chapter 19: EU AI Act Compliance Analysis →](#ch-19)
