# References

Each chapter in this book is built from one file in the Juno project's `docs/` folder
(restructured from the project's original flat `docs/` layout). This table maps every
chapter back to its source.

| Chapter | Source file in `docs/` |
|---|---|
| 1.1 Requirements | `docs/01-getting-started/requirements.md` |
| 1.2 Quickstart: Local Player | `docs/01-getting-started/quickstart-local.md` |
| 1.3 Quickstart: JVM Embedding | `docs/01-getting-started/quickstart-jvm-embedding.md` |
| 1.4 Supported Models | `docs/01-getting-started/supported-models.md` |
| 2.1 Overview | `docs/02-architecture/overview.md` |
| 2.2 Distributed Inference | `docs/02-architecture/distributed-inference.md` |
| 2.3 Handler Routing | `docs/02-architecture/handler-routing.md` |
| 2.4 GPU Acceleration | `docs/02-architecture/gpu-acceleration.md` |
| 2.5 Key Design Decisions | `docs/02-architecture/key-design-decisions.md` |
| 2.6 Module Map | `docs/02-architecture/module-map.md` |
| 3.1 Commands | `docs/03-cli-reference/commands.md` |
| 3.2 Flags | `docs/03-cli-reference/flags.md` |
| 3.3 Local Mode | `docs/03-cli-reference/local-mode.md` |
| 3.4 Cluster Mode | `docs/03-cli-reference/cluster-mode.md` |
| 3.5 LoRA Mode | `docs/03-cli-reference/lora-mode.md` |
| 3.6 Merge Mode | `docs/03-cli-reference/merge-mode.md` |
| 3.7 Test Mode | `docs/03-cli-reference/test-mode.md` |
| 3.8 Diagnostics and Tracing | `docs/03-cli-reference/diagnostics-and-tracing.md` |
| 4.1 Concepts | `docs/04-lora-fine-tuning/concepts.md` |
| 4.2 Architecture Support | `docs/04-lora-fine-tuning/architecture-support.md` |
| 4.3 Training Guide | `docs/04-lora-fine-tuning/training-guide.md` |
| 4.4 Inference with a Trained Adapter | `docs/04-lora-fine-tuning/inference-with-adapter.md` |
| 4.5 Merging Adapters | `docs/04-lora-fine-tuning/merging-adapters.md` |
| 4.6 Programmatic API | `docs/04-lora-fine-tuning/programmatic-api.md` |
| 4.7 Common Pitfalls | `docs/04-lora-fine-tuning/common-pitfalls.md` |
| 4.8 Testing Checklist | `docs/04-lora-fine-tuning/testing-checklist.md` |
| 5.1 OpenAI-Compatible API | `docs/05-rest-api/openai-compatible-api.md` |
| 5.2 Juno Native API | `docs/05-rest-api/juno-native-api.md` |
| 5.3 Error Handling | `docs/05-rest-api/error-handling.md` |
| 5.4 OpenAPI Spec | `docs/05-rest-api/openapi-spec.md` |
| 6.1 On-Prem Cluster | `docs/06-deployment/on-prem-cluster.md` |
| 6.2 AWS Deployment | `docs/06-deployment/aws-deployment.md` |
| 6.3 Windows Notes | `docs/06-deployment/windows.md` |
| 7.1 JFR and Metrics | `docs/07-observability-and-performance/jfr-and-metrics.md` |
| 7.2 Performance Methodology | `docs/07-observability-and-performance/performance-methodology.md` |
| 7.3 Performance Report | `docs/07-observability-and-performance/performance-report.md` |
| 8.1 Build and Test | `docs/08-testing/build-and-test.md` |
| 8.2 GPU Tests | `docs/08-testing/gpu-tests.md` |
| 9.1 License and Patents | `docs/09-legal-and-compliance/license-and-patents.md` |
| 9.2 Third-Party Model Weights | `docs/09-legal-and-compliance/third-party-models.md` |
| 9.3 LoRA and Merge Licensing | `docs/09-legal-and-compliance/lora-and-merge-licensing.md` |
| 9.4 Contributor License Agreement | `docs/09-legal-and-compliance/contributor-license-agreement.md` |
| 9.5 Trademark Policy | `docs/09-legal-and-compliance/trademark.md` |
| 9.6 Export Control | `docs/09-legal-and-compliance/export-control.md` |
| 9.7 EU AI Act Compliance | `docs/09-legal-and-compliance/eu-ai-act-compliance.md` |
| 9.8 Disclaimer of Warranties | `docs/09-legal-and-compliance/disclaimer.md` |
| 10.1 Contributing | `docs/10-community-and-project/contributing.md` |
| 10.2 Governance | `docs/10-community-and-project/governance.md` |
| 10.3 Security Policy | `docs/10-community-and-project/security-policy.md` |
| 10.4 Funding | `docs/10-community-and-project/funding.md` |
| 10.5 Commercial Services | `docs/10-community-and-project/commercial-services.md` |
| 10.6 Contributors | `docs/10-community-and-project/contributors.md` |
| 11.1 Release Notes | `docs/11-releases/release-notes.md` |
| 11.2 Changelog | `docs/11-releases/changelog.md` |

The performance report referenced in Chapter 7.3 and the sample training file referenced
in Chapter 4.3 are generated/data assets, not prose documentation; they live at
`docs/assets/juno_test_matrix.html` and `docs/assets/examples/facts-sample.json` in the
source tree and are linked to directly rather than reproduced in this book.
