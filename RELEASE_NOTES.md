# Juno 0.1.1 Release Notes

**Java Unified Neural Orchestration.** Distributed LLM inference and fine-tuning in pure Java.

License: [Apache 2.0](LICENSE)

Full documentation: **[ml.cab/juno-documentation](https://ml.cab/juno-documentation)**

---

## Requirements

| Component | Version |
|---|---|
| JDK | 25+ |
| Maven (build from source) | 3.9+ |
| NVIDIA GPU (optional) | CUDA 12.x + driver |
| AMD GPU (optional) | ROCm 6+ + driver |

CPU-only inference requires no GPU stack. The `./juno` launcher enforces JDK 25 at startup.

Full requirements: [1.1 Requirements](https://ml.cab/juno-documentation/requirements/).

---

## What is in this release

| Area | Documentation |
|---|---|
| Distributed inference, GPU acceleration, architecture | [Part 2. Architecture](https://ml.cab/juno-documentation/overview/) |
| LoRA fine-tuning, GPU training, DoRA, GGUF merge | [Part 4. LoRA Fine-Tuning](https://ml.cab/juno-documentation/concepts/) |
| OpenAI-compatible and Juno-native REST API | [Part 5. REST API](https://ml.cab/juno-documentation/juno-native-api/) |
| JFR events, health dashboard, performance matrix | [Part 7. Observability and Performance](https://ml.cab/juno-documentation/jfr-and-metrics/) |
| Supported models and quantizations | [1.4 Supported Models](https://ml.cab/juno-documentation/supported-models/) |

---

## Quick start

```bash
mvn clean package -DskipTests

# Download a GGUF, then:
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# With OpenAI-compatible API:
./juno local --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --api-port 8080
```

See [1.2 Quickstart: Local](https://ml.cab/juno-documentation/quickstart-local/)
and [Part 3. CLI Reference](https://ml.cab/juno-documentation/commands/).

---

## Known limitations (0.1.0)

- **Text only**: image or multimodal message content is not supported.
- **OpenAI `n > 1`**: rejected with HTTP 400; only single completions.
- **Partial OpenAI compatibility**: `stop`, `presence_penalty`, `logit_bias`, `user`, `seed` are
  ignored for client compatibility.
- **No built-in auth or TLS** on the REST server; configure at the reverse proxy or network layer
  for production.
- **LoRA merge / redistribution** may trigger model-license obligations; see
  [9.3 LoRA and Merge Licensing](https://ml.cab/juno-documentation/lora-and-merge-licensing/).
- **EU AI Act**: compliance-oriented features (AI disclosure, audit logging, auth) are not yet
  built in; see [9.7 EU AI Act Compliance](https://ml.cab/juno-documentation/eu-ai-act-compliance/).

---

## Documentation

Full reference: **[ml.cab/juno-documentation](https://ml.cab/juno-documentation)**

| Topic | Page |
|---|---|
| Architecture | [Part 2. Architecture](https://ml.cab/juno-documentation/overview/) |
| CLI reference | [Part 3. CLI Reference](https://ml.cab/juno-documentation/commands/) |
| LoRA fine-tuning | [Part 4. LoRA Fine-Tuning](https://ml.cab/juno-documentation/concepts/) |
| REST API | [Part 5. REST API](https://ml.cab/juno-documentation/juno-native-api/) |
| Deployment | [Part 6. Deployment](https://ml.cab/juno-documentation/on-prem-cluster/) |
| Observability and performance | [Part 7. Observability and Performance](https://ml.cab/juno-documentation/jfr-and-metrics/) |
| Legal and compliance | [Part 9. Legal and Compliance](https://ml.cab/juno-documentation/license-and-patents/) |
| Contributing | [10.1 Contributing](https://ml.cab/juno-documentation/contributing/) |
| Security | [10.3 Security Policy](https://ml.cab/juno-documentation/security-policy/) |
| OpenAPI spec | [5.4 OpenAPI Spec](https://ml.cab/juno-documentation/openapi-spec/) |
| Changelog | [11.2 Changelog](https://ml.cab/juno-documentation/changelog/) |

---

## Upgrade / migration

This is the first public release. No prior version migration path.

Artifacts publish under `cab.ml` at version `0.1.0` on Maven Central. Import the BOM:

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>cab.ml</groupId>
      <artifactId>juno-bom</artifactId>
      <version>0.1.0</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>
```