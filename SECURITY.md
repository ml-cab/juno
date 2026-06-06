# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

If you discover a security issue in Juno, please report it privately. Do not open a public GitHub issue for vulnerabilities.

**Email:** [dev@ml.cab](mailto:dev@ml.cab?subject=Juno%20Security%20Report)

**Discord:** [https://discord.gg/VaXKz9sE](https://discord.gg/VaXKz9sE)

Include:

- A description of the issue and affected components (coordinator REST, gRPC node layer, GPU FFI, LoRA training, etc.)
- Steps to reproduce
- Impact assessment if known
- Your contact details for follow-up

We aim to acknowledge reports within a few business days and will coordinate disclosure once a fix is available.

## Scope notes

Juno ships as an inference engine. Production deployments are expected to add network controls (TLS, authentication, rate limiting) at the perimeter. The default REST server does not include built-in auth or TLS; see [RELEASE_NOTES.md](RELEASE_NOTES.md) known limitations.

Model weights (GGUF files) are third-party artifacts. Juno does not vet model contents; obtain models from trusted sources and comply with their licenses.
