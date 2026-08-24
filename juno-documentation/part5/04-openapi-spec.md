(ch-5-4)=
# 5.4. OpenAPI Spec

The full OpenAPI 3.0 specification for the OpenAI-compatible surface is checked into the repo at
[`api/src/main/resources/juno-api.yaml`](https://github.com/ml-cab/juno/blob/main/api/src/main/resources/juno-api.yaml).

Use it to:

- Generate typed clients in languages other than the ones covered by the OpenAI SDKs.
- Validate request and response bodies in CI.
- Drive contract tests against a running coordinator.

The Juno-native endpoints (`/v1/inference`, `/v1/inference/stream`, `/v1/cluster/health`, and the
health/console routes) are not part of this OpenAPI document. See
[Juno native API](#ch-5-1) for those.

## See also

- [Chapter 5.1 -- Juno Native API](#ch-5-1)
- [Chapter 5.2 -- OpenAI-Compatible API](#ch-5-2)

---

[<- 5.3 Error Handling](#ch-5-3) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [6.1 On-Prem Cluster ->](#ch-6-1)