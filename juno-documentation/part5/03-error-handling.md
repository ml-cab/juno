(ch-5-3)=
# 5.3. Error Handling

Error responses from the OpenAI-compatible API follow the OpenAI error envelope:

```json
{"error": {"message": "...", "type": "...", "code": "..."}}
```

| HTTP | `code` | Cause |
|------|--------|-------|
| 400 | `invalid_request` | Missing or empty messages, `n` greater than 1, or an invalid body |
| 503 | `service_unavailable` | No model loaded or model not ready |
| 429 | `rate_limit_exceeded` | Scheduler queue full; `Retry-After` header is set |
| 500 | `internal_error` | Unexpected inference error |

## See also

- [Chapter 5.1 -- Juno Native API](#ch-5-1)
- [Chapter 5.2 -- OpenAI-Compatible API](#ch-5-2)

---

[<- 5.2 OpenAI-Compatible API](#ch-5-2) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [5.4 OpenAPI Spec ->](#ch-5-4)