# Two sessions with the same opening prompt: correctness check on a real model

**Purpose.** Shows on a real model the defect that `CrossSessionPrefixHitTest` reproduces with a synthetic pipeline, and its
fix. Before the fix, a session whose prompt started with a prompt another session had cached skipped that prefill and decoded
over positions it never wrote. After it, a prefix-cache entry is matched only for the session that cached it. This is a
correctness reading, not a throughput reading.

**Pinned: no** (not a timing). CPU, 12 threads, Xeon E5-1650 v2; TinyLlama 1.1B Q4_K_M; greedy decoding (`--temperature 0`),
12 tokens.

**Builds.**

| Side | Tree | Shaded jar (SHA-256, first 16) |
|---|---|---|
| A, before | `74f75d78`, built from `git archive` in scratch | `96ee5784b914097c` |
| B, after | `74f75d78` plus the uncommitted KV work and the prefix-cache fix | `f0c7eb0186b6fe3a` |

## Command

```bash
two-sessions.sh <jar> static|continuous <out-dir>
```

The script starts a local server (`--local --cpu --api-port 18093 --schedule <schedule>`), then sends three requests in order,
each with the same system prompt ("You are a helpful assistant. Answer in one short sentence.") and user message ("What is the
capital of France?"): one stateless, one with `x_juno_session_id` `user-a`, one with `user-b`. Every reply should be the
same.

## Result

| Build | Schedule | Stateless | Session `user-a` | Session `user-b` |
|---|---|---|---|---|
| A | `static` | The capital of France is Paris. | The capital of France is Paris. | **garbled text** (` Резуpitiful""` then `Baseditorialitter`) |
| A | `continuous` | The capital of France is Paris. | The capital of France is Paris. | **garbled text** (same) |
| B | `static` | The capital of France is Paris. | The capital of France is Paris. | The capital of France is Paris. |
| B | `continuous` | The capital of France is Paris. | The capital of France is Paris. | The capital of France is Paris. |

`smoke-consistency.log`: `scripts/performance-tests/smoke-consistency.sh` on build B (`JUNO_JAR` set to it), all 54 checks
pass, including zero prefix-cache hits for stateless static-batch traffic (CPU and GPU).

## How to read it

- On build A, `user-b`'s prompt matched `user-a`'s cached prompt, so its request started decoding at the last prompt position
  with nothing written under its own key before it. The garbled reply is the model reading that empty KV.
- On build B, `user-b` looks up only entries cached under `user-b`, finds none, and prefills its whole prompt.
- `server-*.log` are the four servers' logs (A = `base`, B = `cand`), with color codes and local paths removed.
