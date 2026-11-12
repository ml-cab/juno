# Shared prompt prefixes on a real model: replies equal, time to answer

**Purpose.** Shows on a real model that a request started from a finished request's kept prompt
(`--prefix-cache-tokens`, default on) replies exactly as with sharing off, on both schedules, on the CPU and with CUDA
GPU attention (where the shared rows are copied into the request's device KV), and how much sooner it answers. A
correctness reading first; the times are one unpinned run each, not a gate.

**Pinned: no.** Xeon E5-1650 v2 (12 threads), GeForce GTX 1080; TinyLlama 1.1B Q4_K_M; greedy decoding
(`--temperature 0`), 16 tokens, `-Xmx6g`, `--parallel 1`.

**Builds.** One tree (`74f75d78` plus the uncommitted KV work and the sharing change), at two points of this session:

| Readings | Shaded jar (SHA-256, first 16) | Differs from the final jar by |
|---|---|---|
| `cpu/` | `6f9f165a0d5b7ba8` | the default budget (then one context, 32,768 here) and the startup line; not the sharing path |
| `gpu/` | `c18a3e1c8e228788` | the startup line only |
| final, smoke | `a261d4df8de1a18c` | - |

Each side of every comparison is the same jar: the server started with the default budget against the same server with
`--prefix-cache-tokens 0`.

## Command

```bash
JUNO_ROOT=<repo> shared-prefix-check.sh <jar> static|continuous default|0 <out-dir> cpu|gpu 1
```

The script starts a local server and sends nine requests in order: three stateless with a short system prompt (France,
colour, France again), sessions `user-a` and `user-b` with the France prompt, then three stateless and one session
(`user-c`) with a ~1,000-token system prompt (40 library rules) and different questions. It records each reply and its
`curl` time, and the server's `/v1/cluster/health` counters.

## Result

Replies: **9 of 9 equal** between sharing on and off in all four cases (CPU and GPU, `static` and `continuous`), and
equal to the earlier session-defect reading's replies for the France prompt. Health with sharing on: 7 shared-prefix
hits, 3,961 positions not recomputed, 3,915 held; `prefixHits` 0 (no request resumed from its own key).

Seconds per request, sharing on / off:

| # | Request | CPU `static` | CPU `continuous` | GPU `static` | GPU `continuous` |
|---|---|---|---|---|---|
| 1 | short, first (nothing held) | 6.12 / 6.00 | 6.08 / 6.08 | 0.42 / 0.33 | 0.95 / 0.91 |
| 2 | short, other question | 0.97 / 1.76 | 1.15 / 2.06 | 0.12 / 0.09 | 0.33 / 0.40 |
| 3 | short, France again | 0.81 / 2.28 | 1.15 / 2.65 | 0.12 / 0.14 | 0.53 / 0.60 |
| 4 | session `user-a`, France | 0.77 / 2.32 | 1.10 / 2.77 | 0.09 / 0.12 | 0.49 / 0.64 |
| 5 | session `user-b`, France | 0.72 / 2.27 | 1.08 / 2.67 | 0.08 / 0.11 | 0.44 / 0.64 |
| 6 | long, first (nothing held) | 46.79 / 45.89 | 49.03 / 48.86 | 1.33 / 1.32 | 4.76 / 4.68 |
| 7 | long, other question | **4.44 / 47.82** | **5.38 / 49.68** | **0.94 / 1.33** | **1.78 / 4.75** |
| 8 | session `user-c`, long | 3.15 / 47.83 | 3.84 / 48.79 | 0.53 / 0.93 | 1.14 / 4.19 |
| 9 | long, third question | 4.49 / 47.10 | 5.29 / 49.50 | 0.67 / 0.98 | 1.49 / 4.40 |

## How to read it

- Requests 1 and 6 hold nothing to share from, so both sides compute the whole prompt: their times agree within noise.
- From request 7 on, the long system prompt is shared. On the CPU the answer comes about 10x sooner; on the GPU, where
  prefill is fast, 1.4x (`static`) to 2.7x (`continuous`) sooner.
- Session `user-b` (request 5) opens with `user-a`'s message: before the session fix it replied with garbled text; here
  it starts from the kept copy of `user-a`'s prompt and replies as with sharing off.
- **Watch item, not addressed:** on the GPU `static` lane, request 2 (about 25 shared positions) reads 0.12 s against
  0.09 s. Copying a short prefix into the device KV of every layer can cost more than its prefill on a GPU. One unpinned
  reading; the 16-position minimum may need to be higher with GPU attention.
- `smoke-consistency.log`: `scripts/performance-tests/smoke-consistency.sh` on the final jar (`JUNO_JAR` set to it), 56 of
  56 pass, CPU and GPU. Its static-batch rounds (`--parallel 2`) repeat each prompt after its first, unshared answer; every
  repeat equals it, and the server counts 19 shared-prefix hits on each device.
- `server-*.log` are the servers' logs with color codes and local paths removed.
