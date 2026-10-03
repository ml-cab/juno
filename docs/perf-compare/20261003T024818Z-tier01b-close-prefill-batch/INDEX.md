# Prefill window width bake-off after the prefill-throughput work (2026-10-03)

Purpose: the `compare-prefill-batch.sh` reading the closing matrix of the prefill-throughput work calls for,
taken on HEAD and on the build before that work. **Unpinned single readings**, agent-run, a few minutes
apart: for orientation, not a gate. The gated prefill figures are the pinned closing sweeps.

Command: `compare-prefill-batch.sh --gpu --n-prompt 512 --prefill-values 1,32,512 --no-publish`, TinyLlama
1.1B Q4_K_M, one request per window width, each build from its own tree. `head-first/` is an earlier HEAD run
with the script's default widths (1 and 32), taken 55 minutes before, right after the vision runs.

Builds: HEAD `1ac490a` (shaded jar sha256 `66f02ee7c2908f78`). Baseline `ffd0ca7`, the build the 2026-09-30
reference sweeps measured, rebuilt from an export of that commit (jar `bd19a9306ea7901f`).

| `--prefill-batch` | Baseline `ffd0ca7` pp t/s | HEAD pp t/s | HEAD over baseline | HEAD, earlier run |
|---|---|---|---|---|
| 1 | 60.82 | 60.42 | 0.99x | 59.16 |
| 32 | 152.68 | 338.35 | 2.22x | 236.28 |
| 512 | 196.78 | 787.97 | 4.00x | not run |

How to read it:
- Width 1 is one forward pass per token, the decode path, which the prefill work did not change: flat.
- At widths 32 and 512 every window runs on the device region (more than eight rows), so the gain grows with
  width: the fixed per-layer upload and download are spread over more rows. The baseline's
  `attention_share_pct` of 12% to 14% is 0 on HEAD because attention runs inside the region, where the
  host attention span does not see it.
- The two HEAD readings at width 32 differ by 43%. A single unpinned request is not a measurement at this
  resolution (the first was taken straight after a 9-minute CPU-bound vision run); the 2.2x and 4.0x are
  indicative only.
