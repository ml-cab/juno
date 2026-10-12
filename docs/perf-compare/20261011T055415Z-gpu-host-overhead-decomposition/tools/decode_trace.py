#!/usr/bin/env python3
"""Per-token decode decomposition from an nsys sqlite export.

Token boundaries are the logits downloads (the largest device-to-host copy size that
occurs once per forward pass). For each interval between two consecutive logits
downloads (one decode pass plus sampling) it reports wall time, device-busy time (union
of kernel and copy intervals), device-idle time, kernel time by kernel, copy time, and the
host's CUDA API time by call. Medians over the decode intervals.
Usage: decode_trace.py TRACE.sqlite LAYERS [--logit-bytes N] [--json OUT]
"""
import json
import sqlite3
import statistics
import sys
from collections import defaultdict

db, layers = sys.argv[1], int(sys.argv[2])
out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
kernels = [(s, e, names[n]) for s, e, n in c.execute(
    "select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
copies = [(s, e, k, b) for s, e, k, b in c.execute(
    "select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
api = [(s, e, names[n]) for s, e, n in c.execute(
    "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]

# Logits download: the D2H size whose count is closest to the number of forward passes,
# taken as the largest D2H size occurring at least 8 times.
d2h = defaultdict(int)
for s, e, k, b in copies:
    if k == 2:
        d2h[b] += 1
logit_bytes = (int(sys.argv[sys.argv.index("--logit-bytes") + 1]) if "--logit-bytes" in sys.argv
               else max(b for b, n in d2h.items() if n >= 8))
bounds = [e for s, e, k, b in copies if k == 2 and b == logit_bytes]


def clip(items, lo, hi):
    return [(max(s, lo), min(e, hi)) + tuple(x) for s, e, *x in items if e > lo and s < hi]


def union(iv):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


rows = []
for lo, hi in zip(bounds[1:-1], bounds[2:]):
    ks = clip(kernels, lo, hi)
    cs = clip(copies, lo, hi)
    aps = clip(api, lo, hi)
    busy = union([(s, e) for s, e, *_ in ks] + [(s, e) for s, e, *_ in cs])
    by_kernel = defaultdict(float)
    for s, e, n in ks:
        by_kernel[n] += e - s
    by_api = defaultdict(float)
    n_api = defaultdict(int)
    for s, e, n in aps:
        by_api[n] += e - s
        n_api[n] += 1
    rows.append({
        "wall": hi - lo, "busy": busy, "idle": hi - lo - busy,
        "kernel": union([(s, e) for s, e, *_ in ks]),
        "copy": union([(s, e) for s, e, *_ in cs]),
        "launches": sum(1 for _ in ks), "copies": len(cs),
        "by_kernel": by_kernel, "by_api": by_api, "n_api": n_api,
        "h2d": sum(1 for s, e, k, b in cs if k == 1), "d2h": sum(1 for s, e, k, b in cs if k == 2),
    })

ms = lambda ns: ns / 1e6
med = lambda key: statistics.median(r[key] for r in rows)
res = {
    "intervals": len(rows), "layers": layers, "logit_bytes": logit_bytes,
    "wall_ms": ms(med("wall")), "device_busy_ms": ms(med("busy")), "device_idle_ms": ms(med("idle")),
    "kernel_ms": ms(med("kernel")), "copy_ms": ms(med("copy")),
    "launches": med("launches"), "copies": med("copies"), "h2d": med("h2d"), "d2h": med("d2h"),
    "kernels_ms": {}, "api_ms": {}, "api_count": {},
}
for k in sorted({k for r in rows for k in r["by_kernel"]}):
    res["kernels_ms"][k] = ms(statistics.median(r["by_kernel"].get(k, 0) for r in rows))
for k in sorted({k for r in rows for k in r["by_api"]}):
    res["api_ms"][k] = ms(statistics.median(r["by_api"].get(k, 0) for r in rows))
    res["api_count"][k] = statistics.median(r["n_api"].get(k, 0) for r in rows)
res["kernels_ms"] = dict(sorted(res["kernels_ms"].items(), key=lambda kv: -kv[1]))
res["api_ms"] = dict(sorted(res["api_ms"].items(), key=lambda kv: -kv[1]))

print(f"{db}: {len(rows)} decode intervals, logits {logit_bytes} B, {layers} layers")
print(f"  wall {res['wall_ms']:.3f} ms  busy {res['device_busy_ms']:.3f}  idle {res['device_idle_ms']:.3f}"
      f"  (kernel {res['kernel_ms']:.3f}, copy {res['copy_ms']:.3f})")
print(f"  launches {res['launches']:.0f}  copies {res['copies']:.0f} (h2d {res['h2d']:.0f}, d2h {res['d2h']:.0f})"
      f"  per layer: busy {res['device_busy_ms'] / layers * 1000:.1f} us, launches {res['launches'] / layers:.1f}")
print("  kernels:", ", ".join(f"{k} {v:.3f}" for k, v in list(res["kernels_ms"].items())[:10]))
print("  api:", ", ".join(f"{k} {v:.3f} ({res['api_count'][k]:.0f})" for k, v in res["api_ms"].items()))
if out:
    json.dump(res, open(out, "w"), indent=1)
