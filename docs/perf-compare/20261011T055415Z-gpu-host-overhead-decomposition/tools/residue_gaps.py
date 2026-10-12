#!/usr/bin/env python3
"""Where a prefill window's unattributed time sits: the intervals of the prefill juno.PrefillBatch that no top-level span on the same thread covers, the GC pauses that
overlap them, and the allocation samples taken inside them.
Usage: residue_gaps.py REC.jfr"""
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime

rec = sys.argv[1]


def events(names):
    out = subprocess.run(["jfr", "print", "--json", "--events", names, rec], capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out)["recording"]["events"]


def ts(s):
    base, rest = s[:19], s[19:]
    frac = rest.split("-")[0].split("+")[0]
    return datetime.fromisoformat(base).timestamp() + float("0" + frac)


def dur(s):
    s = s[2:-1]
    return float(s) if s else 0.0


def iv(e):
    v = e["values"]
    st = ts(v["startTime"])
    return st, st + dur(v["duration"])


fp = max(events("juno.PrefillBatch"), key=lambda e: dur(e["values"]["duration"]))
lo, hi = iv(fp)
thread = fp["values"]["eventThread"]["javaThreadId"]
spans = [iv(e) + (e["type"] if "type" in e else e["values"].get("step", ""), e["values"].get("step", ""))
         for e in events("juno.WindowStep,juno.Attention,juno.RmsNorm,juno.Rope,juno.ResidualAdd,juno.SwiGlu")
         if e["values"]["eventThread"]["javaThreadId"] == thread and iv(e)[1] > lo and iv(e)[0] < hi]
spans.sort()
gaps, cur, prev_label = [], lo, "window start"
for s, e, t, step in spans:
    if s > cur + 1e-6:
        gaps.append((cur, s, prev_label, step))
    if e > cur:
        cur, prev_label = e, step
if hi > cur:
    gaps.append((cur, hi, prev_label, "window end"))
total = sum(b - a for a, b, *_ in gaps)
print(f"window {1000 * (hi - lo):.1f} ms, uncovered {1000 * total:.1f} ms in {len(gaps)} gaps")
by_pos = Counter()
for a, b, before, after in gaps:
    by_pos[f"{before} -> {after}"] += 1000 * (b - a)
for k, v in by_pos.most_common(8):
    print(f"  {v:8.1f} ms  {k}")
big = sorted(gaps, key=lambda g: g[0] - g[1])[:5]
for a, b, before, after in big:
    print(f"  largest: {1000 * (b - a):.1f} ms at +{1000 * (a - lo):.1f} ms ({before} -> {after})")
gc = [iv(e) for e in events("jdk.GCPhasePause")]
gc_in = sum(max(0, min(b, gb) - max(a, ga)) for a, b, *_ in gaps for ga, gb in gc)
gc_win = sum(max(0, min(hi, gb) - max(lo, ga)) for ga, gb in gc)
print(f"GC pause inside window {1000 * gc_win:.1f} ms, inside uncovered gaps {1000 * gc_in:.1f} ms")
alloc = Counter()
for e in events("jdk.ObjectAllocationSample"):
    v = e["values"]
    t = ts(v["startTime"])
    if v["eventThread"]["javaThreadId"] != thread or not (lo <= t <= hi):
        continue
    inside = any(a <= t <= b for a, b, *_ in gaps)
    frames = (v.get("stackTrace") or {}).get("frames") or []
    site = next((f["method"]["type"]["name"].split(".")[-1] + "." + f["method"]["name"] + ":" + str(f["lineNumber"])
                 for f in frames if "juno" in f["method"]["type"]["name"]), "?")
    alloc[("gap" if inside else "span", site)] += v["weight"]
print("allocation samples (weight MB) by site, inside gaps vs spans:")
for (where, site), w in alloc.most_common(12):
    print(f"  {where:4s} {w / 1e6:9.1f}  {site}")
