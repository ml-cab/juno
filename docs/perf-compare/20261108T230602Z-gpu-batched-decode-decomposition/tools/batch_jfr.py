#!/usr/bin/env python3
"""Untraced batched decode step from a JFR recording (juno-perf.jfc).

The step: each request's interval between consecutive juno.TokenProduced events, median over every request's
intervals inside the window where all N requests decode (from the latest second token to the earliest last token).
Host terms per step: the juno.MatVec, juno.RmsNorm, juno.Rope, juno.ResidualAdd, juno.SwiGlu and juno.Attention
durations that start inside that window, summed per event type and divided by the steps in the window; GC pauses
(jdk.GCPhasePause) the same way. juno.MatVec spans a whole device matmul call: packing, copies, kernel and the wait
for it. A batched step emits no juno.ForwardPass; a single request's step does, and its median is reported too.
Usage: batch_jfr.py REC.jfr N [--json OUT]
"""
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

SPANS = ["juno.MatVec", "juno.RmsNorm", "juno.Rope", "juno.ResidualAdd", "juno.SwiGlu", "juno.Attention",
         "jdk.GCPhasePause"]


def events(rec, name):
    out = subprocess.run(["jfr", "print", "--json", "--events", name, rec], capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out)["recording"]["events"]


def dur_ms(s):  # ISO-8601 "PT0.018799568S"
    s = s[2:-1]
    return float(s) * 1000 if s else 0.0


def ts(s):
    base, frac_tz = s[:19], s[19:]
    frac = frac_tz.split("-")[0].split("+")[0]
    return datetime.fromisoformat(base).timestamp() + float("0" + frac)


rec, n = sys.argv[1], int(sys.argv[2])
out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
by_req = defaultdict(list)
for e in events(rec, "juno.TokenProduced"):
    by_req[e["values"]["requestId"]].append(ts(e["values"]["startTime"]))
times = [sorted(v) for v in by_req.values()]
lo = max(t[1] for t in times)
hi = min(t[-1] for t in times)
gaps, steps_in = [], []
for t in times:
    inside = [x for x in t if lo <= x <= hi]
    steps_in.append(len(inside) - 1)
    gaps += [(b - a) * 1000 for a, b in zip(inside, inside[1:])]
steps = statistics.median(steps_in)
res = {"requests": len(times), "n": n, "window_s": hi - lo, "steps_in_window": steps,
       "step_ms": statistics.median(gaps), "tokens_per_s": n * 1000 / statistics.median(gaps), "per_step_ms": {},
       "per_step_count": {}}
for name in SPANS:
    tot, cnt = 0.0, 0
    for e in events(rec, name):
        if lo <= ts(e["values"]["startTime"]) <= hi:
            tot += dur_ms(e["values"]["duration"])
            cnt += 1
    res["per_step_ms"][name] = tot / steps if steps else 0.0
    res["per_step_count"][name] = cnt / steps if steps else 0.0
fp = [dur_ms(e["values"]["duration"]) for e in events(rec, "juno.ForwardPass") if e["values"]["startPosition"] > 0]
res["forward_pass_ms"] = statistics.median(fp) if fp else None
print(json.dumps(res))
if out:
    json.dump(res, open(out, "w"), indent=1)
