#!/usr/bin/env python3
"""Untraced decode step from a JFR recording: median juno.ForwardPass duration of the
one-row (decode) passes and median interval between consecutive juno.TokenProduced
events (the whole step: forward pass, sampling, detokenize, scheduling).
Usage: decode_jfr.py REC.jfr"""
import json
import statistics
import subprocess
import sys
from datetime import datetime


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


rec = sys.argv[1]
fp = events(rec, "juno.ForwardPass")
decode = [dur_ms(e["values"]["duration"]) for e in fp if e["values"]["startPosition"] > 0]
tp = sorted(ts(e["values"]["startTime"]) for e in events(rec, "juno.TokenProduced"))
gaps = [(b - a) * 1000 for a, b in zip(tp[1:], tp[2:])]
res = {"decode_passes": len(decode), "forward_ms": statistics.median(decode),
       "token_interval_ms": statistics.median(gaps), "tg_tps": 1000 / statistics.median(gaps)}
print(json.dumps(res))
