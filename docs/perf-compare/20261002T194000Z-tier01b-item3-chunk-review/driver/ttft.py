#!/usr/bin/env python3
"""Prefill latency probe: calibrated raw prompt, max_tokens=1, warmup then measured reps.

usage: ttft.py PORT N_PROMPT [WARMUP] [REPS]
Prints one JSON line: prompt_tokens, per-rep ms, median/min/max ms, prefill t/s at the median.
"""
import json, statistics, sys, time, urllib.request

port, n_prompt = int(sys.argv[1]), int(sys.argv[2])
warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 2
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
base = f"http://127.0.0.1:{port}"

def get(path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.load(r)

def chat(model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return (time.perf_counter() - t0) * 1000.0, out

models = get("/v1/models")
model = (models.get("data") or [{}])[0].get("id") or models["models"][0]["modelId"]
words = lambda n: " ".join(["x"] * max(1, n))
_, probe = chat(model, words(n_prompt))
overhead = probe["usage"]["prompt_tokens"] - n_prompt
prompt = words(n_prompt - overhead)
for _ in range(warmup):
    chat(model, prompt)
ms, toks = [], None
for _ in range(reps):
    t, out = chat(model, prompt)
    ms.append(round(t, 1))
    toks = out["usage"]["prompt_tokens"]
med = statistics.median(ms)
print(json.dumps({"prompt_tokens": toks, "ms": ms, "median_ms": med, "min_ms": min(ms), "max_ms": max(ms),
                  "spread_pct": round(100 * (max(ms) - min(ms)) / med, 1),
                  "prefill_tps_at_median": round(toks / (med / 1000.0), 2)}))
