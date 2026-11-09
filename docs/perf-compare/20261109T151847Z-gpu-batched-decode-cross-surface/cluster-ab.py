#!/usr/bin/env python3
"""One request-stream cluster throughput, --parallel 1, the pre-tier jar (A) and the final jar (B) alternating A B A B A B,
pipeline and tensor, GPU, TinyLlama, unpinned. Each start: one warm-up, then three rounds of 8 concurrent 64-token requests."""
import json, statistics, subprocess, sys, time, urllib.request, concurrent.futures, os
root, jar_a, jar_b, out, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; prompt = "could you please write me a short poem about love and war"
os.makedirs(out, exist_ok=True); base = f"http://127.0.0.1:{port}"
def ask(mid, n):
    body = {"model": mid, "temperature": 0, "max_tokens": n, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["usage"]["completion_tokens"]
def run(jar, ptype, tag):
    log = open(f"{out}/server-{tag}.log", "wb")
    cmd = ["java", "--enable-preview", "--enable-native-access=ALL-UNNAMED", "--add-opens", "java.base/java.lang=ALL-UNNAMED",
           "--add-opens", "java.base/java.nio=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector", "-XX:+UseG1GC", "-Xms512m", "-Xmx4g",
           "-Djuno.byteOrder=BE", "-jar", jar, "--model-path", f"{root}/models/{file}", "--dtype", "FLOAT16", "--byteOrder", "BE",
           "--max-tokens", "64", "--temperature", "0", "--top-k", "0", "--top-p", "0", "--gpu", "--pType", ptype, "--api-port", str(port),
           "--parallel", "1", "--batch-window-ms", "300"]
    srv = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if srv.poll() is not None: raise RuntimeError(f"exited {srv.returncode}")
            try: urllib.request.urlopen(base + "/v1/models", timeout=5).read(); break
            except Exception: time.sleep(2)
        mid = json.load(urllib.request.urlopen(base + "/v1/models", timeout=5))["data"][0]["id"]
        ask(mid, 16); rounds = []
        for _ in range(3):
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(8) as ex: toks = sum(ex.map(lambda _i: ask(mid, 64), range(8)))
            rounds.append(toks / (time.time() - t0))
    finally:
        try: srv.stdin.close()
        except Exception: pass
        srv.terminate()
        try: srv.wait(45)
        except Exception: srv.kill()
        log.close(); time.sleep(6)
    return rounds
res = {}
for ptype in ("pipeline", "tensor"):
    res[ptype] = {"A": [], "B": []}
    for rep in (1, 2, 3):
        for side, jar in (("A", jar_a), ("B", jar_b)):
            r = run(jar, ptype, f"{ptype}-{side}-{rep}"); res[ptype][side].append(r)
            print(f"{ptype} {side} {rep}: " + " ".join(f"{x:.1f}" for x in r), flush=True)
    a = statistics.median(statistics.median(x) for x in res[ptype]["A"]); b = statistics.median(statistics.median(x) for x in res[ptype]["B"])
    res[ptype]["median_A"] = a; res[ptype]["median_B"] = b; res[ptype]["B_over_A"] = b / a
    print(f"{ptype}: A {a:.2f} B {b:.2f} B/A {b/a:.3f}", flush=True)
json.dump(res, open(f"{out}/results.json", "w"), indent=1)
