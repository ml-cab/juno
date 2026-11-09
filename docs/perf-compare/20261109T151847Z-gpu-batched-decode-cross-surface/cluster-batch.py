#!/usr/bin/env python3
"""A decode batch on a real cluster (forked node JVMs), through REST, GPU, greedy: pipeline and tensor at --parallel 1 and 8,
the two settings alternating. Reports replies against the single-request replies, aggregate t/s and mean latency, and
whether the startup statement was printed."""
import json, statistics, subprocess, sys, time, urllib.request, concurrent.futures, os
root, jar, out, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
statement = sys.argv[5] if len(sys.argv) > 5 else ""
file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
consistency = ["What is the capital of France?", "Name one primary color.", "How many legs does a spider have?"]
load_prompt = sys.argv[6] if len(sys.argv) > 6 else "Explain why the sky is blue in a few sentences."
os.makedirs(out, exist_ok=True)
base = f"http://127.0.0.1:{port}"

def ask(mid, prompt, n):
    body = {"model": mid, "temperature": 0, "max_tokens": n, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"], time.time() - t0

def serve(ptype, parallel, tag):
    logpath = f"{out}/server-{tag}.log"
    log = open(logpath, "wb")
    cmd = ["java", "--enable-preview", "--enable-native-access=ALL-UNNAMED", "--add-opens", "java.base/java.lang=ALL-UNNAMED",
           "--add-opens", "java.base/java.nio=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector", "-XX:+UseG1GC", "-Xms512m",
           "-Xmx4g", "-Djuno.byteOrder=BE", "-jar", jar, "--model-path", f"{root}/models/{file}", "--dtype", "FLOAT16",
           "--byteOrder", "BE", "--max-tokens", "64", "--temperature", "0", "--top-k", "0", "--top-p", "0", "--gpu",
           "--pType", ptype, "--api-port", str(port), "--parallel", str(parallel), "--batch-window-ms", "300"]
    srv = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        if srv.poll() is not None:
            raise RuntimeError(f"server exited {srv.returncode}, see {logpath}")
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=5).read(); break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("server did not start")
    return srv, log, logpath, json.load(urllib.request.urlopen(base + "/v1/models", timeout=5))["data"][0]["id"]

def stop(srv, log):
    try:
        srv.stdin.close()
    except Exception:
        pass
    srv.terminate()
    try:
        srv.wait(45)
    except Exception:
        srv.kill()
    log.close()
    time.sleep(6)

result = {"model": file, "runs": []}
single = {}
ok = True
for ptype in ("pipeline", "tensor"):
    for rep in (1, 2):
        for parallel in (1, 8):
            tag = f"{ptype}-p{parallel}-{rep}"
            srv, log, logpath, mid = serve(ptype, parallel, tag)
            try:
                ask(mid, load_prompt, 16)  # JIT warm-up, not scored
                if parallel == 1 and ptype not in single:
                    single[ptype] = [ask(mid, p, 64)[0] for p in consistency]
                rounds = []
                for _ in range(3):
                    t0 = time.time()
                    with concurrent.futures.ThreadPoolExecutor(8) as ex:
                        rs = list(ex.map(lambda _i: ask(mid, load_prompt, 64), range(8)))
                    wall = time.time() - t0
                    rounds.append({"tps": sum(r[1] for r in rs) / wall, "latency_s": statistics.mean(r[2] for r in rs),
                                   "tokens": sum(r[1] for r in rs), "same_reply": len({r[0] for r in rs}) == 1})
                order = [i % len(consistency) for i in range(8)]
                with concurrent.futures.ThreadPoolExecutor(8) as ex:
                    replies = list(ex.map(lambda p: ask(mid, consistency[p], 64)[0], order))
                diffs = sum(1 for k, p in enumerate(order) if replies[k] != single[ptype][p])
            finally:
                stop(srv, log)
            text = open(logpath, errors="replace").read()
            stated = bool(statement) and statement in text
            row = {"pType": ptype, "parallel": parallel, "rep": rep, "rounds": rounds, "differing": diffs, "of": len(order),
                   "statement_printed": stated, "replies": replies}
            result["runs"].append(row)
            med = statistics.median(r["tps"] for r in rounds)
            print(f"{tag}: median {med:.1f} t/s, latency {statistics.median(r['latency_s'] for r in rounds):.2f} s, differ {diffs}/{len(order)}, statement {stated}", flush=True)
            ok = ok and diffs == 0 and (not statement or stated == (parallel > 1))
result["single"] = single
json.dump(result, open(f"{out}/results.json", "w"), indent=1)
print("OK" if ok else "CHECK", flush=True)
sys.exit(0 if ok else 1)
