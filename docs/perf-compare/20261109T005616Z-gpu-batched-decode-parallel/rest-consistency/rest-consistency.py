#!/usr/bin/env python3
"""REST check: single-request replies equal replies batched 8 at a time (--parallel 8), 64 tokens, greedy."""
import json, subprocess, sys, time, urllib.request, concurrent.futures, os
root, jar, out, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
models = [("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "4g"), ("qwen2.5-3b-instruct-q4_k_m.gguf", "6g"),
          ("Phi-3.5-mini-instruct-Q4_K_M.gguf", "6g"), ("mistral-7b-instruct-v0.1-q4_k_m.gguf", "9g")]
prompts = ["What is the capital of France?", "Name one primary color.", "How many legs does a spider have?"]
system = "You are a helpful assistant. Answer in one short sentence."
os.makedirs(out, exist_ok=True)
base = f"http://127.0.0.1:{port}"

def post(payload):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]

def ask(mid, p):
    return post({"model": mid, "temperature": 0, "max_tokens": 64, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompts[p]}]})

results = []
for file, heap in models:
    log = open(f"{out}/{file}.log", "wb")
    cmd = ["java", "--enable-preview", "--enable-native-access=ALL-UNNAMED", "--add-opens", "java.base/java.lang=ALL-UNNAMED",
           "--add-opens", "java.base/java.nio=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector", "-XX:+UseG1GC", "-Xms512m",
           f"-Xmx{heap}", "-Djuno.byteOrder=BE", "-jar", jar, "--model-path", f"{root}/models/{file}", "--dtype", "FLOAT16",
           "--byteOrder", "BE", "--max-tokens", "64", "--temperature", "0", "--top-k", "0", "--top-p", "0", "--nodes", "1",
           "--local", "--gpu", "--api-port", str(port), "--parallel", "8", "--batch-window-ms", "300"]
    srv = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=5).read(); break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("server did not start")
        mid = json.load(urllib.request.urlopen(base + "/v1/models", timeout=5))["data"][0]["id"]
        single = [ask(mid, p) for p in range(3)]
        order = [i % 3 for i in range(8)]
        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            batched = list(ex.map(lambda p: ask(mid, p), order))
        diffs = sum(1 for k, p in enumerate(order) if batched[k] != single[p])
        results.append({"model": file, "differing": diffs, "of": len(order), "single": single, "batched": batched})
        print(f"{file}: {diffs} of {len(order)} batched replies differ from the single-request reply", flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except Exception:
            srv.kill()
        log.close()
    time.sleep(3)
json.dump(results, open(f"{out}/results.json", "w"), indent=1)
