#!/usr/bin/env python3
"""--lora-play with a decode batch, through REST: the startup statement is printed, and replies batched 8 at a time
equal the single-request replies, 64 tokens, greedy."""
import json, subprocess, sys, time, urllib.request, concurrent.futures, os
root, jar, adapter, out, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
prompts = ["What is the capital of France?", "Name one primary color.", "How many legs does a spider have?", "What is your name?"]
system = "You are a helpful assistant. Answer in one short sentence."
statement = "--lora-play decodes a batch"
os.makedirs(out, exist_ok=True)
base = f"http://127.0.0.1:{port}"

def ask(mid, p):
    body = {"model": mid, "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompts[p]}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]

logpath = f"{out}/server.log"
log = open(logpath, "wb")
cmd = ["java", "--enable-preview", "--enable-native-access=ALL-UNNAMED", "--add-opens", "java.base/java.lang=ALL-UNNAMED",
       "--add-opens", "java.base/java.nio=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector", "-XX:+UseG1GC", "-Xms512m",
       "-Xmx4g", "-Djuno.byteOrder=BE", "-jar", jar, "--model-path", f"{root}/models/{file}", "--dtype", "FLOAT16",
       "--byteOrder", "BE", "--max-tokens", "64", "--temperature", "0", "--top-k", "0", "--top-p", "0", "--nodes", "1",
       "--local", "--gpu", "--api-port", str(port), "--parallel", "8", "--batch-window-ms", "300", "--lora-play", adapter]
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
    single = [ask(mid, p) for p in range(len(prompts))]
    order = [i % len(prompts) for i in range(8)]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        batched = list(ex.map(lambda p: ask(mid, p), order))
    wall = time.time() - t0
finally:
    srv.terminate()
    try:
        srv.wait(30)
    except Exception:
        srv.kill()
    log.close()
diffs = sum(1 for k, p in enumerate(order) if batched[k] != single[p])
stated = statement in open(logpath, errors="replace").read()
json.dump({"model": file, "differing": diffs, "of": len(order), "statement_printed": stated, "batched_wall_s": round(wall, 2),
           "prompts": prompts, "single": single, "batched": batched}, open(f"{out}/results.json", "w"), indent=1)
print(f"{diffs} of {len(order)} batched replies differ; statement printed: {stated}; wall {wall:.1f} s", flush=True)
sys.exit(0 if diffs == 0 and stated else 1)
