(ch-3-7)=
# 3.7. Test Mode

`./juno test` runs 8 automated real-model smoke checks (6 pipeline-parallel, 2 tensor-parallel)
against an actual GGUF file, and exits `0` if all pass or `1` if any fail. It is the fastest way
to confirm a build works end to end on real weights before relying on it.

```bash
./juno test --model-path /path/to/model.gguf
```

**Windows (Command Prompt):**

```bat
juno.bat test --model-path models\model.gguf
```

This is distinct from the Maven-driven unit and integration test suites, which run without a
real model file. See [Build and test](#ch-8-1) for the full test matrix.

## See also

- [Chapter 3.1 -- Commands](#ch-3-1)
- [Chapter 8.1 -- Build and Test](#ch-8-1)

---

[<- 3.6 Merge Mode](#ch-3-6) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [3.8 Diagnostics and Tracing ->](#ch-3-8)
