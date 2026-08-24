(ch-6-3)=
# 6.3. Windows Notes

Juno runs on Windows with the same functionality as Linux/macOS, with the differences below.

## Launcher

`juno.bat` at the project root delegates to `scripts\run.bat`. Requires JDK 25+ on `PATH` or
`JAVA_HOME` set. All flags, environment variable overrides, and subcommands documented across
this site are identical across platforms; only the launcher name and path separators change.

## Path and syntax differences

- Use backslashes for paths, for example `--model-path models\model.gguf`.
- Multi-line commands use the caret (`^`) line continuation instead of the backslash (`\`) used
  in bash examples.
- Setting an environment variable for a single command uses `set VAR=value` on its own line
  (Command Prompt) rather than `VAR=value ./command`.

Every code example in this documentation that shows a `./juno ...` command has a Windows
equivalent using `juno.bat` with these substitutions; see the [CLI reference](#ch-3-1) pages for side-by-side examples.

## GPU support

CUDA GPU acceleration is supported on Windows (NVIDIA only). ROCm is Linux-only, so AMD GPU
acceleration and the AMD GPU test suite are not available on Windows. CPU-only inference works
identically on both platforms.

## See also

- [Chapter 1.1 -- Requirements](#ch-1-1)
- [Chapter 3.2 -- Flags](#ch-3-2)
- [Chapter 2.4 -- GPU Acceleration](#ch-2-4)

---

[<- 6.2 AWS Deployment](#ch-6-2) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [7.1 JFR and Metrics ->](#ch-7-1)
