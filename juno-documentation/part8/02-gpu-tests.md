(ch-8-2)=
# 8.2. GPU Tests

**GPU tests** (NVIDIA: requires CUDA 12.x and an NVIDIA GPU):

```bash
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

mvn verify -Pgpu -Dit.model.path=/path/to/model.gguf -pl juno-master \
  --enable-native-access=ALL-UNNAMED
```

**Windows (NVIDIA GPU tests):**
```bat
mvn test -Dgroups=gpu -pl node --enable-native-access=ALL-UNNAMED

mvn verify -Pgpu -Dit.model.path=C:\models\model.gguf -pl juno-master ^
  --enable-native-access=ALL-UNNAMED
```

**GPU tests** (AMD: requires ROCm 6+ and an AMD GPU):

```bash
mvn test -Dgroups=rocm -pl node --enable-native-access=ALL-UNNAMED
```

> **Note:** ROCm is Linux-only. AMD GPU tests are not supported on Windows.

## See also

- [Chapter 8.1 -- Build and Test](#ch-8-1)
- [Chapter 2.4 -- GPU Acceleration](#ch-2-4)

---

[<- 8.1 Build and Test](#ch-8-1) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [9.1 License and Patents ->](#ch-9-1)
