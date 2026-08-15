(ch-4-8)=
# 4.8. Testing Checklist

```bash
mvn test -Dtest=LoraAdapterTest                    # numerical gradient check (most important)
mvn test -Dtest=LoraAdapterSetTest                 # round-trip serialisation v1/v2
mvn test -Dtest=LoraAdamOptimizerTest              # update direction, A-only decay, LoRA+
mvn test -Dtest=LoraTrainableHandlerTest           # adjointness: dot(A*x,v) == dot(A^T*v,x)
mvn test -Dtest=LoraMicrobatchTest                 # bounds, apply/current, blank/default
mvn test -Dtest=LoraCorpusLimitTest                # seeded subsampling, budget limits
mvn test -Dtest=LoraTrainableHandlerGpuBackwardTest  # CPU/GPU parity, speed gates
mvn test -pl node -Dgroups=gpu                     # GPU adjoint, parity (NVIDIA)
mvn test -pl node -Dgroups=rocm                    # GPU adjoint, parity (AMD)
```

## See also

- [Chapter 4.7 -- Common Pitfalls](#ch-4-7)
- [Chapter 8.1 -- Build and Test](#ch-8-1)
- [Chapter 8.2 -- GPU Tests](#ch-8-2)

---

[<- 4.7 Common Pitfalls](#ch-4-7) &nbsp;|&nbsp; [Table of Contents](../index.md) &nbsp;|&nbsp; [5.1 OpenAI-Compatible API ->](#ch-5-1)
