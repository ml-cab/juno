before run.sh change

LOCAL
```
     "name": "TinyLlama-1.1B-Chat-v1.0.Q5_K_M",
      "path": "../models/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile",
      "jfrFile": "juno-TinyLlama-1.1B-Chat-v1.0.Q5_K_M-20260404-010233.jfr",
      "metrics": {
        "jfr.file.bytes": 2021150.0,
        "juno.MatVec.count": 29605.0,
        "juno.MatVec.duration.total_ms": 41845.075087,
        "juno.MatVec.duration.p95_ms": 2.517087,
        "juno.MatVec.backend.cpu.count": 0.0,
        "juno.MatVec.backend.cpu.p95_ms": 0.0,
        "juno.MatVec.backend.cuda.count": 0.0,
        "juno.MatVec.backend.cuda.p95_ms": 0.0,
        "juno.MatVec.backend.cuda_resident.count": 0.0,
        "juno.MatVec.backend.cuda_resident.p95_ms": 0.0,
        "juno.MatVec.backend.quantized_q6_k.count": 4011.0,
        "juno.MatVec.backend.quantized_q6_k.p95_ms": 2.571736,
        "juno.MatVec.backend.quantized_q5_k.count": 25594.0,
        "juno.MatVec.backend.quantized_q5_k.p95_ms": 2.516968,
        "juno.ForwardPass.count": 573.0,
        "juno.ForwardPass.prefill.count": 3.0,
        "juno.ForwardPass.decode.count": 570.0,
        "juno.ForwardPass.prefill.p95_ms": 369.507178,
        "juno.ForwardPass.decode.p95_ms": 197.033898,
        "juno.Tokenizer.encode.count": 3.0,
        "juno.Tokenizer.encode.p95_ms": 13.474224,
        "juno.Tokenizer.decodeToken.count": 62.0,
        "juno.Tokenizer.decodeToken.p95_ms": 0.083519,
        "juno.TemplateFormat.count": 3.0,
        "juno.TemplateFormat.p95_ms": 0.065114,
        "juno.LoraTrainStep.count": 0.0,
        "juno.LoraTrainStep.forward_ms.p95": 0.0,
        "juno.LoraTrainStep.backward_ms.p95": 0.0,
        "juno.LoraTrainStep.optimizer_ms.p95": 0.0
      }
```

CLUSTER

```
      "name": "TinyLlama-1.1B-Chat-v1.0.Q5_K_M",
      "path": "../models/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile",
      "jfrFile": "juno-TinyLlama-1.1B-Chat-v1.0.Q5_K_M-20260404-010101.jfr",
      "metrics": {
        "jfr.file.bytes": 803593.0,
        "juno.MatVec.count": 0.0,
        "juno.MatVec.duration.total_ms": 0.0,
        "juno.MatVec.duration.p95_ms": 0.0,
        "juno.MatVec.backend.cpu.count": 0.0,
        "juno.MatVec.backend.cpu.p95_ms": 0.0,
        "juno.MatVec.backend.cuda.count": 0.0,
        "juno.MatVec.backend.cuda.p95_ms": 0.0,
        "juno.MatVec.backend.cuda_resident.count": 0.0,
        "juno.MatVec.backend.cuda_resident.p95_ms": 0.0,
        "juno.ForwardPass.count": 0.0,
        "juno.ForwardPass.prefill.count": 0.0,
        "juno.ForwardPass.decode.count": 0.0,
        "juno.ForwardPass.prefill.p95_ms": 0.0,
        "juno.ForwardPass.decode.p95_ms": 0.0,
        "juno.Tokenizer.encode.count": 3.0,
        "juno.Tokenizer.encode.p95_ms": 13.423786,
        "juno.Tokenizer.decodeToken.count": 50.0,
        "juno.Tokenizer.decodeToken.p95_ms": 0.076187,
        "juno.TemplateFormat.count": 3.0,
        "juno.TemplateFormat.p95_ms": 0.068662,
        "juno.LoraTrainStep.count": 0.0,
        "juno.LoraTrainStep.forward_ms.p95": 0.0,
        "juno.LoraTrainStep.backward_ms.p95": 0.0,
        "juno.LoraTrainStep.optimizer_ms.p95": 0.0
      }
```

    after run.sh change:

LOCAL

```
      "name": "TinyLlama-1.1B-Chat-v1.0.Q5_K_M",
      "path": "../models/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile",
      "jfrFile": "juno-TinyLlama-1.1B-Chat-v1.0.Q5_K_M-20260404-005531.jfr",
      "metrics": {
        "jfr.file.bytes": 2527259.0,
        "juno.MatVec.count": 38292.0,
        "juno.MatVec.duration.total_ms": 53216.768629,
        "juno.MatVec.duration.p95_ms": 2.510659,
        "juno.MatVec.backend.cpu.count": 0.0,
        "juno.MatVec.backend.cpu.p95_ms": 0.0,
        "juno.MatVec.backend.cuda.count": 0.0,
        "juno.MatVec.backend.cuda.p95_ms": 0.0,
        "juno.MatVec.backend.cuda_resident.count": 0.0,
        "juno.MatVec.backend.cuda_resident.p95_ms": 0.0,
        "juno.MatVec.backend.quantized_q6_k.count": 5189.0,
        "juno.MatVec.backend.quantized_q6_k.p95_ms": 2.527675,
        "juno.MatVec.backend.quantized_q5_k.count": 33103.0,
        "juno.MatVec.backend.quantized_q5_k.p95_ms": 2.51061,
        "juno.ForwardPass.count": 741.0,
        "juno.ForwardPass.prefill.count": 3.0,
        "juno.ForwardPass.decode.count": 738.0,
        "juno.ForwardPass.prefill.p95_ms": 360.675099,
        "juno.ForwardPass.decode.p95_ms": 196.164471,
        "juno.Tokenizer.encode.count": 3.0,
        "juno.Tokenizer.encode.p95_ms": 40.699391,
        "juno.Tokenizer.decodeToken.count": 92.0,
        "juno.Tokenizer.decodeToken.p95_ms": 0.091591,
        "juno.TemplateFormat.count": 3.0,
        "juno.TemplateFormat.p95_ms": 0.065545,
        "juno.LoraTrainStep.count": 0.0,
        "juno.LoraTrainStep.forward_ms.p95": 0.0,
        "juno.LoraTrainStep.backward_ms.p95": 0.0,
        "juno.LoraTrainStep.optimizer_ms.p95": 0.0
      }
```

CLUSTER

```
      "name": "TinyLlama-1.1B-Chat-v1.0.Q5_K_M",
      "path": "../models/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile",
      "jfrFile": "juno-TinyLlama-1.1B-Chat-v1.0.Q5_K_M-20260404-004300.jfr",
      "metrics": {
        "jfr.file.bytes": 851646.0,
        "juno.MatVec.count": 0.0,
        "juno.MatVec.duration.total_ms": 0.0,
        "juno.MatVec.duration.p95_ms": 0.0,
        "juno.MatVec.backend.cpu.count": 0.0,
        "juno.MatVec.backend.cpu.p95_ms": 0.0,
        "juno.MatVec.backend.cuda.count": 0.0,
        "juno.MatVec.backend.cuda.p95_ms": 0.0,
        "juno.MatVec.backend.cuda_resident.count": 0.0,
        "juno.MatVec.backend.cuda_resident.p95_ms": 0.0,
        "juno.ForwardPass.count": 0.0,
        "juno.ForwardPass.prefill.count": 0.0,
        "juno.ForwardPass.decode.count": 0.0,
        "juno.ForwardPass.prefill.p95_ms": 0.0,
        "juno.ForwardPass.decode.p95_ms": 0.0,
        "juno.Tokenizer.encode.count": 4.0,
        "juno.Tokenizer.encode.p95_ms": 17.468139,
        "juno.Tokenizer.decodeToken.count": 55.0,
        "juno.Tokenizer.decodeToken.p95_ms": 0.081545,
        "juno.TemplateFormat.count": 4.0,
        "juno.TemplateFormat.p95_ms": 0.06,
        "juno.LoraTrainStep.count": 0.0,
        "juno.LoraTrainStep.forward_ms.p95": 0.0,
        "juno.LoraTrainStep.backward_ms.p95": 0.0,
        "juno.LoraTrainStep.optimizer_ms.p95": 0.0
      }
```

