# Mean NLL on the model-quality text with LF and with CR LF line ends

**Purpose.** The kit's `quality.*` probes tokenize the model-quality text (`model-quality-nll.txt`, a resource of the
kit jar, built from the tester's checkout) and teacher-force it through the model; `ingest` holds a remote run's mean
NLL to at most 1.01x the reference run's. Before `.gitattributes` existed, a Windows checkout made with Git's installer
default (`core.autocrlf=true`) wrote that text with CR LF line ends, 77 lines, so a Windows run scored a different
token sequence from the Linux reference. This reading measures how far that moves the score, on one model, so that
the effect on the testers' runs at the kit's reference commit is a number.

**Pinned:** no. Not a throughput reading; the quantity is a mean NLL, which does not depend on clocks.

**Build.** Kit jar SHA-256 `666c282aac7502a1b91684848d9058072174316f543fa4f5b444c18cdddc932f` (Juno `d8012b11` plus the
uncommitted CUDA library and kit-probe changes; its quality code and text are those of `80ec949e`, the commit the
testers build). Host: Xeon E5-1650 v2, OpenJDK 25.0.4.1, Linux. Model `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`, CPU,
integer kernels.

**Commands.** The probe's scoring JVM, once on the text as stored (LF) and once with a CR LF copy placed ahead of the
jar on the class path, so the same code reads it:

```
java --enable-preview --enable-native-access=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED \
  --add-opens java.base/java.nio=ALL-UNNAMED --add-modules jdk.incubator.vector -XX:+UseG1GC -Xmx4g \
  -cp kit.jar cab.ml.juno.kit.QualityRunMain models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 2048
java ... -cp crlf:kit.jar cab.ml.juno.kit.QualityRunMain models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 2048
```

where `crlf/cab/ml/juno/kit/model-quality-nll.txt` is the stored text with a CR added before every LF
(`sed 's/$/\r/'`). 2026-11-07, 07:05:30Z to 07:08:28Z. `lf.json` and `crlf.json` are the two result lines.

## Result

| Text | Mean NLL over 2048 predictions, nats | Ratio to LF | Seconds |
|---|---|---|---|
| LF (as stored; a Linux checkout, or Windows with `.gitattributes`) | 2.45609 | 1.000 | 82.5 |
| CR LF (Windows checkout with `core.autocrlf=true`, before `.gitattributes`) | 2.39748 | **0.976** | 84.9 |

The two runs' argmax predictions agree up to position 85 and part at position 86, at the blank line after the text's
first paragraph, where the CR LF text carries its first CRs; from there the two texts are shifted against each other.

## How to read it

- The CR LF text scores **lower**, not higher: a CR before every newline is easy to predict once seen. `ingest`'s bound
  is one-sided (at most 1.01x the reference), so a Windows run made from a CR LF checkout does not fail it falsely. It
  passes while scoring a different text, and in doing so it could hide a numeric regression of up to about 3.5% on this
  model (1.01 / 0.976).
- The probe's own check (integer kernels at most 1.01x the float-activation kernels, both on the tester's host) reads
  the same text on both sides and is unaffected.
- One model only. Other tokenizers split CR differently, so the size of the shift differs per model; its cause does not.
- From the commit that adds `.gitattributes` on, every checkout writes the text with LF on every OS, so the comparison is
  like for like again.
