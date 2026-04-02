// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.productivity;

import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class JfrMetricsExtractor {

    private static final String MAT_VEC = "juno.MatVec";
    private static final String FORWARD = "juno.ForwardPass";
    private static final String TOKENIZER = "juno.Tokenizer";
    private static final String TEMPLATE = "juno.TemplateFormat";
    private static final String LORA_STEP = "juno.LoraTrainStep";

    private JfrMetricsExtractor() {
    }

    static MetricsSnapshot.ModelMetrics extract(Path jfrFile, ModelsConfig.ModelEntry model) throws IOException {
        String jfrName = jfrFile.getFileName().toString();
        long fileBytes = Files.size(jfrFile);

        List<Long> matVecAll = new ArrayList<>();
        Map<String, List<Long>> matVecByBackend = new HashMap<>();

        List<Long> forwardAll = new ArrayList<>();
        List<Long> forwardPrefill = new ArrayList<>();
        List<Long> forwardDecode = new ArrayList<>();

        List<Long> tokEncode = new ArrayList<>();
        List<Long> tokDecodeToken = new ArrayList<>();

        List<Long> template = new ArrayList<>();

        List<Long> loraForwardMs = new ArrayList<>();
        List<Long> loraBackwardMs = new ArrayList<>();
        List<Long> loraOptimizerMs = new ArrayList<>();
        int loraStepCount = 0;

        try (RecordingFile rf = new RecordingFile(jfrFile)) {
            while (rf.hasMoreEvents()) {
                RecordedEvent ev = rf.readEvent();
                String type = ev.getEventType().getName();
                long nano = ev.getDuration().toNanos();

                switch (type) {
                    case MAT_VEC -> {
                        matVecAll.add(nano);
                        if (ev.hasField("backend")) {
                            String b = sanitizeBackend(ev.getString("backend"));
                            matVecByBackend.computeIfAbsent(b, k -> new ArrayList<>()).add(nano);
                        }
                    }
                    case FORWARD -> {
                        forwardAll.add(nano);
                        if (ev.hasField("startPosition")) {
                            int pos = ev.getInt("startPosition");
                            if (pos == 0) {
                                forwardPrefill.add(nano);
                            } else {
                                forwardDecode.add(nano);
                            }
                        }
                    }
                    case TOKENIZER -> {
                        if (ev.hasField("operation")) {
                            String op = ev.getString("operation");
                            if ("encode".equals(op)) {
                                tokEncode.add(nano);
                            } else if ("decodeToken".equals(op)) {
                                tokDecodeToken.add(nano);
                            }
                        }
                    }
                    case TEMPLATE -> template.add(nano);
                    case LORA_STEP -> {
                        loraStepCount++;
                        if (ev.hasField("forwardMs")) {
                            loraForwardMs.add(ev.getLong("forwardMs"));
                        }
                        if (ev.hasField("backwardMs")) {
                            loraBackwardMs.add(ev.getLong("backwardMs"));
                        }
                        if (ev.hasField("optimizerMs")) {
                            loraOptimizerMs.add(ev.getLong("optimizerMs"));
                        }
                    }
                    default -> {
                        /* ignore JDK and other events */
                    }
                }
            }
        }

        Map<String, Double> m = new LinkedHashMap<>();
        m.put("jfr.file.bytes", (double) fileBytes);

        m.put("juno.MatVec.count", (double) matVecAll.size());
        m.put("juno.MatVec.duration.total_ms", JfrPercentiles.sumNanosToMs(matVecAll));
        m.put("juno.MatVec.duration.p95_ms", JfrPercentiles.p95NanosToMs(matVecAll));

        List<String> legacyBackends = List.of("cpu", "cuda", "cuda_resident");
        for (String backend : legacyBackends) {
            List<Long> list = matVecByBackend.getOrDefault(backend, List.of());
            m.put("juno.MatVec.backend." + backend + ".count", (double) list.size());
            m.put("juno.MatVec.backend." + backend + ".p95_ms", JfrPercentiles.p95NanosToMs(list));
        }
        for (Map.Entry<String, List<Long>> e : matVecByBackend.entrySet()) {
            if (legacyBackends.contains(e.getKey())) {
                continue;
            }
            String key = e.getKey();
            m.put("juno.MatVec.backend." + key + ".count", (double) e.getValue().size());
            m.put("juno.MatVec.backend." + key + ".p95_ms", JfrPercentiles.p95NanosToMs(e.getValue()));
        }

        m.put("juno.ForwardPass.count", (double) forwardAll.size());
        m.put("juno.ForwardPass.prefill.count", (double) forwardPrefill.size());
        m.put("juno.ForwardPass.decode.count", (double) forwardDecode.size());
        m.put("juno.ForwardPass.prefill.p95_ms", JfrPercentiles.p95NanosToMs(forwardPrefill));
        m.put("juno.ForwardPass.decode.p95_ms", JfrPercentiles.p95NanosToMs(forwardDecode));

        m.put("juno.Tokenizer.encode.count", (double) tokEncode.size());
        m.put("juno.Tokenizer.encode.p95_ms", JfrPercentiles.p95NanosToMs(tokEncode));
        m.put("juno.Tokenizer.decodeToken.count", (double) tokDecodeToken.size());
        m.put("juno.Tokenizer.decodeToken.p95_ms", JfrPercentiles.p95NanosToMs(tokDecodeToken));

        m.put("juno.TemplateFormat.count", (double) template.size());
        m.put("juno.TemplateFormat.p95_ms", JfrPercentiles.p95NanosToMs(template));

        m.put("juno.LoraTrainStep.count", (double) loraStepCount);
        m.put("juno.LoraTrainStep.forward_ms.p95", JfrPercentiles.p95LongMs(loraForwardMs));
        m.put("juno.LoraTrainStep.backward_ms.p95", JfrPercentiles.p95LongMs(loraBackwardMs));
        m.put("juno.LoraTrainStep.optimizer_ms.p95", JfrPercentiles.p95LongMs(loraOptimizerMs));

        return new MetricsSnapshot.ModelMetrics(model.getName(), model.getPath(), jfrName, m);
    }

    private static String sanitizeBackend(String backend) {
        if (backend == null) {
            return "unknown";
        }
        return backend.replace('-', '_');
    }
}
