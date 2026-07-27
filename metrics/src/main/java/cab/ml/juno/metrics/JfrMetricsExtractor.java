/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cab.ml.juno.metrics;

import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * @author Yevhen Soldatov
 */

final class JfrMetricsExtractor {

    private static final String MAT_VEC = "juno.MatVec";
    private static final String FORWARD = "juno.ForwardPass";
    private static final String TOKENIZER = "juno.Tokenizer";
    private static final String TEMPLATE = "juno.TemplateFormat";
    private static final String LORA_STEP = "juno.LoraTrainStep";
    private static final String LORA_VAL = "juno.LoraValidation";
    private static final String LORA_NORM = "juno.LoraNormRefresh";
    private static final String LORA_MERGE = "juno.LoraMerge";
    private static final String LORA_PLAY = "juno.LoraPlayback";
    private static final String LORA_CKPT = "juno.LoraCheckpoint";
    private static final String TOKEN_PRODUCED = "juno.TokenProduced";

    private JfrMetricsExtractor() {
    }

    static MetricsSnapshot.ModelMetrics extract(Path jfrFile, ModelsConfig.ModelEntry model) throws IOException {
        return extractMerged(List.of(jfrFile), model);
    }

    /**
     * Reads JFR events from every file in {@code jfrFiles} into shared lists, then
     * computes percentiles once over the merged data.
     *
     * <p>This is the correct aggregation strategy for a cluster run where the coordinator
     * records Tokenizer/TemplateFormat events and each node records MatVec/ForwardPass
     * events — combining counts and percentiles before summarising gives accurate
     * cross-JVM metrics rather than per-JVM snapshots.
     *
     * @param jfrFiles one or more JFR files (coordinator + node files); missing files are silently skipped
     * @param model    model entry used for the output snapshot
     */
    static MetricsSnapshot.ModelMetrics extractMerged(List<Path> jfrFiles, ModelsConfig.ModelEntry model)
            throws IOException {

        String jfrName = jfrFiles.isEmpty() ? "merged" : jfrFiles.get(0).getFileName().toString();
        long totalFileBytes = 0;

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
        List<Long> loraTotalMs = new ArrayList<>();
        List<Long> loraFrozenForwardMs = new ArrayList<>();
        List<Long> loraFrozenTransposeMs = new ArrayList<>();
        List<Long> loraAdapterBackwardMs = new ArrayList<>();
        List<Long> loraTransferMs = new ArrayList<>();
        List<Float> loraLoss = new ArrayList<>();
        List<Float> loraGradNorm = new ArrayList<>();
        List<Integer> loraTokens = new ArrayList<>();
        List<Integer> loraPredictions = new ArrayList<>();
        int loraStepCount = 0;
        int loraClippedCount = 0;
        Map<String, List<Long>> loraTotalByAlgo = new TreeMap<>();
        Map<String, Integer> loraCountByAlgo = new TreeMap<>();

        List<Float> valLoss = new ArrayList<>();
        List<Float> valBestLoss = new ArrayList<>();
        List<Long> valDurationMs = new ArrayList<>();
        int valCount = 0;

        List<Long> normDurationMs = new ArrayList<>();
        int normCount = 0;

        List<Long> mergeDurationMs = new ArrayList<>();
        List<Float> mergeRmse = new ArrayList<>();
        List<Float> mergeDeltaRetention = new ArrayList<>();
        int mergeCount = 0;

        List<Long> playLoadMs = new ArrayList<>();
        int playCount = 0;

        int checkpointCount = 0;

        int tokenProducedCount = 0;
        Instant tokenProducedFirst = null;
        Instant tokenProducedLast = null;

        for (Path jfrFile : jfrFiles) {
            if (!Files.isRegularFile(jfrFile) || Files.size(jfrFile) == 0)
                continue;
            totalFileBytes += Files.size(jfrFile);
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
                            if (ev.hasField("forwardMs"))
                                loraForwardMs.add(ev.getLong("forwardMs"));
                            if (ev.hasField("backwardMs"))
                                loraBackwardMs.add(ev.getLong("backwardMs"));
                            if (ev.hasField("optimizerMs"))
                                loraOptimizerMs.add(ev.getLong("optimizerMs"));
                            if (ev.hasField("totalMs"))
                                loraTotalMs.add(ev.getLong("totalMs"));
                            if (ev.hasField("frozenForwardMs"))
                                loraFrozenForwardMs.add(ev.getLong("frozenForwardMs"));
                            if (ev.hasField("frozenTransposeBackwardMs"))
                                loraFrozenTransposeMs.add(ev.getLong("frozenTransposeBackwardMs"));
                            if (ev.hasField("adapterBackwardMs"))
                                loraAdapterBackwardMs.add(ev.getLong("adapterBackwardMs"));
                            if (ev.hasField("transferMs"))
                                loraTransferMs.add(ev.getLong("transferMs"));
                            if (ev.hasField("loss"))
                                loraLoss.add(ev.getFloat("loss"));
                            if (ev.hasField("globalGradNorm"))
                                loraGradNorm.add(ev.getFloat("globalGradNorm"));
                            if (ev.hasField("numTokens"))
                                loraTokens.add(ev.getInt("numTokens"));
                            if (ev.hasField("predictionCount"))
                                loraPredictions.add(ev.getInt("predictionCount"));
                            if (ev.hasField("clipped") && ev.getBoolean("clipped"))
                                loraClippedCount++;
                            if (ev.hasField("algorithm")) {
                                String algo = sanitizeAlgo(ev.getString("algorithm"));
                                if (!algo.isEmpty()) {
                                    loraCountByAlgo.merge(algo, 1, Integer::sum);
                                    if (ev.hasField("totalMs"))
                                        loraTotalByAlgo.computeIfAbsent(algo, k -> new ArrayList<>())
                                                .add(ev.getLong("totalMs"));
                                }
                            }
                        }
                        case LORA_VAL -> {
                            valCount++;
                            if (ev.hasField("loss")) {
                                float loss = ev.getFloat("loss");
                                valLoss.add(loss);
                                boolean best = ev.hasField("bestSoFar") && ev.getBoolean("bestSoFar");
                                if (best)
                                    valBestLoss.add(loss);
                            }
                            if (ev.hasField("durationMs"))
                                valDurationMs.add(ev.getLong("durationMs"));
                        }
                        case LORA_NORM -> {
                            normCount++;
                            if (ev.hasField("durationMs"))
                                normDurationMs.add(ev.getLong("durationMs"));
                        }
                        case LORA_MERGE -> {
                            mergeCount++;
                            if (ev.hasField("durationMs"))
                                mergeDurationMs.add(ev.getLong("durationMs"));
                            if (ev.hasField("rmse"))
                                mergeRmse.add(ev.getFloat("rmse"));
                            if (ev.hasField("deltaRetention"))
                                mergeDeltaRetention.add(ev.getFloat("deltaRetention"));
                        }
                        case LORA_PLAY -> {
                            playCount++;
                            if (ev.hasField("loadMs"))
                                playLoadMs.add(ev.getLong("loadMs"));
                        }
                        case LORA_CKPT -> checkpointCount++;
                        case TOKEN_PRODUCED -> {
                            tokenProducedCount++;
                            Instant ts = ev.getStartTime();
                            if (tokenProducedFirst == null || ts.isBefore(tokenProducedFirst))
                                tokenProducedFirst = ts;
                            if (tokenProducedLast == null || ts.isAfter(tokenProducedLast))
                                tokenProducedLast = ts;
                        }
                        default -> { /* ignore JDK and other events */ }
                    }
                }
            }
        }

        Map<String, Double> m = new LinkedHashMap<>();
        m.put("jfr.file.bytes", (double) totalFileBytes);

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
            if (legacyBackends.contains(e.getKey()))
                continue;
            m.put("juno.MatVec.backend." + e.getKey() + ".count", (double) e.getValue().size());
            m.put("juno.MatVec.backend." + e.getKey() + ".p95_ms", JfrPercentiles.p95NanosToMs(e.getValue()));
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
        m.put("juno.LoraTrainStep.total_ms.p95", JfrPercentiles.p95LongMs(loraTotalMs));
        m.put("juno.LoraTrainStep.frozen_forward_ms.p95", JfrPercentiles.p95LongMs(loraFrozenForwardMs));
        m.put("juno.LoraTrainStep.frozen_transpose_ms.p95", JfrPercentiles.p95LongMs(loraFrozenTransposeMs));
        m.put("juno.LoraTrainStep.adapter_backward_ms.p95", JfrPercentiles.p95LongMs(loraAdapterBackwardMs));
        m.put("juno.LoraTrainStep.transfer_ms.p95", JfrPercentiles.p95LongMs(loraTransferMs));
        m.put("juno.LoraTrainStep.loss.last", JfrPercentiles.lastFloat(loraLoss));
        m.put("juno.LoraTrainStep.loss.mean", JfrPercentiles.meanFloat(loraLoss));
        m.put("juno.LoraTrainStep.grad_norm.p95", JfrPercentiles.p95Float(loraGradNorm));
        m.put("juno.LoraTrainStep.clipped.fraction",
                loraStepCount == 0 ? 0.0 : (double) loraClippedCount / loraStepCount);
        m.put("juno.LoraTrainStep.tokens.total", JfrPercentiles.sumInt(loraTokens));
        m.put("juno.LoraTrainStep.predictions.total", JfrPercentiles.sumInt(loraPredictions));

        for (Map.Entry<String, Integer> e : loraCountByAlgo.entrySet()) {
            String algo = e.getKey();
            m.put("juno.LoraTrainStep.by_algorithm." + algo + ".count", (double) e.getValue());
            m.put("juno.LoraTrainStep.by_algorithm." + algo + ".total_ms.p95",
                    JfrPercentiles.p95LongMs(loraTotalByAlgo.getOrDefault(algo, List.of())));
        }

        m.put("juno.LoraValidation.count", (double) valCount);
        m.put("juno.LoraValidation.loss.last", JfrPercentiles.lastFloat(valLoss));
        m.put("juno.LoraValidation.loss.best",
                valBestLoss.isEmpty() ? JfrPercentiles.minFloat(valLoss) : JfrPercentiles.minFloat(valBestLoss));
        m.put("juno.LoraValidation.duration_ms.p95", JfrPercentiles.p95LongMs(valDurationMs));

        m.put("juno.LoraMerge.count", (double) mergeCount);
        m.put("juno.LoraMerge.duration_ms.p95", JfrPercentiles.p95LongMs(mergeDurationMs));
        m.put("juno.LoraMerge.rmse.last", JfrPercentiles.lastFloat(mergeRmse));
        m.put("juno.LoraMerge.delta_retention.last", JfrPercentiles.lastFloat(mergeDeltaRetention));

        m.put("juno.LoraNormRefresh.count", (double) normCount);
        m.put("juno.LoraNormRefresh.duration_ms.p95", JfrPercentiles.p95LongMs(normDurationMs));

        m.put("juno.LoraPlayback.count", (double) playCount);
        m.put("juno.LoraPlayback.load_ms.p95", JfrPercentiles.p95LongMs(playLoadMs));

        m.put("juno.LoraCheckpoint.count", (double) checkpointCount);

        double elapsedSeconds = 0.0;
        if (tokenProducedFirst != null && tokenProducedLast != null && tokenProducedCount > 1) {
            java.time.Duration span = java.time.Duration.between(tokenProducedFirst, tokenProducedLast);
            elapsedSeconds = span.toNanos() / 1_000_000_000.0;
        }
        double tps = (elapsedSeconds > 0) ? tokenProducedCount / elapsedSeconds : 0.0;

        m.put("juno.TokenProduced.count", (double) tokenProducedCount);
        m.put("juno.TokenProduced.elapsed_seconds", elapsedSeconds);
        m.put("juno.TokenProduced.tps", tps);

        return new MetricsSnapshot.ModelMetrics(model.getName(), model.getPath(), jfrName, m);
    }

    private static String sanitizeBackend(String backend) {
        if (backend == null) {
            return "unknown";
        }
        return backend.replace('-', '_');
    }

    /** Keep CLI vocabulary in JSON keys; strip empty / null. */
    private static String sanitizeAlgo(String algorithm) {
        if (algorithm == null) {
            return "";
        }
        String a = algorithm.strip().toLowerCase();
        return a.isEmpty() ? "" : a;
    }
}
