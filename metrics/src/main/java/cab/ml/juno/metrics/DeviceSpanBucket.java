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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Host-device copies ({@code juno.DeviceStaging}), weight dequantizations
 * ({@code juno.WeightDequant}) and device kernels ({@code juno.DeviceCompute}): the
 * terms {@code juno.MatVec} otherwise hides inside its own span.
 *
 * <p>Both events are totals, not one event per copy: the engine counts every copy
 * and commits one event per site and phase at the end of each recording chunk, so
 * this bucket sums the events. Durations are the events' measured fields
 * ({@code transferNanos}, {@code dequantNanos}) over the copies that were timed
 * ({@code timed_count}); decode-phase copies are counted but not timed, so their
 * {@code total_ms} reads 0 with a non-zero {@code count}. Small synchronous copies
 * are timed one in sixteen, so {@code total_ms} is the measured sum and
 * {@code estimated_total_ms} scales each site's measured mean to all its copies;
 * the breakdown reads the estimate.
 *
 * <p>Staging is split by direction (H2D, D2H, D2D) and phase: {@code prefill} (the
 * issuing forward call covered more than one row), {@code decode} (one row), and
 * {@code other} (outside a forward call: weight uploads, lookup tables, cache
 * growth). As with the other window-split events, a multi-session decode step also
 * covers more than one row, so the split reads as prefill only under single-stream
 * traffic, which is how every benchmark in this repository drives the engine. Per
 * site: {@code juno.DeviceStaging.site.<site>.<phase>.*}, for attribution. The
 * {@code HOST} direction is host work that exists only to stage a copy (packing an
 * activation window to FP16); it is timed like a copy but moves nothing across the
 * bus, so it stays out of the H2D and D2H totals.
 *
 * <p>Device kernels are split the same way: {@code juno.DeviceCompute},
 * {@code juno.DeviceCompute.<phase>} and {@code juno.DeviceCompute.site.<site>.<phase>},
 * each as {@code count}/{@code timed_count}/{@code total_ms}. Every direction, phase,
 * dequant key and known compute site is written on every run, zero or not, so a
 * consumer never has to tell "absent" from "none".
 *
 * @author Yevhen Soldatov
 */
final class DeviceSpanBucket {

    static final String DEVICE_STAGING = "juno.DeviceStaging";
    static final String WEIGHT_DEQUANT = "juno.WeightDequant";
    static final String DEVICE_COMPUTE = "juno.DeviceCompute";

    private static final List<String> DIRECTIONS = List.of("H2D", "D2H", "D2D", "HOST");
    /**
     * Every kernel site the engine times today; always written, so a zero is visible. The
     * matmul and attention sites first, then the prefill-window device region's operations.
     */
    private static final List<String> COMPUTE_SITES = List.of("gemm_half", "gemm_kquant", "gemv_half_batched", "gemm_fp32",
            "mmq_packed", "gqa_attention", "rms_norm", "convert_fp16", "bias_add", "split_qkv", "rope", "kv_append",
            "gqa_attention_region", "swiglu", "residual_add");
    private static final List<String> PHASES = List.of("prefill", "decode", "other");
    private static final List<String> TIMINGS = List.of("device", "host");
    /** Every format the engine dequantizes today; always written, so a zero is visible. */
    private static final List<String> FORMATS = List.of("F32", "F16", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K",
            "Q6_K");

    private static final double NANOS_PER_MS = 1_000_000.0;

    /** Key prefix to {copies, bytes, timed copies, nanos, estimated nanos}. */
    private final Map<String, long[]> staging = new LinkedHashMap<>();
    /** Per-site key prefix to {copies, bytes, timed copies, nanos, estimated nanos}; only sites seen. */
    private final Map<String, long[]> bySite = new TreeMap<>();
    /** Key prefix to {count, timed count, nanos}. */
    private final Map<String, long[]> dequant = new LinkedHashMap<>();
    /** Key prefix to {count, timed count, nanos}: totals and per phase, then per site and phase. */
    private final Map<String, long[]> compute = new LinkedHashMap<>();
    private final Map<String, long[]> computeBySite = new TreeMap<>();

    DeviceSpanBucket() {
        for (String dir : DIRECTIONS) {
            staging.put(DEVICE_STAGING + "." + dir, new long[5]);
            for (String phase : PHASES)
                staging.put(DEVICE_STAGING + "." + dir + "." + phase, new long[5]);
        }
        dequant.put(WEIGHT_DEQUANT, new long[3]);
        for (String timing : TIMINGS)
            dequant.put(WEIGHT_DEQUANT + "." + timing, new long[3]);
        for (String format : FORMATS)
            dequant.put(WEIGHT_DEQUANT + ".format." + format, new long[3]);
        compute.put(DEVICE_COMPUTE, new long[3]);
        for (String phase : PHASES) {
            compute.put(DEVICE_COMPUTE + "." + phase, new long[3]);
            for (String site : COMPUTE_SITES)
                computeBySite.put(DEVICE_COMPUTE + ".site." + site + "." + phase, new long[3]);
        }
    }

    /**
     * Folds one event into this bucket.
     *
     * @return {@code true} if the event was one this bucket accounts for
     */
    boolean accept(RecordedEvent ev, String type) {
        switch (type) {
            case DEVICE_STAGING -> {
                String dir = text(ev, "direction", "unknown");
                String phase = text(ev, "phase", "other");
                long copies = longField(ev, "copies");
                long timed = longField(ev, "timedCopies");
                long nanos = longField(ev, "transferNanos");
                // A sampled site's duration: its measured mean over the copies it made.
                long estimated = timed > 0 ? Math.round((double) nanos * copies / timed) : 0L;
                long[] v = { copies, longField(ev, "bytes"), timed, nanos, estimated };
                add(staging.computeIfAbsent(DEVICE_STAGING + "." + dir, k -> new long[5]), v);
                add(staging.computeIfAbsent(DEVICE_STAGING + "." + dir + "." + phase, k -> new long[5]), v);
                add(bySite.computeIfAbsent(DEVICE_STAGING + ".site." + siteKey(text(ev, "site", "unknown")) + "."
                        + phase, k -> new long[5]), v);
                return true;
            }
            case WEIGHT_DEQUANT -> {
                String timing = text(ev, "timing", null);
                String format = text(ev, "format", null);
                long[] v = { longField(ev, "count"), longField(ev, "timedCount"), longField(ev, "dequantNanos") };
                add(dequant.get(WEIGHT_DEQUANT), v);
                if (timing != null)
                    add(dequant.computeIfAbsent(WEIGHT_DEQUANT + "." + timing, k -> new long[3]), v);
                if (format != null)
                    add(dequant.computeIfAbsent(WEIGHT_DEQUANT + ".format." + format, k -> new long[3]), v);
                return true;
            }
            case DEVICE_COMPUTE -> {
                String phase = text(ev, "phase", "other");
                long[] v = { longField(ev, "count"), longField(ev, "timedCount"), longField(ev, "computeNanos") };
                add(compute.get(DEVICE_COMPUTE), v);
                add(compute.computeIfAbsent(DEVICE_COMPUTE + "." + phase, k -> new long[3]), v);
                add(computeBySite.computeIfAbsent(DEVICE_COMPUTE + ".site." + siteKey(text(ev, "site", "unknown"))
                        + "." + phase, k -> new long[3]), v);
                return true;
            }
            default -> {
                return false;
            }
        }
    }

    void putInto(Map<String, Double> m) {
        putStaging(m, staging);
        putStaging(m, bySite);
        putCounted(m, dequant);
        putCounted(m, compute);
        putCounted(m, computeBySite);
    }

    private static void putCounted(Map<String, Double> m, Map<String, long[]> cells) {
        for (Map.Entry<String, long[]> e : cells.entrySet()) {
            long[] v = e.getValue();
            m.put(e.getKey() + ".count", (double) v[0]);
            m.put(e.getKey() + ".timed_count", (double) v[1]);
            m.put(e.getKey() + ".total_ms", v[2] / NANOS_PER_MS);
        }
    }

    private static void putStaging(Map<String, Double> m, Map<String, long[]> cells) {
        for (Map.Entry<String, long[]> e : cells.entrySet()) {
            long[] v = e.getValue();
            m.put(e.getKey() + ".count", (double) v[0]);
            m.put(e.getKey() + ".bytes", (double) v[1]);
            m.put(e.getKey() + ".timed_count", (double) v[2]);
            m.put(e.getKey() + ".total_ms", v[3] / NANOS_PER_MS);
            m.put(e.getKey() + ".estimated_total_ms", v[4] / NANOS_PER_MS);
        }
    }

    /** "cudaMemcpyAsync(xh H2D batched-gemm)" becomes "cudamemcpyasync_xh_h2d_batched_gemm". */
    static String siteKey(String site) {
        String k = site.toLowerCase(java.util.Locale.ROOT).replaceAll("[^a-z0-9]+", "_");
        k = k.replaceAll("^_+|_+$", "");
        return k.isEmpty() ? "unknown" : k;
    }

    private static String text(RecordedEvent ev, String field, String fallback) {
        if (!ev.hasField(field))
            return fallback;
        String s = ev.getString(field);
        return s == null || s.isEmpty() ? fallback : s;
    }

    private static long longField(RecordedEvent ev, String field) {
        return ev.hasField(field) ? ev.getLong(field) : 0L;
    }

    private static void add(long[] acc, long[] v) {
        for (int i = 0; i < acc.length; i++)
            acc[i] += v[i];
    }
}
