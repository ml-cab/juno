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
import jdk.jfr.consumer.RecordedFrame;
import jdk.jfr.consumer.RecordedStackTrace;
import jdk.jfr.consumer.RecordedThread;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The JVM-side counterpart to the {@code juno.*} event accounting in
 * {@link JfrMetricsExtractor}: garbage-collection pauses, allocation volume and
 * its attribution, hot methods, and time lost to monitor contention or parking.
 *
 * <p>These are the numbers a throughput comparison needs in order to be read
 * honestly. A short measurement window that happens to contain one long
 * collection pause reports a throughput drop that looks exactly like a code
 * regression, and without a recorded pause figure there is no way to tell the
 * two apart after the fact. Allocation volume plays the same role for the
 * opposite question: a change that shifts work off a hot loop but allocates more
 * per token pays for it later, in a pause that lands outside the window that
 * measured the win.
 *
 * <p>Attribution ({@code top_sites}, {@code top_methods}) is deliberately
 * bounded to {@link #TOP_N} entries. It exists to point a reader at the next
 * thing to look at, not to be gated on, and an unbounded histogram would put one
 * key per distinct method into every published result file.
 *
 * @author Yevhen Soldatov
 */
final class JdkEventBucket {

    static final String GC_PHASE_PAUSE = "jdk.GCPhasePause";
    static final String THREAD_ALLOCATION_STATISTICS = "jdk.ThreadAllocationStatistics";
    static final String OBJECT_ALLOCATION_SAMPLE = "jdk.ObjectAllocationSample";
    static final String EXECUTION_SAMPLE = "jdk.ExecutionSample";
    static final String JAVA_MONITOR_ENTER = "jdk.JavaMonitorEnter";
    static final String THREAD_PARK = "jdk.ThreadPark";

    /** How many entries each attribution histogram keeps. */
    private static final int TOP_N = 10;

    private static final double NANOS_PER_MS = 1_000_000.0;

    private int gcPauseCount;
    private long gcPauseTotalNanos;
    private long gcPauseMaxNanos;

    /**
     * {@code jdk.ThreadAllocationStatistics#allocated} is a running total for the
     * thread, not the bytes allocated since the previous sample, so the figure a
     * run allocated is the sum over threads of each thread's largest sample.
     * Summing the events instead multiplies the answer by roughly the number of
     * samples per thread.
     */
    private final Map<Long, Long> allocatedByThread = new HashMap<>();

    private int allocationSampleCount;
    private long allocationSampleWeightBytes;
    private final Map<String, Long> allocationBytesBySite = new HashMap<>();

    private int executionSampleCount;
    private final Map<String, Long> executionSamplesByMethod = new HashMap<>();

    private int monitorEnterCount;
    private long monitorEnterTotalNanos;

    private int threadParkCount;
    private long threadParkTotalNanos;

    /**
     * Folds one JDK event into this bucket.
     *
     * @param type  the event type name, already read by the caller
     * @param nanos the event duration in nanoseconds; zero for the events that carry none
     * @return {@code true} if the event was one this bucket accounts for
     */
    boolean accept(RecordedEvent ev, String type, long nanos) {
        switch (type) {
            case GC_PHASE_PAUSE -> {
                gcPauseCount++;
                gcPauseTotalNanos += nanos;
                if (nanos > gcPauseMaxNanos)
                    gcPauseMaxNanos = nanos;
            }
            case THREAD_ALLOCATION_STATISTICS -> {
                if (!ev.hasField("allocated"))
                    return true;
                long allocated = ev.getLong("allocated");
                allocatedByThread.merge(threadKey(ev, "thread"), allocated, Math::max);
            }
            case OBJECT_ALLOCATION_SAMPLE -> {
                allocationSampleCount++;
                long weight = ev.hasField("weight") ? ev.getLong("weight") : 0L;
                allocationSampleWeightBytes += weight;
                if (weight > 0)
                    allocationBytesBySite.merge(allocationSite(ev), weight, Long::sum);
            }
            case EXECUTION_SAMPLE -> {
                executionSampleCount++;
                String method = topMethod(ev);
                if (method != null)
                    executionSamplesByMethod.merge(method, 1L, Long::sum);
            }
            case JAVA_MONITOR_ENTER -> {
                monitorEnterCount++;
                monitorEnterTotalNanos += nanos;
            }
            case THREAD_PARK -> {
                threadParkCount++;
                threadParkTotalNanos += nanos;
            }
            default -> {
                return false;
            }
        }
        return true;
    }

    /** Writes this bucket's metrics into {@code m}; every key is always written, zero or not. */
    void putInto(Map<String, Double> m) {
        m.put("jdk.GCPhasePause.count", (double) gcPauseCount);
        m.put("jdk.GCPhasePause.total_ms", gcPauseTotalNanos / NANOS_PER_MS);
        m.put("jdk.GCPhasePause.max_ms", gcPauseMaxNanos / NANOS_PER_MS);

        double allocatedBytes = 0.0;
        for (long perThread : allocatedByThread.values())
            allocatedBytes += perThread;
        m.put("jdk.ThreadAllocationStatistics.bytes_total", allocatedBytes);

        m.put("jdk.ObjectAllocationSample.count", (double) allocationSampleCount);
        m.put("jdk.ObjectAllocationSample.weight_total_bytes", (double) allocationSampleWeightBytes);
        putTopN(m, "jdk.ObjectAllocationSample.top_sites.", ".bytes", allocationBytesBySite);

        m.put("jdk.ExecutionSample.count", (double) executionSampleCount);
        putTopN(m, "jdk.ExecutionSample.top_methods.", ".samples", executionSamplesByMethod);

        m.put("jdk.JavaMonitorEnter.count", (double) monitorEnterCount);
        m.put("jdk.JavaMonitorEnter.total_ms", monitorEnterTotalNanos / NANOS_PER_MS);

        m.put("jdk.ThreadPark.count", (double) threadParkCount);
        m.put("jdk.ThreadPark.total_ms", threadParkTotalNanos / NANOS_PER_MS);
    }

    private static void putTopN(Map<String, Double> m, String prefix, String suffix, Map<String, Long> histogram) {
        if (histogram.isEmpty())
            return;
        List<Map.Entry<String, Long>> entries = new ArrayList<>(histogram.entrySet());
        // Ties broken by name so the published key set is stable across runs.
        entries.sort(Comparator.<Map.Entry<String, Long>, Long>comparing(Map.Entry::getValue).reversed()
                .thenComparing(Map.Entry::getKey));
        int limit = Math.min(TOP_N, entries.size());
        for (int i = 0; i < limit; i++) {
            Map.Entry<String, Long> e = entries.get(i);
            m.put(prefix + e.getKey() + suffix, (double) e.getValue());
        }
    }

    /**
     * The frame that allocated, falling back to the type being allocated when the
     * sample carries no stack trace.
     */
    private static String allocationSite(RecordedEvent ev) {
        String frame = topMethod(ev);
        if (frame != null)
            return frame;
        if (ev.hasField("objectClass") && ev.getClass("objectClass") != null)
            return sanitize(ev.getClass("objectClass").getName());
        return "unknown";
    }

    private static String topMethod(RecordedEvent ev) {
        RecordedStackTrace stack = ev.getStackTrace();
        if (stack == null || stack.getFrames().isEmpty())
            return null;
        RecordedFrame top = stack.getFrames().get(0);
        if (top.getMethod() == null || top.getMethod().getType() == null)
            return null;
        return sanitize(top.getMethod().getType().getName() + "." + top.getMethod().getName());
    }

    private static long threadKey(RecordedEvent ev, String field) {
        if (!ev.hasField(field))
            return -1L;
        RecordedThread thread = ev.getThread(field);
        return thread == null ? -1L : thread.getJavaThreadId();
    }

    /** Keeps every emitted key a single dotted segment, so the metric namespace stays parseable. */
    private static String sanitize(String raw) {
        if (raw == null || raw.isEmpty())
            return "unknown";
        StringBuilder sb = new StringBuilder(raw.length());
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            sb.append(Character.isLetterOrDigit(c) ? c : '_');
        }
        return sb.toString();
    }
}
