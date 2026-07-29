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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Yevhen Soldatov
 */

final class JfrPercentiles {

    private JfrPercentiles() {
    }

    /**
     * 95th percentile of durations in nanoseconds, converted to milliseconds. Empty input returns 0.
     */
    static double p95NanosToMs(List<Long> nanos) {
        if (nanos == null || nanos.isEmpty()) {
            return 0.0;
        }
        List<Long> sorted = new ArrayList<>(nanos);
        Collections.sort(sorted);
        int idx = (int) Math.ceil(0.95 * sorted.size()) - 1;
        idx = Math.max(0, Math.min(idx, sorted.size() - 1));
        return sorted.get(idx) / 1_000_000.0;
    }

    static double p95LongMs(List<Long> millis) {
        if (millis == null || millis.isEmpty()) {
            return 0.0;
        }
        List<Long> sorted = new ArrayList<>(millis);
        Collections.sort(sorted);
        int idx = (int) Math.ceil(0.95 * sorted.size()) - 1;
        idx = Math.max(0, Math.min(idx, sorted.size() - 1));
        return sorted.get(idx).doubleValue();
    }

    static double sumNanosToMs(List<Long> nanos) {
        if (nanos == null || nanos.isEmpty()) {
            return 0.0;
        }
        long sum = 0L;
        for (Long n : nanos) {
            sum += n;
        }
        return sum / 1_000_000.0;
    }

    /** Mean of float samples; empty → 0 (never NaN in metrics JSON). */
    static double meanFloat(List<Float> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        double sum = 0.0;
        int n = 0;
        for (Float v : values) {
            if (v == null || !Float.isFinite(v)) {
                continue;
            }
            sum += v;
            n++;
        }
        return n == 0 ? 0.0 : sum / n;
    }

    /** Last finite float sample; empty → 0. */
    static double lastFloat(List<Float> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        for (int i = values.size() - 1; i >= 0; i--) {
            Float v = values.get(i);
            if (v != null && Float.isFinite(v)) {
                return v.doubleValue();
            }
        }
        return 0.0;
    }

    /** Minimum finite float sample; empty → 0. */
    static double minFloat(List<Float> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        double min = Double.POSITIVE_INFINITY;
        for (Float v : values) {
            if (v == null || !Float.isFinite(v)) {
                continue;
            }
            if (v < min) {
                min = v;
            }
        }
        return Double.isInfinite(min) ? 0.0 : min;
    }

    static double p95Float(List<Float> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        List<Float> finite = new ArrayList<>();
        for (Float v : values) {
            if (v != null && Float.isFinite(v)) {
                finite.add(v);
            }
        }
        if (finite.isEmpty()) {
            return 0.0;
        }
        Collections.sort(finite);
        int idx = (int) Math.ceil(0.95 * finite.size()) - 1;
        idx = Math.max(0, Math.min(idx, finite.size() - 1));
        return finite.get(idx).doubleValue();
    }

    static double sumLong(List<Long> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        long sum = 0L;
        for (Long v : values) {
            if (v != null) {
                sum += v;
            }
        }
        return (double) sum;
    }

    static double sumInt(List<Integer> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        long sum = 0L;
        for (Integer v : values) {
            if (v != null) {
                sum += v;
            }
        }
        return (double) sum;
    }
}
