// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.productivity;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
}
