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
}
