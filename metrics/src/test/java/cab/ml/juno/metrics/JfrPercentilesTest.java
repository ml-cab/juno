// Created by Yevhen Soldatov
// Initial implementation: 2026

package cab.ml.juno.metrics;

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

class JfrPercentilesTest {

    @Test
    void p95NanosEmpty() {
        Assertions.assertThat(JfrPercentiles.p95NanosToMs(List.of())).isEqualTo(0.0);
    }

    @Test
    void p95NanosSingle() {
        Assertions.assertThat(JfrPercentiles.p95NanosToMs(List.of(1_500_000L))).isEqualTo(1.5);
    }

    @Test
    void p95LongMs() {
        Assertions.assertThat(JfrPercentiles.p95LongMs(List.of(10L, 20L, 100L))).isEqualTo(100.0);
    }
}
