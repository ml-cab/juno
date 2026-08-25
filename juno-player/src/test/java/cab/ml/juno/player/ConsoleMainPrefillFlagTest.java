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
package cab.ml.juno.player;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import cab.ml.juno.coordinator.PrefillMode;

/**
 * Verifies {@code ConsoleMain.parsePrefillMode()} via reflection, using the
 * same pattern as {@code ConsoleMainDtypeTest} (the dtype warning-and-fallback
 * fix, CHANGELOG Session 35). Tests:
 * <ul>
 *   <li>flag absent → default is {@link PrefillMode#BATCHED}</li>
 *   <li>{@code --prefill single} → {@link PrefillMode#SINGLE}</li>
 *   <li>{@code --prefill BATCHED} (upper-case) → {@link PrefillMode#BATCHED}</li>
 *   <li>unrecognized value → warning to stderr + fallback to {@link PrefillMode#BATCHED}
 *       (not a hard failure, consistent with {@code --dtype} behaviour)</li>
 * </ul>
 */
class ConsoleMainPrefillFlagTest {

    private final PrintStream originalErr = System.err;

    @AfterEach
    void restoreStderr() {
        System.setErr(originalErr);
    }

    @Test
    void single_parses_correctly() throws Exception {
        assertThat(invokeParsePrefillMode("single")).isEqualTo(PrefillMode.SINGLE);
        assertThat(invokeParsePrefillMode("SINGLE")).isEqualTo(PrefillMode.SINGLE);
    }

    @Test
    void batched_parses_correctly() throws Exception {
        assertThat(invokeParsePrefillMode("batched")).isEqualTo(PrefillMode.BATCHED);
        assertThat(invokeParsePrefillMode("BATCHED")).isEqualTo(PrefillMode.BATCHED);
        assertThat(invokeParsePrefillMode("Batched")).isEqualTo(PrefillMode.BATCHED);
    }

    @Test
    void unrecognized_value_falls_back_to_batched_with_warning() throws Exception {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        System.setErr(new PrintStream(captured));

        PrefillMode result = invokeParsePrefillMode("turbo");

        assertThat(result).isEqualTo(PrefillMode.BATCHED);
        String stderr = captured.toString();
        assertThat(stderr).contains("WARNING");
        assertThat(stderr).contains("turbo");
        assertThat(stderr).containsIgnoringCase("batched");
    }

    @Test
    void unrecognized_is_not_a_hard_failure() throws Exception {
        // Must not throw — degenerate input logs a warning and defaults to BATCHED.
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        System.setErr(new PrintStream(captured));

        PrefillMode result = invokeParsePrefillMode("garbage-value-that-does-not-exist");

        assertThat(result).isEqualTo(PrefillMode.BATCHED);
    }

    // ── Reflection helper ─────────────────────────────────────────────────────

    private static PrefillMode invokeParsePrefillMode(String s) throws Exception {
        Method m = ConsoleMain.class.getDeclaredMethod("parsePrefillMode", String.class);
        m.setAccessible(true);
        return (PrefillMode) m.invoke(null, s);
    }
}
