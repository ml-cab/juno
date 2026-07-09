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
package cab.ml.juno.coordinator;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies {@link PrefillMode#parse} behavior: recognised values, case
 * insensitivity, and proper {@link IllegalArgumentException} for unknown inputs.
 */
class PrefillModeTest {

    @ParameterizedTest
    @ValueSource(strings = { "single", "SINGLE", "Single", "sInGlE" })
    void parseSingleVariants(String s) {
        assertEquals(PrefillMode.SINGLE, PrefillMode.parse(s),
                "'" + s + "' must parse to SINGLE");
    }

    @ParameterizedTest
    @ValueSource(strings = { "batched", "BATCHED", "Batched", "BATCHED" })
    void parseBatchedVariants(String s) {
        assertEquals(PrefillMode.BATCHED, PrefillMode.parse(s),
                "'" + s + "' must parse to BATCHED");
    }

    @ParameterizedTest
    @ValueSource(strings = { "batch", "stream", "", "auto", "fast", "1", "SINGLETIMEOUT" })
    void parseUnrecognizedThrows(String s) {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> PrefillMode.parse(s),
                "'" + s + "' must throw IllegalArgumentException");
        assertTrue(ex.getMessage().contains(s),
                "exception message must name the rejected value; got: " + ex.getMessage());
    }

    @Test
    void parseBatchedIsDefault() {
        // BATCHED is the documented default when the flag is absent
        assertEquals(PrefillMode.BATCHED, PrefillMode.parse("batched"));
    }

    @Test
    void enumHasTwoValues() {
        assertEquals(2, PrefillMode.values().length,
                "PrefillMode must have exactly SINGLE and BATCHED");
    }
}
