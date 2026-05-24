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
package cab.ml.juno.node;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.logging.Logger;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Safe HIP runtime detection via Panama FFI.
 *
 * <p>Mirrors {@link CudaAvailability} in structure; delegates library availability
 * to {@link RocmBindings#isAvailable()}. All struct-field offsets use the
 * {@code hipDeviceProp_t} layout from ROCm 7.x headers (Linux x86_64).
 */
public final class RocmAvailability {

    private static final Logger log = Logger.getLogger(RocmAvailability.class.getName());

    private static final boolean AVAILABLE = detect();

    private RocmAvailability() {}

    /**
     * Returns {@code true} if at least one HIP-capable device is present and
     * ROCm libraries loaded successfully.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Returns the number of HIP-capable devices, or 0 if ROCm is unavailable.
     */
    public static int deviceCount() {
        if (!AVAILABLE) return 0;
        RocmBindings rocm = RocmBindings.instance();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment countSeg = arena.allocate(JAVA_INT);
            int rc = (int) rocm.cudaGetDeviceCount.invokeExact(countSeg);
            return (rc == 0) ? countSeg.get(JAVA_INT, 0) : 0;
        } catch (Throwable t) {
            log.warning("hipGetDeviceCount failed: " + t.getMessage());
            return 0;
        }
    }

    /**
     * Returns a human-readable device name (from {@code hipDeviceProp_t.name}),
     * or {@code "unavailable"} on failure.
     */
    public static String deviceName(int index) {
        if (!AVAILABLE) return "unavailable";
        return withDeviceProp(index, prop -> {
            RocmBindings rocm = RocmBindings.instance();
            String name = prop.getString(rocm.PROP_NAME_OFFSET);
            return (name != null) ? name.trim() : "unknown";
        }, "unknown");
    }

    /**
     * Returns total VRAM in bytes for device {@code index}, or 0 on failure.
     */
    public static long vramBytes(int index) {
        if (!AVAILABLE) return 0L;
        return withDeviceProp(index,
            prop -> prop.get(JAVA_LONG, RocmBindings.instance().PROP_TOTAL_MEM_OFFSET),
            0L);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    @FunctionalInterface
    private interface PropReader<T> {
        T read(MemorySegment prop) throws Throwable;
    }

    private static <T> T withDeviceProp(int index, PropReader<T> reader, T fallback) {
        RocmBindings rocm = RocmBindings.instance();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment prop = arena.allocate(rocm.DEVICE_PROP_BYTES);
            int rc = (int) rocm.cudaGetDeviceProperties.invokeExact(prop, index);
            if (rc != 0) return fallback;
            return reader.read(prop);
        } catch (Throwable t) {
            log.warning("hipGetDevicePropertiesR0600 failed: " + t.getMessage());
            return fallback;
        }
    }

    private static boolean detect() {
        if (!RocmBindings.isAvailable()) {
            log.info("ROCm not available — RocmBindings did not load");
            return false;
        }
        RocmBindings rocm = RocmBindings.instance();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment countSeg = arena.allocate(JAVA_INT);
            int rc = (int) rocm.cudaGetDeviceCount.invokeExact(countSeg);
            int n  = countSeg.get(JAVA_INT, 0);
            boolean ok = (rc == 0 && n > 0);
            if (ok) {
                log.info("ROCm available — " + n + " device(s)");
            } else {
                log.info("ROCm not available (rc=" + rc + ", devices=" + n + ")");
            }
            return ok;
        } catch (Throwable t) {
            log.info("ROCm not available — " + t.getMessage());
            return false;
        }
    }
}
