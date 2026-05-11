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
 * Safe CUDA runtime detection via Panama FFI.
 *
 * Delegates library availability to {@link CudaBindings#isAvailable()}.
 * If the CUDA shared libraries (libcudart.so.12, libcublas.so.12) are not on
 * LD_LIBRARY_PATH, {@link #isAvailable()} returns false without throwing.
 *
 * {@link #deviceName} and {@link #vramBytes} allocate a single off-heap
 * cudaDeviceProp struct via a confined arena, read the fields, then release
 * immediately — zero heap allocation during the read.
 *
 * Struct field offsets ({@link CudaBindings#PROP_NAME_OFFSET},
 * {@link CudaBindings#PROP_TOTAL_MEM_OFFSET}) are CUDA 12.x / Linux x86_64.
 */
public final class CudaAvailability {

    private static final Logger log = Logger.getLogger(CudaAvailability.class.getName());

    private static final boolean AVAILABLE = detect();

    private CudaAvailability() {}

    /**
     * Returns true if at least one CUDA-capable device is present and the
     * CUDA libraries were resolved successfully.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Returns the number of CUDA-capable devices, or 0 if CUDA is unavailable.
     */
    public static int deviceCount() {
        if (!AVAILABLE) return 0;
        CudaBindings cuda = CudaBindings.instance();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment countSeg = arena.allocate(JAVA_INT);
            int rc = (int) cuda.cudaGetDeviceCount.invokeExact(countSeg);
            return (rc == 0) ? countSeg.get(JAVA_INT, 0) : 0;
        } catch (Throwable t) {
            log.warning("cudaGetDeviceCount failed: " + t.getMessage());
            return 0;
        }
    }

    /**
     * Returns a human-readable name for device index, or "unknown" on failure.
     */
    public static String deviceName(int index) {
        if (!AVAILABLE) return "unavailable";
        return withDeviceProp(index, prop -> {
            String name = prop.getString(CudaBindings.PROP_NAME_OFFSET);
            return (name != null) ? name.trim() : "unknown";
        }, "unknown");
    }

    /**
     * Returns total VRAM in bytes for device index, or 0 on failure.
     */
    public static long vramBytes(int index) {
        if (!AVAILABLE) return 0L;
        return withDeviceProp(index,
            prop -> prop.get(JAVA_LONG, CudaBindings.PROP_TOTAL_MEM_OFFSET),
            0L);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    @FunctionalInterface
    private interface PropReader<T> {
        T read(MemorySegment prop) throws Throwable;
    }

    private static <T> T withDeviceProp(int index, PropReader<T> reader, T fallback) {
        CudaBindings cuda = CudaBindings.instance();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment prop = arena.allocate(CudaBindings.DEVICE_PROP_BYTES);
            int rc = (int) cuda.cudaGetDeviceProperties.invokeExact(prop, index);
            if (rc != 0) return fallback;
            return reader.read(prop);
        } catch (Throwable t) {
            log.warning("cudaGetDeviceProperties failed: " + t.getMessage());
            return fallback;
        }
    }

    private static boolean detect() {
        if (!CudaBindings.isAvailable()) {
            log.info("CUDA not available — CudaBindings did not load");
            return false;
        }
        CudaBindings cuda = CudaBindings.instance();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment countSeg = arena.allocate(JAVA_INT);
            int rc = (int) cuda.cudaGetDeviceCount.invokeExact(countSeg);
            int n  = countSeg.get(JAVA_INT, 0);
            boolean ok = (rc == 0 && n > 0);
            if (ok) {
                log.info("CUDA available — " + n + " device(s)");
            } else {
                log.info("CUDA not available (rc=" + rc + ", devices=" + n + ")");
            }
            return ok;
        } catch (Throwable t) {
            log.info("CUDA not available — " + t.getMessage());
            return false;
        }
    }
}