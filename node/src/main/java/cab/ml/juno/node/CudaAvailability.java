/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
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

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.cuda.cudart.cudaDeviceProp;
import org.bytedeco.cuda.global.cudart;

import java.util.logging.Logger;

/**
 * Safe CUDA runtime detection.
 *
 * Uses org.bytedeco (JavaCPP) cudart. Calling CUDA directly without checking
 * availability throws when the native library or driver is absent (e.g. CPU-only
 * CI, Intel integrated graphics). This class wraps all cudart calls in a single
 * try/catch so the rest of the codebase can branch on isAvailable() without
 * defensive exception handling everywhere.
 *
 * Usage:
 *   if (CudaAvailability.isAvailable()) {
 *       handler = GpuForwardPassHandler.load(path, ctx);
 *   } else {
 *       handler = CpuForwardPassHandler.load(path, ctx);
 *   }
 */
public final class CudaAvailability {

    private static final Logger log = Logger.getLogger(CudaAvailability.class.getName());

    /** Cached result — detection runs once at class load time. */
    private static final boolean AVAILABLE = detect();

    private CudaAvailability() {}

    /**
     * Returns true if at least one CUDA-capable device is present and
     * initialised. False on any failure: missing driver, no GPU, wrong arch.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Returns the number of CUDA-capable devices, or 0 if CUDA is unavailable.
     */
    public static int deviceCount() {
        if (!AVAILABLE) return 0;
        try (IntPointer count = new IntPointer(1)) {
            int rc = cudart.cudaGetDeviceCount(count);
            return (rc == 0) ? count.get() : 0;
        } catch (Exception e) {
            return 0;
        }
    }

    /**
     * Returns a human-readable description of device {@code index},
     * or "unavailable" if CUDA is not present.
     */
    public static String deviceName(int index) {
        if (!AVAILABLE) return "unavailable";
        try (cudaDeviceProp prop = new cudaDeviceProp()) {
            int rc = cudart.cudaGetDeviceProperties(prop, index);
            if (rc != 0) return "unknown";
            String n = prop.name().getString();
            return n != null ? n.trim().replaceAll("\\0", "") : "unknown";
        } catch (Exception e) {
            return "unknown";
        }
    }

    /**
     * Returns total VRAM in bytes for device {@code index}, or 0 if unavailable.
     */
    public static long vramBytes(int index) {
        if (!AVAILABLE) return 0L;
        try (cudaDeviceProp prop = new cudaDeviceProp()) {
            int rc = cudart.cudaGetDeviceProperties(prop, index);
            return (rc == 0) ? prop.totalGlobalMem() : 0L;
        } catch (Exception e) {
            return 0L;
        }
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private static boolean detect() {
        try {
            IntPointer count = new IntPointer(1);
            int rc = cudart.cudaGetDeviceCount(count);
            int n = count.get();
            count.close();
            boolean ok = (rc == 0 && n > 0);
            if (ok) {
                log.info("CUDA available — " + n + " device(s)");
            } else {
                log.info("CUDA not available (rc=" + rc + ", devices=" + n + ")");
            }
            return ok;
        } catch (UnsatisfiedLinkError | Exception e) {
            log.info("CUDA not available — " + e.getMessage());
            return false;
        }
    }
}
