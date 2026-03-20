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

import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;

/**
 * Enables ANSI escape-sequence rendering on Windows by activating Virtual
 * Terminal Processing (VTP) via the Windows console API, and rewires
 * System.out / System.err to raw UTF-8 byte streams.
 *
 * <p>On Linux / macOS this is a no-op.
 *
 * <p>Call {@link #enable()} as the very first statement of
 * {@code ConsoleMain.main()}, before any output is written.
 *
 * <p>Requires {@code --enable-native-access=ALL-UNNAMED} (already in scripts).
 *
 * <h3>Implementation notes</h3>
 * <ul>
 *   <li>On Windows 10+, GetConsoleMode/SetConsoleMode are forwarded stubs in
 *       kernel32.dll; the real implementation is in kernelbase.dll. We load
 *       kernelbase first and fall back to kernel32 for older Windows.</li>
 *   <li>We use {@code invoke()} rather than {@code invokeExact()} throughout.
 *       {@code invokeExact} throws WrongMethodTypeException when a non-void
 *       return value is discarded at the call site; that exception is silently
 *       swallowed and VTP is never enabled.</li>
 * </ul>
 */
public final class AnsiSupport {

    private static final int STD_OUTPUT_HANDLE = -11;
    private static final int STD_ERROR_HANDLE  = -12;

    private static final int ENABLE_PROCESSED_OUTPUT            = 0x0001;
    private static final int ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;

    private AnsiSupport() {}

    /** Call once, first line of ConsoleMain.main(), before any output. */
    public static void enable() {
        rewireStreams();
        if (!isWindows()) return;
        tryEnableVtp(STD_OUTPUT_HANDLE);
        tryEnableVtp(STD_ERROR_HANDLE);
    }

    // -------------------------------------------------------------------------

    private static void tryEnableVtp(int stdHandleId) {
        // kernelbase.dll holds the real implementation on Windows 10+.
        // kernel32.dll only has forwarder stubs that FFM cannot call through.
        for (String dll : new String[]{"kernelbase", "kernel32"}) {
            if (tryEnableVtpWith(dll, stdHandleId)) return;
        }
    }

    private static boolean tryEnableVtpWith(String dll, int stdHandleId) {
        try {
            Linker       linker = Linker.nativeLinker();
            SymbolLookup lib    = SymbolLookup.libraryLookup(dll, Arena.global());

            MethodHandle getStdHandle = linker.downcallHandle(
                    lib.find("GetStdHandle").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

            MethodHandle getConsoleMode = linker.downcallHandle(
                    lib.find("GetConsoleMode").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS));

            MethodHandle setConsoleMode = linker.downcallHandle(
                    lib.find("SetConsoleMode").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

            // NOTE: use invoke(), not invokeExact().
            // invokeExact() throws WrongMethodTypeException when a non-void
            // return value is discarded at the Java call site — that exception
            // is caught and silently swallowed, leaving VTP permanently off.
            MemorySegment handle = (MemorySegment) getStdHandle.invoke(stdHandleId);

            if (handle == null || MemorySegment.NULL.equals(handle)) return false;
            if (handle.address() == -1L) return false; // INVALID_HANDLE_VALUE

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment modePtr = arena.allocate(ValueLayout.JAVA_INT);

                int ok = (int) getConsoleMode.invoke(handle, modePtr);
                if (ok == 0) return false; // not a real console (pipe / file)

                int current = modePtr.get(ValueLayout.JAVA_INT, 0);
                int desired  = current
                        | ENABLE_PROCESSED_OUTPUT
                        | ENABLE_VIRTUAL_TERMINAL_PROCESSING;

                if (desired != current) {
                    setConsoleMode.invoke(handle, desired); // return value intentionally ignored
                }
                return true;
            }
        } catch (Throwable ignored) {
            return false;
        }
    }

    // -------------------------------------------------------------------------

    /**
     * Replace System.out/err with raw UTF-8 PrintStreams backed by
     * FileDescriptor.out/err, bypassing the platform default charset.
     */
    private static void rewireStreams() {
        try {
            System.setOut(new PrintStream(
                    new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8));
            System.setErr(new PrintStream(
                    new FileOutputStream(FileDescriptor.err), true, StandardCharsets.UTF_8));
        } catch (Throwable ignored) {}
    }

    private static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase().startsWith("win");
    }
}