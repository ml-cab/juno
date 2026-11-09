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
package cab.ml.juno.kit;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;

/**
 * Windows only: the last-write time of the registry key {@code HKCU\Console}, where console settings such as
 * {@code VirtualTerminalLevel} live, read through the FFM API from {@code advapi32} with query access only. Any
 * value written under the key moves it, a value rewritten with the data it already held included, so comparing
 * it before and after a launcher run shows whether the run wrote there; comparing the values would miss a rewrite.
 */
final class ConsoleRegistryKey {

	/** {@code HKEY_CURRENT_USER}: {@code (HKEY)(ULONG_PTR)(LONG)0x80000001}, sign-extended on x64. */
	private static final long HKEY_CURRENT_USER = 0xFFFFFFFF80000001L;
	private static final int KEY_QUERY_VALUE = 0x0001;
	private static final int ERROR_FILE_NOT_FOUND = 2;

	private ConsoleRegistryKey() {
	}

	/**
	 * The key's last-write time in FILETIME ticks (100 ns since 1601), {@code absent} when the key does not exist,
	 * or {@code unreadable: <why>}.
	 */
	static String lastWriteTime() {
		try (Arena a = Arena.ofConfined()) {
			Linker linker = Linker.nativeLinker();
			SymbolLookup advapi = SymbolLookup.libraryLookup("advapi32", a);
			MethodHandle open = linker.downcallHandle(advapi.find("RegOpenKeyExW").orElseThrow(),
					FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS));
			MethodHandle info = linker.downcallHandle(advapi.find("RegQueryInfoKeyW").orElseThrow(),
					FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
							ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
			MethodHandle close = linker.downcallHandle(advapi.find("RegCloseKey").orElseThrow(),
					FunctionDescriptor.of(JAVA_INT, ADDRESS));
			MemorySegment slot = a.allocate(ADDRESS);
			int rc = (int) open.invokeExact(MemorySegment.ofAddress(HKEY_CURRENT_USER),
					a.allocateFrom("Console", StandardCharsets.UTF_16LE), 0, KEY_QUERY_VALUE, slot);
			if (rc == ERROR_FILE_NOT_FOUND)
				return "absent";
			if (rc != 0)
				return "unreadable: RegOpenKeyExW returned " + rc;
			MemorySegment key = slot.get(ADDRESS, 0);
			try {
				MemorySegment fileTime = a.allocate(JAVA_LONG);
				MemorySegment n = MemorySegment.NULL;
				rc = (int) info.invokeExact(key, n, n, n, n, n, n, n, n, n, n, fileTime);
				if (rc != 0)
					return "unreadable: RegQueryInfoKeyW returned " + rc;
				return Long.toUnsignedString(fileTime.get(JAVA_LONG, 0));
			} finally {
				int ignored = (int) close.invokeExact(key);
			}
		} catch (Throwable t) {
			return "unreadable: " + t;
		}
	}
}
