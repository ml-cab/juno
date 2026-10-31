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


import java.util.List;
import java.util.Map;

/** A probe's body; it runs inside its own JVM, started by {@link ProbeRunner} through {@link ProbeMain}. */
interface KitProbe {

	ProbeReport run(String id, Map<String, String> args) throws Exception;

	static int intArg(Map<String, String> args, String key, int fallback) {
		String v = args.get(key);
		return v == null ? fallback : Integer.parseInt(v.trim());
	}

	static long longArg(Map<String, String> args, String key, long fallback) {
		String v = args.get(key);
		return v == null ? fallback : Long.parseLong(v.trim());
	}

	/** {@code v} as the values of a one-run reading, or none when it is not a finite number. */
	static List<Double> one(double v) {
		return Double.isFinite(v) ? List.of(v) : List.of();
	}
}
