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

/**
 * The value of {@code --health-port N}: the TCP port the health sidecar started by {@code --health} listens on,
 * and the one a local cluster's nodes report to. A value that is not a port is refused at the command line rather
 * than replaced by the default.
 */
final class HealthPortOption {

	static final String FLAG = "--health-port";

	private HealthPortOption() {
	}

	static int parse(String value) {
		String v = value == null ? "" : value.strip();
		try {
			int port = Integer.parseInt(v);
			if (port >= 1 && port <= 65535)
				return port;
		} catch (NumberFormatException e) {
			// refused below
		}
		throw new IllegalArgumentException(FLAG + " must be a port number from 1 to 65535, got '" + value + "'");
	}
}
