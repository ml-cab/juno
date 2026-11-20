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

/**
 * A session is busy: a save, restore or delete found a request of it in flight, or a chat request found the session
 * held by one of those. The REST surface answers HTTP 409; the caller retries once the other call has finished.
 */
public final class SessionBusyException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	public SessionBusyException(String message) {
		super(message);
	}
}
