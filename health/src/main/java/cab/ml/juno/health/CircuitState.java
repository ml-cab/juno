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

package cab.ml.juno.health;

/**
 * Circuit breaker state machine.
 *
 * Transitions: CLOSED → OPEN when failure rate exceeds threshold OPEN →
 * HALF_OPEN after waitDuration elapses HALF_OPEN → CLOSED on probe success
 * HALF_OPEN → OPEN on probe failure
 */
public enum CircuitState {

	/** Normal operation — calls pass through. */
	CLOSED,

	/** Failing — calls are rejected immediately. */
	OPEN,

	/** Testing recovery — limited calls allowed through. */
	HALF_OPEN
}
