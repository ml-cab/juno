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
 * Thrown by FaultTolerantPipeline when no node can serve the request.
 *
 * Two scenarios: CIRCUIT_OPEN — all node circuits were OPEN at call time (no
 * attempt made) RETRIES_EXHAUSTED — one or more nodes were tried but all threw
 * exceptions
 *
 * The REST layer converts this to HTTP 503 with a Retry-After hint. The
 * scheduler may re-queue the request or complete the future exceptionally.
 */
public final class PipelineUnavailableException extends RuntimeException {

	private static final long serialVersionUID = PipelineUnavailableException.class.getName().hashCode();

	public enum Reason {
		/**
		 * All node circuit breakers were OPEN — request rejected without attempting a
		 * forward pass.
		 */
		CIRCUIT_OPEN,
		/** At least one attempt was made but all tried nodes threw exceptions. */
		RETRIES_EXHAUSTED
	}

	private final Reason reason;
	private final int attemptsMade;

	public PipelineUnavailableException(Reason reason, int attemptsMade, String detail) {
		super(String.format("[%s] %s (attempts: %d)", reason, detail, attemptsMade));
		this.reason = reason;
		this.attemptsMade = attemptsMade;
	}

	public PipelineUnavailableException(Reason reason, int attemptsMade, String detail, Throwable cause) {
		super(String.format("[%s] %s (attempts: %d)", reason, detail, attemptsMade), cause);
		this.reason = reason;
		this.attemptsMade = attemptsMade;
	}

	public Reason reason() {
		return reason;
	}

	public int attemptsMade() {
		return attemptsMade;
	}

	/**
	 * Whether the failure is potentially transient — worth retrying at the
	 * scheduler level.
	 */
	public boolean isRetryable() {
		return reason == Reason.RETRIES_EXHAUSTED;
	}
}
