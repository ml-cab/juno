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
package cab.ml.juno.kvcache;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * What a launch is told when {@code --cache-type-k q4_0} stores keys at 4 bits. Measured over a fixed 2048-token text
 * on four models, 4-bit keys raised the mean NLL to 1.03x to 2.0x an {@code f16} KV's on three of them, while 8-bit
 * keys with 4-bit values stayed within 1.01x on all four. The loss is the keys': a key enters attention through a dot
 * product and a softmax, and a {@code q4_0} block's one scale is set by its largest value. The setting stays available
 * (it is close to lossless on some models); the notice names the loss and the {@code q8_0} keys alternative. The value
 * type plays no part.
 *
 * <p>Console front ends print {@link #notice} (they turn library logging off unless verbose);
 * {@link SessionKvLayout#fromEnv} logs {@link #announce} once per process for every other entry point.
 */
public final class KvKeyPrecisionNotice {

	private static final Logger log = Logger.getLogger(KvKeyPrecisionNotice.class.getName());

	private static final AtomicBoolean ANNOUNCED = new AtomicBoolean();

	private static final String TEXT = "--cache-type-k q4_0 stores keys at 4 bits, which costs quality on some models:"
			+ " over a fixed 2048-token text the mean NLL rose to 1.03x to 2.0x the f16 KV's on three of four models"
			+ " measured. 8-bit keys with 4-bit values (--cache-type-k q8_0 --cache-type-v q4_0) stayed within 1.01x on"
			+ " all four, at 0.76x of q8_0's KV bytes";

	private KvKeyPrecisionNotice() {
	}

	/** The one-line notice for {@code types}, or {@code null} when the keys are not stored at 4 bits. */
	public static String notice(CacheTypeOptions types) {
		return types.typeK() == KvElementType.Q4_0 ? TEXT : null;
	}

	/** Logs {@link #notice}, once per process, when {@code types} stores keys at 4 bits. */
	static void announce(CacheTypeOptions types) {
		String text = notice(types);
		if (text != null && ANNOUNCED.compareAndSet(false, true))
			log.warning(text);
	}

	static void resetForTest() {
		ANNOUNCED.set(false);
	}
}
