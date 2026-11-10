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

/**
 * The startup notice for a file whose model is trained with a sliding window that the file does not declare.
 *
 * <p>Juno applies a window only where the file declares one ({@link SlidingWindow}); it never infers one. Some
 * conversions of Phi-3-mini-4k omit {@code phi3.attention.sliding_window}, and that model was trained with a
 * 2047-token window on every layer: without it, output past 2047 tokens of context degrades (teacher-forced NLL past
 * position 2047 reads about 10.5 against 2.6 with the window). Nothing else would tell the user, so the shape
 * that model has (architecture {@code phi3}, a 4096-token context, no window key) is named at load. A file that
 * declares the key, at any width including 0, gets no notice; neither does any other context or architecture
 * (Phi-3.5-mini and the 128k models declare 131072).
 */
final class UndeclaredWindowNotice {

	private static final String ARCH = "phi3";
	private static final long CONTEXT = 4096;
	private static final int TRAINED_WINDOW = 2047;

	private UndeclaredWindowNotice() {
	}

	/** The notice for {@code r}, or {@code null} when the file does not have that shape. */
	static String forFile(GgufReader r) {
		if (!ARCH.equals(r.metaString("general.architecture")))
			return null;
		String key = ARCH + ".attention.sliding_window";
		if (r.meta(key) != null || r.metaLong(ARCH + ".context_length", -1) != CONTEXT)
			return null;
		return "This " + ARCH + " file has a " + CONTEXT + "-token context and declares no " + key
				+ ", so attention runs without a window. Phi-3-mini-4k was trained with a " + TRAINED_WINDOW
				+ "-token window: on that model, output degrades past " + TRAINED_WINDOW
				+ " tokens of context. A conversion that declares the key applies it; ./juno gguf-info shows the key.";
	}
}
