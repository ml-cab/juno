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
package cab.ml.juno.tokenizer;

/**
 * A model file declaring a {@code tokenizer.ggml.pre} type Juno has not implemented. Distinct from
 * other argument errors so a launcher can report it as a refused file, in one line, rather than as a
 * crash; still an {@link IllegalArgumentException} for existing callers.
 */
public final class UnsupportedPreTokenizerException extends IllegalArgumentException {

	private static final long serialVersionUID = 1L;

	public UnsupportedPreTokenizerException(String message) {
		super(message);
	}
}
