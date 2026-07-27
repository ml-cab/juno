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
package cab.ml.juno.lora;

/**
 * Algorithm identity for checkpoint / merge metadata. Distinct from QLoRA.
 */
public enum AdapterAlgorithm {

	LORA,
	DORA,
	QA_LORA;

	public static AdapterAlgorithm fromId(int id) {
		AdapterAlgorithm[] values = values();
		if (id < 0 || id >= values.length)
			throw new IllegalArgumentException("unknown AdapterAlgorithm id: " + id);
		return values[id];
	}

	public static AdapterAlgorithm fromMode(LoraMode mode) {
		return switch (mode) {
			case LORA -> LORA;
			case DORA -> DORA;
			case QA_LORA -> QA_LORA;
		};
	}
}
