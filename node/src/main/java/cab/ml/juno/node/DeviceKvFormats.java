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

import java.util.Objects;
import java.util.function.BooleanSupplier;

import cab.ml.juno.kvcache.CacheTypeOptions;

/**
 * The formats of a request's device KV: one for the keys, one for the values. Chosen once per handler from
 * {@code --cache-type-k/-v} ({@link #select}); every {@link DeviceKvCache} the handler allocates carries it, and the
 * attention and KV-write kernels read it from the cache they are given.
 */
record DeviceKvFormats(DeviceKvFormat k, DeviceKvFormat v) {

	/** The FP16 device KV every launch had before the quantized formats. */
	static final DeviceKvFormats F16 = new DeviceKvFormats(DeviceKvFormat.F16, DeviceKvFormat.F16);

	DeviceKvFormats {
		Objects.requireNonNull(k, "k");
		Objects.requireNonNull(v, "v");
	}

	/**
	 * The device formats for a launch's cache types: each side's type where the device can hold it for heads of
	 * {@code headDim} values and the quantized attention kernel loads, FP16 otherwise. {@code quantizedKernelLoads}
	 * is asked only when a type is quantized, so an {@code f16} launch never loads that kernel.
	 */
	static DeviceKvFormats select(CacheTypeOptions types, int headDim, BooleanSupplier quantizedKernelLoads) {
		if (!types.usesQuantized())
			return F16;
		DeviceKvFormats wanted = new DeviceKvFormats(DeviceKvFormat.holding(types.typeK(), headDim),
				DeviceKvFormat.holding(types.typeV(), headDim));
		if (!wanted.quantized() || !quantizedKernelLoads.getAsBoolean())
			return F16;
		return wanted;
	}

	/** Either side quantized: the attention launch takes the quantized kernel. */
	boolean quantized() {
		return k.quantized() || v.quantized();
	}

	/**
	 * Whether the device rows carry the values the host KV of {@code types} carries: each side's format is that side's
	 * type ({@code f16} counts, the device holding the FP16 rounding of the host's FP32 rows as it always has). A
	 * device copy rebuilt from host rows (a shared prefix, a context shift) then equals the one the request writes.
	 */
	boolean holds(CacheTypeOptions types) {
		return k.hostType() == types.typeK() && v.hostType() == types.typeV();
	}

	/** Device bytes of one position, K and V. */
	long bytesPerPosition(int kvDim) {
		return k.rowBytes(kvDim) + v.rowBytes(kvDim);
	}

	/** Device bytes of one session of {@code tokens} positions over {@code layers} layers. */
	long sessionBytes(int layers, int kvDim, int tokens) {
		return (long) layers * tokens * bytesPerPosition(kvDim);
	}

	/** Whole sessions of {@code tokens} positions over {@code layers} layers that fit in {@code freeBytes}. */
	long sessionsThatFit(long freeBytes, int layers, int kvDim, int tokens) {
		return freeBytes <= 0 ? 0 : freeBytes / sessionBytes(layers, kvDim, tokens);
	}

	/** {@code "K q8_0, V q4_0"}. */
	String summary() {
		return "K " + k.hostType().cliName() + ", V " + v.hostType().cliName();
	}
}
