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
 * A rotary position embedding applied in place to a device-resident activation,
 * on its chain's stream: what the decode residency region ({@link ResidentQkvPath})
 * rotates q and k with. {@link CudaRope} is the rotation from an inverse-frequency
 * table (adjacent or split-half pairs); {@link CudaPhi3Rope} is the Phi-3 family's
 * extended rotation.
 */
interface ResidentRope extends AutoCloseable {

	/**
	 * Rotates every valid row of {@code x} in place, row {@code r} at position
	 * {@code startPos + r}. Asynchronous on {@code x}'s chain.
	 *
	 * @return {@code false} (doing nothing) if the kernel failed to load
	 */
	boolean applyResident(ResidentActivation x, int startPos);

	/**
	 * Rotates columns {@code [from, from + width)} of the one valid row of {@code x}
	 * in place at {@code pos}, as whole heads. Asynchronous on {@code x}'s chain.
	 *
	 * @return {@code false} (doing nothing) if the kernel failed to load
	 */
	boolean applyResidentColumns(ResidentActivation x, int from, int width, int pos);

	/**
	 * Rotates every valid row of {@code x} in place, row {@code r} at position
	 * {@code seqLens[r] - 1}, for a decode batch whose rows are different requests at
	 * their own positions; {@code seqLens} is a device array of one int per row.
	 * Asynchronous on {@code x}'s chain.
	 *
	 * @return {@code false} (doing nothing) if the kernel failed to load
	 */
	boolean applyResidentRows(ResidentActivation x, java.lang.foreign.MemorySegment seqLens);

	/** Checks that this rotation can run at {@code pos}, where its table is finite. */
	default void requirePosition(int pos) {
	}

	int headDim();

	@Override
	void close();
}
