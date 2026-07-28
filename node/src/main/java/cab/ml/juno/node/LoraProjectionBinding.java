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
 * Maps one logical LoRA projection onto a physical GGUF tensor slice.
 *
 * <p>For LLaMA/Qwen2/Qwen3 dense layouts each binding covers the full tensor
 * ({@code rowOffset=0}, {@code rowCount=outDim}). For Phi-3, several logical
 * keys share one fused tensor at distinct row ranges.
 *
 * @param projection   logical key ({@code wq}, {@code wk}, …)
 * @param physicalName full GGUF tensor name (e.g. {@code blk.0.attn_qkv.weight})
 * @param rowOffset    first row of this logical slice inside the physical tensor
 * @param rowCount     number of rows owned by this logical projection
 * @param inDim        adapter input dimension
 * @param outDim       adapter output dimension (equals {@code rowCount})
 */
public record LoraProjectionBinding(
		LoraProjection projection,
		String physicalName,
		int rowOffset,
		int rowCount,
		int inDim,
		int outDim) {

	public LoraProjectionBinding {
		if (projection == null)
			throw new IllegalArgumentException("projection is null");
		if (physicalName == null || physicalName.isBlank())
			throw new IllegalArgumentException("physicalName is blank");
		if (rowOffset < 0 || rowCount <= 0 || inDim <= 0 || outDim <= 0)
			throw new IllegalArgumentException("invalid dims");
		if (rowCount != outDim)
			throw new IllegalArgumentException("rowCount must equal outDim");
	}

	/** Absolute layer index encoded in {@link #physicalName}. */
	public int layer() {
		// blk.L.suffix
		int dot = physicalName.indexOf('.', 4);
		if (!physicalName.startsWith("blk.") || dot < 0)
			throw new IllegalStateException("unexpected physical name: " + physicalName);
		return Integer.parseInt(physicalName.substring(4, dot));
	}
}
