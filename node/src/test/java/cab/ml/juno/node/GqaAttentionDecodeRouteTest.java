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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Which attention launches take the wide decode block: rows of one (each over its own cache) at head widths up to 128,
 * FP16 and quantized alike. Prefill tiles (several rows of one cache a block) and 256-wide heads keep the 128-thread
 * tiled kernel.
 */
@DisplayName("GPU attention: which launches run the wide decode block")
class GqaAttentionDecodeRouteTest {

	@Test
	@DisplayName("rows of one at head widths up to 128 run it; prefill tiles and 256-wide heads do not")
	void wideDecodeRoute() {
		for (int headDim : new int[] { 4, 16, 64, 80, 96, 128 })
			assertThat(GqaAttentionKernel.wideDecode(1, headDim)).as("rows of one, head %d", headDim).isTrue();
		for (int headDim : new int[] { 132, 192, 256 })
			assertThat(GqaAttentionKernel.wideDecode(1, headDim)).as("rows of one, head %d", headDim).isFalse();
		for (int rowsPerBlock : new int[] { 2, 4, 8, 16, 32 })
			for (int headDim : new int[] { 64, 128, 256 })
				assertThat(GqaAttentionKernel.wideDecode(rowsPerBlock, headDim))
						.as("%d rows a block, head %d", rowsPerBlock, headDim).isFalse();
	}
}
