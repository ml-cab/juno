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

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Deterministic ForwardPassHandler for tests and integration testing.
 *
 * 
 * Intermediate nodes: returns a fixed-pattern float[] activation of hiddenDim
 * size. Last node: returns logits with all probability mass on a configurable
 * winner token.
 *
 * No GPU, no model weights, no Cuda. Compiles and runs anywhere.
 */
public final class CyclicForwardPassHandler implements ForwardPassHandler {

	private final int winnerToken; // last-node logit winner
	private final AtomicInteger callCount = new AtomicInteger(0);

	public CyclicForwardPassHandler() {
		this.winnerToken = 42;
	}

	public CyclicForwardPassHandler(int winnerToken) {
		this.winnerToken = winnerToken;
	}

	@Override
	public ForwardResult forward(ForwardRequest request, ShardContext context) {
		callCount.incrementAndGet();
		long start = System.nanoTime();

		ForwardPassEvent evt = new ForwardPassEvent();
		evt.begin();

		ForwardResult result;
		if (context.hasOutputProjection()) {
			// Last node — return logits
			float[] logits = new float[context.vocabSize()];
			logits[winnerToken] = 100.0f;
			result = ForwardResult.logits(request.requestId(), logits, System.nanoTime() - start);
		} else {
			// Intermediate node — return activations (deterministic pattern)
			float[] activations = new float[context.hiddenDim()];
			for (int i = 0; i < activations.length; i++) {
				activations[i] = 0.01f * (i % 100);
			}
			result = ForwardResult.activations(request.requestId(), activations, System.nanoTime() - start);
		}

		evt.handlerType = "cyclic";
		evt.requestId = request.requestId();
		evt.startPosition = request.startPosition();
		evt.layerCount = context.endLayer() - context.startLayer();
		evt.hasOutputProjection = context.hasOutputProjection();
		evt.commit();

		return result;
	}

	@Override
	public boolean isReady() {
		return true;
	}

	public int callCount() {
		return callCount.get();
	}
}