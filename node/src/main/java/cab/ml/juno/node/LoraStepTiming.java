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
 * Tier-4/9 wall-time subsets for one chunk gradient computation (milliseconds).
 *
 * <p>Copied onto {@link LoraTrainEvent} after accumulation. Values may be zero on
 * CPU-only runs that skip finer instrumentation.
 */
public final class LoraStepTiming {

	public long frozenForwardMs;
	public long attentionNonlinearMs;
	public long frozenTransposeBackwardMs;
	public long adapterBackwardMs;
	public long transferMs;

	public static LoraStepTiming zero() {
		return new LoraStepTiming();
	}

	public void add(LoraStepTiming o) {
		if (o == null)
			return;
		frozenForwardMs += o.frozenForwardMs;
		attentionNonlinearMs += o.attentionNonlinearMs;
		frozenTransposeBackwardMs += o.frozenTransposeBackwardMs;
		adapterBackwardMs += o.adapterBackwardMs;
		transferMs += o.transferMs;
	}

	public void clear() {
		frozenForwardMs = 0L;
		attentionNonlinearMs = 0L;
		frozenTransposeBackwardMs = 0L;
		adapterBackwardMs = 0L;
		transferMs = 0L;
	}

	/** Apply this timing onto a train-step JFR event. */
	public void apply(LoraTrainEvent event) {
		if (event == null)
			return;
		event.frozenForwardMs = frozenForwardMs;
		event.attentionNonlinearMs = attentionNonlinearMs;
		event.frozenTransposeBackwardMs = frozenTransposeBackwardMs;
		event.adapterBackwardMs = adapterBackwardMs;
		event.transferMs = transferMs;
	}
}
