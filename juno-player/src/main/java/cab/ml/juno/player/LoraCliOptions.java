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
package cab.ml.juno.player;

import java.util.List;

import cab.ml.juno.node.LoraProjection;

/**
 * Parsed LoRA CLI / environment options (testable without {@link ConsoleMain}).
 */
public final class LoraCliOptions {

	public String path;
	public String playPath;
	public int rank = 8;
	public float alpha = -1f;
	public double lr = 1e-4;
	public int maxIters = 50;
	public int maxItersQa = 50;
	public float lossTargetText = 1.8f;
	public float lossTargetQa = 1.2f;
	public float earlyStop = 0.25f;
	public String targets = "qv";
	public int gradientAccumulation = 1;
	public float maxGradNorm = 1.0f;

	public float resolvedAlpha() {
		return alpha < 0f ? rank : alpha;
	}

	public List<LoraProjection> parsedTargets() {
		return LoraProjection.parseTargets(targets);
	}

	public LoraTrainingConfig toTrainingConfig() {
		return LoraTrainingConfig.builder().rank(rank).alpha(resolvedAlpha()).learningRate(lr).targets(parsedTargets())
				.gradientAccumulationSteps(gradientAccumulation).maxGradNorm(maxGradNorm).build();
	}

	/**
	 * Apply a single {@code --lora-*} flag. Returns the next index after consuming
	 * any value argument.
	 *
	 * @return new {@code i}, or {@code -1} if {@code args[i]} is not a LoRA flag
	 */
	public int applyFlag(String[] args, int i) {
		String flag = args[i];
		return switch (flag) {
		case "--lora-path" -> {
			requireValue(args, i, flag);
			path = args[i + 1];
			yield i + 1;
		}
		case "--lora-play" -> {
			requireValue(args, i, flag);
			playPath = args[i + 1];
			yield i + 1;
		}
		case "--lora-rank" -> {
			requireValue(args, i, flag);
			rank = Integer.parseInt(args[i + 1]);
			yield i + 1;
		}
		case "--lora-alpha" -> {
			requireValue(args, i, flag);
			alpha = Float.parseFloat(args[i + 1]);
			yield i + 1;
		}
		case "--lora-lr" -> {
			requireValue(args, i, flag);
			lr = Double.parseDouble(args[i + 1]);
			yield i + 1;
		}
		case "--lora-max-iters" -> {
			requireValue(args, i, flag);
			int n = Integer.parseInt(args[i + 1]);
			maxIters = n;
			maxItersQa = n;
			yield i + 1;
		}
		case "--lora-loss-target-text" -> {
			requireValue(args, i, flag);
			lossTargetText = Float.parseFloat(args[i + 1]);
			yield i + 1;
		}
		case "--lora-loss-target-qa" -> {
			requireValue(args, i, flag);
			lossTargetQa = Float.parseFloat(args[i + 1]);
			yield i + 1;
		}
		case "--lora-steps" -> {
			requireValue(args, i, flag);
			maxIters = Integer.parseInt(args[i + 1]);
			yield i + 1;
		}
		case "--lora-steps-qa" -> {
			requireValue(args, i, flag);
			maxItersQa = Integer.parseInt(args[i + 1]);
			yield i + 1;
		}
		case "--lora-early-stop" -> {
			requireValue(args, i, flag);
			earlyStop = Float.parseFloat(args[i + 1]);
			yield i + 1;
		}
		case "--lora-targets" -> {
			requireValue(args, i, flag);
			targets = args[i + 1];
			parsedTargets(); // validate eagerly
			yield i + 1;
		}
		case "--lora-gradient-accumulation" -> {
			requireValue(args, i, flag);
			gradientAccumulation = Integer.parseInt(args[i + 1]);
			if (gradientAccumulation < 1)
				throw new IllegalArgumentException("--lora-gradient-accumulation must be >= 1");
			yield i + 1;
		}
		case "--lora-max-grad-norm" -> {
			requireValue(args, i, flag);
			maxGradNorm = Float.parseFloat(args[i + 1]);
			if (maxGradNorm < 0f)
				throw new IllegalArgumentException("--lora-max-grad-norm must be >= 0");
			yield i + 1;
		}
		default -> -1;
		};
	}

	public static LoraCliOptions fromEnvDefaults() {
		LoraCliOptions o = new LoraCliOptions();
		String t = System.getenv("LORA_TARGETS");
		if (t != null && !t.isBlank())
			o.targets = t;
		String ga = System.getenv("LORA_GRADIENT_ACCUMULATION");
		if (ga != null && !ga.isBlank())
			o.gradientAccumulation = Integer.parseInt(ga);
		String mn = System.getenv("LORA_MAX_GRAD_NORM");
		if (mn != null && !mn.isBlank())
			o.maxGradNorm = Float.parseFloat(mn);
		return o;
	}

	private static void requireValue(String[] args, int i, String flag) {
		if (i + 1 >= args.length)
			throw new IllegalArgumentException(flag + " requires a value");
	}
}
