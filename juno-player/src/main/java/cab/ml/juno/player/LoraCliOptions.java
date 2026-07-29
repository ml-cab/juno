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
import java.util.Locale;

import cab.ml.juno.lora.LoraLearningRateSchedule;
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
	public String lrSchedule = "constant";
	public int warmupSteps = 0;
	public double minLr = 0.0;
	public double weightDecay = 0.01;
	public double loraPlusRatio = 1.0;
	public float dropout = 0f;
	public long seed = 42L;
	public float validationSplit = 0f;
	public int validationPatience = 0;
	public float validationMinDelta = 0f;

	public String mode = "lora";
	public String scaling = "standard";
	public String init = "kaiming-uniform";
	public int groupWidth = 0;
	public String mergeCapability = "f32-preserve";
	public String architecture = "";
	public String trainDevice = "cpu";

	public float resolvedAlpha() {
		return alpha < 0f ? rank : alpha;
	}

	public List<LoraProjection> parsedTargets() {
		return LoraProjection.parseTargets(targets);
	}

	public LoraLearningRateSchedule.Mode parsedLrSchedule() {
		return switch (lrSchedule.strip().toLowerCase(Locale.ROOT)) {
		case "constant" -> LoraLearningRateSchedule.Mode.CONSTANT;
		case "cosine" -> LoraLearningRateSchedule.Mode.COSINE;
		default -> throw new IllegalArgumentException(
				"--lora-lr-schedule must be constant|cosine (got " + lrSchedule + ")");
		};
	}

	public cab.ml.juno.lora.LoraMode parsedMode() {
		return switch (mode.strip().toLowerCase(Locale.ROOT)) {
		case "lora" -> cab.ml.juno.lora.LoraMode.LORA;
		case "dora" -> cab.ml.juno.lora.LoraMode.DORA;
		case "qa-lora", "qalora" -> cab.ml.juno.lora.LoraMode.QA_LORA;
		default -> throw new IllegalArgumentException("--lora-mode must be lora|dora|qa-lora (got " + mode + ")");
		};
	}

	public cab.ml.juno.lora.MergeCapability parsedMergeCapability() {
		return switch (mergeCapability.strip().toLowerCase(Locale.ROOT)) {
		case "sidecar-only", "sidecar" -> cab.ml.juno.lora.MergeCapability.SIDECAR_ONLY;
		case "f32-preserve", "f32" -> cab.ml.juno.lora.MergeCapability.F32_PRESERVE;
		case "source-type-projected", "projected" -> cab.ml.juno.lora.MergeCapability.SOURCE_TYPE_PROJECTED;
		case "exact-affine" -> throw new IllegalArgumentException(
				"EXACT_AFFINE is unavailable for GGUF K-quants");
		default -> throw new IllegalArgumentException(
				"--lora-merge must be f32-preserve|source-type-projected|sidecar-only (got " + mergeCapability + ")");
		};
	}

	public cab.ml.juno.lora.LoraScaling parsedScaling() {
		return switch (scaling.strip().toLowerCase(Locale.ROOT)) {
		case "standard" -> cab.ml.juno.lora.LoraScaling.STANDARD;
		case "rslora" -> cab.ml.juno.lora.LoraScaling.RANK_STABILIZED;
		default -> throw new IllegalArgumentException(
				"--lora-scaling must be standard|rslora (got " + scaling + ")");
		};
	}

	public cab.ml.juno.lora.LoraInitialization parsedInit() {
		return switch (init.strip().toLowerCase(Locale.ROOT)) {
		case "kaiming-uniform" -> cab.ml.juno.lora.LoraInitialization.KAIMING_UNIFORM;
		case "legacy-normal" -> cab.ml.juno.lora.LoraInitialization.LEGACY_NORMAL;
		default -> throw new IllegalArgumentException(
				"--lora-init must be kaiming-uniform|legacy-normal (got " + init + ")");
		};
	}

	public LoraTrainingConfig toTrainingConfig() {
		return LoraTrainingConfig.builder().rank(rank).alpha(resolvedAlpha()).scaling(parsedScaling())
				.initialization(parsedInit()).mode(parsedMode()).learningRate(lr).targets(parsedTargets())
				.gradientAccumulationSteps(gradientAccumulation).maxGradNorm(maxGradNorm)
				.lrSchedule(parsedLrSchedule()).minLearningRate(minLr).warmupUpdates(warmupSteps)
				.weightDecay(weightDecay).loraPlusRatio(loraPlusRatio).dropout(dropout).seed(seed)
				.validationSplit(validationSplit).validationPatience(validationPatience)
				.validationMinDelta(validationMinDelta).groupWidth(groupWidth)
				.mergeCapability(parsedMergeCapability()).architecture(architecture).trainDevice(trainDevice).build();
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
		case "--lora-lr-schedule" -> {
			requireValue(args, i, flag);
			lrSchedule = args[i + 1];
			parsedLrSchedule();
			yield i + 1;
		}
		case "--lora-warmup-steps" -> {
			requireValue(args, i, flag);
			warmupSteps = Integer.parseInt(args[i + 1]);
			if (warmupSteps < 0)
				throw new IllegalArgumentException("--lora-warmup-steps must be >= 0");
			yield i + 1;
		}
		case "--lora-min-lr" -> {
			requireValue(args, i, flag);
			minLr = Double.parseDouble(args[i + 1]);
			if (!Double.isFinite(minLr) || minLr < 0)
				throw new IllegalArgumentException("--lora-min-lr must be finite and >= 0");
			yield i + 1;
		}
		case "--lora-weight-decay" -> {
			requireValue(args, i, flag);
			weightDecay = Double.parseDouble(args[i + 1]);
			if (!Double.isFinite(weightDecay) || weightDecay < 0)
				throw new IllegalArgumentException("--lora-weight-decay must be finite and >= 0");
			yield i + 1;
		}
		case "--lora-plus-ratio" -> {
			requireValue(args, i, flag);
			loraPlusRatio = Double.parseDouble(args[i + 1]);
			if (!Double.isFinite(loraPlusRatio) || loraPlusRatio <= 0)
				throw new IllegalArgumentException("--lora-plus-ratio must be finite and > 0");
			yield i + 1;
		}
		case "--lora-dropout" -> {
			requireValue(args, i, flag);
			dropout = Float.parseFloat(args[i + 1]);
			cab.ml.juno.lora.LoraDropout.validateRate(dropout);
			yield i + 1;
		}
		case "--lora-seed" -> {
			requireValue(args, i, flag);
			seed = Long.parseLong(args[i + 1]);
			yield i + 1;
		}
		case "--lora-validation-split" -> {
			requireValue(args, i, flag);
			validationSplit = Float.parseFloat(args[i + 1]);
			if (!Float.isFinite(validationSplit) || validationSplit < 0f || validationSplit >= 1f)
				throw new IllegalArgumentException("--lora-validation-split must be in [0, 1)");
			yield i + 1;
		}
		case "--lora-validation-patience" -> {
			requireValue(args, i, flag);
			validationPatience = Integer.parseInt(args[i + 1]);
			if (validationPatience < 0)
				throw new IllegalArgumentException("--lora-validation-patience must be >= 0");
			yield i + 1;
		}
		case "--lora-validation-min-delta" -> {
			requireValue(args, i, flag);
			validationMinDelta = Float.parseFloat(args[i + 1]);
			if (!Float.isFinite(validationMinDelta) || validationMinDelta < 0f)
				throw new IllegalArgumentException("--lora-validation-min-delta must be finite and >= 0");
			yield i + 1;
		}
		case "--lora-mode" -> {
			requireValue(args, i, flag);
			mode = args[i + 1];
			parsedMode();
			yield i + 1;
		}
		case "--lora-scaling" -> {
			requireValue(args, i, flag);
			scaling = args[i + 1];
			parsedScaling();
			yield i + 1;
		}
		case "--lora-init" -> {
			requireValue(args, i, flag);
			init = args[i + 1];
			parsedInit();
			yield i + 1;
		}
		case "--lora-group-width" -> {
			requireValue(args, i, flag);
			groupWidth = Integer.parseInt(args[i + 1]);
			yield i + 1;
		}
		case "--lora-merge" -> {
			requireValue(args, i, flag);
			mergeCapability = args[i + 1];
			parsedMergeCapability();
			yield i + 1;
		}
		default -> -1;
		};
	}

	public static LoraCliOptions fromEnvDefaults() {
		LoraCliOptions o = new LoraCliOptions();
		applyEnv(o, "LORA_TARGETS", v -> o.targets = v);
		applyEnv(o, "LORA_GRADIENT_ACCUMULATION", v -> o.gradientAccumulation = Integer.parseInt(v));
		applyEnv(o, "LORA_MAX_GRAD_NORM", v -> o.maxGradNorm = Float.parseFloat(v));
		applyEnv(o, "LORA_LR_SCHEDULE", v -> o.lrSchedule = v);
		applyEnv(o, "LORA_WARMUP_STEPS", v -> o.warmupSteps = Integer.parseInt(v));
		applyEnv(o, "LORA_MIN_LR", v -> o.minLr = Double.parseDouble(v));
		applyEnv(o, "LORA_WEIGHT_DECAY", v -> o.weightDecay = Double.parseDouble(v));
		applyEnv(o, "LORA_PLUS_RATIO", v -> o.loraPlusRatio = Double.parseDouble(v));
		applyEnv(o, "LORA_DROPOUT", v -> o.dropout = Float.parseFloat(v));
		applyEnv(o, "LORA_SEED", v -> o.seed = Long.parseLong(v));
		applyEnv(o, "LORA_VALIDATION_SPLIT", v -> o.validationSplit = Float.parseFloat(v));
		applyEnv(o, "LORA_VALIDATION_PATIENCE", v -> o.validationPatience = Integer.parseInt(v));
		applyEnv(o, "LORA_VALIDATION_MIN_DELTA", v -> o.validationMinDelta = Float.parseFloat(v));
		applyEnv(o, "LORA_MODE", v -> o.mode = v);
		applyEnv(o, "LORA_SCALING", v -> o.scaling = v);
		applyEnv(o, "LORA_INIT", v -> o.init = v);
		o.parsedLrSchedule();
		o.parsedMode();
		o.parsedScaling();
		o.parsedInit();
		cab.ml.juno.lora.LoraDropout.validateRate(o.dropout);
		return o;
	}

	private static void applyEnv(LoraCliOptions o, String name, java.util.function.Consumer<String> setter) {
		String v = System.getenv(name);
		if (v != null && !v.isBlank())
			setter.accept(v);
	}

	private static void requireValue(String[] args, int i, String flag) {
		if (i + 1 >= args.length)
			throw new IllegalArgumentException(flag + " requires a value");
	}
}
