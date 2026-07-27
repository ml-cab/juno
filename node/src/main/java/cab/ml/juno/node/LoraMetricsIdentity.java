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

import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Collectors;

import cab.ml.juno.lora.LoraAdapterSet;
import cab.ml.juno.lora.LoraInitialization;
import cab.ml.juno.lora.LoraMode;
import cab.ml.juno.lora.LoraScaling;
import cab.ml.juno.lora.MergeCapability;
import cab.ml.juno.lora.QaLoraEntryMeta;

/**
 * Stable CLI-vocabulary identity tags copied onto LoRA JFR events so one
 * recording can be filtered by adapter mode without correlating external logs.
 *
 * <p>Spellings match {@code --lora-mode}, {@code --lora-scaling},
 * {@code --lora-init}, and {@code --lora-merge} (never enum names like
 * {@code QA_LORA}).
 */
public final class LoraMetricsIdentity {

	public final String algorithm;
	public final String scaling;
	public final String initialization;
	public final String architecture;
	public final String trainDevice;
	public final int rank;
	public final float alpha;
	public final float effectiveScale;
	public final String targets;
	public final int groupWidth;
	public final String mergeCapability;

	public LoraMetricsIdentity(String algorithm, String scaling, String initialization, String architecture,
			String trainDevice, int rank, float alpha, float effectiveScale, String targets, int groupWidth,
			String mergeCapability) {
		this.algorithm = Objects.requireNonNullElse(algorithm, "lora");
		this.scaling = Objects.requireNonNullElse(scaling, "standard");
		this.initialization = Objects.requireNonNullElse(initialization, "kaiming-uniform");
		this.architecture = Objects.requireNonNullElse(architecture, "");
		this.trainDevice = Objects.requireNonNullElse(trainDevice, "cpu");
		this.rank = rank;
		this.alpha = alpha;
		this.effectiveScale = effectiveScale;
		this.targets = Objects.requireNonNullElse(targets, "");
		this.groupWidth = Math.max(0, groupWidth);
		this.mergeCapability = Objects.requireNonNullElse(mergeCapability, "f32-preserve");
	}

	public static LoraMetricsIdentity of(LoraMode mode, LoraScaling scaling, LoraInitialization init,
			String architecture, String trainDevice, int rank, float alpha, List<LoraProjection> targets,
			int groupWidth, MergeCapability mergeCapability) {
		LoraScaling s = scaling != null ? scaling : LoraScaling.STANDARD;
		float effective = s.effectiveScale(alpha, Math.max(1, rank));
		return new LoraMetricsIdentity(algorithmLabel(mode), scalingLabel(s), initializationLabel(init),
				architecture != null ? architecture : "", trainDevice != null ? trainDevice : "cpu", rank, alpha,
				effective, targetsLabel(targets), groupWidth, mergeCapabilityLabel(mergeCapability));
	}

	public static String algorithmLabel(LoraMode mode) {
		if (mode == null)
			return "lora";
		return switch (mode) {
		case LORA -> "lora";
		case DORA -> "dora";
		case QA_LORA -> "qa-lora";
		};
	}

	public static String scalingLabel(LoraScaling scaling) {
		if (scaling == null)
			return "standard";
		return switch (scaling) {
		case STANDARD -> "standard";
		case RANK_STABILIZED -> "rslora";
		};
	}

	public static String initializationLabel(LoraInitialization init) {
		if (init == null)
			return "kaiming-uniform";
		return switch (init) {
		case KAIMING_UNIFORM -> "kaiming-uniform";
		case LEGACY_NORMAL -> "legacy-normal";
		};
	}

	public static String mergeCapabilityLabel(MergeCapability cap) {
		if (cap == null)
			return "f32-preserve";
		return switch (cap) {
		case SIDECAR_ONLY -> "sidecar-only";
		case F32_PRESERVE -> "f32-preserve";
		case SOURCE_TYPE_PROJECTED -> "source-type-projected";
		case EXACT_AFFINE -> "exact-affine";
		case UNSUPPORTED -> "unsupported";
		};
	}

	public static String targetsLabel(List<LoraProjection> targets) {
		if (targets == null || targets.isEmpty())
			return "";
		return targets.stream().map(LoraProjection::key).collect(Collectors.joining(","));
	}

	/** Resolve a coarse train-device label from JVM properties / flags. */
	public static String resolveTrainDevice(boolean useGpu) {
		if (!useGpu)
			return "cpu";
		String forced = System.getProperty("juno.gpu.backend", "").strip().toLowerCase(Locale.ROOT);
		if (forced.contains("cuda"))
			return "cuda";
		if (forced.contains("rocm") || forced.contains("hip"))
			return "rocm";
		return "auto";
	}

	public void apply(LoraTrainEvent e) {
		e.algorithm = algorithm;
		e.scaling = scaling;
		e.initialization = initialization;
		e.architecture = architecture;
		e.trainDevice = trainDevice;
		e.rank = rank;
		e.alpha = alpha;
		e.effectiveScale = effectiveScale;
		e.targets = targets;
		e.groupWidth = groupWidth;
	}

	public void apply(LoraValidationEvent e) {
		e.algorithm = algorithm;
		e.scaling = scaling;
		e.initialization = initialization;
		e.architecture = architecture;
		e.trainDevice = trainDevice;
		e.rank = rank;
		e.alpha = alpha;
		e.effectiveScale = effectiveScale;
		e.targets = targets;
		e.groupWidth = groupWidth;
	}

	public void apply(LoraNormRefreshEvent e) {
		e.algorithm = algorithm;
		e.scaling = scaling;
		e.initialization = initialization;
		e.architecture = architecture;
		e.trainDevice = trainDevice;
		e.rank = rank;
		e.alpha = alpha;
		e.effectiveScale = effectiveScale;
		e.targets = targets;
		e.groupWidth = groupWidth;
	}

	public void apply(LoraMergeEvent e) {
		e.algorithm = algorithm;
		e.scaling = scaling;
		e.initialization = initialization;
		e.architecture = architecture;
		e.trainDevice = trainDevice;
		e.rank = rank;
		e.alpha = alpha;
		e.effectiveScale = effectiveScale;
		e.targets = targets;
		e.groupWidth = groupWidth;
		e.mergeCapability = mergeCapability;
	}

	public void apply(LoraPlaybackEvent e) {
		e.algorithm = algorithm;
		e.scaling = scaling;
		e.initialization = initialization;
		e.architecture = architecture;
		e.trainDevice = trainDevice;
		e.rank = rank;
		e.alpha = alpha;
		e.effectiveScale = effectiveScale;
		e.targets = targets;
		e.groupWidth = groupWidth;
	}

	public void apply(LoraCheckpointEvent e) {
		e.algorithm = algorithm;
		e.scaling = scaling;
		e.initialization = initialization;
		e.architecture = architecture;
		e.trainDevice = trainDevice;
		e.rank = rank;
		e.alpha = alpha;
		e.effectiveScale = effectiveScale;
		e.targets = targets;
		e.groupWidth = groupWidth;
	}

	/**
	 * Best-effort identity from a loaded adapter set (checkpoint / merge / playback).
	 */
	public static LoraMetricsIdentity fromAdapterSet(LoraAdapterSet adapters, String architecture,
			String trainDevice) {
		if (adapters == null || adapters.size() == 0) {
			return new LoraMetricsIdentity("lora", "standard", "kaiming-uniform", architecture, trainDevice, 0, 0f, 0f,
					"", 0, "f32-preserve");
		}
		if (!adapters.asQaMap().isEmpty()) {
			var entry = adapters.asQaMap().entrySet().iterator().next();
			var qa = entry.getValue();
			MergeCapability cap = MergeCapability.F32_PRESERVE;
			QaLoraEntryMeta meta = adapters.qaMeta().get(entry.getKey());
			if (meta != null)
				cap = meta.mergeCapability();
			return new LoraMetricsIdentity(algorithmLabel(LoraMode.QA_LORA), scalingLabel(qa.scaling),
					initializationLabel(qa.initialization), architecture, trainDevice, qa.rank, qa.alpha, qa.scale,
					LoraAdapterSet.keyProj(entry.getKey()), qa.groupWidth, mergeCapabilityLabel(cap));
		}
		var entry = adapters.asMap().entrySet().iterator().next();
		var a = entry.getValue();
		return new LoraMetricsIdentity(algorithmLabel(a.mode), scalingLabel(a.scaling),
				initializationLabel(a.initialization), architecture, trainDevice, a.rank, a.alpha, a.scale,
				LoraAdapterSet.keyProj(entry.getKey()), 0, "f32-preserve");
	}
}
