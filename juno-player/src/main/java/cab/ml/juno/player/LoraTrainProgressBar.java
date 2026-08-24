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

import java.util.Locale;

/**
 * Compact LoRA training progress line driven by loss → target, not max passes.
 *
 * <p>
 * Baseline loss is the value at pass 2. Percent is how much of
 * {@code (baseline - target)} has been closed. ETA uses the loss-improvement
 * rate since that baseline. Max-iteration caps are ignored.
 */
public final class LoraTrainProgressBar {

	static final int BAR_WIDTH = 20;
	/** Pass index (1-based) whose loss locks the progress baseline. */
	static final int BASELINE_PASS = 2;
	private static final String ERASE_EOL = "\033[K";

	private LoraTrainProgressBar() {
	}

	/**
	 * Fraction of the baseline→target loss gap already closed, in {@code [0, 100]}.
	 * Returns 100 when {@code currentLoss <= targetLoss}. Returns 0 until a finite
	 * baseline exists (unless the target is already met).
	 */
	public static int percentTowardTarget(float baselineLoss, float currentLoss, float targetLoss) {
		if (Float.isFinite(currentLoss) && Float.isFinite(targetLoss) && currentLoss <= targetLoss)
			return 100;
		if (!Float.isFinite(baselineLoss) || !Float.isFinite(currentLoss) || !Float.isFinite(targetLoss))
			return 0;
		float span = baselineLoss - targetLoss;
		if (!(span > 0f))
			return 0;
		float done = baselineLoss - currentLoss;
		if (!(done > 0f))
			return 0;
		int pct = (int) Math.round(100.0 * done / span);
		if (pct < 0)
			return 0;
		if (pct > 100)
			return 100;
		return pct;
	}

	/** Filled cell count; round-half-up from {@code percent}. */
	public static int filledBars(int percent, int width) {
		if (width <= 0 || percent <= 0)
			return 0;
		if (percent >= 100)
			return width;
		return Math.min(width, (percent * width + 50) / 100);
	}

	/**
	 * ETA from loss improvement rate since baseline.
	 * {@code elapsedSinceBaselineMs} is wall time since the baseline pass.
	 */
	public static long etaMs(float baselineLoss, float currentLoss, float targetLoss, long elapsedSinceBaselineMs) {
		if (!Float.isFinite(baselineLoss) || !Float.isFinite(currentLoss) || !Float.isFinite(targetLoss))
			return 0L;
		if (currentLoss <= targetLoss || elapsedSinceBaselineMs <= 0L)
			return 0L;
		float improved = baselineLoss - currentLoss;
		float remaining = currentLoss - targetLoss;
		if (!(improved > 0f) || !(remaining > 0f))
			return 0L;
		return (long) (elapsedSinceBaselineMs * (remaining / improved));
	}

	public static String formatEta(long etaMs) {
		if (etaMs <= 0)
			return "0s";
		if (etaMs > 60_000)
			return String.format(Locale.ROOT, "%dm%02ds", etaMs / 60_000, (etaMs % 60_000) / 1000);
		return String.format(Locale.ROOT, "%ds", etaMs / 1000);
	}

	/**
	 * Progress line. {@code baselineLoss} is {@link Float#NaN} until pass
	 * {@link #BASELINE_PASS}. {@code elapsedSinceBaselineMs} is 0 until then.
	 */
	public static String render(int pass, float currentLoss, float targetLoss, float baselineLoss, long passMs,
			long elapsedSinceBaselineMs) {
		int pct = percentTowardTarget(baselineLoss, currentLoss, targetLoss);
		int bars = filledBars(pct, BAR_WIDTH);
		String bar = Color.GREEN + "▓".repeat(bars) + Color.DIM + "░".repeat(BAR_WIDTH - bars) + Color.RESET;
		long eta = etaMs(baselineLoss, currentLoss, targetLoss, elapsedSinceBaselineMs);
		return String.format(Locale.ROOT, "\r  pass %3d  loss=%8.4f  %s %3d%%  %4dms/pass  ETA %-8s%s", pass,
				currentLoss, bar, pct, Math.max(0L, passMs), formatEta(eta), ERASE_EOL);
	}
}
