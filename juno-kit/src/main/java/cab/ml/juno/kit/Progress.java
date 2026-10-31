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
package cab.ml.juno.kit;


import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * What the tester sees while probes run: which probe, whether to leave the machine idle, and elapsed and
 * remaining time, re-estimated from the observed pace after every probe. On a terminal a one-line ticker
 * updates every second; otherwise output is plain lines.
 */
final class Progress implements RunSession.Listener {

	private final PrintStream out;
	private final List<Probe> all;
	private final double scale;
	private final boolean live;
	private final long start = System.nanoTime();
	private final List<Probe> done = new ArrayList<>();
	private final List<Duration> took = new ArrayList<>();
	private volatile Thread ticker;
	private volatile long probeStart;

	Progress(PrintStream out, List<Probe> all, double scale, boolean live) {
		this.out = out;
		this.all = List.copyOf(all);
		this.scale = scale;
		this.live = live;
	}

	@Override
	public void started(Probe probe, int index, int total) {
		probeStart = System.nanoTime();
		out.printf(Locale.ROOT, "[%d/%d] %s: %s%n      expected about %s. %s%n", index, total, probe.id(),
				probe.title(), minutes(probe.refCostSeconds() * scale), advice(probe));
		if (live) {
			ticker = Thread.ofPlatform().daemon().start(() -> {
				try {
					while (true) {
						Thread.sleep(1000);
						out.printf(Locale.ROOT, "\r      running %s, total elapsed %s ",
								clock((System.nanoTime() - probeStart) / 1e9), clock((System.nanoTime() - start) / 1e9));
						out.flush();
					}
				} catch (InterruptedException ignored) {
				}
			});
		}
	}

	@Override
	public void finished(Probe probe, ProbeOutcome outcome, int index, int total) {
		Thread t = ticker;
		if (t != null) {
			t.interrupt();
			out.print("\r");
		}
		done.add(probe);
		took.add(Duration.ofMillis(outcome.wallMillis()));
		List<Probe> left = new ArrayList<>(all);
		left.removeIf(p -> done.stream().anyMatch(d -> d.id().equals(p.id())));
		Estimator.Range r = Estimator.remaining(done, took, left, scale);
		out.printf(Locale.ROOT, "      %s in %s%s%s; elapsed %s%s%n", outcome.status(),
				clock(outcome.wallMillis() / 1000.0), outcome.attempts() > 1 ? " (" + outcome.attempts() + " attempts)" : "",
				outcome.note().isEmpty() ? "" : " - " + outcome.note(), clock((System.nanoTime() - start) / 1e9),
				left.isEmpty() ? "" : ", about " + minutes(r.low().toSeconds()) + " to " + minutes(r.high().toSeconds())
						+ " left");
	}

	static String advice(Probe probe) {
		return probe.kind() == ProbeKind.READING ? "Leave the computer idle: this is a measurement."
				: "Light use of the computer is fine: this checks results, it does not time them.";
	}

	static String minutes(double seconds) {
		return seconds < 90 ? Math.round(seconds) + " s" : Math.round(seconds / 60) + " min";
	}

	static String clock(double seconds) {
		long s = Math.round(seconds);
		return s >= 3600 ? String.format(Locale.ROOT, "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
				: String.format(Locale.ROOT, "%d:%02d", s / 60, s % 60);
	}
}
