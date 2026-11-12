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

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Locale;

import jdk.jfr.Event;
import jdk.jfr.Name;
import jdk.jfr.Recording;
import jdk.jfr.StackTrace;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

/**
 * Whether JFR's clock agrees with the operating system's on this machine. Where the CPU's timestamp counter is
 * invariant, the JVM converts JFR's counter ticks at the frequency in the processor's brand string and never
 * measures the counter, so on a CPU whose counter runs at another rate every JFR span is off by the same
 * factor, in either direction. Detection times one JFR event around a sleep against {@link System#nanoTime};
 * where the two differ by more than {@link #MAX_ERROR}, every probe JVM reads JFR from the operating-system
 * clock. A host whose counter agrees keeps the cheaper counter.
 */
final class JfrClockCheck {

	/** The largest disagreement between JFR's clock and the operating system's that keeps the CPU counter. */
	static final double MAX_ERROR = 0.01;
	/** The span timed on both clocks. */
	static final long SPAN_MILLIS = 500;

	private static final String EVENT = "juno.kit.ClockSpan";

	@Name(EVENT)
	@StackTrace(false)
	static final class ClockSpan extends Event {
	}

	private JfrClockCheck() {
	}

	/**
	 * JFR's duration of one span over {@link System#nanoTime}'s, on whichever clock this JVM's JFR reads;
	 * {@code NaN} when JFR could not record it.
	 */
	static double measure() {
		Path file = null;
		try (Recording recording = new Recording()) {
			recording.enable(ClockSpan.class).withThreshold(Duration.ZERO);
			recording.start();
			ClockSpan span = new ClockSpan();
			span.begin();
			long t0 = System.nanoTime();
			Thread.sleep(SPAN_MILLIS);
			long t1 = System.nanoTime();
			span.end();
			span.commit();
			recording.stop();
			file = Files.createTempFile("juno-kit-clock", ".jfr");
			recording.dump(file);
			for (RecordedEvent e : RecordingFile.readAllEvents(file))
				if (e.getEventType().getName().equals(EVENT))
					return e.getDuration().toNanos() / (double) (t1 - t0);
			return Double.NaN;
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			return Double.NaN;
		} catch (Exception e) {
			return Double.NaN;
		} finally {
			if (file != null)
				file.toFile().delete();
		}
	}

	/** True when JFR's clock reads more than {@link #MAX_ERROR} off; false when the check did not run. */
	static boolean disagrees(double ratio) {
		return !Double.isNaN(ratio) && Math.abs(ratio - 1) > MAX_ERROR;
	}

	/**
	 * The operating-system clock where the kernel rejected the counter (the performance harness's rule) or where
	 * JFR's clock disagrees with the operating system's.
	 */
	static boolean osClockNeeded(String clockSource, double ratio) {
		return JvmLauncher.osClockNeeded(clockSource) || disagrees(ratio);
	}

	/** The summary's line: which clock the probes' JFR reads, and why. */
	static String describe(String clockSource, double ratio) {
		if (disagrees(ratio))
			return String.format(Locale.ROOT, "JFR clock: the operating system's; the CPU counter read %.4fx it", ratio);
		if (JvmLauncher.osClockNeeded(clockSource))
			return "JFR clock: the operating system's; kernel clock source " + clockSource;
		if (Double.isNaN(ratio))
			return "JFR clock: the CPU counter, not checked";
		return String.format(Locale.ROOT, "JFR clock: the CPU counter, %.4fx the operating system's", ratio);
	}

	/** The ratio detection recorded in {@code backend}, or {@code NaN}. */
	static double ratioOf(java.util.Map<String, Object> backend) {
		return backend.get("jfrClockRatio") instanceof Number n ? n.doubleValue() : Double.NaN;
	}
}
