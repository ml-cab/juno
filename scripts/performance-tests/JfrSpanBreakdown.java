import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordedFrame;
import jdk.jfr.consumer.RecordedStackTrace;
import jdk.jfr.consumer.RecordedThread;
import jdk.jfr.consumer.RecordingFile;

/**
 * CPU cost breakdown of one Juno JFR recording, split into its prefill and its
 * decode span.
 *
 * <p>A span is bounded by forward-pass events: prefill is every
 * {@code juno.PrefillBatch} window plus every {@code juno.ForwardPass} with
 * {@code windowSize > 1} or {@code startPosition == 0}, decode every other
 * {@code juno.ForwardPass}, and each span runs from its first event's start to
 * its last event's end. Tokens are the windows' {@code windowSize} for prefill
 * and one per pass for decode. Per span it reports:
 * <ul>
 * <li>{@code jdk.ExecutionSample} top frames, as samples and share;</li>
 * <li>the number of distinct sampled threads (not a utilisation figure: the
 * sampler records only a few Java threads per period);</li>
 * <li>bytes allocated, from the {@code jdk.ThreadAllocationStatistics}
 * cumulative counters (the last reading at or before the span's end minus the
 * last at or before its start, per thread, so resolution is the event's
 * period), per token and per second;</li>
 * <li>{@code jdk.ObjectAllocationSample} weight by top frame (attribution, not
 * a rate);</li>
 * <li>{@code jdk.GCPhasePause} count and total, {@code jdk.ThreadPark} and
 * {@code jdk.JavaMonitorEnter} count and total, each restricted to events that
 * start inside the span.</li>
 * </ul>
 *
 * <p>Run as a single-file source program:
 * {@code java JfrSpanBreakdown.java [--top 12] a.jfr [b.jfr ...]}.
 * Prints one JSON object per recording and span.
 */
public final class JfrSpanBreakdown {

	record Pass(long start, long end, boolean prefill) {
	}

	record AllocReading(long time, long allocated) {
	}

	static final class Span {
		final String name;
		long start = Long.MAX_VALUE;
		long end = Long.MIN_VALUE;
		int passes;
		long tokens;
		long samples;
		final Map<String, Long> topFrames = new HashMap<>();
		final Map<String, Long> threadSamples = new HashMap<>();
		final Map<String, Long> allocSites = new HashMap<>();
		long allocSampleWeight;
		int gcCount;
		double gcMs;
		int parkCount;
		double parkMs;
		int monitorCount;
		double monitorMs;
		long allocatedBytes;

		Span(String name) {
			this.name = name;
		}

		boolean contains(long t) {
			return t >= start && t <= end;
		}

		double seconds() {
			return (end - start) / 1e9;
		}
	}

	public static void main(String[] args) throws Exception {
		int top = 12;
		List<Path> files = new ArrayList<>();
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--top" -> top = Integer.parseInt(args[++i]);
			default -> files.add(Path.of(args[i]));
			}
		}
		for (Path f : files)
			analyse(f, top);
	}

	private static long nanos(Instant t) {
		return t.getEpochSecond() * 1_000_000_000L + t.getNano();
	}

	private static void analyse(Path file, int top) throws Exception {
		List<RecordedEvent> events = RecordingFile.readAllEvents(file);
		Span prefill = new Span("prefill");
		Span decode = new Span("decode");
		for (RecordedEvent e : events) {
			String type = e.getEventType().getName();
			boolean batch = type.equals("juno.PrefillBatch");
			if (!batch && !type.equals("juno.ForwardPass"))
				continue;
			int window = e.hasField("windowSize") ? e.getInt("windowSize") : 1;
			boolean isPrefill = batch || window > 1
					|| (e.hasField("startPosition") && e.getInt("startPosition") == 0);
			Span s = isPrefill ? prefill : decode;
			s.start = Math.min(s.start, nanos(e.getStartTime()));
			s.end = Math.max(s.end, nanos(e.getEndTime()));
			s.passes++;
			s.tokens += window;
		}
		List<Span> spans = new ArrayList<>();
		for (Span s : List.of(prefill, decode))
			if (s.passes > 0)
				spans.add(s);

		Map<String, List<AllocReading>> alloc = new HashMap<>();
		for (RecordedEvent e : events) {
			String type = e.getEventType().getName();
			long t = nanos(e.getStartTime());
			if (type.equals("jdk.ThreadAllocationStatistics")) {
				RecordedThread th = e.getThread("thread");
				String key = th == null ? "?" : th.getJavaName() + "#" + th.getJavaThreadId();
				alloc.computeIfAbsent(key, k -> new ArrayList<>()).add(new AllocReading(t, e.getLong("allocated")));
				continue;
			}
			for (Span s : spans) {
				if (!s.contains(t))
					continue;
				switch (type) {
				case "jdk.ExecutionSample" -> {
					s.samples++;
					s.topFrames.merge(topFrame(e.getStackTrace()), 1L, Long::sum);
					RecordedThread th = e.getThread("sampledThread");
					s.threadSamples.merge(th == null ? "?" : th.getJavaName(), 1L, Long::sum);
				}
				case "jdk.ObjectAllocationSample" -> {
					long w = e.getLong("weight");
					s.allocSampleWeight += w;
					s.allocSites.merge(topFrame(e.getStackTrace()), w, Long::sum);
				}
				case "jdk.GCPhasePause" -> {
					s.gcCount++;
					s.gcMs += e.getDuration().toNanos() / 1e6;
				}
				case "jdk.ThreadPark" -> {
					s.parkCount++;
					s.parkMs += e.getDuration().toNanos() / 1e6;
				}
				case "jdk.JavaMonitorEnter" -> {
					s.monitorCount++;
					s.monitorMs += e.getDuration().toNanos() / 1e6;
				}
				default -> {
				}
				}
			}
		}
		for (Span s : spans)
			for (List<AllocReading> readings : alloc.values())
				s.allocatedBytes += Math.max(0, lastAtOrBefore(readings, s.end) - lastAtOrBefore(readings, s.start));

		for (Span s : spans) {
			Set<String> threads = new HashSet<>(s.threadSamples.keySet());
			StringBuilder sb = new StringBuilder();
			sb.append(String.format(Locale.ROOT,
					"{\"file\":\"%s\",\"span\":\"%s\",\"passes\":%d,\"tokens\":%d,\"span_s\":%.3f,"
							+ "\"ms_per_token\":%.1f,\"execution_samples\":%d,"
							+ "\"sampled_threads\":%d,\"allocated_bytes\":%d,"
							+ "\"allocated_bytes_per_token\":%.0f,\"allocated_mb_per_s\":%.1f,"
							+ "\"gc_pauses\":%d,\"gc_pause_ms\":%.2f,\"park_count\":%d,\"park_ms\":%.1f,"
							+ "\"monitor_count\":%d,\"monitor_ms\":%.1f,",
					file.getFileName(), s.name, s.passes, s.tokens, s.seconds(), s.seconds() * 1000 / s.tokens,
					s.samples, threads.size(), s.allocatedBytes, s.allocatedBytes / (double) s.tokens, s.allocatedBytes / 1e6 / s.seconds(),
					s.gcCount, s.gcMs, s.parkCount, s.parkMs, s.monitorCount, s.monitorMs));
			sb.append("\"top_methods\":").append(topJson(s.topFrames, s.samples, top)).append(',');
			sb.append("\"alloc_sample_weight\":").append(s.allocSampleWeight).append(',');
			sb.append("\"alloc_sites\":").append(topJson(s.allocSites, s.allocSampleWeight, top)).append('}');
			System.out.println(sb);
		}
	}

	private static long lastAtOrBefore(List<AllocReading> readings, long t) {
		long v = 0;
		for (AllocReading r : readings)
			if (r.time() <= t)
				v = Math.max(v, r.allocated());
		return v;
	}

	private static String topFrame(RecordedStackTrace st) {
		if (st == null || st.getFrames().isEmpty())
			return "?";
		RecordedFrame f = st.getFrames().get(0);
		return f.getMethod().getType().getName() + "." + f.getMethod().getName();
	}

	private static String topJson(Map<String, Long> m, long total, int top) {
		Map<String, Long> sorted = new LinkedHashMap<>();
		m.entrySet().stream().sorted((a, b) -> Long.compare(b.getValue(), a.getValue())).limit(top)
				.forEach(en -> sorted.put(en.getKey(), en.getValue()));
		StringBuilder sb = new StringBuilder("[");
		for (Map.Entry<String, Long> en : sorted.entrySet()) {
			if (sb.length() > 1)
				sb.append(',');
			sb.append(String.format(Locale.ROOT, "{\"frame\":\"%s\",\"value\":%d,\"share\":%.4f}", en.getKey(),
					en.getValue(), total == 0 ? 0 : en.getValue() / (double) total));
		}
		return sb.append(']').toString();
	}
}
