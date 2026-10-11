import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CyclicBarrier;

/**
 * Multi-threaded sequential-read bandwidth of this host, as a JVM sees it.
 *
 * <p>Reads a buffer far larger than the last-level cache, split into equal
 * chunks that each thread owns statically, and sums it with four independent
 * accumulators per thread so the loop is load-bound rather than add-bound.
 * Threads are persistent and released together by a barrier; one timing covers
 * {@code passes} full reads of the buffer by every thread. Prints one JSON line
 * per thread count with every repetition and the median, in GB/s (1e9 bytes).
 *
 * <p>Run as a single-file source program (see read-bandwidth-microbench.sh):
 * {@code java -Xms5g -Xmx5g ReadBandwidth.java --gib 3 --threads 1,6,12 --reps 3 --passes 4}
 */
public final class ReadBandwidth {

	/** 96 chunks divide evenly among 1, 2, 3, 4, 6, 8, 12, 16, 24 and 32 threads. */
	private static final int CHUNKS = 96;

	private static volatile long sink;

	public static void main(String[] args) throws Exception {
		double gib = 3;
		int[] threadCounts = { 1, 2, 4, 6, 8, 12 };
		int reps = 3;
		int passes = 4;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--gib" -> gib = Double.parseDouble(args[++i]);
			case "--threads" -> threadCounts = Arrays.stream(args[++i].split(",")).mapToInt(Integer::parseInt).toArray();
			case "--reps" -> reps = Integer.parseInt(args[++i]);
			case "--passes" -> passes = Integer.parseInt(args[++i]);
			default -> throw new IllegalArgumentException("unknown argument " + args[i]);
			}
		}
		int chunkLongs = (int) Math.round(gib * (1L << 30) / 8 / CHUNKS);
		long[][] chunks = new long[CHUNKS][];
		for (int c = 0; c < CHUNKS; c++) {
			long[] a = new long[chunkLongs];
			for (int i = 0; i < a.length; i++)
				a[i] = i ^ c;
			chunks[c] = a;
		}
		long bytesPerPass = (long) CHUNKS * chunkLongs * 8;

		for (int t : threadCounts) {
			if (CHUNKS % t != 0)
				throw new IllegalArgumentException(CHUNKS + " chunks do not divide among " + t + " threads");
			measure(chunks, t, 1); // warm-up: JIT and page residency
			double[] gbs = new double[reps];
			for (int r = 0; r < reps; r++) {
				long ns = measure(chunks, t, passes);
				gbs[r] = (double) bytesPerPass * passes / ns;
			}
			double[] sorted = gbs.clone();
			Arrays.sort(sorted);
			System.out.printf(Locale.ROOT, "{\"tool\":\"java\",\"threads\":%d,\"buffer_bytes\":%d,\"passes\":%d,\"gbs\":%s,\"median_gbs\":%.2f}%n",
					t, bytesPerPass, passes, fmt(gbs), sorted[reps / 2]);
		}
	}

	/** Runs {@code passes} full reads on {@code t} threads and returns the wall time in ns. */
	private static long measure(long[][] chunks, int t, int passes) throws Exception {
		CyclicBarrier start = new CyclicBarrier(t + 1);
		CyclicBarrier end = new CyclicBarrier(t + 1);
		int per = CHUNKS / t;
		List<Thread> threads = new ArrayList<>();
		for (int w = 0; w < t; w++) {
			int first = w * per;
			Thread th = new Thread(() -> {
				try {
					start.await();
					long s = 0;
					for (int p = 0; p < passes; p++)
						for (int c = first; c < first + per; c++)
							s += sum(chunks[c]);
					sink += s;
					end.await();
				} catch (Exception e) {
					throw new RuntimeException(e);
				}
			});
			th.start();
			threads.add(th);
		}
		start.await();
		long t0 = System.nanoTime();
		end.await();
		long ns = System.nanoTime() - t0;
		for (Thread th : threads)
			th.join();
		return ns;
	}

	private static long sum(long[] a) {
		long s0 = 0, s1 = 0, s2 = 0, s3 = 0;
		int n = a.length & ~3;
		for (int i = 0; i < n; i += 4) {
			s0 += a[i];
			s1 += a[i + 1];
			s2 += a[i + 2];
			s3 += a[i + 3];
		}
		for (int i = n; i < a.length; i++)
			s0 += a[i];
		return s0 + s1 + s2 + s3;
	}

	private static String fmt(double[] v) {
		StringBuilder sb = new StringBuilder("[");
		for (int i = 0; i < v.length; i++)
			sb.append(i == 0 ? "" : ",").append(String.format(Locale.ROOT, "%.2f", v[i]));
		return sb.append(']').toString();
	}
}
