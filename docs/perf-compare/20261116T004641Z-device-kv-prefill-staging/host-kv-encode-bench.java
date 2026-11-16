import cab.ml.juno.kvcache.DenseKvTensor;
import cab.ml.juno.kvcache.KvElementType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

/** Times DenseKvTensor.writeRows of a 2,048-row window (the prefill region's host KV write) per host KV type. */
public class EncodeBench {
    public static void main(String[] a) throws Exception {
        int rows = 2048;
        for (int kvDim : new int[] { 256, 1024, 3072 }) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment src = arena.allocate((long) rows * kvDim * 4);
                Random r = new Random(1);
                for (long i = 0; i < (long) rows * kvDim; i++) src.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) r.nextGaussian());
                for (KvElementType t : new KvElementType[] { KvElementType.F16, KvElementType.Q8_0 }) {
                    DenseKvTensor kv = new DenseKvTensor(t, kvDim, rows);
                    long best = Long.MAX_VALUE;
                    for (int rep = 0; rep < 15; rep++) {
                        long t0 = System.nanoTime();
                        kv.writeRows(0, rows, src, 0);
                        best = Math.min(best, System.nanoTime() - t0);
                    }
                    System.out.printf("kvDim %4d %-5s writeRows of %d rows: %7.2f ms (%.2f ns/value)%n", kvDim, t, rows, best / 1e6, (double) best / ((long) rows * kvDim));
                }
            }
        }
    }
}
