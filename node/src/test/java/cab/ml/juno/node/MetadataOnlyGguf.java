package cab.ml.juno.node;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Writes a minimal GGUF file with zero tensors and only metadata: either the
 * single key {@code general.architecture}, enough for code that reads the
 * architecture before touching any tensor, or a given set of keys.
 */
final class MetadataOnlyGguf {

	private static final int GGUF_MAGIC = 0x46554747;
	private static final int TYPE_UINT32 = 4;
	private static final int TYPE_INT32 = 5;
	private static final int TYPE_BOOL = 7;
	private static final int TYPE_STRING = 8;
	private static final int TYPE_ARRAY = 9;

	private MetadataOnlyGguf() {
	}

	static Path write(Path dir, String architecture) throws IOException {
		byte[] key = "general.architecture".getBytes(StandardCharsets.UTF_8);
		byte[] value = architecture.getBytes(StandardCharsets.UTF_8);
		ByteBuffer buf = ByteBuffer.allocate(24 + 8 + key.length + 4 + 8 + value.length).order(ByteOrder.LITTLE_ENDIAN);
		buf.putInt(GGUF_MAGIC);
		buf.putInt(3); // version
		buf.putLong(0); // tensor count
		buf.putLong(1); // metadata kv count
		buf.putLong(key.length);
		buf.put(key);
		buf.putInt(TYPE_STRING);
		buf.putLong(value.length);
		buf.put(value);
		Path gguf = dir.resolve(architecture.replace('/', '_') + ".gguf");
		Files.write(gguf, buf.array());
		return gguf;
	}

	/**
	 * Writes {@code fileName} with the keys of {@code metadata} in order. Values:
	 * {@link String}, {@link Integer} (uint32, or int32 when negative),
	 * {@link Boolean}, {@code boolean[]} and {@code int[]} (an int32 array).
	 */
	static Path write(Path dir, String fileName, Map<String, Object> metadata) throws IOException {
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		out.writeBytes(buf(24).putInt(GGUF_MAGIC).putInt(3).putLong(0).putLong(metadata.size()).array());
		for (Map.Entry<String, Object> e : metadata.entrySet()) {
			out.writeBytes(string(e.getKey()));
			out.writeBytes(value(e.getValue()));
		}
		Path gguf = dir.resolve(fileName);
		Files.write(gguf, out.toByteArray());
		return gguf;
	}

	/** An ordered key map for {@link #write(Path, String, Map)}: {@code key, value, key, value, ...}. */
	static Map<String, Object> keys(Object... keyValues) {
		Map<String, Object> m = new LinkedHashMap<>();
		for (int i = 0; i < keyValues.length; i += 2)
			m.put((String) keyValues[i], keyValues[i + 1]);
		return m;
	}

	private static byte[] value(Object v) {
		return switch (v) {
		case String s -> {
			byte[] b = string(s);
			yield buf(4 + b.length).putInt(TYPE_STRING).put(b).array();
		}
		case Integer i -> buf(8).putInt(i < 0 ? TYPE_INT32 : TYPE_UINT32).putInt(i).array();
		case Boolean b -> buf(5).putInt(TYPE_BOOL).put((byte) (b ? 1 : 0)).array();
		case boolean[] a -> {
			ByteBuffer buf = buf(16 + a.length).putInt(TYPE_ARRAY).putInt(TYPE_BOOL).putLong(a.length);
			for (boolean b : a)
				buf.put((byte) (b ? 1 : 0));
			yield buf.array();
		}
		case int[] a -> {
			ByteBuffer buf = buf(16 + 4 * a.length).putInt(TYPE_ARRAY).putInt(TYPE_INT32).putLong(a.length);
			for (int x : a)
				buf.putInt(x);
			yield buf.array();
		}
		default -> throw new IllegalArgumentException("unsupported metadata value " + v.getClass());
		};
	}

	private static byte[] string(String s) {
		byte[] b = s.getBytes(StandardCharsets.UTF_8);
		return buf(8 + b.length).putLong(b.length).put(b).array();
	}

	private static ByteBuffer buf(int n) {
		return ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN);
	}
}
