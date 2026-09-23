package cab.ml.juno.node;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Writes a minimal GGUF file with zero tensors and a single metadata key,
 * {@code general.architecture}. Enough for code that reads the architecture
 * before touching any tensor.
 */
final class MetadataOnlyGguf {

	private static final int GGUF_MAGIC = 0x46554747;
	private static final int TYPE_STRING = 8;

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
}
