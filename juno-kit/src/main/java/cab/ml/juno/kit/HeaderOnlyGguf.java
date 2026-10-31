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


import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A GGUF file holding nothing but {@code general.architecture}: no tensors, no tokenizer. Juno checks the
 * architecture before it reads anything else, so such a file is enough to see an unsupported architecture
 * refused, without downloading a model that has it.
 */
final class HeaderOnlyGguf {

	private static final int ALIGNMENT = 32;
	private static final int TYPE_STRING = 8;

	private HeaderOnlyGguf() {
	}

	static Path write(Path file, String architecture) throws IOException {
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		out.writeBytes("GGUF".getBytes(StandardCharsets.US_ASCII));
		out.writeBytes(le(4).putInt(3).array());
		out.writeBytes(le(8).putLong(0).array());
		out.writeBytes(le(8).putLong(1).array());
		string(out, "general.architecture");
		out.writeBytes(le(4).putInt(TYPE_STRING).array());
		string(out, architecture);
		while (out.size() % ALIGNMENT != 0)
			out.write(0);
		Files.createDirectories(file.toAbsolutePath().getParent());
		return Files.write(file, out.toByteArray());
	}

	private static void string(ByteArrayOutputStream out, String s) {
		byte[] b = s.getBytes(StandardCharsets.UTF_8);
		out.writeBytes(le(8).putLong(b.length).array());
		out.writeBytes(b);
	}

	private static ByteBuffer le(int n) {
		return ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN);
	}
}
