package cab.ml.juno.master;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import cab.ml.juno.player.ClusterHarness;

/**
 * A cluster started with a model whose architecture has no verified handler must
 * fail to start, naming the reason, instead of coming up with nodes that serve
 * fixed stub output. It must also not leave the forked node JVMs running.
 *
 * <p>Uses a metadata-only GGUF (zero tensors), so no model file is needed: the
 * nodes reject the architecture before reading any tensor.
 */
@DisplayName("Cluster start with an unsupported model architecture")
class UnsupportedArchitectureClusterIT {

	@TempDir
	Path dir;

	@Test
	@DisplayName("pipeline-parallel start fails with the node's reason and leaves no node JVMs running")
	void pipeline_start_fails_closed_and_cleans_up() throws Exception {
		Path gguf = writeMetadataOnlyGguf("gemma4");
		ClusterHarness harness = ClusterHarness.threeNodes(gguf.toString(), 6);
		try {
			assertThatThrownBy(harness::start).isInstanceOf(RuntimeException.class)
					.hasMessageContaining("did not load its shard")
					.hasMessageContaining("Unsupported model architecture 'gemma4'");

			long liveChildren = ProcessHandle.current().descendants().filter(ProcessHandle::isAlive).count();
			assertThat(liveChildren).as("forked node JVMs still running after a failed start").isZero();
		} finally {
			harness.stop();
		}
	}

	private Path writeMetadataOnlyGguf(String architecture) throws Exception {
		byte[] key = "general.architecture".getBytes(StandardCharsets.UTF_8);
		byte[] value = architecture.getBytes(StandardCharsets.UTF_8);
		ByteBuffer buf = ByteBuffer.allocate(24 + 8 + key.length + 4 + 8 + value.length).order(ByteOrder.LITTLE_ENDIAN);
		buf.putInt(0x46554747); // "GGUF"
		buf.putInt(3); // version
		buf.putLong(0); // tensor count
		buf.putLong(1); // metadata kv count
		buf.putLong(key.length);
		buf.put(key);
		buf.putInt(8); // string value type
		buf.putLong(value.length);
		buf.put(value);
		Path gguf = dir.resolve(architecture + ".gguf");
		Files.write(gguf, buf.array());
		return gguf;
	}
}
