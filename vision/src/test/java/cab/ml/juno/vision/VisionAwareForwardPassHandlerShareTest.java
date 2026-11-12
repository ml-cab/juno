package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * A request with an image is never the source or the target of a shared KV prefix: its image positions hold that
 * image's patch embeddings under the same placeholder token ids every image uses.
 */
@DisplayName("VisionAwareForwardPassHandler - shared KV prefix")
class VisionAwareForwardPassHandlerShareTest {

	/** Records the shares that reach it and accepts them. */
	private static final class SharingStub extends StubForwardPassHandler {
		final List<String> shares = new ArrayList<>();

		@Override
		public boolean shareKvPrefix(String fromId, String toId, int len) {
			shares.add(fromId + "->" + toId);
			return true;
		}

		@Override
		public boolean supportsKvPrefixSharing() {
			return true;
		}
	}

	private final SharingStub inner = new SharingStub();
	private final VisionAwareForwardPassHandler handler = new VisionAwareForwardPassHandler(inner, 32000, 64,
			"<image>");

	@Test
	@DisplayName("text-only requests: delegated; a request with an image on either side: refused, until evicted")
	void imageRequestsAreNeverShared() {
		assertThat(handler.supportsKvPrefixSharing()).isTrue();
		assertThat(handler.shareKvPrefix("a", "b", 20)).isTrue();

		handler.registerVisionEmbeddings("img", new float[][] { new float[64] });
		assertThat(handler.shareKvPrefix("img", "c", 20)).isFalse();
		assertThat(handler.shareKvPrefix("a", "img", 20)).isFalse();
		assertThat(inner.shares).containsExactly("a->b");

		handler.evict("img");
		assertThat(handler.shareKvPrefix("a", "img", 20)).isTrue();
	}
}
