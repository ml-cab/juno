package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Context shift through the vision decorator: a text-only request reaches the
 * wrapped handler; a request carrying an image fails closed, because a shift
 * would move or drop its image tokens; eviction clears the mark.
 */
@DisplayName("VisionAwareForwardPassHandler - context shift")
class VisionAwareForwardPassHandlerContextShiftTest {

	private final StubForwardPassHandler inner = new StubForwardPassHandler();
	private final VisionAwareForwardPassHandler handler = new VisionAwareForwardPassHandler(inner, 32000, 64,
			"<image>");

	@Test
	@DisplayName("text-only request: delegated to the wrapped handler")
	void textOnlyDelegates() {
		// the stub keeps the interface default, so reaching it shows as its own refusal
		assertThatThrownBy(() -> handler.shiftKv("text", 100, 4, 40)).isInstanceOf(UnsupportedOperationException.class)
				.hasMessageContaining("StubForwardPassHandler");
		assertThat(handler.contextLimit()).isEqualTo(inner.contextLimit());
	}

	@Test
	@DisplayName("request with an image: fails closed naming the image tokens, until evicted")
	void imageRequestFailsClosed() {
		handler.registerVisionEmbeddings("img", new float[][] { new float[64] });
		handler.releaseVisionEmbeddings("img");
		assertThatThrownBy(() -> handler.shiftKv("img", 100, 4, 40)).isInstanceOf(IllegalStateException.class)
				.hasMessageContaining("image tokens");
		handler.evict("img");
		assertThatThrownBy(() -> handler.shiftKv("img", 100, 4, 40))
				.isInstanceOf(UnsupportedOperationException.class);
	}
}
