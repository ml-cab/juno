package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("SessionKvLayout")
class SessionKvLayoutTest {

	private String prevSched;
	private String prevPage;

	@AfterEach
	void restore() {
		restore(ServeScheduleOptions.ENV, prevSched);
		restore(KvPageSizeOptions.ENV, prevPage);
	}

	@Test
	@DisplayName("static default allocates dense layers")
	void static_dense() {
		prevSched = System.getProperty(ServeScheduleOptions.ENV);
		prevPage = System.getProperty(KvPageSizeOptions.ENV);
		System.clearProperty(ServeScheduleOptions.ENV);
		System.clearProperty(KvPageSizeOptions.ENV);

		SessionKvLayout layout = SessionKvLayout.fromEnv(8);
		assertThat(layout.usesPaged()).isFalse();
		assertThat(layout.needsAttentionScratch()).isFalse();
		SessionKvTensor[] k = layout.newKLayers(2);
		assertThat(k[0]).isInstanceOf(DenseKvTensor.class);
		assertThat(layout.policySummary()).contains("kv-page-size ignored");
	}

	@Test
	@DisplayName("continuous allocates paged layers and needs scratch")
	void continuous_paged() {
		prevSched = System.getProperty(ServeScheduleOptions.ENV);
		prevPage = System.getProperty(KvPageSizeOptions.ENV);
		System.setProperty(ServeScheduleOptions.ENV, "continuous");
		System.setProperty(KvPageSizeOptions.ENV, "4");

		SessionKvLayout layout = SessionKvLayout.fromEnv(8);
		assertThat(layout.usesPaged()).isTrue();
		assertThat(layout.needsAttentionScratch()).isTrue();
		SessionKvTensor[] k = layout.newKLayers(1);
		assertThat(k[0]).isInstanceOf(PagedKvTensor.class);
		k[0].writeToken(0, new float[8]);
		assertThat(k[0].capacityTokens()).isEqualTo(4);
		SessionKvLayout.releaseLayers(k);
	}

	private static void restore(String key, String prev) {
		if (prev == null)
			System.clearProperty(key);
		else
			System.setProperty(key, prev);
	}
}
