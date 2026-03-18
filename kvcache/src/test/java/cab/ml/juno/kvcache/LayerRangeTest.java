package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class LayerRangeTest {

	@Test
	void contains_returns_true_for_layers_within_range() {
		LayerRange range = LayerRange.of(0, 8);
		assertThat(range.contains(0)).isTrue();
		assertThat(range.contains(7)).isTrue();
	}

	@Test
	void contains_returns_false_for_endLayer_itself() {
		// endLayer is exclusive
		LayerRange range = LayerRange.of(0, 8);
		assertThat(range.contains(8)).isFalse();
	}

	@Test
	void contains_returns_false_for_layers_outside_range() {
		LayerRange range = LayerRange.of(8, 16);
		assertThat(range.contains(0)).isFalse();
		assertThat(range.contains(7)).isFalse();
		assertThat(range.contains(16)).isFalse();
	}

	@Test
	void all_contains_any_non_negative_layer() {
		LayerRange all = LayerRange.all();
		assertThat(all.contains(0)).isTrue();
		assertThat(all.contains(100)).isTrue();
		assertThat(all.contains(Integer.MAX_VALUE)).isTrue();
	}

	@Test
	void layer_count_is_end_minus_start() {
		assertThat(LayerRange.of(4, 12).layerCount()).isEqualTo(8);
		assertThat(LayerRange.of(0, 1).layerCount()).isEqualTo(1);
	}

	@Test
	void rejects_start_greater_than_end() {
		assertThatThrownBy(() -> LayerRange.of(8, 4)).isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("startLayer");
	}

	@Test
	void rejects_empty_range() {
		assertThatThrownBy(() -> LayerRange.of(4, 4)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void rejects_negative_start() {
		assertThatThrownBy(() -> LayerRange.of(-1, 4)).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void is_unbounded_only_for_all_sentinel() {
		assertThat(LayerRange.all().isUnbounded()).isTrue();
		assertThat(LayerRange.of(0, 32).isUnbounded()).isFalse();
	}

	@Test
	void to_string_is_human_readable() {
		assertThat(LayerRange.of(8, 16).toString()).contains("8").contains("16");
		assertThat(LayerRange.all().toString()).containsIgnoringCase("all");
	}
}
