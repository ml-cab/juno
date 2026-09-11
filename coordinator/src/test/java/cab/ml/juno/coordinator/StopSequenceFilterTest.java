package cab.ml.juno.coordinator;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class StopSequenceFilterTest {

	@Test
	void empty_stops_pass_through() {
		StopSequenceFilter f = new StopSequenceFilter(new String[0]);
		StopSequenceFilter.Outcome o = f.accept("hello");
		assertThat(o.stop()).isFalse();
		assertThat(o.emit()).isEqualTo("hello");
		assertThat(f.text()).isEqualTo("hello");
	}

	@Test
	void stops_when_complete_sequence_appears() {
		StopSequenceFilter f = new StopSequenceFilter(new String[] { "END" });
		assertThat(f.accept("abc").emit()).isEqualTo("abc");
		StopSequenceFilter.Outcome o = f.accept("END");
		assertThat(o.stop()).isTrue();
		assertThat(o.emit()).isEmpty();
		assertThat(f.text()).isEqualTo("abc");
	}

	@Test
	void holds_back_proper_prefix_of_stop() {
		StopSequenceFilter f = new StopSequenceFilter(new String[] { "STOP" });
		StopSequenceFilter.Outcome first = f.accept("ST");
		assertThat(first.stop()).isFalse();
		assertThat(first.emit()).isEmpty();
		StopSequenceFilter.Outcome second = f.accept("OP");
		assertThat(second.stop()).isTrue();
		assertThat(second.emit()).isEmpty();
		assertThat(f.text()).isEmpty();
	}

	@Test
	void finish_releases_held_prefix_when_not_a_stop() {
		StopSequenceFilter f = new StopSequenceFilter(new String[] { "STOP" });
		assertThat(f.accept("ST").emit()).isEmpty();
		StopSequenceFilter.Outcome done = f.finish("");
		assertThat(done.stop()).isFalse();
		assertThat(done.emit()).isEqualTo("ST");
		assertThat(f.text()).isEqualTo("ST");
	}

	@Test
	void strip_stop_from_middle_keeps_prefix_only() {
		StopSequenceFilter f = new StopSequenceFilter(new String[] { "###" });
		StopSequenceFilter.Outcome o = f.accept("hi###more");
		assertThat(o.stop()).isTrue();
		assertThat(o.emit()).isEqualTo("hi");
		assertThat(f.text()).isEqualTo("hi");
	}
}
