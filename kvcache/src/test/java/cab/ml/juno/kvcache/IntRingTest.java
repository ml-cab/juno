package cab.ml.juno.kvcache;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayDeque;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("IntRing")
class IntRingTest {

	@Test
	@DisplayName("behaves as an ArrayDeque over a long random mix of operations, across growth and wrap-around")
	void matchesArrayDeque() {
		IntRing ring = new IntRing();
		ArrayDeque<Integer> ref = new ArrayDeque<>();
		Random rnd = new Random(7);
		for (int i = 0; i < 200_000; i++) {
			int op = rnd.nextInt(10);
			if (op < 5 || ref.isEmpty()) {
				int v = rnd.nextInt(1 << 20);
				ring.addLast(v);
				ref.addLast(v);
			} else if (op < 8) {
				assertThat(ring.pollFirst()).isEqualTo(ref.pollFirst());
			} else {
				assertThat(ring.pollLast()).isEqualTo(ref.pollLast());
			}
			assertThat(ring.size()).isEqualTo(ref.size());
		}
	}

	@Test
	@DisplayName("polling an empty ring is refused")
	void emptyIsRefused() {
		IntRing ring = new IntRing();
		assertThat(ring.isEmpty()).isTrue();
		assertThatThrownBy(ring::pollFirst).isInstanceOf(IllegalStateException.class);
		assertThatThrownBy(ring::pollLast).isInstanceOf(IllegalStateException.class);
	}
}
