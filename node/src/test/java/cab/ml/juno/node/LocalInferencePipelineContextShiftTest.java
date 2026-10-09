package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;

/**
 * A local pipeline shifts every handler that holds a slice of the request's KV,
 * each exactly once even when one handler serves several stages, and reports the
 * lowest context limit among them.
 */
@DisplayName("LocalInferencePipeline - context shift")
class LocalInferencePipelineContextShiftTest {

	private static final class Recording implements ForwardPassHandler {
		final int limit;
		final List<String> shifts = new ArrayList<>();

		Recording(int limit) {
			this.limit = limit;
		}

		@Override
		public ForwardResult forward(ForwardRequest request, ShardContext context) {
			throw new AssertionError();
		}

		@Override
		public boolean isReady() {
			return true;
		}

		@Override
		public int contextLimit() {
			return limit;
		}

		@Override
		public void shiftKv(String requestId, int seqLen, int keep, int discard) {
			shifts.add(requestId + ":" + seqLen + ":" + keep + ":" + discard);
		}
	}

	private static ShardMap twoStages() {
		return new ShardMap("m", 4, List.of(new ShardAssignment("n1", "h", 1, 0, 2, true, false),
				new ShardAssignment("n2", "h", 2, 2, 4, false, true)), Instant.now());
	}

	@Test
	@DisplayName("two handlers: both shifted once; the limit is the lower of the two")
	void twoHandlersShiftedOnce() {
		Recording a = new Recording(4096);
		Recording b = new Recording(32768);
		LocalInferencePipeline p = LocalInferencePipeline.from(twoStages(), List.of(a, b), 10, 8, 2);
		assertThat(p.supportsContextShift()).isTrue();
		assertThat(p.contextLimit()).isEqualTo(4096);
		p.shiftKv("r", 100, 4, 48);
		assertThat(a.shifts).containsExactly("r:100:4:48");
		assertThat(b.shifts).containsExactly("r:100:4:48");
	}

	@Test
	@DisplayName("one handler serving both stages is shifted once, not once per stage")
	void sharedHandlerShiftedOnce() {
		Recording a = new Recording(32768);
		LocalInferencePipeline p = LocalInferencePipeline.from(twoStages(), a, 10, 8, 2);
		p.shiftKv("r", 100, 4, 48);
		assertThat(a.shifts).containsExactly("r:100:4:48");
	}
}
