package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import cab.ml.juno.kvcache.KvElementType;
import cab.ml.juno.kvcache.PagedKvArena;
import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;

@DisplayName("LocalInferencePipeline - the page arenas its handlers hold")
class LocalInferencePipelineKvArenaTest {

	private static ForwardPassHandler holding(PagedKvArena arena) {
		return new ForwardPassHandler() {
			@Override
			public ForwardResult forward(ForwardRequest request, ShardContext context) {
				throw new UnsupportedOperationException();
			}

			@Override
			public boolean isReady() {
				return true;
			}

			@Override
			public PagedKvArena pagedKvArena() {
				return arena;
			}
		};
	}

	private static ShardMap nodes(int n) {
		List<ShardAssignment> a = new java.util.ArrayList<>();
		for (int i = 0; i < n; i++)
			a.add(new ShardAssignment("n" + i, "host", 9091 + i, i, i + 1, i == 0, i == n - 1));
		return new ShardMap("tiny", n, a, Instant.now());
	}

	@Test
	@DisplayName("each arena once, in stage order, whether a handler repeats or two handlers share one; none for dense handlers")
	void distinctArenas() {
		PagedKvArena a = new PagedKvArena(4, 8, KvElementType.F16, KvElementType.F16);
		PagedKvArena b = new PagedKvArena(4, 8, KvElementType.F16, KvElementType.Q8_0);
		ForwardPassHandler first = holding(a);
		LocalInferencePipeline p = LocalInferencePipeline.from(nodes(5),
				List.of(first, holding(b), first, holding(a), holding(null)), 16, 8, 2);
		assertThat(p.pagedKvArenas()).containsExactly(a, b);

		LocalInferencePipeline dense = LocalInferencePipeline.from(nodes(1), holding(null), 16, 8, 2);
		assertThat(dense.pagedKvArenas()).isEmpty();
	}
}
