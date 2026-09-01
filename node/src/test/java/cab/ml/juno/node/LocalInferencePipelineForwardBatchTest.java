package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import cab.ml.juno.registry.ShardAssignment;
import cab.ml.juno.registry.ShardMap;

class LocalInferencePipelineForwardBatchTest {

	private static final int VOCAB = 32000;
	private static final int HIDDEN = 4096;
	private static final int HEADS = 32;

	private ShardMap singleNodeMap() {
		return new ShardMap("tiny", 1,
				List.of(new ShardAssignment("n1", "host1", 9091, 0, 1, true, true)), Instant.now());
	}

	@Test
	void forwardBatch_routes_multi_decode_once_per_stage() {
		AtomicInteger multiDecodeCalls = new AtomicInteger(0);
		CyclicForwardPassHandler delegate = new CyclicForwardPassHandler();
		ForwardPassHandler handler = new ForwardPassHandler() {
			@Override
			public ForwardResult forward(ForwardRequest request, ShardContext context) {
				return delegate.forward(request, context);
			}

			@Override
			public boolean isReady() {
				return delegate.isReady();
			}

			@Override
			public MultiDecodeForwardResult forwardMultiDecode(MultiDecodeForwardRequest request,
					ShardContext context) {
				multiDecodeCalls.incrementAndGet();
				return delegate.forwardMultiDecode(request, context);
			}
		};

		LocalInferencePipeline pipeline = LocalInferencePipeline.from(singleNodeMap(), handler, VOCAB, HIDDEN, HEADS);

		float[][] out = pipeline.forwardBatch(
				List.of("r1", "r2", "r3"),
				List.of(new int[] { 1, 2 }, new int[] { 3, 4, 5 }, new int[] { 9 }),
				List.of(1, 2, 0));

		assertThat(out.length).isEqualTo(3);
		assertThat(multiDecodeCalls.get()).isEqualTo(1);
	}

	@Test
	void forwardBatch_single_request_delegates_to_forward() {
		CyclicForwardPassHandler handler = new CyclicForwardPassHandler(77);
		LocalInferencePipeline pipeline = LocalInferencePipeline.from(singleNodeMap(), handler, VOCAB, HIDDEN, HEADS);

		float[][] out = pipeline.forwardBatch(List.of("solo"), List.of(new int[] { 4, 5 }), List.of(1));

		assertThat(out.length).isEqualTo(1);
		assertThat(out[0][77]).isGreaterThan(0f);
		assertThat(handler.callCount()).isEqualTo(1);
	}
}
