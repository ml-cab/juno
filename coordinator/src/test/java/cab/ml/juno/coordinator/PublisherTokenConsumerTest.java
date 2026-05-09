package cab.ml.juno.coordinator;

import static java.util.concurrent.TimeUnit.SECONDS;
import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Flow;

import org.junit.jupiter.api.Test;

class PublisherTokenConsumerTest {

	@Test
	void publishes_token_pieces_then_finishes() throws Exception {
		PublisherTokenConsumer c = new PublisherTokenConsumer();
		List<String> out = new CopyOnWriteArrayList<>();
		CountDownLatch done = new CountDownLatch(1);

		c.publisher().subscribe(new Flow.Subscriber<String>() {
			private Flow.Subscription subscription;

			@Override
			public void onSubscribe(Flow.Subscription s) {
				this.subscription = s;
				s.request(Long.MAX_VALUE);
			}

			@Override
			public void onNext(String item) {
				out.add(item);
			}

			@Override
			public void onError(Throwable throwable) {
				done.countDown();
			}

			@Override
			public void onComplete() {
				done.countDown();
			}
		});

		c.onToken("hel", 1, 0);
		c.onToken("lo", 2, 1);
		c.finish();

		assertThat(done.await(5, SECONDS)).isTrue();
		assertThat(out).containsExactly("hel", "lo");
	}
}
