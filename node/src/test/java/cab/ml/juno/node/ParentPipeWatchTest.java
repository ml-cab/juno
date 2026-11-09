package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayInputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * A forked node's way to stop with its parent on every OS: the parent closes the node's standard input (or dies,
 * which closes it too), and the node exits through its shutdown hooks. Destroying the process is SIGTERM on Linux,
 * which runs those hooks, but a forced termination on Windows, which runs none.
 */
@DisplayName("ParentPipeWatch")
class ParentPipeWatchTest {

	@Test
	@DisplayName("runs the action once the stream ends, after reading whatever the parent wrote")
	void runs_the_action_at_end_of_stream() throws Exception {
		CountDownLatch ended = new CountDownLatch(1);
		Thread t = ParentPipeWatch.start(new ByteArrayInputStream(new byte[] { 1, 2, 3 }), ended::countDown);
		assertThat(ended.await(5, TimeUnit.SECONDS)).isTrue();
		assertThat(t.isDaemon()).isTrue();
	}

	@Test
	@DisplayName("does not run the action while the parent keeps the pipe open, and runs it when the pipe closes")
	void waits_while_the_pipe_is_open() throws Exception {
		PipedOutputStream parent = new PipedOutputStream();
		PipedInputStream child = new PipedInputStream(parent);
		CountDownLatch ended = new CountDownLatch(1);
		ParentPipeWatch.start(child, ended::countDown);
		parent.write(7);
		parent.flush();
		assertThat(ended.await(300, TimeUnit.MILLISECONDS)).isFalse();
		parent.close();
		assertThat(ended.await(5, TimeUnit.SECONDS)).isTrue();
	}
}
