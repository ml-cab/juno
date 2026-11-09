package cab.ml.juno.node;

import java.io.IOException;
import java.io.InputStream;

/**
 * Stops a forked node together with its parent, on every OS. The parent (the local cluster) holds the write end of
 * the node's standard input and closes it to stop the node; the operating system closes it as well if the parent
 * dies. At end of stream the node exits through {@link System#exit}, so its shutdown hooks run: the gRPC server
 * drains, and a JFR recording started with {@code dumponexit} is written. Destroying the process has that effect
 * on Linux (SIGTERM) but not on Windows, where it is a forced termination and no hook runs.
 */
public final class ParentPipeWatch {

	/** System property, {@code true} in a node the local cluster forked. */
	public static final String PROPERTY = "juno.node.exitWithParent";

	private ParentPipeWatch() {
	}

	/** Whether this JVM was started with {@link #PROPERTY} set to {@code true}. */
	public static boolean enabled() {
		return Boolean.getBoolean(PROPERTY);
	}

	/**
	 * Reads {@code in} to its end on a daemon thread, discarding what it reads, then runs {@code onEnd}. A read
	 * error counts as the end: the pipe is gone with the parent.
	 *
	 * @return the started thread
	 */
	public static Thread start(InputStream in, Runnable onEnd) {
		return Thread.ofPlatform().daemon().name("juno-parent-pipe").start(() -> {
			byte[] buf = new byte[256];
			try {
				while (in.read(buf) >= 0) {
					// the parent writes nothing; whatever arrives is discarded
				}
			} catch (IOException e) {
				// a broken pipe ends the watch as end of stream does
			}
			onEnd.run();
		});
	}
}
