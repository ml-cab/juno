package cab.ml.juno.node;

import java.net.InetAddress;
import java.net.UnknownHostException;

/**
 * The address a node's gRPC server listens on, from {@code -Dnode.bind}. Unset, the server listens on every
 * interface, as a node of a deployed cluster must. The local cluster binds the nodes it forks to the loopback
 * address: it dials them at {@code localhost} only, and a desktop firewall then has nothing to ask the user
 * about (Windows asks the first time a program listens on a network interface).
 */
public final class NodeBindAddress {

	/** System property naming the listen address; unset or blank for every interface. */
	public static final String PROPERTY = "node.bind";

	private NodeBindAddress() {
	}

	/** This JVM's loopback address as text, the value the local cluster passes to its nodes. */
	public static String loopback() {
		return InetAddress.getLoopbackAddress().getHostAddress();
	}

	/**
	 * @param value the property's value
	 * @return the address to listen on, or {@code null} for every interface
	 * @throws IllegalArgumentException if the value does not resolve to an address
	 */
	public static InetAddress resolve(String value) {
		if (value == null || value.isBlank())
			return null;
		String v = value.strip();
		try {
			return InetAddress.getByName(v);
		} catch (UnknownHostException e) {
			throw new IllegalArgumentException("-D" + PROPERTY + "=" + v + " does not resolve to an address of this host");
		}
	}
}
