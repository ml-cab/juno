/*
 * Copyright 2026 Dmytro Soloviov (soulaway)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cab.ml.juno.node;

import jdk.jfr.Category;
import jdk.jfr.DataAmount;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Period;
import jdk.jfr.StackTrace;
import jdk.jfr.Timespan;

/**
 * JFR event: host-device copies from one site in one phase, totalled over a
 * recording chunk.
 *
 * <p>Periodic ({@code endChunk}): {@link DeviceSpanTally} counts every copy and
 * commits one event per site and phase when the chunk ends, so a recording holds
 * the totals of the work done while it ran. A per-copy event would itself cost a
 * visible share of a prefill window, which issues tens of thousands of copies.
 *
 * <p>{@link #transferNanos} is the sum over the {@link #timedCopies} that were
 * timed. An asynchronous copy is timed on the device between two stream events
 * ({@link DeviceSpanTimer}); a synchronous one on the host ({@link DeviceStaging#copy}),
 * after the default stream has drained if it reads a kernel's result back. Copies in
 * the decode phase are counted but not timed: timing several hundred copies per
 * token is what made a per-copy design expensive, and the prefill breakdown is
 * what the figure exists for. Small synchronous copies are timed one in sixteen
 * ({@link DeviceStaging#SAMPLE_EVERY}); a site's duration is then estimated as its
 * mean over {@link #timedCopies} times {@link #copies}.
 */
@Name("juno.DeviceStaging")
@Label("Device Staging")
@Description("Host-device copies of one site in one phase, totalled over the recording chunk")
@Category({ "Juno", "GPU" })
@StackTrace(false)
@Period("endChunk")
public final class DeviceStagingEvent extends Event {

	static final String H2D = "H2D";
	static final String D2H = "D2H";
	static final String D2D = "D2D";
	/** Host work that exists only to stage a copy (packing an activation window to FP16); crosses no bus. */
	static final String HOST = "HOST";
	/** The memcpy-kind code {@link DeviceSpanTally#staging} takes for {@link #HOST} work. */
	static final int HOST_WORK = -1;
	static final String TIMING_DEVICE = "device";
	static final String TIMING_HOST = "host";

	@Label("Site")
	@Description("The copy site, as named in the engine's error messages")
	public String site;

	@Label("Direction")
	@Description("H2D (host to device), D2H (device to host), D2D, or HOST (host work done only to stage a copy)")
	public String direction;

	@Label("Phase")
	@Description("prefill (issued by a forward call over more than one row), decode (one row), "
			+ "or other (outside a forward call: weights, tables, cache growth)")
	public String phase;

	@Label("Copies")
	public long copies;

	@Label("Bytes")
	@DataAmount
	public long bytes;

	@Label("Timed Copies")
	@Description("Copies whose transfer time is included in Transfer Time")
	public long timedCopies;

	@Label("Transfer Time")
	@Description("Summed measured transfer time of the timed copies")
	@Timespan(Timespan.NANOSECONDS)
	public long transferNanos;

	/** Direction label for a {@link GpuBindings} memcpy kind. */
	static String direction(int kind) {
		return switch (kind) {
		case GpuBindings.H2D -> H2D;
		case GpuBindings.D2H -> D2H;
		case HOST_WORK -> HOST;
		default -> D2D;
		};
	}
}
