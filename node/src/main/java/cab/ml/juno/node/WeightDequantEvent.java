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
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Period;
import jdk.jfr.StackTrace;
import jdk.jfr.Timespan;

/**
 * JFR event: weight-matrix dequantizations of one format and timing source,
 * totalled over a recording chunk (periodic, like {@link DeviceStagingEvent}).
 *
 * <p>Two sites count. The packed K-quant GEMM dequantizes its weights to FP16 on
 * the device before every batched call ({@code timing = device}, timed between two
 * stream events); the FP16 upload path dequantizes each weight once on the host at
 * load ({@code timing = host}).
 */
@Name("juno.WeightDequant")
@Label("Weight Dequant")
@Description("Weight dequantizations of one format and timing source, totalled over the recording chunk")
@Category({ "Juno", "GPU" })
@StackTrace(false)
@Period("endChunk")
public final class WeightDequantEvent extends Event {

	@Label("Format")
	@Description("GGUF quantization format, e.g. Q4_K, Q6_K, Q8_0")
	public String format;

	@Label("Timing")
	@Description("device (timed between two stream events) or host (timed on the calling thread)")
	public String timing;

	@Label("Count")
	public long count;

	@Label("Timed Count")
	public long timedCount;

	@Label("Dequant Time")
	@Description("Summed measured duration of the timed dequantizations")
	@Timespan(Timespan.NANOSECONDS)
	public long dequantNanos;

	/** Format label for a GGUF tensor type id. */
	static String format(int ggufType) {
		return switch (ggufType) {
		case 0 -> "F32";
		case 1 -> "F16";
		case 8 -> "Q8_0";
		case 10 -> "Q2_K";
		case 11 -> "Q3_K";
		case 12 -> "Q4_K";
		case 13 -> "Q5_K";
		case 14 -> "Q6_K";
		default -> "type" + ggufType;
		};
	}
}
