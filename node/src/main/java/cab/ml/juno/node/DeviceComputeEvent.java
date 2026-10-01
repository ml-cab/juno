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
 * JFR event: device kernel launches from one site in one phase, totalled over a
 * recording chunk (periodic, like {@link DeviceStagingEvent}).
 *
 * <p>{@code juno.MatVec} spans the whole batched matmul call: host packing, copies,
 * dequantization and the GEMM. {@link DeviceStagingEvent} and
 * {@link WeightDequantEvent} take the first three out of it; this event times the
 * kernel itself, so a prefill breakdown reads the GEMM rather than inferring it as a
 * residue. Sites: {@code gemm_half} (the tiled FP16 GEMM, on FP16 weights or on
 * packed K-quant weights dequantized just before it), {@code gemv_half_batched} (the
 * strided batched FP16 GEMV for windows of two to eight rows), {@code gemm_fp32}
 * (the FP32 BLAS GEMM), {@code mmq_packed} (the packed integer-dot GEMV at decode
 * width) and {@code gqa_attention} (the attention kernel).
 *
 * <p>Asynchronous kernels are timed on the device between two stream events
 * ({@link DeviceSpanTimer}); kernels on the default stream are timed on the host
 * between two drains of that stream ({@link DeviceComputeClock}). As with copies,
 * decode-width launches are counted and not timed.
 */
@Name("juno.DeviceCompute")
@Label("Device Compute")
@Description("Device kernel launches of one site in one phase, totalled over the recording chunk")
@Category({ "Juno", "GPU" })
@StackTrace(false)
@Period("endChunk")
public final class DeviceComputeEvent extends Event {

	static final String GEMM_HALF = "gemm_half";
	static final String GEMV_HALF_BATCHED = "gemv_half_batched";
	static final String GEMM_FP32 = "gemm_fp32";
	static final String MMQ_PACKED = "mmq_packed";
	static final String GQA_ATTENTION = "gqa_attention";

	@Label("Site")
	@Description("The kernel: gemm_half, gemv_half_batched, gemm_fp32, mmq_packed or gqa_attention")
	public String site;

	@Label("Phase")
	@Description("prefill (issued by a forward call over more than one row), decode (one row), "
			+ "or other (outside a forward call)")
	public String phase;

	@Label("Count")
	public long count;

	@Label("Timed Count")
	@Description("Launches whose duration is included in Compute Time")
	public long timedCount;

	@Label("Compute Time")
	@Description("Summed measured duration of the timed launches")
	@Timespan(Timespan.NANOSECONDS)
	public long computeNanos;
}
