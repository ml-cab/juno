/*
 * Created by Yevhen Soldatov
 * Initial implementation: 2026
 *
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

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Per-shard projection weights uploaded once to device memory for pipeline-parallel
 * GPU inference ({@link CudaMatVec#sgemv(DeviceFloatMatrix, float[])}).
 *
 * <p>
 * Not used in tensor-parallel mode ({@code tensorWorldSize > 1}), where each node
 * holds all layers and resident FP32 weights would exceed typical VRAM budgets.
 */
public final class LlamaGpuResidentWeights implements AutoCloseable {

	private static final Logger log = Logger.getLogger(LlamaGpuResidentWeights.class.getName());

	private final DeviceFloatMatrix[] wqD;
	private final DeviceFloatMatrix[] wkD;
	private final DeviceFloatMatrix[] wvD;
	private final DeviceFloatMatrix[] woD;
	private final DeviceFloatMatrix[] wGateD;
	private final DeviceFloatMatrix[] wUpD;
	private final DeviceFloatMatrix[] wDownD;
	private final DeviceFloatMatrix outputProjD;

	private LlamaGpuResidentWeights(DeviceFloatMatrix[] wqD, DeviceFloatMatrix[] wkD, DeviceFloatMatrix[] wvD,
			DeviceFloatMatrix[] woD, DeviceFloatMatrix[] wGateD, DeviceFloatMatrix[] wUpD,
			DeviceFloatMatrix[] wDownD, DeviceFloatMatrix outputProjD) {
		this.wqD = wqD;
		this.wkD = wkD;
		this.wvD = wvD;
		this.woD = woD;
		this.wGateD = wGateD;
		this.wUpD = wUpD;
		this.wDownD = wDownD;
		this.outputProjD = outputProjD;
	}

	/**
	 * Dequantizes each projection matrix via {@link GgufReader#tensorEphemeral(String)}
	 * (no reader cache) and uploads to the GPU.
	 */
	public static LlamaGpuResidentWeights upload(GgufReader r, LlamaConfig cfg, ShardContext ctx, GpuContext gpuCtx)
			throws IOException {
		int startLayer = ctx.startLayer();
		int endLayer = ctx.endLayer();
		int L = endLayer - startLayer;
		int H = cfg.hiddenDim();
		int kvDim = cfg.kvDim();
		int I = cfg.intermediateSize();

		DeviceFloatMatrix[] wqD = new DeviceFloatMatrix[L];
		DeviceFloatMatrix[] wkD = new DeviceFloatMatrix[L];
		DeviceFloatMatrix[] wvD = new DeviceFloatMatrix[L];
		DeviceFloatMatrix[] woD = new DeviceFloatMatrix[L];
		DeviceFloatMatrix[] wGateD = new DeviceFloatMatrix[L];
		DeviceFloatMatrix[] wUpD = new DeviceFloatMatrix[L];
		DeviceFloatMatrix[] wDownD = new DeviceFloatMatrix[L];

		for (int li = 0; li < L; li++) {
			int i = li + startLayer;
			String p = "blk." + i + ".";
			float[] wq = r.tensorEphemeral(p + "attn_q.weight");
			wqD[li] = DeviceFloatMatrix.upload(gpuCtx, wq, H, H);
			float[] wk = r.tensorEphemeral(p + "attn_k.weight");
			wkD[li] = DeviceFloatMatrix.upload(gpuCtx, wk, kvDim, H);
			float[] wv = r.tensorEphemeral(p + "attn_v.weight");
			wvD[li] = DeviceFloatMatrix.upload(gpuCtx, wv, kvDim, H);
			float[] wo = r.tensorEphemeral(p + "attn_output.weight");
			woD[li] = DeviceFloatMatrix.upload(gpuCtx, wo, H, H);
			float[] wg = r.tensorEphemeral(p + "ffn_gate.weight");
			wGateD[li] = DeviceFloatMatrix.upload(gpuCtx, wg, I, H);
			float[] wu = r.tensorEphemeral(p + "ffn_up.weight");
			wUpD[li] = DeviceFloatMatrix.upload(gpuCtx, wu, I, H);
			float[] wd = r.tensorEphemeral(p + "ffn_down.weight");
			wDownD[li] = DeviceFloatMatrix.upload(gpuCtx, wd, H, I);
		}

		DeviceFloatMatrix outD = null;
		if (ctx.hasOutputProjection()) {
			float[] outHost;
			if (r.hasTensor("output.weight")) {
				outHost = r.tensorEphemeral("output.weight");
			} else {
				log.info(
						"output.weight not found — tied embeddings; uploading token_embd.weight as output projection");
				outHost = r.tensorEphemeral("token_embd.weight");
			}
			outD = DeviceFloatMatrix.upload(gpuCtx, outHost, cfg.vocabSize(), H);
		}

		log.info("GPU-resident projection weights uploaded — " + L + " layers  outputProj=" + (outD != null));
		return new LlamaGpuResidentWeights(wqD, wkD, wvD, woD, wGateD, wUpD, wDownD, outD);
	}

	public DeviceFloatMatrix wq(int li) {
		return wqD[li];
	}

	public DeviceFloatMatrix wk(int li) {
		return wkD[li];
	}

	public DeviceFloatMatrix wv(int li) {
		return wvD[li];
	}

	public DeviceFloatMatrix wo(int li) {
		return woD[li];
	}

	public DeviceFloatMatrix wGate(int li) {
		return wGateD[li];
	}

	public DeviceFloatMatrix wUp(int li) {
		return wUpD[li];
	}

	public DeviceFloatMatrix wDown(int li) {
		return wDownD[li];
	}

	public DeviceFloatMatrix outputProj() {
		return outputProjD;
	}

	@Override
	public void close() {
		closeAll(wqD);
		closeAll(wkD);
		closeAll(wvD);
		closeAll(woD);
		closeAll(wGateD);
		closeAll(wUpD);
		closeAll(wDownD);
		if (outputProjD != null)
			outputProjD.close();
	}

	private static void closeAll(DeviceFloatMatrix[] arr) {
		for (DeviceFloatMatrix m : arr) {
			if (m != null && !m.isClosed())
				m.close();
		}
	}
}
