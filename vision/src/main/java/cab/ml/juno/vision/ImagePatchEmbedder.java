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

package cab.ml.juno.vision;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * Converts raw image bytes (JPEG / PNG / GIF / BMP, anything
 * {@link javax.imageio.ImageIO} handles) into normalised float patch tensors
 * ready for the CLIP vision encoder.
 *
 * Processing pipeline per image:
 * <ol>
 *   <li>Decode bytes to {@link BufferedImage}.
 *   <li>Resize to {@code imageSize × imageSize} using bilinear interpolation.
 *   <li>Normalise each channel with the CLIP standard mean/std:
 *       mean = {0.48145466, 0.4578275, 0.40821073},
 *       std  = {0.26862954, 0.26130258, 0.27577711}.
 *   <li>Lay out as {@code float[3 * imageSize * imageSize]} in CHW order
 *       (channel-first): all R pixels, then all G, then all B.
 * </ol>
 *
 * Thread-safe: stateless, all parameters come from the constructor.
 */
public final class ImagePatchEmbedder {

    // CLIP normalisation constants (ImageNet-derived, used by CLIP and variants)
    static final float[] MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    static final float[] STD  = {0.26862954f, 0.26130258f, 0.27577711f};

    private final int imageSize;  // target square resolution (e.g. 336 for LLaVA)
    private final int patchSize;  // patch edge length in pixels

    public ImagePatchEmbedder(VisionConfig cfg) {
        this.imageSize = cfg.imageSize();
        this.patchSize = cfg.patchSize();
    }

    /**
     * Decode, resize, and normalise an image.
     *
     * @param imageBytes raw image bytes (JPEG, PNG, …)
     * @return float[3 * imageSize * imageSize] in CHW order, CLIP-normalised
     * @throws IOException if the bytes cannot be decoded as a known image format
     */
    public float[] toPixelTensor(byte[] imageBytes) throws IOException {
        BufferedImage src = ImageIO.read(new ByteArrayInputStream(imageBytes));
        if (src == null) {
            throw new IOException("ImageIO could not decode the supplied image bytes — "
                    + "verify the format is JPEG, PNG, GIF, or BMP");
        }
        BufferedImage resized = resize(src, imageSize, imageSize);
        return normalise(resized);
    }

    /**
     * Number of patches the encoder will produce from one image.
     * Does not include the CLS token.
     */
    public int numPatches() {
        int grid = imageSize / patchSize;
        return grid * grid;
    }

    // ── Private helpers ────────────────────────────────────────────────────

    private static BufferedImage resize(BufferedImage src, int w, int h) {
        if (src.getWidth() == w && src.getHeight() == h) {
            return src;
        }
        BufferedImage dst = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = dst.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                           RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, w, h, null);
        g.dispose();
        return dst;
    }

    /**
     * Convert a {@code TYPE_INT_RGB} BufferedImage to a CHW float tensor
     * normalised with CLIP mean/std.
     *
     * Layout: out[c * H * W + y * W + x]  where c=0 R, c=1 G, c=2 B.
     */
    private static float[] normalise(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        float[] out = new float[3 * h * w];
        int planeSize = h * w;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255.0f;
                float g = ((rgb >>  8) & 0xFF) / 255.0f;
                float b = ( rgb        & 0xFF) / 255.0f;

                int pix = y * w + x;
                out[pix]              = (r - MEAN[0]) / STD[0];
                out[planeSize + pix]  = (g - MEAN[1]) / STD[1];
                out[2 * planeSize + pix] = (b - MEAN[2]) / STD[2];
            }
        }
        return out;
    }
}