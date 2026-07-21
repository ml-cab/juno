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
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 * Converts raw image bytes (JPEG / PNG / GIF / BMP, anything
 * {@link javax.imageio.ImageIO} handles) into normalised float patch tensors
 * ready for the CLIP vision encoder.
 *
 * Processing pipeline per image:
 * <ol>
 *   <li>Decode bytes to {@link BufferedImage}.
 *   <li>For JPEG: read the EXIF Orientation tag (0x0112) and rotate/flip the
 *       image to upright before anything else. {@link ImageIO#read} does NOT
 *       do this — it returns the raw sensor-orientation pixel grid as-is.
 *       Phone/camera photos routinely carry this tag (portrait shots are
 *       usually stored as landscape pixels + orientation=6); a synthetic
 *       image (a saved PNG icon, a screenshot) essentially never does. Added
 *       2026-07-20 after a real camera photo produced badly wrong vision
 *       output under conditions where a synthetic test image had not.
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

    private static final Logger log = Logger.getLogger(ImagePatchEmbedder.class.getName());

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
     * Decode, EXIF-correct, resize, and normalise an image.
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
        int rawW = src.getWidth();
        int rawH = src.getHeight();

        int orientation = readExifOrientation(imageBytes);
        BufferedImage upright = applyExifOrientation(src, orientation);

        log.info(String.format(
                "Image load: decodedBytes=%d rawDims=%dx%d exifOrientation=%d correctedDims=%dx%d "
                        + "targetSize=%dx%d",
                imageBytes.length, rawW, rawH, orientation, upright.getWidth(), upright.getHeight(), imageSize,
                imageSize));

        BufferedImage resized = resize(upright, imageSize, imageSize);
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

    // ── EXIF orientation ─────────────────────────────────────────────────

    /**
     * Reads the EXIF Orientation tag (0x0112) from a JPEG's APP1 segment.
     * Returns 1 (normal, no correction needed) for non-JPEG bytes, JPEGs
     * with no EXIF segment, or any parsing failure — this is deliberately
     * fail-safe: a wrong read of "1" just skips correction (same as today's
     * pre-existing behavior), it never applies a wrong rotation.
     */
    static int readExifOrientation(byte[] b) {
        try {
            if (b.length < 4 || (b[0] & 0xFF) != 0xFF || (b[1] & 0xFF) != 0xD8) {
                return 1; // not a JPEG (no SOI marker) — PNG/GIF/BMP don't use this mechanism
            }
            int pos = 2;
            while (pos + 4 <= b.length) {
                if ((b[pos] & 0xFF) != 0xFF) {
                    return 1; // malformed marker stream
                }
                int marker = b[pos + 1] & 0xFF;
                if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
                    pos += 2; // markers with no length/payload
                    continue;
                }
                if (marker == 0xD9 || marker == 0xDA) {
                    return 1; // EOI / start-of-scan — no APP1 found before image data
                }
                int segLen = ((b[pos + 2] & 0xFF) << 8) | (b[pos + 3] & 0xFF);
                if (marker == 0xE1 && segLen >= 8) {
                    int segStart = pos + 4;
                    if (isExifHeader(b, segStart)) {
                        int o = parseTiffOrientation(b, segStart + 6);
                        return (o >= 1 && o <= 8) ? o : 1;
                    }
                }
                if (segLen < 2) {
                    return 1; // malformed length
                }
                pos += 2 + segLen;
            }
            return 1;
        } catch (RuntimeException e) {
            // Fail-safe: any unexpected parsing issue just skips correction.
            log.warning("EXIF orientation parse failed, assuming orientation=1: " + e);
            return 1;
        }
    }

    private static boolean isExifHeader(byte[] b, int off) {
        return off + 6 <= b.length && b[off] == 'E' && b[off + 1] == 'x' && b[off + 2] == 'i' && b[off + 3] == 'f'
                && b[off + 4] == 0 && b[off + 5] == 0;
    }

    private static int parseTiffOrientation(byte[] b, int tiffStart) {
        if (tiffStart + 8 > b.length) {
            return 1;
        }
        boolean little;
        if (b[tiffStart] == 'I' && b[tiffStart + 1] == 'I') {
            little = true;
        } else if (b[tiffStart] == 'M' && b[tiffStart + 1] == 'M') {
            little = false;
        } else {
            return 1;
        }
        int ifdOffset = readInt32(b, tiffStart + 4, little);
        int ifdStart = tiffStart + ifdOffset;
        if (ifdStart < 0 || ifdStart + 2 > b.length) {
            return 1;
        }
        int numEntries = readInt16(b, ifdStart, little);
        for (int i = 0; i < numEntries; i++) {
            int entryOffset = ifdStart + 2 + i * 12;
            if (entryOffset + 12 > b.length) {
                break;
            }
            int tag = readInt16(b, entryOffset, little);
            if (tag == 0x0112) {
                return readInt16(b, entryOffset + 8, little);
            }
        }
        return 1;
    }

    private static int readInt16(byte[] b, int off, boolean little) {
        int b0 = b[off] & 0xFF;
        int b1 = b[off + 1] & 0xFF;
        return little ? (b1 << 8 | b0) : (b0 << 8 | b1);
    }

    private static int readInt32(byte[] b, int off, boolean little) {
        int b0 = b[off] & 0xFF, b1 = b[off + 1] & 0xFF, b2 = b[off + 2] & 0xFF, b3 = b[off + 3] & 0xFF;
        return little ? (b3 << 24 | b2 << 16 | b1 << 8 | b0) : (b0 << 24 | b1 << 16 | b2 << 8 | b3);
    }

    /**
     * Rotates/flips {@code img} to upright per the EXIF Orientation value.
     * orientation=1 (or any value outside 1-8) is a no-op. Standard 8-case
     * EXIF orientation transform matrices (widely used recipe; see e.g. the
     * EXIF spec's own diagram, TIFF6 Orientation tag definition, or any of
     * the many public implementations of this exact table).
     */
    static BufferedImage applyExifOrientation(BufferedImage img, int orientation) {
        if (orientation <= 1 || orientation > 8) {
            return img;
        }
        int w = img.getWidth();
        int h = img.getHeight();
        AffineTransform t = new AffineTransform();
        switch (orientation) {
            case 2 -> { // flip horizontal
                t = AffineTransform.getScaleInstance(-1.0, 1.0);
                t.translate(-w, 0);
            }
            case 3 -> { // rotate 180
                t = AffineTransform.getTranslateInstance(w, h);
                t.rotate(Math.PI);
            }
            case 4 -> { // flip vertical
                t = AffineTransform.getScaleInstance(1.0, -1.0);
                t.translate(0, -h);
            }
            case 5 -> { // transpose (flip horizontal + rotate 90 CW)
                t = AffineTransform.getRotateInstance(-Math.PI / 2);
                t.scale(-1.0, 1.0);
            }
            case 6 -> { // rotate 90 CW
                t = AffineTransform.getTranslateInstance(h, 0);
                t.rotate(Math.PI / 2);
            }
            case 7 -> { // transverse (flip horizontal + rotate 90 CCW)
                t = AffineTransform.getScaleInstance(-1.0, 1.0);
                t.translate(-h, 0);
                t.translate(0, w);
                t.rotate(3 * Math.PI / 2);
            }
            case 8 -> { // rotate 90 CCW
                t = AffineTransform.getTranslateInstance(0, w);
                t.rotate(3 * Math.PI / 2);
            }
            default -> {
                return img;
            }
        }
        boolean swapDims = orientation >= 5; // 5,6,7,8 are 90°-family rotations
        int newW = swapDims ? h : w;
        int newH = swapDims ? w : h;
        BufferedImage dst = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = dst.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img, t, null);
        g.dispose();
        return dst;
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