package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import javax.imageio.ImageIO;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("ImagePatchEmbedder — pixel normalisation and patch geometry")
class ImagePatchEmbedderTest {

    // Small config to keep tests fast: 28px image, 14px patches → 4 patches
    private static final int IMAGE_SIZE = 28;
    private static final int PATCH_SIZE = 14;
    private static final int EXPECTED_PATCHES = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE); // 4

    private static VisionConfig cfg;
    private static ImagePatchEmbedder embedder;

    @BeforeAll
    static void setup() {
        cfg = VisionConfig.synthetic(IMAGE_SIZE, PATCH_SIZE, 64, 2, 4, 128);
        embedder = new ImagePatchEmbedder(cfg);
    }

    // ── Geometry ──────────────────────────────────────────────────────────────

    @Test
    @DisplayName("numPatches matches (imageSize / patchSize)^2")
    void num_patches_matches_config() {
        assertThat(embedder.numPatches()).isEqualTo(EXPECTED_PATCHES);
    }

    // ── Pixel tensor shape ────────────────────────────────────────────────────

    @Test
    @DisplayName("toPixelTensor returns float[3 * imageSize * imageSize]")
    void pixel_tensor_shape() throws IOException {
        byte[] jpeg = solidColorJpeg(Color.RED, IMAGE_SIZE, IMAGE_SIZE);
        float[] tensor = embedder.toPixelTensor(jpeg);
        assertThat(tensor).hasSize(3 * IMAGE_SIZE * IMAGE_SIZE);
    }

    // ── Normalisation ─────────────────────────────────────────────────────────

    @Test
    @DisplayName("pure red image: R channel positive, G and B channels negative after CLIP normalisation")
    void red_image_channel_signs() throws IOException {
        // Pure red: R=255, G=0, B=0
        // After normalisation: R = (1.0 - 0.481) / 0.269 > 0
        //                      G = (0.0 - 0.458) / 0.261 < 0
        //                      B = (0.0 - 0.408) / 0.276 < 0
        byte[] jpeg = solidColorJpeg(Color.RED, IMAGE_SIZE, IMAGE_SIZE);
        float[] tensor = embedder.toPixelTensor(jpeg);
        int plane = IMAGE_SIZE * IMAGE_SIZE;

        float rSample = tensor[0];                   // R channel first pixel
        float gSample = tensor[plane];               // G channel first pixel
        float bSample = tensor[2 * plane];           // B channel first pixel

        assertThat(rSample).isGreaterThan(0f);
        assertThat(gSample).isLessThan(0f);
        assertThat(bSample).isLessThan(0f);
    }

    @Test
    @DisplayName("pure white image: all channels near (1-mean)/std")
    void white_image_normalisation_values() throws IOException {
        byte[] jpeg = solidColorJpeg(Color.WHITE, IMAGE_SIZE, IMAGE_SIZE);
        float[] tensor = embedder.toPixelTensor(jpeg);
        int plane = IMAGE_SIZE * IMAGE_SIZE;

        float expectedR = (1.0f - ImagePatchEmbedder.MEAN[0]) / ImagePatchEmbedder.STD[0];
        float expectedG = (1.0f - ImagePatchEmbedder.MEAN[1]) / ImagePatchEmbedder.STD[1];
        float expectedB = (1.0f - ImagePatchEmbedder.MEAN[2]) / ImagePatchEmbedder.STD[2];

        assertThat(tensor[0]).isCloseTo(expectedR, within(0.02f));
        assertThat(tensor[plane]).isCloseTo(expectedG, within(0.02f));
        assertThat(tensor[2 * plane]).isCloseTo(expectedB, within(0.02f));
    }

    @Test
    @DisplayName("pure black image: all channels near (0-mean)/std")
    void black_image_normalisation_values() throws IOException {
        byte[] jpeg = solidColorJpeg(Color.BLACK, IMAGE_SIZE, IMAGE_SIZE);
        float[] tensor = embedder.toPixelTensor(jpeg);
        int plane = IMAGE_SIZE * IMAGE_SIZE;

        float expectedR = (0.0f - ImagePatchEmbedder.MEAN[0]) / ImagePatchEmbedder.STD[0];
        float expectedG = (0.0f - ImagePatchEmbedder.MEAN[1]) / ImagePatchEmbedder.STD[1];
        float expectedB = (0.0f - ImagePatchEmbedder.MEAN[2]) / ImagePatchEmbedder.STD[2];

        // JPEG compression can alter pixel values slightly; use a loose tolerance
        assertThat(tensor[0]).isCloseTo(expectedR, within(0.05f));
        assertThat(tensor[plane]).isCloseTo(expectedG, within(0.05f));
        assertThat(tensor[2 * plane]).isCloseTo(expectedB, within(0.05f));
    }

    // ── Resize ────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("image larger than imageSize is resized — output shape unchanged")
    void oversized_image_resized_correctly() throws IOException {
        byte[] jpeg = solidColorJpeg(Color.BLUE, 512, 512);
        float[] tensor = embedder.toPixelTensor(jpeg);
        assertThat(tensor).hasSize(3 * IMAGE_SIZE * IMAGE_SIZE);
    }

    @Test
    @DisplayName("image smaller than imageSize is resized — output shape unchanged")
    void undersized_image_resized_correctly() throws IOException {
        byte[] jpeg = solidColorJpeg(Color.GREEN, 7, 7);
        float[] tensor = embedder.toPixelTensor(jpeg);
        assertThat(tensor).hasSize(3 * IMAGE_SIZE * IMAGE_SIZE);
    }

    // ── Error handling ────────────────────────────────────────────────────────

    @Test
    @DisplayName("invalid bytes throw IOException with a descriptive message")
    void invalid_bytes_throw_io_exception() {
        byte[] garbage = new byte[] { 0x00, 0x11, 0x22, (byte) 0xFF };
        assertThatThrownBy(() -> embedder.toPixelTensor(garbage))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("ImageIO could not decode");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static byte[] solidColorJpeg(Color color, int w, int h) throws IOException {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(color);
        g.fillRect(0, 0, w, h);
        g.dispose();
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(img, "jpeg", baos);
        return baos.toByteArray();
    }

    // ── EXIF orientation: byte-level parser ─────────────────────────────────────
    //
    // Added 2026-07-20: ImageIO.read() ignores EXIF Orientation entirely — a
    // real phone/camera photo stored with this tag would silently be fed to
    // the vision encoder sideways or upside down. readExifOrientation()/
    // applyExifOrientation() correct for this before anything else runs.

    @Test
    @DisplayName("readExifOrientation: plain JPEG with no EXIF segment returns 1 (no-op)")
    void exif_orientation_no_exif_segment_returns_1() throws IOException {
        byte[] jpeg = solidColorJpeg(Color.RED, 8, 8); // ImageIO.write does not embed EXIF
        assertThat(ImagePatchEmbedder.readExifOrientation(jpeg)).isEqualTo(1);
    }

    @Test
    @DisplayName("readExifOrientation: non-JPEG bytes (PNG) return 1")
    void exif_orientation_non_jpeg_returns_1() throws IOException {
        BufferedImage img = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(img, "png", baos);
        assertThat(ImagePatchEmbedder.readExifOrientation(baos.toByteArray())).isEqualTo(1);
    }

    @Test
    @DisplayName("readExifOrientation: too-short / garbage bytes return 1, never throw")
    void exif_orientation_garbage_bytes_returns_1_no_throw() {
        assertThat(ImagePatchEmbedder.readExifOrientation(new byte[0])).isEqualTo(1);
        assertThat(ImagePatchEmbedder.readExifOrientation(new byte[] { 0x00 })).isEqualTo(1);
        assertThat(ImagePatchEmbedder.readExifOrientation(new byte[] { (byte) 0xFF, (byte) 0xD8, 0x00 }))
                .isEqualTo(1);
    }

    @Test
    @DisplayName("readExifOrientation: synthetic APP1/Exif segment (little-endian TIFF), orientation=6")
    void exif_orientation_parses_synthetic_segment_little_endian() {
        byte[] jpeg = buildJpegWithExifOrientation(6, true);
        assertThat(ImagePatchEmbedder.readExifOrientation(jpeg)).isEqualTo(6);
    }

    @Test
    @DisplayName("readExifOrientation: synthetic APP1/Exif segment (big-endian TIFF), orientation=8")
    void exif_orientation_parses_synthetic_segment_big_endian() {
        byte[] jpeg = buildJpegWithExifOrientation(8, false);
        assertThat(ImagePatchEmbedder.readExifOrientation(jpeg)).isEqualTo(8);
    }

    @Test
    @DisplayName("readExifOrientation: every valid orientation value 1-8 round-trips through the parser")
    void exif_orientation_all_values_roundtrip() {
        for (int o = 1; o <= 8; o++) {
            byte[] jpeg = buildJpegWithExifOrientation(o, o % 2 == 0);
            assertThat(ImagePatchEmbedder.readExifOrientation(jpeg)).as("orientation %d", o).isEqualTo(o);
        }
    }

    /** Builds a minimal SOI + APP1/Exif segment containing exactly one IFD0
     * entry (tag=Orientation) — enough for readExifOrientation() to parse,
     * without needing a full valid/decodable JPEG. */
    private static byte[] buildJpegWithExifOrientation(int orientation, boolean littleEndian) {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        out.write(0xFF); out.write(0xD8); // SOI

        java.io.ByteArrayOutputStream tiff = new java.io.ByteArrayOutputStream();
        if (littleEndian) {
            tiff.write('I'); tiff.write('I');
            writeInt16(tiff, 42, true);
            writeInt32(tiff, 8, true);
        } else {
            tiff.write('M'); tiff.write('M');
            writeInt16(tiff, 42, false);
            writeInt32(tiff, 8, false);
        }
        writeInt16(tiff, 1, littleEndian);       // IFD0: 1 entry
        writeInt16(tiff, 0x0112, littleEndian);  // tag = Orientation
        writeInt16(tiff, 3, littleEndian);       // type = SHORT
        writeInt32(tiff, 1, littleEndian);       // count = 1
        writeInt16(tiff, orientation, littleEndian); // value (first 2 of 4 bytes)
        writeInt16(tiff, 0, littleEndian);           // padding to fill 4-byte value field
        writeInt32(tiff, 0, littleEndian);       // next IFD offset = none
        byte[] tiffBytes = tiff.toByteArray();

        byte[] exifHeader = { 'E', 'x', 'i', 'f', 0, 0 };
        int segLen = 2 + exifHeader.length + tiffBytes.length; // length field includes itself

        out.write(0xFF); out.write(0xE1); // APP1
        out.write((segLen >> 8) & 0xFF);
        out.write(segLen & 0xFF);
        out.writeBytes(exifHeader);
        out.writeBytes(tiffBytes);
        return out.toByteArray();
    }

    private static void writeInt16(java.io.ByteArrayOutputStream out, int v, boolean little) {
        int b0 = v & 0xFF, b1 = (v >> 8) & 0xFF;
        if (little) { out.write(b0); out.write(b1); } else { out.write(b1); out.write(b0); }
    }

    private static void writeInt32(java.io.ByteArrayOutputStream out, int v, boolean little) {
        int b0 = v & 0xFF, b1 = (v >> 8) & 0xFF, b2 = (v >> 16) & 0xFF, b3 = (v >> 24) & 0xFF;
        if (little) {
            out.write(b0); out.write(b1); out.write(b2); out.write(b3);
        } else {
            out.write(b3); out.write(b2); out.write(b1); out.write(b0);
        }
    }

    // ── EXIF orientation: transform geometry (convention-agnostic invariants) ───
    //
    // These deliberately avoid asserting exact CW-vs-CCW pixel placement —
    // that's easy to get backwards in a hand-derived test assertion just as
    // easily as in the implementation, which would give false confidence
    // either way. Instead: dimension swap/preserve (unambiguous per
    // orientation family) and a direction-agnostic round-trip invariant
    // (180° twice = identity, regardless of rotation direction convention).

    @Test
    @DisplayName("applyExifOrientation: orientation=1 is a no-op (same instance)")
    void exif_transform_orientation_1_is_noop() {
        BufferedImage img = new BufferedImage(5, 3, BufferedImage.TYPE_INT_RGB);
        assertThat(ImagePatchEmbedder.applyExifOrientation(img, 1)).isSameAs(img);
    }

    @Test
    @DisplayName("applyExifOrientation: orientation=0 or 9 (invalid) is a no-op (same instance)")
    void exif_transform_invalid_orientation_is_noop() {
        BufferedImage img = new BufferedImage(5, 3, BufferedImage.TYPE_INT_RGB);
        assertThat(ImagePatchEmbedder.applyExifOrientation(img, 0)).isSameAs(img);
        assertThat(ImagePatchEmbedder.applyExifOrientation(img, 9)).isSameAs(img);
    }

    @Test
    @DisplayName("applyExifOrientation: 90°-family orientations (5,6,7,8) swap width and height")
    void exif_transform_90_family_swaps_dimensions() {
        BufferedImage img = new BufferedImage(5, 3, BufferedImage.TYPE_INT_RGB);
        for (int o : new int[] { 5, 6, 7, 8 }) {
            BufferedImage out = ImagePatchEmbedder.applyExifOrientation(img, o);
            assertThat(out.getWidth()).as("orientation %d width", o).isEqualTo(3);
            assertThat(out.getHeight()).as("orientation %d height", o).isEqualTo(5);
        }
    }

    @Test
    @DisplayName("applyExifOrientation: flip/180° orientations (2,3,4) preserve width and height")
    void exif_transform_flip_and_180_preserve_dimensions() {
        BufferedImage img = new BufferedImage(5, 3, BufferedImage.TYPE_INT_RGB);
        for (int o : new int[] { 2, 3, 4 }) {
            BufferedImage out = ImagePatchEmbedder.applyExifOrientation(img, o);
            assertThat(out.getWidth()).as("orientation %d width", o).isEqualTo(5);
            assertThat(out.getHeight()).as("orientation %d height", o).isEqualTo(3);
        }
    }

    @Test
    @DisplayName("applyExifOrientation: rotating 180° twice returns to the original pixel layout "
            + "(direction-agnostic invariant — true regardless of CW/CCW convention)")
    void exif_transform_180_twice_is_identity() {
        BufferedImage img = new BufferedImage(4, 3, BufferedImage.TYPE_INT_RGB);
        // Give every pixel a distinct color so any misplacement is detectable.
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 4; x++)
                img.setRGB(x, y, (x * 10 + y) | 0xFF000000);

        BufferedImage once = ImagePatchEmbedder.applyExifOrientation(img, 3);
        BufferedImage twice = ImagePatchEmbedder.applyExifOrientation(once, 3);

        assertThat(twice.getWidth()).isEqualTo(img.getWidth());
        assertThat(twice.getHeight()).isEqualTo(img.getHeight());
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 4; x++) {
                assertThat(twice.getRGB(x, y)).as("pixel (%d,%d)", x, y).isEqualTo(img.getRGB(x, y));
            }
        }
    }

    @Test
    @DisplayName("applyExifOrientation: orientation=3 (180°) moves the top-left pixel to the "
            + "bottom-right corner — unambiguous regardless of rotation direction convention")
    void exif_transform_180_moves_corner_to_opposite_corner() {
        BufferedImage img = new BufferedImage(4, 3, BufferedImage.TYPE_INT_RGB);
        int cornerColor = 0xFF123456;
        img.setRGB(0, 0, cornerColor);

        BufferedImage rotated = ImagePatchEmbedder.applyExifOrientation(img, 3);

        assertThat(rotated.getRGB(3, 2)).isEqualTo(cornerColor);
    }

    @Test
    @DisplayName("toPixelTensor: a JPEG with orientation=6 produces the EXIF-corrected pixel tensor, "
            + "not the raw sensor-orientation one (end-to-end wiring check)")
    void toPixelTensor_applies_exif_correction_end_to_end() throws IOException {
        // Build a plain solid-color JPEG (no real EXIF), then splice a synthetic
        // orientation=6 APP1 segment in front of it. ImageIO.read() will still
        // decode the underlying image data fine (it just skips the extra APP1
        // marker), but toPixelTensor() must now report the corrected dimensions.
        byte[] plain = solidColorJpeg(Color.BLUE, 6, 4);
        byte[] withExif = spliceExifApp1(plain, 6);

        // Sanity: the raw decode is still 6x4 (unswapped) — proves any
        // dimension swap we observe comes from EXIF correction, not from the
        // image itself.
        BufferedImage rawDecoded = ImageIO.read(new java.io.ByteArrayInputStream(withExif));
        assertThat(rawDecoded.getWidth()).isEqualTo(6);
        assertThat(rawDecoded.getHeight()).isEqualTo(4);

        assertThat(ImagePatchEmbedder.readExifOrientation(withExif)).isEqualTo(6);

        // Full pipeline must not throw, and must still produce a correctly
        // sized tensor for the embedder's configured target size regardless
        // of the source image's (now-corrected) aspect ratio.
        float[] tensor = embedder.toPixelTensor(withExif);
        assertThat(tensor).hasSize(3 * IMAGE_SIZE * IMAGE_SIZE);
    }

    /** Inserts a synthetic orientation APP1/Exif segment right after the SOI
     * marker of an existing, real, decodable JPEG. */
    private static byte[] spliceExifApp1(byte[] realJpeg, int orientation) {
        byte[] app1Only = buildJpegWithExifOrientation(orientation, true); // SOI + APP1
        int app1Len = app1Only.length - 2; // exclude the SOI we already have in realJpeg
        byte[] out = new byte[realJpeg.length + app1Len];
        out[0] = realJpeg[0];
        out[1] = realJpeg[1]; // SOI
        System.arraycopy(app1Only, 2, out, 2, app1Len); // APP1 segment
        System.arraycopy(realJpeg, 2, out, 2 + app1Len, realJpeg.length - 2); // rest of real JPEG
        return out;
    }
}