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
}