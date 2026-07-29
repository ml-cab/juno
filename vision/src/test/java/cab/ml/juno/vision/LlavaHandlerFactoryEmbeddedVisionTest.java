package cab.ml.juno.vision;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests {@link LlavaHandlerFactory#isVisionArchitecture} for the embedded-GGUF
 * case — a single llamafile that bundles both the text model and the vision
 * encoder as two GGUF entries in the same ZIP (moondream2 pattern).
 */
@DisplayName("LlavaHandlerFactory — isVisionArchitecture with embedded vision GGUF")
class LlavaHandlerFactoryEmbeddedVisionTest {

    // ── minimal GGUF builder (standalone, no dependency on node test classes) ─

    private static byte[] minimalGguf(String tensorName) {
        final int MAGIC     = 0x46554747;
        final int ALIGNMENT = 32;
        byte[] nameBytes = tensorName.getBytes(StandardCharsets.UTF_8);
        byte[] data = new byte[4]; // one F32 element
        ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putFloat(1.0f);

        int prePad  = 24 + 8 + nameBytes.length + 4 + 8 + 4 + 8;
        int aligned = ((prePad + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

        ByteBuffer buf = ByteBuffer.allocate(aligned + data.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(MAGIC);
        buf.putInt(3);
        buf.putLong(1);
        buf.putLong(0);
        buf.putLong(nameBytes.length);
        buf.put(nameBytes);
        buf.putInt(1);
        buf.putLong(1);
        buf.putInt(0); // F32
        buf.putLong(0);
        buf.put(new byte[aligned - prePad]);
        buf.put(data);
        return buf.array();
    }

    /**
     * Build a two-entry ZIP llamafile: entry0 = {@code firstGguf},
     * entry1 = {@code secondGguf}.
     */
    private static Path twoGgufLlamafile(Path dir, byte[] firstGguf, byte[] secondGguf)
            throws IOException {
        byte[] stub = "MZ\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
                .getBytes(StandardCharsets.ISO_8859_1);
        String[] names = { "llm.gguf", "mmproj.gguf" };
        byte[][] blobs = { firstGguf, secondGguf };

        long[] localOffsets = new long[2];
        long cursor = stub.length;
        for (int i = 0; i < 2; i++) {
            localOffsets[i] = cursor;
            cursor += 30 + names[i].length() + blobs[i].length;
        }
        long cdOffset = cursor;

        byte[][] cdes = new byte[2][];
        int cdLen = 0;
        for (int i = 0; i < 2; i++) {
            byte[] fn = names[i].getBytes(StandardCharsets.UTF_8);
            ByteBuffer cde = ByteBuffer.allocate(46 + fn.length).order(ByteOrder.LITTLE_ENDIAN);
            cde.putInt(0x02014b50);
            cde.putShort((short) 20); cde.putShort((short) 20);
            cde.putShort((short) 0); cde.putShort((short) 0);
            cde.putShort((short) 0); cde.putShort((short) 0);
            cde.putInt(0);
            cde.putInt(blobs[i].length); cde.putInt(blobs[i].length);
            cde.putShort((short) fn.length);
            cde.putShort((short) 0); cde.putShort((short) 0);
            cde.putShort((short) 0); cde.putShort((short) 0);
            cde.putInt(0);
            cde.putInt((int) localOffsets[i]);
            cde.put(fn);
            cdes[i] = cde.array();
            cdLen += cdes[i].length;
        }

        ByteBuffer eocd = ByteBuffer.allocate(22).order(ByteOrder.LITTLE_ENDIAN);
        eocd.putInt(0x06054b50);
        eocd.putShort((short) 0); eocd.putShort((short) 0);
        eocd.putShort((short) 2); eocd.putShort((short) 2);
        eocd.putInt(cdLen);
        eocd.putInt((int) cdOffset);
        eocd.putShort((short) 0);

        int total = (int) cdOffset + cdLen + 22;
        ByteBuffer out = ByteBuffer.allocate(total);
        out.put(stub);
        for (int i = 0; i < 2; i++) {
            byte[] fn = names[i].getBytes(StandardCharsets.UTF_8);
            ByteBuffer lh = ByteBuffer.allocate(30 + fn.length).order(ByteOrder.LITTLE_ENDIAN);
            lh.putInt(0x04034b50);
            lh.putShort((short) 20); lh.putShort((short) 0);
            lh.putShort((short) 0);  lh.putShort((short) 0); lh.putShort((short) 0);
            lh.putInt(0);
            lh.putInt(blobs[i].length); lh.putInt(blobs[i].length);
            lh.putShort((short) fn.length); lh.putShort((short) 0);
            lh.put(fn);
            out.put(lh.array());
            out.put(blobs[i]);
        }
        for (byte[] cde : cdes) out.put(cde);
        out.put(eocd.array());

        Path file = dir.resolve("test.llamafile");
        Files.write(file, out.array());
        return file;
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("isVisionArchitecture is true when vision encoder is the second GGUF in a llamafile")
    void isVisionArchitecture_trueWhenVisionIsSecondGguf(@TempDir Path tmp) throws IOException {
        byte[] llmGguf    = minimalGguf("token_embd.weight");     // text model — no vision tensors
        byte[] visionGguf = minimalGguf("v.patch_embd.weight");   // vision encoder

        Path llamafile = twoGgufLlamafile(tmp, llmGguf, visionGguf);

        // No --mmproj-path: factory must discover the second GGUF automatically
        assertThat(LlavaHandlerFactory.isVisionArchitecture(llamafile, null)).isTrue();
    }

    @Test
    @DisplayName("isVisionArchitecture is false when the llamafile has only one GGUF with no vision tensors")
    void isVisionArchitecture_falseWhenNoVisionEntry(@TempDir Path tmp) throws IOException {
        byte[] gguf = minimalGguf("token_embd.weight");
        // Single-entry llamafile — no vision GGUF
        byte[] stub = "MZ\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
                .getBytes(StandardCharsets.ISO_8859_1);
        byte[] fn = "llm.gguf".getBytes(StandardCharsets.UTF_8);
        long localOff = stub.length;
        ByteBuffer lh = ByteBuffer.allocate(30 + fn.length).order(ByteOrder.LITTLE_ENDIAN);
        lh.putInt(0x04034b50); lh.putShort((short) 20); lh.putShort((short) 0);
        lh.putShort((short) 0); lh.putShort((short) 0); lh.putShort((short) 0);
        lh.putInt(0); lh.putInt(gguf.length); lh.putInt(gguf.length);
        lh.putShort((short) fn.length); lh.putShort((short) 0); lh.put(fn);

        long cdOff = localOff + 30 + fn.length + gguf.length;
        ByteBuffer cde = ByteBuffer.allocate(46 + fn.length).order(ByteOrder.LITTLE_ENDIAN);
        cde.putInt(0x02014b50); cde.putShort((short) 20); cde.putShort((short) 20);
        cde.putShort((short) 0); cde.putShort((short) 0); cde.putShort((short) 0); cde.putShort((short) 0);
        cde.putInt(0); cde.putInt(gguf.length); cde.putInt(gguf.length);
        cde.putShort((short) fn.length); cde.putShort((short) 0); cde.putShort((short) 0);
        cde.putShort((short) 0); cde.putShort((short) 0); cde.putInt(0);
        cde.putInt((int) localOff); cde.put(fn);

        ByteBuffer eocd = ByteBuffer.allocate(22).order(ByteOrder.LITTLE_ENDIAN);
        eocd.putInt(0x06054b50); eocd.putShort((short) 0); eocd.putShort((short) 0);
        eocd.putShort((short) 1); eocd.putShort((short) 1);
        eocd.putInt(cde.capacity()); eocd.putInt((int) cdOff); eocd.putShort((short) 0);

        int total = stub.length + lh.capacity() + gguf.length + cde.capacity() + 22;
        ByteBuffer file = ByteBuffer.allocate(total);
        file.put(stub); file.put(lh.array()); file.put(gguf);
        file.put(cde.array()); file.put(eocd.array());

        Path llamafile = tmp.resolve("text-only.llamafile");
        Files.write(llamafile, file.array());

        assertThat(LlavaHandlerFactory.isVisionArchitecture(llamafile, null)).isFalse();
    }

    @Test
    @DisplayName("isVisionArchitecture is true when an explicit mmproj path contains v.patch_embd.weight")
    void isVisionArchitecture_trueForExplicitMmprojPath(@TempDir Path tmp) throws IOException {
        byte[] llmGguf    = minimalGguf("token_embd.weight");
        byte[] visionGguf = minimalGguf("v.patch_embd.weight");

        Path modelPath  = tmp.resolve("llm.gguf");
        Path mmprojPath = tmp.resolve("mmproj.gguf");
        Files.write(modelPath, llmGguf);
        Files.write(mmprojPath, visionGguf);

        assertThat(LlavaHandlerFactory.isVisionArchitecture(modelPath, mmprojPath)).isTrue();
    }

    // ── resolveImagePlaceholderString / resolveImageTokenId ───────────────────

    @Test
    @DisplayName("resolveImagePlaceholderString: token 50256 maps to <|endoftext|> (phi-2/moondream2 EOS)")
    void resolveImagePlaceholderString_50256_isEndOfText() throws Exception {
        java.lang.reflect.Method m = LlavaHandlerFactory.class
                .getDeclaredMethod("resolveImagePlaceholderString", int.class);
        m.setAccessible(true);
        String result = (String) m.invoke(null, 50256);
        assertThat(result).isEqualTo("<|endoftext|>");
    }

    @Test
    @DisplayName("resolveImagePlaceholderString: token 32000 maps to <image> (LLaVA/LLaMA default)")
    void resolveImagePlaceholderString_32000_isImage() throws Exception {
        java.lang.reflect.Method m = LlavaHandlerFactory.class
                .getDeclaredMethod("resolveImagePlaceholderString", int.class);
        m.setAccessible(true);
        String result = (String) m.invoke(null, 32000);
        assertThat(result).isEqualTo("<image>");
    }

    @Test
    @DisplayName("resolveImagePlaceholderString: any other token ID falls back to <image>")
    void resolveImagePlaceholderString_otherIds_fallbackToImage() throws Exception {
        java.lang.reflect.Method m = LlavaHandlerFactory.class
                .getDeclaredMethod("resolveImagePlaceholderString", int.class);
        m.setAccessible(true);
        assertThat((String) m.invoke(null, 12345)).isEqualTo("<image>");
        assertThat((String) m.invoke(null, 50257)).isEqualTo("<image>");
    }
}