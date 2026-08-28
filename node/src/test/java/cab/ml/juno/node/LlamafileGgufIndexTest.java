package cab.ml.juno.node;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@DisplayName("LlamafileGgufIndex — scans llamafile ZIP for all embedded GGUF entries")
class LlamafileGgufIndexTest {

    // ── helpers ───────────────────────────────────────────────────────────────

    /** Build a minimal single-tensor GGUF with the given tensor name (0 kv pairs). */
    static byte[] buildMinimalGgufBytes(String tensorName) {
        final int ALIGNMENT = 32;
        final int GGUF_MAGIC = 0x46554747;
        byte[] nameBytes = tensorName.getBytes(StandardCharsets.UTF_8);

        // 1 F32 element = 4 bytes of data
        float[] values = { 1.0f };
        byte[] data = new byte[values.length * 4];
        ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putFloat(values[0]);

        int headerSize = 24;
        int infoSize   = 8 + nameBytes.length + 4 + 8 + 4 + 8; // nameLen+name+ndims+dim+type+offset
        int prePad     = headerSize + infoSize;
        int aligned    = ((prePad + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        int padLen     = aligned - prePad;

        ByteBuffer buf = ByteBuffer.allocate(aligned + data.length).order(ByteOrder.LITTLE_ENDIAN);

        // Header
        buf.putInt(GGUF_MAGIC);
        buf.putInt(3);            // version
        buf.putLong(1);           // 1 tensor
        buf.putLong(0);           // 0 kv pairs

        // Tensor info
        buf.putLong(nameBytes.length);
        buf.put(nameBytes);
        buf.putInt(1);            // ndims = 1
        buf.putLong(values.length); // dim[0]
        buf.putInt(0);            // F32
        buf.putLong(0);           // offset within data section

        buf.put(new byte[padLen]);
        buf.put(data);
        return buf.array();
    }

    /**
     * Wrap one or more GGUF byte arrays inside a minimal ZIP polyglot
     * (APE stub + local headers + central directory + EOCD).
     * Entries are STORED (no compression). Entry filenames are auto-generated
     * as {@code entry0.gguf}, {@code entry1.gguf}, etc.
     */
    static Path buildMultiGgufLlamafile(Path dir, byte[]... ggufBlobs) throws IOException {
        byte[] stub = "MZqFpD\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
                .getBytes(StandardCharsets.ISO_8859_1);

        String[] names = new String[ggufBlobs.length];
        for (int i = 0; i < ggufBlobs.length; i++)
            names[i] = "entry" + i + ".gguf";

        // Compute layout
        long[] localHdrOffsets = new long[ggufBlobs.length];
        long cursor = stub.length;
        for (int i = 0; i < ggufBlobs.length; i++) {
            localHdrOffsets[i] = cursor;
            byte[] fn = names[i].getBytes(StandardCharsets.UTF_8);
            cursor += 30 + fn.length + ggufBlobs[i].length;
        }
        long cdOffset = cursor;

        // Build central directory entries
        byte[][] cdeArrays = new byte[ggufBlobs.length][];
        int cdTotal = 0;
        for (int i = 0; i < ggufBlobs.length; i++) {
            byte[] fn = names[i].getBytes(StandardCharsets.UTF_8);
            ByteBuffer cde = ByteBuffer.allocate(46 + fn.length).order(ByteOrder.LITTLE_ENDIAN);
            cde.putInt(0x02014b50);
            cde.putShort((short) 20); // version made by
            cde.putShort((short) 20); // version needed
            cde.putShort((short) 0);  // flags
            cde.putShort((short) 0);  // compression: STORED
            cde.putShort((short) 0);  // mod time
            cde.putShort((short) 0);  // mod date
            cde.putInt(0);            // CRC
            cde.putInt(ggufBlobs[i].length); // compressed size
            cde.putInt(ggufBlobs[i].length); // uncompressed size
            cde.putShort((short) fn.length);
            cde.putShort((short) 0);  // extra length
            cde.putShort((short) 0);  // comment length
            cde.putShort((short) 0);  // disk number start
            cde.putShort((short) 0);  // internal attributes
            cde.putInt(0);            // external attributes
            cde.putInt((int) localHdrOffsets[i]);
            cde.put(fn);
            cdeArrays[i] = cde.array();
            cdTotal += cdeArrays[i].length;
        }

        // EOCD (22 bytes)
        ByteBuffer eocd = ByteBuffer.allocate(22).order(ByteOrder.LITTLE_ENDIAN);
        eocd.putInt(0x06054b50);
        eocd.putShort((short) 0);  // disk number
        eocd.putShort((short) 0);  // start disk
        eocd.putShort((short) ggufBlobs.length);
        eocd.putShort((short) ggufBlobs.length);
        eocd.putInt(cdTotal);
        eocd.putInt((int) cdOffset);
        eocd.putShort((short) 0);  // comment length

        // Assemble
        int totalSize = (int)(cdOffset) + cdTotal + 22;
        ByteBuffer file = ByteBuffer.allocate(totalSize);
        file.put(stub);
        for (int i = 0; i < ggufBlobs.length; i++) {
            byte[] fn = names[i].getBytes(StandardCharsets.UTF_8);
            ByteBuffer lh = ByteBuffer.allocate(30 + fn.length).order(ByteOrder.LITTLE_ENDIAN);
            lh.putInt(0x04034b50);
            lh.putShort((short) 20);
            lh.putShort((short) 0);
            lh.putShort((short) 0);  // STORED
            lh.putShort((short) 0);
            lh.putShort((short) 0);
            lh.putInt(0);
            lh.putInt(ggufBlobs[i].length);
            lh.putInt(ggufBlobs[i].length);
            lh.putShort((short) fn.length);
            lh.putShort((short) 0);
            lh.put(fn);
            file.put(lh.array());
            file.put(ggufBlobs[i]);
        }
        for (byte[] cde : cdeArrays)
            file.put(cde);
        file.put(eocd.array());

        Path out = dir.resolve("test.llamafile");
        Files.write(out, file.array());
        return out;
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("scanAll returns empty list for a plain .gguf file")
    void scanAll_emptyForPlainGguf(@TempDir Path tmp) throws IOException {
        byte[] ggufBytes = buildMinimalGgufBytes("weight");
        Path gguf = tmp.resolve("model.gguf");
        Files.write(gguf, ggufBytes);

        List<LlamafileGgufIndex.Entry> entries = LlamafileGgufIndex.scanAll(gguf);

        assertThat(entries).isEmpty();
    }

    @Test
    @DisplayName("scanAll returns one entry for a llamafile with a single GGUF")
    void scanAll_singleEntryForSingleGgufLlamafile(@TempDir Path tmp) throws IOException {
        byte[] gguf = buildMinimalGgufBytes("token_embd.weight");
        Path llamafile = buildMultiGgufLlamafile(tmp, gguf);

        List<LlamafileGgufIndex.Entry> entries = LlamafileGgufIndex.scanAll(llamafile);

        assertThat(entries).hasSize(1);
        assertThat(entries.get(0).name()).isEqualTo("entry0.gguf");
        assertThat(entries.get(0).dataOffset()).isPositive();
    }

    @Test
    @DisplayName("scanAll returns two entries for a llamafile bundling two GGUFs (moondream2 pattern)")
    void scanAll_twoEntriesForDualGgufLlamafile(@TempDir Path tmp) throws IOException {
        byte[] textGguf    = buildMinimalGgufBytes("token_embd.weight");
        byte[] visionGguf  = buildMinimalGgufBytes("v.patch_embd.weight");
        Path llamafile = buildMultiGgufLlamafile(tmp, textGguf, visionGguf);

        List<LlamafileGgufIndex.Entry> entries = LlamafileGgufIndex.scanAll(llamafile);

        assertThat(entries).hasSize(2);
        assertThat(entries.get(0).name()).isEqualTo("entry0.gguf");
        assertThat(entries.get(1).name()).isEqualTo("entry1.gguf");
        // offsets must be strictly ordered (first entry comes before second in the file)
        assertThat(entries.get(0).dataOffset()).isLessThan(entries.get(1).dataOffset());
    }

    @Test
    @DisplayName("dataOffset returned by scanAll matches the offset GgufReader.openAtDataOffset requires")
    void dataOffset_readableByGgufReaderOpenAtDataOffset(@TempDir Path tmp) throws IOException {
        byte[] textGguf   = buildMinimalGgufBytes("token_embd.weight");
        byte[] visionGguf = buildMinimalGgufBytes("v.patch_embd.weight");
        Path llamafile = buildMultiGgufLlamafile(tmp, textGguf, visionGguf);

        List<LlamafileGgufIndex.Entry> entries = LlamafileGgufIndex.scanAll(llamafile);
        assertThat(entries).hasSize(2);

        // First entry — text model
        try (GgufReader r = GgufReader.openAtDataOffset(llamafile, entries.get(0).dataOffset())) {
            assertThat(r.hasTensor("token_embd.weight")).isTrue();
            assertThat(r.hasTensor("v.patch_embd.weight")).isFalse();
        }

        // Second entry — vision encoder
        try (GgufReader r = GgufReader.openAtDataOffset(llamafile, entries.get(1).dataOffset())) {
            assertThat(r.hasTensor("v.patch_embd.weight")).isTrue();
            assertThat(r.hasTensor("token_embd.weight")).isFalse();
        }
    }

    @Test
    @DisplayName("scanAll returns empty list for a non-ZIP, non-GGUF file")
    void scanAll_emptyForArbitraryBinaryFile(@TempDir Path tmp) throws IOException {
        Path garbage = tmp.resolve("garbage.bin");
        Files.write(garbage, new byte[]{ 0x00, 0x01, 0x02, 0x03, 0x04, 0x05 });

        List<LlamafileGgufIndex.Entry> entries = LlamafileGgufIndex.scanAll(garbage);

        assertThat(entries).isEmpty();
    }
}