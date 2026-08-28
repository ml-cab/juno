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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Scans a llamafile (ZIP polyglot) for <em>all</em> embedded GGUF data entries.
 *
 * <p>{@link GgufReader#open(Path)} only finds the <em>first</em> GGUF entry in a
 * llamafile ZIP. Some llamafiles bundle more than one model: moondream2, for
 * example, packages the phi-2 LLM and the SigLIP vision encoder as two
 * separate GGUF entries in the same ZIP. This class enumerates every
 * {@code .gguf} entry so callers can locate specific ones — e.g. the vision
 * encoder — without requiring a separate {@code --mmproj-path} file.
 *
 * <h3>Design notes</h3>
 * <ul>
 *   <li>Lives in {@code cab.ml.juno.node} (same package as {@link GgufReader})
 *       so it can access the package-private {@link GgufReader#GGUF_MAGIC} and
 *       {@link GgufReader#readZip64ExtraLocalOffset} constants.
 *   <li>Implements its own EOCD+CD walk rather than delegating to
 *       {@link GgufReader}'s private ZIP code — the two are parallel, not
 *       layered, by design. {@link GgufReader} remains a single-entry reader;
 *       multi-entry scanning is this class's sole responsibility.
 *   <li>The forward-scan fallback used by
 *       {@link GgufReader#findGgufOffsetInZip} (for llamafiles where the EOCD
 *       is beyond the last 65 KB) is <em>not</em> implemented here. If EOCD
 *       resolution fails, {@link #scanAll} returns an empty list. In practice
 *       all known multi-GGUF llamafiles (moondream2, etc.) use ZIP64 with an
 *       accessible EOCD.
 * </ul>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * List<LlamafileGgufIndex.Entry> entries = LlamafileGgufIndex.scanAll(path);
 * // entries.get(0) is the LLM text model (found by GgufReader.open too)
 * // entries.get(1) might be the vision encoder
 * for (LlamafileGgufIndex.Entry e : entries) {
 *     try (GgufReader r = GgufReader.openAtDataOffset(path, e.dataOffset())) {
 *         if (r.hasTensor("v.patch_embd.weight")) { ... }
 *     }
 * }
 * }</pre>
 */
public final class LlamafileGgufIndex {

    private static final Logger log = Logger.getLogger(LlamafileGgufIndex.class.getName());

    /** A single GGUF entry found inside a llamafile ZIP. */
    public record Entry(
            /** ZIP entry filename (e.g. {@code "mmproj.gguf"}). */
            String name,
            /** Absolute byte offset in the llamafile where the GGUF magic begins. */
            long dataOffset
    ) {}

    private LlamafileGgufIndex() {}

    /**
     * Return all GGUF data entries found in the ZIP central directory of
     * {@code path}, in declaration order.
     *
     * <p>Returns an empty list when:
     * <ul>
     *   <li>{@code path} is a plain {@code .gguf} file (starts with GGUF magic)
     *   <li>The file is not a ZIP (not a llamafile)
     *   <li>EOCD resolution fails (malformed ZIP or EOCD beyond final 65 KB)
     * </ul>
     *
     * @param path path to a llamafile or plain .gguf
     * @return immutable list of entries; never {@code null}
     */
    public static List<Entry> scanAll(Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            // Plain .gguf files start with GGUF magic — no ZIP to scan.
            ByteBuffer magic4 = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            ch.read(magic4, 0);
            magic4.flip();
            if (magic4.limit() >= 4 && magic4.getInt(0) == GgufReader.GGUF_MAGIC) {
                return List.of();
            }

            try {
                return scanZipCentralDirectory(ch);
            } catch (IOException e) {
                log.fine("LlamafileGgufIndex: ZIP scan failed for " + path.getFileName()
                        + " — " + e.getMessage());
                return List.of();
            }
        }
    }

    /**
     * Walk the ZIP central directory and collect all entries whose filename
     * ends with {@code .gguf} and whose data starts with GGUF magic.
     */
    private static List<Entry> scanZipCentralDirectory(FileChannel ch) throws IOException {
        long fileSize = ch.size();

        // ── Step 1: locate EOCD ───────────────────────────────────────────────
        // EOCD is 22 bytes + optional comment (max 65535 bytes).
        int searchLen = (int) Math.min(fileSize, 65535 + 22);
        long searchStart = fileSize - searchLen;

        ByteBuffer tail = ByteBuffer.allocate(searchLen).order(ByteOrder.LITTLE_ENDIAN);
        while (tail.hasRemaining()) {
            int r = ch.read(tail, searchStart + tail.position());
            if (r < 0) break;
        }
        tail.flip();
        int actualLen = tail.limit();

        int eocdIdx = -1;
        for (int i = actualLen - 22; i >= 0; i--) {
            if ((tail.getInt(i) & 0xFFFFFFFFL) != 0x06054b50L)
                continue;
            long candidateCdSize   = tail.getInt(i + 12) & 0xFFFFFFFFL;
            long candidateCdOffset = tail.getInt(i + 16) & 0xFFFFFFFFL;
            long eocdAbsPos        = searchStart + i;
            boolean zip64 = (candidateCdSize == 0xFFFFFFFFL || candidateCdOffset == 0xFFFFFFFFL);
            if (!zip64) {
                if (candidateCdSize == 0)              continue;
                if (candidateCdSize > eocdAbsPos)      continue;
                if (candidateCdOffset + candidateCdSize > eocdAbsPos) continue;
                if (candidateCdOffset >= fileSize)     continue;
            }
            eocdIdx = i;
            break;
        }

        if (eocdIdx < 0)
            throw new IOException("EOCD not found in final 65 KB — not a ZIP or EOCD out of range");

        // ── Step 1b: resolve ZIP64 offsets if needed ──────────────────────────
        long rawCdSize   = tail.getInt(eocdIdx + 12) & 0xFFFFFFFFL;
        long rawCdOffset = tail.getInt(eocdIdx + 16) & 0xFFFFFFFFL;
        long cdSize;
        long cdOffset;

        if (rawCdSize == 0xFFFFFFFFL || rawCdOffset == 0xFFFFFFFFL) {
            long eocdAbsPos = searchStart + eocdIdx;
            long locatorPos = eocdAbsPos - 20;
            if (locatorPos < 0)
                throw new IOException("ZIP64 EOCD locator would be before start of file");

            ByteBuffer loc = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
            while (loc.hasRemaining()) {
                int r = ch.read(loc, locatorPos + loc.position());
                if (r < 0) break;
            }
            loc.flip();
            if (loc.limit() < 20 || (loc.getInt(0) & 0xFFFFFFFFL) != 0x07064b50L)
                throw new IOException("ZIP64 EOCD locator signature not found at " + locatorPos);

            long z64EocdAbsPos = loc.getLong(8);
            ByteBuffer z64 = ByteBuffer.allocate(56).order(ByteOrder.LITTLE_ENDIAN);
            while (z64.hasRemaining()) {
                int r = ch.read(z64, z64EocdAbsPos + z64.position());
                if (r < 0) break;
            }
            z64.flip();
            if (z64.limit() < 56 || (z64.getInt(0) & 0xFFFFFFFFL) != 0x06064b50L)
                throw new IOException("ZIP64 EOCD record signature not found at " + z64EocdAbsPos);

            cdSize   = z64.getLong(40);
            cdOffset = z64.getLong(48);
        } else {
            cdSize   = rawCdSize;
            cdOffset = rawCdOffset;
        }

        if (cdSize == 0 || cdOffset >= fileSize)
            throw new IOException("ZIP central directory is empty or out of range");

        // ── Step 2: read the central directory ────────────────────────────────
        ByteBuffer cd = ByteBuffer.allocate((int) cdSize).order(ByteOrder.LITTLE_ENDIAN);
        while (cd.hasRemaining()) {
            int r = ch.read(cd, cdOffset + cd.position());
            if (r < 0) break;
        }
        cd.flip();

        // ── Step 3: walk all entries, collect those whose data is GGUF ────────
        List<Entry> result = new ArrayList<>();
        int cdPos = 0;

        while (cdPos + 46 <= cd.limit()) {
            long sig = cd.getInt(cdPos) & 0xFFFFFFFFL;
            if (sig != 0x02014b50L)
                break; // end of CD

            int fnLen      = cd.getShort(cdPos + 28) & 0xFFFF;
            int extraLen   = cd.getShort(cdPos + 30) & 0xFFFF;
            int commentLen = cd.getShort(cdPos + 32) & 0xFFFF;
            long localHdrOffset = cd.getInt(cdPos + 42) & 0xFFFFFFFFL;

            int nextEntry = cdPos + 46 + fnLen + extraLen + commentLen;
            if (nextEntry > cd.limit())
                break;

            byte[] fnBytes = new byte[fnLen];
            cd.position(cdPos + 46);
            cd.get(fnBytes);
            String filename = new String(fnBytes, StandardCharsets.UTF_8);

            if (filename.endsWith(".gguf")) {
                // Resolve ZIP64 sentinel for local header offset
                if (localHdrOffset == 0xFFFFFFFFL) {
                    localHdrOffset = GgufReader.readZip64ExtraLocalOffset(
                            cd, cdPos + 46 + fnLen, extraLen);
                }

                // Read local file header to get actual data start
                ByteBuffer lh = ByteBuffer.allocate(30).order(ByteOrder.LITTLE_ENDIAN);
                while (lh.hasRemaining()) {
                    int r = ch.read(lh, localHdrOffset + lh.position());
                    if (r < 0) break;
                }
                lh.flip();

                if (lh.limit() >= 30 && (lh.getInt(0) & 0xFFFFFFFFL) == 0x04034b50L) {
                    int localFnLen    = lh.getShort(26) & 0xFFFF;
                    int localExtraLen = lh.getShort(28) & 0xFFFF;
                    long dataStart = localHdrOffset + 30L + localFnLen + localExtraLen;

                    // Confirm GGUF magic at the computed data start
                    ByteBuffer ggufMagic = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                    int read = ch.read(ggufMagic, dataStart);
                    ggufMagic.flip();
                    if (read == 4 && ggufMagic.getInt(0) == GgufReader.GGUF_MAGIC) {
                        log.fine("LlamafileGgufIndex: found GGUF entry \"" + filename
                                + "\" at dataOffset=" + dataStart);
                        result.add(new Entry(filename, dataStart));
                    }
                }
            }

            cdPos = nextEntry;
        }

        return Collections.unmodifiableList(result);
    }
}