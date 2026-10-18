#include <lpl/zarr/Array.hpp>
#include <lpl/zarr/FileStore.hpp>

#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

int gChecks = 0;
int gFailures = 0;

void check(bool ok, const char *what)
{
    ++gChecks;
    if (!ok)
    {
        ++gFailures;
        std::printf("  FAIL  %s\n", what);
    }
}

void checkEq(long long got, long long want, const char *what)
{
    ++gChecks;
    if (got != want)
    {
        ++gFailures;
        std::printf("  FAIL  %s: got %lld, want %lld\n", what, got, want);
    }
}

/** PHerc0358, level 0. */
const char *kRawArray = R"({
  "shape": [ 14744, 7783, 7783 ],
  "chunks": [ 128, 128, 128 ],
  "dtype": "|u1",
  "fill_value": 0,
  "order": "C",
  "filters": null,
  "dimension_separator": "/",
  "compressor": null,
  "zarr_format": 2
})";

/** PHerc0172, level 0. */
const char *kBloscArray =
    R"({"chunks":[128,128,128],"compressor":{"blocksize":0,"clevel":3,"cname":"zstd","id":"blosc","shuffle":1},)"
    R"("dimension_separator":"/","dtype":"|u1","fill_value":0,"filters":null,"order":"C",)"
    R"("shape":[21000,6700,9100],"zarr_format":2})";

/** Raw, with the OTHER separator the corpus uses and a non-zero fill value. */
const char *kDotArray =
    R"({"chunks":[128,128,128],"compressor":null,"dimension_separator":".","dtype":"|u1","fill_value":7,)"
    R"("filters":null,"order":"C","shape":[2048,2048,2048],"zarr_format":2})";

const char *kUnknownCodec =
    R"({"chunks":[64,64],"compressor":{"id":"lzma"},"dtype":"|u1","fill_value":0,"filters":null,)"
    R"("order":"C","shape":[128,128],"zarr_format":2})";

const char *kMultiscale = R"({"multiscales":[{"axes":[{"name":"z"},{"name":"y"},{"name":"x"}],
  "datasets":[{"path":"0"},{"path":"1"},{"path":"2"},{"path":"3"},{"path":"4"},{"path":"5"}]}]})";

const char *kBloscChunks64 =
    R"({"chunks":[64],"compressor":{"blocksize":0,"clevel":5,"cname":"lz4","id":"blosc","shuffle":0},)"
    R"("dtype":"|u1","fill_value":0,"filters":null,"order":"C","shape":[128],"zarr_format":2})";

bool parses(const char *document, lpl::zarr::ArrayMeta &meta)
{
    return lpl::zarr::parseArrayMeta(document, std::strlen(document), meta);
}

/** The 16-byte header blosc writes in front of every chunk. */
struct BloscHeader final {
    lpl::core::u8 formatVersion{2u};
    lpl::core::u8 codecVersion{1u};
    lpl::core::u8 flags{0x02u}; /**< BLOSC_MEMCPYED: the samples follow the header as they are. */
    lpl::core::u8 typeSize{1u};
    lpl::core::u32 uncompressedBytes{0u};
    lpl::core::u32 blockBytes{0u};
    lpl::core::u32 compressedBytes{0u};
};
static_assert(sizeof(BloscHeader) == 16u);
static_assert(std::endian::native == std::endian::little, "blosc writes its header little-endian");

/**
 * @brief A blosc chunk whose header announces @p announcedBytes and whose body carries @p bodyBytes of them.
 */
std::vector<lpl::core::u8> storedBloscChunk(lpl::core::u8 sampleValue, lpl::core::u32 announcedBytes,
                                            lpl::core::u32 bodyBytes)
{
    const BloscHeader header{.uncompressedBytes = announcedBytes,
                             .blockBytes = announcedBytes,
                             .compressedBytes = static_cast<lpl::core::u32>(sizeof(BloscHeader)) + announcedBytes};
    std::vector<lpl::core::u8> chunk(sizeof(BloscHeader) + bodyBytes, sampleValue);
    std::memcpy(chunk.data(), &header, sizeof(header));
    return chunk;
}

/** Serves one value for every key, or nothing when it holds none. */
struct MemoryStore final : lpl::zarr::IZarrStore {
    std::vector<lpl::core::u8> value;

    [[nodiscard]] lpl::zarr::FetchResult read(const char *, std::span<lpl::core::u8> buffer) noexcept override
    {
        if (value.empty())
            return lpl::zarr::FetchResult{lpl::zarr::FetchStatus::Absent, 0u};
        if (value.size() > buffer.size())
            return lpl::zarr::FetchResult{lpl::zarr::FetchStatus::TooLarge, value.size()};
        std::memcpy(buffer.data(), value.data(), value.size());
        return lpl::zarr::FetchResult{lpl::zarr::FetchStatus::Ok, value.size()};
    }

    [[nodiscard]] const char *name() const noexcept override { return "MemoryStore"; }
};

/**
 * @brief What would otherwise be misread rather than refused.
 */
void checkWhatWouldOtherwiseBeMisreadIsRefused(const lpl::zarr::ArrayMeta &raw)
{
    lpl::zarr::ArrayMeta bad{};
    check(!parses(R"({"chunks":[0],"compressor":null,"dtype":"|u1","order":"C","shape":[8]})", bad),
          "a chunk edge of zero is refused");
    checkEq(bad.chunkCount(0), 0, "and what was read before the refusal counts no chunk, not a division by zero");
    check(!parses(R"({"chunks":[-1,-1],"compressor":null,"dtype":"|u1","order":"C","shape":[8,8]})", bad),
          "a negative chunk edge is refused");
    check(!parses(R"({"chunks":[8],"compressor":null,"dtype":"|u1","order":"C","shape":[-8]})", bad),
          "a negative shape is refused");
    check(!parses(R"({"chunks":[4294967296,4294967296,4294967296],"compressor":null,"dtype":"|u1",)"
                  R"("order":"C","shape":[4294967296,4294967296,4294967296]})",
                  bad),
          "a chunk whose byte count overflows is refused");
    check(!parses(R"({"chunks":[1],"compressor":null,"dtype":"|u1","order":"C","shape":[99999999999999999999]})", bad),
          "a number too long for 64 bits is refused, not wrapped");

    const char *cutAfterMinus = R"({"chunks":[1],"compressor":null,"dtype":"|u1","order":"C","shape":[-9]})";
    const auto cutLength = static_cast<lpl::core::usize>(std::strstr(cutAfterMinus, "-9") + 1 - cutAfterMinus);
    check(!lpl::zarr::parseArrayMeta(cutAfterMinus, cutLength, bad),
          "a document cut after a minus sign is refused, not read past its end");

    check(!parses(R"({"chunks":[8],"compressor":null,"dtype":"|u1","fill_value":0,)"
                  R"("filters":[{"dtype":"|u1","id":"delta"}],"order":"C","shape":[8],"zarr_format":2})",
                  bad),
          "an array that needs a filter is refused, not read raw");
    check(!parses(R"({"chunks":[8],"compressor":{"cname":"zstd"},"dtype":"|u1","filters":null,"order":"C",)"
                  R"("shape":[8],"storage":{"id":"zstd"}})",
                  bad),
          "the codec is read inside the compressor, never from a key elsewhere");
    check(!parses(R"({"chunks":[8,8],"compressor":null,"dimension_separator":"x","dtype":"|u1","order":"C",)"
                  R"("shape":[8,8]})",
                  bad),
          "a separator other than '.' or '/' is refused");
    checkEq(static_cast<long long>(bad.chunkBytes()), 0, "and metadata that is not valid sizes no chunk");
    check(!parses(R"({"chunks":[8],"compressor":null,"dtype":"|u1","fill_value":300,"order":"C","shape":[8]})", bad),
          "an unsigned fill value above 255 is refused, not clamped");

    lpl::zarr::ArrayMeta signedBytes{};
    check(parses(R"({"chunks":[8],"compressor":null,"dtype":"|i1","fill_value":-1,"order":"C","shape":[8]})",
                 signedBytes),
          "a signed byte array parses");
    checkEq(signedBytes.fillValue, 0xFF, "and its fill value -1 is the byte 0xFF");

    check(!parses(R"({"chunks":[8],"compressor":null,"dtype":"|u1","order":"","shape":[8]})", bad),
          "an empty order is refused, not read as absent");
    check(!parses(R"({"chunks":[8],"compressor":null,"dtype":"|u1","order":"FFFFFFFF","shape":[8]})", bad),
          "and so is an order too long to be one letter");

    lpl::zarr::ArrayMeta reused = raw;
    check(!lpl::zarr::parseArrayMeta(nullptr, 0u, reused), "a null document is refused");
    checkEq(reused.dimensions, 0, "and leaves nothing of a previous parse behind");
}

/**
 * @brief A blosc chunk cut short.
 *
 * @details When the chunk cut short is read, the scratch still holds the whole chunk read before it.
 */
void checkBloscChunkCutShortFails()
{
    if (!lpl::zarr::codecAvailable(lpl::zarr::Codec::Blosc))
    {
        std::printf("  note: this build has no blosc; the truncated-chunk checks do not run\n");
    }
    else
    {
        lpl::zarr::ArrayMeta packed{};
        check(parses(kBloscChunks64, packed), "a blosc array of 64-byte chunks parses");

        MemoryStore store;
        std::vector<lpl::core::u8> scratch(256u, 0u);
        std::vector<lpl::core::u8> out(64u, 0u);
        store.value = storedBloscChunk(0x11u, 64u, 64u);
        lpl::zarr::FetchResult r = lpl::zarr::readChunk(store, packed, "0", out, scratch);
        check(r.ok() && out[0] == 0x11u && out[63] == 0x11u, "a whole blosc chunk decodes");

        store.value = storedBloscChunk(0x22u, 64u, 10u);
        r = lpl::zarr::readChunk(store, packed, "1", out, scratch);
        check(r.status == lpl::zarr::FetchStatus::Failed,
              "a blosc chunk cut short fails, not decoded from stale bytes");
    }
}

/**
 * @brief The store.
 */
void checkTheFileStore()
{
    char root[256];
    std::snprintf(root, sizeof(root), "/tmp/lpl-zarr-test-%d", static_cast<int>(::getpid()));
    lpl::zarr::FileStore store(root);
    check(store.valid(), "a store opens on a root");

    std::vector<lpl::core::u8> payload(4096u, 0xABu);
    check(store.write("0/1/2/3", payload), "a nested key writes, creating its parents");

    std::vector<lpl::core::u8> back(8192u, 0u);
    lpl::zarr::FetchResult r = store.read("0/1/2/3", back);
    check(r.ok(), "and reads back");
    checkEq(static_cast<long long>(r.size), 4096, "at the size written");
    check(back[0] == 0xABu && back[4095] == 0xABu, "with the bytes written");

    r = store.read("0/1/2/9999", back);
    check(r.status == lpl::zarr::FetchStatus::Absent, "a key that is not there is absent, not failed");

    r = store.read("0/1/2/3", std::span(back).first(16u));
    check(r.status == lpl::zarr::FetchStatus::TooLarge, "a buffer that does not fit says so");
    checkEq(static_cast<long long>(r.size), 4096, "and reports what it would need");

    r = store.read("../../etc/passwd", back);
    check(r.status == lpl::zarr::FetchStatus::Failed, "a key that climbs out of the root is refused");

    r = store.read("0/1/2", back);
    check(r.status == lpl::zarr::FetchStatus::Failed, "a key that names a directory fails, it is not a value");

    const std::string nameTooLong(300u, 'x');
    r = store.read(nameTooLong.c_str(), back);
    check(r.status == lpl::zarr::FetchStatus::Failed, "a key the filesystem cannot look up fails, it is not absent");

    std::error_code removal;
    std::filesystem::remove_all(root, removal);
    check(!removal, "the test removes the directory it wrote");
}

/**
 * @brief An absent chunk is fill, not a hole.
 */
void checkAbsentChunkIsFillNotAHole()
{
    lpl::zarr::ArrayMeta small{};
    const char *doc = R"({"chunks":[4,4,4],"compressor":null,"dimension_separator":"/","dtype":"|u1","fill_value":42,)"
                      R"("filters":null,"order":"C","shape":[8,8,8],"zarr_format":2})";
    check(parses(doc, small), "the small fixture parses");

    char root[256];
    std::snprintf(root, sizeof(root), "/tmp/lpl-zarr-fill-%d", static_cast<int>(::getpid()));
    lpl::zarr::FileStore store(root);
    std::vector<lpl::core::u8> out(small.chunkBytes(), 0u);
    const lpl::zarr::FetchResult r = lpl::zarr::readChunk(store, small, "0/0/0/0", out, {});
    check(r.ok(), "an absent chunk is a successful read");
    check(r.filled, "and says so: successful, and yet not fetched");
    checkEq(static_cast<long long>(r.size), static_cast<long long>(small.chunkBytes()), "of a whole chunk");
    bool allFill = true;
    for (lpl::core::u8 v : out)
    {
        if (v != 42u)
            allFill = false;
    }
    check(allFill, "filled with the declared fill value");
}

} // namespace

/**
 * @brief Reading a zarr array's own description, and the ways that goes silently wrong.
 *
 * @warning The first two `.zarray` documents in this file are **verbatim from real published volumes**, not
 * written for this test: one uncompressed with `/` separators, the other blosc with `/`. Every other
 * document is one of them edited, or written by hand, to isolate a single property. A fixture
 * invented whole would agree with whatever the parser happens to do.
 */
int main()
{
    std::printf("zarr array\n");

    lpl::zarr::ArrayMeta raw{};
    {
        check(parses(kRawArray, raw), "a real raw .zarray parses");
        checkEq(raw.dimensions, 3, "three dimensions");
        checkEq(raw.shape[0], 14744, "shape z");
        checkEq(raw.shape[1], 7783, "shape y");
        checkEq(raw.shape[2], 7783, "shape x");
        checkEq(raw.chunks[0], 128, "chunk edge");
        check(raw.codec == lpl::zarr::Codec::Raw, "a null compressor is the raw path");
        checkEq(raw.separator, '/', "the separator is read from the document");
        checkEq(static_cast<long long>(raw.chunkBytes()), 128LL * 128 * 128, "a chunk is two mebibytes");
        checkEq(raw.chunkCount(0), (14744 + 127) / 128, "chunk count rounds up, as zarr does");
    }

    lpl::zarr::ArrayMeta blosc{};
    {
        const bool opened = parses(kBloscArray, blosc);
        check(blosc.codec == lpl::zarr::Codec::Blosc, "a real blosc .zarray names blosc");
        check(opened == lpl::zarr::codecAvailable(lpl::zarr::Codec::Blosc),
              "and opens exactly when this build has blosc");
        checkEq(blosc.shape[0], 21000, "shape survives a document with no whitespace");
        checkEq(blosc.separator, '/', "and so does the separator");
    }

    {
        lpl::zarr::ArrayMeta dotted{};
        check(parses(kDotArray, dotted), "the dotted variant parses");
        checkEq(dotted.separator, '.', "the other separator is read too, not assumed");
        checkEq(dotted.fillValue, 7, "a non-zero fill value is carried");

        const lpl::core::i64 index[3]{12, 3, 4};
        char key[lpl::zarr::kMaxKeyLength];
        check(lpl::zarr::chunkKey(dotted, "2", index, key), "a key is built");
        check(std::strcmp(key, "2/12.3.4") == 0, "with the level prefix and the declared separator");
        check(lpl::zarr::chunkKey(raw, "0", index, key), "and for the slashed array");
        check(std::strcmp(key, "0/12/3/4") == 0, "the same indices spell a different key");
        check(lpl::zarr::chunkKey(raw, nullptr, index, key), "a bare array needs no prefix");
        check(std::strcmp(key, "12/3/4") == 0, "and gets none");

        const lpl::core::i64 pastLastChunk[3]{16, 3, 4};
        check(!lpl::zarr::chunkKey(dotted, "2", pastLastChunk, key), "an index past the last chunk builds no key");
        const lpl::core::i64 negative[3]{-1, 3, 4};
        check(!lpl::zarr::chunkKey(dotted, "2", negative, key), "nor does a negative one");
    }

    {
        lpl::zarr::ArrayMeta bad{};
        check(!parses(kUnknownCodec, bad), "an unknown compressor makes the array refuse to open");
        check(!parses(R"({"chunks":[128,128,128],"dtype":"|u1")", bad), "a document with no shape is refused");
        check(!parses(R"({"chunks":[8,8],"compressor":null,"dtype":"<u2","fill_value":0,"order":"C","shape":[8,8],)"
                      R"("zarr_format":2})",
                      bad),
              "a sample width this reader does not handle is refused, not truncated");
        check(!parses(R"({"chunks":[8,8],"compressor":null,"dtype":"|u1","fill_value":0,"order":"F","shape":[8,8],)"
                      R"("zarr_format":2})",
                      bad),
              "Fortran order is refused, not transposed");
    }

    checkWhatWouldOtherwiseBeMisreadIsRefused(raw);

    {
        MemoryStore empty;
        std::vector<lpl::core::u8> out(64u, 0u);
        const lpl::zarr::FetchResult r = lpl::zarr::readChunk(empty, lpl::zarr::ArrayMeta{}, "0", out, {});
        check(r.status == lpl::zarr::FetchStatus::Failed, "metadata that never parsed reads nothing");
    }

    {
        lpl::zarr::ArrayMeta eightBytes{};
        check(parses(R"({"chunks":[8],"compressor":null,"dtype":"|u1","order":"C","shape":[16]})", eightBytes),
              "a raw array of 8-byte chunks parses");
        MemoryStore store;
        store.value.assign(16u, 0x33u);
        std::vector<lpl::core::u8> out(8u, 0u);
        const lpl::zarr::FetchResult r = lpl::zarr::readChunk(store, eightBytes, "0", out, {});
        check(r.status == lpl::zarr::FetchStatus::Failed,
              "a raw chunk stored longer than one chunk fails, since no buffer would help");
    }

    checkBloscChunkCutShortFails();

    checkEq(lpl::zarr::countMultiscaleLevels(kMultiscale, std::strlen(kMultiscale)), 6, "the pyramid depth is read");
    checkEq(lpl::zarr::countMultiscaleLevels(kRawArray, std::strlen(kRawArray)), 0,
            "a plain array declares no pyramid, and says so");
    const char *twoGroups = R"({"multiscales":[{"datasets":[{"path":"0"},{"path":"1"}]},{"datasets":[{"path":"0"}]}]})";
    checkEq(lpl::zarr::countMultiscaleLevels(twoGroups, std::strlen(twoGroups)), 2,
            "only the first multiscale group is the pyramid");

    checkTheFileStore();
    checkAbsentChunkIsFillNotAHole();

    std::printf("%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
