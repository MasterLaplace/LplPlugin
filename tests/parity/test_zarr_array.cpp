/**
 * @file test_zarr_array.cpp
 * @brief Reading a zarr array's own description, and the two ways that goes silently wrong.
 *
 * @warning The `.zarray` documents below are **verbatim from real published volumes**, not written
 * for this test. One is uncompressed with `/` separators, the other blosc with `/`; a third is
 * edited to use `.` because the same corpus mixes both. A fixture invented here would agree with
 * whatever the parser happens to do.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/zarr/Array.hpp>
#include <lpl/zarr/FileStore.hpp>

#include <cstdio>
#include <cstring>
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

using namespace lpl;

// PHerc0358, level 0. Uncompressed, and that is not a curiosity: some published volumes really do
// ship raw, which is the fastest path this reader has.
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

// PHerc0172, level 0.
const char *kBloscArray =
    R"({"chunks":[128,128,128],"compressor":{"blocksize":0,"clevel":3,"cname":"zstd","id":"blosc","shuffle":1},)"
    R"("dimension_separator":"/","dtype":"|u1","fill_value":0,"filters":null,"order":"C",)"
    R"("shape":[21000,6700,9100],"zarr_format":2})";

// The same, with the OTHER separator the corpus uses.
const char *kDotArray =
    R"({"chunks":[128,128,128],"compressor":null,"dimension_separator":".","dtype":"|u1","fill_value":7,)"
    R"("filters":null,"order":"C","shape":[256,256,256],"zarr_format":2})";

const char *kUnknownCodec =
    R"({"chunks":[64,64],"compressor":{"id":"lzma"},"dtype":"|u1","fill_value":0,"filters":null,)"
    R"("order":"C","shape":[128,128],"zarr_format":2})";

const char *kMultiscale = R"({"multiscales":[{"axes":[{"name":"z"},{"name":"y"},{"name":"x"}],
  "datasets":[{"path":"0"},{"path":"1"},{"path":"2"},{"path":"3"},{"path":"4"},{"path":"5"}]}]})";

} // namespace

int main()
{
    std::printf("zarr array\n");

    zarr::ArrayMeta raw{};
    {
        check(zarr::parseArrayMeta(kRawArray, std::strlen(kRawArray), raw), "a real raw .zarray parses");
        checkEq(raw.dimensions, 3, "three dimensions");
        checkEq(raw.shape[0], 14744, "shape z");
        checkEq(raw.shape[1], 7783, "shape y");
        checkEq(raw.shape[2], 7783, "shape x");
        checkEq(raw.chunks[0], 128, "chunk edge");
        check(raw.codec == zarr::Codec::Raw, "a null compressor is the raw path");
        checkEq(raw.separator, '/', "the separator is read from the document");
        checkEq(static_cast<long long>(raw.chunkBytes()), 128LL * 128 * 128, "a chunk is two mebibytes");
        checkEq(raw.chunkCount(0), (14744 + 127) / 128, "chunk count rounds up, as zarr does");
    }

    zarr::ArrayMeta blosc{};
    {
        check(zarr::parseArrayMeta(kBloscArray, std::strlen(kBloscArray), blosc), "a real blosc .zarray parses");
        check(blosc.codec == zarr::Codec::Blosc || blosc.codec == zarr::Codec::Unsupported,
              "the compressor is recognised as blosc, or refused when this build lacks it");
        if (blosc.codec == zarr::Codec::Unsupported)
            std::printf("  note: this build has no blosc; the compressed path is refused, not guessed\n");
        checkEq(blosc.shape[0], 21000, "shape survives a document with no whitespace");
        checkEq(blosc.separator, '/', "and so does the separator");
    }

    {
        zarr::ArrayMeta dotted{};
        check(zarr::parseArrayMeta(kDotArray, std::strlen(kDotArray), dotted), "the dotted variant parses");
        // The trap this field exists for: the same corpus mixes '/' and '.', and hard-coding
        // either produces a request for a key that does not exist -- which a reader counts as an
        // absent chunk, so a whole volume comes back reported as empty.
        checkEq(dotted.separator, '.', "the other separator is read too, not assumed");
        checkEq(dotted.fillValue, 7, "a non-zero fill value is carried");

        const core::i64 index[3]{12, 3, 4};
        char key[zarr::kMaxKeyLength];
        check(zarr::chunkKey(dotted, "2", index, key), "a key is built");
        check(std::strcmp(key, "2/12.3.4") == 0, "with the level prefix and the declared separator");
        check(zarr::chunkKey(raw, "0", index, key), "and for the slashed array");
        check(std::strcmp(key, "0/12/3/4") == 0, "the same indices spell a different key");
        check(zarr::chunkKey(raw, nullptr, index, key), "a bare array needs no prefix");
        check(std::strcmp(key, "12/3/4") == 0, "and gets none");
    }

    {
        zarr::ArrayMeta bad{};
        // Refused rather than approximated: decoding with the wrong codec yields bytes, and bytes
        // render, so a viewer would show structure that is entirely an artefact of the mistake.
        check(!zarr::parseArrayMeta(kUnknownCodec, std::strlen(kUnknownCodec), bad),
              "an unknown compressor makes the array refuse to open");

        const char *truncated = R"({"chunks":[128,128,128],"dtype":"|u1")";
        check(!zarr::parseArrayMeta(truncated, std::strlen(truncated), bad), "a document with no shape is refused");

        const char *wideSample =
            R"({"chunks":[8,8],"compressor":null,"dtype":"<u2","fill_value":0,"order":"C","shape":[8,8],"zarr_format":2})";
        check(!zarr::parseArrayMeta(wideSample, std::strlen(wideSample), bad),
              "a sample width this reader does not handle is refused, not truncated");

        const char *fortran =
            R"({"chunks":[8,8],"compressor":null,"dtype":"|u1","fill_value":0,"order":"F","shape":[8,8],"zarr_format":2})";
        check(!zarr::parseArrayMeta(fortran, std::strlen(fortran), bad), "Fortran order is refused, not transposed");
    }

    checkEq(zarr::countMultiscaleLevels(kMultiscale, std::strlen(kMultiscale)), 6, "the pyramid depth is read");
    checkEq(zarr::countMultiscaleLevels(kRawArray, std::strlen(kRawArray)), 0,
            "a plain array declares no pyramid, and says so");

    // ── the store ──────────────────────────────────────────────────────────
    {
        char root[256];
        std::snprintf(root, sizeof(root), "/tmp/lpl-zarr-test-%d", static_cast<int>(::getpid()));
        zarr::FileStore store(root);
        check(store.valid(), "a store opens on a root");

        std::vector<core::u8> payload(4096u, 0xABu);
        check(store.write("0/1/2/3", payload.data(), payload.size()), "a nested key writes, creating its parents");

        std::vector<core::u8> back(8192u, 0u);
        zarr::FetchResult r = store.read("0/1/2/3", back.data(), back.size());
        check(r.ok(), "and reads back");
        checkEq(static_cast<long long>(r.size), 4096, "at the size written");
        check(back[0] == 0xABu && back[4095] == 0xABu, "with the bytes written");

        r = store.read("0/1/2/9999", back.data(), back.size());
        check(r.status == zarr::FetchStatus::Absent, "a key that is not there is absent, not failed");

        r = store.read("0/1/2/3", back.data(), 16u);
        check(r.status == zarr::FetchStatus::TooLarge, "a buffer that does not fit says so");
        checkEq(static_cast<long long>(r.size), 4096, "and reports what it would need");

        // Keys arrive from metadata, and metadata can arrive from a network.
        r = store.read("../../etc/passwd", back.data(), back.size());
        check(r.status == zarr::FetchStatus::Failed, "a key that climbs out of the root is refused");
    }

    // ── an absent chunk is fill, not a hole ────────────────────────────────
    {
        zarr::ArrayMeta small{};
        const char *doc =
            R"({"chunks":[4,4,4],"compressor":null,"dimension_separator":"/","dtype":"|u1","fill_value":42,)"
            R"("filters":null,"order":"C","shape":[8,8,8],"zarr_format":2})";
        check(zarr::parseArrayMeta(doc, std::strlen(doc), small), "the small fixture parses");

        char root[256];
        std::snprintf(root, sizeof(root), "/tmp/lpl-zarr-fill-%d", static_cast<int>(::getpid()));
        zarr::FileStore store(root);
        std::vector<core::u8> out(small.chunkBytes(), 0u);
        // Absence is a normal answer in this format: a chunk that is entirely fill value is simply
        // not stored. A reader that called it an error would report most sparse volumes as broken.
        const zarr::FetchResult r = zarr::readChunk(store, small, "0/0/0/0", out.data(), out.size(), nullptr, 0u);
        check(r.ok(), "an absent chunk is a successful read");
        check(r.filled, "and says so: successful, and yet not fetched");
        checkEq(static_cast<long long>(r.size), static_cast<long long>(small.chunkBytes()), "of a whole chunk");
        bool allFill = true;
        for (core::u8 v : out)
        {
            if (v != 42u)
                allFill = false;
        }
        check(allFill, "filled with the declared fill value");
    }

    std::printf("%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
