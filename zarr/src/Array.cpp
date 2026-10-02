/**
 * @file Array.cpp
 * @brief Parsing `.zarray`, building chunk keys, decoding chunks.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/zarr/Array.hpp>
#include <lpl/zarr/Codec.hpp>

namespace lpl::zarr {

namespace {

/**
 * A scanner, not a JSON parser, and the difference is stated rather than hidden.
 *
 * `.zarray` is machine-written from a fixed schema: a flat object of known keys with numbers,
 * strings, null and one nested object. Finding a key and reading the value after it is enough for
 * exactly that, and pulling in a general parser to read six fields would be a dependency the
 * kernel-adjacent half of this tree cannot take. What this will NOT survive is a hand-edited
 * document with the same key twice, or a string value containing the text of another key -- both
 * of which a real writer never produces.
 */
struct Scanner final {
    const char *begin{nullptr};
    const char *end{nullptr};

    [[nodiscard]] static bool matches(const char *p, const char *stop, const char *needle) noexcept
    {
        for (; *needle != '\0'; ++needle, ++p)
        {
            if (p >= stop || *p != *needle)
                return false;
        }
        return true;
    }

    /// @return Position just past `"key"` and its colon, or nullptr.
    [[nodiscard]] const char *afterKey(const char *needle) const noexcept
    {
        for (const char *p = begin; p < end; ++p)
        {
            if (*p != '"' || !matches(p + 1, end, needle))
                continue;
            const char *q = p + 1;
            while (q < end && *q != '"')
                ++q;
            if (q >= end || q == p + 1)
                continue;
            // The name must end exactly where the needle does, or "shape" would match "shapes".
            if (static_cast<core::usize>(q - (p + 1)) != [needle] {
                    core::usize n = 0;
                    while (needle[n] != '\0')
                        ++n;
                    return n;
                }())
                continue;
            ++q;
            while (q < end && (*q == ' ' || *q == '\n' || *q == '\r' || *q == '\t'))
                ++q;
            if (q >= end || *q != ':')
                continue;
            ++q;
            while (q < end && (*q == ' ' || *q == '\n' || *q == '\r' || *q == '\t'))
                ++q;
            return q;
        }
        return nullptr;
    }

    [[nodiscard]] bool integer(const char *needle, core::i64 &out) const noexcept
    {
        const char *p = afterKey(needle);
        if (p == nullptr)
            return false;
        bool negative = false;
        if (*p == '-')
        {
            negative = true;
            ++p;
        }
        if (p >= end || *p < '0' || *p > '9')
            return false;
        core::i64 v = 0;
        while (p < end && *p >= '0' && *p <= '9')
            v = v * 10 + (*p++ - '0');
        out = negative ? -v : v;
        return true;
    }

    /// Reads `[a, b, c]` of integers.
    [[nodiscard]] core::u32 integerArray(const char *needle, core::i64 *out, core::u32 capacity) const noexcept
    {
        const char *p = afterKey(needle);
        if (p == nullptr || *p != '[')
            return 0u;
        ++p;
        core::u32 n = 0u;
        while (p < end && *p != ']')
        {
            while (p < end && (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t'))
                ++p;
            if (p >= end || *p == ']')
                break;
            bool negative = false;
            if (*p == '-')
            {
                negative = true;
                ++p;
            }
            if (*p < '0' || *p > '9')
                return 0u;
            core::i64 v = 0;
            while (p < end && *p >= '0' && *p <= '9')
                v = v * 10 + (*p++ - '0');
            if (n < capacity)
                out[n] = negative ? -v : v;
            ++n;
            while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t'))
                ++p;
        }
        return n;
    }

    /// Copies a string value; returns length written, 0 when absent or not a string.
    [[nodiscard]] core::u32 string(const char *needle, char *out, core::u32 capacity) const noexcept
    {
        const char *p = afterKey(needle);
        if (p == nullptr || *p != '"')
            return 0u;
        ++p;
        core::u32 n = 0u;
        while (p < end && *p != '"')
        {
            if (n + 1u >= capacity)
                return 0u;
            out[n++] = *p++;
        }
        out[n] = '\0';
        return n;
    }

    [[nodiscard]] bool isNull(const char *needle) const noexcept
    {
        const char *p = afterKey(needle);
        return p != nullptr && matches(p, end, "null");
    }
};

[[nodiscard]] bool sameText(const char *a, const char *b) noexcept
{
    for (; *a != '\0' && *b != '\0'; ++a, ++b)
    {
        if (*a != *b)
            return false;
    }
    return *a == *b;
}

/// Appends a decimal integer; returns false when it would not fit.
[[nodiscard]] bool appendInteger(char *out, core::u32 &at, core::u32 capacity, core::i64 v) noexcept
{
    char tmp[24];
    core::u32 n = 0u;
    const bool negative = v < 0;
    core::u64 u = negative ? static_cast<core::u64>(-v) : static_cast<core::u64>(v);
    do
    {
        tmp[n++] = static_cast<char>('0' + (u % 10u));
        u /= 10u;
    } while (u != 0u && n < sizeof(tmp));
    if (negative)
        tmp[n++] = '-';
    if (at + n + 1u > capacity)
        return false;
    while (n != 0u)
        out[at++] = tmp[--n];
    return true;
}

} // namespace

bool parseArrayMeta(const char *json, core::usize length, ArrayMeta &out) noexcept
{
    if (json == nullptr || length == 0u)
        return false;

    const Scanner s{json, json + length};
    out = ArrayMeta{};

    const core::u32 dims = s.integerArray("shape", out.shape, kMaxDimensions);
    if (dims == 0u || dims > kMaxDimensions)
        return false;
    if (s.integerArray("chunks", out.chunks, kMaxDimensions) != dims)
        return false;
    out.dimensions = dims;

    char dtype[16];
    if (s.string("dtype", dtype, sizeof(dtype)) == 0u)
        return false;
    // Only single-byte samples: that is what every tomographic volume this serves publishes, and
    // guessing an endianness and a width from a two-character code is how a viewer renders noise.
    if (!sameText(dtype, "|u1") && !sameText(dtype, "|i1") && !sameText(dtype, "u1"))
        return false;
    out.itemSize = 1u;

    core::i64 fill = 0;
    if (s.integer("fill_value", fill))
        out.fillValue = static_cast<core::u8>(fill < 0 ? 0 : (fill > 255 ? 255 : fill));

    char order[8];
    if (s.string("order", order, sizeof(order)) != 0u)
        out.cOrder = sameText(order, "C");
    if (!out.cOrder)
        return false;

    char sep[8];
    out.separator = (s.string("dimension_separator", sep, sizeof(sep)) != 0u && sep[0] != '\0') ? sep[0] : '.';

    if (s.isNull("compressor"))
    {
        out.codec = Codec::Raw;
    }
    else
    {
        char id[24];
        if (s.string("id", id, sizeof(id)) == 0u)
            out.codec = Codec::Unsupported;
        else if (sameText(id, "blosc"))
            out.codec = Codec::Blosc;
        else if (sameText(id, "zstd"))
            out.codec = Codec::Zstd;
        else
            out.codec = Codec::Unsupported;
    }
    if (out.codec != Codec::Raw && !codecAvailable(out.codec))
        out.codec = Codec::Unsupported;

    return out.valid();
}

core::u32 countMultiscaleLevels(const char *json, core::usize length) noexcept
{
    if (json == nullptr || length == 0u)
        return 0u;
    const Scanner s{json, json + length};
    if (s.afterKey("multiscales") == nullptr)
        return 0u;

    // Count `"path"` entries inside the datasets array. The datasets list is the pyramid, and its
    // length is the only thing a streamer needs from this document.
    core::u32 levels = 0u;
    for (const char *p = json; p + 6 < json + length; ++p)
    {
        if (*p == '"' && Scanner::matches(p + 1, json + length, "path\""))
            ++levels;
    }
    return levels;
}

bool chunkKey(const ArrayMeta &meta, const char *prefix, const core::i64 *index, char *out) noexcept
{
    if (out == nullptr || index == nullptr || !meta.valid())
        return false;

    core::u32 at = 0u;
    if (prefix != nullptr)
    {
        for (const char *p = prefix; *p != '\0'; ++p)
        {
            if (at + 2u >= kMaxKeyLength)
                return false;
            out[at++] = *p;
        }
        if (at + 2u >= kMaxKeyLength)
            return false;
        out[at++] = '/';
    }
    for (core::u32 d = 0u; d < meta.dimensions; ++d)
    {
        if (d != 0u)
        {
            if (at + 2u >= kMaxKeyLength)
                return false;
            out[at++] = meta.separator;
        }
        if (!appendInteger(out, at, kMaxKeyLength, index[d]))
            return false;
    }
    out[at] = '\0';
    return true;
}

FetchResult readChunk(IZarrStore &store, const ArrayMeta &meta, const char *key, core::u8 *out, core::usize capacity,
                      core::u8 *scratch, core::usize scratchCapacity) noexcept
{
    const core::usize want = meta.chunkBytes();
    if (out == nullptr || capacity < want)
        return FetchResult{FetchStatus::TooLarge, want};

    if (meta.codec == Codec::Raw)
    {
        FetchResult r = store.read(key, out, capacity);
        if (r.status == FetchStatus::Absent)
        {
            for (core::usize i = 0u; i < want; ++i)
                out[i] = meta.fillValue;
            return FetchResult{FetchStatus::Ok, want, true};
        }
        if (r.ok() && r.size != want)
            return FetchResult{FetchStatus::Failed, r.size};
        return r;
    }

    if (scratch == nullptr || scratchCapacity == 0u)
        return FetchResult{FetchStatus::Failed, 0u};

    FetchResult raw = store.read(key, scratch, scratchCapacity);
    if (raw.status == FetchStatus::Absent)
    {
        for (core::usize i = 0u; i < want; ++i)
            out[i] = meta.fillValue;
        return FetchResult{FetchStatus::Ok, want, true};
    }
    if (!raw.ok())
        return raw;

    const core::usize produced = decodeChunk(meta.codec, scratch, raw.size, out, capacity);
    if (produced != want)
        return FetchResult{FetchStatus::Failed, produced};
    return FetchResult{FetchStatus::Ok, produced};
}

} // namespace lpl::zarr
