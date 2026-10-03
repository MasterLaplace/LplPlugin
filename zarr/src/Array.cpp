/**
 * @file Array.cpp
 * @brief Parsing `.zarray`, building chunk keys, decoding chunks.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/zarr/Array.hpp>

#include <limits>

namespace lpl::zarr {

namespace {

[[nodiscard]] constexpr bool isJsonSpace(char c) noexcept { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

[[nodiscard]] const char *skipSpaces(const char *p, const char *end) noexcept
{
    while (p < end && isJsonSpace(*p))
        ++p;
    return p;
}

/// @return Position just past @p prefix when the text at @p p starts with it, or nullptr.
[[nodiscard]] const char *pastPrefix(const char *p, const char *end, const char *prefix) noexcept
{
    for (; *prefix != '\0'; ++prefix, ++p)
    {
        if (p >= end || *p != *prefix)
            return nullptr;
    }
    return p;
}

/// Moves @p p past a JSON integer. Refuses a fraction, an exponent, and what an i64 cannot hold.
[[nodiscard]] bool readInteger(const char *&p, const char *end, core::i64 &out) noexcept
{
    const bool negative = p < end && *p == '-';
    if (negative)
        ++p;
    if (p >= end || *p < '0' || *p > '9')
        return false;

    core::i64 magnitude = 0;
    for (; p < end && *p >= '0' && *p <= '9'; ++p)
    {
        const core::i64 digit = *p - '0';
        if (magnitude > (std::numeric_limits<core::i64>::max() - digit) / 10)
            return false;
        magnitude = magnitude * 10 + digit;
    }
    if (p < end && (*p == '.' || *p == 'e' || *p == 'E'))
        return false;
    out = negative ? -magnitude : magnitude;
    return true;
}

/// @return Position just past the bracket that closes the one at @p open, or nullptr when the
/// document ends first. Brackets inside strings do not count.
[[nodiscard]] const char *pastMatchingBracket(const char *open, const char *end) noexcept
{
    core::u32 depth = 0u;
    bool inString = false;
    for (const char *p = open; p < end; ++p)
    {
        if (*p == '"')
            inString = !inString;
        if (inString)
            continue;
        if (*p == '{' || *p == '[')
            ++depth;
        else if ((*p == '}' || *p == ']') && --depth == 0u)
            return p + 1;
    }
    return nullptr;
}

/**
 * A scanner, not a JSON parser, and the difference is stated rather than hidden.
 *
 * `.zarray` is machine-written from a fixed schema: a flat object of known keys with numbers,
 * strings, null and one nested object. Finding a key and reading the value after it is enough for
 * exactly that, and pulling in a general parser to read six fields would be a dependency the
 * kernel-adjacent half of this tree cannot take. What this will NOT survive is a hand-edited
 * document with the same key twice, a string value containing the text of another key, or an
 * escaped quote inside a string -- all of which a real writer never produces.
 */
struct Scanner final {
    const char *begin{nullptr};
    const char *end{nullptr};

    /// @return Position of the value after `"key":`, or nullptr when the key is absent or has no value.
    [[nodiscard]] const char *afterKey(const char *key) const noexcept
    {
        for (const char *p = begin; p < end; ++p)
        {
            if (*p != '"')
                continue;
            const char *q = pastPrefix(p + 1, end, key);
            // The name must end exactly where the key does, or "shape" would match "shapes".
            if (q == nullptr || q >= end || *q != '"')
                continue;
            q = skipSpaces(q + 1, end);
            if (q >= end || *q != ':')
                continue;
            q = skipSpaces(q + 1, end);
            return q < end ? q : nullptr;
        }
        return nullptr;
    }

    [[nodiscard]] bool has(const char *key) const noexcept { return afterKey(key) != nullptr; }

    [[nodiscard]] bool isNull(const char *key) const noexcept
    {
        const char *p = afterKey(key);
        return p != nullptr && pastPrefix(p, end, "null") != nullptr;
    }

    [[nodiscard]] bool integer(const char *key, core::i64 &out) const noexcept
    {
        const char *p = afterKey(key);
        return p != nullptr && readInteger(p, end, out);
    }

    /// Reads `[a, b, c]` of integers. @return How many it holds, or 0 when it is malformed.
    [[nodiscard]] core::u32 integerArray(const char *key, core::i64 *out, core::u32 capacity) const noexcept
    {
        const char *p = afterKey(key);
        if (p == nullptr || *p != '[')
            return 0u;
        ++p;
        core::u32 n = 0u;
        for (;;)
        {
            while (p < end && (*p == ',' || isJsonSpace(*p)))
                ++p;
            if (p >= end)
                return 0u;
            if (*p == ']')
                return n;
            core::i64 value = 0;
            if (!readInteger(p, end, value))
                return 0u;
            if (n < capacity)
                out[n] = value;
            ++n;
        }
    }

    /// Copies a string value; returns length written, 0 when absent, not a string, empty, too long
    /// for @p capacity, or unterminated.
    [[nodiscard]] core::u32 string(const char *key, char *out, core::u32 capacity) const noexcept
    {
        const char *p = afterKey(key);
        if (p == nullptr || *p != '"')
            return 0u;
        core::u32 n = 0u;
        for (++p; p < end && *p != '"'; ++p)
        {
            if (n + 1u >= capacity)
                return 0u;
            out[n++] = *p;
        }
        if (p >= end)
            return 0u;
        out[n] = '\0';
        return n;
    }

    /// @return Whether the value is a string of exactly one character, which lands in @p out.
    [[nodiscard]] bool letter(const char *key, char &out) const noexcept
    {
        char text[2];
        if (string(key, text, sizeof(text)) != 1u)
            return false;
        out = text[0];
        return true;
    }

    /// @return A scanner over the object or array @p key holds, or an empty one when it holds neither.
    [[nodiscard]] Scanner nested(const char *key) const noexcept
    {
        const char *open = afterKey(key);
        if (open == nullptr || (*open != '{' && *open != '['))
            return Scanner{};
        const char *close = pastMatchingBracket(open, end);
        return close == nullptr ? Scanner{} : Scanner{open, close};
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

[[nodiscard]] bool readSampleType(const Scanner &document, bool &signedSamples) noexcept
{
    char dtype[16];
    if (document.string("dtype", dtype, sizeof(dtype)) == 0u)
        return false;
    // Only single-byte samples: that is what every tomographic volume this serves publishes, and
    // guessing an endianness and a width from a two-character code is how a viewer renders noise.
    signedSamples = sameText(dtype, "|i1");
    return signedSamples || sameText(dtype, "|u1") || sameText(dtype, "u1");
}

[[nodiscard]] bool readFillValue(const Scanner &document, bool signedSamples, core::u8 &fill) noexcept
{
    if (!document.has("fill_value") || document.isNull("fill_value"))
        return true;
    core::i64 value = 0;
    if (!document.integer("fill_value", value))
        return false;
    const core::i64 lowest = signedSamples ? std::numeric_limits<core::i8>::min() : 0;
    const core::i64 highest =
        signedSamples ? std::numeric_limits<core::i8>::max() : std::numeric_limits<core::u8>::max();
    if (value < lowest || value > highest)
        return false;
    fill = static_cast<core::u8>(value);
    return true;
}

/// The compressor's own "id", and only that one: a filter can carry an "id" too.
[[nodiscard]] Codec codecOfCompressor(const Scanner &document) noexcept
{
    if (document.isNull("compressor"))
        return Codec::Raw;
    char id[24];
    if (document.nested("compressor").string("id", id, sizeof(id)) == 0u)
        return Codec::Unsupported;
    if (sameText(id, "blosc"))
        return Codec::Blosc;
    if (sameText(id, "zstd"))
        return Codec::Zstd;
    return Codec::Unsupported;
}

[[nodiscard]] bool indexWithinArray(const ArrayMeta &meta, const core::i64 *index) noexcept
{
    for (core::u32 d = 0u; d < meta.dimensions; ++d)
    {
        if (index[d] < 0 || index[d] >= meta.chunkCount(d))
            return false;
    }
    return true;
}

[[nodiscard]] bool appendChar(char *key, core::u32 &at, char c) noexcept
{
    if (at + 2u > kMaxKeyLength)
        return false;
    key[at++] = c;
    return true;
}

[[nodiscard]] bool appendText(char *key, core::u32 &at, const char *text) noexcept
{
    for (; *text != '\0'; ++text)
    {
        if (!appendChar(key, at, *text))
            return false;
    }
    return true;
}

[[nodiscard]] bool appendInteger(char *key, core::u32 &at, core::u64 value) noexcept
{
    char reversed[20];
    core::u32 digits = 0u;
    do
    {
        reversed[digits++] = static_cast<char>('0' + value % 10u);
        value /= 10u;
    } while (value != 0u);
    if (at + digits + 1u > kMaxKeyLength)
        return false;
    while (digits != 0u)
        key[at++] = reversed[--digits];
    return true;
}

[[nodiscard]] FetchResult fillAbsentChunk(core::u8 fillValue, std::span<core::u8> chunk) noexcept
{
    for (core::u8 &sample : chunk)
        sample = fillValue;
    return FetchResult{FetchStatus::Ok, chunk.size(), true};
}

} // namespace

bool parseArrayMeta(const char *json, core::usize length, ArrayMeta &out) noexcept
{
    out = ArrayMeta{};
    if (json == nullptr || length == 0u)
        return false;

    const Scanner document{json, json + length};
    const core::u32 dims = document.integerArray("shape", out.shape, kMaxDimensions);
    if (dims == 0u || dims > kMaxDimensions)
        return false;
    if (document.integerArray("chunks", out.chunks, kMaxDimensions) != dims)
        return false;
    out.dimensions = dims;

    bool signedSamples = false;
    if (!readSampleType(document, signedSamples) || !readFillValue(document, signedSamples, out.fillValue))
        return false;
    char order = 'C';
    if (document.has("order") && (!document.letter("order", order) || order != 'C'))
        return false;
    if (document.has("filters") && !document.isNull("filters"))
        return false;
    if (document.has("dimension_separator") && !document.letter("dimension_separator", out.separator))
        return false;

    out.codec = codecOfCompressor(document);
    return out.valid() && codecAvailable(out.codec);
}

core::u32 countMultiscaleLevels(const char *json, core::usize length) noexcept
{
    if (json == nullptr || length == 0u)
        return 0u;
    const Scanner datasets = Scanner{json, json + length}.nested("multiscales").nested("datasets");
    if (datasets.begin == nullptr || *datasets.begin != '[')
        return 0u;

    core::u32 levels = 0u;
    const char *p = datasets.begin + 1;
    while (p < datasets.end)
    {
        if (*p != '{')
        {
            ++p;
            continue;
        }
        const char *close = pastMatchingBracket(p, datasets.end);
        if (close == nullptr)
            break;
        if (Scanner{p, close}.has("path"))
            ++levels;
        p = close;
    }
    return levels;
}

bool chunkKey(const ArrayMeta &meta, const char *prefix, const core::i64 *index, char *out) noexcept
{
    if (out == nullptr || index == nullptr || !meta.valid() || !indexWithinArray(meta, index))
        return false;

    core::u32 at = 0u;
    if (prefix != nullptr && (!appendText(out, at, prefix) || !appendChar(out, at, '/')))
        return false;
    for (core::u32 d = 0u; d < meta.dimensions; ++d)
    {
        if (d != 0u && !appendChar(out, at, meta.separator))
            return false;
        if (!appendInteger(out, at, static_cast<core::u64>(index[d])))
            return false;
    }
    out[at] = '\0';
    return true;
}

FetchResult readChunk(IZarrStore &store, const ArrayMeta &meta, const char *key, std::span<core::u8> out,
                      std::span<core::u8> scratch) noexcept
{
    if (!meta.valid())
        return FetchResult{FetchStatus::Failed, 0u};
    const core::usize chunkBytes = meta.chunkBytes();
    if (out.size() < chunkBytes)
        return FetchResult{FetchStatus::TooLarge, chunkBytes};

    if (meta.codec == Codec::Raw)
    {
        const FetchResult stored = store.read(key, out);
        if (stored.status == FetchStatus::Absent)
            return fillAbsentChunk(meta.fillValue, out.first(chunkBytes));
        if (stored.status == FetchStatus::TooLarge || (stored.ok() && stored.size != chunkBytes))
            return FetchResult{FetchStatus::Failed, stored.size};
        return stored;
    }

    if (scratch.empty())
        return FetchResult{FetchStatus::Failed, 0u};
    const FetchResult stored = store.read(key, scratch);
    if (stored.status == FetchStatus::Absent)
        return fillAbsentChunk(meta.fillValue, out.first(chunkBytes));
    if (!stored.ok())
        return stored;

    const core::usize produced = decodeChunk(meta.codec, scratch.first(stored.size), out);
    if (produced != chunkBytes)
        return FetchResult{FetchStatus::Failed, produced};
    return FetchResult{FetchStatus::Ok, produced};
}

} // namespace lpl::zarr
