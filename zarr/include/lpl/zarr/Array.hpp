/**
 * @file Array.hpp
 * @brief Reading what a zarr array says about itself, and building the key of one chunk.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ZARR_ARRAY_HPP
#    define LPL_ZARR_ARRAY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/zarr/Codec.hpp>
#    include <lpl/zarr/Store.hpp>

#    include <limits>
#    include <span>

namespace lpl::zarr {

/// Most dimensions an array may declare. A volume uses three.
inline constexpr core::u32 kMaxDimensions = 4u;

/// Longest store key this reader will build, including the terminator.
inline constexpr core::u32 kMaxKeyLength = 128u;

/**
 * @struct ArrayMeta
 * @brief One level of a zarr array of single-byte samples in C order, as its own `.zarray` describes it.
 */
struct ArrayMeta final {
    core::i64 shape[kMaxDimensions]{};
    core::i64 chunks[kMaxDimensions]{};
    core::u32 dimensions{0u};
    Codec codec{Codec::Raw};
    core::u8 fillValue{0u}; ///< The byte an absent chunk is made of: a signed -1 is 0xFF.
    char separator{'.'};    ///< The character between chunk indices in a key, '.' or '/'.

    /**
     * @return Whether the fields describe an array this reader can address: one to
     * @ref kMaxDimensions dimensions, no negative extent, chunks of at least one sample whose byte
     * count fits a `usize`, a separator of '.' or '/', and a codec the format names.
     */
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        if (dimensions < 1u || dimensions > kMaxDimensions || codec == Codec::Unsupported)
            return false;
        if (separator != '.' && separator != '/')
            return false;
        core::usize bytes = 1u;
        for (core::u32 d = 0u; d < dimensions; ++d)
        {
            if (shape[d] < 0 || chunks[d] < 1)
                return false;
            if (static_cast<core::u64>(chunks[d]) > std::numeric_limits<core::usize>::max() / bytes)
                return false;
            bytes *= static_cast<core::usize>(chunks[d]);
        }
        return true;
    }

    /// @return Bytes one decoded chunk occupies, or 0 when the fields are not valid().
    [[nodiscard]] constexpr core::usize chunkBytes() const noexcept
    {
        if (!valid())
            return 0u;
        core::usize bytes = 1u;
        for (core::u32 d = 0u; d < dimensions; ++d)
            bytes *= static_cast<core::usize>(chunks[d]);
        return bytes;
    }

    /// @return Chunks along @p axis, rounded up the way zarr rounds; 0 when the fields are not
    /// valid() or the array has no such axis.
    [[nodiscard]] constexpr core::i64 chunkCount(core::u32 axis) const noexcept
    {
        if (!valid() || axis >= dimensions)
            return 0;
        return shape[axis] / chunks[axis] + (shape[axis] % chunks[axis] != 0 ? 1 : 0);
    }
};

/**
 * @brief Parses a `.zarray` document.
 *
 * @warning **The separator is READ, never assumed.** One corpus mixes both: Scroll 1's volumes use
 * `/` and another volume in the same bucket uses `.`. Hard-coding it does not produce an error --
 * it produces a request for a key that does not exist, which a reader counts as an absent chunk,
 * so an entire volume comes back reported as empty. That failure has been paid once already, in
 * this project's Python reader, and it is the reason this is a field rather than a constant.
 *
 * @warning **An unknown compressor, a filter or Fortran order is refused, not ignored.** Each one
 * changes what the bytes mean; Codec.hpp says why a refusal beats a guess.
 *
 * @return Whether the document describes an array of the kind @ref ArrayMeta holds (single-byte
 * samples, C order, no filter, a fill value the sample type can hold) that is
 * @ref ArrayMeta::valid and whose codec this build has. On false, @p out holds what was read
 * before the refusal, nothing for a null or empty document, so an array refused for a codec this
 * build lacks still names that codec.
 */
[[nodiscard]] bool parseArrayMeta(const char *json, core::usize length, ArrayMeta &out) noexcept;

/**
 * @brief Number of levels a multiscale group declares in its `.zattrs`.
 *
 * OME-NGFF puts the pyramid in `multiscales[0].datasets[].path`. Counting the entries is all a
 * streamer needs -- the paths themselves are the level indices in every volume this has met, and
 * a reader that trusted the count over the paths would break the day one is not.
 *
 * @return Levels found, or 0 when the document declares no multiscale group.
 */
[[nodiscard]] core::u32 countMultiscaleLevels(const char *json, core::usize length) noexcept;

/**
 * @brief Builds the store key of one chunk.
 *
 * @param meta     The level's metadata; supplies the separator.
 * @param prefix   Level prefix, for example "0". May be null for a bare array.
 * @param index    Chunk indices, @p meta.dimensions of them.
 * @param out      Destination, at least @ref kMaxKeyLength bytes.
 * @return Whether the key fit, false too when @p meta is not valid or an index lies outside the array.
 */
[[nodiscard]] bool chunkKey(const ArrayMeta &meta, const char *prefix, const core::i64 *index, char *out) noexcept;

/**
 * @brief Reads and decodes one chunk into @p out.
 *
 * @param scratch Room for the stored bytes of a compressed chunk; unused for a raw one.
 * @retval Ok       The chunk is in @p out; `filled` says it is the fill value of a key the store lacks.
 * @retval TooLarge @p out is smaller than one chunk, or the stored bytes of a compressed chunk do not
 *                  fit @p scratch; `size` is the bytes needed.
 * @retval Failed   @p meta is not valid, a compressed array came with no scratch, the store failed,
 *                  a raw chunk is not exactly one chunk long, or the bytes did not decode to one chunk.
 */
[[nodiscard]] FetchResult readChunk(IZarrStore &store, const ArrayMeta &meta, const char *key, std::span<core::u8> out,
                                    std::span<core::u8> scratch) noexcept;

} // namespace lpl::zarr

#endif // LPL_ZARR_ARRAY_HPP
