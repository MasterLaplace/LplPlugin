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

namespace lpl::zarr {

/// Dimensions this reader handles. Three is what a volume is; more is a different problem.
inline constexpr core::u32 kMaxDimensions = 4u;

/// Longest store key this reader will build, including the terminator.
inline constexpr core::u32 kMaxKeyLength = 128u;

/**
 * @struct ArrayMeta
 * @brief One level of a zarr array, as its own `.zarray` describes it.
 */
struct ArrayMeta final {
    core::i64 shape[kMaxDimensions]{};
    core::i64 chunks[kMaxDimensions]{};
    core::u32 dimensions{0u};
    core::u32 itemSize{1u}; ///< Bytes per sample.
    Codec codec{Codec::Raw};
    core::u8 fillValue{0u};
    char separator{'.'}; ///< The character between chunk indices in a key.
    bool cOrder{true};   ///< Row-major. Fortran order is refused rather than silently mis-read.

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return dimensions >= 1u && dimensions <= kMaxDimensions && itemSize >= 1u && codec != Codec::Unsupported;
    }

    /// @return Bytes one decoded chunk occupies.
    [[nodiscard]] constexpr core::usize chunkBytes() const noexcept
    {
        core::usize n = itemSize;
        for (core::u32 d = 0u; d < dimensions; ++d)
            n *= static_cast<core::usize>(chunks[d]);
        return n;
    }

    /// @return Chunks along @p axis, rounded up the way zarr rounds.
    [[nodiscard]] constexpr core::i64 chunkCount(core::u32 axis) const noexcept
    {
        return (shape[axis] + chunks[axis] - 1) / chunks[axis];
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
 * @warning **An unknown compressor is refused, not ignored.** Decoding a chunk with the wrong codec
 * yields bytes, and bytes render.
 *
 * @return Whether the document parsed into something usable.
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
 * @return Whether the key fit.
 */
[[nodiscard]] bool chunkKey(const ArrayMeta &meta, const char *prefix, const core::i64 *index, char *out) noexcept;

/**
 * @brief Reads and decodes one chunk into @p out.
 *
 * @warning An absent chunk is filled with @ref ArrayMeta::fillValue and reported Ok, because that
 * is what absence means in this format -- not an error, and not a hole.
 */
[[nodiscard]] FetchResult readChunk(IZarrStore &store, const ArrayMeta &meta, const char *key, core::u8 *out,
                                    core::usize capacity, core::u8 *scratch, core::usize scratchCapacity) noexcept;

} // namespace lpl::zarr

#endif // LPL_ZARR_ARRAY_HPP
