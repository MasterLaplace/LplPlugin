/**
 * @file Codec.hpp
 * @brief Unpacking a chunk, and admitting when this build cannot.
 *
 * @warning **A codec this build lacks is REFUSED, never approximated.** Handing compressed bytes to
 * the raw path produces a full buffer of plausible garbage, and garbage renders -- a viewer would
 * show a structure that is entirely an artefact of the mistake. So the availability question is
 * asked before the metadata is accepted, and an array that names a codec this binary does not have
 * fails to open with a reason.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ZARR_CODEC_HPP
#    define LPL_ZARR_CODEC_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::zarr {

/**
 * @enum Codec
 * @brief How a chunk's bytes are packed.
 */
enum class Codec : core::u8 {
    Raw = 0,     ///< `"compressor": null`. Real volumes ship this way; it is also the fastest path.
    Blosc,       ///< The blosc container, whatever sub-codec it declares in its own header.
    Zstd,        ///< Bare zstd.
    Unsupported, ///< Named in the metadata and not implemented. Refused, never guessed.
};

/// @return Whether this build can unpack @p codec.
[[nodiscard]] bool codecAvailable(Codec codec) noexcept;

/**
 * @brief Unpacks @p size bytes of @p codec from @p in into @p out.
 * @return Bytes produced, or 0 on failure.
 */
[[nodiscard]] core::usize decodeChunk(Codec codec, const core::u8 *in, core::usize size, core::u8 *out,
                                      core::usize capacity) noexcept;

/// @return Human-readable name, for a log line that has to say what it could not open.
[[nodiscard]] const char *codecName(Codec codec) noexcept;

} // namespace lpl::zarr

#endif // LPL_ZARR_CODEC_HPP
