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

#    include <span>

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
 * @brief Unpacks the @p codec bytes of @p in into @p out; Raw bytes are copied as they are.
 * @return Bytes produced, or 0 when @p in is empty or rejected by the codec (blosc also when it is
 * shorter than its header announces), the result does not fit @p out, or this build lacks
 * @p codec. A count other than 0 does not prove the bytes intact: Raw copies a short input as it is.
 */
[[nodiscard]] core::usize decodeChunk(Codec codec, std::span<const core::u8> in, std::span<core::u8> out) noexcept;

/// @return Human-readable name, for a log line that has to say what it could not open.
[[nodiscard]] const char *codecName(Codec codec) noexcept;

} // namespace lpl::zarr

#endif // LPL_ZARR_CODEC_HPP
