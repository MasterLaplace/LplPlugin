/**
 * @file Codec.hpp
 * @brief Portable PPM (P6) image import/export to/from byte buffers.
 *
 * PPM is chosen as the portable interchange format because it has no compression
 * and a trivial, deterministic byte layout — the same encoder/decoder runs on
 * the Linux oracle and the freestanding kernel (where an embedded PPM byte array
 * serves as a compiled-in asset). Alpha is dropped on write and set opaque on
 * read (PPM is RGB).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_IMAGE_CODEC_HPP
#    define LPL_IMAGE_CODEC_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/image/Image.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::image {

/**
 * @brief Encode @p image as a binary PPM (P6) into @p out (replacing it).
 * @return false if the image is empty.
 */
[[nodiscard]] bool writePpm(const Image &image, pmr::vector<core::u8> &out);

/**
 * @brief Decode a binary PPM (P6) from @p data / @p length into @p out.
 *
 * Tolerates leading whitespace and `#` comment lines in the header. Pixels are
 * read as opaque RGBA (alpha = 255). Returns false on a malformed header,
 * unsupported maxval (!= 255) or truncated pixel data.
 */
[[nodiscard]] bool readPpm(const core::u8 *data, core::usize length, Image &out);

} // namespace lpl::image

#endif // LPL_IMAGE_CODEC_HPP
