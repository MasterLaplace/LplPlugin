/**
 * @file EccBaker.hpp
 * @brief Host-side: compute and attach the parity section.
 *
 * Encoding is the expensive half and belongs where there is a heap; the kernel only
 * ever decodes.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_EDITOR_ECCBAKER_HPP
#    define LPL_LPL_EDITOR_ECCBAKER_HPP

#    include <lpl/core/Types.hpp>

#    include <vector>

namespace lpl::editor {

/**
 * @struct EccPolicy
 * @brief How much damage a cartridge should be able to shrug off.
 *
 * The two numbers are not independent, and the pair is what a caller should think
 * about: with @c parityShards symbols a codeword survives floor(s/2) wrong ones, and
 * because the layout is transversal a burst confined to one row costs exactly one
 * symbol per codeword. So a pack with 32 rows and 4 parity symbols survives a burst
 * up to two rows wide — a sixteenth of the image — anywhere in it.
 */
struct EccPolicy {
    core::u32 dataShards{32u};  ///< Rows the protected span is cut into.
    core::u32 parityShards{4u}; ///< Parity symbols per column.
};

/**
 * @brief Appends a transversal parity section to a baked pack.
 *
 * Appended LAST on purpose: it protects every byte between the header and itself,
 * which is the section table and every other payload. Inserting it earlier would
 * mean protecting bytes that move when it is inserted.
 *
 * The header is rewritten afterwards — total size, section count and content hash —
 * so the result is a valid pack rather than a valid pack with a tail.
 *
 * @param image  A pack image as bakeGamePack produced it.
 * @param policy How many rows and how much parity.
 * @return The image with its parity section, or the input unchanged when it is not a
 *         pack.
 */
[[nodiscard]] std::vector<core::u8> attachEcc(const std::vector<core::u8> &image, const EccPolicy &policy = EccPolicy{});

} // namespace lpl::editor

#endif // LPL_LPL_EDITOR_ECCBAKER_HPP
