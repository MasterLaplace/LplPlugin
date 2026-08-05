/**
 * @file EccSection.hpp
 * @brief Parity section: a cartridge that survives a bad sector.
 *
 * The pack format detects corruption with a hash but cannot repair it. This adds a
 * transversal parity section so the ring-0 reader corrects instead of refusing.
 * Beyond a threshold it still fails loudly: a silently repaired-wrong world is worse
 * than a refused one.
 *
 * **Why transversal.** The failure a stored cartridge actually suffers is a BURST —
 * a bad sector, a scratch, a page that did not come back — not a scatter of
 * independent bit flips. So the protected bytes are laid out as `dataShards`
 * contiguous rows, and a codeword is taken DOWN a column, one symbol per row. A burst
 * confined to one row then puts a single wrong symbol into each codeword it touches,
 * which is the case Reed-Solomon is strongest at. Protecting each row on its own
 * would instead concentrate the whole burst in one codeword and lose it.
 *
 * The section does not protect itself. If the parity is what got damaged, the content
 * hash still says the image is bad and the reader refuses — which is the right answer,
 * because a repair driven by corrupt parity is exactly the silent wrong world this
 * format refuses to produce.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_PACK_ECCSECTION_HPP
#    define LPL_LPL_PACK_ECCSECTION_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::pack {

/**
 * @struct EccV1
 * @brief Header of the parity section; the parity bytes follow it.
 */
struct EccV1 {
    core::u32 protectedOffset; ///< First protected byte, from the start of the pack.
    core::u32 protectedBytes;  ///< How many bytes are protected.
    core::u32 rowBytes;        ///< Bytes per row; also the number of codewords.
    core::u32 dataShards;      ///< Rows the protected span is cut into.
    core::u32 parityShards;    ///< Parity symbols per codeword; follows as rowBytes each.
};
static_assert(sizeof(EccV1) == 20u, "GamePack ecc layout is wire format");

/**
 * @struct EccRepairReport
 * @brief What a repair attempt found.
 */
struct EccRepairReport {
    bool present{false};           ///< The pack carried a parity section.
    bool repaired{false};          ///< Every damaged codeword was corrected.
    core::u32 codewords{0u};       ///< Codewords examined.
    core::u32 damagedCodewords{0u}; ///< Codewords that were not already clean.
    core::u32 correctedBytes{0u};  ///< Symbols actually changed.
};

/**
 * @brief Repairs @p bytes in place using its parity section, if it has one.
 *
 * Takes a MUTABLE buffer, which is the whole point: a reader that can only look at
 * the bytes can report damage and nothing more. The kernel copies a GRUB module into
 * memory it owns before calling this.
 *
 * @param bytes     The pack image; modified where it was wrong.
 * @param size      Bytes available.
 * @param outReport Receives what happened.
 * @return true when the image is now consistent with its parity, including the case
 *         where it always was. False means damaged beyond the code's bound.
 */
[[nodiscard]] bool repairPack(core::u8 *bytes, core::u32 size, EccRepairReport &outReport) noexcept;

} // namespace lpl::pack

#endif // LPL_LPL_PACK_ECCSECTION_HPP
