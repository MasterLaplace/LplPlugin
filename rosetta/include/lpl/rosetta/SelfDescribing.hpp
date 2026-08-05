/**
 * @file SelfDescribing.hpp
 * @brief Embedding the specification in the artifact header.
 *
 * What makes a .lplpak a Rosetta object rather than an opaque blob: the reader is
 * not assumed, it is included.
 *
 * The specification is bytes, not prose, and it is bytes a machine can act on — an
 * @ref Interpreter built from it runs exactly the opcodes it names and refuses the
 * ones it does not. That is what turns "the spec is engraved" from a claim into a
 * test: engrave it, throw the source away, rebuild the reader from the engraving, and
 * see whether the payload comes back.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_ROSETTA_SELFDESCRIBING_HPP
#    define LPL_LPL_ROSETTA_SELFDESCRIBING_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/rosetta/MinimalIsa.hpp>

namespace lpl::rosetta {

/**
 * @brief Bytes a mnemonic occupies in the specification, NUL padding included.
 */
inline constexpr core::u32 kMnemonicBytes = 8u;

/**
 * @brief The four bytes a specification starts with.
 *
 * "LPLI" — a reader that finds them knows it is holding an instruction-set
 * description rather than payload. Four bytes is the cheapest thing that can be
 * recognised at a glance under a microscope.
 */
inline constexpr core::u8 kSpecificationMagic[4] = {'L', 'P', 'L', 'I'};

/**
 * @struct IsaEntry
 * @brief One row of the engraved table: what an opcode is called and how wide it is.
 */
struct IsaEntry {
    core::u8 opcode{0u};             ///< Numeric value in the instruction stream.
    core::u8 operands{0u};           ///< Operand bytes it reads, 0 to 3.
    char mnemonic[kMnemonicBytes]{}; ///< NUL-padded name.
};

/**
 * @struct IsaTable
 * @brief Every opcode a specification names.
 */
struct IsaTable {
    core::u32 instructionBytes{0u}; ///< Bytes per instruction.
    core::u32 registerCount{0u};    ///< Registers the machine has.
    core::u32 count{0u};            ///< Rows below.
    IsaEntry entry[static_cast<core::u32>(Opcode::Count)]{};
};

/**
 * @brief Bytes @ref emitSpecification writes.
 *
 * Fixed, so an engraving can reserve the room before it knows what goes in it.
 */
inline constexpr core::u32 kSpecificationBytes =
    4u + 4u + 4u + 4u + static_cast<core::u32>(Opcode::Count) * (2u + kMnemonicBytes);

/**
 * @brief Writes this build's instruction set as bytes.
 *
 * Layout, in order: the magic, the instruction width, the register count, the row
 * count, then one row per opcode. Everything is a byte or a four-byte little-endian
 * word, and nothing is a pointer or an offset — an offset is a rule, and the point of
 * this file is to have as few rules as possible.
 *
 * @param out  Receives @ref kSpecificationBytes bytes.
 * @param size Room available.
 * @return Bytes written, or 0 when there is not enough room.
 */
core::u32 emitSpecification(core::u8 *out, core::u32 size) noexcept;

/**
 * @brief Reads a specification back into a table.
 * @param bytes    The engraved specification.
 * @param size     Its length.
 * @param outTable Receives the table.
 * @return false when the magic is wrong or the bytes are short.
 */
[[nodiscard]] bool readSpecification(const core::u8 *bytes, core::u32 size, IsaTable &outTable) noexcept;

} // namespace lpl::rosetta

#endif // LPL_LPL_ROSETTA_SELFDESCRIBING_HPP
