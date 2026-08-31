/**
 * @file MinimalIsa.hpp
 * @brief Ten opcodes, chosen to be re-implementable from a drawing.
 *
 * LOAD STORE ADD XOR JMP IF and their few companions. The selection criterion is
 * not expressiveness but how little a future reader must be told before an
 * emulator becomes writable.
 *
 * Three decisions follow from that criterion and none of them would follow from any
 * other:
 *
 * - **Fixed four-byte instructions.** A variable-length encoding is denser and needs
 *   a decoding table to read at all; a fixed one can be read with a ruler. Whoever
 *   finds this has to be able to segment the stream before they can do anything else.
 * - **Eight registers and a flat byte memory.** No stack, no flags, no addressing
 *   modes. Every one of those is a rule that has to be transmitted, and every rule
 *   transmitted is a rule that can be lost.
 * - **Wrapping arithmetic, no traps.** A machine that can fault has a fault model,
 *   which is more specification. This one always continues, and a program that runs
 *   past its budget simply stops.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_ROSETTA_MINIMALISA_HPP
#    define LPL_LPL_ROSETTA_MINIMALISA_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::rosetta {

/**
 * @brief Bytes one instruction occupies: opcode, then three operand bytes.
 *
 * Fixed, and that is the point. See the file comment.
 */
inline constexpr core::u32 kInstructionBytes = 4u;

/**
 * @brief Registers the machine has.
 *
 * Eight, so a register index fits a byte with room to spare and a reader never has to
 * be told about an encoding trick.
 */
inline constexpr core::u32 kRegisterCount = 8u;

/**
 * @enum Opcode
 * @brief The ten. Their numeric values are part of the engraved specification.
 */
enum class Opcode : core::u8 {
    Halt = 0u,       ///< Stop. Operands ignored.
    Set = 1u,        ///< r[a] = (b << 8) | c. The only way a constant enters.
    Load = 2u,       ///< r[a] = memory[r[b] + c].
    Store = 3u,      ///< memory[r[b] + c] = r[a] (low byte).
    Add = 4u,        ///< r[a] = r[b] + r[c], wrapping.
    Sub = 5u,        ///< r[a] = r[b] - r[c], wrapping.
    Xor = 6u,        ///< r[a] = r[b] ^ r[c].
    Shift = 7u,      ///< r[a] = r[b] << c when c < 128, else r[b] >> (c - 128).
    Jump = 8u,       ///< Program counter = (b << 8) | c, in instructions.
    JumpIfZero = 9u, ///< The same, when r[a] is zero.
    Count = 10u      ///< Not an opcode: the number of them.
};

/**
 * Bytes a mnemonic occupies, in the table and in storage alike.
 *
 * @warning **One number, because it used to be two facts.** The engraved table gives every
 * mnemonic a fixed eight-byte field while the names themselves were bare literals of whatever
 * length they happened to be, so writing a row meant walking off the end of a five-character
 * string to fill an eight-byte slot. That is a real out-of-bounds read, it survived because a
 * second loop overwrote the garbage before anyone could see it, and only `-O2` ever reported it.
 * Storing the names at the field's own width removes it by construction rather than by argument.
 */
inline constexpr core::u32 kMnemonicBytes = 8u;

/**
 * @brief The words a specification spells the opcodes with, NUL-padded to the field width.
 *
 * Words rather than numbers, for the reason every named thing in this project is one: a number
 * means whatever the enumeration happened to be the day it was written. Here it matters more than
 * usual -- the whole artifact exists to be read by someone who does not have the enumeration.
 *
 * @warning A name that does not leave room for its terminator fails to COMPILE, which is the
 * truncation rule stated where it cannot be forgotten. The emitter used to carry that rule as a
 * runtime branch nothing exercised.
 */
inline constexpr char kOpcodeMnemonics[static_cast<core::u32>(Opcode::Count)][kMnemonicBytes] = {
    "HALT", "SET", "LOAD", "STORE", "ADD", "SUB", "XOR", "SHIFT", "JUMP", "JZ",
};

/**
 * @brief The word a specification spells @p opcode with.
 *
 * @param opcode The instruction.
 * @return Its mnemonic, or "?" when the value is not one of the ten.
 */
[[nodiscard]] constexpr const char *opcodeName(Opcode opcode) noexcept
{
    const auto index = static_cast<core::u32>(opcode);
    return index < static_cast<core::u32>(Opcode::Count) ? kOpcodeMnemonics[index] : "?";
}

/**
 * @brief How many of the three operand bytes @p opcode actually reads.
 *
 * Engraved with the mnemonic, because a reader who knows an instruction is four bytes
 * still cannot tell which of them mean anything.
 *
 * @param opcode The instruction.
 * @return 0 to 3.
 */
[[nodiscard]] constexpr core::u32 operandCount(Opcode opcode) noexcept
{
    switch (opcode)
    {
    case Opcode::Halt: return 0u;
    case Opcode::Jump: return 2u;
    case Opcode::Set:
    case Opcode::Load:
    case Opcode::Store:
    case Opcode::Add:
    case Opcode::Sub:
    case Opcode::Xor:
    case Opcode::Shift:
    case Opcode::JumpIfZero: return 3u;
    case Opcode::Count: break;
    }
    return 0u;
}

/**
 * @brief Packs one instruction into its four bytes.
 * @param opcode The instruction.
 * @param a      First operand.
 * @param b      Second operand.
 * @param c      Third operand.
 * @param out    Receives @ref kInstructionBytes bytes.
 */
constexpr void encodeInstruction(Opcode opcode, core::u8 a, core::u8 b, core::u8 c, core::u8 *out) noexcept
{
    out[0] = static_cast<core::u8>(opcode);
    out[1] = a;
    out[2] = b;
    out[3] = c;
}

} // namespace lpl::rosetta

#endif // LPL_LPL_ROSETTA_MINIMALISA_HPP
