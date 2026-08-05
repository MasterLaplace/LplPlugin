/**
 * @file SelfDescribing.cpp
 * @brief The instruction set, as bytes an artifact can carry.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/rosetta/SelfDescribing.hpp>

namespace lpl::rosetta {

namespace {

/**
 * @brief Writes a little-endian word.
 * @param out   Destination.
 * @param value What to write.
 */
void writeWord(core::u8 *out, core::u32 value) noexcept
{
    out[0] = static_cast<core::u8>(value & 0xFFu);
    out[1] = static_cast<core::u8>((value >> 8) & 0xFFu);
    out[2] = static_cast<core::u8>((value >> 16) & 0xFFu);
    out[3] = static_cast<core::u8>((value >> 24) & 0xFFu);
}

/**
 * @brief Reads a little-endian word.
 * @param bytes Source.
 * @return The value.
 */
[[nodiscard]] core::u32 readWord(const core::u8 *bytes) noexcept
{
    return static_cast<core::u32>(bytes[0]) | (static_cast<core::u32>(bytes[1]) << 8) |
           (static_cast<core::u32>(bytes[2]) << 16) | (static_cast<core::u32>(bytes[3]) << 24);
}

} // namespace

core::u32 emitSpecification(core::u8 *out, core::u32 size) noexcept
{
    if (out == nullptr || size < kSpecificationBytes)
        return 0u;

    core::u32 cursor = 0u;
    for (core::u32 i = 0u; i < 4u; ++i)
        out[cursor++] = kSpecificationMagic[i];

    writeWord(out + cursor, kInstructionBytes);
    cursor += 4u;
    writeWord(out + cursor, kRegisterCount);
    cursor += 4u;
    writeWord(out + cursor, static_cast<core::u32>(Opcode::Count));
    cursor += 4u;

    for (core::u32 i = 0u; i < static_cast<core::u32>(Opcode::Count); ++i)
    {
        const Opcode opcode = static_cast<Opcode>(i);
        out[cursor++] = static_cast<core::u8>(i);
        out[cursor++] = static_cast<core::u8>(operandCount(opcode));

        const char *name = opcodeName(opcode);
        for (core::u32 c = 0u; c < kMnemonicBytes; ++c)
            out[cursor + c] = name[c] == '\0' ? core::u8{0} : static_cast<core::u8>(name[c]);
        // The mnemonic is copied up to the first NUL and the rest is zeroed above; a
        // name longer than the field is truncated rather than allowed to run into the
        // next row, because a row that overruns takes the whole table with it.
        for (core::u32 c = 0u; c < kMnemonicBytes; ++c)
        {
            if (name[c] == '\0')
            {
                for (core::u32 pad = c; pad < kMnemonicBytes; ++pad)
                    out[cursor + pad] = 0u;
                break;
            }
        }
        cursor += kMnemonicBytes;
    }

    return cursor;
}

bool readSpecification(const core::u8 *bytes, core::u32 size, IsaTable &outTable) noexcept
{
    outTable = IsaTable{};
    if (bytes == nullptr || size < 16u)
        return false;

    for (core::u32 i = 0u; i < 4u; ++i)
        if (bytes[i] != kSpecificationMagic[i])
            return false;

    outTable.instructionBytes = readWord(bytes + 4u);
    outTable.registerCount = readWord(bytes + 8u);
    const core::u32 declared = readWord(bytes + 12u);

    const core::u32 rowBytes = 2u + kMnemonicBytes;
    const core::u32 capacity = static_cast<core::u32>(Opcode::Count);
    const core::u32 rows = declared > capacity ? capacity : declared;

    // A count from an engraving is input, not a promise: a plate that lost a corner
    // can name more rows than it carries bytes for.
    core::u32 cursor = 16u;
    for (core::u32 i = 0u; i < rows; ++i)
    {
        if (cursor + rowBytes > size)
            break;
        outTable.entry[outTable.count].opcode = bytes[cursor];
        outTable.entry[outTable.count].operands = bytes[cursor + 1u];
        for (core::u32 c = 0u; c < kMnemonicBytes; ++c)
            outTable.entry[outTable.count].mnemonic[c] = static_cast<char>(bytes[cursor + 2u + c]);
        ++outTable.count;
        cursor += rowBytes;
    }

    return outTable.count != 0u;
}

} // namespace lpl::rosetta
