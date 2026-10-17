#ifndef LPL_ROSETTA_MINIMALISA_INL
#define LPL_ROSETTA_MINIMALISA_INL

namespace lpl::rosetta {

constexpr const char *opcodeName(Opcode opcode) noexcept
{
    const auto index = static_cast<core::u32>(opcode);
    return index < static_cast<core::u32>(Opcode::Count) ? kOpcodeMnemonics[index] : "?";
}

constexpr core::u32 operandCount(Opcode opcode) noexcept
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

constexpr void encodeInstruction(Opcode opcode, core::u8 a, core::u8 b, core::u8 c, core::u8 *out) noexcept
{
    out[0] = static_cast<core::u8>(opcode);
    out[1] = a;
    out[2] = b;
    out[3] = c;
}

} // namespace lpl::rosetta

#endif // LPL_ROSETTA_MINIMALISA_INL
