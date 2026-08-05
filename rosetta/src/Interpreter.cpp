/**
 * @file Interpreter.cpp
 * @brief The reference machine: eight registers, a flat memory, no traps.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/rosetta/Interpreter.hpp>

#include <lpl/rosetta/SelfDescribing.hpp>

namespace lpl::rosetta {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Folds one word into a running FNV-1a hash.
 * @param hash Running value.
 * @param word Word to absorb.
 */
void fold(core::u32 &hash, core::u32 word) noexcept { hash = (hash ^ word) * kFnv1aPrime; }

} // namespace

Interpreter Interpreter::reference() noexcept
{
    Interpreter machine;
    for (core::u32 i = 0u; i < static_cast<core::u32>(Opcode::Count); ++i)
        machine._implemented[i] = 1u;
    machine._knownOpcodes = static_cast<core::u32>(Opcode::Count);
    return machine;
}

bool Interpreter::fromSpecification(const core::u8 *specification, core::u32 size, Interpreter &outMachine) noexcept
{
    outMachine = Interpreter{};

    IsaTable table{};
    if (!readSpecification(specification, size, table))
        return false;
    if (table.count == 0u)
        return false;

    for (core::u32 i = 0u; i < table.count; ++i)
    {
        const core::u8 opcode = table.entry[i].opcode;
        if (opcode >= static_cast<core::u32>(Opcode::Count))
            continue;
        if (outMachine._implemented[opcode] == 0u)
        {
            outMachine._implemented[opcode] = 1u;
            ++outMachine._knownOpcodes;
        }
    }
    return outMachine._knownOpcodes != 0u;
}

bool Interpreter::knows(core::u8 opcode) const noexcept
{
    return opcode < static_cast<core::u32>(Opcode::Count) && _implemented[opcode] != 0u;
}

bool Interpreter::run(const core::u8 *program, core::u32 programSize, core::u8 *memory, core::u32 memorySize,
                      core::u32 budget, ExecutionReport &outReport) const noexcept
{
    outReport = ExecutionReport{};
    if (program == nullptr || programSize < kInstructionBytes)
        return false;

    core::u32 registers[kRegisterCount]{};
    const core::u32 instructions = programSize / kInstructionBytes;
    core::u32 counter = 0u;
    core::u32 hash = kFnv1aOffsetBasis;

    while (outReport.steps < budget)
    {
        if (counter >= instructions)
            break;

        const core::u8 *const word = program + static_cast<core::usize>(counter) * kInstructionBytes;
        const core::u8 rawOpcode = word[0];
        const core::u8 a = word[1];
        const core::u8 b = word[2];
        const core::u8 c = word[3];

        // An opcode this machine was never told about stops it. Not a trap and not a
        // no-op: a reader rebuilt from an incomplete specification has to FAIL, or the
        // engraving could omit half the ISA and nothing would say so.
        if (!knows(rawOpcode))
            break;

        ++outReport.steps;
        fold(hash, (static_cast<core::u32>(rawOpcode) << 24) | (static_cast<core::u32>(a) << 16) |
                       (static_cast<core::u32>(b) << 8) | c);

        const auto reg = [&registers](core::u8 index) -> core::u32 & { return registers[index % kRegisterCount]; };
        bool jumped = false;

        switch (static_cast<Opcode>(rawOpcode))
        {
        case Opcode::Halt: outReport.halted = true; break;
        case Opcode::Set: reg(a) = (static_cast<core::u32>(b) << 8) | c; break;
        case Opcode::Load: {
            const core::u32 address = reg(b) + c;
            reg(a) = address < memorySize && memory != nullptr ? memory[address] : 0u;
            break;
        }
        case Opcode::Store: {
            const core::u32 address = reg(b) + c;
            if (address < memorySize && memory != nullptr)
                memory[address] = static_cast<core::u8>(reg(a) & 0xFFu);
            break;
        }
        case Opcode::Add: reg(a) = reg(b) + reg(c); break;
        case Opcode::Sub: reg(a) = reg(b) - reg(c); break;
        case Opcode::Xor: reg(a) = reg(b) ^ reg(c); break;
        case Opcode::Shift:
            reg(a) = c < 128u ? (reg(b) << (c % 32u)) : (reg(b) >> ((c - 128u) % 32u));
            break;
        case Opcode::Jump:
            counter = (static_cast<core::u32>(b) << 8) | c;
            jumped = true;
            break;
        case Opcode::JumpIfZero:
            if (reg(a) == 0u)
            {
                counter = (static_cast<core::u32>(b) << 8) | c;
                jumped = true;
            }
            break;
        case Opcode::Count: break;
        }

        fold(hash, reg(a));

        if (outReport.halted)
            break;
        if (!jumped)
            ++counter;
    }

    for (core::u32 i = 0u; i < kRegisterCount; ++i)
    {
        outReport.registers[i] = registers[i];
        fold(hash, registers[i]);
    }
    outReport.traceSignature = hash;
    return outReport.halted;
}

} // namespace lpl::rosetta
