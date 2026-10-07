#include <lpl/math/Random.hpp>
#include <lpl/rosetta/Bootstrap.hpp>
#include <lpl/rosetta/Engraving.hpp>
#include <lpl/rosetta/Interpreter.hpp>
#include <lpl/rosetta/Parity.hpp>
#include <lpl/rosetta/SelfDescribing.hpp>
#include <lpl/std/vector.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(rosetta);

namespace {

constexpr lpl::core::u32 kOpcodeCount = static_cast<lpl::core::u32>(lpl::rosetta::Opcode::Count);

/**
 * @brief The specification the plate carries, emitted once per test that reads it.
 */
struct Specification {
    lpl::core::u8 bytes[lpl::rosetta::kSpecificationBytes]{};
    lpl::core::u32 written = 0u;

    Specification() : written(lpl::rosetta::emitSpecification(bytes, lpl::rosetta::kSpecificationBytes)) {}
};

[[nodiscard]] bool spellsEveryMnemonic(const lpl::rosetta::IsaTable &table)
{
    bool spelt = true;

    for (lpl::core::u32 entry = 0u; entry < table.count; ++entry)
    {
        const char *reference = lpl::rosetta::opcodeName(static_cast<lpl::rosetta::Opcode>(table.entry[entry].opcode));

        for (lpl::core::u32 character = 0u; reference[character] != '\0'; ++character)
            spelt = spelt && table.entry[entry].mnemonic[character] == reference[character];
    }
    return spelt;
}

} // namespace

LPL_TEST(ten_opcodes_each_with_a_mnemonic)
{
    bool named = true;

    for (lpl::core::u32 opcode = 0u; opcode < kOpcodeCount; ++opcode)
    {
        const char *name = lpl::rosetta::opcodeName(static_cast<lpl::rosetta::Opcode>(opcode));

        named = named && name[0] != '?' && name[0] != '\0';
    }
    test.check(kOpcodeCount == 10u, "the instruction set has ten opcodes");
    test.check(lpl::rosetta::kInstructionBytes == 4u, "an instruction is four bytes");
    test.check(named, "and every opcode has a mnemonic a specification can spell");
}

LPL_TEST(specification_is_bytes_that_read_back)
{
    const Specification specification;
    lpl::rosetta::IsaTable table{};
    lpl::rosetta::IsaTable junk{};
    lpl::core::u8 rubble[32]{};

    test.check(specification.written == lpl::rosetta::kSpecificationBytes, "the specification is the size it declares");
    if (!test.check(lpl::rosetta::readSpecification(specification.bytes, specification.written, table),
                    "it reads back"))
        return;
    test.check(table.count == kOpcodeCount, "naming every opcode");
    test.check(table.instructionBytes == lpl::rosetta::kInstructionBytes &&
                   table.registerCount == lpl::rosetta::kRegisterCount,
               "with the instruction width and the register count");
    test.check(spellsEveryMnemonic(table), "and every mnemonic, character for character");
    test.check(!lpl::rosetta::readSpecification(rubble, sizeof(rubble), junk),
               "rubble is refused, not read as an empty machine");
}

/**
 * @brief A machine built from the specification alone runs a program; built from half of it, it
 *        knows fewer opcodes instead of falling back on the compiled-in ones.
 */
LPL_TEST(machine_rebuilt_from_the_bytes_runs_a_program)
{
    const Specification specification;
    lpl::rosetta::Interpreter rebuilt;
    lpl::rosetta::Interpreter partial;
    const lpl::core::u32 halfRows = 16u + 5u * (2u + lpl::rosetta::kMnemonicBytes);
    lpl::core::u8 program[3u * lpl::rosetta::kInstructionBytes]{};
    lpl::rosetta::ExecutionReport report{};

    if (!test.check(lpl::rosetta::Interpreter::fromSpecification(specification.bytes, specification.written, rebuilt),
                    "an interpreter is built from the specification alone"))
        return;
    test.check(rebuilt.knownOpcodes() == kOpcodeCount, "and knows every opcode the specification names");
    test.check(lpl::rosetta::Interpreter::fromSpecification(specification.bytes, halfRows, partial),
               "half a specification still builds a machine");
    test.check(partial.knownOpcodes() < rebuilt.knownOpcodes(), "one that knows fewer opcodes");

    lpl::rosetta::encodeInstruction(lpl::rosetta::Opcode::Set, 0u, 0u, 42u, program);
    lpl::rosetta::encodeInstruction(lpl::rosetta::Opcode::Xor, 0u, 0u, 0u, program + 4u);
    lpl::rosetta::encodeInstruction(lpl::rosetta::Opcode::Halt, 0u, 0u, 0u, program + 8u);
    test.check(rebuilt.run(program, sizeof(program), nullptr, 0u, 64u, report), "the rebuilt machine halts");
    test.check(report.registers[0] == 0u, "having XORed the register with itself");
    test.check(report.steps == 3u, "in three instructions");
}

/**
 * @brief An engraved plate gives back its payload and the specification of its own reader.
 */
LPL_TEST(engraved_plate_carries_its_reader)
{
    lpl::core::u8 payload[64]{};
    lpl::math::Random stream{0x0F5Eu};
    lpl::rosetta::Engraving plate;
    lpl::pmr::vector<lpl::core::u8> specification;
    lpl::pmr::vector<lpl::core::u8> recovered;
    lpl::rosetta::EngravingReport report{};
    lpl::rosetta::Interpreter fromPlate;

    for (lpl::core::u32 index = 0u; index < sizeof(payload); ++index)
        payload[index] = static_cast<lpl::core::u8>(stream.next() & 0xFFu);
    plate.setParityShare(200u);
    if (!test.check(plate.engrave(lpl::rosetta::standardBootstrap(), payload, sizeof(payload)), "the plate engraves"))
        return;

    lpl::pmr::vector<lpl::core::u8> image = plate.image();

    if (!test.check(lpl::rosetta::Engraving::read(image.data(), static_cast<lpl::core::u32>(image.size()),
                                                  specification, recovered, report),
                    "the intact plate reads"))
        return;
    test.check(report.replicasIntact == lpl::rosetta::kBootstrapCopies, "with every replica intact");

    bool same = recovered.size() == sizeof(payload);

    for (lpl::core::usize index = 0u; same && index < recovered.size(); ++index)
        same = recovered[index] == payload[index];
    test.check(same, "the payload comes back byte for byte");
    test.check(lpl::rosetta::Interpreter::fromSpecification(
                   specification.data(), static_cast<lpl::core::u32>(specification.size()), fromPlate),
               "a reader is rebuilt from the plate's own specification");
    test.check(fromPlate.knownOpcodes() == kOpcodeCount, "and it knows the whole instruction set");
}

/**
 * @brief Gate P12 rosetta: the canonical program, run by the reader rebuilt from the engraving,
 *        halts and folds the same trace, plate and payload on both targets.
 *
 * @details Halting matters more than the step count: a program whose exit jump lands short loops
 *          until its budget and still folds a stable trace.
 */
LPL_TEST(rebuilt_reader_runs_the_canonical_program)
{
    lpl::rosetta::RosettaFoldResult folded{};

    lpl::rosetta::foldRosettaState(folded);
    test.check(folded.selfHosting == 1u, "the reader rebuilt from the engraving runs the program identically");
    test.check(folded.halted == 1u, "the program halts instead of running out its budget");
    test.check(folded.steps > 16u && folded.steps < 512u, "in a step count fitting sixteen bytes of work");
    test.check(folded.rebuiltOpcodes == kOpcodeCount, "and the rebuilt reader knows every opcode");

    test.measureHexadecimal("trace_signature", folded.traceSignature);
    test.measureHexadecimal("specification_signature", folded.specSignature);
    test.measureHexadecimal("plate_signature", folded.plateSignature);
    test.measureHexadecimal("payload_signature", folded.payloadSignature);
    test.measure("steps", folded.steps);
    test.measure("plate_bytes", folded.plateBytes);
}
