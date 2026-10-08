#include <lpl/math/Random.hpp>
#include <lpl/rosetta/Bootstrap.hpp>
#include <lpl/rosetta/Engraving.hpp>
#include <lpl/rosetta/Interpreter.hpp>
#include <lpl/rosetta/SelfDescribing.hpp>
#include <lpl/std/vector.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(rosetta);

namespace {

constexpr lpl::core::u32 kOpcodeCount = static_cast<lpl::core::u32>(lpl::rosetta::Opcode::Count);
constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;
constexpr lpl::core::u32 kCanonicalMemoryBytes = 16u;
constexpr lpl::core::u32 kCanonicalStepBudget = 4096u;

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

/**
 * @struct Instruction
 * @brief One instruction of the canonical program, before it is encoded.
 */
struct Instruction {
    lpl::rosetta::Opcode opcode; /**< What the instruction does. */
    lpl::core::u8 a;             /**< First operand. */
    lpl::core::u8 b;             /**< Second operand. */
    lpl::core::u8 c;             /**< Third operand. */
};

/**
 * @brief The canonical program: XOR a sixteen-byte buffer with a rolling key.
 *
 * @details It exercises every class of opcode the ISA has (a constant, a load, arithmetic, a store,
 *          a conditional and a jump) in the shortest program that does work a reader would
 *          recognise. A program that only added numbers would leave the memory opcodes untested,
 *          and those are the ones a decompressor needs. The registers are r0 the index, r1 the
 *          limit, r2 scratch, r3 the key and r4 one.
 *
 *          The exit jumps to 12, not 11: aimed at 11 it lands on the jump that closes the loop, so
 *          the program runs until its budget instead of halting, while producing exactly the right
 *          memory, because the loop is idempotent past the sixteenth byte. The trace signature was
 *          stable and the payload correct; only the step count sitting at the budget said that
 *          anything was wrong.
 */
constexpr Instruction kCanonicalProgram[] = {
    {lpl::rosetta::Opcode::Set,        0u, 0u, 0u   }, // 0: index = 0
    {lpl::rosetta::Opcode::Set,        1u, 0u, 16u  }, // 1: limit = 16
    {lpl::rosetta::Opcode::Set,        3u, 0u, 0x5Au}, // 2: key = 0x5A
    {lpl::rosetta::Opcode::Set,        4u, 0u, 1u   }, // 3: one = 1
    {lpl::rosetta::Opcode::Load,       2u, 0u, 0u   }, // 4: scratch = memory[index]
    {lpl::rosetta::Opcode::Xor,        2u, 2u, 3u   }, // 5: scratch ^= key
    {lpl::rosetta::Opcode::Store,      2u, 0u, 0u   }, // 6: memory[index] = scratch
    {lpl::rosetta::Opcode::Add,        3u, 3u, 4u   }, // 7: key += 1, rolling
    {lpl::rosetta::Opcode::Add,        0u, 0u, 4u   }, // 8: ++index
    {lpl::rosetta::Opcode::Sub,        5u, 0u, 1u   }, // 9: r5 = index - limit
    {lpl::rosetta::Opcode::JumpIfZero, 5u, 0u, 12u  }, // 10: done when equal
    {lpl::rosetta::Opcode::Jump,       0u, 0u, 4u   }, // 11: otherwise round again
    {lpl::rosetta::Opcode::Halt,       0u, 0u, 0u   }, // 12
};

/**
 * @struct RosettaFoldResult
 * @brief What gate P12 rosetta records: the signatures both targets must reproduce, and the
 *        counters behind them.
 */
struct RosettaFoldResult {
    lpl::core::u32 traceSignature{0u};   /**< Fold of every instruction retired and its result. */
    lpl::core::u32 specSignature{0u};    /**< Fold of the engraved instruction-set description. */
    lpl::core::u32 plateSignature{0u};   /**< Fold of the whole plate image. */
    lpl::core::u32 payloadSignature{0u}; /**< Fold of the payload read back off the plate. */
    lpl::core::u32 steps{0u};            /**< Instructions the canonical program retired. */
    lpl::core::u32 halted{0u};           /**< 1 when it reached HALT rather than its budget. */
    lpl::core::u32 plateBytes{0u};       /**< Size of the plate. */
    lpl::core::u32 rebuiltOpcodes{0u};   /**< Opcodes an interpreter rebuilt from the plate knows. */
    lpl::core::u32 selfHosting{0u};      /**< 1 when the rebuilt reader decoded the plate. */
};

[[nodiscard]] lpl::core::u32 foldBytes(const lpl::core::u8 *bytes, lpl::core::u32 size) noexcept
{
    lpl::core::u32 hash = kFnv1aOffsetBasis;

    for (lpl::core::u32 index = 0u; index < size; ++index)
        hash = (hash ^ bytes[index]) * kFnv1aPrime;
    return hash;
}

void buildCanonicalProgram(lpl::pmr::vector<lpl::core::u8> &out)
{
    out.clear();
    for (const Instruction &instruction : kCanonicalProgram)
    {
        lpl::core::u8 word[lpl::rosetta::kInstructionBytes]{};

        lpl::rosetta::encodeInstruction(instruction.opcode, instruction.a, instruction.b, instruction.c, word);
        for (lpl::core::u32 index = 0u; index < lpl::rosetta::kInstructionBytes; ++index)
            out.push_back(word[index]);
    }
}

/**
 * @brief The memory the canonical program XORs: byte i holds 7i + 3.
 */
void fillCanonicalMemory(lpl::core::u8 (&memory)[kCanonicalMemoryBytes])
{
    for (lpl::core::u32 index = 0u; index < kCanonicalMemoryBytes; ++index)
        memory[index] = static_cast<lpl::core::u8>(index * 7u + 3u);
}

/**
 * @brief Runs the canonical program, engraves the canonical plate, and folds both.
 *
 * @details The stage that proves something is the last one: a reader rebuilt from what was
 *          engraved runs the same program to the same trace. If it cannot, the specification the
 *          plate carries is not enough to rebuild the reader, which is the one thing the whole
 *          artifact claims.
 *
 * @param out Receives the signatures; a stage that fails leaves the later fields at zero.
 */
void foldRosettaState(RosettaFoldResult &out)
{
    out = RosettaFoldResult{};

    lpl::pmr::vector<lpl::core::u8> program;
    lpl::core::u8 memory[kCanonicalMemoryBytes]{};
    const lpl::rosetta::Interpreter machine = lpl::rosetta::Interpreter::reference();
    lpl::rosetta::ExecutionReport execution{};

    buildCanonicalProgram(program);
    fillCanonicalMemory(memory);
    (void) machine.run(program.data(), static_cast<lpl::core::u32>(program.size()), memory, kCanonicalMemoryBytes,
                       kCanonicalStepBudget, execution);
    out.traceSignature = execution.traceSignature;
    out.steps = execution.steps;
    out.halted = execution.halted ? 1u : 0u;

    lpl::core::u8 specification[lpl::rosetta::kSpecificationBytes]{};
    const lpl::core::u32 specificationBytes =
        lpl::rosetta::emitSpecification(specification, lpl::rosetta::kSpecificationBytes);

    out.specSignature = foldBytes(specification, specificationBytes);

    const lpl::rosetta::Bootstrap bootstrap = lpl::rosetta::standardBootstrap();
    lpl::rosetta::Engraving plate;

    plate.setMedium(lpl::rosetta::Medium::FusedQuartz);
    plate.setParityShare(200u);
    if (!plate.engrave(bootstrap, memory, kCanonicalMemoryBytes))
        return;
    out.plateBytes = static_cast<lpl::core::u32>(plate.image().size());
    out.plateSignature = foldBytes(plate.image().data(), out.plateBytes);

    lpl::pmr::vector<lpl::core::u8> working = plate.image();
    lpl::pmr::vector<lpl::core::u8> engravedSpecification;
    lpl::pmr::vector<lpl::core::u8> readPayload;
    lpl::rosetta::EngravingReport report{};

    if (!lpl::rosetta::Engraving::read(working.data(), static_cast<lpl::core::u32>(working.size()),
                                       engravedSpecification, readPayload, report))
        return;
    out.payloadSignature = foldBytes(readPayload.data(), static_cast<lpl::core::u32>(readPayload.size()));

    lpl::rosetta::Interpreter rebuilt;

    if (!lpl::rosetta::Interpreter::fromSpecification(
            engravedSpecification.data(), static_cast<lpl::core::u32>(engravedSpecification.size()), rebuilt))
        return;
    out.rebuiltOpcodes = rebuilt.knownOpcodes();

    lpl::core::u8 rebuiltMemory[kCanonicalMemoryBytes]{};
    lpl::rosetta::ExecutionReport rebuiltRun{};

    fillCanonicalMemory(rebuiltMemory);
    (void) rebuilt.run(program.data(), static_cast<lpl::core::u32>(program.size()), rebuiltMemory,
                       kCanonicalMemoryBytes, kCanonicalStepBudget, rebuiltRun);
    out.selfHosting = (rebuiltRun.traceSignature == execution.traceSignature && report.payloadRecovered) ? 1u : 0u;
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
    RosettaFoldResult folded{};

    foldRosettaState(folded);
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
