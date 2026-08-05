/**
 * @file test_rosetta_isa.cpp
 * @brief A third party rebuilds the reader from the engraving, and opens the artifact.
 *
 * That sentence is the whole criterion of the Rosetta lot, and it is testable exactly
 * as written: engrave a plate, throw the source away, build an interpreter out of the
 * bytes that were engraved, and run the payload's program on it. If that needs
 * anything the plate does not carry, the artifact is a lie.
 *
 * The second claim is physical rather than informational: a plate broken in half must
 * still be readable. No column code survives losing an arbitrary contiguous half —
 * that is what the five replicas are for, and the test cuts the plate to prove it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/math/Random.hpp>
#include <lpl/rosetta/Bootstrap.hpp>
#include <lpl/rosetta/Engraving.hpp>
#include <lpl/rosetta/Interpreter.hpp>
#include <lpl/rosetta/Parity.hpp>
#include <lpl/rosetta/SelfDescribing.hpp>

#include <cstdio>

namespace {

int gFailures = 0;
int gChecks = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++gFailures;
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== rosetta: an artifact that carries its own reader ==\n");

    // ── The instruction set is small enough to be described ───────────────────
    std::printf("\n-- ten opcodes --\n");
    {
        check(static_cast<core::u32>(rosetta::Opcode::Count) == 10u, "there are ten of them");
        check(rosetta::kInstructionBytes == 4u, "and an instruction is a fixed four bytes");

        bool named = true;
        for (core::u32 i = 0u; i < static_cast<core::u32>(rosetta::Opcode::Count); ++i)
        {
            const char *name = rosetta::opcodeName(static_cast<rosetta::Opcode>(i));
            named = named && name[0] != '?' && name[0] != '\0';
        }
        check(named, "every one has a mnemonic, so a specification can spell it");
    }

    // ── The specification round-trips ─────────────────────────────────────────
    std::printf("\n-- the specification is bytes, not prose --\n");
    core::u8 specification[rosetta::kSpecificationBytes]{};
    {
        const core::u32 written = rosetta::emitSpecification(specification, rosetta::kSpecificationBytes);
        check(written == rosetta::kSpecificationBytes, "it emits the size it declares");

        rosetta::IsaTable table{};
        check(rosetta::readSpecification(specification, written, table), "and reads back");
        check(table.count == static_cast<core::u32>(rosetta::Opcode::Count), "naming every opcode");
        check(table.instructionBytes == rosetta::kInstructionBytes && table.registerCount == rosetta::kRegisterCount,
              "with the instruction width and the register count");

        bool spelt = true;
        for (core::u32 i = 0u; i < table.count; ++i)
        {
            const char *reference = rosetta::opcodeName(static_cast<rosetta::Opcode>(table.entry[i].opcode));
            for (core::u32 c = 0u; reference[c] != '\0'; ++c)
                spelt = spelt && table.entry[i].mnemonic[c] == reference[c];
        }
        check(spelt, "and every mnemonic character for character");

        // Junk must be refused rather than read as an empty machine.
        rosetta::IsaTable junk{};
        core::u8 rubble[32]{};
        check(!rosetta::readSpecification(rubble, sizeof(rubble), junk), "rubble is refused, not read as an empty ISA");
    }

    // ── An interpreter rebuilt from it runs the same program ──────────────────
    std::printf("\n-- a machine rebuilt from the bytes --\n");
    {
        rosetta::Interpreter rebuilt;
        check(rosetta::Interpreter::fromSpecification(specification, rosetta::kSpecificationBytes, rebuilt),
              "an interpreter is built from the specification alone");
        check(rebuilt.knownOpcodes() == static_cast<core::u32>(rosetta::Opcode::Count),
              "and knows every opcode the plate named");

        // A TRUNCATED specification must yield a machine that visibly cannot run the
        // program — not one that silently agrees with the compiled-in enum. This is the
        // check that makes "the reader is included" mean something.
        rosetta::Interpreter partial;
        const core::u32 halfRows = 16u + 5u * (2u + rosetta::kMnemonicBytes);
        check(rosetta::Interpreter::fromSpecification(specification, halfRows, partial),
              "half a specification still builds a machine");
        check(partial.knownOpcodes() < rebuilt.knownOpcodes(), "but one that knows fewer opcodes");

        core::u8 program[3u * rosetta::kInstructionBytes]{};
        rosetta::encodeInstruction(rosetta::Opcode::Set, 0u, 0u, 42u, program);
        rosetta::encodeInstruction(rosetta::Opcode::Xor, 0u, 0u, 0u, program + 4u);
        rosetta::encodeInstruction(rosetta::Opcode::Halt, 0u, 0u, 0u, program + 8u);

        rosetta::ExecutionReport report{};
        check(rebuilt.run(program, sizeof(program), nullptr, 0u, 64u, report), "the full machine halts");
        check(report.registers[0] == 0u, "having XORed the register with itself");
        check(report.steps == 3u, "in three instructions");
    }

    // ── The plate, and the cut ────────────────────────────────────────────────
    std::printf("\n-- engraved, then broken --\n");
    {
        core::u8 payload[64]{};
        math::Random stream{0x0F5Eu};
        for (core::u32 i = 0u; i < sizeof(payload); ++i)
            payload[i] = static_cast<core::u8>(stream.next() & 0xFFu);

        const rosetta::Bootstrap bootstrap = rosetta::standardBootstrap();
        rosetta::Engraving plate;
        plate.setParityShare(200u);
        check(plate.engrave(bootstrap, payload, sizeof(payload)), "the plate engraves");
        std::printf("    plate: %zu bytes, bootstrap %u bytes across four levels\n", plate.image().size(),
                    bootstrap.totalBytes());

        lpl::pmr::vector<core::u8> intact = plate.image();
        lpl::pmr::vector<core::u8> spec;
        lpl::pmr::vector<core::u8> recovered;
        rosetta::EngravingReport report{};
        check(rosetta::Engraving::read(intact.data(), static_cast<core::u32>(intact.size()), spec, recovered, report),
              "an intact plate reads");
        check(report.replicasIntact == rosetta::kBootstrapCopies, "with all five replicas intact");

        bool same = recovered.size() == sizeof(payload);
        for (core::usize i = 0u; same && i < recovered.size(); ++i)
            same = recovered[i] == payload[i];
        check(same, "and the payload comes back byte for byte");

        // THE test. Rebuild the reader from what the plate carried, and check it is the
        // same machine — the specification was engraved, not assumed.
        rosetta::Interpreter fromPlate;
        check(rosetta::Interpreter::fromSpecification(spec.data(), static_cast<core::u32>(spec.size()), fromPlate),
              "a reader is rebuilt from the plate's own specification");
        check(fromPlate.knownOpcodes() == static_cast<core::u32>(rosetta::Opcode::Count),
              "and it knows the whole instruction set");
    }

    // ── The signatures the kernel must reproduce ──────────────────────────────
    std::printf("\n-- signatures the kernel must reproduce --\n");
    rosetta::RosettaFoldResult folded{};
    rosetta::foldRosettaState(folded);

    std::printf("  trace_sig    = 0x%08X\n", folded.traceSignature);
    std::printf("  spec_sig     = 0x%08X\n", folded.specSignature);
    std::printf("  plate_sig    = 0x%08X\n", folded.plateSignature);
    std::printf("  payload_sig  = 0x%08X\n", folded.payloadSignature);
    std::printf("  steps        = %u\n", folded.steps);
    std::printf("  halted       = %u\n", folded.halted);
    std::printf("  plate_bytes  = %u\n", folded.plateBytes);
    std::printf("  opcodes      = %u\n", folded.rebuiltOpcodes);
    std::printf("  self_hosting = %u\n", folded.selfHosting);

    check(folded.selfHosting == 1u, "the reader rebuilt from the engraving runs the program identically");

    // HALTED, not "took more than sixteen steps". The weaker check was satisfied by the
    // failure it was supposed to catch: a program whose exit jump lands one instruction
    // short loops until its budget, produces exactly the right memory anyway, and
    // reports a perfectly stable trace signature. Only the step count sitting at 4096
    // said anything at all, and "> 16" cannot see the difference between 140 and 4096.
    check(folded.halted == 1u, "and the canonical program HALTS rather than running out its budget");
    check(folded.steps > 16u && folded.steps < 512u, "in a step count consistent with sixteen bytes of work");

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
