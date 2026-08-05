/**
 * @file Parity.cpp
 * @brief The canonical program and the canonical plate.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/rosetta/Parity.hpp>

#include <lpl/rosetta/Bootstrap.hpp>
#include <lpl/rosetta/Engraving.hpp>
#include <lpl/rosetta/Interpreter.hpp>
#include <lpl/rosetta/SelfDescribing.hpp>

namespace lpl::rosetta {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief FNV-1a over a byte span.
 * @param bytes Span.
 * @param size  Its length.
 * @return The signature.
 */
[[nodiscard]] core::u32 foldBytes(const core::u8 *bytes, core::u32 size) noexcept
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < size; ++i)
        hash = (hash ^ bytes[i]) * kFnv1aPrime;
    return hash;
}

/**
 * @brief The canonical program: XOR a sixteen-byte buffer with a rolling key.
 *
 * Chosen because it exercises every class of opcode the ISA has — a constant, a load,
 * arithmetic, a store, a conditional and a jump — in the shortest program that does
 * something a reader would recognise as work. A program that only added numbers would
 * leave the memory opcodes untested, and those are the ones a decompressor needs.
 *
 * @param out Receives the instruction stream.
 */
void buildParityProgram(lpl::pmr::vector<core::u8> &out)
{
    out.clear();
    const auto emit = [&out](Opcode opcode, core::u8 a, core::u8 b, core::u8 c) {
        core::u8 word[kInstructionBytes]{};
        encodeInstruction(opcode, a, b, c, word);
        for (core::u32 i = 0u; i < kInstructionBytes; ++i)
            out.push_back(word[i]);
    };

    // r0 = index, r1 = limit, r2 = scratch, r3 = key, r4 = one
    emit(Opcode::Set, 0u, 0u, 0u);    // 0: index = 0
    emit(Opcode::Set, 1u, 0u, 16u);   // 1: limit = 16
    emit(Opcode::Set, 3u, 0u, 0x5Au); // 2: key = 0x5A
    emit(Opcode::Set, 4u, 0u, 1u);    // 3: one = 1
    emit(Opcode::Load, 2u, 0u, 0u);   // 4: scratch = memory[index]
    emit(Opcode::Xor, 2u, 2u, 3u);    // 5: scratch ^= key
    emit(Opcode::Store, 2u, 0u, 0u);  // 6: memory[index] = scratch
    emit(Opcode::Add, 3u, 3u, 4u);    // 7: key += 1  (rolling)
    emit(Opcode::Add, 0u, 0u, 4u);    // 8: ++index
    emit(Opcode::Sub, 5u, 0u, 1u);    // 9: r5 = index - limit
    // 12, not 11. Aiming the exit at 11 lands on the JUMP that closes the loop, so
    // the program runs until its budget rather than halting — and it does so while
    // producing exactly the right memory, because the loop is idempotent past the
    // sixteenth byte. The trace signature was stable, the payload was correct, and the
    // only thing that said anything was wrong was the step count sitting at the budget.
    emit(Opcode::JumpIfZero, 5u, 0u, 12u); // 10: done when equal
    emit(Opcode::Jump, 0u, 0u, 4u);        // 11: otherwise round again
    emit(Opcode::Halt, 0u, 0u, 0u);        // 12
}

} // namespace

void foldRosettaState(RosettaFoldResult &out)
{
    out = RosettaFoldResult{};

    // ── The machine ───────────────────────────────────────────────────────────
    lpl::pmr::vector<core::u8> program;
    buildParityProgram(program);

    core::u8 memory[16]{};
    for (core::u32 i = 0u; i < 16u; ++i)
        memory[i] = static_cast<core::u8>(i * 7u + 3u);

    const Interpreter machine = Interpreter::reference();
    ExecutionReport execution{};
    (void) machine.run(program.data(), static_cast<core::u32>(program.size()), memory, 16u, 4096u, execution);

    out.traceSignature = execution.traceSignature;
    out.steps = execution.steps;
    out.halted = execution.halted ? 1u : 0u;

    // ── The plate ─────────────────────────────────────────────────────────────
    core::u8 specification[kSpecificationBytes]{};
    const core::u32 specBytes = emitSpecification(specification, kSpecificationBytes);
    out.specSignature = foldBytes(specification, specBytes);

    const Bootstrap bootstrap = standardBootstrap();
    Engraving plate;
    plate.setMedium(Medium::FusedQuartz);
    plate.setParityShare(200u);
    if (!plate.engrave(bootstrap, memory, 16u))
        return;

    out.plateBytes = static_cast<core::u32>(plate.image().size());
    out.plateSignature = foldBytes(plate.image().data(), out.plateBytes);

    // ── The only honest test: rebuild the reader from what was engraved ───────
    lpl::pmr::vector<core::u8> working = plate.image();
    lpl::pmr::vector<core::u8> engravedSpec;
    lpl::pmr::vector<core::u8> readPayload;
    EngravingReport report{};
    if (!Engraving::read(working.data(), static_cast<core::u32>(working.size()), engravedSpec, readPayload, report))
        return;

    out.payloadSignature = foldBytes(readPayload.data(), static_cast<core::u32>(readPayload.size()));

    Interpreter rebuilt;
    if (!Interpreter::fromSpecification(engravedSpec.data(), static_cast<core::u32>(engravedSpec.size()), rebuilt))
        return;
    out.rebuiltOpcodes = rebuilt.knownOpcodes();

    // The rebuilt machine must run the same program to the same trace. If it cannot,
    // the specification the plate carries is not enough to rebuild the reader — which
    // is the one thing the whole artifact claims.
    core::u8 rebuiltMemory[16]{};
    for (core::u32 i = 0u; i < 16u; ++i)
        rebuiltMemory[i] = static_cast<core::u8>(i * 7u + 3u);
    ExecutionReport rebuiltRun{};
    (void) rebuilt.run(program.data(), static_cast<core::u32>(program.size()), rebuiltMemory, 16u, 4096u, rebuiltRun);

    out.selfHosting = (rebuiltRun.traceSignature == execution.traceSignature && report.payloadRecovered) ? 1u : 0u;
}

} // namespace lpl::rosetta
