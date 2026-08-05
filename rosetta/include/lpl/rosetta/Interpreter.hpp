/**
 * @file Interpreter.hpp
 * @brief The reference interpreter for that instruction set.
 *
 * Exists so the engraved specification can be tested against something rather
 * than merely asserted to be sufficient.
 *
 * The machine is deliberately boring: eight registers, a flat byte memory, a program
 * counter in instructions, and a step budget. Nothing traps, nothing faults, nothing
 * is undefined — a machine with a fault model has more specification to transmit, and
 * everything transmitted is something that can be lost.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_ROSETTA_INTERPRETER_HPP
#    define LPL_LPL_ROSETTA_INTERPRETER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/rosetta/MinimalIsa.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::rosetta {

/**
 * @struct ExecutionReport
 * @brief What running a program did, in numbers a test can compare.
 */
struct ExecutionReport {
    bool halted{false};                    ///< Reached a HALT rather than the budget.
    core::u32 steps{0u};                   ///< Instructions retired.
    core::u32 traceSignature{0u};          ///< FNV-1a over every retired instruction and its result.
    core::u32 registers[kRegisterCount]{}; ///< Final register file.
};

/**
 * @class Interpreter
 * @brief Executes the minimal ISA over a byte memory.
 *
 * The opcode table is a MEMBER rather than a switch on the enum, so an interpreter
 * rebuilt from an engraved specification runs the opcodes that specification names —
 * and an incomplete engraving produces an interpreter that visibly cannot run the
 * program, instead of one that silently agrees with the compiled-in enum.
 */
class Interpreter {
public:
    Interpreter() noexcept = default;

    /**
     * @brief Builds an interpreter that knows the ten opcodes of this build.
     * @return The reference machine.
     */
    [[nodiscard]] static Interpreter reference() noexcept;

    /**
     * @brief Builds an interpreter from an engraved specification.
     *
     * This is the honest test of the whole artifact: if the specification does not
     * carry enough to rebuild the machine, the machine rebuilt from it will not run
     * the payload's decompressor, and the artifact was a lie.
     *
     * @param specification Bytes as @ref emitSpecification wrote them.
     * @param size          Their length.
     * @param outMachine    Receives the machine on success.
     * @return false when the specification is malformed or names no opcodes.
     */
    [[nodiscard]] static bool fromSpecification(const core::u8 *specification, core::u32 size,
                                                Interpreter &outMachine) noexcept;

    /**
     * @brief Runs @p program over @p memory.
     * @param program     Instruction stream, @ref kInstructionBytes per instruction.
     * @param programSize Its length in bytes.
     * @param memory      Data memory, read and written in place.
     * @param memorySize  Its length.
     * @param budget      Maximum instructions to retire.
     * @param outReport   Receives the tally.
     * @return true when the program halted within the budget.
     */
    [[nodiscard]] bool run(const core::u8 *program, core::u32 programSize, core::u8 *memory, core::u32 memorySize,
                           core::u32 budget, ExecutionReport &outReport) const noexcept;

    /**
     * @brief Opcodes this machine was told about.
     * @return The count; ten for @ref reference.
     */
    [[nodiscard]] core::u32 knownOpcodes() const noexcept { return _knownOpcodes; }

private:
    /**
     * @brief Does this machine implement @p opcode?
     * @param opcode Numeric opcode.
     * @return true when the specification named it.
     */
    [[nodiscard]] bool knows(core::u8 opcode) const noexcept;

    core::u8 _implemented[static_cast<core::u32>(Opcode::Count)]{};
    core::u32 _knownOpcodes{0u};
};

} // namespace lpl::rosetta

#endif // LPL_LPL_ROSETTA_INTERPRETER_HPP
