/**
 * @file Parity.hpp
 * @brief The constexpr program both sides execute.
 *
 * A fixed program must produce the same trace on host and kernel.
 *
 * Unlike gate P11, whose two sides run different code on purpose, this one is the
 * ordinary contract: one interpreter, two targets, and the claim is that a machine
 * built out of wrapping integer arithmetic does not depend on which of them ran it.
 * That is worth folding anyway — the ISA exists to be re-implemented by a stranger,
 * and a machine whose result depends on the host is one no stranger can reproduce.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_ROSETTA_PARITY_HPP
#    define LPL_LPL_ROSETTA_PARITY_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::rosetta {

/**
 * @struct RosettaFoldResult
 * @brief The signatures the kernel must reproduce.
 */
struct RosettaFoldResult {
    core::u32 traceSignature{0u};   ///< Fold of every instruction retired and its result.
    core::u32 specSignature{0u};    ///< Fold of the engraved instruction-set description.
    core::u32 plateSignature{0u};   ///< Fold of the whole plate image.
    core::u32 payloadSignature{0u}; ///< Fold of the payload read back off the plate.
    core::u32 steps{0u};            ///< Instructions the canonical program retired.
    core::u32 halted{0u};           ///< 1 when it reached HALT rather than its budget.
    core::u32 plateBytes{0u};       ///< Size of the plate.
    core::u32 rebuiltOpcodes{0u};   ///< Opcodes an interpreter rebuilt from the plate knows.
    core::u32 selfHosting{0u};      ///< 1 when the rebuilt reader decoded the plate.
};

/**
 * @brief Runs the canonical program, engraves the canonical plate, and folds both.
 *
 * One function, called by the host oracle and by the kernel smoke, for the reason
 * every parity case in this project is one function: two copies of the parameters are
 * two things that can be edited apart.
 *
 * @param out Receives the signatures.
 */
void foldRosettaState(RosettaFoldResult &out);

} // namespace lpl::rosetta

#endif // LPL_LPL_ROSETTA_PARITY_HPP
