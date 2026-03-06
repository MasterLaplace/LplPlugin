/**
 * @file RollbackVerifier.hpp
 * @brief Debug utilities for verifying rollback determinism and state immutability.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_NETCODE_ROLLBACKVERIFIER_HPP
#    define LPL_NET_NETCODE_ROLLBACKVERIFIER_HPP

#    include <lpl/core/Assert.hpp>
#    include <lpl/core/Log.hpp>
#    include <lpl/core/Types.hpp>

#    include <cstring>
#    include <span>
#    include <vector>

namespace lpl::net::netcode {

/**
 * @class RollbackVerifier
 * @brief Static debug utility for verifying netcode determinism properties.
 *
 * Provides compile-time-gated checks to ensure:
 * - **Immutability**: a rollback + re-simulate produces byte-identical output
 * - **Determinism**: identical inputs always produce identical outputs
 *
 * Enabled via `LPL_DEBUG_ROLLBACK` preprocessor define.
 */
class RollbackVerifier {
public:
    /**
     * @brief Captures a state snapshot as a flat byte buffer.
     * @param data Pointer to state memory.
     * @param size Size in bytes.
     * @return Copy of the state bytes.
     */
    [[nodiscard]] static std::vector<core::byte> captureSnapshot(const void *data, core::usize size)
    {
        const auto *bytes = static_cast<const core::byte *>(data);
        return {bytes, bytes + size};
    }

    /**
     * @brief Verifies that two snapshots are byte-identical.
     *
     * Call this after a rollback+re-simulation to ensure the state matches
     * the original forward simulation. A mismatch indicates non-determinism.
     *
     * @param before Snapshot captured before rollback.
     * @param after  Snapshot captured after rollback + re-simulate.
     * @return true if byte-identical, false otherwise.
     */
    [[nodiscard]] static bool verifyImmutability(std::span<const core::byte> before, std::span<const core::byte> after)
    {
        if (before.size() != after.size())
        {
            core::Log::warn("[RollbackVerifier] Size mismatch: {} vs {}", before.size(), after.size());
            return false;
        }

        if (std::memcmp(before.data(), after.data(), before.size()) != 0)
        {
            // Find first divergence offset for diagnostics
            for (core::usize i = 0; i < before.size(); ++i)
            {
                if (before[i] != after[i])
                {
                    core::Log::warn("[RollbackVerifier] Divergence at byte offset {}", i);
                    break;
                }
            }
            return false;
        }

        return true;
    }

    /**
     * @brief Runs two identical simulations and asserts deterministic output.
     *
     * @tparam SimFn Callable: void(void* state, core::usize size)
     * @param state     Mutable state buffer.
     * @param size      Size of state buffer.
     * @param simulate  Function that advances the simulation one tick.
     * @return true if both runs produced identical state.
     */
    template <typename SimFn>
    [[nodiscard]] static bool verifyDeterminism(void *state, core::usize size, SimFn &&simulate)
    {
        auto snapshot1 = captureSnapshot(state, size);
        simulate(state, size);
        auto result1 = captureSnapshot(state, size);

        // Restore original state
        std::memcpy(state, snapshot1.data(), size);

        // Run again
        simulate(state, size);
        auto result2 = captureSnapshot(state, size);

        return verifyImmutability(std::span<const core::byte>{result1}, std::span<const core::byte>{result2});
    }
};

} // namespace lpl::net::netcode

#endif // LPL_NET_NETCODE_ROLLBACKVERIFIER_HPP
