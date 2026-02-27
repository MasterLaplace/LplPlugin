/**
 * @file StateHash.hpp
 * @brief FNV-1a incremental hash for deterministic desync detection.
 *
 * Each simulation tick, clientes and server hash their entire game state.
 * The resulting 8-byte digest is exchanged over the network.  A mismatch
 * signals a determinism failure and triggers a full-state reconciliation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_STATE_HASH_HPP
    #define LPL_MATH_STATE_HASH_HPP

    #include <lpl/core/Types.hpp>
    #include <lpl/core/Concepts.hpp>

    #include <cstring>
    #include <span>

namespace lpl::math {

/**
 * @brief Incremental FNV-1a hasher for game-state desync detection.
 */
class StateHash final {
public:
    static constexpr core::u64 kOffsetBasis = 14695981039346656037ULL;
    static constexpr core::u64 kPrime       = 1099511628211ULL;

    constexpr StateHash() = default;

    /**
     * @brief Feed a span of raw bytes into the hash.
     * @param data Byte span.
     * @return Reference to this hasher (for chaining).
     */
    StateHash &hashBytes(std::span<const core::byte> data);

    /**
     * @brief Feed a trivially-copyable value into the hash.
     * @tparam T Blittable type.
     * @param value Value to hash.
     * @return Reference to this hasher (for chaining).
     */
    template <core::Blittable T>
    StateHash &combine(const T &value)
    {
        const auto *ptr = reinterpret_cast<const core::byte *>(&value);
        return hashBytes({ptr, sizeof(T)});
    }

    /**
     * @brief Finalise and return the current digest.
     * @return 64-bit FNV-1a hash.
     */
    [[nodiscard]] constexpr core::u64 digest() const { return _hash; }

    /**
     * @brief Reset the hasher to its initial state.
     */
    constexpr void reset() { _hash = kOffsetBasis; }

    /**
     * @brief Compare two digests for equality.
     */
    [[nodiscard]] static constexpr bool match(core::u64 a, core::u64 b) { return a == b; }

private:
    core::u64 _hash = kOffsetBasis;
};

} // namespace lpl::math

#endif // LPL_MATH_STATE_HASH_HPP
