/**
 * @file IClockBackend.hpp
 * @brief Abstract clock / timing backend interface.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PLATFORM_ICLOCKBACKEND_HPP
#    define LPL_PLATFORM_ICLOCKBACKEND_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::platform {

/**
 * @class IClockBackend
 * @brief Strategy interface for the platform tick / timestamp contract.
 *
 * Concrete implementations: a std::chrono::steady_clock backend on Linux, the
 * kernel clock (PIT/RTC + rdtsc) backend in-kernel. The tick count is monotonic
 * but may wrap, so consumers take modular deltas. The timestamp counter is for
 * sub-tick, non-authoritative timing only and must never feed the deterministic
 * Fixed32 simulation authority.
 */
class IClockBackend {
public:
    virtual ~IClockBackend() = default;

    /** @brief Monotonic tick count (may wrap; use modular deltas). */
    [[nodiscard]] virtual core::u32 tickCount() const noexcept = 0;

    /** @brief Tick frequency in Hz. */
    [[nodiscard]] virtual core::u32 tickHertz() const noexcept = 0;

    /** @brief High-resolution CPU timestamp counter for sub-tick timing. */
    [[nodiscard]] virtual core::u64 timestampCounter() const noexcept = 0;

    /** @brief Returns a human-readable name. */
    [[nodiscard]] virtual const char *name() const noexcept = 0;
};

} // namespace lpl::platform

#endif // LPL_PLATFORM_ICLOCKBACKEND_HPP
