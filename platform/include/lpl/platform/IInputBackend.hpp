/**
 * @file IInputBackend.hpp
 * @brief Abstract input-event backend interface.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PLATFORM_IINPUTBACKEND_HPP
#    define LPL_PLATFORM_IINPUTBACKEND_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::platform {

/**
 * @class IInputBackend
 * @brief Strategy interface for draining platform input events.
 *
 * Concrete implementations: a GLFW key/event backend on Linux, the PS/2
 * keyboard SPSC ring (ISR producer -> engine consumer) in-kernel. The engine
 * drains decoded characters; the platform owns decoding and device state.
 */
class IInputBackend {
public:
    virtual ~IInputBackend() = default;

    /** @brief Pop one decoded character; false if the ring is empty. */
    [[nodiscard]] virtual bool tryPopCharacter(char &outCharacter) = 0;

    /** @brief Number of decoded characters currently waiting. */
    [[nodiscard]] virtual core::u32 pendingCount() const noexcept = 0;

    /** @brief Returns a human-readable name. */
    [[nodiscard]] virtual const char *name() const noexcept = 0;
};

} // namespace lpl::platform

#endif // LPL_PLATFORM_IINPUTBACKEND_HPP
