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

    /**
     * @brief Pop one relative pointer motion, with the buttons held at the time.
     *
     * Relative rather than absolute on purpose: a PS/2 device reports movement,
     * and turning movement into a coordinate needs a screen the platform layer
     * does not own. A camera wants the delta as it is; a cursor accumulates.
     *
     * Not pure: a backend with no pointing device is a legitimate backend, and
     * the default is the honest answer — nothing moved. Making this pure would
     * force every headless and every kernel-without-a-mouse build to write the
     * same empty override.
     *
     * @param outDeltaX  Rightward movement.
     * @param outDeltaY  UPWARD movement, as a mouse reports it — not screen-down.
     *                   The two conventions differ by a sign, and picking one here
     *                   is what stops each caller from picking a different one.
     * @param outButtons Bit 0 left, bit 1 right, bit 2 middle.
     */
    [[nodiscard]] virtual bool tryPopPointerMotion(core::i32 &outDeltaX, core::i32 &outDeltaY,
                                                   core::u32 &outButtons)
    {
        outDeltaX = 0;
        outDeltaY = 0;
        outButtons = 0u;
        return false;
    }

    /** @brief Whether a pointing device is present at all. */
    [[nodiscard]] virtual bool hasPointer() const noexcept { return false; }

    /** @brief Returns a human-readable name. */
    [[nodiscard]] virtual const char *name() const noexcept = 0;
};

} // namespace lpl::platform

#endif // LPL_PLATFORM_IINPUTBACKEND_HPP
