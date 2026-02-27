/**
 * @file IHapticDevice.hpp
 * @brief Abstract haptic device interface (Strategy pattern).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_HAPTIC_IHAPTICDEVICE_HPP
    #define LPL_HAPTIC_IHAPTICDEVICE_HPP

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/FixedPoint.hpp>

namespace lpl::haptic {

/** @brief Haptic feedback effect descriptor. */
struct HapticEffect
{
    core::u16 motorId{0};
    core::f32 intensity{0.0f};
    core::f32 durationMs{0.0f};
    core::f32 frequency{0.0f};
};

/** @brief Haptic device capabilities. */
struct HapticCapabilities
{
    core::u16 motorCount{0};
    bool supportsFrequency{false};
    bool supportsDirectional{false};
};

/**
 * @brief Abstract interface for haptic output devices.
 *
 * Strategy pattern: concrete implementations handle specific
 * haptic hardware (gloves, vests, controllers, BCI actuators).
 */
class IHapticDevice
{
public:
    virtual ~IHapticDevice() = default;

    /**
     * @brief Initialize the device.
     * @return Success or error.
     */
    [[nodiscard]] virtual core::Expected<void> init() = 0;

    /** @brief Shut down the device and release resources. */
    virtual void shutdown() = 0;

    /** @brief Query device capabilities. */
    [[nodiscard]] virtual HapticCapabilities capabilities() const noexcept = 0;

    /**
     * @brief Submit a haptic effect to the device.
     * @param effect Effect descriptor.
     * @return Success or error.
     */
    [[nodiscard]] virtual core::Expected<void> submitEffect(const HapticEffect& effect) = 0;

    /** @brief Cancel all pending effects. */
    virtual void cancelAll() = 0;

    /** @brief Human-readable device name. */
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace lpl::haptic

#endif // LPL_HAPTIC_IHAPTICDEVICE_HPP
