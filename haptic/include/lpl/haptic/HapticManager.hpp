/**
 * @file HapticManager.hpp
 * @brief Façade managing multiple haptic devices.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_HAPTIC_HAPTICMANAGER_HPP
    #define LPL_HAPTIC_HAPTICMANAGER_HPP

#include <lpl/haptic/IHapticDevice.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <memory>
#include <vector>

namespace lpl::haptic {

/**
 * @brief Façade that aggregates multiple IHapticDevice instances and
 *        provides a unified API for haptic feedback dispatch.
 */
class HapticManager
{
public:
    HapticManager();
    ~HapticManager();

    HapticManager(const HapticManager&) = delete;
    HapticManager& operator=(const HapticManager&) = delete;

    /**
     * @brief Register a haptic device.
     * @param device Owned device pointer.
     */
    void addDevice(std::unique_ptr<IHapticDevice> device);

    /**
     * @brief Initialize all registered devices.
     * @return Success or first error encountered.
     */
    [[nodiscard]] core::Expected<void> initAll();

    /** @brief Shut down all devices. */
    void shutdownAll();

    /**
     * @brief Submit a haptic pulse to all devices (broadcast).
     * @param effect Effect descriptor.
     */
    void broadcast(const HapticEffect& effect);

    /**
     * @brief Submit a spatial haptic event.
     *
     * Maps a world-space contact point to the appropriate body region
     * and dispatches to the relevant device / motor.
     * @param worldPosition Contact position in world space.
     * @param intensity Normalized intensity [0, 1].
     */
    void playAtPosition(const math::Vec3<math::Fixed32>& worldPosition,
                        core::f32 intensity);

    /** @brief Cancel all effects on all devices. */
    void cancelAll();

    /** @brief Number of registered devices. */
    [[nodiscard]] core::usize deviceCount() const noexcept;

private:
    std::vector<std::unique_ptr<IHapticDevice>> _devices;
};

} // namespace lpl::haptic

#endif // LPL_HAPTIC_HAPTICMANAGER_HPP
