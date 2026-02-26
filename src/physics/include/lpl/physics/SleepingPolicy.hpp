// /////////////////////////////////////////////////////////////////////////////
/// @file SleepingPolicy.hpp
/// @brief Determines when rigid bodies can enter/leave sleep state.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/math/Vec3.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/core/Types.hpp>

namespace lpl::physics {

// /////////////////////////////////////////////////////////////////////////////
/// @class SleepingPolicy
/// @brief Configurable threshold-based sleeping policy.
///
/// A body is put to sleep when its linear and angular velocity stay below
/// the configured thresholds for @c kSleepFrames consecutive ticks.
// /////////////////////////////////////////////////////////////////////////////
class SleepingPolicy
{
public:
    static constexpr core::u32 kSleepFrames = 60;

    /// @brief Evaluates whether a body should be awake or asleep.
    /// @param linearVelocity  Current linear velocity.
    /// @param angularVelocity Current angular velocity.
    /// @param sleepCounter    Mutable counter for consecutive idle frames.
    /// @return @c true if the body should sleep.
    [[nodiscard]] static bool shouldSleep(
        const math::Vec3<math::Fixed32>& linearVelocity,
        const math::Vec3<math::Fixed32>& angularVelocity,
        core::u32& sleepCounter) noexcept;

    /// @brief Forces a body awake (resets its counter).
    static void wake(core::u32& sleepCounter) noexcept;

private:
    static constexpr math::Fixed32 kLinearThreshold  = math::Fixed32::fromRaw(655);
    static constexpr math::Fixed32 kAngularThreshold = math::Fixed32::fromRaw(655);
};

} // namespace lpl::physics
