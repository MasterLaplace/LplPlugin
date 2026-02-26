// /////////////////////////////////////////////////////////////////////////////
/// @file SleepingPolicy.cpp
/// @brief SleepingPolicy implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/physics/SleepingPolicy.hpp>

namespace lpl::physics {

bool SleepingPolicy::shouldSleep(
    const math::Vec3<math::Fixed32>& linearVelocity,
    const math::Vec3<math::Fixed32>& angularVelocity,
    core::u32& sleepCounter) noexcept
{
    const auto linSq = linearVelocity.lengthSquared();
    const auto angSq = angularVelocity.lengthSquared();

    const auto linThSq = kLinearThreshold * kLinearThreshold;
    const auto angThSq = kAngularThreshold * kAngularThreshold;

    if (linSq < linThSq && angSq < angThSq)
    {
        ++sleepCounter;
        return sleepCounter >= kSleepFrames;
    }

    sleepCounter = 0;
    return false;
}

void SleepingPolicy::wake(core::u32& sleepCounter) noexcept
{
    sleepCounter = 0;
}

} // namespace lpl::physics
