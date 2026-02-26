// /////////////////////////////////////////////////////////////////////////////
/// @file HapticManager.cpp
/// @brief HapticManager implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/haptic/HapticManager.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::haptic {

HapticManager::HapticManager() = default;
HapticManager::~HapticManager() { shutdownAll(); }

void HapticManager::addDevice(std::unique_ptr<IHapticDevice> device)
{
    LPL_ASSERT(device != nullptr);
    devices_.push_back(std::move(device));
}

core::Expected<void> HapticManager::initAll()
{
    for (auto& dev : devices_)
    {
        auto result = dev->init();
        if (!result)
        {
            core::Log::error("HapticManager: failed to init device");
            return result;
        }
        core::Log::info("HapticManager: initialized device");
    }
    return {};
}

void HapticManager::shutdownAll()
{
    for (auto& dev : devices_)
    {
        dev->shutdown();
    }
}

void HapticManager::broadcast(const HapticEffect& effect)
{
    for (auto& dev : devices_)
    {
        [[maybe_unused]] auto result = dev->submitEffect(effect);
    }
}

void HapticManager::playAtPosition(
    const math::Vec3<math::Fixed32>& /*worldPosition*/,
    core::f32 /*intensity*/)
{
    LPL_ASSERT(false && "HapticManager::playAtPosition not yet implemented — needs body mapping");
}

void HapticManager::cancelAll()
{
    for (auto& dev : devices_)
    {
        dev->cancelAll();
    }
}

core::usize HapticManager::deviceCount() const noexcept
{
    return devices_.size();
}

} // namespace lpl::haptic
