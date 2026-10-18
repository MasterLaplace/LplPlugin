#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/haptic/HapticManager.hpp>

namespace lpl::haptic {

HapticManager::HapticManager() = default;
HapticManager::~HapticManager() { shutdownAll(); }

void HapticManager::addDevice(std::unique_ptr<IHapticDevice> device)
{
    LPL_ASSERT(device != nullptr);
    _devices.push_back(std::move(device));
}

core::Expected<void> HapticManager::initAll()
{
    for (auto &dev : _devices)
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
    for (auto &dev : _devices)
    {
        dev->shutdown();
    }
}

void HapticManager::broadcast(const HapticEffect &effect)
{
    for (auto &dev : _devices)
    {
        [[maybe_unused]] auto result = dev->submitEffect(effect);
    }
}

void HapticManager::playAtPosition(const math::Vec3<math::Fixed32> & /*worldPosition*/, core::f32 /*intensity*/)
{
    // TODO(#317): map the world position to the nearest haptic actuator and modulate its intensity.
    LPL_ASSERT(false && "unimplemented");
}

void HapticManager::cancelAll()
{
    for (auto &dev : _devices)
    {
        dev->cancelAll();
    }
}

core::usize HapticManager::deviceCount() const noexcept { return _devices.size(); }

} // namespace lpl::haptic
