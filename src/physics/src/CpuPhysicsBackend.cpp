// /////////////////////////////////////////////////////////////////////////////
/// @file CpuPhysicsBackend.cpp
/// @brief CPU physics backend implementation stub.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/physics/CpuPhysicsBackend.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::physics {

struct CpuPhysicsBackend::Impl
{
    ecs::Registry& registry;
    explicit Impl(ecs::Registry& r) : registry{r} {}
};

CpuPhysicsBackend::CpuPhysicsBackend(ecs::Registry& registry)
    : impl_{std::make_unique<Impl>(registry)}
{}

CpuPhysicsBackend::~CpuPhysicsBackend() = default;

core::Expected<void> CpuPhysicsBackend::init()
{
    core::Log::info("CpuPhysicsBackend::init");
    return {};
}

core::Expected<void> CpuPhysicsBackend::step(core::f32 /*dt*/)
{
    LPL_ASSERT(false && "CpuPhysicsBackend::step not yet implemented");
    return {};
}

void CpuPhysicsBackend::shutdown()
{
    core::Log::info("CpuPhysicsBackend::shutdown");
}

const char* CpuPhysicsBackend::name() const noexcept
{
    return "CpuPhysicsBackend";
}

} // namespace lpl::physics
