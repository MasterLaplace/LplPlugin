// /////////////////////////////////////////////////////////////////////////////
/// @file CpuPhysicsBackend.hpp
/// @brief CPU-only reference physics backend.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/physics/IPhysicsBackend.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::ecs { class Registry; }

namespace lpl::physics {

// /////////////////////////////////////////////////////////////////////////////
/// @class CpuPhysicsBackend
/// @brief Single-threaded CPU physics step: integrate → broad → narrow →
///        solve → sleep.
// /////////////////////////////////////////////////////////////////////////////
class CpuPhysicsBackend final : public IPhysicsBackend,
                                 public core::NonCopyable<CpuPhysicsBackend>
{
public:
    /// @brief Constructs with a reference to the ECS registry.
    /// @param registry Registry containing Position, Velocity, Mass, AABB, etc.
    explicit CpuPhysicsBackend(ecs::Registry& registry);
    ~CpuPhysicsBackend() override;

    [[nodiscard]] core::Expected<void> init() override;
    [[nodiscard]] core::Expected<void> step(core::f32 dt) override;
    void shutdown() override;
    [[nodiscard]] const char* name() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lpl::physics
