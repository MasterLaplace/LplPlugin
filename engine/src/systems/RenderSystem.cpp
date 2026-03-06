#include <lpl/engine/systems/RenderSystem.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/math/Vec3.hpp>

namespace lpl::engine::systems {

static const ecs::ComponentAccess kRenderAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadOnly},
    {ecs::ComponentId::AABB,     ecs::AccessMode::ReadOnly},
};

static const ecs::SystemDescriptor kRenderSystemDesc{
    "RenderSystem",
    ecs::SchedulePhase::Render,
    std::span<const ecs::ComponentAccess>{kRenderAccesses}
};

RenderSystem::RenderSystem(ecs::Registry& registry, render::IRenderer* renderer)
    : _registry(registry)
    , _renderer(renderer)
{
}

const ecs::SystemDescriptor& RenderSystem::descriptor() const noexcept
{
    return kRenderSystemDesc;
}

void RenderSystem::execute(core::f32 /*dt*/)
{
    // @todo (GPU dispatch): In a fully featured Vulkan ECS integration, 
    // we would extract the transform matrices of all entities in _world
    // and build an instance buffer or push constants to _renderer here.

    if (!_renderer) return;

    for (const auto& part : _registry.partitions())
    {
        if (!part) continue;
        for (const auto& chunkPtr : part->chunks())
        {
            if (!chunkPtr) continue;
            auto& chunk = *chunkPtr;
            
            auto* positions = static_cast<const math::Vec3<float>*>(
                chunk.readComponent(ecs::ComponentId::Position));
            auto* sizes = static_cast<const math::Vec3<float>*>(
                chunk.readComponent(ecs::ComponentId::AABB));

            if (!positions) continue; // Not renderable if no position

            for (core::u32 i = 0; i < chunk.count(); ++i)
            {
                // Simulate submitting to renderer
                [[maybe_unused]] auto p = positions[i];
                if (sizes)
                {
                    [[maybe_unused]] auto s = sizes[i];
                }
            }
        }
    }
}

} // namespace lpl::engine::systems
