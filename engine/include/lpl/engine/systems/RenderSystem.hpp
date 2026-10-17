#pragma once

#ifndef LPL_ENGINE_SYSTEMS_RENDER_SYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_RENDER_SYSTEM_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/System.hpp>

namespace lpl::ecs {
class Registry;
}
namespace lpl::render {
class IRenderer;
}

namespace lpl::engine::systems {

/**
 * @class RenderSystem
 * @brief Reads ECS position and size for an IRenderer, to which it submits nothing yet.
 */
class RenderSystem final : public ecs::ISystem {
public:
    RenderSystem(ecs::Registry &registry, render::IRenderer *renderer);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;

    /**
     * @brief Walks the entities that have a position, and their sizes.
     *
     * The authoritative transform is Fixed32; a real renderer converts to float
     * (toFloat) at submit — rendering is a non-authoritative path.
     *
     * @todo #521: In a fully featured Vulkan ECS integration, the transform
     * matrices of all entities would be extracted here and built into an instance buffer
     * or push constants for the renderer. Nothing is submitted yet.
     */
    void execute(core::f32 dt) override;

private:
    ecs::Registry &_registry;
    render::IRenderer *_renderer;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_RENDER_SYSTEM_HPP
