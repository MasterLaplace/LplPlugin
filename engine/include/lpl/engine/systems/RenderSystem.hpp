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
 * @brief Reads ECS position/size/rotation and submits draw calls to IRenderer.
 */
class RenderSystem final : public ecs::ISystem {
public:
    RenderSystem(ecs::Registry &registry, render::IRenderer *renderer);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    ecs::Registry &_registry;
    render::IRenderer *_renderer;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_RENDER_SYSTEM_HPP
