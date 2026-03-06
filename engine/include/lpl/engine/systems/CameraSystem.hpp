#pragma once

#ifndef LPL_ENGINE_SYSTEMS_CAMERA_SYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_CAMERA_SYSTEM_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/math/Vec3.hpp>

struct GLFWwindow;

namespace lpl::ecs {
class Registry;
}

namespace lpl::engine::systems {

struct CameraData {
    math::Vec3<float> position{0.f, 80.f, -120.f};
    math::Vec3<float> front{0.f, -0.4f, 1.f};
    math::Vec3<float> up{0.f, 1.f, 0.f};
    float speed = 300.f;
    float yaw = 90.f;
    float pitch = -20.f;
};

/**
 * @class CameraSystem
 * @brief Updates a local Camera struct based on GLFW inputs and local player entity pos.
 */
class CameraSystem final : public ecs::ISystem {
public:
    CameraSystem(CameraData &cameraData, ecs::Registry &registry, GLFWwindow *window, const core::u32 &myEntityId,
                 const bool &connected);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    CameraData &_cameraData;
    ecs::Registry &_registry;
    GLFWwindow *_window;
    const core::u32 &_myEntityId;
    const bool &_connected;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_CAMERA_SYSTEM_HPP
