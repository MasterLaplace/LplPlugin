#pragma once

#ifndef LPL_ENGINE_SYSTEMS_LOCAL_INPUT_SYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_LOCAL_INPUT_SYSTEM_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/System.hpp>

struct GLFWwindow;

namespace lpl::input {
class InputManager;
}

namespace lpl::engine::systems {

/**
 * @class LocalInputSystem
 * @brief Reads local hardware inputs (e.g. GLFW) and writes to InputManager.
 */
class LocalInputSystem final : public ecs::ISystem {
public:
    LocalInputSystem(input::InputManager &inputManager, GLFWwindow *window, const core::u32 &myEntityId,
                     const bool &connected);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    input::InputManager &_inputManager;
    GLFWwindow *_window;
    const core::u32 &_myEntityId;
    const bool &_connected;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_LOCAL_INPUT_SYSTEM_HPP
