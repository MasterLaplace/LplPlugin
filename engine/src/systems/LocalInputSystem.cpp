#include <lpl/engine/systems/LocalInputSystem.hpp>
#include <lpl/input/InputManager.hpp>

#ifdef LPL_HAS_RENDERER
#    include <GLFW/glfw3.h>
#endif

namespace lpl::engine::systems {

static const ecs::SystemDescriptor kLocalInputSystemDesc{"LocalInputSystem", ecs::SchedulePhase::Input, {}};

LocalInputSystem::LocalInputSystem(input::InputManager &inputManager, GLFWwindow *window, const core::u32 &myEntityId,
                                   const bool &connected)
    : _inputManager(inputManager), _window(window), _myEntityId(myEntityId), _connected(connected)
{
}

const ecs::SystemDescriptor &LocalInputSystem::descriptor() const noexcept { return kLocalInputSystemDesc; }

void LocalInputSystem::execute(core::f32 /*dt*/)
{
    if (!_connected || _myEntityId == 0)
        return;

#ifdef LPL_HAS_RENDERER
    if (!_window)
        return;

    [[maybe_unused]] auto &_ = _inputManager.getOrCreate(_myEntityId);

    constexpr int trackedKeys[] = {GLFW_KEY_W,    GLFW_KEY_A,     GLFW_KEY_S,  GLFW_KEY_D,
                                   GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_UP, GLFW_KEY_DOWN};
    for (int key : trackedKeys)
    {
        bool pressed = (glfwGetKey(_window, key) == GLFW_PRESS);
        _inputManager.setKeyState(_myEntityId, static_cast<core::u16>(key), pressed);
    }
#endif
}

} // namespace lpl::engine::systems
