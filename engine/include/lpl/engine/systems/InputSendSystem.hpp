#pragma once

#ifndef LPL_ENGINE_SYSTEMS_INPUT_SEND_SYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_INPUT_SEND_SYSTEM_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/System.hpp>
#    include <memory>

namespace lpl::input {
class InputManager;
}
namespace lpl::net::transport {
class ITransport;
}

namespace lpl::engine::systems {

/**
 * @class InputSendSystem
 * @brief Serializes local InputManager state and sends it to the server.
 */
class InputSendSystem final : public ecs::ISystem {
public:
    InputSendSystem(input::InputManager &inputManager, std::shared_ptr<net::transport::ITransport> transport,
                    const core::u32 &myEntityId, const bool &connected);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;

    /**
     * @brief Sends this tick's input of the local player to the server.
     *
     * The payload is the entity id (32 bits), the key count and each tracked key with
     * its state (none without a renderer), the axis count (8 bits, always zero), then the
     * neural data (3 floats + 1 bool = 97 bits).
     */
    void execute(core::f32 dt) override;

private:
    input::InputManager &_inputManager;
    std::shared_ptr<net::transport::ITransport> _transport;
    const core::u32 &_myEntityId;
    const bool &_connected;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_INPUT_SEND_SYSTEM_HPP
