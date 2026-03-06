#include <lpl/engine/systems/InputSendSystem.hpp>
#include <lpl/input/InputManager.hpp>
#include <lpl/net/protocol/Bitstream.hpp>
#include <lpl/net/protocol/PacketBuilder.hpp>
#include <lpl/net/transport/ITransport.hpp>
#include <vector>

#ifdef LPL_HAS_RENDERER
#    include <GLFW/glfw3.h>
#endif

namespace lpl::engine::systems {

static const ecs::SystemDescriptor kInputSendSystemDesc{"InputSendSystem", ecs::SchedulePhase::Network, {}};

InputSendSystem::InputSendSystem(input::InputManager &inputManager,
                                 std::shared_ptr<net::transport::ITransport> transport, const core::u32 &myEntityId,
                                 const bool &connected)
    : _inputManager(inputManager), _transport(std::move(transport)), _myEntityId(myEntityId), _connected(connected)
{
}

const ecs::SystemDescriptor &InputSendSystem::descriptor() const noexcept { return kInputSendSystemDesc; }

void InputSendSystem::execute(core::f32 /*dt*/)
{
    if (!_connected || _myEntityId == 0 || !_transport)
        return;

    const auto *state = _inputManager.getState(_myEntityId);
    if (!state)
        return;

    net::protocol::Bitstream stream;

    // EntityId (32 bits)
    stream.writeU32(_myEntityId);

#ifdef LPL_HAS_RENDERER
    constexpr int trackedKeys[] = {GLFW_KEY_W,    GLFW_KEY_A,     GLFW_KEY_S,  GLFW_KEY_D,
                                   GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_UP, GLFW_KEY_DOWN};
    const core::u16 keyCount = sizeof(trackedKeys) / sizeof(trackedKeys[0]);
    stream.writeU16(keyCount);

    for (int key : trackedKeys)
    {
        stream.writeU16(static_cast<core::u16>(key));
        stream.writeBool(state->getKey(static_cast<core::u16>(key)));
    }
#else
    stream.writeU16(0);
#endif

    // Axis count (8 bits)
    stream.writeU8(0);

    // Neural data (3 floats + 1 bool = 97 bits)
    stream.writeFloat(state->neural.alpha);
    stream.writeFloat(state->neural.beta);
    stream.writeFloat(state->neural.concentration);
    stream.writeBool(state->blinkPrev);

    // Convert bitstream to payload bytes
    auto payload = stream.data();
    std::vector<core::byte> payloadVec{payload.begin(), payload.end()};

    [[maybe_unused]] auto res = net::protocol::sendInputs(*_transport, nullptr, payloadVec);
}

} // namespace lpl::engine::systems
