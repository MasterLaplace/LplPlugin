#include <lpl/core/Log.hpp>
#include <lpl/input/InputManager.hpp>

#include <lpl/std/cmath.hpp>
#include <lpl/std/unordered_map.hpp>

namespace lpl::input {

/**
 * @name Key constants matching legacy WASD
 * @{
 */
static constexpr core::u16 KEY_W = 87;
static constexpr core::u16 KEY_A = 65;
static constexpr core::u16 KEY_S = 83;
static constexpr core::u16 KEY_D = 68;
/** @} */

struct InputManager::Impl {
    pmr::vector<pmr::unique_ptr<IInputSource>> sources;
    InputState state{};
    NeuralInputState neuralState{};
    pmr::unordered_map<core::u32, PerEntityInput> entityStates;
};

InputManager::InputManager() : _impl{pmr::make_unique<Impl>()} {}

InputManager::~InputManager() = default;

void InputManager::addSource(pmr::unique_ptr<IInputSource> source) { _impl->sources.push_back(std::move(source)); }

core::Expected<void> InputManager::init()
{
    for (auto &source : _impl->sources)
    {
        auto result = source->init();
        if (!result.has_value())
        {
            return result;
        }
    }
    return {};
}

core::Expected<void> InputManager::poll()
{
    for (auto &source : _impl->sources)
    {
        auto result = source->poll();
        if (!result.has_value())
        {
            core::Log::warn("InputManager: poll failed for source");
        }
    }
    ++_impl->state.sequence;
    return {};
}

void InputManager::shutdown()
{
    for (auto &source : _impl->sources)
    {
        source->shutdown();
    }
}

void InputManager::setKeyState(core::u32 entityId, core::u16 key, bool pressed)
{
    getOrCreate(entityId).setKey(key, pressed);
}

void InputManager::setAxis(core::u32 entityId, core::u8 axisId, float value)
{
    getOrCreate(entityId).setAxis(axisId, value);
}

void InputManager::setNeural(core::u32 entityId, float alpha, float beta, float concentration, bool blink)
{
    auto &state = getOrCreate(entityId);

    state.neural.alpha = alpha;
    state.neural.beta = beta;
    state.neural.concentration = concentration;

    if (blink && !state.blinkPrev)
    {
        state.setAxis(15, 1.0f);
    }
    state.blinkPrev = blink;
}

const PerEntityInput *InputManager::getState(core::u32 entityId) const
{
    auto it = _impl->entityStates.find(entityId);
    return it != _impl->entityStates.end() ? &it->second : nullptr;
}

PerEntityInput *InputManager::getStateMut(core::u32 entityId)
{
    auto it = _impl->entityStates.find(entityId);
    return it != _impl->entityStates.end() ? &it->second : nullptr;
}

PerEntityInput &InputManager::getOrCreate(core::u32 entityId) { return _impl->entityStates[entityId]; }

void InputManager::removeEntity(core::u32 entityId) { _impl->entityStates.erase(entityId); }

bool InputManager::hasEntity(core::u32 entityId) const { return _impl->entityStates.contains(entityId); }

void InputManager::updateGroundedState(core::u32 entityId, float currentVelY, float threshold)
{
    auto *state = getStateMut(entityId);
    if (!state)
    {
        return;
    }
    state->isGrounded = pmr::fabs(currentVelY) < threshold;
}

math::Vec3<math::Fixed32>
InputManager::computeMovementVelocity(core::u32 entityId, math::Vec3<math::Fixed32> currentVel, math::Fixed32 speed)
{
    auto *state = getStateMut(entityId);
    if (!state)
    {
        return currentVel;
    }

    math::Fixed32 dx{};
    math::Fixed32 dz{};

    if (state->getKey(KEY_W))
        dz = dz - math::Fixed32::one();
    if (state->getKey(KEY_S))
        dz = dz + math::Fixed32::one();
    if (state->getKey(KEY_A))
        dx = dx - math::Fixed32::one();
    if (state->getKey(KEY_D))
        dx = dx + math::Fixed32::one();

    math::Fixed32 neuralScale = math::Fixed32::one();
    const math::Fixed32 concentration = math::Fixed32::fromFloat(state->neural.concentration);

    if (concentration > math::Fixed32::fromFloat(0.001f))
    {
        neuralScale = math::Fixed32::fromFloat(0.70f) + concentration * math::Fixed32::fromFloat(0.60f);
    }

    const math::Fixed32 finalSpeed = speed * neuralScale;

    currentVel.x = dx * finalSpeed;
    currentVel.z = dz * finalSpeed;

    if (state->getAxis(15) > 0.5f && state->isGrounded)
    {
        currentVel.y = math::Fixed32::fromInt(15);
        state->isGrounded = false;
    }
    state->setAxis(15, 0.0f);

    return currentVel;
}

const InputState &InputManager::currentState() const noexcept { return _impl->state; }

const NeuralInputState &InputManager::currentNeuralState() const noexcept { return _impl->neuralState; }

} // namespace lpl::input
