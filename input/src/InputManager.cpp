/**
 * @file InputManager.cpp
 * @brief InputManager implementation with per-entity neural features.
 *
 * Ported from legacy engine/InputManager.hpp. Implements per-entity input
 * maps, neural speed modulation, rising-edge blink detection, and
 * grounded-state tracking.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/input/InputManager.hpp>
#include <lpl/core/Log.hpp>

#include <cmath>
#include <unordered_map>

namespace lpl::input {

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct InputManager::Impl
{
    std::vector<std::unique_ptr<IInputSource>> sources;
    InputState                                 state{};
    NeuralInputState                           neuralState{};
    std::unordered_map<core::u32, PerEntityInput> entityStates;
};

// ========================================================================== //
//  Source management                                                         //
// ========================================================================== //

InputManager::InputManager()
    : _impl{std::make_unique<Impl>()}
{}

InputManager::~InputManager() = default;

void InputManager::addSource(std::unique_ptr<IInputSource> source)
{
    _impl->sources.push_back(std::move(source));
}

core::Expected<void> InputManager::init()
{
    for (auto& source : _impl->sources)
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
    for (auto& source : _impl->sources)
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
    for (auto& source : _impl->sources)
    {
        source->shutdown();
    }
}

// ========================================================================== //
//  Per-entity state (ported from legacy engine/InputManager.hpp)             //
// ========================================================================== //

// Key constants matching legacy WASD
static constexpr core::u16 KEY_W = 87;
static constexpr core::u16 KEY_A = 65;
static constexpr core::u16 KEY_S = 83;
static constexpr core::u16 KEY_D = 68;

void InputManager::setKeyState(core::u32 entityId, core::u16 key, bool pressed)
{
    getOrCreate(entityId).setKey(key, pressed);
}

void InputManager::setAxis(core::u32 entityId, core::u8 axisId, float value)
{
    getOrCreate(entityId).setAxis(axisId, value);
}

void InputManager::setNeural(core::u32 entityId, float alpha, float beta,
                              float concentration, bool blink)
{
    auto& state = getOrCreate(entityId);

    // Rising-edge blink detection: only trigger if previous was false, current is true
    // This is idempotent — repeated calls with blink=true won't re-trigger
    state.neural.alpha = alpha;
    state.neural.beta = beta;
    state.neural.concentration = concentration;

    // blinkPrev tracks previous state for rising-edge detection in computeMovementVelocity
    // We don't update blinkPrev here — that's done in computeMovementVelocity
    // Instead, we store the current blink state in the axis system
    if (blink && !state.blinkPrev)
    {
        // Rising edge: mark blink detected
        state.setAxis(15, 1.0f); // Axis 15 = blink trigger
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

PerEntityInput &InputManager::getOrCreate(core::u32 entityId)
{
    return _impl->entityStates[entityId];
}

void InputManager::removeEntity(core::u32 entityId)
{
    _impl->entityStates.erase(entityId);
}

bool InputManager::hasEntity(core::u32 entityId) const
{
    return _impl->entityStates.contains(entityId);
}

void InputManager::updateGroundedState(core::u32 entityId, float currentVelY,
                                        float threshold)
{
    auto* state = getStateMut(entityId);
    if (!state)
    {
        return;
    }
    state->isGrounded = std::fabs(currentVelY) < threshold;
}

math::Vec3<float> InputManager::computeMovementVelocity(
    core::u32 entityId,
    math::Vec3<float> currentVel,
    float speed)
{
    auto* state = getStateMut(entityId);
    if (!state)
    {
        return currentVel;
    }

    // WASD movement direction
    float dx = 0.0f;
    float dz = 0.0f;

    if (state->getKey(KEY_W)) dz -= 1.0f;
    if (state->getKey(KEY_S)) dz += 1.0f;
    if (state->getKey(KEY_A)) dx -= 1.0f;
    if (state->getKey(KEY_D)) dx += 1.0f;

    // Neural speed modulation: concentration [0..1] → scale [0.70x..1.30x]
    float neuralScale = 1.0f;
    float concentration = state->neural.concentration;

    if (concentration > 0.001f)
    {
        // Linear interpolation: 0.0 → 0.70, 0.5 → 1.00, 1.0 → 1.30
        neuralScale = 0.70f + concentration * 0.60f;
    }

    float finalSpeed = speed * neuralScale;

    // Apply movement
    currentVel.x = dx * finalSpeed;
    currentVel.z = dz * finalSpeed;

    // Blink-based jump (rising-edge, grounded only)
    if (state->getAxis(15) > 0.5f && state->isGrounded)
    {
        currentVel.y = 15.0f; // Jump impulse
        state->isGrounded = false;
    }
    state->setAxis(15, 0.0f); // Clear blink trigger

    return currentVel;
}

// ========================================================================== //
//  Global state                                                              //
// ========================================================================== //

const InputState& InputManager::currentState() const noexcept
{
    return _impl->state;
}

const NeuralInputState& InputManager::currentNeuralState() const noexcept
{
    return _impl->neuralState;
}

} // namespace lpl::input
