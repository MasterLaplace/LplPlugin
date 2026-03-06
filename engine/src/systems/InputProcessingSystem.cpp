/**
 * @file InputProcessingSystem.cpp
 * @brief Drains InputEvent queue and feeds per-entity state into InputManager.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/engine/systems/InputProcessingSystem.hpp>

namespace lpl::engine::systems {

// ========================================================================== //
//  Descriptor                                                                //
// ========================================================================== //

static const ecs::ComponentAccess kInputAccesses[] = {
    {ecs::ComponentId::InputSnapshot, ecs::AccessMode::ReadWrite},
};

static const ecs::SystemDescriptor kInputProcessingDesc{"InputProcessing", ecs::SchedulePhase::Input,
                                                        std::span<const ecs::ComponentAccess>{kInputAccesses}};

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct InputProcessingSystem::Impl {
    EventQueues &queues;
    input::InputManager &inputManager;

    Impl(EventQueues &q, input::InputManager &im) : queues{q}, inputManager{im} {}
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

InputProcessingSystem::InputProcessingSystem(EventQueues &queues, input::InputManager &inputManager)
    : _impl{std::make_unique<Impl>(queues, inputManager)}
{
}

InputProcessingSystem::~InputProcessingSystem() = default;

const ecs::SystemDescriptor &InputProcessingSystem::descriptor() const noexcept { return kInputProcessingDesc; }

void InputProcessingSystem::execute(core::f32 /*dt*/)
{
    auto events = _impl->queues.inputs.drain();

    for (const auto &ev : events)
    {
        const core::u32 eid = ev.entityId;

        // Ensure per-entity input state exists
        [[maybe_unused]] auto &_ = _impl->inputManager.getOrCreate(eid);

        // Apply key states
        for (const auto &k : ev.keys)
        {
            _impl->inputManager.setKeyState(eid, k.key, k.pressed);
        }

        // Apply axis values
        for (const auto &a : ev.axes)
        {
            _impl->inputManager.setAxis(eid, a.axisId, a.value);
        }

        // Apply neural input (BCI)
        if (ev.hasNeural)
        {
            _impl->inputManager.setNeural(eid, ev.neural.alpha, ev.neural.beta, ev.neural.concentration,
                                          ev.neural.blink);
        }
    }
}

} // namespace lpl::engine::systems
