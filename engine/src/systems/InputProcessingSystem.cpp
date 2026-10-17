#include <lpl/engine/systems/InputProcessingSystem.hpp>

namespace lpl::engine::systems {

static const ecs::ComponentAccess kInputAccesses[] = {
    {ecs::ComponentId::InputSnapshot, ecs::AccessMode::ReadWrite},
};

static const ecs::SystemDescriptor kInputProcessingDesc{"InputProcessing", ecs::SchedulePhase::Input,
                                                        std::span<const ecs::ComponentAccess>{kInputAccesses}};

struct InputProcessingSystem::Impl {
    EventQueues &queues;
    input::InputManager &inputManager;
    net::session::SessionManager *sessions;

    Impl(EventQueues &q, input::InputManager &im, net::session::SessionManager *sm)
        : queues{q}, inputManager{im}, sessions{sm}
    {
    }
};

InputProcessingSystem::InputProcessingSystem(EventQueues &queues, input::InputManager &inputManager,
                                             net::session::SessionManager *sessions)
    : _impl{std::make_unique<Impl>(queues, inputManager, sessions)}
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

        if (_impl->sessions != nullptr)
        {
            if (auto *session = _impl->sessions->find(eid))
                session->touch();
        }

        [[maybe_unused]] auto &_ = _impl->inputManager.getOrCreate(eid);

        for (const auto &k : ev.keys)
        {
            _impl->inputManager.setKeyState(eid, k.key, k.pressed);
        }

        for (const auto &a : ev.axes)
        {
            _impl->inputManager.setAxis(eid, a.axisId, a.value);
        }

        if (ev.hasNeural)
        {
            _impl->inputManager.setNeural(eid, ev.neural.alpha, ev.neural.beta, ev.neural.concentration,
                                          ev.neural.blink);
        }
    }
}

} // namespace lpl::engine::systems
