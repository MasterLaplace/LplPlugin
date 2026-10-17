#include <lpl/core/Log.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/engine/systems/SessionSystem.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/net/protocol/Protocol.hpp>

#include <cstring>

namespace lpl::engine::systems {

static const ecs::ComponentAccess kSessionAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Velocity, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Health,   ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Mass,     ecs::AccessMode::ReadWrite},
};

static const ecs::SystemDescriptor kSessionDesc{"Session", ecs::SchedulePhase::Input,
                                                std::span<const ecs::ComponentAccess>{kSessionAccesses}};

struct SessionSystem::Impl {
    net::session::SessionManager &sessionManager;
    EventQueues &queues;
    std::shared_ptr<net::transport::ITransport> transport;
    input::InputManager &inputManager;
    ecs::WorldPartition &world;
    ecs::Registry &registry;
    core::f64 sessionTimeoutMs;

    Impl(net::session::SessionManager &sm, EventQueues &q, std::shared_ptr<net::transport::ITransport> t,
         input::InputManager &im, ecs::WorldPartition &w, ecs::Registry &reg, core::f64 timeout)
        : sessionManager{sm}, queues{q}, transport{std::move(t)}, inputManager{im}, world{w}, registry{reg},
          sessionTimeoutMs{timeout}
    {
    }

    /**
     * @brief Tears down everything a departing client owned: its avatar entity, its
     *        input state and its spatial-index entry.
     *
     * Keyed on the entity id, which is also the session's playerId (see the one network
     * identity of @ref SessionSystem::execute). Without this a disconnect would leak an
     * avatar that lingers and is broadcast — AOI then sends the neighbours an
     * EntityDestroy on the next tick, once the entity is gone from the registry and the
     * grid.
     */
    void teardown(const net::session::Session &session)
    {
        const ecs::EntityId avatar = session.boundEntity();
        if (avatar.isValid())
        {
            [[maybe_unused]] auto sr = world.remove(avatar);
            [[maybe_unused]] auto dr = registry.destroyEntity(avatar);
        }
        inputManager.removeEntity(session.playerId());
    }
};

SessionSystem::SessionSystem(net::session::SessionManager &sessionManager, EventQueues &queues,
                             std::shared_ptr<net::transport::ITransport> transport, input::InputManager &inputManager,
                             ecs::WorldPartition &world, ecs::Registry &registry, core::f64 sessionTimeoutMs)
    : _impl{std::make_unique<Impl>(sessionManager, queues, std::move(transport), inputManager, world, registry,
                                   sessionTimeoutMs)}
{
}

SessionSystem::~SessionSystem() = default;

const ecs::SystemDescriptor &SessionSystem::descriptor() const noexcept { return kSessionDesc; }

void SessionSystem::execute(core::f32 /*dt*/)
{
    for (const auto &ev : _impl->queues.disconnects.drain())
    {
        if (!ev.source.valid())
            continue;
        if (auto *session = _impl->sessionManager.findByAddress(ev.source))
        {
            _impl->teardown(*session);
            [[maybe_unused]] auto d = _impl->sessionManager.disconnect(session->playerId());
        }
    }

    if (_impl->sessionTimeoutMs > 0.0)
    {
        [[maybe_unused]] auto reaped = _impl->sessionManager.reapTimedOut(
            _impl->sessionTimeoutMs, [this](const net::session::Session &s) { _impl->teardown(s); });
    }

    for (const auto &ev : _impl->queues.snapshotAcks.drain())
    {
        if (auto *session = _impl->sessionManager.findByAddress(ev.source))
            session->setAckedSnapshotSeq(ev.seq);
    }

    auto events = _impl->queues.connects.drain();

    for (const auto &ev : events)
    {
        if (ev.source.valid() && _impl->sessionManager.findByAddress(ev.source) != nullptr)
        {
            continue;
        }

        ecs::Archetype playerArch;
        playerArch.add(ecs::ComponentId::Position);
        playerArch.add(ecs::ComponentId::Velocity);
        playerArch.add(ecs::ComponentId::Mass);
        playerArch.add(ecs::ComponentId::AABB);
        playerArch.add(ecs::ComponentId::Health);
        playerArch.add(ecs::ComponentId::SleepState);

        auto entityResult = _impl->registry.createEntity(playerArch);
        if (!entityResult.has_value())
        {
            core::Log::error("SessionSystem: failed to create entity");
            continue;
        }

        auto entityId = entityResult.value();

        const core::u32 netId = entityId.raw();

        auto sessionResult = _impl->sessionManager.connect(netId);
        if (!sessionResult.has_value())
        {
            [[maybe_unused]] auto d = _impl->registry.destroyEntity(entityId);
            continue;
        }

        auto *session = sessionResult.value();
        if (ev.source.valid())
        {
            session->setAddress(ev.source);
        }

        session->bindEntity(entityId);

        [[maybe_unused]] auto &_ = _impl->inputManager.getOrCreate(netId);

        auto refResult = _impl->registry.resolve(entityId);
        if (!refResult.has_value())
            continue;

        auto ref = refResult.value();
        auto &partition = _impl->registry.getOrCreatePartition(playerArch);
        const auto &chunks = partition.chunks();
        if (ref.chunkIndex < static_cast<core::u32>(chunks.size()))
        {
            auto &chunk = *chunks[ref.chunkIndex];

            math::Vec3<math::Fixed32> pos{math::Fixed32::fromInt(0), math::Fixed32::fromInt(10),
                                          math::Fixed32::fromInt(0)};
            if (auto *wpos = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Position)))
                wpos[ref.localIndex] = pos;
            if (auto *rpos = const_cast<math::Vec3<math::Fixed32> *>(
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk.readComponent(ecs::ComponentId::Position))))
                rpos[ref.localIndex] = pos;

            math::Vec3<math::Fixed32> vel{math::Fixed32::zero(), math::Fixed32::zero(), math::Fixed32::zero()};
            if (auto *wvel = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Velocity)))
                wvel[ref.localIndex] = vel;

            math::Fixed32 mass = math::Fixed32::one();
            if (auto *wmass = static_cast<math::Fixed32 *>(chunk.writeComponent(ecs::ComponentId::Mass)))
                wmass[ref.localIndex] = mass;
            if (auto *rmass = const_cast<math::Fixed32 *>(
                    static_cast<const math::Fixed32 *>(chunk.readComponent(ecs::ComponentId::Mass))))
                rmass[ref.localIndex] = mass;

            math::Vec3<math::Fixed32> size{math::Fixed32::fromInt(1), math::Fixed32::fromInt(2),
                                           math::Fixed32::fromInt(1)};
            if (auto *wsize = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::AABB)))
                wsize[ref.localIndex] = size;
            if (auto *rsize = const_cast<math::Vec3<math::Fixed32> *>(
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk.readComponent(ecs::ComponentId::AABB))))
                rsize[ref.localIndex] = size;

            core::i32 hp = 100;
            if (auto *whp = static_cast<core::i32 *>(chunk.writeComponent(ecs::ComponentId::Health)))
                whp[ref.localIndex] = hp;
            if (auto *rhp = const_cast<core::i32 *>(
                    static_cast<const core::i32 *>(chunk.readComponent(ecs::ComponentId::Health))))
                rhp[ref.localIndex] = hp;

            if (auto *ws = static_cast<core::u8 *>(chunk.writeComponent(ecs::ComponentId::SleepState)))
                ws[ref.localIndex] = 0;
        }

        auto fixedPos = math::Vec3<math::Fixed32>{math::Fixed32{0}, math::Fixed32::fromFloat(10.0f), math::Fixed32{0}};
        [[maybe_unused]] auto res = _impl->world.insertOrUpdate(entityId, fixedPos);

        net::protocol::PacketHeader header{};
        header.magic = net::protocol::kProtocolMagic;
        header.version = net::protocol::kProtocolVersion;
        header.type = net::protocol::PacketType::HandshakeAck;
        header.payloadSize = 4;

        core::byte packet[sizeof(header) + 4];
        std::memcpy(packet, &header, sizeof(header));
        std::memcpy(packet + sizeof(header), &netId, 4);

        [[maybe_unused]] auto sendRes =
            _impl->transport->send(std::span<const core::byte>{packet, sizeof(packet)}, session->address());

        core::Log::info("SessionSystem: client connected, entity assigned");
    }
}

} // namespace lpl::engine::systems
