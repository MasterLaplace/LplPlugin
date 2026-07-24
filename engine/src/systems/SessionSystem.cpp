/**
 * @file SessionSystem.cpp
 * @brief Handles client connections: creates entities, sends welcome.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#include <lpl/core/Log.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/engine/systems/SessionSystem.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/net/protocol/Protocol.hpp>

#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>

namespace lpl::engine::systems {

// ========================================================================== //
//  Descriptor                                                                //
// ========================================================================== //

static const ecs::ComponentAccess kSessionAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Velocity, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Health,   ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Mass,     ecs::AccessMode::ReadWrite},
};

static const ecs::SystemDescriptor kSessionDesc{"Session", ecs::SchedulePhase::Input,
                                                std::span<const ecs::ComponentAccess>{kSessionAccesses}};

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

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

    /// Tear down everything a departing client owned: its avatar entity, its
    /// input state and its spatial-index entry. Keyed on the entity id, which is
    /// also the session's playerId (see the id-unification note above). Without
    /// this a disconnect would leak an avatar that lingers and is broadcast — AOI
    /// then sends the neighbours an EntityDestroy on the next tick, once the
    /// entity is gone from the registry and the grid.
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

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

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
    // ─── Departures first, so a slot freed this tick can be reused ─── //
    // Explicit disconnects: the client asked to leave.
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

    // Idle timeout: a client that stopped sending is treated as gone. Input is
    // the heartbeat (InputProcessingSystem touches the session every tick a client
    // sends), so this only fires for clients that truly went silent.
    if (_impl->sessionTimeoutMs > 0.0)
    {
        [[maybe_unused]] auto reaped =
            _impl->sessionManager.reapTimedOut(_impl->sessionTimeoutMs,
                                               [this](const net::session::Session &s) { _impl->teardown(s); });
    }

    auto events = _impl->queues.connects.drain();

    for (const auto &ev : events)
    {
        // Legacy parity (SessionManager::handleConnections): a client is known by
        // its endpoint, so a handshake from an address that already has a session
        // is a retransmission — common on UDP — and must NOT spawn a second
        // player. The previous check asked isDuplicate() about a freshly minted
        // id, which by construction never collides, so it could never fire.
        if (ev.source.valid() && _impl->sessionManager.findByAddress(ev.source) != nullptr)
        {
            continue;
        }

        // ─── Create entity in Registry with full component set ─── //
        // Mirrors legacy SessionManager::connect() which created a full
        // EntitySnapshot with Position, Velocity, Mass, AABB, Health.
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

        // ONE network identity: the entity's ECS id. The legacy keyed session,
        // input and the welcome on a single publicId; the modern port had split it
        // into a separate player counter (session/input) and the ECS id
        // (broadcast/hash), so the input a client sent — tagged with the welcome
        // id — never matched the entity MovementSystem drives (keyed by the ECS
        // id), and the client could not find its own avatar in a snapshot. Keying
        // everything on entityId.raw() closes the loop: welcome → client input →
        // InputProcessing → Movement all name the same entity.
        const core::u32 netId = entityId.raw();

        auto sessionResult = _impl->sessionManager.connect(netId);
        if (!sessionResult.has_value())
        {
            // Undo the entity: a session id collision means we would leak it.
            [[maybe_unused]] auto d = _impl->registry.destroyEntity(entityId);
            continue;
        }

        // Store the client's network address in the session for broadcast
        auto *session = sessionResult.value();
        if (ev.source.valid())
        {
            session->setAddress(ev.source);
        }

        // Bind the avatar to the session: this is the client's centre of interest.
        // AoiBroadcastSystem reads it back (Session::boundEntity) to know where to
        // run the radius query; without the bind the session has a null entity and
        // AOI could not place the client in the world.
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

            // Position: spawn at {0, 10, 0} (legacy default) — authoritative Fixed32
            math::Vec3<math::Fixed32> pos{math::Fixed32::fromInt(0), math::Fixed32::fromInt(10),
                                          math::Fixed32::fromInt(0)};
            if (auto *wpos = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Position)))
                wpos[ref.localIndex] = pos;
            if (auto *rpos = const_cast<math::Vec3<math::Fixed32> *>(
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk.readComponent(ecs::ComponentId::Position))))
                rpos[ref.localIndex] = pos;

            // Velocity: zero
            math::Vec3<math::Fixed32> vel{math::Fixed32::zero(), math::Fixed32::zero(), math::Fixed32::zero()};
            if (auto *wvel = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Velocity)))
                wvel[ref.localIndex] = vel;

            // Mass: 1.0
            math::Fixed32 mass = math::Fixed32::one();
            if (auto *wmass = static_cast<math::Fixed32 *>(chunk.writeComponent(ecs::ComponentId::Mass)))
                wmass[ref.localIndex] = mass;
            if (auto *rmass = const_cast<math::Fixed32 *>(
                    static_cast<const math::Fixed32 *>(chunk.readComponent(ecs::ComponentId::Mass))))
                rmass[ref.localIndex] = mass;

            // AABB (size): {1, 2, 1}
            math::Vec3<math::Fixed32> size{math::Fixed32::fromInt(1), math::Fixed32::fromInt(2),
                                           math::Fixed32::fromInt(1)};
            if (auto *wsize = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::AABB)))
                wsize[ref.localIndex] = size;
            if (auto *rsize = const_cast<math::Vec3<math::Fixed32> *>(
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk.readComponent(ecs::ComponentId::AABB))))
                rsize[ref.localIndex] = size;

            // Health: 100
            core::i32 hp = 100;
            if (auto *whp = static_cast<core::i32 *>(chunk.writeComponent(ecs::ComponentId::Health)))
                whp[ref.localIndex] = hp;
            if (auto *rhp = const_cast<core::i32 *>(
                    static_cast<const core::i32 *>(chunk.readComponent(ecs::ComponentId::Health))))
                rhp[ref.localIndex] = hp;

            // SleepState: awake
            if (auto *ws = static_cast<core::u8 *>(chunk.writeComponent(ecs::ComponentId::SleepState)))
                ws[ref.localIndex] = 0;
        }

        // Insert into spatial index
        auto fixedPos = math::Vec3<math::Fixed32>{math::Fixed32{0}, math::Fixed32::fromFloat(10.0f), math::Fixed32{0}};
        [[maybe_unused]] auto res = _impl->world.insertOrUpdate(entityId, fixedPos);

        // Send welcome packet with entity ID
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
