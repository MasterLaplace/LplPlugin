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
    core::u32 nextEntityId{100};

    Impl(net::session::SessionManager &sm, EventQueues &q, std::shared_ptr<net::transport::ITransport> t,
         input::InputManager &im, ecs::WorldPartition &w, ecs::Registry &reg)
        : sessionManager{sm}, queues{q}, transport{std::move(t)}, inputManager{im}, world{w}, registry{reg}
    {
    }
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

SessionSystem::SessionSystem(net::session::SessionManager &sessionManager, EventQueues &queues,
                             std::shared_ptr<net::transport::ITransport> transport, input::InputManager &inputManager,
                             ecs::WorldPartition &world, ecs::Registry &registry)
    : _impl{std::make_unique<Impl>(sessionManager, queues, std::move(transport), inputManager, world, registry)}
{
}

SessionSystem::~SessionSystem() = default;

const ecs::SystemDescriptor &SessionSystem::descriptor() const noexcept { return kSessionDesc; }

void SessionSystem::execute(core::f32 /*dt*/)
{
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

        // Only mint an id once the connection is accepted, so a rejected
        // duplicate does not burn one.
        const core::u32 newId = _impl->nextEntityId++;

        auto sessionResult = _impl->sessionManager.connect(newId);
        if (!sessionResult.has_value())
        {
            continue;
        }

        // Store the client's network address in the session for broadcast
        auto *session = sessionResult.value();
        if (ev.source.valid())
        {
            session->setAddress(ev.source);
        }

        [[maybe_unused]] auto &_ = _impl->inputManager.getOrCreate(newId);

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
        std::memcpy(packet + sizeof(header), &newId, 4);

        [[maybe_unused]] auto sendRes =
            _impl->transport->send(std::span<const core::byte>{packet, sizeof(packet)}, session->address());

        core::Log::info("SessionSystem: client connected, entity assigned");
    }
}

} // namespace lpl::engine::systems
