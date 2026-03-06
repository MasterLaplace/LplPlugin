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
        const core::u32 newId = _impl->nextEntityId++;

        if (_impl->sessionManager.isDuplicate(newId))
        {
            continue;
        }

        auto sessionResult = _impl->sessionManager.connect(newId);
        if (!sessionResult.has_value())
        {
            continue;
        }

        // Store the client's network address in the session for broadcast
        auto *session = sessionResult.value();
        if (ev.rawAddrLen > 0)
        {
            session->setAddress(ev.rawAddr.data(), ev.rawAddrLen);
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

            // Position: spawn at {0, 10, 0} (legacy default)
            math::Vec3<float> pos{0.0f, 10.0f, 0.0f};
            if (auto *wpos = static_cast<math::Vec3<float> *>(chunk.writeComponent(ecs::ComponentId::Position)))
                wpos[ref.localIndex] = pos;
            if (auto *rpos = const_cast<math::Vec3<float> *>(
                    static_cast<const math::Vec3<float> *>(chunk.readComponent(ecs::ComponentId::Position))))
                rpos[ref.localIndex] = pos;

            // Velocity: zero
            math::Vec3<float> vel{0.0f, 0.0f, 0.0f};
            if (auto *wvel = static_cast<math::Vec3<float> *>(chunk.writeComponent(ecs::ComponentId::Velocity)))
                wvel[ref.localIndex] = vel;

            // Mass: 1.0
            float mass = 1.0f;
            if (auto *wmass = static_cast<float *>(chunk.writeComponent(ecs::ComponentId::Mass)))
                wmass[ref.localIndex] = mass;
            if (auto *rmass =
                    const_cast<float *>(static_cast<const float *>(chunk.readComponent(ecs::ComponentId::Mass))))
                rmass[ref.localIndex] = mass;

            // AABB (size): {1, 2, 1}
            math::Vec3<float> size{1.0f, 2.0f, 1.0f};
            if (auto *wsize = static_cast<math::Vec3<float> *>(chunk.writeComponent(ecs::ComponentId::AABB)))
                wsize[ref.localIndex] = size;
            if (auto *rsize = const_cast<math::Vec3<float> *>(
                    static_cast<const math::Vec3<float> *>(chunk.readComponent(ecs::ComponentId::AABB))))
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
