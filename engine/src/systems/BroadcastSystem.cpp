#include <lpl/core/Log.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/engine/systems/BroadcastSystem.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/net/protocol/Bitstream.hpp>
#include <lpl/net/protocol/Protocol.hpp>

#include <algorithm>
#include <cstring>
#include <vector>

namespace lpl::engine::systems {

static const ecs::ComponentAccess kBroadcastAccesses[] = {
    {ecs::ComponentId::Position,    ecs::AccessMode::ReadOnly},
    {ecs::ComponentId::AABB,        ecs::AccessMode::ReadOnly},
    {ecs::ComponentId::Health,      ecs::AccessMode::ReadOnly},
    {ecs::ComponentId::NetworkSync, ecs::AccessMode::ReadOnly},
};

static const ecs::SystemDescriptor kBroadcastDesc{"Broadcast", ecs::SchedulePhase::Network,
                                                  std::span<const ecs::ComponentAccess>{kBroadcastAccesses}};

struct BroadcastSystem::Impl {
    net::session::SessionManager &sessionManager;
    std::shared_ptr<net::transport::ITransport> transport;
    ecs::WorldPartition &world;
    ecs::Registry &registry;

    net::protocol::Bitstream stream; /**< Bitstream for packet assembly. */

    Impl(net::session::SessionManager &sm, std::shared_ptr<net::transport::ITransport> t, ecs::WorldPartition &w,
         ecs::Registry &reg)
        : sessionManager{sm}, transport{std::move(t)}, world{w}, registry{reg}
    {
    }

    /**
     * @brief Prefixes the stream with a StateSnapshot header and broadcasts it.
     *
     * SessionManager::broadcastState splits anything longer than kMaxPayloadSize into
     * datagrams of that size, so the packet is passed whole.
     */
    void flushStream()
    {
        if (stream.bitsWritten() == 0)
            return;

        auto payload = stream.data();
        const core::u32 payloadSize = static_cast<core::u32>(payload.size());

        std::vector<core::byte> pkt(sizeof(net::protocol::PacketHeader) + payloadSize);
        auto &header = *reinterpret_cast<net::protocol::PacketHeader *>(pkt.data());
        header.magic = net::protocol::kProtocolMagic;
        header.version = net::protocol::kProtocolVersion;
        header.type = net::protocol::PacketType::StateSnapshot;
        header.flags = 0;
        header.padding = 0;
        header.sequence = 0;
        header.payloadSize = payloadSize;

        std::memcpy(pkt.data() + sizeof(header), payload.data(), payloadSize);

        sessionManager.broadcastState(*transport, std::span<const core::byte>{pkt.data(), pkt.size()});

        stream.reset();
    }
};

BroadcastSystem::BroadcastSystem(net::session::SessionManager &sessionManager,
                                 std::shared_ptr<net::transport::ITransport> transport, ecs::WorldPartition &world,
                                 ecs::Registry &registry)
    : _impl{std::make_unique<Impl>(sessionManager, std::move(transport), world, registry)}
{
}

BroadcastSystem::~BroadcastSystem() = default;

const ecs::SystemDescriptor &BroadcastSystem::descriptor() const noexcept { return kBroadcastDesc; }

void BroadcastSystem::execute(core::f32 /*dt*/)
{
    if (_impl->sessionManager.activeCount() == 0)
        return;

    _impl->stream.reset();

    static constexpr core::u32 kEntityBytes = 32;
    static constexpr core::u32 kCountHeaderBytes = 2;
    static constexpr core::u32 kMaxEntitiesPerPacket =
        (net::session::SessionManager::kMaxPayloadSize - kCountHeaderBytes) / kEntityBytes;

    struct EntityRecord {
        core::u32 id;
        float px, py, pz;
        float sx, sy, sz;
        core::i32 hp;
    };
    std::vector<EntityRecord> records;

    const auto &partitions = _impl->registry.partitions();
    for (const auto &part : partitions)
    {
        if (!part)
            continue;

        const auto &archetype = part->archetype();
        if (!archetype.has(ecs::ComponentId::Position))
            continue;

        for (const auto &chunk : part->chunks())
        {
            const core::u32 count = chunk->count();
            if (count == 0)
                continue;

            const auto *positions =
                static_cast<const math::Vec3<math::Fixed32> *>(chunk->readComponent(ecs::ComponentId::Position));
            const auto *sizes =
                archetype.has(ecs::ComponentId::AABB) ?
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk->readComponent(ecs::ComponentId::AABB)) :
                    nullptr;
            const auto *health = archetype.has(ecs::ComponentId::Health) ?
                                     static_cast<const core::i32 *>(chunk->readComponent(ecs::ComponentId::Health)) :
                                     nullptr;

            if (!positions)
                continue;

            auto entityIds = chunk->entities();

            for (core::u32 i = 0; i < count; ++i)
            {
                records.push_back({entityIds[i].raw(), positions[i].x.toFloat(), positions[i].y.toFloat(),
                                   positions[i].z.toFloat(), sizes ? sizes[i].x.toFloat() : 1.0f,
                                   sizes ? sizes[i].y.toFloat() : 1.0f, sizes ? sizes[i].z.toFloat() : 1.0f,
                                   health ? health[i] : 100});
            }
        }
    }

    const core::u32 total = static_cast<core::u32>(records.size());
    core::u32 offset = 0;

    while (offset < total || offset == 0)
    {
        const core::u32 batchSize = std::min(kMaxEntitiesPerPacket, total - offset);

        _impl->stream.reset();
        _impl->stream.writeU16(static_cast<core::u16>(batchSize));

        for (core::u32 i = 0; i < batchSize; ++i)
        {
            const auto &rec = records[offset + i];
            _impl->stream.writeU32(rec.id);
            _impl->stream.writeFloat(rec.px);
            _impl->stream.writeFloat(rec.py);
            _impl->stream.writeFloat(rec.pz);
            _impl->stream.writeFloat(rec.sx);
            _impl->stream.writeFloat(rec.sy);
            _impl->stream.writeFloat(rec.sz);
            _impl->stream.writeI32(rec.hp);
        }

        _impl->flushStream();
        offset += batchSize;

        if (total == 0)
            break;
    }
}

} // namespace lpl::engine::systems
