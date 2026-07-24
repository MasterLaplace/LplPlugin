/**
 * @file AoiBroadcastSystem.cpp
 * @brief Area-of-interest server broadcast implementation.
 *
 * For each connected client, queries the spatial index around the client's own
 * avatar and sends only what is near it, as a delta against what that client was
 * last told about:
 *   - EntitySpawn  (0x20): entities newly inside the radius   [u16 count][32B]*
 *   - StateDelta   (0x12): entities still inside, current xf   [u16 count][32B]*
 *   - EntityDestroy(0x21): entities that left the radius       [u16 count][u32]*
 * The 32-byte entity layout matches StateSnapshot:
 *   [id:4][posX:4][posY:4][posZ:4][sizeX:4][sizeY:4][sizeZ:4][hp:4]
 *
 * Every datagram for the tick is handed to the transport in one sendBatch, the
 * same batching win broadcastState takes.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-23
 * @copyright MIT License
 */

#include <lpl/engine/systems/AoiBroadcastSystem.hpp>

#include <lpl/ecs/Partition.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/net/protocol/Bitstream.hpp>
#include <lpl/net/protocol/Protocol.hpp>
#include <lpl/std/vector.hpp>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lpl::engine::systems {

// ========================================================================== //
//  Descriptor                                                                //
// ========================================================================== //

static const ecs::ComponentAccess kAoiAccesses[] = {
    {ecs::ComponentId::Position,    ecs::AccessMode::ReadOnly},
    {ecs::ComponentId::AABB,        ecs::AccessMode::ReadOnly},
    {ecs::ComponentId::Health,      ecs::AccessMode::ReadOnly},
    {ecs::ComponentId::NetworkSync, ecs::AccessMode::ReadOnly},
};

static const ecs::SystemDescriptor kAoiDesc{"AoiBroadcast", ecs::SchedulePhase::Network,
                                            std::span<const ecs::ComponentAccess>{kAoiAccesses}};

// ========================================================================== //
//  Wire-format constants                                                     //
// ========================================================================== //

namespace {

constexpr core::u32 kEntityBytes = 32;      ///< id + 6 floats + hp.
constexpr core::u32 kIdBytes = 4;           ///< one raw entity id.
constexpr core::u32 kCountHeaderBytes = 2;  ///< u16 count prefix.

/// Entities that fit in one datagram: (1400 - 2) / 32 = 43.
constexpr core::u32 kMaxEntitiesPerPacket =
    (net::session::SessionManager::kMaxPayloadSize - kCountHeaderBytes) / kEntityBytes;
/// Despawn ids that fit in one datagram: (1400 - 2) / 4 = 349.
constexpr core::u32 kMaxIdsPerPacket = (net::session::SessionManager::kMaxPayloadSize - kCountHeaderBytes) / kIdBytes;

} // namespace

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct AoiBroadcastSystem::Impl {
    net::session::SessionManager &sessionManager;
    std::shared_ptr<net::transport::ITransport> transport;
    ecs::WorldPartition &world;
    ecs::Registry &registry;
    math::Fixed32 interestRadius;

    /// One entity's authoritative snapshot, collected once per tick. Position and
    /// size stay Fixed32 (authoritative) until the wire boundary, so the radius
    /// query centre is exact and the float conversion happens only when serialised.
    struct Record {
        core::u32 id;
        math::Vec3<math::Fixed32> pos;
        math::Vec3<math::Fixed32> size;
        core::i32 hp;
    };

    // ---- Per-tick scratch, reused (capacity kept) so a tick does not allocate.
    std::vector<Record> records;              ///< Every entity with a Position, this tick.
    std::unordered_map<core::u32, core::u32> idToRecord; ///< raw id -> index into `records`.
    pmr::vector<ecs::EntityId> neighbors;     ///< queryRadius output.
    std::unordered_set<core::u32> neighborSet;///< This session's neighbours (raw ids).
    std::vector<core::u32> enteredRecs;       ///< Record indices new to the client.
    std::vector<core::u32> movedRecs;         ///< Record indices still known to the client.
    std::vector<core::u32> leftIds;           ///< Raw ids that left the client's radius.

    // ---- Per-session persistent state: what each client currently knows about.
    std::unordered_map<core::u32, std::unordered_set<core::u32>> known; ///< playerId -> known ids.
    std::vector<core::u32> seenPlayers;       ///< playerIds present this tick (for reaping `known`).

    // ---- Outbound batching. Buffers are pooled so their bytes outlive sendBatch
    // without reallocating each tick; a vector move keeps a buffer's data pointer.
    std::vector<std::vector<core::byte>> packetPool;
    core::usize packetUsed{0};
    std::vector<net::transport::Datagram> batch;
    net::protocol::Bitstream stream;

    Impl(net::session::SessionManager &sm, std::shared_ptr<net::transport::ITransport> t, ecs::WorldPartition &w,
         ecs::Registry &reg, math::Fixed32 radius)
        : sessionManager{sm}, transport{std::move(t)}, world{w}, registry{reg}, interestRadius{radius}
    {
    }

    /// Grabs the next pooled buffer, writes header + payload into it, and queues a
    /// datagram pointing at it (borrowing @p address, which outlives the tick).
    void emitPacket(net::protocol::PacketType type, std::span<const core::byte> payload, const net::Endpoint *address)
    {
        if (packetUsed >= packetPool.size())
            packetPool.emplace_back();
        auto &buffer = packetPool[packetUsed++];

        buffer.resize(sizeof(net::protocol::PacketHeader) + payload.size());

        auto &header = *reinterpret_cast<net::protocol::PacketHeader *>(buffer.data());
        header.magic = net::protocol::kProtocolMagic;
        header.version = net::protocol::kProtocolVersion;
        header.type = type;
        header.flags = 0;
        header.padding = 0;
        header.sequence = 0;
        header.payloadSize = static_cast<core::u32>(payload.size());

        if (!payload.empty())
            std::memcpy(buffer.data() + sizeof(header), payload.data(), payload.size());

        batch.push_back(net::transport::Datagram{std::span<const core::byte>{buffer.data(), buffer.size()}, address});
    }

    /// Serialises record indices as one or more 32-byte-entity packets of @p type.
    void emitEntities(net::protocol::PacketType type, const std::vector<core::u32> &recIndices,
                      const net::Endpoint *address)
    {
        const core::u32 total = static_cast<core::u32>(recIndices.size());
        for (core::u32 offset = 0; offset < total; offset += kMaxEntitiesPerPacket)
        {
            const core::u32 batchSize = std::min(kMaxEntitiesPerPacket, total - offset);

            stream.reset();
            stream.writeU16(static_cast<core::u16>(batchSize));
            for (core::u32 i = 0; i < batchSize; ++i)
            {
                const Record &rec = records[recIndices[offset + i]];
                stream.writeU32(rec.id);
                stream.writeFloat(rec.pos.x.toFloat());
                stream.writeFloat(rec.pos.y.toFloat());
                stream.writeFloat(rec.pos.z.toFloat());
                stream.writeFloat(rec.size.x.toFloat());
                stream.writeFloat(rec.size.y.toFloat());
                stream.writeFloat(rec.size.z.toFloat());
                stream.writeI32(rec.hp);
            }
            emitPacket(type, stream.data(), address);
        }
    }

    /// Serialises raw ids as one or more EntityDestroy packets.
    void emitDestroy(const std::vector<core::u32> &ids, const net::Endpoint *address)
    {
        const core::u32 total = static_cast<core::u32>(ids.size());
        for (core::u32 offset = 0; offset < total; offset += kMaxIdsPerPacket)
        {
            const core::u32 batchSize = std::min(kMaxIdsPerPacket, total - offset);

            stream.reset();
            stream.writeU16(static_cast<core::u16>(batchSize));
            for (core::u32 i = 0; i < batchSize; ++i)
                stream.writeU32(ids[offset + i]);
            emitPacket(net::protocol::PacketType::EntityDestroy, stream.data(), address);
        }
    }

    /// Rebuilds `records` and `idToRecord` from the registry (one pass over all
    /// entities that carry a Position).
    void collectRecords()
    {
        records.clear();
        idToRecord.clear();

        for (const auto &part : registry.partitions())
        {
            if (!part)
                continue;

            const auto &archetype = part->archetype();
            if (!archetype.has(ecs::ComponentId::Position))
                continue;

            const bool hasSize = archetype.has(ecs::ComponentId::AABB);
            const bool hasHealth = archetype.has(ecs::ComponentId::Health);

            for (const auto &chunk : part->chunks())
            {
                const core::u32 count = chunk->count();
                if (count == 0)
                    continue;

                const auto *positions =
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk->readComponent(ecs::ComponentId::Position));
                if (!positions)
                    continue;

                const auto *sizes =
                    hasSize ? static_cast<const math::Vec3<math::Fixed32> *>(chunk->readComponent(ecs::ComponentId::AABB))
                            : nullptr;
                const auto *health =
                    hasHealth ? static_cast<const core::i32 *>(chunk->readComponent(ecs::ComponentId::Health)) : nullptr;

                const auto entityIds = chunk->entities();
                for (core::u32 i = 0; i < count; ++i)
                {
                    const core::u32 raw = entityIds[i].raw();
                    Record rec{};
                    rec.id = raw;
                    rec.pos = positions[i];
                    rec.size = sizes ? sizes[i]
                                     : math::Vec3<math::Fixed32>{math::Fixed32::one(), math::Fixed32::one(),
                                                                 math::Fixed32::one()};
                    rec.hp = health ? health[i] : 100;
                    idToRecord[raw] = static_cast<core::u32>(records.size());
                    records.push_back(rec);
                }
            }
        }
    }
};

// ========================================================================== //
//  Public                                                                    //
// ========================================================================== //

AoiBroadcastSystem::AoiBroadcastSystem(net::session::SessionManager &sessionManager,
                                       std::shared_ptr<net::transport::ITransport> transport, ecs::WorldPartition &world,
                                       ecs::Registry &registry, math::Fixed32 interestRadius)
    : _impl{std::make_unique<Impl>(sessionManager, std::move(transport), world, registry, interestRadius)}
{
}

AoiBroadcastSystem::~AoiBroadcastSystem() = default;

const ecs::SystemDescriptor &AoiBroadcastSystem::descriptor() const noexcept { return kAoiDesc; }

void AoiBroadcastSystem::execute(core::f32 /*dt*/)
{
    auto &d = *_impl;

    d.packetUsed = 0;
    d.batch.clear();
    d.seenPlayers.clear();

    // One snapshot pass over the whole world; each session then filters it by its
    // own radius. This is what turns O(clients × N) serialisation into O(N) +
    // O(clients × neighbours).
    d.collectRecords();

    d.sessionManager.forEach([&](net::session::Session &session) {
        if (session.state() != net::session::SessionState::Connected)
            return;

        const core::u32 playerId = session.playerId();
        d.seenPlayers.push_back(playerId);

        const ecs::EntityId avatar = session.boundEntity();
        auto &known = d.known[playerId]; // creates an empty set on first sight

        if (!avatar.isValid())
            return; // no avatar yet: nothing to centre the query on

        const auto centerIt = d.idToRecord.find(avatar.raw());
        if (centerIt == d.idToRecord.end())
            return; // avatar not in the world this tick

        const math::Vec3<math::Fixed32> center = d.records[centerIt->second].pos;

        // Neighbours within the radius (grid-cell granular — an interest query,
        // not exact geometry). Keep only ids we actually have a record for.
        d.neighbors.clear();
        d.world.queryRadius(center, d.interestRadius, d.neighbors);

        d.neighborSet.clear();
        for (const ecs::EntityId id : d.neighbors)
        {
            if (d.idToRecord.find(id.raw()) != d.idToRecord.end())
                d.neighborSet.insert(id.raw());
        }

        // Diff against what this client already knows.
        d.enteredRecs.clear();
        d.movedRecs.clear();
        d.leftIds.clear();

        for (const core::u32 raw : d.neighborSet)
        {
            const core::u32 recIndex = d.idToRecord.find(raw)->second;
            if (known.find(raw) == known.end())
                d.enteredRecs.push_back(recIndex);
            else
                d.movedRecs.push_back(recIndex);
        }
        for (const core::u32 raw : known)
        {
            if (d.neighborSet.find(raw) == d.neighborSet.end())
                d.leftIds.push_back(raw);
        }

        const net::Endpoint *address = session.address();
        if (!d.enteredRecs.empty())
            d.emitEntities(net::protocol::PacketType::EntitySpawn, d.enteredRecs, address);
        if (!d.movedRecs.empty())
            d.emitEntities(net::protocol::PacketType::StateDelta, d.movedRecs, address);
        if (!d.leftIds.empty())
            d.emitDestroy(d.leftIds, address);

        // The neighbour set becomes what the client now knows. Swap, not copy: the
        // scratch is cleared at the top of the next session anyway.
        known.swap(d.neighborSet);
    });

    // Drop known-sets of clients that are gone, so a disconnected player's memory
    // does not linger (and a recycled playerId does not inherit a stale set).
    if (d.known.size() != d.seenPlayers.size())
    {
        for (auto it = d.known.begin(); it != d.known.end();)
        {
            if (d.sessionManager.find(it->first) == nullptr)
                it = d.known.erase(it);
            else
                ++it;
        }
    }

    if (!d.batch.empty())
    {
        [[maybe_unused]] auto result = d.transport->sendBatch(d.batch);
    }
}

} // namespace lpl::engine::systems
