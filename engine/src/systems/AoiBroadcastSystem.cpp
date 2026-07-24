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
#include <lpl/net/protocol/EntityDelta.hpp>
#include <lpl/net/protocol/Protocol.hpp>
#include <lpl/net/relevancy/Relevancy.hpp>
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
    core::u32 keyframeInterval; ///< Ticks between full re-sends of an in-range entity (§6.2.5).
    core::u32 budgetBytes;      ///< Per-client delta byte budget per tick; 0 = unlimited (§6.2.7).
    float nearRadiusSq{0.0f};   ///< Squared radius of the full-rate near ring; 0 = LOD off (§6.2.6).
    core::u32 lodFarInterval{1};///< Update interval (ticks) for entities beyond the near ring.
    core::u64 tickCounter{0};   ///< Advances each execute; a keyframe tick forces a full snapshot.

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

    /// The last snapshot the server sent each client, per entity — the baseline a
    /// field delta is computed against (§6.2.5). Mirrors `known`: an id enters
    /// with its spawn snapshot, updates on each delta, and is erased when it
    /// leaves the radius or the client disconnects.
    std::unordered_map<core::u32, std::unordered_map<core::u32, net::protocol::EntitySnapshot>> lastSent;

    /// Tick at which each entity was last actually sent to each client. Feeds the
    /// staleness term of the relevancy priority so a starved entity rises until it
    /// is sent, and lets an over-budget entity age instead of being dropped for
    /// good (§6.2.7).
    std::unordered_map<core::u32, std::unordered_map<core::u32, core::u64>> lastSentTick;

    // ---- Outbound batching. Buffers are pooled so their bytes outlive sendBatch
    // without reallocating each tick; a vector move keeps a buffer's data pointer.
    std::vector<std::vector<core::byte>> packetPool;
    core::usize packetUsed{0};
    std::vector<net::transport::Datagram> batch;
    net::protocol::Bitstream stream;

    Impl(net::session::SessionManager &sm, std::shared_ptr<net::transport::ITransport> t, ecs::WorldPartition &w,
         ecs::Registry &reg, math::Fixed32 radius, core::u32 keyframe, core::u32 budget)
        : sessionManager{sm}, transport{std::move(t)}, world{w}, registry{reg}, interestRadius{radius},
          keyframeInterval{keyframe}, budgetBytes{budget}
    {
    }

    /// The wire snapshot of a collected record (Fixed32 → float at the boundary).
    [[nodiscard]] net::protocol::EntitySnapshot snapshotOf(core::u32 recIndex) const
    {
        const Record &rec = records[recIndex];
        net::protocol::EntitySnapshot s{};
        s.id = rec.id;
        s.px = rec.pos.x.toFloat();
        s.py = rec.pos.y.toFloat();
        s.pz = rec.pos.z.toFloat();
        s.sx = rec.size.x.toFloat();
        s.sy = rec.size.y.toFloat();
        s.sz = rec.size.z.toFloat();
        s.hp = rec.hp;
        return s;
    }

    /// True when this tick re-sends in-range entities in full, so a lost delta
    /// self-heals. keyframeInterval <= 1 means "always full" (delta compression
    /// off); otherwise every keyframeInterval-th tick is a keyframe.
    [[nodiscard]] bool isKeyframeTick() const noexcept
    {
        return keyframeInterval <= 1 || (tickCounter % keyframeInterval == 0);
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

    // ---- Field-delta scratch for one session (reused, capacity kept).
    struct Candidate {
        net::protocol::EntitySnapshot snap;
        core::u8 mask;
        float priority;
    };
    std::vector<Candidate> candidates;
    std::vector<std::pair<net::protocol::EntitySnapshot, core::u8>> deltaScratch;

    /// Byte size of one field-delta on the wire (id + mask + one float per bit).
    [[nodiscard]] static core::u32 deltaBytes(core::u8 mask) noexcept
    {
        return kIdBytes + 1u + static_cast<core::u32>(__builtin_popcount(mask)) * 4u;
    }

    /// Serialises the still-in-range entities for @p playerId as field-masked
    /// deltas (§6.2.5) under a relevancy priority and a byte budget (§6.2.7):
    ///   - dormancy: an entity whose fields did not change sends nothing between
    ///     keyframes, so a resting world costs no traffic;
    ///   - priority: the rest are ordered by proximity + staleness, so the closest
    ///     and the longest-unsent go first;
    ///   - budget: only the top entities up to @c budgetBytes are sent; the others
    ///     age (their staleness rises) and win a later tick — nothing starves.
    /// The per-client baseline and last-sent tick are updated only for entities
    /// actually sent. Fragments on the real (variable) byte size.
    void emitDeltas(core::u32 playerId, const std::vector<core::u32> &recIndices,
                    const math::Vec3<math::Fixed32> &center, const net::Endpoint *address)
    {
        auto &baseline = lastSent[playerId];
        auto &sentTick = lastSentTick[playerId];
        const bool keyframe = isKeyframeTick();

        // 1. Build the due, scored candidate set (dormant entities drop out here).
        candidates.clear();
        for (const core::u32 recIndex : recIndices)
        {
            net::protocol::EntitySnapshot cur = snapshotOf(recIndex);
            auto it = baseline.find(cur.id);
            const bool known = (it != baseline.end());
            core::u8 mask = (keyframe || !known) ? static_cast<core::u8>(net::protocol::FieldAll)
                                                 : net::protocol::computeFieldMask(it->second, cur);

            const auto &pos = records[recIndex].pos;
            const float dx = (pos.x - center.x).toFloat();
            const float dy = (pos.y - center.y).toFloat();
            const float dz = (pos.z - center.z).toFloat();
            const float distanceSq = dx * dx + dy * dy + dz * dz;

            auto tickIt = sentTick.find(cur.id);
            const core::u64 last = (tickIt != sentTick.end()) ? tickIt->second : 0;
            const auto ticksSince = static_cast<core::u32>(tickCounter > last ? tickCounter - last : 0);

            if (!keyframe)
            {
                if (mask == 0)
                    continue; // dormant: unchanged and not a keyframe → no traffic
                // Network LOD (§6.2.6): a far entity updates on a slower cadence.
                const core::u32 interval = net::relevancy::lodUpdateInterval(distanceSq, nearRadiusSq, lodFarInterval);
                if (ticksSince < interval)
                    continue; // far ring, not its tick yet — its change batches
            }

            candidates.push_back(Candidate{cur, mask, net::relevancy::priority(distanceSq, ticksSince)});
        }

        // 2. Highest priority first.
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate &a, const Candidate &b) { return a.priority > b.priority; });

        // 3. Emit under budget; the rest age for a later tick.
        deltaScratch.clear();
        core::u32 spent = 0;
        for (const auto &c : candidates)
        {
            const core::u32 cost = deltaBytes(c.mask);
            if (budgetBytes != 0 && !deltaScratch.empty() && spent + cost > budgetBytes)
                break; // always send at least one to guarantee progress
            deltaScratch.emplace_back(c.snap, c.mask);
            baseline[c.snap.id] = c.snap;   // the client will now hold this
            sentTick[c.snap.id] = tickCounter;
            spent += cost;
        }

        // Pack into datagrams, fragmenting when the next entity would overflow.
        // An entity's wire size is known from its mask (id + mask byte + one float
        // per set bit), so a group's boundary is found by arithmetic — no
        // throwaway serialisation pass.
        const core::u32 maxPayload = net::session::SessionManager::kMaxPayloadSize;
        core::u32 i = 0;
        const auto n = static_cast<core::u32>(deltaScratch.size());
        while (i < n)
        {
            core::u32 bytes = kCountHeaderBytes;
            core::u32 j = i;
            for (; j < n; ++j)
            {
                const core::u32 entBytes =
                    kIdBytes + 1u + static_cast<core::u32>(__builtin_popcount(deltaScratch[j].second)) * 4u;
                if (bytes + entBytes > maxPayload && j > i)
                    break;
                bytes += entBytes;
            }

            stream.reset();
            stream.writeU16(static_cast<core::u16>(j - i));
            for (core::u32 k = i; k < j; ++k)
                net::protocol::writeEntityDelta(stream, deltaScratch[k].first, deltaScratch[k].second);

            emitPacket(net::protocol::PacketType::StateDelta, stream.data(), address);
            i = j;
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
    : AoiBroadcastSystem{sessionManager, std::move(transport), world, registry, interestRadius, 60, 0}
{
}

AoiBroadcastSystem::AoiBroadcastSystem(net::session::SessionManager &sessionManager,
                                       std::shared_ptr<net::transport::ITransport> transport, ecs::WorldPartition &world,
                                       ecs::Registry &registry, math::Fixed32 interestRadius, core::u32 keyframeInterval)
    : AoiBroadcastSystem{sessionManager, std::move(transport), world, registry, interestRadius, keyframeInterval, 0}
{
}

AoiBroadcastSystem::AoiBroadcastSystem(net::session::SessionManager &sessionManager,
                                       std::shared_ptr<net::transport::ITransport> transport, ecs::WorldPartition &world,
                                       ecs::Registry &registry, math::Fixed32 interestRadius, core::u32 keyframeInterval,
                                       core::u32 budgetBytes)
    : _impl{std::make_unique<Impl>(sessionManager, std::move(transport), world, registry, interestRadius,
                                   keyframeInterval, budgetBytes)}
{
}

AoiBroadcastSystem::~AoiBroadcastSystem() = default;

void AoiBroadcastSystem::setNetworkLod(math::Fixed32 nearRadius, core::u32 farInterval) noexcept
{
    const float r = nearRadius.toFloat();
    _impl->nearRadiusSq = (r > 0.0f) ? r * r : 0.0f;
    _impl->lodFarInterval = farInterval < 1 ? 1 : farInterval;
}

const ecs::SystemDescriptor &AoiBroadcastSystem::descriptor() const noexcept { return kAoiDesc; }

void AoiBroadcastSystem::execute(core::f32 /*dt*/)
{
    auto &d = *_impl;

    d.packetUsed = 0;
    d.batch.clear();
    d.seenPlayers.clear();
    ++d.tickCounter;

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
        auto &baseline = d.lastSent[playerId];
        auto &sentTick = d.lastSentTick[playerId];
        if (!d.enteredRecs.empty())
        {
            // A spawn is a full snapshot; record it as the delta baseline (and its
            // send tick) so the next tick's field delta and priority have a base.
            d.emitEntities(net::protocol::PacketType::EntitySpawn, d.enteredRecs, address);
            for (const core::u32 recIndex : d.enteredRecs)
            {
                const core::u32 rawId = d.records[recIndex].id;
                baseline[rawId] = d.snapshotOf(recIndex);
                sentTick[rawId] = d.tickCounter;
            }
        }
        if (!d.movedRecs.empty())
            d.emitDeltas(playerId, d.movedRecs, center, address);
        if (!d.leftIds.empty())
        {
            d.emitDestroy(d.leftIds, address);
            for (const core::u32 raw : d.leftIds)
            {
                baseline.erase(raw); // no longer replicated to this client
                sentTick.erase(raw);
            }
        }

        // The neighbour set becomes what the client now knows. Swap, not copy: the
        // scratch is cleared at the top of the next session anyway.
        known.swap(d.neighborSet);
    });

    // Drop the per-client memory (known set AND delta baseline) of clients that
    // are gone, so a disconnected player's state does not linger — and a recycled
    // playerId does not inherit a stale set or baseline.
    if (d.known.size() != d.seenPlayers.size())
    {
        for (auto it = d.known.begin(); it != d.known.end();)
        {
            if (d.sessionManager.find(it->first) == nullptr)
            {
                d.lastSent.erase(it->first);
                d.lastSentTick.erase(it->first);
                it = d.known.erase(it);
            }
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
