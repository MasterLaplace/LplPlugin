#include <lpl/engine/systems/AoiBroadcastSystem.hpp>

#ifdef LPL_HAS_NET

#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/ecs/Partition.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/engine/EventQueue.hpp>
#    include <lpl/engine/PacketDispatch.hpp>
#    include <lpl/engine/systems/BroadcastSystem.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/net/Endpoint.hpp>
#    include <lpl/net/protocol/Protocol.hpp>
#    include <lpl/net/session/SessionManager.hpp>

#    include <algorithm>
#    include <cstdio>
#    include <memory>
#    include <set>
#    include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

/**
 * @brief Capturing transport: records, per destination, the entity ids each datagram carried and what packet
 *        type delivered them. sendBatch's default loops over send(), so overriding send() captures every
 *        fragment.
 */
class CapturingTransport final : public lpl::net::transport::ITransport {
public:
    struct Packet {
        lpl::net::Endpoint dest;
        lpl::net::protocol::PacketType type;
        lpl::core::u8 flags;
        std::vector<lpl::core::u32> ids;
    };
    std::vector<Packet> packets;

    lpl::core::Expected<void> open() override { return {}; }
    void close() override {}
    const char *name() const noexcept override { return "CapturingTransport"; }
    lpl::core::Expected<lpl::core::u32> receive(std::span<lpl::core::byte>, lpl::net::Endpoint *) override
    {
        return lpl::core::u32{0};
    }

    lpl::core::Expected<lpl::core::u32> send(std::span<const lpl::core::byte> data,
                                             const lpl::net::Endpoint *address) override
    {
        lpl::net::protocol::PacketHeader header{};
        std::span<const lpl::core::byte> payload;
        if (lpl::engine::detail::parsePacket(data, header, payload))
        {
            lpl::engine::EventQueues q;
            lpl::net::Endpoint src{};
            lpl::engine::detail::dispatchPacket(header, payload, src, q);

            Packet p{};
            p.dest = address ? *address : lpl::net::Endpoint{};
            p.type = header.type;
            p.flags = header.flags;
            for (const auto &ev : q.states.drain())
                for (const auto &e : ev.entities)
                    p.ids.push_back(e.id);
            for (const auto &ev : q.spawns.drain())
                for (const auto &e : ev.entities)
                    p.ids.push_back(e.id);
            for (const auto &ev : q.deltas.drain())
                for (const auto &e : ev.entities)
                    p.ids.push_back(e.id);
            for (const auto &ev : q.destroys.drain())
                for (const auto id : ev.ids)
                    p.ids.push_back(id);
            packets.push_back(std::move(p));
        }
        return static_cast<lpl::core::u32>(data.size());
    }

    void clear() { packets.clear(); }

    [[nodiscard]] std::set<lpl::core::u32> idsFor(const lpl::net::Endpoint &dest,
                                                  lpl::net::protocol::PacketType type) const
    {
        std::set<lpl::core::u32> out;
        for (const auto &p : packets)
            if (p.dest == dest && p.type == type)
                for (const auto id : p.ids)
                    out.insert(id);
        return out;
    }

    [[nodiscard]] std::set<lpl::core::u32> allIdsFor(const lpl::net::Endpoint &dest) const
    {
        std::set<lpl::core::u32> out;
        for (const auto &p : packets)
            if (p.dest == dest)
                for (const auto id : p.ids)
                    out.insert(id);
        return out;
    }

    /**
     * @brief Ids delivered to @p dest in a StateDelta whose Compressed flag matches @p compressed.
     */
    [[nodiscard]] std::set<lpl::core::u32> deltaIdsByCompression(const lpl::net::Endpoint &dest, bool compressed) const
    {
        std::set<lpl::core::u32> out;
        for (const auto &p : packets)
        {
            if (p.dest != dest || p.type != lpl::net::protocol::PacketType::StateDelta)
                continue;
            const bool isCompressed = lpl::net::protocol::hasFlag(p.flags, lpl::net::protocol::PacketFlag::Compressed);
            if (isCompressed == compressed)
                for (const auto id : p.ids)
                    out.insert(id);
        }
        return out;
    }

    [[nodiscard]] lpl::core::usize totalIdsSent() const
    {
        lpl::core::usize n = 0;
        for (const auto &p : packets)
            n += p.ids.size();
        return n;
    }
};

const lpl::ecs::ComponentId kIds[] = {lpl::ecs::ComponentId::Position, lpl::ecs::ComponentId::AABB,
                                      lpl::ecs::ComponentId::Health};
/** @brief A single shared archetype so every entity lands in one partition. */
const lpl::ecs::Archetype kArch{kIds};

[[nodiscard]] lpl::ecs::EntityId spawnEntity(lpl::ecs::Registry &registry)
{
    auto res = registry.createEntity(kArch);
    return res.has_value() ? res.value() : lpl::ecs::EntityId{};
}

/**
 * @brief Writes an entity's authoritative transform into BOTH SoA buffers (the broadcast reads the
 *        read-buffer) and updates the spatial index — what a real tick's SessionSystem + PhysicsSystem would
 *        leave behind.
 */
void placeEntity(lpl::ecs::Registry &registry, lpl::ecs::WorldPartition &world, lpl::ecs::EntityId id, float x, float y,
                 float z)
{
    using Vec = lpl::math::Vec3<lpl::math::Fixed32>;
    const Vec pos{lpl::math::Fixed32::fromFloat(x), lpl::math::Fixed32::fromFloat(y), lpl::math::Fixed32::fromFloat(z)};
    const Vec size{lpl::math::Fixed32::one(), lpl::math::Fixed32::one(), lpl::math::Fixed32::one()};

    auto ref = registry.resolve(id);
    if (ref.has_value())
    {
        auto &partition = registry.getOrCreatePartition(kArch);
        const auto &chunks = partition.chunks();
        if (ref.value().chunkIndex < static_cast<lpl::core::u32>(chunks.size()))
        {
            auto &chunk = *chunks[ref.value().chunkIndex];
            const lpl::core::u32 i = ref.value().localIndex;

            if (auto *w = static_cast<Vec *>(chunk.writeComponent(lpl::ecs::ComponentId::Position)))
                w[i] = pos;
            if (auto *r =
                    const_cast<Vec *>(static_cast<const Vec *>(chunk.readComponent(lpl::ecs::ComponentId::Position))))
                r[i] = pos;
            if (auto *w = static_cast<Vec *>(chunk.writeComponent(lpl::ecs::ComponentId::AABB)))
                w[i] = size;
            if (auto *r = const_cast<Vec *>(static_cast<const Vec *>(chunk.readComponent(lpl::ecs::ComponentId::AABB))))
                r[i] = size;
            if (auto *w = static_cast<lpl::core::i32 *>(chunk.writeComponent(lpl::ecs::ComponentId::Health)))
                w[i] = 100;
            if (auto *r = const_cast<lpl::core::i32 *>(
                    static_cast<const lpl::core::i32 *>(chunk.readComponent(lpl::ecs::ComponentId::Health))))
                r[i] = 100;
        }
    }

    [[maybe_unused]] auto res = world.insertOrUpdate(id, pos);
}

[[nodiscard]] lpl::net::session::Session *joinClient(lpl::net::session::SessionManager &sessions,
                                                     lpl::core::u32 playerId, const lpl::net::Endpoint &addr,
                                                     lpl::ecs::EntityId avatar)
{
    auto joined = sessions.connect(playerId);
    if (!joined.has_value())
        return nullptr;
    auto *session = joined.value();
    session->setAddress(addr);
    session->bindEntity(avatar);
    return session;
}

/**
 * @brief Tick 1: everything a client sees for the first time is a spawn.
 *
 * @param aoi       The AOI system of the two-cluster world.
 * @param transport The transport it sends through.
 * @param alice     Client A's address.
 * @param bob       Client B's address.
 * @param eAvatarA  A's avatar.
 * @param eNearA    The entity near A.
 * @param eAvatarB  B's avatar.
 * @param eNearB    The entity near B.
 * @param eRoamer   The entity far from everyone.
 */
void checkFirstSightIsASpawn(lpl::engine::systems::AoiBroadcastSystem &aoi,
                             const std::shared_ptr<CapturingTransport> &transport, const lpl::net::Endpoint &alice,
                             const lpl::net::Endpoint &bob, lpl::ecs::EntityId eAvatarA, lpl::ecs::EntityId eNearA,
                             lpl::ecs::EntityId eAvatarB, lpl::ecs::EntityId eNearB, lpl::ecs::EntityId eRoamer)
{
    transport->clear();
    aoi.execute(1.0f / 60.0f);

    const auto aliceSpawn = transport->idsFor(alice, lpl::net::protocol::PacketType::EntitySpawn);
    const auto bobSpawn = transport->idsFor(bob, lpl::net::protocol::PacketType::EntitySpawn);

    check(aliceSpawn.count(eAvatarA.raw()) && aliceSpawn.count(eNearA.raw()),
          "A's first packet spawns its own cluster");
    check(!aliceSpawn.count(eAvatarB.raw()) && !aliceSpawn.count(eNearB.raw()), "A never receives B's distant cluster");
    check(bobSpawn.count(eAvatarB.raw()) && bobSpawn.count(eNearB.raw()), "B's first packet spawns its own cluster");
    check(!bobSpawn.count(eAvatarA.raw()) && !bobSpawn.count(eNearA.raw()), "B never receives A's distant cluster");

    check(transport->idsFor(alice, lpl::net::protocol::PacketType::StateDelta).empty(), "nothing is a delta on tick 1");
    check(transport->idsFor(alice, lpl::net::protocol::PacketType::EntityDestroy).empty(),
          "nothing is destroyed on tick 1");
    check(!aliceSpawn.count(eRoamer.raw()), "the far roamer is outside A's radius");
}

/**
 * @brief Tick 2: dormancy — an unchanged entity sends nothing, a moved one deltas.
 *
 * @details eNearA slides, and stays in range.
 *
 * @param registry  The two-cluster world's registry.
 * @param world     Its spatial index.
 * @param aoi       Its AOI system.
 * @param transport The transport the AOI system sends through.
 * @param alice     Client A's address.
 * @param eAvatarA  A's avatar.
 * @param eNearA    The entity near A.
 */
void checkUnchangedEntityIsDormantAndMovedOneDeltas(lpl::ecs::Registry &registry, lpl::ecs::WorldPartition &world,
                                                    lpl::engine::systems::AoiBroadcastSystem &aoi,
                                                    const std::shared_ptr<CapturingTransport> &transport,
                                                    const lpl::net::Endpoint &alice, lpl::ecs::EntityId eAvatarA,
                                                    lpl::ecs::EntityId eNearA)
{
    placeEntity(registry, world, eNearA, 6.0f, 0.0f, 0.0f);
    transport->clear();
    aoi.execute(1.0f / 60.0f);

    const auto aliceSpawn = transport->idsFor(alice, lpl::net::protocol::PacketType::EntitySpawn);
    const auto aliceDelta = transport->idsFor(alice, lpl::net::protocol::PacketType::StateDelta);

    check(aliceSpawn.empty(), "an already-known entity is not spawned again");
    check(aliceDelta.count(eNearA.raw()), "a known entity that moved is sent as a delta");
    check(!aliceDelta.count(eAvatarA.raw()), "an unchanged entity is dormant — no traffic (§6.2.7)");
    check(transport->idsFor(alice, lpl::net::protocol::PacketType::EntityDestroy).empty(),
          "nothing left A's radius, so nothing is destroyed");
}

/**
 * @brief Tick 3: one entity leaves the radius, one enters, the avatar moves.
 *
 * @details eNearA leaves A's radius, eRoamer enters it, and the avatar itself slides.
 *
 * @param registry  The two-cluster world's registry.
 * @param world     Its spatial index.
 * @param aoi       Its AOI system.
 * @param transport The transport the AOI system sends through.
 * @param alice     Client A's address.
 * @param eAvatarA  A's avatar.
 * @param eNearA    The entity near A.
 * @param eRoamer   The entity far from everyone.
 */
void checkLeavingIsADestroyEnteringASpawnMovingADelta(lpl::ecs::Registry &registry, lpl::ecs::WorldPartition &world,
                                                      lpl::engine::systems::AoiBroadcastSystem &aoi,
                                                      const std::shared_ptr<CapturingTransport> &transport,
                                                      const lpl::net::Endpoint &alice, lpl::ecs::EntityId eAvatarA,
                                                      lpl::ecs::EntityId eNearA, lpl::ecs::EntityId eRoamer)
{
    placeEntity(registry, world, eNearA, 0.0f, 5000.0f, 0.0f);
    placeEntity(registry, world, eRoamer, 10.0f, 0.0f, 0.0f);
    placeEntity(registry, world, eAvatarA, 1.0f, 0.0f, 0.0f);

    transport->clear();
    aoi.execute(1.0f / 60.0f);

    const auto aliceSpawn = transport->idsFor(alice, lpl::net::protocol::PacketType::EntitySpawn);
    const auto aliceDelta = transport->idsFor(alice, lpl::net::protocol::PacketType::StateDelta);
    const auto aliceDestroy = transport->idsFor(alice, lpl::net::protocol::PacketType::EntityDestroy);

    check(aliceSpawn.count(eRoamer.raw()), "an entity entering the radius is spawned");
    check(aliceDestroy.count(eNearA.raw()), "an entity leaving the radius is destroyed");
    check(aliceDelta.count(eAvatarA.raw()), "an entity that moved and stayed in range is a delta");
    check(!aliceDelta.count(eNearA.raw()), "the departed entity is not also sent as a delta");
    check(!aliceSpawn.count(eAvatarA.raw()), "the still-present avatar is not re-spawned");
}

/**
 * @brief Purge: a disconnected client's memory is dropped, not leaked.
 *
 * @details Re-joining B on the same avatar must spawn its cluster afresh, proving the known-set was reaped (a
 *          lingering set would send deltas instead).
 *
 * @param sessions  The two-cluster world's sessions.
 * @param aoi       Its AOI system.
 * @param transport The transport the AOI system sends through.
 * @param bob       Client B's address.
 * @param eAvatarB  B's avatar.
 */
void checkDisconnectedClientMemoryIsPurged(lpl::net::session::SessionManager &sessions,
                                           lpl::engine::systems::AoiBroadcastSystem &aoi,
                                           const std::shared_ptr<CapturingTransport> &transport,
                                           const lpl::net::Endpoint &bob, lpl::ecs::EntityId eAvatarB)
{
    [[maybe_unused]] auto d = sessions.disconnect(2);
    transport->clear();
    aoi.execute(1.0f / 60.0f);
    check(transport->allIdsFor(bob).empty(), "a disconnected client receives nothing");
    check(joinClient(sessions, 2, bob, eAvatarB) != nullptr, "client B re-joins");
    transport->clear();
    aoi.execute(1.0f / 60.0f);
    check(!transport->idsFor(bob, lpl::net::protocol::PacketType::EntitySpawn).empty(),
          "a re-joined client is spawned its cluster afresh (known-set was reaped)");
}

/**
 * @brief Non-regression + the whole point: AOI ≪ full broadcast.
 *
 * @details A fresh, identical dispersed world driven once by each system. The full broadcast (the
 *          interestRadius==0 fallback) sends every entity to every client; AOI sends each client only its own
 *          cluster.
 *
 * @param cellSize The spatial index's cell size.
 * @param radius   The interest radius.
 * @param alice    Client A's address.
 * @param bob      Client B's address.
 */
void checkAoiSendsLessThanTheFullBroadcast(lpl::math::Fixed32 cellSize, lpl::math::Fixed32 radius,
                                           const lpl::net::Endpoint &alice, const lpl::net::Endpoint &bob)
{
    lpl::ecs::Registry reg2;
    lpl::ecs::WorldPartition world2{cellSize, 4096};
    lpl::net::session::SessionManager sm2;

    const auto a0 = spawnEntity(reg2);
    const auto a1 = spawnEntity(reg2);
    const auto b0 = spawnEntity(reg2);
    const auto b1 = spawnEntity(reg2);
    placeEntity(reg2, world2, a0, 0.0f, 0.0f, 0.0f);
    placeEntity(reg2, world2, a1, 5.0f, 0.0f, 0.0f);
    placeEntity(reg2, world2, b0, 1000.0f, 0.0f, 0.0f);
    placeEntity(reg2, world2, b1, 1005.0f, 0.0f, 0.0f);
    check(joinClient(sm2, 1, alice, a0) != nullptr && joinClient(sm2, 2, bob, b0) != nullptr,
          "two dispersed clients join the comparison world");

    auto fullTransport = std::make_shared<CapturingTransport>();
    lpl::engine::systems::BroadcastSystem full{sm2, fullTransport, world2, reg2};
    full.execute(1.0f / 60.0f);

    const auto aliceFull = fullTransport->allIdsFor(alice);
    check(aliceFull.count(a0.raw()) && aliceFull.count(a1.raw()) && aliceFull.count(b0.raw()) &&
              aliceFull.count(b1.raw()),
          "the full-broadcast fallback sends every entity to every client");
    const lpl::core::usize fullTotal = fullTransport->totalIdsSent();

    auto aoiTransport = std::make_shared<CapturingTransport>();
    lpl::engine::systems::AoiBroadcastSystem aoi2{sm2, aoiTransport, world2, reg2, radius};
    aoi2.execute(1.0f / 60.0f);

    const auto aliceAoi = aoiTransport->allIdsFor(alice);
    check(!aliceAoi.count(b0.raw()) && !aliceAoi.count(b1.raw()), "AOI keeps the far cluster out of A's stream");
    const lpl::core::usize aoiTotal = aoiTransport->totalIdsSent();

    std::printf("  full broadcast sent %zu entity-records, AOI sent %zu\n", static_cast<size_t>(fullTotal),
                static_cast<size_t>(aoiTotal));
    check(aoiTotal < fullTotal, "AOI serialises strictly fewer entity-records than the full broadcast");
}

/**
 * @brief Bandwidth budget + priority + anti-starvation (§6.2.7).
 *
 * @details One client, six movers all in range at increasing distance (5, 10, 15, 20, 25, 30). A byte budget
 *          that fits ~two deltas per tick forces the server to choose: it must send the closest first, yet still
 *          serve every entity within a few ticks (nothing starves, because a skipped entity's staleness rises
 *          until it wins). Each single-axis delta is 9 bytes (id + mask + one float); a 20-byte budget therefore
 *          admits exactly two per tick. Keyframes are 1000 ticks apart.
 *
 *          The first tick spawns everything (spawns are not budgeted — a client must learn who is there), which
 *          establishes the baseline for the delta stream. Then, for six ticks, enough to serve all six, every
 *          mover is nudged so all are due, and the union of the delta ids is collected.
 *
 * @param cellSize The spatial index's cell size.
 * @param alice    The client's address.
 */
void checkBudgetSendsTheClosestFirstAndStarvesNone(lpl::math::Fixed32 cellSize, const lpl::net::Endpoint &alice)
{
    lpl::ecs::Registry reg3;
    lpl::ecs::WorldPartition world3{cellSize, 4096};
    lpl::net::session::SessionManager sm3;

    const auto avatar = spawnEntity(reg3);
    placeEntity(reg3, world3, avatar, 0.0f, 0.0f, 0.0f);

    constexpr int kMovers = 6;
    lpl::ecs::EntityId movers[kMovers];
    float moverX[kMovers];
    for (int i = 0; i < kMovers; ++i)
    {
        movers[i] = spawnEntity(reg3);
        moverX[i] = 5.0f * static_cast<float>(i + 1);
        placeEntity(reg3, world3, movers[i], moverX[i], 0.0f, 0.0f);
    }
    check(joinClient(sm3, 1, alice, avatar) != nullptr, "the budget client joins");

    auto budgetTransport = std::make_shared<CapturingTransport>();
    const lpl::core::u32 kBudget = 20;
    lpl::engine::systems::AoiBroadcastSystem budgeted{
        sm3, budgetTransport, world3, reg3, lpl::math::Fixed32::fromFloat(100.0f), 1000, kBudget};

    budgeted.execute(1.0f / 60.0f);

    std::set<lpl::core::u32> served;
    int perTickMax = 0;
    std::set<lpl::core::u32> firstDeltaTick;
    for (int tick = 0; tick < kMovers; ++tick)
    {
        for (int i = 0; i < kMovers; ++i)
        {
            moverX[i] += 0.5f;
            placeEntity(reg3, world3, movers[i], moverX[i], 0.0f, 0.0f);
        }
        budgetTransport->clear();
        budgeted.execute(1.0f / 60.0f);

        const auto delta = budgetTransport->idsFor(alice, lpl::net::protocol::PacketType::StateDelta);
        perTickMax = std::max(perTickMax, static_cast<int>(delta.size()));
        if (tick == 0)
            firstDeltaTick = delta;
        for (const auto id : delta)
            served.insert(id);
    }

    check(perTickMax <= 2, "the byte budget caps the delta stream (~2 entities per tick)");
    check(firstDeltaTick.count(movers[0].raw()) && firstDeltaTick.count(movers[1].raw()),
          "the closest entities are sent first");
    bool allServed = true;
    for (int i = 0; i < kMovers; ++i)
        if (!served.count(movers[i].raw()))
            allServed = false;
    check(allServed, "every entity is served within a few ticks — nothing starves");
}

/**
 * @brief Network LOD: a far entity updates on a slower cadence (§6.2.6).
 *
 * @details One client at the origin, a near entity and a far one, both moving every tick. With a full-rate near
 *          ring and a 4-tick far ring, the near entity is sent every tick and the far one only every fourth —
 *          measured over 8 ticks. The near entity stands inside the near ring, the far one between the near
 *          ring and the interest radius; keyframes are 1000 ticks apart. The first tick spawns both and sets the
 *          baseline.
 *
 * @param cellSize The spatial index's cell size.
 * @param alice    The client's address.
 */
void checkFarEntityUpdatesOnASlowerCadence(lpl::math::Fixed32 cellSize, const lpl::net::Endpoint &alice)
{
    lpl::ecs::Registry reg4;
    lpl::ecs::WorldPartition world4{cellSize, 4096};
    lpl::net::session::SessionManager sm4;

    const auto avatar = spawnEntity(reg4);
    const auto nearE = spawnEntity(reg4);
    const auto farE = spawnEntity(reg4);
    placeEntity(reg4, world4, avatar, 0.0f, 0.0f, 0.0f);
    placeEntity(reg4, world4, nearE, 5.0f, 0.0f, 0.0f);
    placeEntity(reg4, world4, farE, 60.0f, 0.0f, 0.0f);
    check(joinClient(sm4, 1, alice, avatar) != nullptr, "the LOD client joins");

    auto lodTransport = std::make_shared<CapturingTransport>();
    lpl::engine::systems::AoiBroadcastSystem lod{sm4, lodTransport, world4, reg4, lpl::math::Fixed32::fromFloat(100.0f),
                                                 1000};
    constexpr lpl::core::u32 kFarInterval = 4;
    lod.setNetworkLod(lpl::math::Fixed32::fromFloat(20.0f), kFarInterval);

    lod.execute(1.0f / 60.0f);

    float nearX = 5.0f, farX = 60.0f;
    int nearSends = 0, farSends = 0;
    constexpr int kTicks = 8;
    for (int t = 0; t < kTicks; ++t)
    {
        nearX += 0.5f;
        farX += 0.5f;
        placeEntity(reg4, world4, nearE, nearX, 0.0f, 0.0f);
        placeEntity(reg4, world4, farE, farX, 0.0f, 0.0f);
        lodTransport->clear();
        lod.execute(1.0f / 60.0f);
        const auto delta = lodTransport->idsFor(alice, lpl::net::protocol::PacketType::StateDelta);
        if (delta.count(nearE.raw()))
            ++nearSends;
        if (delta.count(farE.raw()))
            ++farSends;
    }

    std::printf("  over %d ticks: near sent %d times, far sent %d times\n", kTicks, nearSends, farSends);
    check(nearSends == kTicks, "the near entity updates every tick (full rate)");
    check(farSends == kTicks / static_cast<int>(kFarInterval),
          "the far entity updates once per far-interval (fewer packets, LOD)");
    check(farSends < nearSends, "network LOD sends the far entity strictly less often than the near one");
}

/**
 * @brief Strict acked-baseline: resend until confirmed, then stop (§6.2.5).
 *
 * @details With reliableBaseline on, the delta baseline advances only on the client's ack. An unacked change is
 *          resent every tick (reliability); once acked, the entity goes dormant. Driven by feeding the session
 *          ack sequences directly, with keyframes 100000 ticks apart:
 *           - tick 1: the entity enters → spawn at sequence 1. Client acks it;
 *           - tick 2: nothing moved and the spawn is acked → dormant;
 *           - the entity moves. Tick 3 sends the delta (sequence 3), NOT yet acked;
 *           - tick 4: still not acked and unchanged since → the change is RESENT;
 *           - the client finally acks up to the last sent sequence (tick 4). Tick 5: the change is confirmed,
 *             the baseline advances, dormant again.
 *
 * @param cellSize The spatial index's cell size.
 * @param alice    The client's address.
 */
void checkChangeIsResentUntilAcked(lpl::math::Fixed32 cellSize, const lpl::net::Endpoint &alice)
{
    lpl::ecs::Registry reg5;
    lpl::ecs::WorldPartition world5{cellSize, 4096};
    lpl::net::session::SessionManager sm5;

    const auto avatar = spawnEntity(reg5);
    const auto ent = spawnEntity(reg5);
    placeEntity(reg5, world5, avatar, 0.0f, 0.0f, 0.0f);
    placeEntity(reg5, world5, ent, 5.0f, 0.0f, 0.0f);
    auto *session = joinClient(sm5, 1, alice, avatar);
    check(session != nullptr, "the acked-baseline client joins");

    auto tr = std::make_shared<CapturingTransport>();
    lpl::engine::systems::AoiBroadcastSystem strict{sm5,   tr, world5, reg5, lpl::math::Fixed32::fromFloat(100.0f),
                                                    100000};
    strict.setReliableBaseline(true);

    auto entInDelta = [&]() {
        return tr->idsFor(alice, lpl::net::protocol::PacketType::StateDelta).count(ent.raw()) > 0;
    };

    strict.execute(1.0f / 60.0f);
    session->setAckedSnapshotSeq(1);

    tr->clear();
    strict.execute(1.0f / 60.0f);
    check(!entInDelta(), "after the spawn is acked, an unchanged entity is dormant");

    placeEntity(reg5, world5, ent, 6.0f, 0.0f, 0.0f);
    tr->clear();
    strict.execute(1.0f / 60.0f);
    check(entInDelta(), "a change is sent as a delta");

    tr->clear();
    strict.execute(1.0f / 60.0f);
    check(entInDelta(), "an unacknowledged change is resent (reliability, not lost)");

    session->setAckedSnapshotSeq(4);
    tr->clear();
    strict.execute(1.0f / 60.0f);
    check(!entInDelta(), "once the change is acked, it stops being resent");
}

/**
 * @brief Precision LOD: far entities are quantized, near ones are not (§6.2.6).
 *
 * @details The near entity stands inside the near ring, the far one in the far ring, which is updated every tick
 *          so it can be observed; keyframes are 100000 ticks apart. Both are spawned, then both move so each is a
 *          delta, and how each was encoded is observed.
 *
 * @param cellSize The spatial index's cell size.
 * @param alice    The client's address.
 */
void checkFarEntitiesAreQuantizedNearOnesAreNot(lpl::math::Fixed32 cellSize, const lpl::net::Endpoint &alice)
{
    lpl::ecs::Registry reg6;
    lpl::ecs::WorldPartition world6{cellSize, 4096};
    lpl::net::session::SessionManager sm6;

    const auto avatar = spawnEntity(reg6);
    const auto nearE = spawnEntity(reg6);
    const auto farE = spawnEntity(reg6);
    placeEntity(reg6, world6, avatar, 0.0f, 0.0f, 0.0f);
    placeEntity(reg6, world6, nearE, 5.0f, 0.0f, 0.0f);
    placeEntity(reg6, world6, farE, 60.0f, 0.0f, 0.0f);
    check(joinClient(sm6, 1, alice, avatar) != nullptr, "the precision-LOD client joins");

    auto tr = std::make_shared<CapturingTransport>();
    lpl::engine::systems::AoiBroadcastSystem lod{sm6, tr, world6, reg6, lpl::math::Fixed32::fromFloat(100.0f), 100000};
    lod.setNetworkLod(lpl::math::Fixed32::fromFloat(20.0f), 1);
    lod.setPrecisionLod(lpl::math::Fixed32::fromFloat(1024.0f), 16);

    lod.execute(1.0f / 60.0f);

    placeEntity(reg6, world6, nearE, 6.0f, 0.0f, 0.0f);
    placeEntity(reg6, world6, farE, 61.0f, 0.0f, 0.0f);
    tr->clear();
    lod.execute(1.0f / 60.0f);

    const auto full = tr->deltaIdsByCompression(alice, false);
    const auto quant = tr->deltaIdsByCompression(alice, true);
    check(full.count(nearE.raw()), "the near entity's delta is full precision (uncompressed)");
    check(quant.count(farE.raw()), "the far entity's delta is quantized (compressed)");
    check(!quant.count(nearE.raw()) && !full.count(farE.raw()),
          "near and far are not encoded the same way — precision falls off with distance");
}

} // namespace

/**
 * @brief Area-of-interest (AOI) broadcast test.
 *
 * @details Proves systems::AoiBroadcastSystem sends each client only the entities within its interest radius,
 *          as a per-session delta (spawn on enter, destroy on leave, delta while it stays), and that the
 *          zero-radius fallback (BroadcastSystem) is the full O(clients × N) broadcast. Everything is MEASURED
 *          per destination: a capturing transport records which entity ids each client's datagrams carried, the
 *          way test-transport-batching counts syscalls — a gain unmeasured is a claim, not a result (book §1.5).
 *
 *          The first checks run on two clusters ~1000 units apart, one client centred on each, and a roamer that
 *          starts far from everyone.
 */
int main()
{
    std::printf("== area-of-interest broadcast ==\n");

    const auto radius = lpl::math::Fixed32::fromFloat(50.0f);
    const auto cellSize = lpl::math::Fixed32::fromFloat(10.0f);

    const auto alice = lpl::net::Endpoint::fromOctets(127, 0, 0, 1, 40001);
    const auto bob = lpl::net::Endpoint::fromOctets(127, 0, 0, 1, 40002);

    lpl::ecs::Registry registry;
    lpl::ecs::WorldPartition world{cellSize, 4096};
    lpl::net::session::SessionManager sessions;

    const auto eAvatarA = spawnEntity(registry);
    const auto eNearA = spawnEntity(registry);
    const auto eAvatarB = spawnEntity(registry);
    const auto eNearB = spawnEntity(registry);
    const auto eRoamer = spawnEntity(registry);

    placeEntity(registry, world, eAvatarA, 0.0f, 0.0f, 0.0f);
    placeEntity(registry, world, eNearA, 5.0f, 0.0f, 0.0f);
    placeEntity(registry, world, eAvatarB, 1000.0f, 0.0f, 0.0f);
    placeEntity(registry, world, eNearB, 1005.0f, 0.0f, 0.0f);
    placeEntity(registry, world, eRoamer, 0.0f, 5000.0f, 0.0f);

    check(joinClient(sessions, 1, alice, eAvatarA) != nullptr, "client A joins, bound to its avatar");
    check(joinClient(sessions, 2, bob, eAvatarB) != nullptr, "client B joins, bound to its avatar");

    auto transport = std::make_shared<CapturingTransport>();
    lpl::engine::systems::AoiBroadcastSystem aoi{sessions, transport, world, registry, radius};

    checkFirstSightIsASpawn(aoi, transport, alice, bob, eAvatarA, eNearA, eAvatarB, eNearB, eRoamer);
    checkUnchangedEntityIsDormantAndMovedOneDeltas(registry, world, aoi, transport, alice, eAvatarA, eNearA);
    checkLeavingIsADestroyEnteringASpawnMovingADelta(registry, world, aoi, transport, alice, eAvatarA, eNearA, eRoamer);
    checkDisconnectedClientMemoryIsPurged(sessions, aoi, transport, bob, eAvatarB);
    checkAoiSendsLessThanTheFullBroadcast(cellSize, radius, alice, bob);
    checkBudgetSendsTheClosestFirstAndStarvesNone(cellSize, alice);
    checkFarEntityUpdatesOnASlowerCadence(cellSize, alice);
    checkChangeIsResentUntilAcked(cellSize, alice);
    checkFarEntitiesAreQuantizedNearOnesAreNot(cellSize, alice);

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#else

#    include <cstdio>

int main()
{
    std::printf("aoi test skipped: built without LPL_HAS_NET\n");
    return 0;
}

#endif
