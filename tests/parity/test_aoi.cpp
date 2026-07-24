/*
** LplPlugin — area-of-interest (AOI) broadcast test
**
** Proves systems::AoiBroadcastSystem sends each client only the entities within
** its interest radius, as a per-session delta (spawn on enter, destroy on leave,
** delta while it stays), and that the zero-radius fallback (BroadcastSystem) is
** the full O(clients × N) broadcast. Everything is MEASURED per destination: a
** capturing transport records which entity ids each client's datagrams carried,
** the way test-transport-batching counts syscalls — a gain unmeasured is a claim,
** not a result (book §1.5).
*/

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

using namespace lpl;

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++g_failures;
}

// ─────────────────────────────────────────────────────────────────────────────
// Capturing transport: records, per destination, the entity ids each datagram
// carried and what packet type delivered them. sendBatch's default loops over
// send(), so overriding send() captures every fragment.
// ─────────────────────────────────────────────────────────────────────────────
class CapturingTransport final : public net::transport::ITransport {
public:
    struct Packet {
        net::Endpoint dest;
        net::protocol::PacketType type;
        std::vector<core::u32> ids;
    };
    std::vector<Packet> packets;

    core::Expected<void> open() override { return {}; }
    void close() override {}
    const char *name() const noexcept override { return "CapturingTransport"; }
    core::Expected<core::u32> receive(std::span<core::byte>, net::Endpoint *) override { return core::u32{0}; }

    core::Expected<core::u32> send(std::span<const core::byte> data, const net::Endpoint *address) override
    {
        net::protocol::PacketHeader header{};
        std::span<const core::byte> payload;
        if (engine::detail::parsePacket(data, header, payload))
        {
            engine::EventQueues q;
            net::Endpoint src{};
            engine::detail::dispatchPacket(header, payload, src, q);

            Packet p{};
            p.dest = address ? *address : net::Endpoint{};
            p.type = header.type;
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
        return static_cast<core::u32>(data.size());
    }

    void clear() { packets.clear(); }

    [[nodiscard]] std::set<core::u32> idsFor(const net::Endpoint &dest, net::protocol::PacketType type) const
    {
        std::set<core::u32> out;
        for (const auto &p : packets)
            if (p.dest == dest && p.type == type)
                for (const auto id : p.ids)
                    out.insert(id);
        return out;
    }

    [[nodiscard]] std::set<core::u32> allIdsFor(const net::Endpoint &dest) const
    {
        std::set<core::u32> out;
        for (const auto &p : packets)
            if (p.dest == dest)
                for (const auto id : p.ids)
                    out.insert(id);
        return out;
    }

    [[nodiscard]] core::usize totalIdsSent() const
    {
        core::usize n = 0;
        for (const auto &p : packets)
            n += p.ids.size();
        return n;
    }
};

// A single shared archetype so every entity lands in one partition.
const ecs::ComponentId kIds[] = {ecs::ComponentId::Position, ecs::ComponentId::AABB, ecs::ComponentId::Health};
const ecs::Archetype kArch{kIds};

[[nodiscard]] ecs::EntityId spawnEntity(ecs::Registry &registry)
{
    auto res = registry.createEntity(kArch);
    return res.has_value() ? res.value() : ecs::EntityId{};
}

// Writes an entity's authoritative transform into BOTH SoA buffers (the broadcast
// reads the read-buffer) and updates the spatial index — what a real tick's
// SessionSystem + PhysicsSystem would leave behind.
void placeEntity(ecs::Registry &registry, ecs::WorldPartition &world, ecs::EntityId id, float x, float y, float z)
{
    using Vec = math::Vec3<math::Fixed32>;
    const Vec pos{math::Fixed32::fromFloat(x), math::Fixed32::fromFloat(y), math::Fixed32::fromFloat(z)};
    const Vec size{math::Fixed32::one(), math::Fixed32::one(), math::Fixed32::one()};

    auto ref = registry.resolve(id);
    if (ref.has_value())
    {
        auto &partition = registry.getOrCreatePartition(kArch);
        const auto &chunks = partition.chunks();
        if (ref.value().chunkIndex < static_cast<core::u32>(chunks.size()))
        {
            auto &chunk = *chunks[ref.value().chunkIndex];
            const core::u32 i = ref.value().localIndex;

            if (auto *w = static_cast<Vec *>(chunk.writeComponent(ecs::ComponentId::Position)))
                w[i] = pos;
            if (auto *r = const_cast<Vec *>(static_cast<const Vec *>(chunk.readComponent(ecs::ComponentId::Position))))
                r[i] = pos;
            if (auto *w = static_cast<Vec *>(chunk.writeComponent(ecs::ComponentId::AABB)))
                w[i] = size;
            if (auto *r = const_cast<Vec *>(static_cast<const Vec *>(chunk.readComponent(ecs::ComponentId::AABB))))
                r[i] = size;
            if (auto *w = static_cast<core::i32 *>(chunk.writeComponent(ecs::ComponentId::Health)))
                w[i] = 100;
            if (auto *r =
                    const_cast<core::i32 *>(static_cast<const core::i32 *>(chunk.readComponent(ecs::ComponentId::Health))))
                r[i] = 100;
        }
    }

    [[maybe_unused]] auto res = world.insertOrUpdate(id, pos);
}

[[nodiscard]] net::session::Session *joinClient(net::session::SessionManager &sessions, core::u32 playerId,
                                                const net::Endpoint &addr, ecs::EntityId avatar)
{
    auto joined = sessions.connect(playerId);
    if (!joined.has_value())
        return nullptr;
    auto *session = joined.value();
    session->setAddress(addr);
    session->bindEntity(avatar);
    return session;
}

} // namespace

int main()
{
    std::printf("== area-of-interest broadcast ==\n");

    const auto radius = math::Fixed32::fromFloat(50.0f);
    const auto cellSize = math::Fixed32::fromFloat(10.0f);

    const auto alice = net::Endpoint::fromOctets(127, 0, 0, 1, 40001);
    const auto bob = net::Endpoint::fromOctets(127, 0, 0, 1, 40002);

    // Two clusters ~1000 units apart, one client centred on each.
    ecs::Registry registry;
    ecs::WorldPartition world{cellSize, 4096};
    net::session::SessionManager sessions;

    const auto eAvatarA = spawnEntity(registry);
    const auto eNearA = spawnEntity(registry);
    const auto eAvatarB = spawnEntity(registry);
    const auto eNearB = spawnEntity(registry);
    const auto eRoamer = spawnEntity(registry); // starts far from everyone

    placeEntity(registry, world, eAvatarA, 0.0f, 0.0f, 0.0f);
    placeEntity(registry, world, eNearA, 5.0f, 0.0f, 0.0f);
    placeEntity(registry, world, eAvatarB, 1000.0f, 0.0f, 0.0f);
    placeEntity(registry, world, eNearB, 1005.0f, 0.0f, 0.0f);
    placeEntity(registry, world, eRoamer, 0.0f, 5000.0f, 0.0f);

    check(joinClient(sessions, 1, alice, eAvatarA) != nullptr, "client A joins, bound to its avatar");
    check(joinClient(sessions, 2, bob, eAvatarB) != nullptr, "client B joins, bound to its avatar");

    auto transport = std::make_shared<CapturingTransport>();
    engine::systems::AoiBroadcastSystem aoi{sessions, transport, world, registry, radius};

    // ── Tick 1: everything a client sees for the first time is a spawn ───────── //
    transport->clear();
    aoi.execute(1.0f / 60.0f);

    {
        const auto aliceSpawn = transport->idsFor(alice, net::protocol::PacketType::EntitySpawn);
        const auto bobSpawn = transport->idsFor(bob, net::protocol::PacketType::EntitySpawn);

        check(aliceSpawn.count(eAvatarA.raw()) && aliceSpawn.count(eNearA.raw()),
              "A's first packet spawns its own cluster");
        check(!aliceSpawn.count(eAvatarB.raw()) && !aliceSpawn.count(eNearB.raw()),
              "A never receives B's distant cluster");
        check(bobSpawn.count(eAvatarB.raw()) && bobSpawn.count(eNearB.raw()), "B's first packet spawns its own cluster");
        check(!bobSpawn.count(eAvatarA.raw()) && !bobSpawn.count(eNearA.raw()),
              "B never receives A's distant cluster");

        check(transport->idsFor(alice, net::protocol::PacketType::StateDelta).empty(), "nothing is a delta on tick 1");
        check(transport->idsFor(alice, net::protocol::PacketType::EntityDestroy).empty(),
              "nothing is destroyed on tick 1");
        check(!aliceSpawn.count(eRoamer.raw()), "the far roamer is outside A's radius");
    }

    // ── Tick 2: dormancy — an unchanged entity sends nothing, a moved one deltas ─ //
    placeEntity(registry, world, eNearA, 6.0f, 0.0f, 0.0f); // eNearA slides, stays in range
    transport->clear();
    aoi.execute(1.0f / 60.0f);

    {
        const auto aliceSpawn = transport->idsFor(alice, net::protocol::PacketType::EntitySpawn);
        const auto aliceDelta = transport->idsFor(alice, net::protocol::PacketType::StateDelta);

        check(aliceSpawn.empty(), "an already-known entity is not spawned again");
        check(aliceDelta.count(eNearA.raw()), "a known entity that moved is sent as a delta");
        check(!aliceDelta.count(eAvatarA.raw()), "an unchanged entity is dormant — no traffic (§6.2.7)");
        check(transport->idsFor(alice, net::protocol::PacketType::EntityDestroy).empty(),
              "nothing left A's radius, so nothing is destroyed");
    }

    // ── Tick 3: one entity leaves the radius, one enters, the avatar moves ───── //
    placeEntity(registry, world, eNearA, 0.0f, 5000.0f, 0.0f); // leaves A's radius
    placeEntity(registry, world, eRoamer, 10.0f, 0.0f, 0.0f);  // enters A's radius
    placeEntity(registry, world, eAvatarA, 1.0f, 0.0f, 0.0f);  // the avatar itself slides

    transport->clear();
    aoi.execute(1.0f / 60.0f);

    {
        const auto aliceSpawn = transport->idsFor(alice, net::protocol::PacketType::EntitySpawn);
        const auto aliceDelta = transport->idsFor(alice, net::protocol::PacketType::StateDelta);
        const auto aliceDestroy = transport->idsFor(alice, net::protocol::PacketType::EntityDestroy);

        check(aliceSpawn.count(eRoamer.raw()), "an entity entering the radius is spawned");
        check(aliceDestroy.count(eNearA.raw()), "an entity leaving the radius is destroyed");
        check(aliceDelta.count(eAvatarA.raw()), "an entity that moved and stayed in range is a delta");
        check(!aliceDelta.count(eNearA.raw()), "the departed entity is not also sent as a delta");
        check(!aliceSpawn.count(eAvatarA.raw()), "the still-present avatar is not re-spawned");
    }

    // ── Purge: a disconnected client's memory is dropped, not leaked ─────────── //
    {
        [[maybe_unused]] auto d = sessions.disconnect(2);
        transport->clear();
        aoi.execute(1.0f / 60.0f);
        check(transport->allIdsFor(bob).empty(), "a disconnected client receives nothing");
        // Re-joining B on the same avatar must spawn its cluster afresh, proving
        // the known-set was reaped (a lingering set would send deltas instead).
        check(joinClient(sessions, 2, bob, eAvatarB) != nullptr, "client B re-joins");
        transport->clear();
        aoi.execute(1.0f / 60.0f);
        check(!transport->idsFor(bob, net::protocol::PacketType::EntitySpawn).empty(),
              "a re-joined client is spawned its cluster afresh (known-set was reaped)");
    }

    // ── Non-regression + the whole point: AOI ≪ full broadcast ──────────────── //
    // A fresh, identical dispersed world driven once by each system. The full
    // broadcast (the interestRadius==0 fallback) sends every entity to every
    // client; AOI sends each client only its own cluster.
    {
        ecs::Registry reg2;
        ecs::WorldPartition world2{cellSize, 4096};
        net::session::SessionManager sm2;

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
        engine::systems::BroadcastSystem full{sm2, fullTransport, world2, reg2};
        full.execute(1.0f / 60.0f);

        const auto aliceFull = fullTransport->allIdsFor(alice);
        check(aliceFull.count(a0.raw()) && aliceFull.count(a1.raw()) && aliceFull.count(b0.raw()) &&
                  aliceFull.count(b1.raw()),
              "the full-broadcast fallback sends every entity to every client");
        const core::usize fullTotal = fullTransport->totalIdsSent();

        auto aoiTransport = std::make_shared<CapturingTransport>();
        engine::systems::AoiBroadcastSystem aoi2{sm2, aoiTransport, world2, reg2, radius};
        aoi2.execute(1.0f / 60.0f);

        const auto aliceAoi = aoiTransport->allIdsFor(alice);
        check(!aliceAoi.count(b0.raw()) && !aliceAoi.count(b1.raw()), "AOI keeps the far cluster out of A's stream");
        const core::usize aoiTotal = aoiTransport->totalIdsSent();

        std::printf("  full broadcast sent %zu entity-records, AOI sent %zu\n", static_cast<size_t>(fullTotal),
                    static_cast<size_t>(aoiTotal));
        check(aoiTotal < fullTotal, "AOI serialises strictly fewer entity-records than the full broadcast");
    }

    // ── Bandwidth budget + priority + anti-starvation (§6.2.7) ───────────────── //
    // One client, six movers all in range at increasing distance. A byte budget
    // that fits ~two deltas per tick forces the server to choose: it must send the
    // closest first, yet still serve every entity within a few ticks (nothing
    // starves, because a skipped entity's staleness rises until it wins).
    {
        ecs::Registry reg3;
        ecs::WorldPartition world3{cellSize, 4096};
        net::session::SessionManager sm3;

        const auto avatar = spawnEntity(reg3);
        placeEntity(reg3, world3, avatar, 0.0f, 0.0f, 0.0f);

        constexpr int kMovers = 6;
        ecs::EntityId movers[kMovers];
        float moverX[kMovers];
        for (int i = 0; i < kMovers; ++i)
        {
            movers[i] = spawnEntity(reg3);
            moverX[i] = 5.0f * static_cast<float>(i + 1); // 5,10,15,20,25,30 — all in radius
            placeEntity(reg3, world3, movers[i], moverX[i], 0.0f, 0.0f);
        }
        check(joinClient(sm3, 1, alice, avatar) != nullptr, "the budget client joins");

        auto budgetTransport = std::make_shared<CapturingTransport>();
        // Each single-axis delta is 9 bytes (id + mask + one float); a 20-byte
        // budget therefore admits exactly two per tick.
        const core::u32 kBudget = 20;
        engine::systems::AoiBroadcastSystem budgeted{sm3,   budgetTransport,       world3,
                                                     reg3,  math::Fixed32::fromFloat(100.0f), /*keyframe*/ 1000,
                                                     kBudget};

        // Tick 1: everything is spawned (spawns are not budgeted — a client must
        // learn who is there). Establishes the baseline for the delta stream.
        budgeted.execute(1.0f / 60.0f);

        std::set<core::u32> served; // union of delta ids across the budgeted ticks
        int perTickMax = 0;
        std::set<core::u32> firstDeltaTick;
        for (int tick = 0; tick < kMovers; ++tick) // enough ticks to serve all six
        {
            for (int i = 0; i < kMovers; ++i) // nudge every mover so all are due
            {
                moverX[i] += 0.5f;
                placeEntity(reg3, world3, movers[i], moverX[i], 0.0f, 0.0f);
            }
            budgetTransport->clear();
            budgeted.execute(1.0f / 60.0f);

            const auto delta = budgetTransport->idsFor(alice, net::protocol::PacketType::StateDelta);
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

    // ── Network LOD: a far entity updates on a slower cadence (§6.2.6) ───────── //
    // One client at the origin, a near entity and a far one, both moving every
    // tick. With a full-rate near ring and a 4-tick far ring, the near entity is
    // sent every tick and the far one only every fourth — measured over 8 ticks.
    {
        ecs::Registry reg4;
        ecs::WorldPartition world4{cellSize, 4096};
        net::session::SessionManager sm4;

        const auto avatar = spawnEntity(reg4);
        const auto nearE = spawnEntity(reg4);
        const auto farE = spawnEntity(reg4);
        placeEntity(reg4, world4, avatar, 0.0f, 0.0f, 0.0f);
        placeEntity(reg4, world4, nearE, 5.0f, 0.0f, 0.0f);   // inside the near ring
        placeEntity(reg4, world4, farE, 60.0f, 0.0f, 0.0f);   // near ring < d < interest radius
        check(joinClient(sm4, 1, alice, avatar) != nullptr, "the LOD client joins");

        auto lodTransport = std::make_shared<CapturingTransport>();
        engine::systems::AoiBroadcastSystem lod{sm4,  lodTransport, world4, reg4, math::Fixed32::fromFloat(100.0f),
                                                /*keyframe*/ 1000};
        constexpr core::u32 kFarInterval = 4;
        lod.setNetworkLod(math::Fixed32::fromFloat(20.0f), kFarInterval);

        lod.execute(1.0f / 60.0f); // tick 1: spawn both, baseline set

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
            const auto delta = lodTransport->idsFor(alice, net::protocol::PacketType::StateDelta);
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
