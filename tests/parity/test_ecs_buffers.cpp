/**
 * @file test_ecs_buffers.cpp
 * @brief The double-buffer contract, which nothing tested until it was suspected.
 *
 * @warning Written because a walk built on top of the ECS lost its positions every other tick, and the
 * ECS was the obvious suspect: `swapBuffers` is base machinery that predates almost everything
 * here and had no test of its own. It turned out to be innocent -- the fault was a goal chosen
 * afresh every tick, so the traveller walked back the way it came -- but the absence of this file
 * is what made five wrong hypotheses possible. A base feature with no test is a base feature
 * every future bug gets blamed on.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/WorldPosition.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>

#include <cstdio>

namespace {

int gChecks = 0;
int gFailures = 0;

/**
 * @brief Records one assertion.
 *
 * @param what Description.
 * @param ok   Whether it held.
 */
void check(const char *what, bool ok)
{
    ++gChecks;
    if (ok)
        return;
    ++gFailures;
    std::printf("  (fail) %s\n", what);
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("── a write survives a swap\n");
    {
        ecs::Registry registry;
        ecs::Archetype archetype;
        archetype.add(ecs::ComponentId::Position);
        const auto entity = registry.createEntity(archetype);
        check("an entity is created", static_cast<bool>(entity));

        bool everyReadSawTheLastWrite = true;
        for (int tick = 0; tick < 6; ++tick)
        {
            core::u32 row = 0u;
            ecs::Chunk *chunk = registry.chunkOf(entity.value(), row);
            if (chunk == nullptr)
            {
                everyReadSawTheLastWrite = false;
                break;
            }
            auto *positions =
                static_cast<math::Vec3<math::Fixed32> *>(chunk->writeComponent(ecs::ComponentId::Position));
            // @warning The write side, which is what every system in this repository reads: `swapBuffers`
            // copies back to front and THEN swaps, so the write buffer already carries this
            // frame's values and a read-modify-write on it is both current and published.
            const float expected = static_cast<float>(tick * 10);
            if (positions[row].x.toFloat() != expected)
                everyReadSawTheLastWrite = false;
            positions[row].x = math::Fixed32::fromFloat(expected + 10.0f);
            registry.swapAllBuffers();
        }
        check("six ticks of read-modify-write keep their value", everyReadSawTheLastWrite);
    }

    std::printf("── and does so with several components, walked by partition\n");
    {
        // The shape a real system uses: more than one component, reached by iterating partitions
        // rather than by entity. Both differences were suspected of losing writes; neither does.
        ecs::Registry registry;
        ecs::Archetype archetype;
        archetype.add(ecs::ComponentId::Position);
        archetype.add(ecs::ComponentId::Historical);
        const auto entity = registry.createEntity(archetype);
        check("an entity of two components is created", static_cast<bool>(entity));

        bool positionsHeld = true;
        bool identityHeld = true;
        for (int tick = 0; tick < 6; ++tick)
        {
            for (const auto &partition : registry.partitions())
            {
                if (partition == nullptr || !partition->archetype().has(ecs::ComponentId::Historical))
                    continue;
                for (const auto &chunk : partition->chunks())
                {
                    if (chunk == nullptr)
                        continue;
                    auto *positions =
                        static_cast<math::Vec3<math::Fixed32> *>(chunk->writeComponent(ecs::ComponentId::Position));
                    auto *identity = static_cast<core::u32 *>(chunk->writeComponent(ecs::ComponentId::Historical));
                    if (positions == nullptr || identity == nullptr)
                    {
                        positionsHeld = false;
                        break;
                    }
                    if (tick == 0)
                        identity[0] = 4242u;
                    else if (identity[0] != 4242u)
                    {
                        // @warning An identity written ONCE, at creation, and never again. A component
                        // that only the first tick writes is the case a swap can silently lose,
                        // and it is exactly how a body forgets who it is.
                        identityHeld = false;
                    }
                    if (positions[0].x.toFloat() != static_cast<float>(tick * 10))
                        positionsHeld = false;
                    positions[0].x = math::Fixed32::fromFloat(static_cast<float>(tick * 10) + 10.0f);
                }
            }
            registry.swapAllBuffers();
        }
        check("positions survive when reached through a partition walk", positionsHeld);
        check("and a value written only once survives every later swap", identityHeld);
    }

    std::printf("── a position that is a chunk plus an offset\n");
    {
        const auto chunk = lpl::math::Fixed32::fromFloat(512.0f);
        lpl::ecs::WorldPosition p;
        p.localX = lpl::math::Fixed32::fromFloat(500.0f);

        // A step that crosses the boundary carries into the cell index rather than growing the
        // local coordinate -- which is what keeps every local value small enough that a product
        // of two of them cannot leave Q16.16.
        p.localX = p.localX + lpl::math::Fixed32::fromFloat(20.0f);
        lpl::ecs::normaliseWorldPosition(p, chunk);
        check("crossing a boundary moves the cell", p.chunkX == 1);
        check("and folds the offset back inside", p.localX == lpl::math::Fixed32::fromFloat(8.0f));

        // @warning Backwards too, and this is the direction a naive fold gets wrong: a body at local 2
        // stepping back 10 is at 504 of the PREVIOUS chunk, not at -8 of this one.
        p.localX = p.localX - lpl::math::Fixed32::fromFloat(20.0f);
        lpl::ecs::normaliseWorldPosition(p, chunk);
        check("stepping back crosses the other way", p.chunkX == 0);
        check("with the offset inside again", p.localX >= lpl::math::Fixed32{} && p.localX < chunk);

        // Range: a world that reaches a planet. Earth's circumference is 40 075 km, which is
        // 78 272 chunks of 512 m -- the point of the split, in one number.
        lpl::ecs::WorldPosition far;
        far.chunkX = 78272;
        far.localX = lpl::math::Fixed32::fromFloat(511.5f);
        lpl::ecs::normaliseWorldPosition(far, chunk);
        check("a cell far past what Q16.16 spans is untouched", far.chunkX == 78272);
        check("and its offset is still exact", far.localX == lpl::math::Fixed32::fromFloat(511.5f));

        // The delta of two positions is a chunk count plus a remainder, never one flat number:
        // that is where the overflow used to be.
        lpl::core::i32 dcx = 0;
        lpl::core::i32 dcz = 0;
        lpl::math::Fixed32 dlx{};
        lpl::math::Fixed32 dlz{};
        lpl::ecs::worldDelta(p, far, chunk, dcx, dcz, dlx, dlz);
        check("a delta across 78 272 chunks is expressed exactly", dcx == 78272 && dcz == 0);
        check("with the leftover inside one chunk", dlx > -chunk && dlx < chunk && dlz == lpl::math::Fixed32{});
    }

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
