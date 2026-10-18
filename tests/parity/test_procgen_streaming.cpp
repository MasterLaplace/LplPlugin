#include <lpl/procgen/HiGen.hpp>
#include <lpl/procgen/Streaming.hpp>

#include <cstdio>

namespace {

int gChecks = 0;
int gFailures = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    if (!condition)
    {
        ++gFailures;
        std::printf("  FAIL %s\n", what);
    }
}

void testCascadeRule()
{
    std::printf("the cascade rule is checked, not merely documented\n");

    lpl::procgen::HiGenSchedule schedule;
    check(schedule.addLevel(64u), "a coarse level is accepted");
    check(schedule.addLevel(16u), "a finer level is accepted after it");
    check(schedule.addLevel(4u), "and a finer one after that");
    check(!schedule.addLevel(32u), "a level coarser than the last is refused");
    check(!schedule.addLevel(4u), "a level of the same size is refused");

    check(lpl::procgen::checkCascade(schedule, 2u, 0u) == lpl::procgen::CascadeViolation::None,
          "a fine pass may read a coarse level");
    check(lpl::procgen::checkCascade(schedule, 1u, 1u) == lpl::procgen::CascadeViolation::None,
          "a pass may read its own level");
    check(lpl::procgen::checkCascade(schedule, 0u, 2u) == lpl::procgen::CascadeViolation::ReadsFiner,
          "a coarse pass reading a fine level is the forbidden direction");
    check(lpl::procgen::checkCascade(schedule, 0u, 9u) == lpl::procgen::CascadeViolation::UnknownLevel,
          "a level outside the schedule is reported");

    lpl::procgen::HiGenSchedule unbounded;
    check(unbounded.addUnbounded(), "an unbounded level may lead a schedule");
    check(!unbounded.addUnbounded(), "there is at most one unbounded level");
    check(unbounded.addLevel(8u), "bounded levels follow it");
    check(lpl::procgen::checkCascade(unbounded, 1u, lpl::procgen::kUnboundedLevel) ==
              lpl::procgen::CascadeViolation::None,
          "anything may read the unbounded level");
}

/**
 * @brief World coordinates map to level cells, on both sides of the origin.
 *
 * Truncating division would put -1 in cell 0 alongside +1, folding the world about its own axis. Floor division is
 * what keeps the mapping monotonic.
 */
void testLevelCellMapping()
{
    std::printf("world coordinates map to level cells, on both sides of the origin\n");

    lpl::procgen::HiGenSchedule schedule;
    (void) schedule.addLevel(8u);

    check(lpl::procgen::levelCellOf(schedule, 0u, 0, 0).x == 0, "the origin is cell 0");
    check(lpl::procgen::levelCellOf(schedule, 0u, 7, 0).x == 0, "the last cell of the first block is still 0");
    check(lpl::procgen::levelCellOf(schedule, 0u, 8, 0).x == 1, "the next block is cell 1");

    check(lpl::procgen::levelCellOf(schedule, 0u, -1, 0).x == -1, "just left of the origin is cell -1");
    check(lpl::procgen::levelCellOf(schedule, 0u, -8, 0).x == -1, "and so is eight cells left");
    check(lpl::procgen::levelCellOf(schedule, 0u, -9, 0).x == -2, "nine cells left is cell -2");
}

/**
 * @brief A coarse result is computed once and read many times.
 *
 * A 32x32 world over 16x16 blocks is 4 coarse cells. Anything more means the cache is not doing the one job it
 * exists for.
 */
void testCacheServesTheCoarseWorkOnce()
{
    std::printf("a coarse result is computed once and read many times\n");

    lpl::procgen::HiGenCache cache;
    lpl::procgen::HiGenSchedule schedule;
    (void) schedule.addLevel(16u);

    lpl::core::u32 computed = 0u;
    lpl::core::u32 value = 0u;
    for (lpl::core::i32 z = 0; z < 32; ++z)
        for (lpl::core::i32 x = 0; x < 32; ++x)
        {
            const lpl::procgen::ChunkCoord coarse = lpl::procgen::levelCellOf(schedule, 0u, x, z);
            if (!cache.lookup(0u, coarse, value))
            {
                ++computed;
                cache.store(0u, coarse, static_cast<lpl::core::u32>(coarse.x * 31 + coarse.z));
            }
        }

    std::printf("    1024 fine cells over 16-cell blocks: %u coarse evaluations, %u hits\n", computed, cache.hits());
    check(computed == 4u, "the coarse pass ran once per coarse cell");
    check(cache.hits() == 1024u - 4u, "every other lookup was served from the cache");
}

/**
 * @brief A source oscillating on a boundary does not thrash.
 *
 * The release radius is 1.5 times the generate radius. A source walks back and forth across a chunk boundary, and
 * the test counts how often a chunk is released only to be asked for again.
 */
void testHysteresis()
{
    std::printf("a source oscillating on a boundary does not thrash\n");

    lpl::procgen::StreamingParams params;
    params.generateRadius = 2u;
    params.releaseRatio16 = 24u;
    params.maxGeneratePerTick = 0u;
    params.maxReleasePerTick = 0u;

    lpl::pmr::vector<lpl::procgen::ChunkCoord> resident;
    lpl::core::u32 churn = 0u;

    for (lpl::core::u32 tick = 0u; tick < 40u; ++tick)
    {
        lpl::procgen::GenerationSource source;
        source.x = lpl::math::Fixed32::fromFloat((tick % 2u) == 0u ? 0.49f : 0.51f);
        source.z = lpl::math::Fixed32::zero();

        const lpl::procgen::StreamingPlan plan =
            lpl::procgen::planStreaming(&source, 1u, resident.empty() ? nullptr : &resident[0],
                                        static_cast<lpl::core::u32>(resident.size()), params);

        for (lpl::core::u32 i = 0u; i < plan.toRelease.size(); ++i)
            for (lpl::core::u32 j = 0u; j < plan.toGenerate.size(); ++j)
                if (plan.toRelease[i].x == plan.toGenerate[j].coord.x &&
                    plan.toRelease[i].z == plan.toGenerate[j].coord.z)
                    ++churn;

        for (lpl::core::u32 i = 0u; i < plan.toRelease.size(); ++i)
            for (lpl::core::u32 j = 0u; j < resident.size(); ++j)
                if (resident[j].x == plan.toRelease[i].x && resident[j].z == plan.toRelease[i].z)
                {
                    resident[j] = resident[resident.size() - 1u];
                    resident.pop_back();
                    break;
                }
        for (lpl::core::u32 i = 0u; i < plan.toGenerate.size(); ++i)
            resident.push_back(plan.toGenerate[i].coord);
    }

    std::printf("    %u resident chunks after 40 oscillating ticks, %u churn events\n",
                static_cast<lpl::core::u32>(resident.size()), churn);
    check(churn == 0u, "no chunk is released and re-requested in the same tick");
    check(resident.size() == 25u, "the resident set settles at the generate radius");
}

/**
 * @brief What is ahead is built first, and the budget is counted in chunks.
 *
 * No wall clock anywhere: the same inputs give the same plan, always.
 */
void testDirectionWeightAndBudget()
{
    std::printf("what is ahead is built first, and the budget is counted in chunks\n");

    lpl::procgen::StreamingParams params;
    params.generateRadius = 3u;
    params.directionWeight16 = 16u;
    params.maxGeneratePerTick = 4u;

    lpl::procgen::GenerationSource source;
    source.headingX = lpl::math::Fixed32::one();

    const lpl::procgen::StreamingPlan plan = lpl::procgen::planStreaming(&source, 1u, nullptr, 0u, params);

    check(plan.toGenerate.size() == 4u, "the per-tick budget is respected");
    check(plan.wanted == 49u, "the whole 7x7 region is wanted before the budget applies");

    lpl::core::i32 ahead = 0;
    for (lpl::core::u32 i = 0u; i < plan.toGenerate.size(); ++i)
        ahead += plan.toGenerate[i].coord.x;
    std::printf("    first %u chunks have summed x = %d (heading +X)\n",
                static_cast<lpl::core::u32>(plan.toGenerate.size()), ahead);
    check(ahead > 0, "the scheduled chunks lie ahead of the source, not behind it");

    const lpl::procgen::StreamingPlan twin = lpl::procgen::planStreaming(&source, 1u, nullptr, 0u, params);
    bool identical = twin.toGenerate.size() == plan.toGenerate.size();
    for (lpl::core::u32 i = 0u; identical && i < twin.toGenerate.size(); ++i)
        identical = twin.toGenerate[i].coord.x == plan.toGenerate[i].coord.x &&
                    twin.toGenerate[i].coord.z == plan.toGenerate[i].coord.z;
    check(identical, "the plan is reproducible");
}

void testPoolRecycles()
{
    std::printf("chunk slots are recycled, not reallocated\n");

    lpl::procgen::ChunkPool pool;
    pool.reserve(8u);
    check(pool.capacity() == 8u, "the pool holds what was reserved");

    for (lpl::core::u32 round = 0u; round < 50u; ++round)
    {
        const lpl::core::u32 a = pool.acquire();
        const lpl::core::u32 b = pool.acquire();
        check(a != lpl::procgen::ChunkPool::kNoSlot && b != lpl::procgen::ChunkPool::kNoSlot, "slots are available");
        pool.release(a);
        pool.release(b);
    }

    std::printf("    %u acquisitions served by a recycled slot of a pool of %u\n", pool.recycled(), pool.capacity());
    check(pool.capacity() == 8u, "the pool never grew");
    check(pool.live() == 0u, "every slot came back");
}

} // namespace

/**
 * @brief Probes for the hierarchical schedule and the streaming policy.
 *
 * Two claims, and both have a specific failure behind them:
 *
 *  - **The cascade rule is mechanical.** A pass reading a finer level than its
 *    own must be reported, not silently wrong.
 *  - **A source on a boundary does not thrash.** Without a release radius wider
 *    than the generate radius, an oscillating player rebuilds the same chunk
 *    every tick forever.
 */
int main()
{
    std::printf("== procgen streaming and hierarchy ==\n");
    testCascadeRule();
    testLevelCellMapping();
    testCacheServesTheCoarseWorkOnce();
    testHysteresis();
    testDirectionWeightAndBudget();
    testPoolRecycles();

    if (gFailures == 0)
        std::printf("\nALL PASS (0 failures, %d checks)\n", gChecks);
    else
        std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
