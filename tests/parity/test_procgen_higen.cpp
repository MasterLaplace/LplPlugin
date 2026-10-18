#include <lpl/procgen/Chunking.hpp>
#include <lpl/procgen/HiGen.hpp>

#include <cstdio>
#include <string>

static int failures = 0;

static void check(bool ok, const std::string &what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

namespace {

constexpr lpl::core::u32 kFnvOffset = 0x811C9DC5u;
constexpr lpl::core::u32 kFnvPrime = 0x01000193u;

void fold(lpl::core::u32 &digest, lpl::core::u32 value)
{
    for (lpl::core::u32 byte = 0u; byte < 4u; ++byte)
    {
        digest ^= (value >> (byte * 8u)) & 0xFFu;
        digest *= kFnvPrime;
    }
}

/**
 * @brief The world the hierarchy is a hierarchy OF.
 */
lpl::procgen::ChunkParams worldParams()
{
    lpl::procgen::ChunkParams params;
    params.size = 32u;
    params.worldSeed = 90210u;
    return params;
}

/** @brief A pass, plus the counter that says how often it actually ran. */
struct CountedHeight {
    const lpl::procgen::ChunkParams *params{nullptr};
    lpl::core::u32 invocations{0u};

    lpl::core::u32 operator()(lpl::core::i32 worldX, lpl::core::i32 worldZ)
    {
        ++invocations;
        return static_cast<lpl::core::u32>(lpl::procgen::sampleWorldHeight(*params, worldX, worldZ).raw());
    }
};

/** @brief Side of the square of world cells every run covers. */
constexpr lpl::core::i32 kDomain = 64;

/**
 * @brief Folds the domain, taking each cell's value from the coarsest level given.
 *
 * The cascade rule made concrete: a fine cell reads the value of the COARSE cell
 * that covers it. With a single level of cellSize 1 that is the cell itself, which
 * is why the one-level run must equal the flat one exactly.
 *
 * A value missing from the cache is sampled at the coarse cell's ORIGIN, not at the
 * fine cell: a coarse answer that varied inside its own cell would not be a coarse
 * answer.
 */
lpl::core::u32 runHierarchy(const lpl::procgen::HiGenSchedule &schedule, lpl::core::u32 level, CountedHeight &pass,
                            lpl::procgen::HiGenCache &cache)
{
    lpl::core::u32 digest = kFnvOffset;
    for (lpl::core::i32 z = 0; z < kDomain; ++z)
        for (lpl::core::i32 x = 0; x < kDomain; ++x)
        {
            const lpl::procgen::ChunkCoord cell = lpl::procgen::levelCellOf(schedule, level, x, z);
            lpl::core::u32 value = 0u;
            if (!cache.lookup(level, cell, value))
            {
                const lpl::core::i32 side = static_cast<lpl::core::i32>(schedule.levels[level].cellSize);
                value = pass(cell.x * side, cell.z * side);
                cache.store(level, cell, value);
            }
            fold(digest, value);
        }
    return digest;
}

/**
 * @brief One level is the flat pipeline, to the bit.
 */
void checkOneLevelIsTheFlatPipeline(const lpl::procgen::ChunkParams &params)
{
    std::printf("-- a single level is the flat pipeline --\n");
    lpl::core::u32 flatDigest = kFnvOffset;
    lpl::core::u32 flatInvocations = 0u;
    CountedHeight pass{&params, 0u};
    for (lpl::core::i32 z = 0; z < kDomain; ++z)
        for (lpl::core::i32 x = 0; x < kDomain; ++x)
            fold(flatDigest, pass(x, z));
    flatInvocations = pass.invocations;

    lpl::procgen::HiGenSchedule schedule;
    check(schedule.addLevel(1u), "a one-level hierarchy is legal");

    CountedHeight hierarchical{&params, 0u};
    lpl::procgen::HiGenCache cache;
    const lpl::core::u32 digest = runHierarchy(schedule, 0u, hierarchical, cache);
    std::printf("    flat fold=0x%08X (%u invocations), one level fold=0x%08X (%u invocations)\n", flatDigest,
                flatInvocations, digest, hierarchical.invocations);
    check(digest == flatDigest, "one level folds bit-identically to the flat run");
    check(hierarchical.invocations == flatInvocations, "and costs exactly the same, which is the honest baseline");
    check(cache.hits() == 0u, "nothing repeats at cellSize 1, so nothing is served from the cache");
}

/**
 * @brief Coarser levels cost less, and by how much is the whole argument.
 *
 * Coarsest first is the execution order, not a style: a level may only read what a
 * coarser one has already produced.
 *
 * The saving is quadratic in the cell size, which is the reason a hierarchy is worth
 * its bookkeeping at all: 16x16 world cells share one answer.
 */
void checkCoarserLevelsCostLess(const lpl::procgen::ChunkParams &params)
{
    std::printf("\n-- the cache IS the hierarchy --\n");
    lpl::procgen::HiGenSchedule schedule;
    check(schedule.addLevel(16u), "a coarse level");
    check(schedule.addLevel(4u), "a middle level");
    check(schedule.addLevel(1u), "and a fine one, coarsest first");
    check(schedule.levelCount == 3u, "three levels");
    check(!schedule.addLevel(8u), "a level coarser than the last is refused");

    lpl::core::u32 invocationsPerLevel[3]{};
    for (lpl::core::u32 level = 0u; level < schedule.levelCount; ++level)
    {
        CountedHeight pass{&params, 0u};
        lpl::procgen::HiGenCache cache;
        (void) runHierarchy(schedule, level, pass, cache);
        invocationsPerLevel[level] = pass.invocations;
        const lpl::core::u32 side = schedule.levels[level].cellSize;
        const lpl::core::u32 expected =
            (static_cast<lpl::core::u32>(kDomain) / side) * (static_cast<lpl::core::u32>(kDomain) / side);
        std::printf("    cellSize %2u: %5u invocations, %5u cache hits (expected %u distinct cells)\n", side,
                    pass.invocations, cache.hits(), expected);
        check(pass.invocations == expected, "a coarse pass runs once per coarse cell, not once per world cell");
        check(cache.hits() + cache.misses() == static_cast<lpl::core::u32>(kDomain) * kDomain,
              "every lookup is accounted for as a hit or a miss");
        check(cache.size() == expected, "and the cache holds exactly the distinct coarse cells");
    }

    check(invocationsPerLevel[0] < invocationsPerLevel[1], "coarser costs strictly less than middling");
    check(invocationsPerLevel[1] < invocationsPerLevel[2], "and middling strictly less than fine");
    check(invocationsPerLevel[2] / invocationsPerLevel[0] == 256u, "a 16-cell level costs 256 times less");
}

/**
 * @brief The rule is mechanical, not a comment.
 */
void checkTheCascadeRuleIsMechanical()
{
    std::printf("\n-- a pass may not read a finer level --\n");
    lpl::procgen::HiGenSchedule schedule;
    (void) schedule.addUnbounded();
    (void) schedule.addLevel(8u);
    (void) schedule.addLevel(2u);

    check(lpl::procgen::checkCascade(schedule, 2u, 1u) == lpl::procgen::CascadeViolation::None,
          "a fine pass may read a coarse level");
    check(lpl::procgen::checkCascade(schedule, 1u, 2u) == lpl::procgen::CascadeViolation::ReadsFiner,
          "a coarse pass may NOT read a fine one — the forest does not follow the trees");
    check(lpl::procgen::checkCascade(schedule, 1u, 7u) == lpl::procgen::CascadeViolation::UnknownLevel,
          "and a level nobody declared is an error, not a guess");
    check(!schedule.addUnbounded(), "there is at most one unbounded level");
}

/**
 * @brief Forgetting, because a cache that outlives its parameters lies.
 */
void checkClearingForgetsTheCache(const lpl::procgen::ChunkParams &params)
{
    std::printf("\n-- clearing --\n");
    lpl::procgen::HiGenSchedule schedule;
    (void) schedule.addLevel(8u);
    CountedHeight pass{&params, 0u};
    lpl::procgen::HiGenCache cache;
    (void) runHierarchy(schedule, 0u, pass, cache);
    const lpl::core::u32 before = pass.invocations;
    cache.clear();
    check(cache.size() == 0u, "clearing empties the cache");
    (void) runHierarchy(schedule, 0u, pass, cache);
    check(pass.invocations == before * 2u, "so the work is genuinely done again, not silently reused");
}

} // namespace

/**
 * @brief The hierarchy's two claims, measured: same world, fewer invocations.
 *
 * HiGen was the last phase of the procedural work with no test target at all, and
 * the two things it asserts about itself are exactly the two a comment cannot
 * settle:
 *
 *  1. **It is a refactor, not a rewrite.** Run the same pipeline through a
 *     single-level hierarchy and the world must fold BIT-IDENTICALLY to the flat
 *     one. A hierarchy that changes the world while claiming to reorganise the work
 *     is the most expensive kind of wrong: every downstream signature moves and the
 *     cause is three layers away.
 *
 *  2. **The cache is the point.** Without it, running a coarse pass "per level cell"
 *     degenerates into running it once per fine cell — the flat pipeline with extra
 *     bookkeeping. So the invocation count between one level and three is not a
 *     nicety, it is the entire justification for HiGenCache existing, and it is
 *     measured here rather than asserted in prose.
 *
 * The pass under test is @c procgen::sampleWorldHeight, because a real generator
 * function is what makes claim 1 mean anything: it is an fBm with octaves, so a
 * hierarchy that rounded a coordinate anywhere would show up immediately as a
 * different fold.
 */
int main()
{
    std::printf("== hierarchical generation ==\n\n");

    const lpl::procgen::ChunkParams params = worldParams();

    checkOneLevelIsTheFlatPipeline(params);
    checkCoarserLevelsCostLess(params);
    checkTheCascadeRuleIsMechanical();
    checkClearingForgetsTheCache(params);

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
