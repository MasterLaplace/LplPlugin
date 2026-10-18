#include <lpl/ai/AiMap.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(ai_map);

namespace {

constexpr lpl::core::u32 kSide = 12u;                                                  /**< Cells along each axis. */
constexpr lpl::core::u8 kWalk = static_cast<lpl::core::u8>(lpl::ai::Locomotion::Walk); /**< The body's one mode. */

/**
 * @brief An open field with a walkable goal cell walled in on its eight sides, so no route reaches it.
 *
 * @return The map.
 */
[[nodiscard]] lpl::ai::AiMap openFieldAroundAWalledGoal()
{
    lpl::ai::AiMap map{kSide, kSide};
    for (lpl::core::u32 z = 0u; z < kSide; ++z)
        for (lpl::core::u32 x = 0u; x < kSide; ++x)
        {
            const bool ring = x >= 8u && x <= 10u && z >= 8u && z <= 10u && !(x == 9u && z == 9u);
            map.setCapability(x, z, ring ? lpl::core::u8{0} : kWalk);
        }
    return map;
}

} // namespace

/**
 * @brief A search that runs dry expands each state it reached once, from its best cost.
 *
 * @details A state pushed again at a lower cost leaves its older entries in the heap. The test that was meant to
 *          drop them compared the popped estimate, cost plus heuristic, with the best cost alone, so it never held:
 *          every stale entry was expanded again when it came out.
 *
 *          The goal is unreachable so the search empties its heap: every entry comes out, and the count of
 *          expansions is the count of reached states only if each stale one was dropped. Turns are charged, which
 *          is what makes a state cheaper by a later route than by the first.
 */
LPL_TEST(a_search_expands_each_state_once)
{
    const lpl::ai::AiMap map = openFieldAroundAWalledGoal();
    lpl::ai::AiMapParams params;
    params.turnCost = 8u;
    lpl::pmr::vector<lpl::core::u32> path;
    lpl::ai::PathSearchStats stats;

    const lpl::core::u32 cost = map.findPath(1u, 1u, 9u, 9u, kWalk, params, path, &stats);

    test.check(cost == lpl::ai::AiMap::kNoPath && path.empty(), "the walled goal has no route");
    test.check(stats.skipped != 0u, "the heap held stale entries");
    test.check(stats.expanded == stats.reached, "and the search expanded each reached state once");
    test.measure("reached", stats.reached);
    test.measure("expanded", stats.expanded);
    test.measure("skipped", stats.skipped);
}

/**
 * @brief A search that cannot start reports nothing done.
 */
LPL_TEST(a_search_from_a_wall_reports_nothing)
{
    const lpl::ai::AiMap map = openFieldAroundAWalledGoal();
    lpl::pmr::vector<lpl::core::u32> path;
    lpl::ai::PathSearchStats stats{7u, 7u, 7u};

    const lpl::core::u32 cost = map.findPath(8u, 8u, 1u, 1u, kWalk, lpl::ai::AiMapParams{}, path, &stats);

    test.check(cost == lpl::ai::AiMap::kNoPath, "a start inside a wall has no route");
    test.check(stats.reached == 0u && stats.expanded == 0u && stats.skipped == 0u, "and the search reports nothing");
}
