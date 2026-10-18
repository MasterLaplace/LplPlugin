#include <lpl/net/relevancy/Relevancy.hpp>

#include <cstdio>

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

/**
 * @brief Proximity: closer wins at equal staleness.
 */
void checkCloserWinsAtEqualStaleness()
{
    check(lpl::net::relevancy::priority(1.0f, 0) > lpl::net::relevancy::priority(100.0f, 0),
          "closer entity outranks farther (equal staleness)");
    check(lpl::net::relevancy::priority(0.0f, 5) > lpl::net::relevancy::priority(50.0f, 5),
          "at the avatar it outranks a distant one (equal staleness)");
}

/**
 * @brief Staleness: waiting raises priority monotonically.
 */
void checkWaitingRaisesPriorityMonotonically()
{
    check(lpl::net::relevancy::priority(10.0f, 3) > lpl::net::relevancy::priority(10.0f, 0),
          "the same entity rises as it waits");
    bool monotone = true;
    for (lpl::core::u32 t = 1; t < 200; ++t)
        if (!(lpl::net::relevancy::priority(10.0f, t) > lpl::net::relevancy::priority(10.0f, t - 1)))
            monotone = false;
    check(monotone, "priority is strictly increasing in staleness");
}

/**
 * @brief Anti-starvation: a far, long-waiting entity beats a close, just-sent one.
 *
 * @details The close, just-sent entity has the best possible proximity and no wait. The crossover is
 *          bounded by the proximity weight: a far entity catches the closest one within kProximityWeight
 *          ticks and passes it on the next, so nobody waits forever.
 */
void checkFarLongWaitingEntityBeatsCloseJustSentOne()
{
    const float closeJustSent = lpl::net::relevancy::priority(0.0f, 0);
    check(lpl::net::relevancy::priority(1.0e6f, 1000) > closeJustSent,
          "a far entity starved long enough outranks the closest just-sent one");

    check(lpl::net::relevancy::priority(1.0e9f, static_cast<lpl::core::u32>(lpl::net::relevancy::kProximityWeight) +
                                                    1) > closeJustSent,
          "the starvation crossover is bounded by the proximity weight");
}

/**
 * @brief Dormancy gate: only changed or keyframe entities are due.
 */
void checkOnlyChangedOrKeyframeEntitiesAreDue()
{
    check(!lpl::net::relevancy::isDue(false, false), "an unchanged entity between keyframes is not due");
    check(lpl::net::relevancy::isDue(true, false), "a changed entity is due");
    check(lpl::net::relevancy::isDue(false, true), "a keyframe re-sends even an unchanged entity");
}

} // namespace

/**
 * @brief Replication relevancy / priority test (book §6.2.7).
 *
 * @details Proves the pure priority scorer has the two properties the budgeted broadcast relies on: for
 *          equal staleness, a closer entity outranks a farther one; and staleness grows without bound, so
 *          an entity skipped for being far or over budget eventually outranks any close one — the scheme
 *          cannot starve.
 */
int main()
{
    std::printf("== replication relevancy ==\n");

    checkCloserWinsAtEqualStaleness();
    checkWaitingRaisesPriorityMonotonically();
    checkFarLongWaitingEntityBeatsCloseJustSentOne();
    checkOnlyChangedOrKeyframeEntitiesAreDue();

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
