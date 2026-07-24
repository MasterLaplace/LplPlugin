/*
** LplPlugin — replication relevancy / priority test (book §6.2.7)
**
** Proves the pure priority scorer has the two properties the budgeted broadcast
** relies on: for equal staleness, a closer entity outranks a farther one; and
** staleness grows without bound, so an entity skipped for being far or over
** budget eventually outranks any close one — the scheme cannot starve.
*/

#include <lpl/net/relevancy/Relevancy.hpp>

#include <cstdio>

using namespace lpl;
namespace rel = lpl::net::relevancy;

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

} // namespace

int main()
{
    std::printf("== replication relevancy ==\n");

    // ── Proximity: closer wins at equal staleness ──────────────────────────── //
    check(rel::priority(1.0f, 0) > rel::priority(100.0f, 0), "closer entity outranks farther (equal staleness)");
    check(rel::priority(0.0f, 5) > rel::priority(50.0f, 5),
          "at the avatar it outranks a distant one (equal staleness)");

    // ── Staleness: waiting raises priority monotonically ───────────────────── //
    check(rel::priority(10.0f, 3) > rel::priority(10.0f, 0), "the same entity rises as it waits");
    bool monotone = true;
    for (core::u32 t = 1; t < 200; ++t)
        if (!(rel::priority(10.0f, t) > rel::priority(10.0f, t - 1)))
            monotone = false;
    check(monotone, "priority is strictly increasing in staleness");

    // ── Anti-starvation: a far, long-waiting entity beats a close, just-sent one //
    const float closeJustSent = rel::priority(0.0f, 0); // best possible proximity, no wait
    check(rel::priority(1.0e6f, 1000) > closeJustSent,
          "a far entity starved long enough outranks the closest just-sent one");

    // The crossover is bounded by the proximity weight: a far entity catches the
    // closest one within kProximityWeight ticks, so nobody waits forever.
    check(rel::priority(1.0e9f, static_cast<core::u32>(rel::kProximityWeight) + 1) > closeJustSent,
          "the starvation crossover is bounded by the proximity weight");

    // ── Dormancy gate: only changed or keyframe entities are due ───────────── //
    check(!rel::isDue(/*changed*/ false, /*keyframe*/ false), "an unchanged entity between keyframes is not due");
    check(rel::isDue(true, false), "a changed entity is due");
    check(rel::isDue(false, true), "a keyframe re-sends even an unchanged entity");

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
