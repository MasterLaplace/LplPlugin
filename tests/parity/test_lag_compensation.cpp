/*
** LplPlugin — lag compensation test (book §6.2.9)
**
** Proves the server can rewind an entity to the instant a client saw and validate
** a hit against where it WAS, not where it has since moved. A shot aimed at the
** target's past position connects when validated in that past, and would miss if
** validated against the present — which is exactly the unfairness lag comp fixes.
*/

#include <lpl/net/netcode/LagCompensation.hpp>

#include <cstdio>

using namespace lpl;
using net::netcode::LagCompensator;

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

math::Vec3<float> v3(float x, float y = 0.0f, float z = 0.0f) { return {x, y, z}; }

} // namespace

int main()
{
    std::printf("== lag compensation ==\n");

    // A target runs along +X: x=0 at t=0, x=10 at t=1, x=20 at t=2 (the present).
    LagCompensator lag;
    const core::u32 target = 42;
    lag.record(target, 0.0, v3(0.0f));
    lag.record(target, 1.0, v3(10.0f));
    lag.record(target, 2.0, v3(20.0f));

    // ── Rewind reconstructs past position (interpolated) ───────────────────── //
    {
        bool found = false;
        check(lag.positionAt(target, 1.0, found).x == 10.0f && found, "rewind to a recorded tick is exact");
        check(lag.positionAt(target, 1.5, found).x == 15.0f, "rewind between ticks interpolates");
        (void) lag.positionAt(999, 1.0, found);
        check(!found, "an unknown entity reports not found");
    }

    // ── The core guarantee: a shot fair in the past, unfair in the present ──── //
    {
        // The shooter is at (10, 5, 0) firing straight down -Y. The client saw the
        // target at t=1, where it stood at x=10 — directly in the line of fire.
        const auto origin = v3(10.0f, 5.0f, 0.0f);
        const auto down = v3(0.0f, -1.0f, 0.0f);
        const float radius = 1.0f;

        check(lag.rayHitsAt(target, 1.0, origin, down, radius),
              "the shot validated at the time the client saw (t=1) HITS");
        check(!lag.rayHitsAt(target, 2.0, origin, down, radius),
              "the same shot validated against the present (t=2, target at x=20) MISSES");
    }

    // ── A shot that misses even in the past stays a miss ───────────────────── //
    {
        const auto origin = v3(0.0f, 5.0f, 0.0f);
        const auto down = v3(0.0f, -1.0f, 0.0f);
        check(!lag.rayHitsAt(target, 1.0, origin, down, 1.0f),
              "a shot nowhere near the target's past position still misses");
        check(lag.rayHitsAt(target, 0.0, origin, down, 1.0f),
              "but the same aim connects at t=0, when the target was at the origin");
    }

    check(lag.trackedCount() == 1, "one entity is tracked");
    lag.forget(target);
    check(lag.trackedCount() == 0, "a despawned entity is forgotten");

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
