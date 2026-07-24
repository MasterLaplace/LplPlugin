/*
** LplPlugin — snapshot interpolation + dead reckoning test (book §6.2.4)
**
** Proves the client render-smoothing of remote entities: rendering the recent
** past interpolates between the two bracketing snapshots (so a late or dropped
** packet is hidden), and running past the newest snapshot extrapolates along the
** last velocity (dead reckoning) instead of freezing.
*/

#include <lpl/net/netcode/Interpolation.hpp>

#include <cmath>
#include <cstdio>

using namespace lpl;
using net::netcode::SnapshotInterpolator;

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

math::Vec3<float> v3(float x, float y = 0.0f, float z = 0.0f) { return {x, y, z}; }

} // namespace

int main()
{
    std::printf("== snapshot interpolation + dead reckoning ==\n");

    // ── Linear interpolation between two snapshots ─────────────────────────── //
    {
        SnapshotInterpolator interp;
        interp.addSample(0.0, v3(0.0f));
        interp.addSample(1.0, v3(10.0f));

        check(near(interp.sample(0.5).x, 5.0f), "the midpoint interpolates halfway");
        check(near(interp.sample(0.25).x, 2.5f), "a quarter of the way interpolates to a quarter");
        check(near(interp.sample(0.0).x, 0.0f) && near(interp.sample(1.0).x, 10.0f), "the endpoints are exact");
    }

    // ── Rendering the past hides a dropped packet ──────────────────────────── //
    {
        // The client renders at now - delay. A snapshot for t=1 was dropped; the
        // buffer holds only t=0 and t=2, yet rendering t=1 (in the past) still
        // yields a smooth midpoint instead of a freeze.
        SnapshotInterpolator interp;
        interp.addSample(0.0, v3(0.0f));
        interp.addSample(2.0, v3(20.0f)); // the t=1 snapshot never arrived
        check(near(interp.sample(1.0).x, 10.0f), "a missing snapshot is bridged by interpolation");
    }

    // ── Clamp before the oldest sample ─────────────────────────────────────── //
    {
        SnapshotInterpolator interp;
        interp.addSample(5.0, v3(3.0f));
        interp.addSample(6.0, v3(4.0f));
        check(near(interp.sample(0.0).x, 3.0f), "a render time before the oldest clamps to the oldest");
    }

    // ── Dead reckoning past the newest snapshot ────────────────────────────── //
    {
        SnapshotInterpolator interp;
        interp.addSample(0.0, v3(0.0f), v3(0.0f));
        interp.addSample(1.0, v3(10.0f), v3(10.0f)); // moving +10 units/s at the last sample

        check(!interp.extrapolating(0.5), "inside the buffer it interpolates, not extrapolates");
        check(interp.extrapolating(1.5), "past the newest it extrapolates");
        check(near(interp.sample(1.5).x, 15.0f), "dead reckoning advances along the last velocity");
        check(near(interp.sample(2.0).x, 20.0f), "further ahead extends the extrapolation linearly");
    }

    // ── Out-of-order / duplicate snapshots are ignored ─────────────────────── //
    {
        SnapshotInterpolator interp;
        interp.addSample(0.0, v3(0.0f));
        interp.addSample(1.0, v3(10.0f));
        interp.addSample(0.5, v3(999.0f)); // late packet for a superseded tick
        check(interp.count() == 2, "an out-of-order snapshot is dropped");
        check(near(interp.sample(0.5).x, 5.0f), "the stale sample did not corrupt the interpolation");
    }

    // ── Capacity: only the most recent snapshots are kept ──────────────────── //
    {
        SnapshotInterpolator interp{4};
        for (int i = 0; i < 10; ++i)
            interp.addSample(static_cast<core::f64>(i), v3(static_cast<float>(i)));
        check(interp.count() == 4, "the ring keeps at most `capacity` snapshots");
        check(near(interp.newestTime(), 9.0), "the newest snapshot is retained");
    }

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
