#include <lpl/net/netcode/Interpolation.hpp>

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void check(bool cond, const char *what)
{
    std::printf("  %s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond)
        ++g_failures;
}

bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

lpl::math::Vec3<float> v3(float x, float y = 0.0f, float z = 0.0f) { return {x, y, z}; }

/**
 * @brief Linear interpolation between two snapshots.
 */
void checkLinearInterpolationBetweenTwoSnapshots()
{
    lpl::net::netcode::SnapshotInterpolator interp;
    interp.addSample(0.0, v3(0.0f));
    interp.addSample(1.0, v3(10.0f));

    check(near(interp.sample(0.5).x, 5.0f), "the midpoint interpolates halfway");
    check(near(interp.sample(0.25).x, 2.5f), "a quarter of the way interpolates to a quarter");
    check(near(interp.sample(0.0).x, 0.0f) && near(interp.sample(1.0).x, 10.0f), "the endpoints are exact");
}

/**
 * @brief Rendering the past hides a dropped packet.
 *
 * @details The client renders at now - delay. A snapshot for t=1 was dropped; the buffer holds only t=0 and
 *          t=2, yet rendering t=1 (in the past) still yields a smooth midpoint instead of a freeze.
 */
void checkRenderingThePastHidesADroppedPacket()
{
    lpl::net::netcode::SnapshotInterpolator interp;
    interp.addSample(0.0, v3(0.0f));
    interp.addSample(2.0, v3(20.0f));
    check(near(interp.sample(1.0).x, 10.0f), "a missing snapshot is bridged by interpolation");
}

/**
 * @brief Clamp before the oldest sample.
 */
void checkRenderTimeBeforeTheOldestSampleClamps()
{
    lpl::net::netcode::SnapshotInterpolator interp;
    interp.addSample(5.0, v3(3.0f));
    interp.addSample(6.0, v3(4.0f));
    check(near(interp.sample(0.0).x, 3.0f), "a render time before the oldest clamps to the oldest");
}

/**
 * @brief Dead reckoning past the newest snapshot.
 *
 * @details The entity moves at +10 units/s at the last sample.
 */
void checkDeadReckoningPastTheNewestSnapshot()
{
    lpl::net::netcode::SnapshotInterpolator interp;
    interp.addSample(0.0, v3(0.0f), v3(0.0f));
    interp.addSample(1.0, v3(10.0f), v3(10.0f));

    check(!interp.extrapolating(0.5), "inside the buffer it interpolates, not extrapolates");
    check(interp.extrapolating(1.5), "past the newest it extrapolates");
    check(near(interp.sample(1.5).x, 15.0f), "dead reckoning advances along the last velocity");
    check(near(interp.sample(2.0).x, 20.0f), "further ahead extends the extrapolation linearly");
}

/**
 * @brief Out-of-order / duplicate snapshots are ignored.
 *
 * @details The sample at t=0.5 is a late packet for a superseded tick.
 */
void checkOutOfOrderSnapshotsAreIgnored()
{
    lpl::net::netcode::SnapshotInterpolator interp;
    interp.addSample(0.0, v3(0.0f));
    interp.addSample(1.0, v3(10.0f));
    interp.addSample(0.5, v3(999.0f));
    check(interp.count() == 2, "an out-of-order snapshot is dropped");
    check(near(interp.sample(0.5).x, 5.0f), "the stale sample did not corrupt the interpolation");
}

/**
 * @brief Capacity: only the most recent snapshots are kept.
 */
void checkOnlyTheMostRecentSnapshotsAreKept()
{
    lpl::net::netcode::SnapshotInterpolator interp{4};
    for (int i = 0; i < 10; ++i)
        interp.addSample(static_cast<lpl::core::f64>(i), v3(static_cast<float>(i)));
    check(interp.count() == 4, "the ring keeps at most `capacity` snapshots");
    check(near(interp.newestTime(), 9.0), "the newest snapshot is retained");
}

} // namespace

/**
 * @brief Snapshot interpolation + dead reckoning test (book §6.2.4).
 *
 * @details Proves the client render-smoothing of remote entities: rendering the recent past interpolates
 *          between the two bracketing snapshots (so a late or dropped packet is hidden), and running past the
 *          newest snapshot extrapolates along the last velocity (dead reckoning) instead of freezing.
 */
int main()
{
    std::printf("== snapshot interpolation + dead reckoning ==\n");

    checkLinearInterpolationBetweenTwoSnapshots();
    checkRenderingThePastHidesADroppedPacket();
    checkRenderTimeBeforeTheOldestSampleClamps();
    checkDeadReckoningPastTheNewestSnapshot();
    checkOutOfOrderSnapshotsAreIgnored();
    checkOnlyTheMostRecentSnapshotsAreKept();

    std::printf(g_failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
