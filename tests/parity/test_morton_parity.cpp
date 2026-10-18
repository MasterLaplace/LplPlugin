#include <cstdio>
#include <cstdlib>
#include <lpl/core/Log.hpp>
#include <lpl/math/Morton.hpp>

static int failures = 0;

static void checkMorton(lpl::core::i32 x, lpl::core::i32 y, lpl::core::i32 z)
{
    const lpl::core::u64 code = lpl::math::morton::encode3D(x, y, z);
    lpl::core::i32 dx = 0, dy = 0, dz = 0;
    lpl::math::morton::decode3D(code, dx, dy, dz);

    if (dx != x || dy != y || dz != z)
    {
        std::printf("  FAIL: encode(%d,%d,%d)=%llu -> decode=(%d,%d,%d)\n", x, y, z,
                    static_cast<unsigned long long>(code), dx, dy, dz);
        ++failures;
    }
    else
    {
        std::printf("  PASS: roundtrip(%d,%d,%d) code=%llu\n", x, y, z, static_cast<unsigned long long>(code));
    }
}

/**
 * @brief Zero round-trips.
 */
static void checkZeroRoundTrips() { checkMorton(0, 0, 0); }

/**
 * @brief The unit axes round-trip.
 */
static void checkUnitAxesRoundTrip()
{
    checkMorton(1, 0, 0);
    checkMorton(0, 1, 0);
    checkMorton(0, 0, 1);
}

/**
 * @brief Powers of two round-trip.
 */
static void checkPowersOfTwoRoundTrip()
{
    checkMorton(2, 4, 8);
    checkMorton(16, 32, 64);
}

/**
 * @brief The largest 10-bit values round-trip.
 */
static void checkLargestTenBitValuesRoundTrip() { checkMorton(1023, 1023, 1023); }

/**
 * @brief Mixed values round-trip.
 */
static void checkMixedValuesRoundTrip()
{
    checkMorton(42, 137, 255);
    checkMorton(100, 200, 300);
    checkMorton(511, 0, 511);
}

/**
 * @brief Negative values round-trip, supported through kMortonBias.
 */
static void checkNegativeValuesRoundTrip()
{
    checkMorton(-1, -1, -1);
    checkMorton(-100, 50, -200);
}

/**
 * @brief Ordering: encode(a) < encode(b) for monotonically increasing coordinates.
 */
static void checkCodesGrowWithTheCoordinates()
{
    const lpl::core::u64 c1 = lpl::math::morton::encode3D(0, 0, 0);
    const lpl::core::u64 c2 = lpl::math::morton::encode3D(1, 1, 1);
    if (c1 < c2)
    {
        std::printf("  PASS: ordering (0,0,0) < (1,1,1)\n");
    }
    else
    {
        std::printf("  FAIL: ordering %llu >= %llu\n", static_cast<unsigned long long>(c1),
                    static_cast<unsigned long long>(c2));
        ++failures;
    }
}

/**
 * @brief Parity test: Morton encoding/decoding roundtrip verification.
 *
 * Verifies that morton encode3D/decode3D operations are lossless for
 * a range of coordinate inputs.
 */
int main()
{
    lpl::core::Log::info("=== Morton3D Parity Test ===");

    checkZeroRoundTrips();

    checkUnitAxesRoundTrip();

    checkPowersOfTwoRoundTrip();

    checkLargestTenBitValuesRoundTrip();

    checkMixedValuesRoundTrip();

    checkNegativeValuesRoundTrip();

    checkCodesGrowWithTheCoordinates();

    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
