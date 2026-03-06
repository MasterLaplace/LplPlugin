/**
 * @file test_morton_parity.cpp
 * @brief Parity test: Morton encoding/decoding roundtrip verification.
 *
 * Verifies that morton encode3D/decode3D operations are lossless for
 * a range of coordinate inputs.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#include <lpl/math/Morton.hpp>
#include <lpl/core/Log.hpp>
#include <cstdio>
#include <cstdlib>

using namespace lpl;

static int failures = 0;

static void checkMorton(core::i32 x, core::i32 y, core::i32 z)
{
    const core::u64 code = math::morton::encode3D(x, y, z);
    core::i32 dx = 0, dy = 0, dz = 0;
    math::morton::decode3D(code, dx, dy, dz);

    if (dx != x || dy != y || dz != z)
    {
        std::printf("  FAIL: encode(%d,%d,%d)=%llu -> decode=(%d,%d,%d)\n",
                    x, y, z, static_cast<unsigned long long>(code), dx, dy, dz);
        ++failures;
    }
    else
    {
        std::printf("  PASS: roundtrip(%d,%d,%d) code=%llu\n",
                    x, y, z, static_cast<unsigned long long>(code));
    }
}

int main()
{
    core::Log::info("=== Morton3D Parity Test ===");

    // Zero
    checkMorton(0, 0, 0);

    // Unit axes
    checkMorton(1, 0, 0);
    checkMorton(0, 1, 0);
    checkMorton(0, 0, 1);

    // Powers of two
    checkMorton(2, 4, 8);
    checkMorton(16, 32, 64);

    // Max 10-bit values (Morton3D supports 10 bits per axis)
    checkMorton(1023, 1023, 1023);

    // Mixed values
    checkMorton(42, 137, 255);
    checkMorton(100, 200, 300);
    checkMorton(511, 0, 511);

    // Negative values (supported via kMortonBias)
    checkMorton(-1, -1, -1);
    checkMorton(-100, 50, -200);

    // Ordering: encode(a) < encode(b) for monotonically increasing coords
    {
        const core::u64 c1 = math::morton::encode3D(0, 0, 0);
        const core::u64 c2 = math::morton::encode3D(1, 1, 1);
        if (c1 < c2)
        {
            std::printf("  PASS: ordering (0,0,0) < (1,1,1)\n");
        }
        else
        {
            std::printf("  FAIL: ordering %llu >= %llu\n",
                        static_cast<unsigned long long>(c1),
                        static_cast<unsigned long long>(c2));
            ++failures;
        }
    }

    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
