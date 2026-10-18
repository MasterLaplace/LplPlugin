#include <cstdint>
#include <cstdio>

#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Simd.hpp>

static int failures = 0;

/** @brief A spread of raw Q16.16 values: zero, ±small, ±1.0, ±large, near-limits. */
static const lpl::core::i32 kSamples[] = {0,          1,           -1,      0x8000,   -0x8000,  0x10000,   -0x10000,
                                          0x00018000, -0x00018000, 0x34000, -0x34000, 98304,    -131072,   212992,
                                          655,        -655,        123456,  -123456,  0x400000, -0x400000, 0x7FFFFF,
                                          -0x7FFFFF,  3,           7,       13,       -3,       -7,        -13};
static constexpr int N = static_cast<int>(sizeof(kSamples) / sizeof(kSamples[0]));

/**
 * @brief Tests all lane-aligned quads over the cartesian product of samples: each lane of the SimdFixed4
 *        operation must equal the scalar Fixed32 one.
 */
template <typename ScalarOp, typename SimdOp> static void checkOp(const char *label, ScalarOp scalar, SimdOp simd)
{
    int mismatches = 0;
    for (int a = 0; a < N; ++a)
    {
        for (int b0 = 0; b0 < N; b0 += 4)
        {
            lpl::core::i32 lhs[4], rhs[4], out[4];
            for (int k = 0; k < 4; ++k)
            {
                lhs[k] = kSamples[a];
                rhs[k] = kSamples[(b0 + k) % N];
            }
            lpl::math::simd::SimdFixed4 vr =
                simd(lpl::math::simd::SimdFixed4::load(lhs), lpl::math::simd::SimdFixed4::load(rhs));
            vr.store(out);
            for (int k = 0; k < 4; ++k)
            {
                const lpl::core::i32 expected =
                    scalar(lpl::math::Fixed32::fromRaw(lhs[k]), lpl::math::Fixed32::fromRaw(rhs[k])).raw();
                if (out[k] != expected)
                {
                    if (mismatches < 5)
                        std::printf("    MISMATCH %s: raw %d op %d -> simd %d, scalar %d\n", label, lhs[k], rhs[k],
                                    out[k], expected);
                    ++mismatches;
                }
            }
        }
    }
    std::printf("  %s: %s (%d mismatches over %d quads)\n", mismatches == 0 ? "PASS" : "FAIL", label, mismatches,
                N * ((N + 3) / 4));
    if (mismatches != 0)
        ++failures;
}

/**
 * @brief Proves SimdFixed4 folds bit-identical to scalar FixedPoint<i32,16>.
 *
 * This is the parity GATE that lets a host SIMD fast path coexist with the
 * kernel's scalar Fixed32 path: for add/sub/mul, every SimdFixed4 lane must
 * equal the scalar Fixed32 result exactly (raw i32 == raw i32) — otherwise the
 * authoritative signatures would diverge between the Linux oracle and the i686
 * kernel. Integer SIMD makes this achievable where float SIMD never could.
 *
 * The SSE2 path is baseline on x86_64 (always active). Compile with the impl:
 *   g++ -std=gnu++23 -I core/include -I math/include \
 *       tests/parity/test_simd_fixed_parity.cpp math/src/Simd.cpp -o /tmp/t
 */
int main()
{
    std::printf("== SimdFixed4 vs scalar Fixed32 parity ==\n");
#if defined(LPL_ARCH_X64)
    std::printf("  (SSE2 vector path)\n\n");
#else
    std::printf("  (scalar fallback path)\n\n");
#endif

    checkOp(
        "add", [](lpl::math::Fixed32 x, lpl::math::Fixed32 y) { return x + y; },
        [](lpl::math::simd::SimdFixed4 x, lpl::math::simd::SimdFixed4 y) { return x + y; });
    checkOp(
        "sub", [](lpl::math::Fixed32 x, lpl::math::Fixed32 y) { return x - y; },
        [](lpl::math::simd::SimdFixed4 x, lpl::math::simd::SimdFixed4 y) { return x - y; });
    checkOp(
        "mul", [](lpl::math::Fixed32 x, lpl::math::Fixed32 y) { return x * y; },
        [](lpl::math::simd::SimdFixed4 x, lpl::math::simd::SimdFixed4 y) { return x * y; });

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
