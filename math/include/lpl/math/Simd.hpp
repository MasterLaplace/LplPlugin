/**
 * @file Simd.hpp
 * @brief Thin wrappers over AVX2 / SSE4.1 / ARM NEON intrinsics.
 *
 * Provides a portable SimdFloat4 / SimdFloat8 abstraction used by the
 * physics system for vectorised integration loops.  Falls back to scalar
 * emulation when SIMD is unavailable.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_SIMD_HPP
#    define LPL_MATH_SIMD_HPP

#    include <lpl/core/Platform.hpp>
#    include <lpl/core/Types.hpp>

#    if defined(LPL_ARCH_X64) || defined(LPL_ARCH_X86)
#        include <immintrin.h>
#    elif defined(LPL_ARCH_ARM64)
#        include <arm_neon.h>
#    endif

namespace lpl::math::simd {

/**
 * @brief 128-bit SIMD register wrapping 4 Q16.16 fixed-point values (raw i32).
 *
 * DETERMINISTIC: every operation is bit-identical to the scalar
 * FixedPoint<i32,16> arithmetic (add/sub are raw integer add/sub; multiply is
 * @c (a*b + 0x8000) >> 16 with the same round-half-up bias). A host SIMD fast
 * path therefore folds the exact same signature as the kernel's scalar path —
 * unlike float SIMD, whose FMA/rounding may diverge across targets. This is why
 * integer SIMD is *safe* for authoritative state where float SIMD is not.
 *
 * The SSE path needs SSE4.1 (@c pmuldq). When unavailable, the scalar fallback
 * below is used — still deterministic, just not vectorised.
 */
struct SimdFixed4 {
#    if defined(LPL_ARCH_X64)
    __m128i reg; ///< 4 raw Q16.16 lanes. SSE2 baseline — always active, like SimdFloat4.
#    else
    core::i32 data[4]; ///< Scalar fallback (ARM64 / generic).
#    endif

    static SimdFixed4 load(const core::i32 *raw);
    static SimdFixed4 splat(core::i32 raw);
    void store(core::i32 *raw) const;

    SimdFixed4 operator+(SimdFixed4 rhs) const;
    SimdFixed4 operator-(SimdFixed4 rhs) const;
    SimdFixed4 operator*(SimdFixed4 rhs) const;
};

#    if defined(LPL_ARCH_X64)

/**
 * @brief 128-bit SIMD register wrapping 4 floats (SSE).
 */
struct SimdFloat4 {
    __m128 reg;

    static SimdFloat4 load(const float *ptr);
    static SimdFloat4 splat(float val);
    void store(float *ptr) const;

    SimdFloat4 operator+(SimdFloat4 rhs) const;
    SimdFloat4 operator-(SimdFloat4 rhs) const;
    SimdFloat4 operator*(SimdFloat4 rhs) const;

    static SimdFloat4 fma(SimdFloat4 a, SimdFloat4 b, SimdFloat4 c);
};

/**
 * @brief 256-bit SIMD register wrapping 8 floats (AVX/AVX2).
 * @note Only available when compiled with -mavx or -mavx2.
 */
#        ifdef __AVX__
struct SimdFloat8 {
    __m256 reg;

    static SimdFloat8 load(const float *ptr);
    static SimdFloat8 splat(float val);
    void store(float *ptr) const;

    SimdFloat8 operator+(SimdFloat8 rhs) const;
    SimdFloat8 operator-(SimdFloat8 rhs) const;
    SimdFloat8 operator*(SimdFloat8 rhs) const;

    static SimdFloat8 fma(SimdFloat8 a, SimdFloat8 b, SimdFloat8 c);
};
#        endif // __AVX__

#    elif defined(LPL_ARCH_ARM64)

struct SimdFloat4 {
    float32x4_t reg;

    static SimdFloat4 load(const float *ptr);
    static SimdFloat4 splat(float val);
    void store(float *ptr) const;

    SimdFloat4 operator+(SimdFloat4 rhs) const;
    SimdFloat4 operator-(SimdFloat4 rhs) const;
    SimdFloat4 operator*(SimdFloat4 rhs) const;

    static SimdFloat4 fma(SimdFloat4 a, SimdFloat4 b, SimdFloat4 c);
};

#    else

struct SimdFloat4 {
    float data[4];

    static SimdFloat4 load(const float *ptr);
    static SimdFloat4 splat(float val);
    void store(float *ptr) const;

    SimdFloat4 operator+(SimdFloat4 rhs) const;
    SimdFloat4 operator-(SimdFloat4 rhs) const;
    SimdFloat4 operator*(SimdFloat4 rhs) const;

    static SimdFloat4 fma(SimdFloat4 a, SimdFloat4 b, SimdFloat4 c);
};

#    endif

} // namespace lpl::math::simd

#endif // LPL_MATH_SIMD_HPP
