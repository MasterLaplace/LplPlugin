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
    #define LPL_MATH_SIMD_HPP

    #include <lpl/core/Platform.hpp>
    #include <lpl/core/Types.hpp>

    #if defined(LPL_ARCH_X64) || defined(LPL_ARCH_X86)
        #include <immintrin.h>
    #elif defined(LPL_ARCH_ARM64)
        #include <arm_neon.h>
    #endif

namespace lpl::math::simd {

#if defined(LPL_ARCH_X64)

/**
 * @brief 128-bit SIMD register wrapping 4 floats (SSE).
 */
struct SimdFloat4 {
    __m128 reg;

    static SimdFloat4 load(const float *ptr);
    static SimdFloat4 splat(float val);
    void              store(float *ptr) const;

    SimdFloat4 operator+(SimdFloat4 rhs) const;
    SimdFloat4 operator-(SimdFloat4 rhs) const;
    SimdFloat4 operator*(SimdFloat4 rhs) const;

    static SimdFloat4 fma(SimdFloat4 a, SimdFloat4 b, SimdFloat4 c);
};

/**
 * @brief 256-bit SIMD register wrapping 8 floats (AVX2).
 */
struct SimdFloat8 {
    __m256 reg;

    static SimdFloat8 load(const float *ptr);
    static SimdFloat8 splat(float val);
    void              store(float *ptr) const;

    SimdFloat8 operator+(SimdFloat8 rhs) const;
    SimdFloat8 operator-(SimdFloat8 rhs) const;
    SimdFloat8 operator*(SimdFloat8 rhs) const;

    static SimdFloat8 fma(SimdFloat8 a, SimdFloat8 b, SimdFloat8 c);
};

#elif defined(LPL_ARCH_ARM64)

struct SimdFloat4 {
    float32x4_t reg;

    static SimdFloat4 load(const float *ptr);
    static SimdFloat4 splat(float val);
    void              store(float *ptr) const;

    SimdFloat4 operator+(SimdFloat4 rhs) const;
    SimdFloat4 operator-(SimdFloat4 rhs) const;
    SimdFloat4 operator*(SimdFloat4 rhs) const;

    static SimdFloat4 fma(SimdFloat4 a, SimdFloat4 b, SimdFloat4 c);
};

#else

struct SimdFloat4 {
    float data[4];

    static SimdFloat4 load(const float *ptr);
    static SimdFloat4 splat(float val);
    void              store(float *ptr) const;

    SimdFloat4 operator+(SimdFloat4 rhs) const;
    SimdFloat4 operator-(SimdFloat4 rhs) const;
    SimdFloat4 operator*(SimdFloat4 rhs) const;

    static SimdFloat4 fma(SimdFloat4 a, SimdFloat4 b, SimdFloat4 c);
};

#endif

} // namespace lpl::math::simd

#endif // LPL_MATH_SIMD_HPP
