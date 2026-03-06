/**
 * @file Simd.cpp
 * @brief SIMD wrapper implementation — SSE/AVX2, NEON, scalar fallback.
 *
 * On x86_64, SimdFloat4 always uses SSE (baseline).  SimdFloat8 is only
 * available when the translation unit is compiled with -mavx or -mavx2
 * (checked via __AVX__).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/math/Simd.hpp>

namespace lpl::math::simd {

// ============================================================================
//  x86_64 — SSE (SimdFloat4)
// ============================================================================

#if defined(LPL_ARCH_X64)

SimdFloat4 SimdFloat4::load(const float *ptr) { return {_mm_load_ps(ptr)}; }
SimdFloat4 SimdFloat4::splat(float val) { return {_mm_set1_ps(val)}; }
void SimdFloat4::store(float *ptr) const { _mm_store_ps(ptr, reg); }

SimdFloat4 SimdFloat4::operator+(SimdFloat4 rhs) const { return {_mm_add_ps(reg, rhs.reg)}; }
SimdFloat4 SimdFloat4::operator-(SimdFloat4 rhs) const { return {_mm_sub_ps(reg, rhs.reg)}; }
SimdFloat4 SimdFloat4::operator*(SimdFloat4 rhs) const { return {_mm_mul_ps(reg, rhs.reg)}; }

SimdFloat4 SimdFloat4::fma(SimdFloat4 a, SimdFloat4 b, SimdFloat4 c)
{
#    ifdef __FMA__
    return {_mm_fmadd_ps(a.reg, b.reg, c.reg)};
#    else
    return {_mm_add_ps(_mm_mul_ps(a.reg, b.reg), c.reg)};
#    endif
}

// ── SimdFloat8 (AVX/AVX2) ───────────────────────────────────────────────────

#    ifdef __AVX__

SimdFloat8 SimdFloat8::load(const float *ptr) { return {_mm256_load_ps(ptr)}; }
SimdFloat8 SimdFloat8::splat(float val) { return {_mm256_set1_ps(val)}; }
void SimdFloat8::store(float *ptr) const { _mm256_store_ps(ptr, reg); }

SimdFloat8 SimdFloat8::operator+(SimdFloat8 rhs) const { return {_mm256_add_ps(reg, rhs.reg)}; }
SimdFloat8 SimdFloat8::operator-(SimdFloat8 rhs) const { return {_mm256_sub_ps(reg, rhs.reg)}; }
SimdFloat8 SimdFloat8::operator*(SimdFloat8 rhs) const { return {_mm256_mul_ps(reg, rhs.reg)}; }

SimdFloat8 SimdFloat8::fma(SimdFloat8 a, SimdFloat8 b, SimdFloat8 c)
{
#        ifdef __FMA__
    return {_mm256_fmadd_ps(a.reg, b.reg, c.reg)};
#        else
    return {_mm256_add_ps(_mm256_mul_ps(a.reg, b.reg), c.reg)};
#        endif
}

#    endif // __AVX__

// ============================================================================
//  ARM64 — NEON (SimdFloat4 only)
// ============================================================================

#elif defined(LPL_ARCH_ARM64)

SimdFloat4 SimdFloat4::load(const float *ptr) { return {vld1q_f32(ptr)}; }
SimdFloat4 SimdFloat4::splat(float val) { return {vdupq_n_f32(val)}; }
void SimdFloat4::store(float *ptr) const { vst1q_f32(ptr, reg); }

SimdFloat4 SimdFloat4::operator+(SimdFloat4 rhs) const { return {vaddq_f32(reg, rhs.reg)}; }
SimdFloat4 SimdFloat4::operator-(SimdFloat4 rhs) const { return {vsubq_f32(reg, rhs.reg)}; }
SimdFloat4 SimdFloat4::operator*(SimdFloat4 rhs) const { return {vmulq_f32(reg, rhs.reg)}; }

SimdFloat4 SimdFloat4::fma(SimdFloat4 a, SimdFloat4 b, SimdFloat4 c)
{
    return {vfmaq_f32(c.reg, a.reg, b.reg)}; // NEON FMA: c + a*b
}

// ============================================================================
//  Scalar fallback
// ============================================================================

#else

SimdFloat4 SimdFloat4::load(const float *ptr)
{
    return {
        {ptr[0], ptr[1], ptr[2], ptr[3]}
    };
}

SimdFloat4 SimdFloat4::splat(float val)
{
    return {
        {val, val, val, val}
    };
}

void SimdFloat4::store(float *ptr) const
{
    ptr[0] = data[0];
    ptr[1] = data[1];
    ptr[2] = data[2];
    ptr[3] = data[3];
}

SimdFloat4 SimdFloat4::operator+(SimdFloat4 rhs) const
{
    return {
        {data[0] + rhs.data[0], data[1] + rhs.data[1], data[2] + rhs.data[2], data[3] + rhs.data[3]}
    };
}

SimdFloat4 SimdFloat4::operator-(SimdFloat4 rhs) const
{
    return {
        {data[0] - rhs.data[0], data[1] - rhs.data[1], data[2] - rhs.data[2], data[3] - rhs.data[3]}
    };
}

SimdFloat4 SimdFloat4::operator*(SimdFloat4 rhs) const
{
    return {
        {data[0] * rhs.data[0], data[1] * rhs.data[1], data[2] * rhs.data[2], data[3] * rhs.data[3]}
    };
}

SimdFloat4 SimdFloat4::fma(SimdFloat4 a, SimdFloat4 b, SimdFloat4 c)
{
    return {
        {a.data[0] * b.data[0] + c.data[0], a.data[1] * b.data[1] + c.data[1], a.data[2] * b.data[2] + c.data[2],
         a.data[3] * b.data[3] + c.data[3]}
    };
}

#endif

} // namespace lpl::math::simd
