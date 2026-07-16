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

// ── SimdFixed4 (SSE2 integer, deterministic Q16.16) ─────────────────────────
// Pure SSE2 (baseline on x86_64) so it is always active, exactly like
// SimdFloat4. Bit-identical to scalar FixedPoint<i32,16> — validated by
// tests/parity/test_simd_fixed_parity.cpp.

namespace {
// Signed 32x32 -> 64 product of the EVEN 32-bit lanes (0,2), in SSE2.
// pmuldq is SSE4.1; we derive the signed product from the unsigned one
// (pmuludq, SSE2) with the standard two-term sign correction:
//   signed(a*b) = unsigned(a*b) - (a<0 ? b:0)·2^32 - (b<0 ? a:0)·2^32  (mod 2^64)
inline __m128i signedMulEven(__m128i x, __m128i y)
{
    const __m128i up = _mm_mul_epu32(x, y);              // even lanes: unsigned 32x32->64
    const __m128i sx = _mm_srai_epi32(x, 31);            // per-lane 0xFFFFFFFF if x<0
    const __m128i sy = _mm_srai_epi32(y, 31);
    const __m128i cx = _mm_slli_epi64(_mm_and_si128(sx, y), 32); // (x<0 ? y:0) into high dword
    const __m128i cy = _mm_slli_epi64(_mm_and_si128(sy, x), 32); // (y<0 ? x:0) into high dword
    return _mm_sub_epi64(_mm_sub_epi64(up, cx), cy);
}
} // namespace

SimdFixed4 SimdFixed4::load(const core::i32 *raw)
{
    return {_mm_loadu_si128(reinterpret_cast<const __m128i *>(raw))};
}
SimdFixed4 SimdFixed4::splat(core::i32 raw) { return {_mm_set1_epi32(raw)}; }
void SimdFixed4::store(core::i32 *raw) const { _mm_storeu_si128(reinterpret_cast<__m128i *>(raw), reg); }

SimdFixed4 SimdFixed4::operator+(SimdFixed4 rhs) const { return {_mm_add_epi32(reg, rhs.reg)}; }
SimdFixed4 SimdFixed4::operator-(SimdFixed4 rhs) const { return {_mm_sub_epi32(reg, rhs.reg)}; }

SimdFixed4 SimdFixed4::operator*(SimdFixed4 rhs) const
{
    const __m128i bias = _mm_set1_epi64x(1LL << 15); // round-half-up, matches scalar operator*
    // Products for even lanes (0,2) and odd lanes (1,3), then round + >>16.
    __m128i prodE = _mm_srli_epi64(_mm_add_epi64(signedMulEven(reg, rhs.reg), bias), 16);
    __m128i prodO = _mm_srli_epi64(
        _mm_add_epi64(signedMulEven(_mm_srli_si128(reg, 4), _mm_srli_si128(rhs.reg, 4)), bias), 16);
    // Only the low dword of each qword is kept (result fits in i32; there
    // logical and arithmetic 64-bit shift agree). Reinterleave to lane order.
    __m128i e = _mm_shuffle_epi32(prodE, _MM_SHUFFLE(3, 1, 2, 0)); // [E0,E1,..]
    __m128i o = _mm_shuffle_epi32(prodO, _MM_SHUFFLE(3, 1, 2, 0)); // [O0,O1,..]
    return {_mm_unpacklo_epi32(e, o)};                            // [E0,O0,E1,O1] = lanes 0..3
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

// ── SimdFixed4 scalar fallback (deterministic, matches SSE2 path) ────────────

SimdFixed4 SimdFixed4::load(const core::i32 *raw)
{
    return {
        {raw[0], raw[1], raw[2], raw[3]}
    };
}
SimdFixed4 SimdFixed4::splat(core::i32 raw)
{
    return {
        {raw, raw, raw, raw}
    };
}
void SimdFixed4::store(core::i32 *raw) const
{
    raw[0] = data[0];
    raw[1] = data[1];
    raw[2] = data[2];
    raw[3] = data[3];
}

SimdFixed4 SimdFixed4::operator+(SimdFixed4 rhs) const
{
    return {
        {data[0] + rhs.data[0], data[1] + rhs.data[1], data[2] + rhs.data[2], data[3] + rhs.data[3]}
    };
}
SimdFixed4 SimdFixed4::operator-(SimdFixed4 rhs) const
{
    return {
        {data[0] - rhs.data[0], data[1] - rhs.data[1], data[2] - rhs.data[2], data[3] - rhs.data[3]}
    };
}
SimdFixed4 SimdFixed4::operator*(SimdFixed4 rhs) const
{
    SimdFixed4 s;
    for (int i = 0; i < 4; ++i)
    {
        core::i64 w = static_cast<core::i64>(data[i]) * static_cast<core::i64>(rhs.data[i]);
        w += (1LL << 15);
        s.data[i] = static_cast<core::i32>(w >> 16);
    }
    return s;
}

#endif

} // namespace lpl::math::simd
