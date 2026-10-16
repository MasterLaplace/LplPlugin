#include <lpl/math/Simd.hpp>

namespace lpl::math::simd {

#if defined(LPL_ARCH_X64)

namespace {

/**
 * @brief The signed 32x32 -> 64 products of the EVEN 32-bit lanes (0, 2), in SSE2.
 *
 * pmuldq, the signed multiply, is SSE4.1. The signed product is derived from the unsigned one
 * (pmuludq, SSE2) with the standard two-term sign correction:
 * signed(a*b) = unsigned(a*b) - (a<0 ? b:0)·2^32 - (b<0 ? a:0)·2^32 (mod 2^64). An arithmetic
 * shift by 31 makes each lane all ones when it is negative, which selects the other factor, and
 * that factor is shifted into the high dword before it is subtracted.
 *
 * @param x The first factors.
 * @param y The second factors.
 * @return The two 64-bit products.
 */
inline __m128i signedMulEven(__m128i x, __m128i y)
{
    const __m128i up = _mm_mul_epu32(x, y);
    const __m128i sx = _mm_srai_epi32(x, 31);
    const __m128i sy = _mm_srai_epi32(y, 31);
    const __m128i cx = _mm_slli_epi64(_mm_and_si128(sx, y), 32);
    const __m128i cy = _mm_slli_epi64(_mm_and_si128(sy, x), 32);
    return _mm_sub_epi64(_mm_sub_epi64(up, cx), cy);
}

} // namespace

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

SimdFixed4 SimdFixed4::load(const core::i32 *raw) { return {_mm_loadu_si128(reinterpret_cast<const __m128i *>(raw))}; }
SimdFixed4 SimdFixed4::splat(core::i32 raw) { return {_mm_set1_epi32(raw)}; }
void SimdFixed4::store(core::i32 *raw) const { _mm_storeu_si128(reinterpret_cast<__m128i *>(raw), reg); }

SimdFixed4 SimdFixed4::operator+(SimdFixed4 rhs) const { return {_mm_add_epi32(reg, rhs.reg)}; }
SimdFixed4 SimdFixed4::operator-(SimdFixed4 rhs) const { return {_mm_sub_epi32(reg, rhs.reg)}; }

SimdFixed4 SimdFixed4::operator*(SimdFixed4 rhs) const
{
    const __m128i bias = _mm_set1_epi64x(1LL << 15);
    __m128i prodE = _mm_srli_epi64(_mm_add_epi64(signedMulEven(reg, rhs.reg), bias), 16);
    __m128i prodO =
        _mm_srli_epi64(_mm_add_epi64(signedMulEven(_mm_srli_si128(reg, 4), _mm_srli_si128(rhs.reg, 4)), bias), 16);
    __m128i e = _mm_shuffle_epi32(prodE, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i o = _mm_shuffle_epi32(prodO, _MM_SHUFFLE(3, 1, 2, 0));
    return {_mm_unpacklo_epi32(e, o)};
}

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

#elif defined(LPL_ARCH_ARM64)

SimdFloat4 SimdFloat4::load(const float *ptr) { return {vld1q_f32(ptr)}; }
SimdFloat4 SimdFloat4::splat(float val) { return {vdupq_n_f32(val)}; }
void SimdFloat4::store(float *ptr) const { vst1q_f32(ptr, reg); }

SimdFloat4 SimdFloat4::operator+(SimdFloat4 rhs) const { return {vaddq_f32(reg, rhs.reg)}; }
SimdFloat4 SimdFloat4::operator-(SimdFloat4 rhs) const { return {vsubq_f32(reg, rhs.reg)}; }
SimdFloat4 SimdFloat4::operator*(SimdFloat4 rhs) const { return {vmulq_f32(reg, rhs.reg)}; }

SimdFloat4 SimdFloat4::fma(SimdFloat4 a, SimdFloat4 b, SimdFloat4 c) { return {vfmaq_f32(c.reg, a.reg, b.reg)}; }

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
