/**
 * @file XorKernel.cpp
 * @brief The two kernels behind lpl::codec::xorRow.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/codec/XorKernel.hpp>

#include <lpl/core/Platform.hpp>

#if defined(LPL_ARCH_X64)
#    include <emmintrin.h>
#endif

namespace lpl::codec {

XorPath activeXorPath() noexcept
{
#if defined(LPL_ARCH_X64)
    return XorPath::Sse2;
#else
    return XorPath::Scalar;
#endif
}

void xorRow(core::u64 *destination, const core::u64 *source, core::u32 words) noexcept
{
    if (destination == nullptr || source == nullptr)
        return;

    core::u32 i = 0u;

#if defined(LPL_ARCH_X64)
    // Two 128-bit lanes per iteration, i.e. four words, so the vector path and the
    // scalar path below consume the same stride. Unaligned loads: the rows ARE
    // aligned (BitMatrix guarantees it), but a droplet payload handed in by net/ or
    // pack/ is whatever the caller had, and on every processor this module targets
    // an unaligned load of an aligned address costs the same as an aligned one.
    for (; i + 4u <= words; i += 4u)
    {
        __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(destination + i));
        __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(destination + i + 2u));
        const __m128i sourceLo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + i));
        const __m128i sourceHi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + i + 2u));
        lo = _mm_xor_si128(lo, sourceLo);
        hi = _mm_xor_si128(hi, sourceHi);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(destination + i), lo);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(destination + i + 2u), hi);
    }
#else
    // Four independent XORs rather than one: a dependent chain stalls on the load,
    // and the ports are there whether or not they are used.
    for (; i + 4u <= words; i += 4u)
    {
        destination[i] ^= source[i];
        destination[i + 1u] ^= source[i + 1u];
        destination[i + 2u] ^= source[i + 2u];
        destination[i + 3u] ^= source[i + 3u];
    }
#endif

    for (; i < words; ++i)
        destination[i] ^= source[i];
}

void xorRowInto(core::u64 *destination, const core::u64 *a, const core::u64 *b, core::u32 words) noexcept
{
    if (destination == nullptr || a == nullptr || b == nullptr)
        return;

    core::u32 i = 0u;

#if defined(LPL_ARCH_X64)
    for (; i + 4u <= words; i += 4u)
    {
        const __m128i lo = _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i)),
                                         _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i)));
        const __m128i hi = _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i + 2u)),
                                         _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i + 2u)));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(destination + i), lo);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(destination + i + 2u), hi);
    }
#else
    for (; i + 4u <= words; i += 4u)
    {
        destination[i] = a[i] ^ b[i];
        destination[i + 1u] = a[i + 1u] ^ b[i + 1u];
        destination[i + 2u] = a[i + 2u] ^ b[i + 2u];
        destination[i + 3u] = a[i + 3u] ^ b[i + 3u];
    }
#endif

    for (; i < words; ++i)
        destination[i] = a[i] ^ b[i];
}

bool rowIsZero(const core::u64 *row, core::u32 words) noexcept
{
    if (row == nullptr)
        return true;
    core::u64 accumulator = 0u;
    for (core::u32 i = 0u; i < words; ++i)
        accumulator |= row[i];
    return accumulator == 0u;
}

core::u32 firstNonZeroWord(const core::u64 *row, core::u32 words) noexcept
{
    if (row == nullptr)
        return words;
    for (core::u32 i = 0u; i < words; ++i)
        if (row[i] != 0u)
            return i;
    return words;
}

} // namespace lpl::codec
