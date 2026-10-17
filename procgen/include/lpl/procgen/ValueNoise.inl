#ifndef LPL_PROCGEN_VALUENOISE_INL
#define LPL_PROCGEN_VALUENOISE_INL

namespace lpl::procgen {

inline core::u32 ValueNoise2D::hash2(core::i32 x, core::i32 z, core::u32 seed) noexcept
{
    core::u32 h = seed * 0x9E3779B1u;
    h ^= static_cast<core::u32>(x) * 73856093u;
    h = (h << 5) | (h >> 27);
    h ^= static_cast<core::u32>(z) * 19349663u;
    h ^= h >> 13;
    h *= 0x85EBCA6Bu;
    h ^= h >> 16;
    return h;
}

inline ValueNoise2D::Fixed32 ValueNoise2D::latticeValue(core::i32 x, core::i32 z, core::u32 seed) noexcept
{
    const core::u32 h = hash2(x, z, seed);
    const core::i32 raw = static_cast<core::i32>(h & 0x1FFFFu) - 0x10000;
    return Fixed32::fromRaw(raw);
}

inline ValueNoise2D::Fixed32 ValueNoise2D::smoothstep(Fixed32 t) noexcept
{
    const Fixed32 three = Fixed32::fromInt(3);
    const Fixed32 two = Fixed32::fromInt(2);
    return t * t * (three - two * t);
}

inline ValueNoise2D::Fixed32 ValueNoise2D::sample(Fixed32 x, Fixed32 z, core::u32 seed) noexcept
{
    const core::i32 x0 = x.raw() >> 16;
    const core::i32 z0 = z.raw() >> 16;
    const Fixed32 fx = Fixed32::fromRaw(x.raw() & 0xFFFF);
    const Fixed32 fz = Fixed32::fromRaw(z.raw() & 0xFFFF);
    const Fixed32 ux = smoothstep(fx);
    const Fixed32 uz = smoothstep(fz);

    const Fixed32 v00 = latticeValue(x0, z0, seed);
    const Fixed32 v10 = latticeValue(x0 + 1, z0, seed);
    const Fixed32 v01 = latticeValue(x0, z0 + 1, seed);
    const Fixed32 v11 = latticeValue(x0 + 1, z0 + 1, seed);

    const Fixed32 a = lerp(v00, v10, ux);
    const Fixed32 b = lerp(v01, v11, ux);
    return lerp(a, b, uz);
}

inline ValueNoise2D::Fixed32 ValueNoise2D::fbm(Fixed32 x, Fixed32 z, core::u32 octaves, core::u32 seed,
                                               Fixed32 lacunarity, Fixed32 persistence) noexcept
{
    Fixed32 sum = Fixed32::zero();
    Fixed32 norm = Fixed32::zero();
    Fixed32 amp = Fixed32::one();
    Fixed32 freq = Fixed32::one();
    for (core::u32 o = 0; o < octaves; ++o)
    {
        sum = sum + amp * sample(x * freq, z * freq, seed + o);
        norm = norm + amp;
        amp = amp * persistence;
        freq = freq * lacunarity;
        if (freq.raw() > kMaxOctaveFrequencyRaw || amp.raw() == 0)
            break;
    }
    return (norm == Fixed32::zero()) ? Fixed32::zero() : sum / norm;
}

inline ValueNoise2D::Fixed32 ValueNoise2D::ridged(Fixed32 x, Fixed32 z, core::u32 octaves, core::u32 seed,
                                                  Fixed32 lacunarity, Fixed32 persistence) noexcept
{
    Fixed32 sum = Fixed32::zero();
    Fixed32 norm = Fixed32::zero();
    Fixed32 amp = Fixed32::one();
    Fixed32 freq = Fixed32::one();
    for (core::u32 o = 0; o < octaves; ++o)
    {
        const Fixed32 folded = Fixed32::one() - sample(x * freq, z * freq, seed + o).abs();
        sum = sum + amp * (folded * folded);
        norm = norm + amp;
        amp = amp * persistence;
        freq = freq * lacunarity;
        if (freq.raw() > kMaxOctaveFrequencyRaw || amp.raw() == 0)
            break;
    }
    return (norm == Fixed32::zero()) ? Fixed32::zero() : sum / norm;
}

inline ValueNoise2D::Fixed32 ValueNoise2D::billow(Fixed32 x, Fixed32 z, core::u32 octaves, core::u32 seed,
                                                  Fixed32 lacunarity, Fixed32 persistence) noexcept
{
    Fixed32 sum = Fixed32::zero();
    Fixed32 norm = Fixed32::zero();
    Fixed32 amp = Fixed32::one();
    Fixed32 freq = Fixed32::one();
    for (core::u32 o = 0; o < octaves; ++o)
    {
        sum = sum + amp * sample(x * freq, z * freq, seed + o).abs();
        norm = norm + amp;
        amp = amp * persistence;
        freq = freq * lacunarity;
        if (freq.raw() > kMaxOctaveFrequencyRaw || amp.raw() == 0)
            break;
    }
    return (norm == Fixed32::zero()) ? Fixed32::zero() : sum / norm;
}

inline void ValueNoise2D::warp(Fixed32 &x, Fixed32 &z, core::u32 seed, Fixed32 strength, Fixed32 frequency) noexcept
{
    if (strength.raw() == 0)
        return;
    const Fixed32 wx = x * frequency;
    const Fixed32 wz = z * frequency;
    const Fixed32 offsetX = fbm(wx, wz, 3u, seed ^ 0x7F4A7C15u) * strength;
    const Fixed32 offsetZ = fbm(wx, wz, 3u, seed ^ 0x2545F491u) * strength;
    x = x + offsetX;
    z = z + offsetZ;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_VALUENOISE_INL
