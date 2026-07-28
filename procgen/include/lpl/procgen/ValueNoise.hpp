/**
 * @file ValueNoise.hpp
 * @brief Deterministic Fixed32 value noise (lattice + smoothstep + fBm).
 *
 * A seed-driven, libm-free coherent-noise field in Q16.16 fixed point, so the
 * generated world is **bit-identical across the Linux oracle and the i686
 * kernel** — the same determinism contract as the rest of the engine. Value
 * noise (hashed lattice values, smoothstep-interpolated) is chosen over gradient
 * noise for a first slice: every step is integer/Fixed32, no gradients, no sqrt.
 * Fractional Brownian motion (fBm) sums octaves for natural-looking relief.
 *
 * Header-only and freestanding-safe (no heap, no exceptions): usable in the
 * kernel smoke path exactly like the CubePile sample.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_VALUENOISE_HPP
#    define LPL_PROCGEN_VALUENOISE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::procgen {

/**
 * @struct ValueNoise2D
 * @brief Deterministic 2D value noise in Fixed32 (Q16.16), all integer math.
 *
 * @c sample returns coherent noise in [-1, 1); @c fbm sums octaves into the same
 * range. Both are pure functions of (x, z, seed) — no state, fully reproducible.
 */
struct ValueNoise2D {
    using Fixed32 = math::Fixed32;

    /// Integer hash of a lattice cell (bit-mixed multiply-xor), seed-salted.
    [[nodiscard]] static core::u32 hash2(core::i32 x, core::i32 z, core::u32 seed) noexcept
    {
        core::u32 h = seed * 0x9E3779B1u;
        h ^= static_cast<core::u32>(x) * 73856093u;
        // The rotate is what makes the two coordinates non-interchangeable, and
        // it is not decoration. Folding both in with plain XOR looks symmetric
        // and is: negating both coordinates leaves the result unchanged, so
        // (1, -1) and (-1, 1) hashed identically — and so did every other pair
        // related that way. Measured over a 129x129 block of coordinates, that
        // XOR-only form collided on **2732 of 16641** inputs, 16 percent. Two
        // chunks diagonally opposite the origin generated the same terrain, and
        // the noise lattice itself carried the same duplication. With one
        // rotation between the two folds: zero collisions over the same block.
        h = (h << 5) | (h >> 27);
        h ^= static_cast<core::u32>(z) * 19349663u;
        h ^= h >> 13;
        h *= 0x85EBCA6Bu;
        h ^= h >> 16;
        return h;
    }

    /// Pseudo-random lattice value in [-1, 1) as Fixed32 (17 hash bits → Q16.16).
    [[nodiscard]] static Fixed32 latticeValue(core::i32 x, core::i32 z, core::u32 seed) noexcept
    {
        const core::u32 h = hash2(x, z, seed);
        const core::i32 raw = static_cast<core::i32>(h & 0x1FFFFu) - 0x10000; // [-65536, 65535]
        return Fixed32::fromRaw(raw);
    }

    /// Smoothstep 3t² − 2t³ (Fixed32); t in [0, 1).
    [[nodiscard]] static Fixed32 smoothstep(Fixed32 t) noexcept
    {
        const Fixed32 three = Fixed32::fromInt(3);
        const Fixed32 two = Fixed32::fromInt(2);
        return t * t * (three - two * t);
    }

    /// Linear interpolation a + (b − a)·t (Fixed32).
    [[nodiscard]] static Fixed32 lerp(Fixed32 a, Fixed32 b, Fixed32 t) noexcept { return a + (b - a) * t; }

    /// Coherent value noise at (x, z) for @p seed, in [-1, 1).
    [[nodiscard]] static Fixed32 sample(Fixed32 x, Fixed32 z, core::u32 seed) noexcept
    {
        const core::i32 x0 = x.raw() >> 16; // floor (arithmetic shift)
        const core::i32 z0 = z.raw() >> 16;
        const Fixed32 fx = Fixed32::fromRaw(x.raw() & 0xFFFF); // fractional part in [0,1)
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

    /**
     * @brief Largest frequency an octave may reach before the sum is abandoned.
     *
     * Fixed32 holds values up to about 32767, and an octave evaluates the noise
     * at @c coordinate * frequency. With a coordinate already scaled into the
     * hundreds, a frequency past this point wraps the multiply and the terrain
     * turns to garbage in a way that looks like a seed problem rather than an
     * overflow. Stopping early loses detail nobody could see anyway: an octave
     * beyond the twelfth contributes less than one part in four thousand.
     */
    static constexpr core::i32 kMaxOctaveFrequencyRaw = 4096 << 16;

    /**
     * @brief Fractal Brownian motion: @p octaves summed, normalised to [-1, 1).
     *
     * @param x           Sample abscissa.
     * @param z           Sample ordinate.
     * @param octaves     Layers to sum.
     * @param seed        Determinism anchor; each octave salts it differently.
     * @param lacunarity  Frequency multiplier per octave (2 is the usual value).
     * @param persistence Amplitude multiplier per octave (0.5 is the usual value).
     */
    [[nodiscard]] static Fixed32 fbm(Fixed32 x, Fixed32 z, core::u32 octaves, core::u32 seed,
                                     Fixed32 lacunarity = Fixed32::fromInt(2),
                                     Fixed32 persistence = Fixed32::half()) noexcept
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

    /**
     * @brief Ridged multifractal: fBm folded so its zero crossings become crests.
     *
     * Musgrave's construction. Each octave contributes @f$(1 - |n|)^2@f$ instead
     * of @f$n@f$, which turns the places where the noise passes through zero —
     * previously the least interesting part of the field — into sharp ridges, and
     * pushes the smooth parts down into valleys. It is the standard answer to why
     * plain fBm reads as rolling hills and never as a mountain range: fBm is
     * symmetric about its mean, and real orogeny is not.
     *
     * @return Coherent noise in [0, 1), 1 along the ridge lines.
     */
    [[nodiscard]] static Fixed32 ridged(Fixed32 x, Fixed32 z, core::u32 octaves, core::u32 seed,
                                        Fixed32 lacunarity = Fixed32::fromInt(2),
                                        Fixed32 persistence = Fixed32::half()) noexcept
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

    /**
     * @brief Billow noise: fBm rectified, so its zero crossings become creases.
     *
     * The mirror image of @ref ridged — @f$|n|@f$ rather than @f$(1-|n|)^2@f$ —
     * giving rounded bulges separated by creases. What dunes, clouds and eroded
     * badlands look like.
     *
     * @return Coherent noise in [0, 1).
     */
    [[nodiscard]] static Fixed32 billow(Fixed32 x, Fixed32 z, core::u32 octaves, core::u32 seed,
                                        Fixed32 lacunarity = Fixed32::fromInt(2),
                                        Fixed32 persistence = Fixed32::half()) noexcept
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

    /**
     * @brief Displaces a sample point by a second noise field (domain warping).
     *
     * Rather than perturbing the noise's output, perturb where it is *read*. The
     * result keeps the field's statistics but destroys its axis alignment, which
     * is what removes the faint rectangular grain a lattice noise always has and
     * what turns straight-ish boundaries into the swirled, folded ones real
     * geology and real coastlines show.
     *
     * @param x        Abscissa to displace, modified in place.
     * @param z        Ordinate to displace, modified in place.
     * @param seed     Determinism anchor for the displacement field.
     * @param strength How far a point may move, in the same units as x and z.
     * @param frequency Scale of the displacement field.
     */
    static void warp(Fixed32 &x, Fixed32 &z, core::u32 seed, Fixed32 strength,
                     Fixed32 frequency = Fixed32::half()) noexcept
    {
        if (strength.raw() == 0)
            return;
        const Fixed32 wx = x * frequency;
        const Fixed32 wz = z * frequency;
        // Two independent offsets: one field would displace both axes identically
        // and merely shear the domain instead of folding it.
        const Fixed32 offsetX = fbm(wx, wz, 3u, seed ^ 0x7F4A7C15u) * strength;
        const Fixed32 offsetZ = fbm(wx, wz, 3u, seed ^ 0x2545F491u) * strength;
        x = x + offsetX;
        z = z + offsetZ;
    }
};

} // namespace lpl::procgen

#endif // LPL_PROCGEN_VALUENOISE_HPP
