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

    /// Fractal Brownian motion: @p octaves of value noise, normalised to [-1, 1).
    /// Each octave doubles frequency and halves amplitude (persistence 0.5).
    [[nodiscard]] static Fixed32 fbm(Fixed32 x, Fixed32 z, core::u32 octaves, core::u32 seed) noexcept
    {
        const Fixed32 two = Fixed32::fromInt(2);
        Fixed32 sum = Fixed32::zero();
        Fixed32 norm = Fixed32::zero();
        Fixed32 amp = Fixed32::one();
        Fixed32 freq = Fixed32::one();
        for (core::u32 o = 0; o < octaves; ++o)
        {
            sum = sum + amp * sample(x * freq, z * freq, seed + o);
            norm = norm + amp;
            amp = amp * Fixed32::half();
            freq = freq * two;
        }
        return (norm == Fixed32::zero()) ? Fixed32::zero() : sum / norm;
    }
};

} // namespace lpl::procgen

#endif // LPL_PROCGEN_VALUENOISE_HPP
