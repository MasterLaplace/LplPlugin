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
 * Header-only and freestanding-safe (no heap, no exceptions): usable in ring 0
 * exactly like the CubePile sample.
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

    /**
     * @brief Integer hash of a lattice cell (bit-mixed multiply-xor), seed-salted.
     *
     * The rotate between the two coordinate folds is what makes the two
     * coordinates non-interchangeable, and it is not decoration. Folding both in
     * with plain XOR looks symmetric and is: negating both coordinates leaves the
     * result unchanged, so (1, -1) and (-1, 1) hashed identically — and so did
     * every other pair related that way. Measured over a 129x129 block of
     * coordinates, that XOR-only form collided on **2732 of 16641** inputs, 16
     * percent. Two chunks diagonally opposite the origin generated the same
     * terrain, and the noise lattice itself carried the same duplication. With one
     * rotation between the two folds: zero collisions over the same block.
     */
    [[nodiscard]] static core::u32 hash2(core::i32 x, core::i32 z, core::u32 seed) noexcept;

    /**
     * @brief Pseudo-random lattice value in [-1, 1) as Fixed32 (17 hash bits → Q16.16).
     *
     * The low 17 bits of the hash, offset to the raw range [-65536, 65535].
     */
    [[nodiscard]] static Fixed32 latticeValue(core::i32 x, core::i32 z, core::u32 seed) noexcept;

    /**
     * @brief Smoothstep 3t² − 2t³ (Fixed32); t in [0, 1).
     */
    [[nodiscard]] static Fixed32 smoothstep(Fixed32 t) noexcept;

    /**
     * @brief Linear interpolation a + (b − a)·t (Fixed32).
     */
    [[nodiscard]] static Fixed32 lerp(Fixed32 a, Fixed32 b, Fixed32 t) noexcept { return a + (b - a) * t; }

    /**
     * @brief Coherent value noise at (x, z) for @p seed, in [-1, 1).
     *
     * The lattice cell is the floor of each coordinate, taken by an arithmetic
     * shift of the raw value, and the fraction in [0, 1) is its low 16 bits.
     */
    [[nodiscard]] static Fixed32 sample(Fixed32 x, Fixed32 z, core::u32 seed) noexcept;

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
                                     Fixed32 persistence = Fixed32::half()) noexcept;

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
                                        Fixed32 persistence = Fixed32::half()) noexcept;

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
                                        Fixed32 persistence = Fixed32::half()) noexcept;

    /**
     * @brief Displaces a sample point by a second noise field (domain warping).
     *
     * Rather than perturbing the noise's output, perturb where it is *read*. The
     * result keeps the field's statistics but destroys its axis alignment, which
     * is what removes the faint rectangular grain a lattice noise always has and
     * what turns straight-ish boundaries into the swirled, folded ones real
     * geology and real coastlines show.
     *
     * Two independent offsets: one field would displace both axes identically
     * and merely shear the domain instead of folding it.
     *
     * @param x        Abscissa to displace, modified in place.
     * @param z        Ordinate to displace, modified in place.
     * @param seed     Determinism anchor for the displacement field.
     * @param strength How far a point may move, in the same units as x and z.
     * @param frequency Scale of the displacement field.
     */
    static void warp(Fixed32 &x, Fixed32 &z, core::u32 seed, Fixed32 strength,
                     Fixed32 frequency = Fixed32::half()) noexcept;
};

} // namespace lpl::procgen

#    include "ValueNoise.inl"

#endif // LPL_PROCGEN_VALUENOISE_HPP
