/**
 * @file RenderParity.hpp
 * @brief The FNV-1a step and the Q16.16 quantization every render signature folds with.
 *
 * The render folds compared between the host and the i686 kernel (topology, ray tracer, command
 * buffer, foveation, rasterizer, the cube-pile sample) share these constants, so a signature of
 * one is comparable with a signature of another, and the seed and the prime live in one place.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_RENDERPARITY_HPP
#    define LPL_RENDER_RENDERPARITY_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::render {

namespace detail {

/// FNV-1a 32-bit parameters, shared by every cross-target signature fold so the
/// seed/prime live in exactly one place (see Topology/RayTracer/Pbr/CommandBuffer/
/// Foveated/SoftwareRasterizer). Changing these rebaselines all parity oracles.
inline constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
inline constexpr core::u32 kFnv1aPrime = 0x01000193u;

/// Q16.16 quantization scale used to fold non-authoritative float results into a
/// cross-target signature (multiply, truncate to i32, then fnv1aStep). One place
/// for the "quantize a float to Q16.16 before hashing" idiom shared by the render
/// signature folds. (Distinct from integer Q16 weight arithmetic in Texture.hpp.)
inline constexpr core::f32 kQ16FoldScale = 65536.0f;

[[nodiscard]] inline core::u32 fnv1aStep(core::u32 hash, core::u32 value) noexcept
{
    for (core::u32 i = 0; i < 4u; ++i)
    {
        hash ^= (value >> (i * 8u)) & 0xFFu;
        hash *= kFnv1aPrime;
    }
    return hash;
}

} // namespace detail

} // namespace lpl::render

#endif // LPL_RENDER_RENDERPARITY_HPP
