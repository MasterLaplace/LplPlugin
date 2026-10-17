/**
 * @file Transform2D.hpp
 * @brief Deterministic 2D affine transform (2x3) in Fixed32 Q16.16.
 *
 * Built on the Fixed32 / CORDIC authority so composition and point mapping are
 * bit-identical across the Linux oracle and the freestanding kernel — scene
 * transforms are authoritative state and must not use float. The matrix is
 *   | a c tx |
 *   | b d ty |
 * applied as x' = a*x + c*y + tx, y' = b*x + d*y + ty.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SCENE_TRANSFORM2D_HPP
#    define LPL_SCENE_TRANSFORM2D_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::scene {

/** @brief A 2x3 affine transform with Fixed32 components. */
struct Transform2D {
    math::Fixed32 a{math::Fixed32::fromInt(1)};
    math::Fixed32 b{math::Fixed32::fromInt(0)};
    math::Fixed32 c{math::Fixed32::fromInt(0)};
    math::Fixed32 d{math::Fixed32::fromInt(1)};
    math::Fixed32 tx{math::Fixed32::fromInt(0)};
    math::Fixed32 ty{math::Fixed32::fromInt(0)};

    [[nodiscard]] static constexpr Transform2D identity() noexcept { return Transform2D{}; }

    /**
     * @brief Pure translation.
     */
    [[nodiscard]] static Transform2D translation(math::Fixed32 x, math::Fixed32 y) noexcept;

    /**
     * @brief Translation + rotation + non-uniform scale (T * R * S).
     * @param x,y Translation.
     * @param angle Rotation in Fixed32 radians (via CORDIC).
     * @param sx,sy Scale factors.
     */
    [[nodiscard]] static Transform2D fromTRS(math::Fixed32 x, math::Fixed32 y, math::Fixed32 angle, math::Fixed32 sx,
                                             math::Fixed32 sy) noexcept;

    /**
     * @brief Compose: returns (*this) applied after @p inner (this * inner).
     */
    [[nodiscard]] Transform2D operator*(const Transform2D &inner) const noexcept;

    /**
     * @brief Map a point through the transform.
     */
    void apply(math::Fixed32 x, math::Fixed32 y, math::Fixed32 &outX, math::Fixed32 &outY) const noexcept;
};

} // namespace lpl::scene

#    include "Transform2D.inl"

#endif // LPL_SCENE_TRANSFORM2D_HPP
