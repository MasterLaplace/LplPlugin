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

using math::Fixed32;

/** @brief A 2x3 affine transform with Fixed32 components. */
struct Transform2D {
    Fixed32 a{Fixed32::fromInt(1)};
    Fixed32 b{Fixed32::fromInt(0)};
    Fixed32 c{Fixed32::fromInt(0)};
    Fixed32 d{Fixed32::fromInt(1)};
    Fixed32 tx{Fixed32::fromInt(0)};
    Fixed32 ty{Fixed32::fromInt(0)};

    [[nodiscard]] static constexpr Transform2D identity() noexcept { return Transform2D{}; }

    /** @brief Pure translation. */
    [[nodiscard]] static Transform2D translation(Fixed32 x, Fixed32 y) noexcept
    {
        Transform2D t;
        t.tx = x;
        t.ty = y;
        return t;
    }

    /**
     * @brief Translation + rotation + non-uniform scale (T * R * S).
     * @param x,y Translation.
     * @param angle Rotation in Fixed32 radians (via CORDIC).
     * @param sx,sy Scale factors.
     */
    [[nodiscard]] static Transform2D fromTRS(Fixed32 x, Fixed32 y, Fixed32 angle, Fixed32 sx, Fixed32 sy) noexcept
    {
        Fixed32 s{Fixed32::fromInt(0)};
        Fixed32 co{Fixed32::fromInt(0)};
        math::Cordic::sincos(angle, s, co);
        Transform2D t;
        t.a = co * sx;
        t.b = s * sx;
        t.c = -(s * sy);
        t.d = co * sy;
        t.tx = x;
        t.ty = y;
        return t;
    }

    /** @brief Compose: returns (*this) applied after @p inner (this * inner). */
    [[nodiscard]] Transform2D operator*(const Transform2D &inner) const noexcept
    {
        Transform2D r;
        r.a = a * inner.a + c * inner.b;
        r.b = b * inner.a + d * inner.b;
        r.c = a * inner.c + c * inner.d;
        r.d = b * inner.c + d * inner.d;
        r.tx = a * inner.tx + c * inner.ty + tx;
        r.ty = b * inner.tx + d * inner.ty + ty;
        return r;
    }

    /** @brief Map a point through the transform. */
    void apply(Fixed32 x, Fixed32 y, Fixed32 &outX, Fixed32 &outY) const noexcept
    {
        outX = a * x + c * y + tx;
        outY = b * x + d * y + ty;
    }
};

} // namespace lpl::scene

#endif // LPL_SCENE_TRANSFORM2D_HPP
