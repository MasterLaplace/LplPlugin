/**
 * @file Cordic.hpp
 * @brief CORDIC algorithm for trigonometric functions in fixed-point.
 *
 * Computes sin, cos, and atan2 using only shifts and additions,
 * requiring no FPU.  Designed for deterministic simulation on
 * heterogeneous hardware (RTOS, embedded, FPGA).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_CORDIC_HPP
#    define LPL_MATH_CORDIC_HPP

#    include "FixedPoint.hpp"

namespace lpl::math {

/**
 * @brief CORDIC trigonometric engine operating on Fixed32 values.
 *
 * The lookup table of atan(2^{-i}) values is pre-computed at compile
 * time.  The number of iterations equals FracBits for full precision.
 */
class Cordic final {
public:
    Cordic() = delete;

    /**
     * @brief Compute sine of an angle in radians.
     * @param angle Angle in Fixed32 radians.
     * @return sin(angle) as Fixed32.
     */
    [[nodiscard]] static Fixed32 sin(Fixed32 angle);

    /**
     * @brief Compute cosine of an angle in radians.
     * @param angle Angle in Fixed32 radians.
     * @return cos(angle) as Fixed32.
     */
    [[nodiscard]] static Fixed32 cos(Fixed32 angle);

    /**
     * @brief Simultaneously compute sine and cosine.
     *
     * The angle is first brought into the range the rotation can actually reach. The rotation steers
     * it to zero using a table of atan(2^-i), so the largest angle it can consume is their sum: 1.7433
     * radians, a little under a hundred degrees. Past that it runs out of table and simply stops
     * turning — sin and cos freeze at their values there and never move again.
     *
     * That was live, and in the authoritative path. CharacterController accumulates its heading
     * without bound and resolves the walk direction through here, so a player who turned more than a
     * hundred degrees kept walking in the direction they had at a hundred degrees. It read as the
     * controls going dead rather than as a maths fault, which is why it survived: the first sixty
     * degrees of every turn work perfectly.
     *
     * The reduction is exact and integer, so it costs the determinism contract nothing: it wraps into
     * (-pi, pi] with one remainder, then folds the two outer quadrants into the inner two by half a
     * turn and a sign flip — cos and sin are both odd about a half-turn. What is left is at most
     * pi/2 = 1.5708, inside the table's reach with room to spare.
     *
     * @param angle  Angle in Fixed32 radians.
     * @param[out] outSin Result sine value.
     * @param[out] outCos Result cosine value.
     */
    static void sincos(Fixed32 angle, Fixed32 &outSin, Fixed32 &outCos);

    /**
     * @brief Compute the four-quadrant arctangent.
     *
     * Resolves all four quadrants, in (-pi, pi]. Worth stating because the rotation underneath cannot:
     * vectoring CORDIC accumulates angles from a table of atan(2^-i), so the largest angle it can ever
     * produce is their sum — 1.7433 radians, a shade under a hundred degrees. That is the RIGHT
     * half-plane and nothing else, and for x < 0 the loop simply saturates there.
     *
     * That was live: atan2(0, -1) came back as 1.7433 instead of pi. The function was wrong over half
     * its domain, and wrong in the one way its NAME exists to rule out — resolving all four quadrants
     * is the whole difference between atan2 and atan. It survived because nothing called it; the first
     * caller walked a body due north and it set off west.
     *
     * The fix is the standard one: reflect the input through the origin into the half the rotation can
     * reach, and put the half-turn back afterwards. The sign of that half-turn comes from the ORIGINAL
     * y, so the result stays in (-pi, pi].
     *
     * @param y Y component.
     * @param x X component.
     * @return atan2(y, x) as Fixed32 radians.
     */
    [[nodiscard]] static Fixed32 atan2(Fixed32 y, Fixed32 x);
};

} // namespace lpl::math

#endif // LPL_MATH_CORDIC_HPP
