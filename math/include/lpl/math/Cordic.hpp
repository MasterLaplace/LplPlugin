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
    #define LPL_MATH_CORDIC_HPP

    #include "FixedPoint.hpp"

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
     * @param angle  Angle in Fixed32 radians.
     * @param[out] outSin Result sine value.
     * @param[out] outCos Result cosine value.
     */
    static void sincos(Fixed32 angle, Fixed32 &outSin, Fixed32 &outCos);

    /**
     * @brief Compute the four-quadrant arctangent.
     * @param y Y component.
     * @param x X component.
     * @return atan2(y, x) as Fixed32 radians.
     */
    [[nodiscard]] static Fixed32 atan2(Fixed32 y, Fixed32 x);
};

} // namespace lpl::math

#endif // LPL_MATH_CORDIC_HPP
