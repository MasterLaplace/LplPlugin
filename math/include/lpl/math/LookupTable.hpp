/**
 * @file LookupTable.hpp
 * @brief Compile-time generated trigonometric lookup tables.
 *
 * Provides pre-computed sin/cos tables with linear interpolation,
 * as an alternative to CORDIC when cache residency is guaranteed.
 * Lookup is ~3 cycles when the table fits in L1.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_LOOKUP_TABLE_HPP
    #define LPL_MATH_LOOKUP_TABLE_HPP

    #include "FixedPoint.hpp"

    #include <array>
    #include <cmath>

namespace lpl::math {

/**
 * @brief Fixed-size trigonometric LUT with linear interpolation.
 * @tparam Size Number of entries covering [0, 2*pi).
 */
template <core::u32 Size = 4096>
class LookupTable final {
public:
    LookupTable() = delete;

    /**
     * @brief Lookup sin(angle) via table + lerp.
     * @param angle Angle in Fixed32 radians.
     * @return Approximated sin value as Fixed32.
     */
    [[nodiscard]] static Fixed32 sin(Fixed32 angle);

    /**
     * @brief Lookup cos(angle) via table + lerp.
     * @param angle Angle in Fixed32 radians.
     * @return Approximated cos value as Fixed32.
     */
    [[nodiscard]] static Fixed32 cos(Fixed32 angle);

private:
    static constexpr auto kTable = [] {
        std::array<core::i32, Size> tbl{};
        for (core::u32 i = 0; i < Size; ++i) {
            double angle = (static_cast<double>(i) / Size) * 2.0 * 3.14159265358979323846;
            tbl[i] = static_cast<core::i32>(std::sin(angle) * Fixed32::kOne);
        }
        return tbl;
    }();
};

} // namespace lpl::math

#endif // LPL_MATH_LOOKUP_TABLE_HPP
