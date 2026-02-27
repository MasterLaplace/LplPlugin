/**
 * @file Cordic.cpp
 * @brief CORDIC implementation with pre-computed atan lookup table.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#include "lpl/math/Cordic.hpp"

#include <array>
#include <cmath>

namespace lpl::math {

namespace {

constexpr core::u32 kIterations = 16;

constexpr auto kAtanTable = [] {
    std::array<core::i32, kIterations> tbl{};
    for (core::u32 i = 0; i < kIterations; ++i)
        tbl[i] = static_cast<core::i32>(std::atan(1.0 / (1 << i)) * Fixed32::kOne);
    return tbl;
}();

constexpr core::i32 kGain = static_cast<core::i32>(0.607252935 * Fixed32::kOne);

} // anonymous namespace

Fixed32 Cordic::sin([[maybe_unused]] Fixed32 angle)
{
    Fixed32 s, c;
    sincos(angle, s, c);
    return s;
}

Fixed32 Cordic::cos([[maybe_unused]] Fixed32 angle)
{
    Fixed32 s, c;
    sincos(angle, s, c);
    return c;
}

void Cordic::sincos(
    [[maybe_unused]] Fixed32 angle,
    [[maybe_unused]] Fixed32 &outSin,
    [[maybe_unused]] Fixed32 &outCos
) {
    core::i32 x = kGain;
    core::i32 y = 0;
    core::i32 z = angle.raw();

    for (core::u32 i = 0; i < kIterations; ++i) {
        core::i32 dx = y >> i;
        core::i32 dy = x >> i;
        if (z >= 0) {
            x -= dx;
            y += dy;
            z -= kAtanTable[i];
        } else {
            x += dx;
            y -= dy;
            z += kAtanTable[i];
        }
    }

    outCos = Fixed32::fromRaw(x);
    outSin = Fixed32::fromRaw(y);
}

Fixed32 Cordic::atan2([[maybe_unused]] Fixed32 yVal, [[maybe_unused]] Fixed32 xVal)
{
    core::i32 x = xVal.raw();
    core::i32 y = yVal.raw();
    core::i32 z = 0;

    for (core::u32 i = 0; i < kIterations; ++i) {
        core::i32 dx = y >> i;
        core::i32 dy = x >> i;
        if (y >= 0) {
            x += dx;
            y -= dy;
            z += kAtanTable[i];
        } else {
            x -= dx;
            y += dy;
            z -= kAtanTable[i];
        }
    }

    return Fixed32::fromRaw(z);
}

} // namespace lpl::math
