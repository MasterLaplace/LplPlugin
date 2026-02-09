#pragma once

#include <cstdint>

namespace Morton {

[[nodiscard]] constexpr inline uint32_t part1by1(uint32_t x) noexcept
{
    x &= 0x0000ffffU;
    x = (x | (x << 8u)) & 0x00FF00FFU;
    x = (x | (x << 4u)) & 0x0F0F0F0FU;
    x = (x | (x << 2u)) & 0x33333333U;
    x = (x | (x << 1u)) & 0x55555555U;
    return x;
}

[[nodiscard]] constexpr inline uint32_t encode2D(uint32_t x, uint32_t y) noexcept
{
    return (part1by1(y) << 1) | part1by1(x);
}

} // Morton
