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

[[nodiscard]] constexpr inline uint64_t part1by2(uint64_t x) noexcept
{
    x &= 0x001fffffULL;
    x = (x | (x << 32)) & 0x001f00000000ffffULL;
    x = (x | (x << 16)) & 0x001f0000ff0000ffULL;
    x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

[[nodiscard]] constexpr inline uint64_t encode3D(uint32_t x, uint32_t y, uint32_t z) noexcept
{
    return (part1by2(z) << 2) | (part1by2(y) << 1) | part1by2(x);
}

} // Morton
