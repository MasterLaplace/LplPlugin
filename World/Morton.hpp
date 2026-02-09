#pragma once
#include <cstdint>

/**
 * @brief Morton encoding/decoding utilities (Z-order curve)
 *
 * All functions are constexpr for compile-time evaluation and
 * marked inline for future __host__ __device__ CUDA compatibility.
 */
namespace Optimizing::World::Morton {

// ============================================================================
// 2D Morton code (32-bit)
// ============================================================================

/// Spread bits of x: insert a 0-bit between each pair of bits.
/// Example: 0b1011 → 0b01_00_01_01
[[nodiscard]] constexpr inline uint32_t part1by1(uint32_t x) noexcept
{
    x &= 0x0000ffffU;
    x = (x | (x << 8)) & 0x00FF00FFU;
    x = (x | (x << 4)) & 0x0F0F0F0FU;
    x = (x | (x << 2)) & 0x33333333U;
    x = (x | (x << 1)) & 0x55555555U;
    return x;
}

/// Compact bits of x: inverse of part1by1.
/// Extracts every other bit and packs them together.
[[nodiscard]] constexpr inline uint32_t compact1by1(uint32_t x) noexcept
{
    x &= 0x55555555U;
    x = (x | (x >> 1)) & 0x33333333U;
    x = (x | (x >> 2)) & 0x0F0F0F0FU;
    x = (x | (x >> 4)) & 0x00FF00FFU;
    x = (x | (x >> 8)) & 0x0000FFFFU;
    return x;
}

[[nodiscard]] constexpr inline uint32_t encode2D(uint32_t x, uint32_t y) noexcept
{
    return (part1by1(y) << 1) | part1by1(x);
}

constexpr inline void decode2D(uint32_t morton, uint32_t &x, uint32_t &y) noexcept
{
    x = compact1by1(morton);
    y = compact1by1(morton >> 1);
}

// ============================================================================
// 3D Morton code (64-bit)
// ============================================================================

/// Spread bits of x: insert two 0-bits between each bit (for 3D interleaving).
/// Supports up to 21 bits of input.
[[nodiscard]] constexpr inline uint64_t part1by2(uint64_t x) noexcept
{
    x &= 0x1fffffULL;
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

/// Compact bits of x: inverse of part1by2.
[[nodiscard]] constexpr inline uint64_t compact1by2(uint64_t x) noexcept
{
    x &= 0x1249249249249249ULL;
    x = (x ^ (x >> 2))  & 0x10c30c30c30c30c3ULL;
    x = (x ^ (x >> 4))  & 0x100f00f00f00f00fULL;
    x = (x ^ (x >> 8))  & 0x1f0000ff0000ffULL;
    x = (x ^ (x >> 16)) & 0x1f00000000ffffULL;
    x = (x ^ (x >> 32)) & 0x1fffffULL;
    return x;
}

[[nodiscard]] constexpr inline uint64_t encode3D(uint64_t x, uint64_t y, uint64_t z) noexcept
{
    return (part1by2(z) << 2) | (part1by2(y) << 1) | part1by2(x);
}

constexpr inline void decode3D(uint64_t code, uint64_t &x, uint64_t &y, uint64_t &z) noexcept
{
    x = compact1by2(code);
    y = compact1by2(code >> 1);
    z = compact1by2(code >> 2);
}

} // namespace Optimizing::World::Morton
