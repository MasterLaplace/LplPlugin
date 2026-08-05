/**
 * @file GaloisField.hpp
 * @brief Arithmetic over GF(2) and GF(256).
 *
 * On GF(2) addition is XOR and multiplication is AND, which is why a 512-bit
 * vector register performs 512 additions in one instruction. GF(256) carries the
 * Reed-Solomon symbols; its log/antilog tables are constexpr so no runtime
 * initialisation can differ between the oracle and the kernel.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CODEC_GALOISFIELD_HPP
#    define LPL_LPL_CODEC_GALOISFIELD_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::codec {

/**
 * @brief GF(2) addition: exclusive or.
 *
 * Named rather than written inline at each site, because the whole module rests on
 * the identity and a reader should meet it once with its name on it: over the field
 * with two elements, adding is XOR and subtracting is the same operation, which is
 * why every decode below is symmetric and why undoing a combination costs exactly
 * what making it cost.
 */
[[nodiscard]] constexpr core::u64 gf2Add(core::u64 a, core::u64 b) noexcept { return a ^ b; }

/**
 * @brief GF(2) multiplication: conjunction. One instruction over sixty-four coefficients.
 */
[[nodiscard]] constexpr core::u64 gf2Mul(core::u64 a, core::u64 b) noexcept { return a & b; }

/**
 * @brief The primitive polynomial of GF(256): x^8 + x^4 + x^3 + x^2 + 1.
 *
 * 0x11D, the one Reed-Solomon implementations have agreed on since the CD. The
 * choice is arbitrary in the algebra and is NOT arbitrary on the wire: a decoder
 * using a different modulus reads a different field, so this constant is part of
 * the format the way the FNV offset basis is.
 */
inline constexpr core::u32 kGf256Modulus = 0x11Du;

/**
 * @struct Gf256Tables
 * @brief Antilog and log of the field, generated at compile time.
 *
 * The antilog table is doubled — 512 entries rather than 255 — so a multiplication
 * can add two logs without a conditional subtraction of the field order. A branch
 * per symbol is not a rounding error in a decoder that runs one per byte.
 */
struct Gf256Tables {
    core::u8 exp[512]{}; ///< exp[i] = generator^i, repeated once.
    core::u8 log[256]{}; ///< log[x] = i such that generator^i == x; log[0] is unused.
};

/**
 * @brief Builds the field. constexpr, so the kernel image carries the table rather than code.
 */
[[nodiscard]] constexpr Gf256Tables makeGf256Tables() noexcept
{
    Gf256Tables tables{};
    core::u32 value = 1u;
    for (core::u32 i = 0u; i < 255u; ++i)
    {
        tables.exp[i] = static_cast<core::u8>(value);
        tables.log[value] = static_cast<core::u8>(i);
        value <<= 1;
        if ((value & 0x100u) != 0u)
            value ^= kGf256Modulus;
    }
    for (core::u32 i = 255u; i < 512u; ++i)
        tables.exp[i] = tables.exp[i - 255u];
    return tables;
}

/// The field itself. One instance, read by the oracle and by ring 0.
inline constexpr Gf256Tables kGf256 = makeGf256Tables();

/**
 * @brief GF(256) addition is still XOR: the characteristic is two whatever the extension.
 */
[[nodiscard]] constexpr core::u8 gf256Add(core::u8 a, core::u8 b) noexcept { return static_cast<core::u8>(a ^ b); }

/**
 * @brief GF(256) multiplication by logarithms.
 */
[[nodiscard]] constexpr core::u8 gf256Mul(core::u8 a, core::u8 b) noexcept
{
    if (a == 0u || b == 0u)
        return 0u;
    return kGf256.exp[static_cast<core::u32>(kGf256.log[a]) + static_cast<core::u32>(kGf256.log[b])];
}

/**
 * @brief GF(256) division. Dividing by zero yields zero rather than trapping: the
 *        callers feed it syndromes, where a zero divisor means the error locator
 *        already said there is nothing there.
 */
[[nodiscard]] constexpr core::u8 gf256Div(core::u8 a, core::u8 b) noexcept
{
    if (a == 0u || b == 0u)
        return 0u;
    return kGf256.exp[static_cast<core::u32>(kGf256.log[a]) + 255u - static_cast<core::u32>(kGf256.log[b])];
}

/**
 * @brief Multiplicative inverse; zero maps to zero for the reason gf256Div does.
 */
[[nodiscard]] constexpr core::u8 gf256Inv(core::u8 a) noexcept
{
    if (a == 0u)
        return 0u;
    return kGf256.exp[255u - static_cast<core::u32>(kGf256.log[a])];
}

/**
 * @brief generator^exponent, the root evaluation Chien search walks.
 */
[[nodiscard]] constexpr core::u8 gf256Pow(core::u32 exponent) noexcept { return kGf256.exp[exponent % 255u]; }

/**
 * @brief @p base raised to @p exponent, by repeated logarithm rather than squaring.
 */
[[nodiscard]] constexpr core::u8 gf256PowOf(core::u8 base, core::u32 exponent) noexcept
{
    if (base == 0u)
        return exponent == 0u ? 1u : 0u;
    return kGf256.exp[(static_cast<core::u32>(kGf256.log[base]) * exponent) % 255u];
}

/**
 * @brief Evaluates a polynomial at @p x by Horner's rule.
 *
 * Coefficients ASCEND: index i is the coefficient of x^i. That is the order
 * Berlekamp-Massey produces its locator in and the order the generator polynomial is
 * built in, so it is the order the whole module uses. One convention, because a
 * polynomial read backwards evaluates to a different value and nothing says so.
 *
 * @param coefficients Lowest degree first.
 * @param count        Number of coefficients.
 * @param x            Point of evaluation.
 * @return The value at @p x.
 */
[[nodiscard]] constexpr core::u8 gf256Evaluate(const core::u8 *coefficients, core::u32 count, core::u8 x) noexcept
{
    core::u8 accumulator = 0u;
    for (core::u32 i = count; i > 0u; --i)
        accumulator = gf256Add(gf256Mul(accumulator, x), coefficients[i - 1u]);
    return accumulator;
}

} // namespace lpl::codec

#endif // LPL_LPL_CODEC_GALOISFIELD_HPP
