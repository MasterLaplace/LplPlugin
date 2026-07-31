/**
 * @file MipTexture.hpp
 * @brief A texture with its mip chain, and trilinear sampling by level of detail.
 *
 * A texture without mips is not merely lower quality at a distance, it is WRONG at
 * a distance: one pixel covers many texels and the sampler answers for one of
 * them, so which texel wins changes as the camera moves and the surface crawls
 * with noise. On a terrain covered in grain that reads as static, and no amount of
 * bilinear filtering fixes it — bilinear averages four texels when the pixel
 * covers four hundred.
 *
 * The chain is built once by box-filtering halves, and the level is chosen from
 * how much world distance one pixel covers. Both halves matter: mips with a level
 * that never changes are decoration.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_MIP_TEXTURE_HPP
#    define LPL_RENDER_MIP_TEXTURE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/render/Texture.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::render {

/**
 * @class MipTexture
 * @brief One texture plus its successively halved copies.
 */
class MipTexture {
public:
    MipTexture() = default;

    /** @brief Builds the chain down to (or towards) a 1x1 level. */
    explicit MipTexture(const Texture &base, core::u32 maxLevels = 6u)
    {
        _levels.push_back(base);
        while (_levels.size() < maxLevels &&
               (_levels[_levels.size() - 1u].width() > 1u || _levels[_levels.size() - 1u].height() > 1u))
            _levels.push_back(_levels[_levels.size() - 1u].halved());
    }

    [[nodiscard]] core::u32 levelCount() const noexcept { return static_cast<core::u32>(_levels.size()); }
    [[nodiscard]] bool empty() const noexcept { return _levels.empty(); }

    /**
     * @brief Trilinear sample: bilinear within two levels, linear between them.
     *
     * @param level Fractional level of detail; 0 is the full-size texture.
     */
    [[nodiscard]] core::u32 sample(core::u32 uQ16, core::u32 vQ16, core::f32 level) const noexcept
    {
        if (_levels.empty())
            return 0u;
        if (level < 0.0f)
            level = 0.0f;
        const core::u32 last = static_cast<core::u32>(_levels.size()) - 1u;
        const core::u32 low = static_cast<core::u32>(level);
        if (low >= last)
            return _levels[last].sampleBilinear(uQ16, vQ16);

        const core::u32 a = _levels[low].sampleBilinear(uQ16, vQ16);
        const core::u32 b = _levels[low + 1u].sampleBilinear(uQ16, vQ16);
        const core::u32 t = static_cast<core::u32>((level - static_cast<core::f32>(low)) * 256.0f);

        core::u32 out = 0u;
        for (core::u32 shift = 0u; shift <= 16u; shift += 8u)
        {
            const core::u32 from = (a >> shift) & 0xFFu;
            const core::u32 to = (b >> shift) & 0xFFu;
            out |= (((from * (256u - t) + to * t) >> 8) & 0xFFu) << shift;
        }
        return out;
    }

    /**
     * @brief Level of detail from the world distance one pixel covers.
     *
     * @c log2 without libm: the exponent field of a float IS its integer log2, and
     * the mantissa's top bits interpolate between the powers well enough for a
     * level selection. Doing it any other way would mean a table or a series for a
     * quantity that is already sitting in the bit pattern.
     */
    [[nodiscard]] static core::f32 levelForFootprint(core::f32 texelsPerPixel) noexcept
    {
        if (texelsPerPixel <= 1.0f)
            return 0.0f;
        core::u32 bits = 0u;
        __builtin_memcpy(&bits, &texelsPerPixel, sizeof(bits));
        const core::i32 exponent = static_cast<core::i32>((bits >> 23) & 0xFFu) - 127;
        const core::f32 mantissa = static_cast<core::f32>(bits & 0x7FFFFFu) * (1.0f / 8388608.0f);
        return static_cast<core::f32>(exponent) + mantissa;
    }

private:
    pmr::vector<Texture> _levels;
};

} // namespace lpl::render

#endif // LPL_RENDER_MIP_TEXTURE_HPP
