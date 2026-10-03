/**
 * @file Transfer.hpp
 * @brief Turning a density into a colour and an opacity, without ever deciding "solid" or "not".
 *
 * @warning **This module renders the field, it does not threshold it, and that is the central
 * decision.** Looking at real level-0 samples settles it in one image: the sheets are bright
 * ribbons three to five samples thick, the gap between them is darker material rather than air,
 * and a hard cut anywhere in that band shreds the ribbons into fragments. Measured on PHerc0172,
 * the cut that matches the physical duty cycle of a sheet leaves 14.4 % of matter standing and
 * breaks every ribbon in the slice. A ramp keeps the ribbon, keeps its soft edge, and keeps the
 * reader able to see that the edge is soft -- which for a research instrument is the whole point:
 * a binary surface silently asserts a boundary that the scan never resolved.
 *
 * @warning **A curve that means the same thing at every level is a curve that moves with the
 * level.** Downsampling preserves the mean and squeezes the spread, so a window fixed on the
 * level-0 distribution swallows the whole histogram a few levels up. @ref TransferFunction::forLevel
 * rescales the window toward the mean by the measured spread ratio, so the same matter stays
 * visible as the ray gets further away and the samples get coarser.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOXEL_TRANSFER_HPP
#    define LPL_VOXEL_TRANSFER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/voxel/Volume.hpp>

namespace lpl::voxel {

/**
 * @struct TransferFunction
 * @brief Density to emitted colour and absorption, as a table the marcher indexes directly.
 *
 * Colour is **not** premultiplied: the compositor multiplies by the step's alpha, and a table that
 * arrived premultiplied would be multiplied twice the first time somebody changed the step size.
 *
 * @warning Alpha is opacity **per level-0 sample of path length**, not per step. A marcher that
 * takes bigger steps far away must correct for it, or the same matter turns denser purely because
 * the camera moved back -- the classic direct-volume-rendering artefact, and the one that makes a
 * level-of-detail transition look like a change in the object.
 */
struct TransferFunction final {
    core::f32 red[256]{};
    core::f32 green[256]{};
    core::f32 blue[256]{};
    core::f32 alpha[256]{};

    core::u8 firstVisible{255}; ///< Lowest density whose alpha is non-zero; empty-space skipping reads it.

    core::f32 lowEdge{0.0f};    ///< Density where the ramp this table was built from leaves zero.
    core::f32 highEdge{255.0f}; ///< Density where that ramp reaches @ref peakAlpha.
    core::f32 peakAlpha{0.0f};  ///< Opacity per level-0 sample at and above @ref highEdge.

    /**
     * @brief The same curve, rescaled for a coarser level.
     *
     * @warning Returns a copy on purpose. A marcher crossing three levels in one ray needs all
     * three at once, and mutating a shared one would make the picture depend on the order the
     * bricks happened to be visited.
     *
     * @warning **Only the ramp survives the rescaling.** The coarser curve is rebuilt from
     * @ref lowEdge, @ref highEdge and @ref peakAlpha with the colours of @ref rampTransfer, so a
     * table edited by hand keeps its edits at level 0 and loses them at every coarser level whose
     * spread ratio is under one. A table filled by hand without those three fields loses more:
     * @ref peakAlpha is zero by default, so at those levels it paints NOTHING.
     */
    [[nodiscard]] TransferFunction forLevel(const DensityProfile &profile, core::u32 level) const noexcept;

    /// @return Whether a brick or a cell whose largest sample is @p highest can paint anything.
    [[nodiscard]] constexpr bool reachesVisible(core::u8 highest) const noexcept { return highest >= firstVisible; }
};

/**
 * @brief Builds a ramp between two densities, with a colour that warms as it rises.
 *
 * @warning **This is the only curve this module ships, on purpose.** What a density *means* --
 * bone, sheet, tissue, smoke -- belongs to whoever knows the instrument that produced it, and a
 * generic engine that shipped a dozen named presets would be asserting knowledge it does not have.
 * Callers build their own from a measured @ref DensityProfile.
 */
[[nodiscard]] TransferFunction rampTransfer(core::u8 floorSample, core::u8 sheetSample, core::f32 peakAlpha) noexcept;

/**
 * @brief Peak opacity such that @p samples of full-density path reach @p opacity.
 *
 * @warning **This exists so nobody has to guess a peak alpha, and guessing is what makes a volume
 * render either fog or a wall.** The number that means something is physical -- "a sheet should be
 * nearly solid once the ray has crossed its own thickness" -- and it depends on how many samples
 * that thickness is, which depends on the level and the instrument. Stating the intent and
 * deriving the constant keeps the two from drifting apart when either changes.
 *
 * @param samples  Path length through full-density matter, in level-0 samples.
 * @param opacity  Accumulated opacity wanted after that path, in (0,1).
 */
[[nodiscard]] core::f32 alphaForOpaqueAfter(core::f32 samples, core::f32 opacity) noexcept;

/**
 * @brief Fraction of light left after crossing @p samples level-0 samples of opacity @p perSample.
 *
 * Exactly (1 - @p perSample)^n over the whole samples, and linear across the fractional remainder.
 *
 * @return 1 when @p perSample or @p samples is at most 0, and 0 when @p perSample is at least 1.
 */
[[nodiscard]] core::f32 transparencyAfter(core::f32 perSample, core::f32 samples) noexcept;

} // namespace lpl::voxel

#endif // LPL_VOXEL_TRANSFER_HPP
