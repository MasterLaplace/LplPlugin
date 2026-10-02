/**
 * @file Volume.hpp
 * @brief What a scanned volume is, in numbers, whichever volume it is.
 *
 * @warning **Nothing in this module names a subject.** A tomographic stack is a tomographic stack:
 * the same arithmetic serves a carbonised scroll at 7.91 micrometres, a fragment at 1.129, a
 * medical series at 300, or a simulation grid with no physical unit at all. A viewer that only
 * opens one of them is a demo rather than an instrument, so every number a renderer needs is
 * carried here and filled from the volume's own metadata. Opening a second subject is a different
 * @ref VolumeGeometry, never a second code path.
 *
 * @warning **The sample size is the number that makes scale a decision instead of an accident.** A
 * body is metres tall and the structure worth walking through can be tens of micrometres across,
 * so something must state the ratio. Stating it here, once, is what lets the same walk read
 * correctly at 7.91 micrometres and at 1.129: the world keeps its size, the sampling gets finer.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOXEL_VOLUME_HPP
#    define LPL_VOXEL_VOLUME_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/voxel/Brick.hpp>
#    include <lpl/voxel/Mosaic.hpp>

namespace lpl::voxel {

/**
 * @struct VolumeGeometry
 * @brief The shape of one scanned volume and the scale it is walked at.
 */
struct VolumeGeometry final {
    core::i64 samples[3]{0, 0, 0};    ///< Level-0 extent, in (z, y, x) order -- the zarr axis order.
    core::u32 levels{1u};             ///< Published pyramid levels, finest first.
    core::f32 voxelMicrometres{1.0f}; ///< Edge of one level-0 sample, in micrometres.

    /**
     * How many metres of walked world one micrometre of scroll becomes.
     *
     * @warning This is a **staging** decision, not a measurement, and it is the one that makes the
     * result legible or not. It defaults to 1 because this module has no opinion about anybody's
     * subject; the caller picks it from the feature it wants a body to be able to walk between.
     * On a Herculaneum roll the gap between neighbouring sheets is about 260 micrometres, so
     * 11538 puts that gap at three metres and a body walks a corridor. Leaving it at 1 -- one
     * sample, one metre, the obvious first idea -- makes a person fourteen micrometres tall and
     * that corridor fourteen metres wide: arithmetically correct, and unreadable.
     */
    core::f32 metresPerMicrometre{1.0f};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return samples[0] > 0 && samples[1] > 0 && samples[2] > 0 && levels > 0u && levels <= kMaxPyramidLevels &&
               voxelMicrometres > 0.0f;
    }

    /// @return Edge of one level-0 sample, in walked metres.
    [[nodiscard]] constexpr core::f32 metresPerSample() const noexcept
    {
        return voxelMicrometres * metresPerMicrometre * 1e-6f;
    }

    /// @return Extent of the whole volume along @p axis, in walked metres.
    [[nodiscard]] constexpr core::f32 extentMetres(core::u32 axis) const noexcept
    {
        return static_cast<core::f32>(samples[axis]) * metresPerSample();
    }

    /// @return Level-0 samples along @p axis at @p level, rounded up the way the pyramid rounds.
    [[nodiscard]] constexpr core::i64 samplesAtLevel(core::u32 axis, core::u32 level) const noexcept
    {
        const core::i64 d = static_cast<core::i64>(1) << level;
        return (samples[axis] + d - 1) / d;
    }

    /// @return Bricks along @p axis at @p level.
    [[nodiscard]] constexpr core::i32 bricksAtLevel(core::u32 axis, core::u32 level) const noexcept
    {
        const core::i64 n = samplesAtLevel(axis, level);
        return static_cast<core::i32>((n + kBrickEdge - 1) / kBrickEdge);
    }
};

/**
 * @struct DensityProfile
 * @brief What the samples of one volume actually look like, measured rather than assumed.
 *
 * @warning **This cannot be a constant, and treating it as one is the mistake that makes a viewer
 * work on exactly one scroll.** A carbonised roll is not air and matter: the gap between two
 * sheets is filled with lower-density material, not vacuum, so the whole distribution sits in a
 * narrow band whose position and width belong to the volume. Measured on PHerc0172 at 7.91 um the
 * band is roughly 115 to 190 around a mean of 147 -- and there is no reason for the next scroll,
 * scanned at another energy on another beamline, to land anywhere near that.
 */
struct DensityProfile final {
    core::u8 floorSample{0};   ///< Below this, nothing is drawn: the medium, not a sheet.
    core::u8 sheetSample{255}; ///< At and above this, a sheet is fully present.
    core::f32 mean{0.0f};      ///< Mean of non-empty samples at level 0.
    core::f32 deviation{1.0f}; ///< Standard deviation of non-empty samples at level 0.

    /**
     * Standard deviation at each level, divided by the level-0 one.
     *
     * @warning **This is the correction the pyramid demands, and skipping it makes the level of
     * detail lie about how much matter there is.** Downsampling is a box filter: it preserves the
     * mean exactly and squeezes the spread. Measured on one paired cubic millimetre of PHerc0172,
     * the deviation falls from 23.5 to 9.0 across six levels, so a window placed on the level-0
     * distribution catches almost everything by level 4. Rescaling the window toward the mean by
     * this ratio keeps the same matter visible at every level.
     *
     * @warning It is filled by measurement, not by that example. Another instrument's pyramid may
     * not even be a box filter.
     */
    core::f32 spreadRatio[kMaxPyramidLevels]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    [[nodiscard]] constexpr bool valid() const noexcept { return sheetSample > floorSample; }

    /// @return Where the bulk of the matter sits: everything below this is medium, not structure.
    [[nodiscard]] constexpr core::f32 bulkCeiling(core::f32 deviations) const noexcept
    {
        return mean + deviation * deviations;
    }
};

/**
 * @brief Measures a profile from a set of resident bricks.
 *
 * @param bricks    Level-0 bricks to read; more is better, and they should not all come from one
 *                  region -- the inside of a roll is compressed and the outside is not, and a
 *                  profile taken only at the centre reports a solid block.
 * @param count     How many.
 * @param quantile  Fraction of non-empty matter the sheet window should select, in [0,1]. The
 *                  physical duty cycle of sheet thickness over spire pitch is the honest default:
 *                  about 40 um of sheet every 300 um is 0.133.
 * @param windowDeviations  How far below that edge the window's floor sits, in measured standard
 *                  deviations.
 *
 * @warning **The width is in deviations, not in raw sample values, and that is what makes it
 * portable.** A width stated in sample values is calibrated on one beamline. Measured on
 * PHerc0172 the matter spans about three deviations, so a width of one is a narrow band near the
 * top -- structure against a transparent medium -- while three paints everything and renders fog.
 * The first run of this renderer derived the width from a low percentile and did exactly that: a
 * plausible cream-coloured haze with no sheets in it.
 */
[[nodiscard]] DensityProfile measureProfile(const BrickView *bricks, core::u32 count, core::f32 quantile,
                                            core::f32 windowDeviations = 1.0f) noexcept;

/**
 * @brief Fills @ref DensityProfile::spreadRatio from bricks covering the same region at each level.
 *
 * @warning The bricks must cover the **same physical region**, cropped, or the measurement is not
 * of downsampling at all. A whole level-5 brick spans 32 mm and leaves the scroll, so comparing
 * whole bricks reports the emptiness around the object and calls it contrast loss. That mistake
 * was made once here already, and the paired version is what produced the ratios above.
 *
 * @param perLevel  One brick per level, index 0 being level 0. Entries may be invalid: their
 *                  ratio is left at the previous level's, which degrades rather than lies.
 * @param count     Entries in @p perLevel.
 */
void measureSpread(DensityProfile &profile, const BrickView *perLevel, core::u32 count) noexcept;

/**
 * @brief Fills @ref DensityProfile::spreadRatio from a whole resident set, cropped to one box.
 *
 * @warning **The crop is the entire point, and skipping it produces a confidently wrong number.**
 * A resident set holds a small ring of fine bricks and a large one of coarse bricks, so a coarse
 * level covers far more of the world -- including the emptiness outside the subject. Measure each
 * level over its own bricks and the coarse levels report that emptiness as a collapse in contrast,
 * which is a real effect of something else entirely. That mistake was made once here, on whole
 * bricks, and the cropped version is what produced the ratios this renderer uses.
 *
 * @warning **Unmeasured, every ratio is one, and one means "no correction".** That is not a
 * neutral default: it leaves a curve calibrated on the finest level painting the entire histogram
 * a few levels up, which is exactly what makes a level of detail visible as tiling.
 *
 * @param profile     Filled in place; only @ref DensityProfile::spreadRatio is touched.
 * @param mosaic      The resident set.
 * @param centre      Box centre, in level-0 samples, (z, y, x).
 * @param halfExtent  Half the box edge, in level-0 samples. Small enough that the finest level
 *                    covers it, or the comparison is not paired.
 */
void measureSpreadOverBox(DensityProfile &profile, const BrickMosaic &mosaic, const core::i64 centre[3],
                          core::i64 halfExtent) noexcept;

} // namespace lpl::voxel

#endif // LPL_VOXEL_VOLUME_HPP
