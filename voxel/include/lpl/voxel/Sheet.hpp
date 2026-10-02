/**
 * @file Sheet.hpp
 * @brief Following a surface that is only implied by the samples, and refusing to jump to the next one.
 *
 * @warning **Generating a surface matters more than loading one, because most of a scroll has no
 * traced surface at all.** A corpus publishes segments for the parts somebody has already worked
 * on; everywhere else there is nothing to load, and "nothing to load" is precisely where an
 * instrument earns its keep. So the tracer here is not a fallback for a missing file -- it is the
 * primary way a surface comes into being, and reading one from disk is the special case.
 *
 * @warning **The failure mode this file exists to neutralise is the SHEET JUMP.** A rolled scroll
 * is a stack of parallel surfaces tens of micrometres apart. A walk that steps onto the neighbour
 * produces a path that is perfectly smooth, perfectly plausible and WRONG -- and nothing about its
 * shape says so. The walk therefore refuses to advance when recentring moves it further than
 * @ref SheetTraceParams::maximumRecentre: a genuine follow corrects a fraction of a sample, a jump
 * corrects an interline. That distinction is the whole safety property, and it is counted rather
 * than silently applied.
 *
 * @warning **This does not judge whether the surface being followed is the RIGHT one.** It follows
 * the one it is placed on. Choosing where to start is the caller's problem and a different kind of
 * problem.
 *
 * @warning **The field has to be a ridge, not a plateau.** On a raw tomographic scan the sheets ARE
 * bright ridges, so the samples work directly. On a thresholded prediction -- a volume carrying
 * only two values -- there is no gradient inside the matter at all: the structure tensor sees
 * nothing until it touches an edge, and recentring on the maximum becomes an argmax over a
 * constant, which returns the first index and glues the walk to the EDGE of the sheet instead of
 * its middle. A binary field must be turned into a distance field before it comes here.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOXEL_SHEET_HPP
#    define LPL_VOXEL_SHEET_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/voxel/Mosaic.hpp>

namespace lpl::voxel {

/**
 * @struct SheetTraceParams
 * @brief How far to step, how hard to look, and what counts as a jump.
 *
 * The defaults are the ones this project measured while following real sheets, not preferences.
 */
struct SheetTraceParams final {
    core::f32 step{1.0f};
    /**< Step length in level-0 samples. Longer than a sample steps over the structure being
         followed; much shorter pays for recentring that finds nothing new. */

    core::i32 tensorRadius{2};
    /**< Window radius of the structure tensor, in samples. Too small and noise orients the normal;
         too large and the window spans two sheets, so the normal becomes their average. */

    core::f32 maximumRecentre{3.0f};
    /**< @warning The safety property. Beyond this, a recentring is not a correction -- it is a
         change of sheet, and the walk stops rather than produce a smooth plausible lie.

         ⚠ **The number that actually bounds safety is half the local spacing between sheets, and
         this is far below it on purpose.** On a Herculaneum roll the spacing is about 38 samples
         at 7.91 um, so anything under 19 cannot reach the neighbour; three is six times
         conservative. It is not two, which is the value this project's Python tracer uses,
         because that tracer runs on a smoothed PREDICTION and this one runs on the raw scan,
         where the ridge maximum jitters more. Measured: at two, every row of a real patch ended
         "would have jumped" and the patch came back empty; at three, all sixty-six rows ran to
         their full length. A limit tuned on one field is not a limit on another. */

    core::f32 searchRadius{3.0f};
    /**< How far along the normal to look for the ridge when recentring, once the walk is on it. */

    core::f32 seedSearchRadius{10.0f};
    /**< @warning **Finding the sheet and staying on it are different problems, and using one radius
         for both makes the tracer useless on real data.** A seed dropped at an arbitrary point sits
         on average a third of the spacing away from any ridge -- about ten samples on a Herculaneum
         roll -- so a three-sample search finds nothing, the walk starts in the medium and stops on
         its first step. Measured: with the step radius used for the seed, every row of a real patch
         ended "left the matter" and the patch came back empty. This radius applies to the FIRST
         snap only; afterwards the tight one is what keeps the walk from wandering to a neighbour. */

    core::u8 floorSample{0u};
    /**< Below this the walk stops: it has left the matter. */

    core::u32 maximumSteps{4096u};

    core::i32 ridgeSmoothing{1};
    /**< Radius, in samples, of the average taken when looking for the ridge.

         @warning **The jitter this removes is in the FIELD, not in the surface.** A tomographic
         ridge wobbles by a sample from one position to the next simply because the samples are
         noisy, and a recentring that chases every wobble writes that noise into the geometry: rows
         traced side by side then disagree by a few samples and the patch comes out corrugated,
         which reads as a crumpled sheet. Averaging a small window before looking for the maximum
         is not smoothing the result -- it is asking the question of a less noisy field. Zero turns
         it off. */
};

/**
 * @enum SheetStop
 * @brief Why a trace ended. Every reason is distinct because they mean different things.
 */
enum class SheetStop : core::u8 {
    Budget = 0,   ///< Ran out of steps. The sheet probably continues.
    LeftMatter,   ///< Fell below the floor: the sheet ended, or the walk fell off it.
    LeftResident, ///< Walked out of the bricks in memory. Not a fact about the scroll.
    WouldJump,    ///< Recentring exceeded the limit. **The sheet is still there; the walk refused.**
    Degenerate,   ///< No usable normal: the field is flat here.
};

/**
 * @struct SheetTrace
 * @brief A traced path along one surface, and what happened to it.
 */
struct SheetTrace final {
    core::u32 count{0};            ///< Points written.
    core::u32 recentrings{0};      ///< Steps that were pulled back onto the ridge.
    core::u32 refusedJumps{0};     ///< Steps refused as sheet changes. Non-zero is informative, not a fault.
    core::f32 totalRecentre{0.0f}; ///< Sum of recentring distances; the average says how well it fits.
    SheetStop stop{SheetStop::Budget};
};

/**
 * @brief Walks one surface from @p seed in direction @p heading.
 *
 * @param mosaic    Resident samples.
 * @param seed      Start, in level-0 samples (z, y, x). Snapped onto the ridge before the first step.
 * @param heading   Initial direction; projected into the sheet plane at every step, which is what
 *                  makes the walk follow curvature instead of leaving on a tangent.
 * @param params    Step, window, and the jump limit.
 * @param out       Destination for the path, in level-0 samples.
 * @param capacity  Points available.
 */
[[nodiscard]] SheetTrace traceSheet(const BrickMosaic &mosaic, const math::Vec3<core::f32> &seed,
                                    const math::Vec3<core::f32> &heading, const SheetTraceParams &params,
                                    math::Vec3<core::f32> *out, core::u32 capacity) noexcept;

/**
 * @brief The local sheet normal at a point, from the structure tensor.
 *
 * @warning **Not a finite difference.** Inside a sheet the gradient is small and noisy and its
 * direction wanders; the structure tensor accumulates gradients over a window and takes the
 * dominant eigenvector, which is stable exactly where the plain difference is not. The eigenvector
 * comes from power iteration rather than a closed form: no libm, and the answer is a direction, so
 * the sign is arbitrary and the caller must not read meaning into it.
 *
 * @return Whether a usable normal was found.
 */
[[nodiscard]] bool sheetNormal(const BrickMosaic &mosaic, const math::Vec3<core::f32> &at, core::i32 radius,
                               math::Vec3<core::f32> &outNormal) noexcept;

/// Widest patch a single trace call fills. Bounded so a patch needs no allocation of its own.
inline constexpr core::u32 kMaxPatchWidth = 512u;

/**
 * @struct SheetPatch
 * @brief A rectangular piece of one surface, as a grid of points.
 *
 * A trace is a curve; a patch is what you can actually look at. It is grown by tracing across the
 * sheet from each point of a first trace along it -- so every row is itself a refusal-guarded walk,
 * and a patch that stops short says where the surface stopped being followable.
 */
struct SheetPatch final {
    core::u32 columns{0};
    core::u32 rows{0};
    core::u32 refusedJumps{0};
    core::u32 shortRows{0}; ///< Rows that ended before the full width: the patch has a ragged edge.

    /**
     * How many row walks ended for each reason, indexed by @ref SheetStop.
     *
     * @warning **"Short rows" alone is not a diagnosis.** A patch that comes back empty says
     * nothing about whether the sheet ended, the walk fell off it, the bricks ran out, or the
     * guard refused -- and those call for four different responses. Reporting only the count sends
     * the operator to guess, which is the thing this project keeps paying for.
     */
    core::u32 stops[5]{};
    /// Row-major, @ref rows * @ref columns points. Rows shorter than @ref columns are padded with
    /// the last point they reached, and counted in @ref shortRows rather than left undefined.
    math::Vec3<core::f32> *points{nullptr};

    [[nodiscard]] const math::Vec3<core::f32> &at(core::u32 row, core::u32 column) const noexcept
    {
        return points[static_cast<core::usize>(row) * columns + column];
    }
};

/**
 * @brief Grows a patch of surface around @p seed.
 *
 * @param out       Storage for @p rows * @p columns points, owned by the caller.
 */
[[nodiscard]] SheetPatch traceSheetPatch(const BrickMosaic &mosaic, const math::Vec3<core::f32> &seed,
                                         const SheetTraceParams &params, core::u32 rows, core::u32 columns,
                                         math::Vec3<core::f32> *out) noexcept;

/**
 * @brief How far each point of a patch sits from the average of its neighbours.
 *
 * The number that says whether a surface is corrugated. Reported rather than assumed, because
 * "the trace looks crumpled" and "the sheet is crumpled" are different claims and only one of
 * them is about the scroll.
 *
 * @return Root-mean-square deviation, in level-0 samples.
 */
[[nodiscard]] core::f32 patchRoughness(const SheetPatch &patch) noexcept;

/**
 * @brief Pulls each point of a patch toward the average of its neighbours.
 *
 * @warning **This ALTERS the traced geometry, and that is why it is a separate call rather than
 * part of tracing.** Everything else in this file reports what the samples said; this changes it.
 * A relaxed patch is easier to look at and is no longer, strictly, where the walk went -- so a
 * measurement against another surface should say which one it used, and a caller that cares about
 * the trace itself should keep the unrelaxed points.
 *
 * @warning The border is held fixed. Letting it move would shrink the patch a little on every
 * iteration, and a surface that quietly retreats from its own edge is worse than a rough one.
 *
 * @param strength    How far toward the neighbour average, in [0,1]. Half is a gentle pass.
 * @param iterations  Repeats. Each one widens the neighbourhood the result depends on.
 */
void relaxPatch(SheetPatch &patch, core::f32 strength, core::u32 iterations) noexcept;

/**
 * @brief Relaxes a patch and puts it back on the ridge after every pass.
 *
 * @warning **This is the honest version of smoothing a trace, and the difference from
 * @ref relaxPatch matters.** Plain relaxation moves points toward their neighbours and stops
 * asking the samples anything -- push it far enough and the surface becomes its own average plane,
 * which scores perfectly on roughness and is no longer the sheet. Here every pass is followed by a
 * recentring along the local normal, so the result is smooth AND still where the matter is: the
 * smoothing removes lateral disagreement between rows, and the data puts each point back on the
 * crest.
 *
 * @warning It does not fix the reason the rows disagreed. Each row of a patch is walked
 * independently, so two neighbouring walks accumulate their own drift -- measured on a real
 * scroll at three samples of root-mean-square deviation, on a sheet five samples thick. A tracer
 * whose rows constrained each other would not need this; that is a different algorithm.
 */
void relaxPatchOnRidge(const BrickMosaic &mosaic, SheetPatch &patch, const SheetTraceParams &params, core::f32 strength,
                       core::u32 iterations) noexcept;

} // namespace lpl::voxel

#endif // LPL_VOXEL_SHEET_HPP
