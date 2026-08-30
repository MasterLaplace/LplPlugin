/**
 * @file ReliefParity.hpp
 * @brief The relief determinism gate: a world standing on measured ground, on two targets.
 *
 * @warning **What this gate claims, and what it deliberately does not.** It claims that a world whose
 * lowest frequency is a survey rather than a generator comes out bit-identical on the Linux oracle
 * and in ring 0 -- the projection, the cell lookup, the border blend, the detail layer added back,
 * and a body walking across all of it. It does NOT claim the samples are SRTM: they are built here
 * from an integer formula so that both sides can hold the same ones without a file. The reader that
 * turns real tiles into these numbers is measured separately, against the real Peloponnese, in
 * `test-relief`.
 *
 * @warning **The section is not crossed here, and that is stated rather than glossed.** Reading a
 * `.lplknow` section in ring 0 is what gate P18 already proves, and the host round-trip -- resample,
 * bake, reopen, stand on it -- is asserted in `test-relief`. What is new on this path is the
 * ARITHMETIC, so that is what is folded on both targets.
 *
 * The canonical field is a coast: land in the west falling to bathymetry in the east, with a ridge
 * across it and a hole punched in the survey. Every one of those exists to exercise a branch --
 * ground above and below sea level, a slope for a body to walk down, and a gap that must fall back
 * to invented ground rather than to thirty-two kilometres below the sea.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_RELIEFPARITY_HPP
#    define LPL_ENGINE_RELIEFPARITY_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::engine {

/**
 * @struct ReliefFoldResult
 * @brief Signatures and counts both targets must agree on.
 */
struct ReliefFoldResult {
    core::u32 sampleSignature{0u}; ///< The survey itself, before anything reads it.
    core::u32 heightSignature{0u}; ///< Ground over a window spanning inside, border and outside.
    core::u32 walkSignature{0u};   ///< Where a body went, step by step.
    core::u32 coastSignature{0u};  ///< Which cells came out sea and which land.

    /**
     * Cells the survey answered for.
     *
     * @warning Counted because the signatures alone cannot tell a working field from an absent one:
     * a run where the relief never applies folds perfectly on both targets and proves nothing.
     */
    core::u32 measuredCells{0u};
    core::u32 inventedCells{0u}; ///< Cells outside the survey, or inside its hole.
    core::u32 blendedCells{0u};  ///< Cells in the border band, part measured and part invented.
    core::u32 seaCells{0u};      ///< Cells at or below the world's sea level.

    core::u32 walkSteps{0u}; ///< Steps the body actually took.
    core::i32 descended{0};  ///< Raw Q16.16 height the walk lost, start to finish.
};

/**
 * @brief Runs the canonical relief world and folds it.
 *
 * @return The signatures.
 */
[[nodiscard]] ReliefFoldResult foldReliefParity();

/**
 * @brief The same world with no survey behind it.
 *
 * @warning **The control, and without it the gate is satisfied by a field that never applies.** Every
 * signature above is stable on both targets whether or not the relief is ever read; only comparing
 * against the world that ignores it shows that it was. The two must DIFFER, and the counts say by
 * how much.
 *
 * @return The signatures of the purely invented world.
 */
[[nodiscard]] ReliefFoldResult foldInventedParity();

} // namespace lpl::engine

#endif // LPL_ENGINE_RELIEFPARITY_HPP
