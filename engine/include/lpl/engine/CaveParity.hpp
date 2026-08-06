/**
 * @file CaveParity.hpp
 * @brief The determinism gate for a cave you can walk into.
 *
 * Every gate before this one folds a world that was GENERATED. This one folds a world
 * that was generated and then WALKED — because the claim being made is not that two
 * targets build the same cave, it is that a body entering it ends up in the same place
 * on both. Those are different claims, and only the second one is what a player has.
 *
 * The chain it exercises is the whole of the feature and nothing else:
 * @ref procgen::buildCaveWarren decides the cave, @ref procgen::caveWarrenSpanAt turns
 * a column into a floor and a ceiling, and @ref CharacterController decides where a
 * body may stand given that pair. Each link is Fixed32 and each one can disagree
 * between targets on its own; folding only the first would pass a run in which the
 * walker fell through the floor in ring 0.
 *
 * @warning What is folded here is authoritative state only — the body's position,
 *          velocity, heading and contact flags, and the cave's own cells. Nothing
 *          about how any of it is drawn.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-06
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_CAVE_PARITY_HPP
#    define LPL_ENGINE_CAVE_PARITY_HPP

#    include <lpl/procgen/CaveWarren.hpp>

namespace lpl::engine {

/**
 * @struct CaveFoldResult
 * @brief What the two targets have to agree about, cave and walker both.
 */
struct CaveFoldResult {
    core::u32 warrenSignature{0u}; ///< FNV-1a over the cave: cells, cover, volume, adit.
    core::u32 walkSignature{0u};   ///< FNV-1a over the body's authoritative state, every tick.
    core::u32 spanSignature{0u};   ///< FNV-1a over the floor and ceiling along the way in.

    core::u32 coveredColumns{0u};
    core::u32 openCells{0u};
    core::u32 reachableCells{0u};
    core::u32 apertureCells{0u};
    core::u32 pathLength{0u};

    /**
     * @brief Ticks the walker spent with rock over its head.
     *
     * The counter that makes this gate discriminating rather than merely stable. A
     * signature alone is satisfied by a run in which the body never left the shelf:
     * two targets would agree perfectly about a walker that failed to get in. Zero
     * here means the feature did not happen.
     */
    core::u32 enclosedTicks{0u};
    core::u32 descendedLevels{0u}; ///< Voxel levels between the highest and lowest floor stood on.
    core::u32 blocked{0u};         ///< Moves refused because the rise ahead was a wall.
    core::u32 headBumps{0u};       ///< Ticks the body's head met a ceiling.
    core::u32 navigable{0u};       ///< Whether the deepest gallery can be reached from the mouth.
    core::u32 kind{0u};            ///< The resolved procgen::CaveKind.
};

/**
 * @brief Builds the parity cave, walks a body into it, and folds both.
 *
 * The site is found rather than invented: the first cave-mouth landmark of the walked
 * world's own lattice that carries a warren. A synthetic warren would fold just as
 * deterministically and would prove nothing about the world anybody plays, which is
 * the same argument @ref procgen::foldEndlessPatch makes for sampling the real chunk
 * scheme.
 *
 * @return The signatures and counters.
 */
[[nodiscard]] CaveFoldResult foldCaveParity();

/**
 * @brief The same walk against a cave whose way in has been WALLED UP.
 *
 * The negative control, and this file would be worth much less without it. "The body
 * got inside" is satisfied by a collider that lets everything through, so the claim
 * only means something alongside a run where the body must NOT get inside and does
 * not. Sealing is done by filling the doorway columns with rock after the fact, so the
 * two runs differ in exactly one thing.
 *
 * @return The same fold, of a walk that should never report a tick underground.
 */
[[nodiscard]] CaveFoldResult foldSealedCaveParity();

} // namespace lpl::engine

#endif // LPL_ENGINE_CAVE_PARITY_HPP
