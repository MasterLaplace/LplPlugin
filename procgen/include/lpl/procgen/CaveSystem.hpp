/**
 * @file CaveSystem.hpp
 * @brief Caves that are actually caves: layered, linked, and open to the sky.
 *
 * What this module produced before was a **2D dungeon plan extruded into one flat
 * layer, buried seven units down, with no connection to the world above it**. It
 * was honest about being an instrument — it made `Dungeon`'s output legible — but
 * it was not a cave. A cave is a hole in the ground that leads somewhere.
 *
 * Three things separate the two, and none of them is a new generator:
 *
 *  - **Layers.** One plan is a floor. Several plans stacked, each with its own
 *    character, is a system — and every generator here already produces plans.
 *  - **Shafts.** A shaft is only valid where BOTH layers are hollow at the same
 *    (x, z), which is the one thing a per-layer generator cannot know. This is
 *    the piece that was missing.
 *  - **Entrances.** Shafts that pierce the SURFACE. Without them the whole system
 *    is a sealed void that no player will ever see, which is the state the module
 *    was actually in.
 *
 * And then the guarantee, the same one every other pass here carries: every
 * hollow cell must be reachable from at least one entrance. A cave nobody can
 * enter is not content, and neither is a chamber walled off inside one.
 *
 * The alternative — a 3D density field and an isosurface — is the real answer and
 * is deliberately not taken: it needs a renderer that consumes meshes, and until
 * one does, it would be geometry nothing can draw.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_CAVESYSTEM_HPP
#    define LPL_PROCGEN_CAVESYSTEM_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Biome.hpp>
#    include <lpl/procgen/Dungeon.hpp>
#    include <lpl/procgen/Extrusion.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/QualityGate.hpp>

namespace lpl::procgen {

/// Most layers a cave system may stack. Bounded so the volume stays predictable.
inline constexpr core::u32 kMaxCaveLayers = 8u;

/**
 * @struct CaveSystemParams
 * @brief How deep the system goes, and what each layer is like.
 */
struct CaveSystemParams {
    core::u32 width{64u};  ///< Cells along X.
    core::u32 depth{64u};  ///< Cells along Z.
    core::u32 seed{1337u}; ///< Determinism anchor.

    core::u32 layers{3u};         ///< Stacked plans; clamped to @ref kMaxCaveLayers.
    core::u32 levelsPerLayer{2u}; ///< Voxel levels each layer occupies.

    /**
     * @brief Fill probability of the shallowest layer.
     *
     * Deeper layers get more of it, so the system opens out as it descends:
     * cramped tunnels near the surface, caverns at the bottom. One parameter, and
     * the depth does the rest — which is cheaper than describing every layer and
     * gives the descent a direction.
     */
    core::f32 topFill{0.46f};
    core::f32 deepFill{0.38f};     ///< Fill probability of the deepest layer.
    core::u32 automatonSteps{5u};  ///< Smoothing steps per layer.
    core::u32 minChamberSize{24u}; ///< Pockets smaller than this are filled in.

    core::u32 shaftsPerPair{3u};      ///< Shafts attempted between each pair of layers.
    core::u32 entrances{2u};          ///< Shafts attempted through the surface.
    core::f32 entranceMaxSlope{1.5f}; ///< Steepest ground an entrance may open on.
};

/**
 * @struct CaveShaft
 * @brief A vertical link, and what it joins.
 */
struct CaveShaft {
    core::u32 x{0u};          ///< Column.
    core::u32 z{0u};          ///< Row.
    core::u32 upperLayer{0u}; ///< Shallower layer index.
    core::u32 lowerLayer{0u}; ///< Deeper layer index.
    bool surface{false};      ///< True when the shaft opens onto the world above.
};

/**
 * @struct CaveSystem
 * @brief A stack of plans, the shafts joining them, and the way in.
 */
struct CaveSystem {
    DungeonMap layer[kMaxCaveLayers]; ///< Plan per layer, shallowest first.
    core::u32 layerCount{0u};         ///< Layers actually generated.

    lpl::pmr::vector<CaveShaft> shafts; ///< Every vertical link, entrances included.
    core::u32 entranceCount{0u};        ///< Shafts that pierce the surface.
    core::u32 hollowCells{0u};          ///< Open cells across every layer.
    core::u32 reachableCells{0u};       ///< Open cells reachable from an entrance.
    core::u32 repairedCells{0u};        ///< Cells the reachability repair had to open.

    /// @return Whether every hollow cell can be reached from an entrance.
    [[nodiscard]] bool fullyReachable() const noexcept { return hollowCells != 0u && reachableCells == hollowCells; }
};

/**
 * @brief Generates a layered cave system under a terrain.
 *
 * @param params  Layer count, fill and shaft budget.
 * @param surface Terrain the entrances must break through; may be empty, in which
 *                case no entrance is cut and @c entranceCount stays 0.
 * @param biomes  Optional biome map: an entrance will not open under water or on
 *                bare summits, which is the difference between a cave mouth and a
 *                hole in a glacier.
 * @return The system.
 */
[[nodiscard]] CaveSystem generateCaveSystem(const CaveSystemParams &params, const Heightfield &surface,
                                            const BiomeMap *biomes = nullptr);

/**
 * @brief Opens walls until every hollow cell is reachable from an entrance.
 *
 * Runs in three dimensions over the stack, so a chamber sealed on its own layer
 * can be joined by a new shaft as readily as by a doorway. Without this the
 * layering makes the old problem worse rather than better: more layers means more
 * places to be sealed in.
 *
 * @param system System to repair in place.
 * @param seed   Stream for tie-breaking.
 * @return Number of cells opened.
 */
/**
 * @brief Judges a stack the way @ref evaluateLevel judges one plan.
 *
 * The playability gate could not judge a layered system at all. @ref GateCriteria was
 * asked of the flat @c DungeonMap, which this generator leaves empty because it fills
 * a @ref CaveSystem instead — so a recipe asking for layers reported zero open cells
 * and failed a world that was perfectly navigable, and a document had to switch the
 * gate off to use the generator. Switching a check off to use a feature is how a check
 * stops meaning anything.
 *
 * The measurements are the same ones, taken in three dimensions: reachability floods
 * from the entrances through the shafts, and a cell's neighbours are the four beside
 * it plus whatever shaft touches it. Two consequences worth naming:
 *
 * - `goalReachable` means **the deepest layer can be reached from the surface**. That
 *   is the failure a stack has and a plan does not — more layers means more places to
 *   be sealed in — and nothing that looks at one floor can see it.
 * - `pathLength` is the deepest reachable cell's distance from the way in, so
 *   @ref GateCriteria::minPathLength keeps its meaning: reject a system whose bottom
 *   is one step from daylight.
 *
 * @param system The stack to measure; an empty one measures as all zeroes.
 * @return The measurements, for @ref passesGate.
 */
[[nodiscard]] LevelQuality evaluateCaveSystem(const CaveSystem &system);

core::u32 repairCaveReachability(CaveSystem &system, core::u32 seed);

/**
 * @brief Turns the stack into one voxel volume.
 *
 * Layer 0 sits at the top of the volume and the deepest layer at the bottom, so
 * the volume reads the way the world does rather than the way the array is
 * indexed.
 *
 * @param system  The system.
 * @param params  For @c levelsPerLayer.
 * @param rock    Material id for solid rock; 0 leaves it empty.
 * @return The volume.
 */
[[nodiscard]] VoxelVolume caveVolume(const CaveSystem &system, const CaveSystemParams &params, core::u8 rock);

/// @brief FNV-1a fold of every layer and every shaft, for determinism checks.
[[nodiscard]] core::u32 foldCaveSystem(const CaveSystem &system);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CAVESYSTEM_HPP
