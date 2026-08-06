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

    /**
     * @brief Which generator fills a FLOOR.
     *
     * Three of the four kinds are fillings and one is the container: a stack is what
     * @c Layered names, so a floor that claimed to be @c Layered would be a stack
     * inside a stack. It is clamped to @c Cellular rather than switched on, and a
     * caller that wants a mixed stack asks for it through @ref mixLayerKinds — which
     * is exactly what resolving a recipe's @c Layered does.
     *
     * @c Auto never arrives here either: @ref chooseCaveKind resolves it, because the
     * evidence it needs is about a PLACE and this struct describes a cave.
     */
    CaveKind layerKind{CaveKind::Cellular};

    /**
     * @brief Give every floor its own character. This is what @c Layered resolves to.
     *
     * The claim this file opens with — "several plans stacked, EACH WITH ITS OWN
     * CHARACTER, is a system" — was implemented as nothing but a fill probability that
     * drifted with depth. Every floor was the same automaton, so the sentence was not
     * true of the code under it. With this set, the generator itself varies with depth,
     * and a stack can be a natural cave over a warren of rooms.
     */
    bool mixLayerKinds{false};

    // ── What the PLACE is like, for a mixed stack ───────────────────────────
    //
    // Two scalars rather than a callback, for the reason ITerrainQuery gives: a
    // generator that took a lambda from its owner could not be given a fake in a test
    // nor say in its own parameters what it depends on. Both are ignored unless
    // @ref mixLayerKinds is set.

    bool settled{false};     ///< People were here. People dig rooms, not fissures.
    core::f32 wetness{0.5f}; ///< How much the ground dissolves; high ground branches.
};

/**
 * @struct CaveContext
 * @brief The evidence a place offers about what kind of underground it should have.
 *
 * What @c CaveKind::Auto is resolved from. Everything in it is already to hand
 * wherever the question is asked, which is the test a rule like this has to pass:
 * evidence that has to be invented for the rule is not evidence.
 */
struct CaveContext {
    bool settled{false};      ///< A settlement stands within reach of this place.
    core::f32 wetness{0.5f};  ///< Ground moisture in [0, 1].
    core::u32 layerCount{1u}; ///< Floors the underground will have.
};

/// Moisture at or above which ground is taken to dissolve rather than fracture.
inline constexpr core::f32 kKarstWetness = 0.58f;

/**
 * @brief Resolves @c CaveKind::Auto from what a place actually offers.
 *
 * Not a die roll. Each branch is a claim about the world that a player can read off
 * the result, which is the difference between a procedural default and noise:
 *
 *  - people leave ROOMS. A settlement within reach means somebody dug here, and what
 *    they dug is a partition of rectangles joined by corridors, not a fissure.
 *  - wet ground DISSOLVES. Above @ref kKarstWetness the underground is karst, and
 *    karst branches — which is what diffusion-limited aggregation produces and why it
 *    is in this enum at all.
 *  - anything deep enough to have several floors has had time to become several
 *    things, so it gets the mixed stack.
 *  - otherwise the automaton, which is what the default has always been.
 *
 * A pure function of its argument: no seed, no clock, no global. Two targets asking
 * about one place get one answer, and so does the same target asked twice.
 *
 * @param context What the place offers.
 * @return A concrete kind; never @c Auto.
 */
[[nodiscard]] CaveKind chooseCaveKind(const CaveContext &context);

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
 * @brief Recounts @c hollowCells and @c reachableCells after a caller edits the layers.
 *
 * @ref generateCaveSystem leaves both correct and @ref repairCaveReachability keeps
 * them so, which is enough for a caller that only reads. It is not enough for one that
 * WRITES — masking cells out or forcing a passage open changes what is hollow, and the
 * repair reads @c hollowCells to decide when it is finished. A stale count makes it
 * hunt for cells that are no longer there.
 *
 * Exported rather than left private because the caller that edits is the only one that
 * knows it has edited; deriving it inside the repair would recount on every round for
 * every caller that never touched anything.
 *
 * @param system System to recount, in place.
 */
void recountCaveReachability(CaveSystem &system);

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
