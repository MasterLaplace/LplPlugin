/**
 * @file WorldRecipe.hpp
 * @brief A replayable procedural world: the recipe, not the 100 000 trees.
 *
 * This is the unit the rest of the project trades in. A world is not a list of
 * entities, it is a seed plus the passes applied to it — so it costs a few
 * hundred bytes to store, a few hundred tokens for an AI to author, and it can
 * be replayed anywhere the engine runs, including ring 0 where there is no JSON
 * parser and no filesystem. A `.lplscene` document stores this recipe; baking it
 * to explicit entities stays possible for archival and diffing, but the recipe
 * is the source of truth.
 *
 * A recipe is a declarative description of a @ref WorldBuilder pipeline, and
 * deliberately nothing more. There is exactly one implementation of every pass,
 * and this is the serialisable way to ask for them: what the kernel replays, what
 * a cartridge carries and what the editor authors are then the same world, not
 * three worlds that resemble each other. The module used to hold a second, older
 * generator reachable only from here, which meant the passes under the parity
 * gate were not the passes anything else ran — the gate was stable precisely
 * because it exercised nothing.
 *
 * Because every pass is authoritative Fixed32 (value noise with no libm, erosion
 * and drainage as grid relaxations with a fixed visit order, a heap-free
 * breadth-first gate), baking the same recipe on the Linux oracle and inside the
 * i686 kernel must fold the SAME FNV-1a signatures, bit for bit.
 * @ref parityWorldRecipe is the canonical recipe both sides bake for that gate;
 * it lives here, in one constexpr function, precisely so the two callers cannot
 * drift apart by editing their own copy of the parameters.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_WORLDRECIPE_HPP
#    define LPL_PROCGEN_WORLDRECIPE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/WorldBuilder.hpp>

namespace lpl::ecs {
class Registry;
}

namespace lpl::procgen {

/**
 * @brief Scatter rules a single recipe may carry.
 *
 * Bounded on purpose. A recipe is a wire object before it is an engine object,
 * and a wire object with an unbounded list needs a length prefix, a heap and a
 * bounds check in ring 0. Four kinds of prop is what a hand-authored world uses;
 * anything richer is a job for several recipes rather than for an open array.
 */
inline constexpr core::u32 kMaxScatterRules = 4u;

/**
 * @struct WorldRecipe
 * @brief An ordered set of deterministic passes producing a world from a seed.
 *
 * The pass order is part of the contract and is fixed by @ref bakeWorld, not by
 * the caller: terrain, erosion, water, climate, underground, civilisation,
 * verdict, materialisation. Reordering would change the entity creation order and
 * therefore the fold, so a recipe is only comparable to another recipe baked by
 * the same @ref bakeWorld.
 *
 * Each pass carries its own parameter struct rather than a flattened copy of its
 * fields: the parameters are the module's own, so a pass that grows a knob is
 * expressible here the day it is written, with no translation layer to update.
 * (The **wire** form in `lpl::pack` is flattened, which is a different problem
 * with a different answer — see lpl/pack/GamePack.hpp.)
 */
struct WorldRecipe {
    core::u32 seed{1337u};    ///< Master seed; every pass derives its own stream.
    core::u32 width{24u};     ///< Terrain cells along X.
    core::u32 depth{24u};     ///< Terrain cells along Z.
    core::f32 cellSize{0.5f}; ///< World units between two cells.

    NoiseParams terrain{};       ///< Base terrain layer.
    core::f32 heightLow{-8.0f};  ///< Normalised range floor, when @ref normalizeTerrain.
    core::f32 heightHigh{16.0f}; ///< Normalised range ceiling.

    ThermalErosionParams thermal{};     ///< Talus relaxation.
    HydraulicErosionParams hydraulic{}; ///< Capacity-limited water erosion.
    RiverParams rivers{};               ///< How much drainage becomes visible water.
    MoistureParams climate{};           ///< Rainfall, wind, coastal reach.
    ClimateParams axes{};               ///< The six climate axes' shaping.
    BiomeParams biomes{};               ///< Sea, shore and summit thresholds.
    CaveParams caves{};                 ///< Underground layer.
    SettlementParams settlement{};      ///< Town layout.
    RoadParams roads{};                 ///< Road network grown over the town.
    GateCriteria gate{};                ///< What the underground must satisfy.

    ScatterRule scatter[kMaxScatterRules]{}; ///< Prop rules, first @ref scatterCount used.
    core::u32 scatterCount{0u};              ///< Rules actually in play.

    bool normalizeTerrain{true};  ///< Rescale the field before any absolute threshold reads it.
    bool erodeTerrain{true};      ///< Run both erosion models.
    bool carveRivers{true};       ///< Route drainage and cut river beds.
    bool classifyBiomes{true};    ///< Compute moisture and classify.
    bool carveCaves{true};        ///< Generate the underground layer.
    bool placeSettlement{true};   ///< Lay a town onto the terrain.
    bool growRoads{true};         ///< Grow a road network over it.
    bool materializeGround{true}; ///< Emit one entity per ground cell (off: props only).
    bool checkPlayability{true};  ///< Judge the underground against @ref gate.
};

/**
 * @struct WorldRecipeResult
 * @brief What baking a recipe produced, in a form a C caller can print.
 *
 * Deliberately free of Fixed32 and bool so the kernel smoke can copy it into a
 * plain C struct field by field without a conversion that might differ between
 * targets.
 *
 * Three signatures rather than one, and that is the point of the gate. The state
 * fold only sees where entities ended up, so a pass that reshapes the terrain
 * without moving a cube — every climate pass, most of erosion — would be
 * invisible to it. Folding the height field and the biome map puts the *grids*
 * under the determinism contract, which is where all the arithmetic that could
 * diverge between targets actually happens.
 */
struct WorldRecipeResult {
    core::u32 entityCount{0u};     ///< Entities materialised by all passes.
    core::u32 stateSignature{0u};  ///< FNV-1a fold of authoritative Fixed32 entity state.
    core::u32 heightSignature{0u}; ///< FNV-1a fold of the final height field.
    core::u32 biomeSignature{0u};  ///< FNV-1a fold of the biome map.
    core::u32 riverCells{0u};      ///< Cells carved as river.
    core::u32 roadCells{0u};       ///< Cells the road network occupies.
    core::u32 lakeCells{0u};       ///< Cells holding standing water.
    core::u32 dungeonFloor{0u};    ///< Open cells in the underground layer.
    core::u32 settlementPlots{0u}; ///< Building footprints laid out.
    core::u32 gateReachable{0u};   ///< 1 if the underground's goal is reachable.
    core::u32 gateVisited{0u};     ///< Cells the gate's flood actually reached.
    core::u32 gatePathLength{0u};  ///< Steps from entrance to exit.
    core::u32 ok{0u};              ///< 1 if the world is non-empty AND passes its gate.
};

/**
 * @brief The canonical recipe baked by both the Linux oracle and the kernel.
 *
 * Sized for the kernel's 4 MiB heap, not a desktop's: a 24x24 terrain plus a
 * modest scatter is a few hundred entities, comparable to the CubePile sample
 * that already boots. Changing anything here re-folds the gate on BOTH sides at
 * once, which is the point.
 *
 * It exercises the passes on purpose, not the cheapest possible path: noise,
 * both erosions, depression filling and drainage, river carving, the rainfall and
 * rain-shadow climate, Whittaker classification, blue-noise scatter, a cellular
 * cave with its connectivity repair, a Voronoi-districted settlement, and the
 * playability verdict. A gate that skips the interesting passes proves the
 * uninteresting ones.
 *
 * @return The parity recipe.
 */
[[nodiscard]] constexpr WorldRecipe parityWorldRecipe() noexcept
{
    WorldRecipe recipe{};

    recipe.seed = 1337u;
    recipe.width = 24u;
    recipe.depth = 24u;
    recipe.cellSize = 0.5f;

    recipe.terrain.seed = 1337u;
    recipe.terrain.frequency = 0.15f;
    recipe.terrain.amplitude = 12.0f;
    recipe.terrain.octaves = 4u;
    recipe.terrain.lacunarity = 2.0f;
    recipe.terrain.persistence = 0.5f;
    recipe.terrain.kind = NoiseKind::Fbm;

    recipe.heightLow = -8.0f;
    recipe.heightHigh = 16.0f;

    // Fewer iterations than the desktop defaults: the shapes converge quickly on
    // a 24x24 field, and ring 0 pays for every pass in boot time.
    recipe.thermal.iterations = 8u;
    recipe.hydraulic.iterations = 12u;

    recipe.caves.width = 24u;
    recipe.caves.depth = 24u;
    recipe.caves.seed = 0xCA4Eu;
    recipe.caves.minRegionSize = 12u;

    recipe.settlement.districtSize = 8u;

    // A 24x24 world is one district wide; three rewrite rounds fill it without
    // the turtle spending most of its string refused against the border.
    recipe.roads.iterations = 3u;
    recipe.roads.stepLength = 2u;

    // A 24x24 cave is small; the desktop defaults would reject most seeds for
    // being cramped rather than for being unplayable.
    recipe.gate.minPathLength = 4u;
    recipe.gate.minWalkableCells = 16u;

    recipe.scatter[0].biome = BiomeId::Grassland;
    recipe.scatter[0].density = 0.06f;
    recipe.scatter[0].halfExtent = 0.2f;
    recipe.scatter[0].tag = 1u;
    recipe.scatterCount = 1u;

    return recipe;
}

/**
 * @brief FNV-1a fold of every entity's authoritative Fixed32 state.
 *
 * Walks partitions then chunks in storage order and folds the raw Q16.16 words
 * of Position and AABB. Raw ints, never floats: the fold must be an identity on
 * the bits, not on a rounded decimal rendering of them.
 *
 * @param registry World to fold.
 * @return The 32-bit signature (offset basis if the world is empty).
 */
[[nodiscard]] core::u32 foldWorldState(const ecs::Registry &registry) noexcept;

/**
 * @brief Runs every enabled pass of @p recipe into @p registry and folds it.
 * @param registry Destination world (should be empty on entry).
 * @param recipe   The passes to run.
 * @return Entity count, the three signatures, and the playability verdict.
 */
WorldRecipeResult bakeWorld(ecs::Registry &registry, const WorldRecipe &recipe);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_WORLDRECIPE_HPP
