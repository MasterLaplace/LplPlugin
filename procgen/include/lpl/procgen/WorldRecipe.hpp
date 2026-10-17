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
 * bounds check in ring 0.
 *
 * Eight rather than four, because four was a real ceiling: one rule per biome is
 * how vegetation is actually written — a taiga is conifers, a savanna is scrub, a
 * marsh is reeds — and four runs out at the fifth kind of plant. A viewer that
 * wanted six had to write `WorldBuilder` calls by hand, and a world built that way
 * cannot be saved, baked, replayed in ring 0 or asked for by an intelligence.
 * `lpl::pack::kWireScatterRules` must match, and a static_assert says so.
 */
inline constexpr core::u32 kMaxScatterRules = 8u;

/**
 * @brief Longest roadside grammar a recipe may carry, NUL included.
 *
 * A fixed array rather than a string, for the reason the whole recipe is bounded: it
 * is a wire object before it is an engine object, and a wire object with an unbounded
 * string needs a length prefix, a heap and a bounds check in ring 0. Sixty-four
 * characters is several times the longest pattern the grammar was designed for
 * (`{[A,P]:2,[BL,P]:1}*,[G,P]` is twenty-five).
 */
inline constexpr core::u32 kMaxRoadsidePattern = 64u;

namespace detail {

/**
 * @brief Character-by-character equality. No libc: this module is freestanding.
 */
[[nodiscard]] constexpr bool sameName(const char *a, const char *b) noexcept;

} // namespace detail

/**
 * @brief Spells a cave kind for a document.
 * @return The word a document spells @p kind with.
 */
[[nodiscard]] constexpr const char *caveKindName(CaveKind kind) noexcept;

/**
 * @brief Reads a cave kind from a document.
 *
 * A WORD, never an index, for the reason every named thing in this format is one: an
 * index means whatever the enumeration happened to be on the day it was written, so
 * reordering the enum silently reinterprets every document already on disk.
 *
 * @param name The spelling, as @ref caveKindName produces it.
 * @param outKind Set on success.
 * @return false when the word is not one of the five.
 */
[[nodiscard]] constexpr bool caveKindByName(const char *name, CaveKind &outKind) noexcept;

/**
 * @brief Spells a province distance metric for a document.
 * @return The word a document spells @p metric with.
 */
[[nodiscard]] constexpr const char *distanceMetricName(DistanceMetric metric) noexcept;

/**
 * @brief Reads a province distance metric from a document.
 *
 * See @ref caveKindByName for why it is a word.
 */
[[nodiscard]] constexpr bool distanceMetricByName(const char *name, DistanceMetric &outMetric) noexcept;

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
    core::u32 seed{1337u};    /**< Master seed; every pass derives its own stream. */
    core::u32 width{24u};     /**< Terrain cells along X. */
    core::u32 depth{24u};     /**< Terrain cells along Z. */
    core::f32 cellSize{0.5f}; /**< World units between two cells. */

    NoiseParams terrain{};       /**< Base terrain layer. */
    core::f32 heightLow{-8.0f};  /**< Normalised range floor, when @ref normalizeTerrain. */
    core::f32 heightHigh{16.0f}; /**< Normalised range ceiling. */

    /**
     * @brief Lowest the finished ground may sit, or 0 to leave the field where it fell.
     *
     * The built-in physics stops a body half a unit above the origin and knows
     * nothing about a heightfield, so a world whose ground dips under that line has
     * **two** floors: the real one, and a flat invisible one the bodies actually come
     * to rest on. The fix is to move the world, once, before a single entity exists —
     * shifting it afterwards and then chasing everything already placed in it is the
     * wrong way round, and it does not work, because "everything" grows.
     *
     * A measurement, not a constant: the shift is `clearance - lowest cell`, which is
     * only knowable once erosion and the rivers have run. That is why this is its own
     * pass late in the pipeline rather than a value folded into @ref heightLow, which
     * is applied before either.
     *
     * Every absolute threshold travels with it — sea level, the mountain and snow
     * lines, the climate's sea level, the floors the settlement and the roads refuse
     * to build below — so the classification still sees the relative elevations it was
     * tuned for. Forgetting one of them is what puts roads under the new ground.
     */
    core::f32 groundClearance{0.0f};

    ThermalErosionParams thermal{};     /**< Talus relaxation. */
    HydraulicErosionParams hydraulic{}; /**< Capacity-limited water erosion. */
    RiverParams rivers{};               /**< How much drainage becomes visible water. */
    MoistureParams climate{};           /**< Rainfall, wind, coastal reach. */
    ClimateParams axes{};               /**< The six climate axes' shaping. */
    BiomeParams biomes{};               /**< Sea, shore and summit thresholds. */
    VoronoiParams provinces{};          /**< Surface regions, when @ref partitionRegions. */
    CaveParams caves{};                 /**< Underground layer, when @ref caveKind is Cellular. */
    BspDungeonParams rooms{};           /**< Underground layer, when @ref caveKind is Bsp. */
    DlaParams aggregation{};            /**< Underground layer, when @ref caveKind is Dla. */
    CaveSystemParams caveSystem{};      /**< Underground layer, when @ref caveKind is Layered. */
    SettlementParams settlement{};      /**< Town layout. */
    BuildingGrammarParams buildings{};  /**< How a plot becomes a volume, when @ref raiseBuildings. */
    RoadParams roads{};                 /**< Road network grown over the town. */
    GateCriteria gate{};                /**< What the underground must satisfy. */

    ScatterRule scatter[kMaxScatterRules]{}; /**< Prop rules, first @ref scatterCount used. */
    core::u32 scatterCount{0u};              /**< Rules actually in play. */

    /**
     * @brief Terrace steps, or 0 to leave the field smooth.
     *
     * A count rather than a count-plus-flag: zero steps and "no terracing" are the
     * same world, so a second switch could only ever contradict the first.
     */
    core::u32 terraceSteps{0u};

    CaveKind caveKind{CaveKind::Cellular}; /**< Which underground generator runs. See @ref CaveKind. */

    char roadsidePattern[kMaxRoadsidePattern]{}; /**< Roadside decoration's L-system; empty leaves the verges bare. */
    core::u32 roadsideLevels{0u};                /**< Voxel levels of the roadside modules; 0 also leaves them bare. */

    bool normalizeTerrain{true};  /**< Rescale the field before any absolute threshold reads it. */
    bool erodeTerrain{true};      /**< Run both erosion models. */
    bool carveRivers{true};       /**< Route drainage and cut river beds. */
    bool classifyBiomes{true};    /**< Compute moisture and classify. */
    bool partitionRegions{false}; /**< Divide the surface into Voronoi provinces. */
    bool carveCaves{true};        /**< Generate the underground layer. */
    bool placeSettlement{true};   /**< Lay a town onto the terrain. */
    bool raiseBuildings{false};   /**< Extrude the plots with the shape grammar. */
    bool growRoads{true};         /**< Grow a road network over it. */
    bool materializeGround{true}; /**< Emit one entity per ground cell (off: props only). */
    bool checkPlayability{true};  /**< Judge the underground against @ref gate. */
};

/**
 * @struct WorldRecipeResult
 * @brief What baking a recipe produced, every field a word.
 *
 * Free of Fixed32 and bool, so the engine test records each field as the same
 * word on both targets, without a conversion that might differ between them.
 *
 * Three signatures rather than one, and that is the point of the gate. The state
 * fold only sees where entities ended up, so a pass that reshapes the terrain
 * without moving a cube — every climate pass, most of erosion — would be
 * invisible to it. Folding the height field and the biome map puts the *grids*
 * under the determinism contract, which is where all the arithmetic that could
 * diverge between targets actually happens.
 */
struct WorldRecipeResult {
    core::u32 entityCount{0u};     /**< Entities materialised by all passes. */
    core::u32 stateSignature{0u};  /**< FNV-1a fold of authoritative Fixed32 entity state. */
    core::u32 heightSignature{0u}; /**< FNV-1a fold of the final height field. */
    core::u32 biomeSignature{0u};  /**< FNV-1a fold of the biome map. */
    core::u32 riverCells{0u};      /**< Cells carved as river. */
    core::u32 roadCells{0u};       /**< Cells the road network occupies. */
    core::u32 lakeCells{0u};       /**< Cells holding standing water. */
    core::u32 dungeonFloor{0u};    /**< Open cells in the underground layer. */
    core::u32 settlementPlots{0u}; /**< Building footprints laid out. */
    core::u32 gateReachable{0u};   /**< 1 if the underground's goal is reachable. */
    core::u32 gateVisited{0u};     /**< Cells the gate's flood actually reached. */
    core::u32 gatePathLength{0u};  /**< Steps from entrance to exit. */
    core::u32 ok{0u};              /**< 1 if the world is non-empty AND passes its gate. */

    /**
     * @brief Whether two bakes produced the same world, field for field.
     */
    [[nodiscard]] bool operator==(const WorldRecipeResult &) const noexcept = default;
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
 * Where it departs from the defaults, it says why:
 *
 * - Fewer erosion iterations than the desktop defaults: the shapes converge quickly
 *   on a 24x24 field, and ring 0 pays for every pass in boot time.
 * - Three road rewrite rounds: a 24x24 world is one district wide; three rewrite
 *   rounds fill it without the turtle spending most of its string refused against
 *   the border.
 * - A milder gate: a 24x24 cave is small; the desktop defaults would reject most
 *   seeds for being cramped rather than for being unplayable.
 *
 * @return The parity recipe.
 */
[[nodiscard]] constexpr WorldRecipe parityWorldRecipe() noexcept;

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
 * @brief Configures @p builder with every enabled pass of @p recipe, in order.
 *
 * The pass order is the contract, and it lives here so that the cartridge, the
 * editor, the parity gate and the in-kernel viewer all run the SAME pipeline.
 * A caller that wants the grids and not the entities stops after this; one that
 * wants a populated world calls @ref bakeWorld, which is this plus
 * materialisation.
 *
 * It is not a style preference: two callers ordering erosion and rivers differently
 * would produce two different worlds from the same recipe, and the parity gate would
 * be comparing worlds that were never meant to match. It is a free function rather
 * than a step inside @ref bakeWorld because a caller may want the GRIDS without the
 * entities — a viewer draws a heightfield, it does not need one entity per ground
 * cell. Before this existed the in-kernel viewer built its own pipeline by hand, which
 * is exactly how a project ends up with two generators and a gate that exercises the
 * one nothing else runs.
 *
 * The order, and why each pass sits where it does:
 *
 * - The terrain, then the measured relief when one is given, then the normalisation.
 * - Terracing before erosion, because erosion is what softens the steps back into
 *   something that looks cut rather than stamped. The other way round gives sharp
 *   benches on a smooth hill, which reads as a rendering artefact.
 * - Both erosions, then the rivers.
 * - The ground clearance. The world is put in the frame the physics expects HERE —
 *   after erosion and the rivers, because the lowest cell is not knowable before
 *   them, and before a single entity exists, because moving the ground under things
 *   already standing on it means chasing a list that only ever grows. See
 *   @ref WorldRecipe::groundClearance.
 * - Provinces before the climate block: the districting is a partition of the
 *   SURFACE, so it belongs with the terrain that is now final, and a settlement laid
 *   out later can be anchored on it.
 * - The climate and the biomes, then the scatter.
 * - The underground. @ref CaveKind::Auto is resolved here from evidence: the map that
 *   has already been built, not a guess about it. The moisture field is computed by
 *   the climate pass above, and whether people live here is a thing the recipe states
 *   outright. Reading a field that exists is the difference between a procedural
 *   default and one invented for the occasion. The switch that picks the generator
 *   still names @c Auto, with no default, so a new kind cannot be added without the
 *   compiler pointing at it.
 * - The settlement, then its buildings: the grammar needs the plots the settlement
 *   laid out, so it cannot run before one exists. Raising a town is what turns a
 *   painted footprint into a silhouette.
 * - The roads, after the settlement, never before: the road field is anchored on the
 *   districts the town laid out, so growing first would steer the network by a town
 *   that does not exist yet. And the verges after the roads they decorate.
 * - The playability verdict.
 *
 * @warning **Measured ground displaces the LOWEST frequency, so it lands right after
 *          the terrain and not later.** Erosion softening real relief is what makes a
 *          survey read as terrain rather than as a photograph pasted onto one, and
 *          normalising afterwards keeps the whole field in the frame the rest of the
 *          recipe expects. Injected rather than named by the recipe, for the same
 *          reason `ChunkParams::relief` is a pointer beside serialisable noise: a
 *          recipe is a wire object and a survey is data that has to come from outside
 *          it.
 *
 * @param builder Destination builder, freshly constructed with the recipe's seed.
 * @param recipe  The passes to configure.
 * @return The vertical shift @ref WorldRecipe::groundClearance applied, 0 when none.
 *         A caller that reads absolute heights afterwards — where the sea is, what
 *         counts as too deep to walk in — needs it, because the world it is about to
 *         measure is not at the altitude its own recipe named. Returned rather than
 *         recomputed: it is only knowable at the moment erosion finished.
 */
core::f32 applyRecipe(WorldBuilder &builder, const WorldRecipe &recipe, const ReliefBlend *relief = nullptr);

/**
 * @brief Runs every enabled pass of @p recipe into @p registry and folds it.
 *
 * A recipe that declines the check is trivially "not failing" it (@c gateReachable
 * reads 1), so the ok flag stays a statement about the world, not about the gate.
 *
 * @param registry Destination world (should be empty on entry).
 * @param recipe   The passes to run.
 * @return Entity count, the three signatures, and the playability verdict.
 */
WorldRecipeResult bakeWorld(ecs::Registry &registry, const WorldRecipe &recipe);

} // namespace lpl::procgen

#    include "WorldRecipe.inl"

#endif // LPL_PROCGEN_WORLDRECIPE_HPP
