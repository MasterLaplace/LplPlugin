/**
 * @file WorldBuilder.hpp
 * @brief The one thing a caller should need to know about procgen.
 *
 * Every pass in this module is usable on its own, and none of them should have
 * to be. Building a world is a sentence, not an assembly job:
 *
 * @code
 * lpl::ecs::Registry registry;
 * lpl::procgen::WorldBuilder{1337}
 *     .terrain(128, 128)
 *     .erode()
 *     .rivers()
 *     .biomes()
 *     .scatterInBiome(lpl::procgen::BiomeId::Forest, 0.08f)
 *     .materialize(registry);
 * @endcode
 *
 * Every step has defaults that produce something reasonable, so a caller states
 * only what it cares about. Every step also has an explicit-parameter overload,
 * so nothing is hidden behind the convenience — the fluent form is a shortcut,
 * not a different engine.
 *
 * Order is enforced rather than assumed. `rivers()` needs drainage, which needs
 * terrain; `biomes()` needs moisture, which rivers produce. Calling them out of
 * order is not an error to diagnose at the call site: the builder runs whatever
 * a step depends on if it has not run yet. That is what makes the short form
 * safe to write.
 *
 * The builder holds the intermediate grids, so they are all still available
 * afterwards (@ref heightfield, @ref biomeMap, @ref drainage) for a renderer, a
 * minimap, or a test that wants to check an invariant.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_WORLDBUILDER_HPP
#    define LPL_PROCGEN_WORLDBUILDER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedMath.hpp>
#    include <lpl/math/Random.hpp>
#    include <lpl/procgen/Aggregation.hpp>
#    include <lpl/procgen/Biome.hpp>
#    include <lpl/procgen/CaveSystem.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/procgen/Dungeon.hpp>
#    include <lpl/procgen/Erosion.hpp>
#    include <lpl/procgen/Extrusion.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Hydrology.hpp>
#    include <lpl/procgen/LSystem.hpp>
#    include <lpl/procgen/QualityGate.hpp>
#    include <lpl/procgen/Routing.hpp>
#    include <lpl/procgen/Settlement.hpp>
#    include <lpl/procgen/ShapeGrammar.hpp>
#    include <lpl/procgen/Voronoi.hpp>
#    include <lpl/procgen/WaveFunctionCollapse.hpp>
#    include <lpl/std/vector.hpp>

#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/ecs/Entity.hpp>

#    include <span>

namespace lpl::ecs {
class Registry;
}

namespace lpl::procgen {

/**
 * @struct ScatterRule
 * @brief Where a kind of prop is allowed to appear, and how densely.
 *
 * Placement is blue noise, not a per-cell coin flip. A coin flip is white noise:
 * it puts two trees in the same square as readily as it leaves a clearing, so
 * trunks intersect and the result reads as static rather than as vegetation. Both
 * of the surveys this module follows reject it explicitly and name Poisson-disk
 * sampling as the replacement, and the module already had a sampler for it.
 *
 * The exclusion radius is not fixed either. It scales with how suitable the cell
 * is — dense where the ground is flat and well watered, sparse where it is steep
 * or dry — which is the variable-radius Poisson distribution the literature
 * describes, and what produces a thinning tree line instead of a hard edge.
 */
struct ScatterRule {
    BiomeId biome{BiomeId::Grassland}; /**< Biome this rule applies to. */
    core::f32 density{0.05f};          /**< Share of the biome's area to cover, in [0, 1]. */
    core::f32 halfExtent{0.25f};       /**< Prop AABB half-size. */
    core::f32 maxSlope{2.0f};          /**< Steepest ground it will stand on. */
    core::f32 minMoisture{0.0f};       /**< Driest ground it tolerates. */
    core::f32 maxMoisture{1.0f};       /**< Wettest ground it tolerates. */
    core::f32 moistureAffinity{0.5f};  /**< How much wetter ground packs it closer, in [0, 1]. */
    core::u32 tag{0u};                 /**< Caller-defined kind (tree, rock, ...). */

    /**
     * @brief Height above which this prop thins out, as a share of the map's range.
     *
     * The tree line. Above it the exclusion radius grows without bound, so the
     * stand does not stop at a drawn contour — it thins, which is what a real
     * tree line looks like. 1 disables it.
     */
    core::f32 treeLine{1.0f};

    /**
     * @brief How sharply density falls once past @ref treeLine.
     *
     * Higher is a harder edge. 0 leaves the prop indifferent to altitude.
     */
    core::f32 altitudeFalloff{0.0f};

    /**
     * @brief Furthest this prop grows from running water, in cells.
     *
     * The phreatophyte case the flora survey names: willows and mangroves live on
     * the water's edge and nowhere else. 0 disables the test, which is what
     * anything that does not care about rivers wants.
     */
    core::u32 maxRiverDistance{0u};

    /**
     * @brief Share of the world's regions this prop is allowed to exist in, in [0, 1].
     *
     * Endemism. A species restricted to a biome appears everywhere that biome
     * does, which makes a large world feel uniform: the same forest, over and
     * over. Drawing a subset of *regions* instead means crossing a mountain range
     * changes what grows, and that a distant valley can hold something found
     * nowhere else — which is the only reason to walk to it.
     *
     * 1 means cosmopolitan: present wherever the biome allows. The draw is keyed
     * to the world seed and the rule's tag, so the same world always endows the
     * same regions.
     */
    core::f32 endemicShare{1.0f};
    /**
     * @brief Whether this prop is something bodies bump into.
     *
     * A collidable prop is given the full physics archetype with a mass of **zero**,
     * which is how the solver is told "immovable": it inverts mass as
     * @f$1/m@f$ guarded to zero, so a zero-mass party receives none of the positional
     * correction and none of the impulse, and the moving party receives all of it. A
     * tree stops a boulder and does not itself budge — without needing a flag, a
     * separate code path, or a second collision system.
     *
     * Off by default. A prop that does not need to be hit costs nothing this way, and
     * turning it on changes the entity's archetype, which a caller should ask for
     * rather than inherit.
     */
    bool collidable{false};
};

/**
 * @struct RoadParams
 * @brief A road network grown by a grammar and steered by a tensor field.
 *
 * Roads are the one structure a pure L-system cannot produce on its own. A
 * grammar rewrites symbols; it has no idea where the town is, where the river
 * runs, or that a 40-degree slope is not a street. Parish and Müller's answer is
 * a grammar under tension between **global goals** — a field that says which way
 * a road here ought to run — and **local constraints** — the ground refusing it.
 * Both halves are here: the field is baked from the districts of the world's
 * regions, and the ground vetoes afterwards.
 */
struct RoadParams {
    core::u32 seed{0u};          /**< 0 derives a stream from the world seed. */
    core::u32 iterations{4u};    /**< Grammar rewrite rounds. */
    core::u32 stepLength{3u};    /**< Cells advanced per segment. */
    core::f32 conform{0.75f};    /**< How strongly the field overrides the turtle, in [0, 1]. */
    core::f32 maxSlope{1.2f};    /**< Steepest ground a road will climb. */
    core::f32 minHeight{0.0f};   /**< Below this the road would be under water. */
    core::u32 gridDistricts{2u}; /**< Planned districts imposing a bearing on their surroundings. */
    bool arterials{true};        /**< Route trunk roads between district centres before growing streets. */
};

/**
 * @struct BuiltWorldStats
 * @brief A summary of what was built.
 */
struct BuiltWorldStats {
    core::u32 terrainCells{0u};      /**< Heightfield cells. */
    core::u32 terrainEntities{0u};   /**< Entities created for the ground. */
    core::u32 propEntities{0u};      /**< Entities created by scatter rules. */
    core::u32 riverCells{0u};        /**< Cells carved as river. */
    core::u32 dungeonFloor{0u};      /**< Walkable dungeon cells, when one was generated. */
    bool dungeonConnected{false};    /**< Whether that dungeon is fully navigable. */
    core::u32 regionCount{0u};       /**< Surface regions, when partitioned. */
    core::u32 settlementPlots{0u};   /**< Buildings laid out, when a settlement was placed. */
    bool settlementConnected{false}; /**< Whether its streets form one network. */
    core::u32 lakeCells{0u};         /**< Cells holding standing water. */
    core::u32 roadCells{0u};         /**< Cells the road network occupies, when grown. */
    core::u32 townVoxels{0u};        /**< Solid voxels in the raised town, when extruded. */
    core::u32 roadsideModules{0u};   /**< Modules placed along the roads, when decorated. */
    core::u32 undergroundVoxels{0u}; /**< Solid voxels in the raised underground, when extruded. */
    core::u32 caveLayers{0u};        /**< Layers of the cave system, when dug. */
    core::u32 caveEntrances{0u};     /**< Shafts that pierce the surface. */
    core::u32 caveHollow{0u};        /**< Hollow cells across every cave layer. */
    core::u32 caveReachable{0u};     /**< Hollow cells reachable from an entrance. */
    core::u32 heightSignature{0u};   /**< Fold of the final heightfield. */
    core::u32 biomeSignature{0u};    /**< Fold of the biome map. */
    core::u32 climateSignature{0u};  /**< Fold of the six climate axes. */
};

/**
 * @class WorldBuilder
 * @brief Composes the procedural passes into one world.
 */
class WorldBuilder {
public:
    /**
     * @brief Starts a world.
     * @param seed Master seed; every pass derives its own stream from it.
     */
    explicit WorldBuilder(core::u32 seed) noexcept : _seed(seed) {}

    /**
     * @name Terrain
     * @{
     */
    /**
     * @brief Generates a heightfield with default noise settings.
     */
    WorldBuilder &terrain(core::u32 width, core::u32 depth);

    /**
     * @brief Generates a heightfield with explicit noise settings.
     *
     * A caller that leaves the noise seed at its default still gets a world that
     * follows the builder's seed — otherwise every world would be the same one
     * until someone noticed the second seed.
     *
     * After @ref chunk, the terrain samples the SAME function as a whole map, only
     * at absolute world coordinates. A non-square chunk is not a chunk: the
     * coordinate system would stop being a grid, so a @p width that differs from
     * @p depth falls back to the plain path rather than lie about it.
     */
    WorldBuilder &terrain(core::u32 width, core::u32 depth, const NoiseParams &noise);

    /**
     * @brief Lays measured ground over the terrain already generated.
     *
     * @warning **The same blend the streamed world uses, called per cell** -- @ref applyRelief, not a
     * copy of it. A bounded map and the streamed world it is a map OF must agree at every cell, and
     * two implementations of "how does a survey meet invented ground" would be two different places
     * wearing one name. This file's own header already makes that argument about the pass order.
     *
     * @warning It runs BEFORE normalisation, terracing and erosion, because it displaces the LOWEST
     * frequency: erosion softening real ground is what makes a survey look like terrain rather than
     * like a photograph pasted onto one, and normalising afterwards keeps the whole field in the
     * frame the rest of the recipe expects.
     *
     * @warning A bounded grid cell (x, z) is world cell (x, z). The bounded world is a WINDOW on the
     * same coordinates the streamed one uses, so a survey lands in both at the same place; an origin
     * of its own here would be a second answer to where the ground is.
     *
     * @param blend The survey and its detail layer. A null mosaic is a no-op.
     * @return This builder.
     */
    WorldBuilder &relief(const ReliefBlend &blend);

    /**
     * @brief Adds a second noise layer (ridges, roughness) over the terrain.
     */
    WorldBuilder &addLayer(const NoiseParams &noise);

    /**
     * @brief Rescales the terrain into [@p low, @p high] world units.
     */
    WorldBuilder &normalize(core::f32 low, core::f32 high);

    /**
     * @brief Quantises the terrain into @p steps plateaus.
     */
    WorldBuilder &terraces(core::u32 steps);
    /** @} */

    /**
     * @name Erosion
     * @{
     */
    /**
     * @brief Runs both erosion models with defaults.
     */
    WorldBuilder &erode();

    /**
     * @brief Runs talus-angle erosion only.
     */
    WorldBuilder &erodeThermal(const ThermalErosionParams &params);

    /**
     * @brief Runs rainfall erosion only.
     */
    WorldBuilder &erodeHydraulic(const HydraulicErosionParams &params);
    /** @} */

    /**
     * @name Water
     * @{
     */
    /**
     * @brief Routes drainage and carves rivers with defaults.
     */
    WorldBuilder &rivers();

    /**
     * @brief Routes drainage and carves rivers with explicit settings.
     *
     * Carving changes the terrain, so the drainage that produced it is stale:
     * moisture, climate and biomes are invalidated, so that they reflect the
     * valleys the rivers just cut instead of the slopes that predated them, and
     * the drainage is recomputed at once, because the river mask is read from it.
     *
     * The mask is kept, not just the count. Where the water runs is what a
     * phreatophyte scatter rule and a road router both need to know, and
     * recomputing it from a stale drainage would answer about the wrong terrain.
     */
    WorldBuilder &rivers(const RiverParams &params);

    /**
     * @brief Raises everything below @p level to it, making a flat sea floor.
     */
    WorldBuilder &seaLevel(core::f32 level);
    /** @} */

    /**
     * @name Climate
     * @{
     */
    /**
     * @brief Classifies biomes with defaults (runs drainage/moisture if needed).
     */
    WorldBuilder &biomes();

    /**
     * @brief Classifies biomes with explicit thresholds.
     *
     * The lakes come free with the drainage: Priority-Flood already raised every
     * basin to its spill level, so the difference between the filled surface and
     * the terrain is standing water. Reporting only a raised-cell count and
     * classifying those cells as meadow was throwing the answer away.
     */
    WorldBuilder &biomes(const BiomeParams &params);

    /**
     * @brief Sets the climate inputs (wind, sea influence, rain shadow).
     */
    WorldBuilder &climate(const MoistureParams &params);

    /**
     * @brief Sets how the six climate axes are shaped, and invalidates the biomes.
     */
    WorldBuilder &climateAxes(const ClimateParams &params);
    /** @} */

    /**
     * @name Population
     * @{
     */
    /**
     * @brief Adds a scatter rule for one biome.
     */
    WorldBuilder &scatterInBiome(BiomeId biome, core::f32 density);

    /**
     * @brief Adds a fully specified scatter rule.
     */
    WorldBuilder &scatter(const ScatterRule &rule);
    /** @} */

    /**
     * @name Underground
     * @{
     */
    /**
     * @brief Generates a BSP dungeon alongside the surface.
     */
    WorldBuilder &dungeon(const BspDungeonParams &params);

    /**
     * @brief Generates a cave system alongside the surface.
     */
    WorldBuilder &caves(const CaveParams &params);

    /**
     * @brief Grows a dendritic cave by diffusion-limited aggregation.
     */
    WorldBuilder &dlaCaves(const DlaParams &params);

    /**
     * @brief Digs a layered cave system under the terrain, open to the sky.
     *
     * Replaces the single buried plan with a stack joined by shafts, at least one
     * of which pierces the surface. Without an entrance the whole system is a
     * void no player can reach, which is what the flat version was.
     *
     * @param params Layer count, fill and shaft budget.
     */
    WorldBuilder &caveSystem(const CaveSystemParams &params);
    /** @} */

    /**
     * @name Civilisation
     * @{
     */
    /**
     * @brief Partitions the surface into regions (provinces, territories).
     */
    WorldBuilder &regions(core::u32 regionSize);

    /**
     * @brief Partitions the surface with an explicit metric and warp.
     *
     * The partition covers the terrain, so its extent is taken from the
     * heightfield rather than trusted from the caller.
     */
    WorldBuilder &regions(const VoronoiParams &params);

    /**
     * @brief Lays a settlement onto the terrain, refusing slopes and water.
     *
     * A settlement must match the terrain it stands on, so its extent is taken
     * from the heightfield rather than trusted from the caller.
     */
    WorldBuilder &settlement(const SettlementParams &params);

    /**
     * @brief Grows a road network with default settings.
     */
    WorldBuilder &roads();

    /**
     * @brief Grows a road network across the terrain.
     *
     * Runs after @ref regions when there are some, so the field can be anchored on
     * the districts they lay out. Without regions the field falls back to a single
     * radial centre at the middle of the map, which is what a road network with
     * nothing to serve looks like.
     *
     * The network is laid in three steps:
     *
     * - Global goals: where roads want to run. A road network exists to serve
     *   something; inventing its own centres while a town sits next to it is how a
     *   generator ends up drawing two unrelated worlds on the same grid. The first
     *   @c gridDistricts districts get an imposed bearing, which is the planned
     *   grid; the rest get the radial web of an old centre.
     * - Arterials first: routed, not grown. A grammar has no destination, so the
     *   network it produces connects whatever it happens to reach. The trunk roads
     *   are therefore routed between the places that exist — district centres —
     *   paying for slope and water, and the grammar then adds texture over a
     *   skeleton that already goes somewhere.
     * - Local constraints: the ground has the last word. A cell too steep or below
     *   @c minHeight is erased after drawing rather than avoided during, because a
     *   turtle that refuses a step has to decide what to do instead, and every
     *   answer to that question is a heuristic. Erasing is honest: the grammar
     *   proposed, the terrain disposed.
     *
     * @param params Grammar, steering and the ground's veto.
     */
    WorldBuilder &roads(const RoadParams &params);
    /** @} */

    /**
     * @name Volume
     * @{
     */
    /**
     * @brief Raises the settlement plan into a voxel volume.
     *
     * The 2.5D step: a plan plus a height rule becomes a volume, at the cost of a
     * planar solve. Plots become blocks whose height varies with the footprint —
     * a bigger building is a taller one — roads and plazas stay flat, and
     * unbuildable ground stays empty.
     *
     * One material per cell kind, and a height rule per material. Plots are taller
     * than roads by construction rather than by a special case, which is what
     * makes the plan and the volume one relationship instead of two.
     *
     * The plan alone cannot say that a bigger footprint is a taller building,
     * because every plot cell looks the same to it; the footprints can. They raise
     * only the cells the map still calls Plot, though. A footprint is a rectangle
     * and the map is the truth: parts of that rectangle came back as road, plaza
     * or unbuildable, and raising the rectangle wholesale puts a six-storey block
     * on top of the street that serves it. Measured before the guard: the tallest
     * ROAD column was as tall as the tallest building.
     *
     * @param params Level count, base level and fill mode.
     */
    WorldBuilder &extrudeTown(const ExtrusionParams &params);

    /**
     * @brief Raises the underground plan into a voxel volume.
     *
     * Walls become full-height columns and floor stays open, so the result is a
     * volume a mesher or a collision builder can walk rather than a picture of a
     * plan. Same 2.5D argument as @ref extrudeTown, applied to the layer that
     * needs it most: a cave read as a flat image is not a cave.
     *
     * @param params Level count, base level and fill mode.
     */
    WorldBuilder &extrudeUnderground(const ExtrusionParams &params);

    /**
     * @brief Raises the settlement's plots into articulated buildings.
     *
     * Replaces @ref extrudeTown's flat prisms with a base course, storeys and a
     * roof. A town of prisms reads as a bar chart; the articulation is what makes
     * it read as architecture.
     *
     * The volume is tall enough for the tallest building the parameters can
     * produce, so a storey is never silently clipped: a roof that vanished because
     * the volume was one level short is a bug that looks like a design choice.
     *
     * @param params Grammar parameters.
     */
    WorldBuilder &buildings(const BuildingGrammarParams &params);

    /**
     * @brief Places fence or lamp modules along the road network.
     *
     * The linear application of the same grammar the buildings use. A refused
     * grammar decorates nothing; it does not half-decorate.
     *
     * @param grammarText The `{[A,P]:2,...}*,[G,P]` string.
     * @param levels      Height of the decoration volume.
     */
    WorldBuilder &roadside(const char *grammarText, core::u32 levels);
    /** @} */

    /**
     * @name Tiles
     * @{
     */
    /**
     * @brief Solves a tile arrangement over the terrain's footprint.
     *
     * When a biome map exists it is used as a preset, pinning water and rock so the
     * solver decorates the world it was given rather than inventing an unrelated
     * one. Without that, a tile pass and a heightfield pass produce two worlds that
     * merely happen to be the same size.
     *
     * A preset can contradict the very rules the solve must satisfy: a rock summit
     * whose neighbour is a lake pins two tiles the tile set forbids from touching,
     * and no arrangement exists. The solver cannot report that usefully — it
     * exhausts its attempts and hands back a grid that violates the rules, silently.
     * So the preset is made SATISFIABLE before it is handed over: where two pinned
     * neighbours may not touch, the earlier one in scan order is released, and the
     * solver fills it with whatever transition the tile set provides. Releasing
     * rather than refusing is the right call here because the pins are a preference
     * (this cell looks like rock) while the adjacency rules are a constraint (rock
     * never abuts water).
     *
     * A solve that did not succeed hands back a grid anyway, and that grid violates
     * the very rules the pass exists to satisfy — measured at seven violations on a
     * 48x48 world before this check existed, with every cell filled so nothing
     * looked wrong. Keeping it silently was the bug: a caller has no way to tell a
     * solved arrangement from a failed one by looking at it, so @ref tilesSolved
     * says which one it got.
     *
     * A preset can be locally consistent and still globally unsatisfiable, so when
     * the pinned solve fails, the pass falls back to an unpinned one. That loses
     * the agreement with the biome map, which is a real loss — and it is strictly
     * better than an arrangement that breaks the adjacency rules, because the rules
     * are what the tiles MEAN.
     *
     * @param tiles  Adjacency rules.
     * @param params Solver budgets; width and depth are taken from the terrain.
     */
    WorldBuilder &tiles(const TileSet &tiles, const WfcParams &params);
    /** @} */

    /**
     * @name Validation
     * @{
     */
    /**
     * @brief Measures the generated dungeon against playability criteria.
     *
     * The module's own rule is that a bottom-up generator guarantees nothing until
     * something has checked it. This is that check, reachable from the fluent form:
     * the verdict lands in @ref lastQuality and @ref gatePassed, so a caller can
     * try the next seed instead of shipping a level nobody can finish.
     *
     * A layered system dug by @ref caveSystem does not fill @ref dungeonMap, so
     * judging the flat map would report zero open cells and fail a world that is
     * perfectly navigable: without a flat map, the layered system is judged. The
     * builder knows which underground it generated; asking the right one is its
     * business rather than the caller's, or every caller repeats the test and one
     * of them gets it wrong.
     *
     * With neither there is nothing to judge. Saying "passed" would be a claim
     * about a level that was never generated, so the verdict is false and the
     * measurements empty.
     *
     * @param criteria What the level has to satisfy.
     */
    WorldBuilder &validate(const GateCriteria &criteria);
    /** @} */

    /**
     * @name Output
     * @{
     */
    /**
     * @brief Creates the entities the world is made of.
     *
     * Ground cells become one cube entity each; scatter rules add props on top.
     * This is the only step that touches the ECS — everything before it works on
     * grids, which is what makes the passes composable and the result inspectable.
     *
     * @param registry Destination world.
     * @return What was built.
     */
    BuiltWorldStats materialize(ecs::Registry &registry);

    /**
     * @brief Creates the scattered props, and nothing else.
     *
     * @ref materialize turns every ground cell into an entity as well, which is right
     * for a world made of cubes and wrong for anything that draws the terrain as a
     * surface: asking for a few hundred trees should not cost sixteen thousand ground
     * entities. Same rules, same streams, same placement — only the ground is left out.
     *
     * @param registry Destination world.
     * @param outIds   Receives the created entity ids (may be null): the decoration
     *                 first, then the collidable props, each in placement order.
     *                 A caller that regenerates its world needs them: the registry has
     *                 no bulk clear, deliberately, so whoever created entities is the
     *                 one who can retire them.
     * @return What was built; @c terrainEntities is zero by construction.
     */
    BuiltWorldStats materializeProps(ecs::Registry &registry, lpl::pmr::vector<ecs::EntityId> *outIds = nullptr);

    /**
     * @brief Runs every pending pass without creating entities.
     *
     * For a caller that wants the grids (a minimap, a server that only needs
     * collision, a test) without paying for entities.
     *
     * @return What would have been built, minus the entity counts.
     */
    BuiltWorldStats bakeGrids();
    /** @} */

    /**
     * @name Inspection
     * @{
     */
    [[nodiscard]] const Heightfield &heightfield() const noexcept { return _height; }
    [[nodiscard]] const Heightfield &moisture() const noexcept { return _moisture; }
    [[nodiscard]] const BiomeMap &biomeMap() const noexcept { return _biomes; }
    /**
     * @brief The six climate axes the classification read.
     */
    [[nodiscard]] const ClimateField &climateField() const noexcept { return _climate; }
    [[nodiscard]] const DrainageNetwork &drainage() const noexcept { return _drainage; }
    [[nodiscard]] const DungeonMap &dungeonMap() const noexcept { return _dungeon; }
    [[nodiscard]] const VoronoiDiagram &regionMap() const noexcept { return _regions; }
    [[nodiscard]] const SettlementMap &settlementMap() const noexcept { return _settlement; }

    /**
     * @brief The building footprints the settlement laid out.
     *
     * The map says which cells are plot; this says which plot each belongs to, and
     * therefore where one building ends and the next begins. Anything that wants to
     * *raise* the town rather than paint it needs the footprints, not the mask.
     */
    [[nodiscard]] const lpl::pmr::vector<BuildingPlot> &plots() const noexcept { return _plots; }
    [[nodiscard]] const TileGrid &tileMap() const noexcept { return _tiles; }
    /**
     * @brief Whether the last tile solve actually satisfied every adjacency rule.
     */
    [[nodiscard]] bool tilesSolved() const noexcept { return _tilesSolved; }
    [[nodiscard]] const LevelQuality &lastQuality() const noexcept { return _quality; }
    /**
     * @brief Whether the last @ref validate passed its criteria; false until validate runs.
     */
    [[nodiscard]] bool gatePassed() const noexcept { return _gatePassed; }
    [[nodiscard]] core::u32 seed() const noexcept { return _seed; }

    /**
     * @brief Standing water as a 0/1 mask, filled by @ref biomes.
     */
    [[nodiscard]] const Grid<core::u8> &lakeMap() const noexcept { return _lakes; }
    /**
     * @brief Which cells the drainage marked as river.
     */
    [[nodiscard]] const Grid<core::u8> &riverMap() const noexcept { return _rivers; }
    /**
     * @brief The road network as a 0/1 mask, empty until @ref roads has run.
     */
    [[nodiscard]] const Grid<core::u8> &roadMap() const noexcept { return _roads; }
    /**
     * @brief The raised town, empty until @ref extrudeTown has run.
     */
    [[nodiscard]] const VoxelVolume &townVolume() const noexcept { return _townVolume; }
    /**
     * @brief The layered cave system, when one was dug.
     */
    [[nodiscard]] const CaveSystem &caves() const noexcept { return _caveSystem; }
    /**
     * @brief Fences and lamps placed along the roads.
     */
    [[nodiscard]] const VoxelVolume &roadsideVolume() const noexcept { return _roadsideVolume; }
    /**
     * @brief The raised underground, empty until @ref extrudeUnderground has run.
     */
    [[nodiscard]] const VoxelVolume &undergroundVolume() const noexcept { return _undergroundVolume; }
    /** @} */

    /**
     * @brief Spacing between grid cells in world units (default 1).
     */
    WorldBuilder &cellSize(core::f32 size) noexcept;

    /**
     * @brief Makes this world one chunk of an endless one.
     *
     * The terrain is then sampled at absolute world coordinates rather than from
     * the grid's own origin, which is the whole trick behind a seamless infinite
     * world: two neighbouring chunks ask the same function about the cells they
     * share and necessarily get the same answer. Set before @ref terrain.
     *
     * @param coord Which chunk of the endless world this is.
     */
    WorldBuilder &chunk(ChunkCoord coord) noexcept;

private:
    void ensureTerrain();
    void ensureDrainage();

    /**
     * @brief Computes the moisture, unless it is still current.
     *
     * Climate has to agree with the classification about where the sea is, or the
     * coastal term would measure the distance to a shoreline the biome map does
     * not have: the moisture takes the biome parameters' sea level.
     */
    void ensureMoisture();

    /**
     * @brief Computes the six climate axes, unless they are still current.
     *
     * One sea level for the whole pipeline. The moisture's coast term, the
     * continentalness axis and the ocean test all have to agree about where the
     * water is, or the map has three shorelines in three different places.
     */
    void ensureClimate();

    void ensureBiomes();

    /** One cell that will become an entity. */
    struct Placement {
        math::Fixed32 x, y, z;
        math::Fixed32 halfExtent;
        bool collidable{false};
    };

    /**
     * @brief Grid cell to world position, honouring cellSize and centring the map.
     */
    void cellToWorld(core::u32 x, core::u32 z, math::Fixed32 &outX, math::Fixed32 &outZ) const;

    /**
     * @brief Appends every scatter rule's accepted placements; returns how many.
     *
     * One stream per rule so adding a rule cannot shift the others.
     */
    core::u32 collectProps(lpl::pmr::vector<Placement> &placements);

    /**
     * @brief Creates and fills one entity per placement, recording ids when asked.
     *
     * Two archetypes, filled in two passes. A collidable prop needs Velocity and
     * Mass for the physics to see it at all — the solver skips any partition
     * missing them — so it cannot share an archetype with decoration, and
     * interleaving the two would scatter each kind across both partitions with no
     * way to tell which row is which.
     *
     * A placement whose entity could not be created is left out of the fill, so
     * each entity receives its own placement rather than the next one's.
     */
    void emit(ecs::Registry &registry, const lpl::pmr::vector<Placement> &placements,
              lpl::pmr::vector<ecs::EntityId> *outIds);

    /**
     * @brief Writes placements onto the entities that were created for them.
     *
     * Writes through IDENTITY, never through row order. A collidable prop's
     * archetype is {Position, Velocity, Mass, AABB} — exactly a loose body's,
     * because that is what the solver requires to see either of them. So props and
     * bodies share a partition, and the earlier version of this function walked
     * that partition from its first row and wrote the first `expected` of them.
     * Whoever else owned those rows lost them.
     *
     * Measured, not reasoned about: with twenty-four bodies already in the world,
     * scattering twenty-three trees overwrote TWENTY-THREE of them — their mass
     * became a prop's zero and their position became a tree's. In the viewer that
     * is the reported symptom exactly: regenerate, and boulders hang motionless in
     * mid-air. They were never stuck. A body with zero mass is one gravity does not
     * apply to, and it stops where it last was.
     *
     * Resolving each created entity costs one lookup per prop and removes the
     * assumption entirely: an entity is written where the registry says it lives.
     *
     * Both buffers are written, like every other generator: the simulation reads
     * the front buffer on its first tick and would otherwise start from
     * uninitialised memory. A collidable prop gets mass zero, the solver's word for
     * immovable, and a zero velocity that gravity never touches (it too is gated
     * on mass) keeps it that way. Nothing else is needed to make a tree an
     * obstacle.
     *
     * @param registry   World holding the entities.
     * @param archetype  Archetype the entities were created in.
     * @param placements What to write.
     * @param created    The entities, in the same order as @p placements. Identity,
     *                   not row order: props share an archetype with loose bodies,
     *                   and addressing rows by position overwrites whatever else
     *                   lives in the partition.
     */
    void fillPlacements(ecs::Registry &registry, const ecs::Archetype &archetype,
                        const lpl::pmr::vector<Placement> &placements, const lpl::pmr::vector<ecs::EntityId> &created);

    /**
     * @brief Cells a rule's biome, slope and moisture filters admit.
     *
     * Nothing stands on a cliff. Checking slope here rather than letting physics
     * sort it out keeps the world plausible before a single tick has run.
     *
     * Endemism: the regions this species was granted are drawn once per rule from
     * the world seed and the rule's tag, so the answer is a property of the world
     * rather than of when the rule happened to run. The raw Q16.16 word IS the
     * share scaled by 65536, so it is the threshold already. Multiplying by
     * fromInt(0x10000) would have been the natural spelling and would have
     * overflowed Fixed32 to zero — granting the species nothing, everywhere,
     * silently. A species granted nowhere is not endemic, it is extinct — and
     * silently dropping a rule is exactly the kind of nothing-happened bug that
     * hides for months — so it is given one region, picked by the next draw of its
     * own stream.
     *
     * Hygrometry: the distance to running water is computed only when a rule asks.
     *
     * The tree line is deliberately NOT tested here. A cell above it is still
     * eligible; it just ends up with a larger exclusion radius in
     * @ref selectBlueNoise, so the stand thins. Rejecting outright would draw a
     * contour line across the mountainside.
     */
    [[nodiscard]] lpl::pmr::vector<core::u32> eligibleCells(const ScatterRule &rule) const;

    /**
     * @brief Thins @p cells down to a blue-noise subset honouring the rule's density.
     *
     * The base spacing is the one that would put the target count of points in
     * this many cells if they were on a square lattice. Deriving it from the
     * requested density rather than asking for a radius keeps the parameter
     * meaning "how much of the ground is covered", which is the thing a caller
     * actually knows.
     *
     * Dart-throwing against an occupancy grid. Candidates are drawn from the
     * eligible set in a shuffled order, so the result does not favour the top-left
     * corner the way a raster sweep does, and each is accepted only if no accepted
     * point lies within its exclusion radius. The distance is squared, so the
     * exclusion zone is a disc rather than the square a Chebyshev test would give
     * — a square one leaves visible rows and columns in the result.
     *
     * Variable radius: wetter ground supports a denser stand, and thin cold air
     * supports a sparser one. This is the Lipschitz-constrained density function
     * the flora survey describes, reduced to the two inputs that matter here. Past
     * the tree line the radius grows with how far past it the cell sits, so the
     * stand THINS instead of stopping at a drawn contour. A hard cut in
     * @ref eligibleCells would put a straight edge across a mountainside, which no
     * forest has ever done.
     *
     * The cells come back in scan order rather than in the shuffled draw order:
     * the entities are written to chunks in this sequence, so a stable order keeps
     * the world fold independent of how the shuffle happened to land. They are
     * read back off the occupancy grid rather than by sorting the accepted list.
     * The grid is already indexed by position, so one sweep gives scan order for
     * free — sorting would be the one quadratic step in an otherwise linear pass.
     */
    void selectBlueNoise(const ScatterRule &rule, lpl::pmr::vector<core::u32> &cells, math::Random random) const;

    core::u32 _seed;
    core::f32 _cellSize{1.0f};

    Heightfield _height;
    Heightfield _moisture;
    ClimateField _climate;
    BiomeMap _biomes;
    DrainageNetwork _drainage;
    DungeonMap _dungeon;
    VoronoiDiagram _regions;
    SettlementMap _settlement;
    lpl::pmr::vector<BuildingPlot> _plots;
    Grid<core::u8> _lakes;
    Grid<core::u8> _rivers;
    Grid<core::u8> _roads;
    VoxelVolume _townVolume;
    VoxelVolume _roadsideVolume;
    CaveSystem _caveSystem;
    VoxelVolume _undergroundVolume;
    TileGrid _tiles;
    LevelQuality _quality{};

    BiomeParams _biomeParams{};
    MoistureParams _moistureParams{};
    ClimateParams _climateParams{};
    lpl::pmr::vector<ScatterRule> _scatterRules;

    bool _drainageReady{false};
    bool _moistureReady{false};
    bool _climateReady{false};
    bool _tilesSolved{false};
    bool _biomesReady{false};
    core::u32 _riverCells{0u};
    core::u32 _dungeonFloor{0u};
    bool _dungeonConnected{false};
    core::u32 _settlementPlots{0u};
    bool _settlementConnected{false};
    core::u32 _lakeCells{0u};
    core::u32 _roadCells{0u};
    core::u32 _townVoxels{0u};
    core::u32 _roadsideModules{0u};
    core::u32 _undergroundVoxels{0u};
    ChunkCoord _chunk{};
    bool _isChunk{false};
    bool _gatePassed{false};
};

} // namespace lpl::procgen

#endif // LPL_PROCGEN_WORLDBUILDER_HPP
