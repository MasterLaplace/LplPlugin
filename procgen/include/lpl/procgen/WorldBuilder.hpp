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
#    include <lpl/procgen/Aggregation.hpp>
#    include <lpl/procgen/Biome.hpp>
#    include <lpl/procgen/CaveSystem.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/procgen/Dungeon.hpp>
#    include <lpl/procgen/Erosion.hpp>
#    include <lpl/procgen/Extrusion.hpp>
#    include <lpl/procgen/FixedMath.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Hydrology.hpp>
#    include <lpl/procgen/LSystem.hpp>
#    include <lpl/procgen/QualityGate.hpp>
#    include <lpl/procgen/Random.hpp>
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
    BiomeId biome{BiomeId::Grassland}; ///< Biome this rule applies to.
    core::f32 density{0.05f};          ///< Share of the biome's area to cover, in [0, 1].
    core::f32 halfExtent{0.25f};       ///< Prop AABB half-size.
    core::f32 maxSlope{2.0f};          ///< Steepest ground it will stand on.
    core::f32 minMoisture{0.0f};       ///< Driest ground it tolerates.
    core::f32 maxMoisture{1.0f};       ///< Wettest ground it tolerates.
    core::f32 moistureAffinity{0.5f};  ///< How much wetter ground packs it closer, in [0, 1].
    core::u32 tag{0u};                 ///< Caller-defined kind (tree, rock, ...).

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
 * Both halves are here: the field is baked from the settlement's districts, and
 * the ground vetoes afterwards.
 */
struct RoadParams {
    core::u32 seed{0u};          ///< 0 derives a stream from the world seed.
    core::u32 iterations{4u};    ///< Grammar rewrite rounds.
    core::u32 stepLength{3u};    ///< Cells advanced per segment.
    core::f32 conform{0.75f};    ///< How strongly the field overrides the turtle, in [0, 1].
    core::f32 maxSlope{1.2f};    ///< Steepest ground a road will climb.
    core::f32 minHeight{0.0f};   ///< Below this the road would be under water.
    core::u32 gridDistricts{2u}; ///< Planned districts imposing a bearing on their surroundings.
    bool arterials{true};        ///< Route trunk roads between district centres before growing streets.
};

/**
 * @struct WorldStats
 * @brief A summary of what was built.
 */
struct BuiltWorldStats {
    core::u32 terrainCells{0u};      ///< Heightfield cells.
    core::u32 terrainEntities{0u};   ///< Entities created for the ground.
    core::u32 propEntities{0u};      ///< Entities created by scatter rules.
    core::u32 riverCells{0u};        ///< Cells carved as river.
    core::u32 dungeonFloor{0u};      ///< Walkable dungeon cells, when one was generated.
    bool dungeonConnected{false};    ///< Whether that dungeon is fully navigable.
    core::u32 regionCount{0u};       ///< Surface regions, when partitioned.
    core::u32 settlementPlots{0u};   ///< Buildings laid out, when a settlement was placed.
    bool settlementConnected{false}; ///< Whether its streets form one network.
    core::u32 lakeCells{0u};         ///< Cells holding standing water.
    core::u32 roadCells{0u};         ///< Cells the road network occupies, when grown.
    core::u32 townVoxels{0u};        ///< Solid voxels in the raised town, when extruded.
    core::u32 roadsideModules{0u};   ///< Modules placed along the roads, when decorated.
    core::u32 undergroundVoxels{0u}; ///< Solid voxels in the raised underground, when extruded.
    core::u32 caveLayers{0u};        ///< Layers of the cave system, when dug.
    core::u32 caveEntrances{0u};     ///< Shafts that pierce the surface.
    core::u32 caveHollow{0u};        ///< Hollow cells across every cave layer.
    core::u32 caveReachable{0u};     ///< Hollow cells reachable from an entrance.
    core::u32 heightSignature{0u};   ///< Fold of the final heightfield.
    core::u32 biomeSignature{0u};    ///< Fold of the biome map.
    core::u32 climateSignature{0u};  ///< Fold of the six climate axes.
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

    // ── Terrain ─────────────────────────────────────────────────────────────

    /// @brief Generates a heightfield with default noise settings.
    WorldBuilder &terrain(core::u32 width, core::u32 depth);

    /// @brief Generates a heightfield with explicit noise settings.
    WorldBuilder &terrain(core::u32 width, core::u32 depth, const NoiseParams &noise);

    /// @brief Adds a second noise layer (ridges, roughness) over the terrain.
    WorldBuilder &addLayer(const NoiseParams &noise);

    /// @brief Rescales the terrain into [@p low, @p high] world units.
    WorldBuilder &normalize(core::f32 low, core::f32 high);

    /// @brief Quantises the terrain into @p steps plateaus.
    WorldBuilder &terraces(core::u32 steps);

    // ── Erosion ─────────────────────────────────────────────────────────────

    /// @brief Runs both erosion models with defaults.
    WorldBuilder &erode();

    /// @brief Runs talus-angle erosion only.
    WorldBuilder &erodeThermal(const ThermalErosionParams &params);

    /// @brief Runs rainfall erosion only.
    WorldBuilder &erodeHydraulic(const HydraulicErosionParams &params);

    // ── Water ───────────────────────────────────────────────────────────────

    /// @brief Routes drainage and carves rivers with defaults.
    WorldBuilder &rivers();

    /// @brief Routes drainage and carves rivers with explicit settings.
    WorldBuilder &rivers(const RiverParams &params);

    /// @brief Raises everything below @p level to it, making a flat sea floor.
    WorldBuilder &seaLevel(core::f32 level);

    // ── Climate ─────────────────────────────────────────────────────────────

    /// @brief Classifies biomes with defaults (runs drainage/moisture if needed).
    WorldBuilder &biomes();

    /// @brief Classifies biomes with explicit thresholds.
    WorldBuilder &biomes(const BiomeParams &params);

    /// @brief Sets the climate inputs (wind, sea influence, rain shadow).
    WorldBuilder &climate(const MoistureParams &params);

    /// @brief Sets how the six climate axes are shaped, and invalidates the biomes.
    WorldBuilder &climateAxes(const ClimateParams &params);

    // ── Population ──────────────────────────────────────────────────────────

    /// @brief Adds a scatter rule for one biome.
    WorldBuilder &scatterInBiome(BiomeId biome, core::f32 density);

    /// @brief Adds a fully specified scatter rule.
    WorldBuilder &scatter(const ScatterRule &rule);

    // ── Underground ─────────────────────────────────────────────────────────

    /// @brief Generates a BSP dungeon alongside the surface.
    WorldBuilder &dungeon(const BspDungeonParams &params);

    /// @brief Generates a cave system alongside the surface.
    WorldBuilder &caves(const CaveParams &params);

    /// @brief Grows a dendritic cave by diffusion-limited aggregation.
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

    // ── Civilisation ────────────────────────────────────────────────────────

    /// @brief Partitions the surface into regions (provinces, territories).
    WorldBuilder &regions(core::u32 regionSize);

    /// @brief Partitions the surface with an explicit metric and warp.
    WorldBuilder &regions(const VoronoiParams &params);

    /// @brief Lays a settlement onto the terrain, refusing slopes and water.
    WorldBuilder &settlement(const SettlementParams &params);

    /// @brief Grows a road network with default settings.
    WorldBuilder &roads();

    /**
     * @brief Grows a road network across the terrain.
     *
     * Runs after @ref settlement when there is one, so the field can be anchored
     * on the districts the town already has. Without a settlement the field falls
     * back to a single radial centre at the middle of the map, which is what a
     * road network with nothing to serve looks like.
     *
     * @param params Grammar, steering and the ground's veto.
     */
    WorldBuilder &roads(const RoadParams &params);

    // ── Volume ──────────────────────────────────────────────────────────────

    /**
     * @brief Raises the settlement plan into a voxel volume.
     *
     * The 2.5D step: a plan plus a height rule becomes a volume, at the cost of a
     * planar solve. Plots become blocks whose height varies with the footprint —
     * a bigger building is a taller one — roads and plazas stay flat, and
     * unbuildable ground stays empty.
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
     * @param params Grammar parameters.
     */
    WorldBuilder &buildings(const BuildingGrammarParams &params);

    /**
     * @brief Places fence or lamp modules along the road network.
     *
     * The linear application of the same grammar the buildings use.
     *
     * @param grammarText The `{[A,P]:2,...}*,[G,P]` string.
     * @param levels      Height of the decoration volume.
     */
    WorldBuilder &roadside(const char *grammarText, core::u32 levels);

    // ── Tiles ───────────────────────────────────────────────────────────────

    /**
     * @brief Solves a tile arrangement over the terrain's footprint.
     *
     * When a biome map exists it is used as a preset, pinning water and rock so the
     * solver decorates the world it was given rather than inventing an unrelated
     * one. Without that, a tile pass and a heightfield pass produce two worlds that
     * merely happen to be the same size.
     *
     * @param tiles  Adjacency rules.
     * @param params Solver budgets; width and depth are taken from the terrain.
     */
    WorldBuilder &tiles(const TileSet &tiles, const WfcParams &params);

    // ── Validation ──────────────────────────────────────────────────────────

    /**
     * @brief Measures the generated dungeon against playability criteria.
     *
     * The module's own rule is that a bottom-up generator guarantees nothing until
     * something has checked it. This is that check, reachable from the fluent form:
     * the verdict lands in @ref lastQuality and @ref gatePassed, so a caller can
     * try the next seed instead of shipping a level nobody can finish.
     *
     * @param criteria What the level has to satisfy.
     */
    WorldBuilder &validate(const GateCriteria &criteria);

    // ── Output ──────────────────────────────────────────────────────────────

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
     * @param outIds   Receives the created entity ids, in placement order (may be null).
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

    // ── Inspection ──────────────────────────────────────────────────────────

    [[nodiscard]] const Heightfield &heightfield() const noexcept { return _height; }
    [[nodiscard]] const Heightfield &moisture() const noexcept { return _moisture; }
    [[nodiscard]] const BiomeMap &biomeMap() const noexcept { return _biomes; }
    /// @brief The six climate axes the classification read.
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
    /// @brief Whether the last tile solve actually satisfied every adjacency rule.
    [[nodiscard]] bool tilesSolved() const noexcept { return _tilesSolved; }
    [[nodiscard]] const LevelQuality &lastQuality() const noexcept { return _quality; }
    [[nodiscard]] bool gatePassed() const noexcept { return _gatePassed; }
    [[nodiscard]] core::u32 seed() const noexcept { return _seed; }

    /// Standing water as a 0/1 mask, filled by @ref biomes.
    [[nodiscard]] const Grid<core::u8> &lakeMap() const noexcept { return _lakes; }
    /// @brief Which cells the drainage marked as river.
    [[nodiscard]] const Grid<core::u8> &riverMap() const noexcept { return _rivers; }
    /// The road network as a 0/1 mask, empty until @ref roads has run.
    [[nodiscard]] const Grid<core::u8> &roadMap() const noexcept { return _roads; }
    /// The raised town, empty until @ref extrudeTown has run.
    [[nodiscard]] const VoxelVolume &townVolume() const noexcept { return _townVolume; }
    /// @brief The layered cave system, when one was dug.
    [[nodiscard]] const CaveSystem &caves() const noexcept { return _caveSystem; }
    /// @brief Fences and lamps placed along the roads.
    [[nodiscard]] const VoxelVolume &roadsideVolume() const noexcept { return _roadsideVolume; }
    /// The raised underground, empty until @ref extrudeUnderground has run.
    [[nodiscard]] const VoxelVolume &undergroundVolume() const noexcept { return _undergroundVolume; }

    /// @brief Spacing between grid cells in world units (default 1).
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
    void ensureMoisture();
    void ensureClimate();
    void ensureBiomes();

    /// One cell that will become an entity.
    struct Placement {
        math::Fixed32 x, y, z;
        math::Fixed32 halfExtent;
        bool collidable{false};
    };

    /// Grid cell to world position, honouring cellSize and centring the map.
    void cellToWorld(core::u32 x, core::u32 z, math::Fixed32 &outX, math::Fixed32 &outZ) const;

    /// Appends every scatter rule's accepted placements; returns how many.
    core::u32 collectProps(lpl::pmr::vector<Placement> &placements);

    /// Creates and fills one entity per placement, recording ids when asked.
    void emit(ecs::Registry &registry, const lpl::pmr::vector<Placement> &placements,
              lpl::pmr::vector<ecs::EntityId> *outIds);

    /// Writes one archetype's placements into that archetype's own partition.
    /**
     * @brief Writes placements onto the entities that were created for them.
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

    /// Cells a rule's biome, slope and moisture filters admit.
    [[nodiscard]] lpl::pmr::vector<core::u32> eligibleCells(const ScatterRule &rule) const;

    /// Thins @p cells down to a blue-noise subset honouring the rule's density.
    void selectBlueNoise(const ScatterRule &rule, lpl::pmr::vector<core::u32> &cells, Random random) const;

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
    bool _gatePassed{true};
};

} // namespace lpl::procgen

#endif // LPL_PROCGEN_WORLDBUILDER_HPP
