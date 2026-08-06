/**
 * @file Landmark.hpp
 * @brief Placing a THING in an endless world: a cave mouth, a village.
 *
 * Terrain in an endless world is a function of position, so it needs no siting rule.
 * A landmark is not: it occupies an area, it is either here or it is not, and the
 * answer has to be the same for every chunk that overlaps it — including the chunk
 * that holds only its far corner and was generated an hour later on another machine.
 *
 * The mechanism is the one @ref EndlessRiverParams already uses for trunk rivers, one
 * level up: a coarse LANDMARK grid, sparser than a chunk, whose every cell either
 * carries a site or does not as a pure function of its own coordinates. A chunk then
 * asks about the landmark cells within reach of it, which is a bounded search, and
 * gets the same answers as its neighbours by construction.
 *
 * Two things in here are easy to get wrong and are the reason it is a module rather
 * than a hundred lines in a sample:
 *
 *  - **A site is judged on the RAW height field**, @ref sampleWorldHeight, never on a
 *    chunk's eroded and carved one. A chunk does not have its neighbour's field, so a
 *    rule that read it would answer differently depending on who asked — and the whole
 *    point is that it cannot. It costs a little accuracy at a cliff edge; it buys the
 *    only property that matters.
 *  - **Carving reach and drawing ownership are different questions.** Every chunk within
 *    reach of a site must carve it, or the ground disagrees across a seam. Exactly ONE
 *    chunk may draw it — the one holding its centre — or a village appears once per
 *    chunk that can see it.
 *
 * What is NOT here: how a village is laid out. That is @ref generateSettlementOnTerrain
 * and @ref BuildingPlot, which already do roads, districts, plots and slope refusal.
 * A streamed village is a small BOUNDED settlement derived from its site, not a second
 * town generator.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_LANDMARK_HPP
#    define LPL_PROCGEN_LANDMARK_HPP

#    include <lpl/procgen/ChunkResidency.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/procgen/Settlement.hpp>

namespace lpl::procgen {

/// What a site is for. Each kind reads the same terrain and wants opposite things of it.
enum class LandmarkKind : core::u8 {
    CaveMouth = 0, ///< An opening in a hillside. Wants a STEEP face.
    Settlement,    ///< Somewhere people would live. Wants FLAT ground.
    Count
};

/**
 * @struct LandmarkParams
 * @brief How often a kind of site occurs, and what ground it will accept.
 */
struct LandmarkParams {
    /**
     * @brief World cells per landmark cell.
     *
     * The spacing of the lattice candidates are drawn from, so it is also the closest
     * two sites of one kind can be. Large enough that a site's own footprint fits well
     * inside it, or two neighbours overlap and the search reach has to grow.
     */
    core::u32 cellSpan{40u};

    core::u32 oneIn{5u}; ///< One landmark cell in N carries a site.

    /**
     * @brief Cells the site occupies, measured from its centre.
     *
     * Also the reach: a chunk must consider every landmark cell whose site could touch
     * it, which is every cell within `radius` world cells of its own extent.
     */
    core::u32 radius{7u};

    /**
     * @brief How far above the sea the ground must be.
     *
     * For a cave this is the answer to the question a flooded cave asks: we do not stop
     * the water getting in, we decline to put a mouth where the water is. A margin
     * rather than "above sea level", because a mouth exactly at the waterline is a
     * mouth the first wave fills.
     */
    core::f32 clearanceAboveSea{2.0f};

    core::f32 maxHeight{1.0e9f}; ///< Above this the site is refused (snow, bare rock).

    /**
     * @brief The slope band the site will accept, as a height difference over @ref radius.
     *
     * A cave mouth wants the steep end and a village the flat end, so the two kinds are
     * the same rule with the band reversed rather than two rules.
     */
    core::f32 minRelief{0.0f};
    core::f32 maxRelief{1.0e9f};
};

/// A cave mouth: steep ground, well clear of the water, not up in the snow.
[[nodiscard]] constexpr LandmarkParams caveMouthDefaults() noexcept
{
    LandmarkParams params{};
    // Measured, over 441 landmark cells of the walked world, at radius 4:
    //
    //   | cellSpan | oneIn | minRelief | one mouth per |
    //   |----------|-------|-----------|---------------|
    //   |    24    |   2   |    2.2    |   4.9 chunks  |
    //   |    32    |   2   |    2.2    |  10.1 chunks  |
    //   |    32    |   3   |    2.2    |  17.4 chunks  |
    //   |    40    |   3   |    5.0    | 408.0 chunks  |
    //
    // The last row is what this used to be, and it is why the world had no caves in it:
    // one mouth per four hundred chunks is one a walker will never meet. Ten is close
    // enough to find by walking and far enough apart not to read as a lattice.
    params.cellSpan = 24u;
    params.oneIn = 2u;
    params.radius = 4u;
    params.clearanceAboveSea = 3.0f;
    // A face, not a plain — and deliberately a little MORE relief than the shelf is deep,
    // or the cut is deeper than the hill it is cut into and the mouth is a pit in a field.
    params.minRelief = 2.2f;
    params.maxRelief = 1.0e9f;
    return params;
}

/// A village: flat ground, above the flood line, below the tree line.
[[nodiscard]] constexpr LandmarkParams settlementDefaults() noexcept
{
    LandmarkParams params{};
    // Measured on the coastal walked world, at maxRelief 6 and radius 11, against a
    // resident set of fifty-six chunks:
    //
    //   | cellSpan | oneIn | one village per | resident on average |
    //   |----------|-------|-----------------|---------------------|
    //   |    56    |   2   |   49.0 chunks   |        1.14         |
    //   |    64    |   2   |   59.2 chunks   |        0.95         |
    //   |    96    |   2   |  108.6 chunks   |        0.52         |
    //   |    96    |   3   |  156.8 chunks   |        0.36         |
    //
    // One resident on average: a village is usually on screen or just over the hill, and
    // never a suburb. The last row is what this used to be, and half a village resident is
    // a world where you walk for a quarter of an hour without meeting one.
    //
    // The span is also what keeps two of them apart: 56 minus the 34 cells of jitter leaves
    // 22, which is exactly two radii, so neighbours touch at worst and never overlap.
    params.cellSpan = 56u;
    params.oneIn = 2u;
    params.radius = 11u;
    params.clearanceAboveSea = 3.0f;
    params.maxHeight = 34.0f;
    params.minRelief = 0.0f;
    // Flat enough to build on. The settlement pass refuses steep CELLS on its own; this
    // refuses steep SITES, which is the cheaper question and the one asked first.
    params.maxRelief = 4.0f;
    return params;
}

/**
 * @struct LandmarkSite
 * @brief One placed landmark, in world cells.
 */
struct LandmarkSite {
    core::i32 cellX{0};     ///< World column of its centre.
    core::i32 cellZ{0};     ///< World row of its centre.
    core::f32 height{0.0f}; ///< Raw ground height there. The pad and the floor come from it.
    core::f32 relief{0.0f}; ///< The height difference that qualified it.
    core::u32 seed{0u};     ///< Derived from its landmark cell; everything else derives from this.
    core::u32 facing{0u};   ///< kNeighbor8 index of the downhill direction.
    core::u32 radius{0u};   ///< Copy of the params', so a consumer needs one argument fewer.
    LandmarkKind kind{LandmarkKind::CaveMouth};
};

/**
 * @brief Whether one landmark cell carries a site, and where exactly.
 *
 * A pure function of (@p landmarkX, @p landmarkZ, the world seed, the kind). Every chunk
 * that asks gets the same answer, which is the whole contract.
 *
 * The centre is JITTERED inside its cell rather than placed at the middle of it: sites on
 * a lattice read as a grid the moment two are on screen together, and a grid of villages
 * is more obviously artificial than no villages.
 *
 * @param params     World parameters.
 * @param landmarks  How often this kind occurs and what ground it accepts.
 * @param kind       Which kind is being asked about.
 * @param seaLevel   Where the water is.
 * @param landmarkX  Landmark column, absolute.
 * @param landmarkZ  Landmark row, absolute.
 * @param out        Receives the site when there is one.
 * @return true when this cell carries a site.
 */
[[nodiscard]] bool landmarkAt(const ChunkParams &params, const LandmarkParams &landmarks, LandmarkKind kind,
                              core::f32 seaLevel, core::i32 landmarkX, core::i32 landmarkZ, LandmarkSite &out);

/**
 * @brief Every site of one kind that can reach a chunk, whether or not its centre is in it.
 *
 * This is the CARVE list. A site whose centre is next door still lowers ground here, and a
 * chunk that skipped it would disagree with its neighbour along their seam.
 *
 * @param params    World parameters.
 * @param landmarks Siting rule.
 * @param kind      Which kind.
 * @param seaLevel  Where the water is.
 * @param coord     Chunk asking.
 * @param emit      Called with each site: `emit(const LandmarkSite &)`.
 */
template <typename Emit>
void forEachLandmarkNear(const ChunkParams &params, const LandmarkParams &landmarks, LandmarkKind kind,
                         core::f32 seaLevel, ChunkCoord coord, Emit &&emit)
{
    if (params.size == 0u || landmarks.cellSpan == 0u)
        return;

    const core::i32 span = static_cast<core::i32>(landmarks.cellSpan);
    const core::i32 reach = static_cast<core::i32>(landmarks.radius);
    const core::i32 chunkCells = static_cast<core::i32>(params.size);
    const core::i32 minCellX = coord.x * chunkCells - reach;
    const core::i32 maxCellX = coord.x * chunkCells + chunkCells - 1 + reach;
    const core::i32 minCellZ = coord.z * chunkCells - reach;
    const core::i32 maxCellZ = coord.z * chunkCells + chunkCells - 1 + reach;

    // Floor division on both ends: truncation folds the world about its own axis, so a
    // site at -1 and one at +1 would land in the same landmark cell.
    const core::i32 fromX = floorDivChunk(minCellX, span);
    const core::i32 toX = floorDivChunk(maxCellX, span);
    const core::i32 fromZ = floorDivChunk(minCellZ, span);
    const core::i32 toZ = floorDivChunk(maxCellZ, span);

    for (core::i32 lz = fromZ; lz <= toZ; ++lz)
        for (core::i32 lx = fromX; lx <= toX; ++lx)
        {
            LandmarkSite site;
            if (!landmarkAt(params, landmarks, kind, seaLevel, lx, lz, site))
                continue;
            // The jitter can push a centre outside the band the cell scan bounded, so the
            // footprint is tested rather than trusted.
            if (site.cellX + static_cast<core::i32>(site.radius) < minCellX ||
                site.cellX - static_cast<core::i32>(site.radius) > maxCellX ||
                site.cellZ + static_cast<core::i32>(site.radius) < minCellZ ||
                site.cellZ - static_cast<core::i32>(site.radius) > maxCellZ)
                continue;
            emit(site);
        }
}

/**
 * @brief Whether a site's centre belongs to a chunk, i.e. whose job it is to DRAW it.
 *
 * Half-open on purpose: a centre exactly on a border belongs to the chunk it starts, and
 * a closed test would give it to both.
 *
 * @param params World parameters.
 * @param site   The site.
 * @param coord  Chunk asking.
 * @return true when this chunk owns the site.
 */
[[nodiscard]] bool chunkOwnsLandmark(const ChunkParams &params, const LandmarkSite &site, ChunkCoord coord);

/**
 * @brief The relief at a given QUANTILE of the sites a rule would otherwise accept.
 *
 * ⚠ `LandmarkParams::minRelief` and `maxRelief` are absolute metres, and an absolute
 * threshold against a distribution that moves is this repository's most-repeated mistake —
 * the river threshold was calibrated for exactly this reason. It bit twice more here in one
 * afternoon: a village tolerance of 2.4 m admitted two sites in three and a half thousand
 * chunks, and the same 6.0 m that gave one village per forty-nine chunks gave none once the
 * terrain shaping was fixed and the world grew hills.
 *
 * So the caller states a SHARE — how many of the candidate sites should qualify — and this
 * returns the metre value that produces it on the terrain that actually exists.
 *
 * **Chunk independence is preserved, and it is the whole difficulty.** A quantile over "the
 * world" does not exist: there is no total. The window is therefore a fixed block of
 * landmark cells at the origin — a function of the parameters alone, identical for every
 * chunk that asks, computed ONCE when a plan is built and never per chunk.
 *
 * @param params      World parameters.
 * @param landmarks   Siting rule; its own relief band is ignored.
 * @param kind        Which kind.
 * @param seaLevel    Where the water is.
 * @param quantile    Fraction of accepted sites BELOW the returned relief, in (0, 1).
 * @return The relief at that quantile, or zero when no site passes the other rules.
 */
[[nodiscard]] core::f32 calibrateLandmarkRelief(const ChunkParams &params, const LandmarkParams &landmarks,
                                                LandmarkKind kind, core::f32 seaLevel, core::f32 quantile);

/**
 * @struct VillagePlan
 * @brief One settlement, laid out on ground derived from its site.
 *
 * A small bounded world, positioned. The layout is @ref generateSettlementOnTerrain's —
 * roads, districts, plots, slope refusal — so a streamed village is the same village a
 * bounded map would grow, and the two cannot drift.
 */
struct VillagePlan {
    LandmarkSite site{};
    SettlementMap map{};                    ///< Village-local grid of cells.
    lpl::pmr::vector<BuildingPlot> plots{}; ///< Footprints, in village-local cells.
    core::i32 originX{0};                   ///< World column of the map's (0, 0).
    core::i32 originZ{0};                   ///< World row of it.
    core::f32 padHeight{0.0f};              ///< The one level the ground is flattened to.
    core::u32 side{0u};                     ///< Cells along each edge of the map.
};

/**
 * @brief Lays out the village of a site.
 *
 * Chunk-independent because it reads nothing but the site, and a site is a function of
 * its own coordinates. Regenerated by every chunk that touches the village rather than
 * cached, which is a real cost — bounded by @ref LandmarkParams::radius — and the reason
 * the village is small.
 *
 * The ground it is laid out on is the RAW field, and it is then FLATTENED to one level:
 * a village on a slope has its streets at whatever height each cell happened to be, and
 * a building standing on two levels at once shears. One level per village is the same
 * decision `buildPlotDatum` made for a bounded plot, for the same reason.
 *
 * @param params World parameters.
 * @param site   The site, from @ref landmarkAt.
 * @return The plan. Empty (side == 0) when the site could not support a village.
 */
[[nodiscard]] VillagePlan planVillage(const ChunkParams &params, const LandmarkSite &site);

/**
 * @struct LandmarkBuilding
 * @brief One building of a village, in WORLD cells and world units.
 *
 * What a renderer needs and nothing else. The village-local grid is an implementation
 * detail of the layout; a renderer that had to know about it would have to know about
 * @ref VillagePlan, and then every consumer would carry the settlement module.
 */
struct LandmarkBuilding {
    core::f32 minX{0.0f};
    core::f32 minZ{0.0f};
    core::f32 maxX{0.0f};
    core::f32 maxZ{0.0f};
    core::f32 baseY{0.0f};  ///< The pad it stands on.
    core::f32 height{0.0f}; ///< How far it rises above the pad.
    core::u32 storeys{1u};  ///< Storeys, from the plot's own footprint area.
    core::u16 district{0u}; ///< Which district, so a village is not one material.
};

/**
 * @brief Every building of a plan, in world coordinates.
 *
 * @param plan  The village.
 * @param emit  Called with each building: `emit(const LandmarkBuilding &)`.
 */
template <typename Emit> void forEachVillageBuilding(const VillagePlan &plan, Emit &&emit)
{
    for (core::u32 i = 0u; i < plan.plots.size(); ++i)
    {
        const BuildingPlot &plot = plan.plots[i];
        if (plot.width == 0u || plot.depth == 0u)
            continue;

        LandmarkBuilding building;
        building.minX = static_cast<core::f32>(plan.originX + static_cast<core::i32>(plot.x));
        building.minZ = static_cast<core::f32>(plan.originZ + static_cast<core::i32>(plot.z));
        building.maxX = building.minX + static_cast<core::f32>(plot.width);
        building.maxZ = building.minZ + static_cast<core::f32>(plot.depth);
        building.baseY = plan.padHeight;
        // Storeys from the footprint's area, not from a die roll: a wide plot is a hall
        // and a narrow one is a tower, which is the relationship a town actually has.
        const core::u32 area = plot.width * plot.depth;
        building.storeys = area >= 12u ? 1u : (area >= 6u ? 2u : 3u);
        building.height = 2.6f * static_cast<core::f32>(building.storeys);
        building.district = plot.district;
        emit(building);
    }
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_LANDMARK_HPP
