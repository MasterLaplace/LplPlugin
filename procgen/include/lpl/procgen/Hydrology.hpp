/**
 * @file Hydrology.hpp
 * @brief Where water goes, and what that says about the land.
 *
 * Once terrain has slopes, the drainage network is not a choice — it follows
 * from them. Computing it gives two things at once: rivers to carve, and a
 * moisture field, which is the second axis every biome classification needs (the
 * first being elevation). The report's TWI note is the same idea: a cell that
 * drains a large area and lies on a gentle slope is wet.
 *
 * Everything here is integer flow accumulation over the grid, so it is exact and
 * order-independent: cells are processed from highest to lowest, which
 * guarantees a cell's own inflow is final before it passes water on.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_HYDROLOGY_HPP
#    define LPL_PROCGEN_HYDROLOGY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/procgen/Heightfield.hpp>

namespace lpl::procgen {

/// How many upstream cells drain through each cell (itself included).
using FlowAccumulation = Grid<core::u32>;

/// Index into @ref kNeighbor8X / @ref kNeighbor8Z of the cell each one drains
/// to, or @ref kNoFlow where there is nowhere lower to go.
using FlowDirection = Grid<core::u8>;

/// Marker in a @ref FlowDirection for a cell with nowhere lower to go.
inline constexpr core::u8 kNoFlow = 0xFFu;

/**
 * @struct DrainageNetwork
 * @brief The full result of routing water over a heightfield.
 */
struct DrainageNetwork {
    FlowDirection direction;       ///< Downhill neighbour per cell (D8 index).
    FlowAccumulation accumulation; ///< Upstream cell count per cell.
    Heightfield filled;            ///< The depression-filled surface routing used.
    core::u32 maxAccumulation{0u}; ///< Largest accumulation found (the river mouth).
    core::u32 raisedCells{0u};     ///< Cells the fill lifted, i.e. the lake surface.
};

/**
 * @struct RiverParams
 * @brief How much of the drainage network becomes visible water.
 */
struct RiverParams {
    /**
     * @brief Share of the map's cells that become river, in [0, 1].
     *
     * A quantile of the accumulation distribution, not a fraction of the largest
     * flow: accumulation is heavy-tailed and its spread grows with the map, so a
     * fixed share of the maximum floods a small world and barely wets a large
     * one. Asking for a share of the *cells* means the same number here reads the
     * same at every size — the same reason climate distances are relative to the
     * map and biome thresholds are applied after normalisation.
     */
    core::f32 density{0.04f};
    core::f32 carveDepth{1.5f}; ///< How deep the strongest river cuts.
    core::u32 smoothing{1u};    ///< Smoothing passes over the carved bed.
};

/**
 * @brief Raises every closed depression to its spill level.
 *
 * A heightfield built from noise is full of local minima, and a local minimum is
 * a cell with nowhere to drain: every upstream cell's water dies there. Routing
 * without filling first therefore does not produce a drainage network, it
 * produces several hundred unrelated puddles — measurably so, the largest one
 * carrying under two percent of the map whatever the map's size. Filling is not
 * a refinement of flow routing, it is a precondition of it.
 *
 * Priority-Flood (Barnes, Lehman and Mulla): flood inward from the border,
 * always from the lowest cell seen so far, raising each cell it reaches to just
 * above the cell it came from. Every cell ends up with a strictly descending
 * path to the border, so no sink survives. The "just above" step is one Q16.16
 * tick, which is also what lets water cross a filled basin in a definite
 * direction instead of stalling on a plateau.
 *
 * @param field Terrain to fill in place.
 * @return Number of cells raised — the extent of the lakes this created.
 */
core::u32 fillDepressions(Heightfield &field);

/**
 * @brief Routes every cell's water to its steepest downhill neighbour (D8).
 *
 * Depressions are filled on a private copy first, so the caller's terrain is
 * untouched while routing still sees a surface with no sinks; that surface is
 * returned in @c DrainageNetwork::filled, because it is also the water level of
 * whatever lakes the fill implied.
 *
 * Eight directions, not four. Four confines a channel to the cardinal axes, so
 * every river runs in visible right-angled staircases; eight halves the bias for
 * the cost of comparing diagonal drops against @c kInvSqrt2, since a diagonal
 * step covers more ground for the same fall.
 *
 * @param field Terrain to route over.
 * @return The drainage network (empty grids when @p field is empty).
 */
[[nodiscard]] DrainageNetwork computeDrainage(const Heightfield &field);

/**
 * @brief Lowers the terrain along the strongest drainage paths.
 *
 * Depth follows the stream power law, @f$E = K A^{m} S^{n}@f$ with the standard
 * bedrock exponents @f$m = 1/2@f$, @f$n = 1@f$: incision grows with the square
 * root of the drainage area and linearly with the local slope. Area alone —
 * which is what a naive carve uses — gouges the same trench through a flat delta
 * as through a mountain gorge, and flattens the one place a river should be
 * cutting hardest. The slope term is what produces gorges upstream and shallow
 * meanders where the land runs out.
 *
 * The strongest cell receives exactly @c carveDepth; everything else scales
 * against it, so the parameter keeps meaning what it says.
 *
 * @param field    Terrain to carve in place.
 * @param network  Drainage computed from @p field.
 * @param params   Threshold and depth.
 * @return Number of cells that became river bed.
 */
core::u32 carveRivers(Heightfield &field, const DrainageNetwork &network, const RiverParams &params);

/**
 * @brief Marks which cells the depression fill drowned — the lakes.
 *
 * Free, and it was being thrown away. Priority-Flood raises every cell inside a
 * basin to its spill level, so the difference between the filled surface and the
 * original terrain IS the water: where it is positive there is standing water,
 * and how positive says how deep. A generator that fills depressions and then
 * reports only a raised-cell count has already computed its lakes and discarded
 * them.
 *
 * @warning The fill raises every cell of a basin, including the ones it lifted by
 *          a single Q16.16 tick — a puddle 1/65536 of a unit deep. Measured on a
 *          64x64 world, taking every raised cell called **20% of the map** a
 *          lake. @p minDepth is what separates a body of water from the epsilon
 *          slope the algorithm leaves behind, and it is not optional in practice.
 *
 * @param network  Drainage whose @c filled surface holds the spill levels.
 * @param original The terrain the fill was computed from.
 * @param minDepth How deep the water must stand, in world units.
 * @return A grid of 0/1 flags, empty when the two do not match in size.
 */
[[nodiscard]] Grid<core::u8> lakeMask(const DrainageNetwork &network, const Heightfield &original,
                                      core::f32 minDepth = 0.5f);

/**
 * @brief How deep the water stands at each cell, in world units.
 * @param network  Drainage whose @c filled surface holds the spill levels.
 * @param original The terrain the fill was computed from.
 * @return Depth per cell (zero on dry land), empty when sizes disagree.
 */
[[nodiscard]] Heightfield lakeDepth(const DrainageNetwork &network, const Heightfield &original);

/**
 * @brief Marks which cells are river.
 *
 * @p density is a quantile: the strongest-flowing cells are taken until that
 * share of the map is covered. Ties are kept whole, so the result never depends
 * on visit order and the covered share can slightly exceed what was asked.
 *
 * @param network Drainage network.
 * @param density Share of cells to mark, in [0, 1].
 * @return A grid of 0/1 flags.
 */
[[nodiscard]] Grid<core::u8> riverMask(const DrainageNetwork &network, core::f32 density);

/// Value in a distance grid for a cell no seed can reach.
inline constexpr core::u32 kUnreachedFromSea = 0xFFFFu;

/**
 * @brief How many cells away the nearest marked cell is, everywhere.
 *
 * Two chamfer sweeps rather than a wavefront queue: cardinal distances come out
 * exact, diagonals land within the usual 3:4 approximation, and it costs one pass
 * each way instead of a queue the size of the map. Good enough for every falloff
 * in this module, and it is one implementation rather than one per caller.
 *
 * @param seeds Non-zero marks a source cell (distance 0).
 * @return Cell distances, @ref kUnreachedFromSea where no source exists.
 */
[[nodiscard]] Grid<core::u32> chamferDistance(const Grid<core::u8> &seeds);

/**
 * @brief How many cells inland every cell lies.
 *
 * Two chamfer sweeps rather than a wavefront queue: the cardinal distances come
 * out exact, diagonals land within the usual 3:4 approximation, and it costs one
 * pass each way instead of a queue the size of the map. Good enough for a falloff,
 * and it is the same number the moisture's coast term and the climate's
 * continentalness axis both need — so it is computed once, here, rather than
 * spelled out twice and left to drift apart.
 *
 * @param field    Terrain.
 * @param seaLevel Height at or below which a cell is sea (distance 0).
 * @return Cell distances, @ref kUnreachedFromSea where no sea exists.
 */
[[nodiscard]] Grid<core::u32> distanceToSea(const Heightfield &field, math::Fixed32 seaLevel);

/**
 * @struct MoistureParams
 * @brief What decides how wet a cell is.
 *
 * The weights need not sum to one; the result is clamped to [0, 1] afterwards.
 * Setting one to zero removes that influence entirely, which is the intended way
 * to say "this world has no ocean" or "wind does not matter here".
 *
 * @note Every distance here is expressed **relative to the map**, not in cells,
 *       and deliberately so. Climate is the lowest-frequency thing in a world: a
 *       continent has a handful of wet and dry belts whether it is a hundred cells
 *       across or a thousand. A frequency fixed in cells would hold the belt
 *       *width* constant instead, so doubling the map would double the number of
 *       biome bands and a player would cross a climate every few paces.
 */
struct MoistureParams {
    core::f32 rainfallWeight{0.4f};  ///< Contribution of the baseline rainfall field.
    core::f32 flowWeight{0.3f};      ///< Contribution of upstream drainage.
    core::f32 altitudeWeight{0.1f};  ///< Contribution of being low-lying.
    core::f32 coastWeight{0.3f};     ///< Contribution of being near the sea.
    core::f32 seaLevel{0.0f};        ///< Height at or below which a cell is sea.
    core::f32 coastReach{0.25f};     ///< Share of the map's longer axis the sea's influence carries inland.
    core::f32 rainShadow{0.55f};     ///< How much a windward ridge dries the land behind it, in [0, 1].
    core::u32 windDirection{0u};     ///< Prevailing wind, as an index into kNeighbor4X/Z.
    core::u32 smoothing{2u};         ///< Diffusion passes; 0 leaves moisture in the channels.
    core::u32 rainfallSeed{0x2A17u}; ///< Seed of the baseline rainfall field.
    core::f32 rainfallBelts{2.5f};   ///< Wet/dry belts across the map's longer axis.
    core::u32 rainfallOctaves{3u};   ///< fBm octaves of the rainfall field.
};

/**
 * @brief Builds a moisture field in [0, 1] from drainage, altitude, sea and wind.
 *
 * Wet means "water passes through here and does not rush away". Five things
 * decide it, and each is in the literature for a reason:
 *
 *  - **Baseline rainfall**, an independent noise field. This one is easy to leave
 *    out and it must not be: without it, moisture becomes a pure function of the
 *    terrain, so the only cells that end up wet are the ones at the bottom — which
 *    are underwater and get classified by altitude before climate is ever
 *    consulted. Measured on a 128x128 world, that alone made forest, rainforest
 *    and marsh impossible whatever the thresholds were set to. It is also what
 *    decorrelates climate from elevation, so a desert and a forest can sit at the
 *    same height instead of the map reading as contour bands.
 *  - **Upstream drainage**, on a logarithmic scale. Accumulation is savagely
 *    skewed — a handful of trunk cells carry most of the map — so a linear term
 *    leaves everything but the trunk bone dry.
 *  - **Altitude**, because air holds less water as it rises.
 *  - **Distance to the sea**, which is where the water comes from. A continental
 *    interior is dry however well it drains.
 *  - **Rain shadow**: air crossing a ridge sheds its water climbing the windward
 *    face and arrives dry on the other side. Scanning along the wind and
 *    carrying a decaying "highest ground crossed so far" reproduces it, and it is
 *    what puts a desert immediately behind a mountain range instead of
 *    distributing deserts by latitude alone.
 *
 * Smoothing afterwards turns a one-cell channel into a valley-wide wet band,
 * which is what a biome map needs — vegetation does not stop at the water's edge.
 *
 * @param field   Terrain (for the altitude, coast and shadow terms).
 * @param network Drainage network (for the accumulation term).
 * @param params  Weights, sea level, wind.
 * @return A grid of Fixed32 in [0, 1].
 */
[[nodiscard]] Heightfield computeMoisture(const Heightfield &field, const DrainageNetwork &network,
                                          const MoistureParams &params);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_HYDROLOGY_HPP
