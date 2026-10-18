/**
 * @file Voronoi.hpp
 * @brief Partitioning space into regions, which is how a world gets structure.
 *
 * Noise gives texture but no organisation: every cell is as good as its
 * neighbour and nothing has a boundary. A Voronoi diagram gives the opposite —
 * discrete regions with sharp, irregular edges — which is what provinces,
 * biome patches, city districts and territory all are.
 *
 * Sites are placed by jittering one point per coarse cell rather than scattering
 * them freely. Free scattering needs a global search to find the nearest site;
 * a jittered grid bounds the search to the 3x3 block of coarse cells around a
 * query, or the 5x5 block under a strong jitter, which turns an O(sites) lookup
 * into O(1) and removes the only reason this would have been expensive.
 *
 * Distances are compared without ever taking a square root — squared for the
 * Euclidean metric, directly for the other two — which keeps this exact in
 * Fixed32 and free of any transcendental.
 *
 * A jittered grid does leave one visible signature: the borders are straight
 * segments, so a partition looks like a partition. Warping the query point
 * through a noise field before measuring distance folds those segments into
 * irregular, interlocking boundaries while keeping every region connected and the
 * lookup still O(1).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_VORONOI_HPP
#    define LPL_PROCGEN_VORONOI_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/** Region index per cell. */
using RegionMap = Grid<core::u16>;

/** Marks a cell no region claimed (only when the diagram is empty). */
inline constexpr core::u16 kNoRegion = 0xFFFFu;

/**
 * @struct VoronoiSite
 * @brief One region's centre, in grid coordinates.
 */
struct VoronoiSite {
    math::Fixed32 x{};    /**< Column (fractional). */
    math::Fixed32 z{};    /**< Row (fractional). */
    core::u16 region{0u}; /**< Index this site owns. */
};

/**
 * @enum DistanceMetric
 * @brief Which notion of "nearest" partitions the space.
 *
 * The metric decides the *shape* of the regions far more than the site placement
 * does, and each of the three has a use the others cannot serve.
 */
enum class DistanceMetric : core::u8 {
    /**
     * @brief Straight-line distance.
     *
     * Organic, convex, cell-like regions: villages, provinces, biome patches,
     * anything grown rather than planned.
     */
    Euclidean = 0,

    /**
     * @brief Taxicab distance, @f$|dx| + |dz|@f$.
     *
     * Boundaries become rectilinear staircases and diamonds, which is what makes
     * this the metric for city blocks — and it means the region borders can be
     * used directly as the street grid instead of needing roads drawn separately.
     */
    Manhattan,

    /**
     * @brief Chessboard distance, @f$\max(|dx|, |dz|)@f$.
     *
     * Regions are forced toward axis-aligned squares: rigid, regimented layouts.
     */
    Chebyshev
};

/**
 * @struct VoronoiParams
 * @brief Grid size, region scale, metric and how organic the borders are.
 */
struct VoronoiParams {
    core::u32 width{64u};                             /**< Cells along X. */
    core::u32 depth{64u};                             /**< Cells along Z. */
    core::u32 seed{1337u};                            /**< Determinism anchor. */
    core::u32 cellSize{12u};                          /**< Coarse cell edge; larger means bigger regions. */
    core::f32 jitter{0.8f};                           /**< How far a site strays in its cell, clamped to [0, 1]. */
    DistanceMetric metric{DistanceMetric::Euclidean}; /**< Which distance decides ownership. */
    core::f32 warpStrength{0.0f};                     /**< Domain warp in cells; 0 leaves borders polygonal. */
};

/**
 * @struct VoronoiDiagram
 * @brief The computed partition.
 */
struct VoronoiDiagram {
    RegionMap regions;                   /**< Region index per cell. */
    lpl::pmr::vector<VoronoiSite> sites; /**< Region centres, indexed by region. */
    core::u32 regionCount{0u};           /**< Number of regions. */
};

/**
 * @brief Builds a Voronoi partition over a grid.
 *
 * One region per coarse cell, indexed in scan order so a region id is
 * reproducible from its coarse coordinates alone.
 *
 * A region id is a u16 and 0xFFFF means "no region", so a partition finer than
 * that would wrap ids into the sentinel and silently mark cells unclaimed.
 * Refusing outright beats producing a diagram whose region 65535 is invisible.
 *
 * Each cell compares only the sites of the coarse cells around its home coarse
 * cell, a jittered site never leaving its own cell: one ring of them, or two
 * when the jitter lets a site two cells away be the nearest (above about 0.657
 * under the Euclidean metric, above 1/3 under Manhattan). The block is widened by
 * however many coarse cells the warp can displace the query point. A site wins
 * only when strictly nearer: ties go to the first in scan order, so the
 * partition is the same on every target.
 *
 * @param params Size, seed and region scale.
 * @return The diagram (empty when the grid or cell size is degenerate).
 */
[[nodiscard]] VoronoiDiagram computeVoronoi(const VoronoiParams &params);

/**
 * @brief Marks cells that touch a different region — the region borders.
 *
 * Borders are where interesting things go: walls, roads, coastlines, the edge
 * of a territory. Deriving them from the partition rather than drawing them
 * separately guarantees they agree with it.
 *
 * @param diagram Partition to trace.
 * @return A grid of 0/1 flags.
 */
[[nodiscard]] Grid<core::u8> regionBorders(const VoronoiDiagram &diagram);

/**
 * @brief Distance from each cell to its own site, squared.
 *
 * The classic "F1" Voronoi field. Useful as a mask: near 0 at a region's centre
 * and largest at its edges, so it fades anything placed per-region.
 *
 * @param diagram Partition to measure.
 * @param metric  Distance to measure with; pass the one the diagram was built on.
 * @return A Fixed32 grid of distances (squared, for the Euclidean metric, and then
 *         Fixed32::max() for a cell about 181 units or more from its site, whose
 *         square Q16.16 cannot hold).
 */
[[nodiscard]] Grid<math::Fixed32> regionDistanceField(const VoronoiDiagram &diagram,
                                                      DistanceMetric metric = DistanceMetric::Euclidean);

/**
 * @brief Counts the cells each region owns.
 * @param diagram   Partition to measure.
 * @param outCounts Receives one count per region; must hold regionCount entries.
 */
void countRegionCells(const VoronoiDiagram &diagram, core::u32 *outCounts);

/**
 * @brief FNV-1a fold of a region map, for determinism checks.
 * @param map Map to fold.
 * @return The 32-bit signature.
 */
[[nodiscard]] core::u32 foldRegionMap(const RegionMap &map);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_VORONOI_HPP
