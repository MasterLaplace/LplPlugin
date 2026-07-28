/**
 * @file Settlement.hpp
 * @brief Roads, districts and building plots — places people would live.
 *
 * A settlement is not scattered buildings: it is a road network with things
 * arranged along it. So the generation runs in that order — connect first,
 * subdivide second, build last — which is also why the result is walkable by
 * construction rather than by a repair pass.
 *
 * The three stages, each borrowing the tool that fits it:
 *
 *  1. **Districts** come from a Voronoi partition. Districts are exactly what a
 *     Voronoi diagram is: irregular regions with hard borders, meeting at
 *     junctions.
 *  2. **Roads** run along the district borders, which means the network agrees
 *     with the districts instead of cutting across them. A spanning pass then
 *     links every district centre to the network, so nothing is enclaved.
 *  3. **Plots** are the cells left inside a district that face a road. Facing a
 *     road is the rule that makes a plot a plot; without it buildings end up
 *     stranded in the middle of blocks.
 *
 * Terrain-aware: a settlement placed on a heightfield refuses slopes and water,
 * so a town does not climb a cliff or stand in a lake.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_SETTLEMENT_HPP
#    define LPL_PROCGEN_SETTLEMENT_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Voronoi.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/// What a settlement cell is.
enum class SettlementCell : core::u8 {
    Empty = 0, ///< Nothing built here.
    Road,      ///< Walkable street.
    Plaza,     ///< Open space at a district centre.
    Plot,      ///< A building footprint.
    Blocked    ///< Unbuildable: too steep, or under water.
};

/// A settlement layout.
using SettlementMap = Grid<SettlementCell>;

/**
 * @struct BuildingPlot
 * @brief One footprint, and which district it belongs to.
 */
struct BuildingPlot {
    core::u32 x{0u};        ///< Left edge.
    core::u32 z{0u};        ///< Top edge.
    core::u32 width{0u};    ///< Extent along X.
    core::u32 depth{0u};    ///< Extent along Z.
    core::u16 district{0u}; ///< Owning region.
};

/**
 * @struct SettlementParams
 * @brief Size, seed, and the shape of the town.
 */
struct SettlementParams {
    core::u32 width{96u};        ///< Map width.
    core::u32 depth{96u};        ///< Map depth.
    core::u32 seed{1337u};       ///< Determinism anchor.
    core::u32 districtSize{16u}; ///< Voronoi coarse cell; larger means bigger blocks.
    core::u32 roadWidth{1u};     ///< Street thickness.
    core::u32 plazaRadius{2u};   ///< Open space carved at each district centre.
    core::u32 minPlot{2u};       ///< Smallest building footprint edge.
    core::u32 maxPlot{4u};       ///< Largest building footprint edge.
    core::f32 plotDensity{0.6f}; ///< Chance an eligible road-facing spot is built on.
    core::f32 maxSlope{1.5f};    ///< Steepest ground a settlement will occupy.
    core::f32 minHeight{0.5f};   ///< Lowest ground it will occupy (keeps it out of water).
};

/**
 * @struct SettlementReport
 * @brief What was laid out.
 */
struct SettlementReport {
    core::u32 districts{0u};    ///< Voronoi regions covering the area.
    core::u32 roadCells{0u};    ///< Cells that became street.
    core::u32 plazaCells{0u};   ///< Cells that became open space.
    core::u32 plots{0u};        ///< Buildings placed.
    core::u32 blockedCells{0u}; ///< Cells refused as unbuildable.
    bool roadsConnected{false}; ///< Is the whole street network one piece?
};

/**
 * @brief Lays out a settlement on flat ground.
 * @param params    Size, seed and layout knobs.
 * @param outPlots  Receives the building footprints (may be null).
 * @param outReport Receives the layout statistics (may be null).
 * @return The settlement map.
 */
[[nodiscard]] SettlementMap generateSettlement(const SettlementParams &params,
                                               lpl::pmr::vector<BuildingPlot> *outPlots = nullptr,
                                               SettlementReport *outReport = nullptr);

/**
 * @brief Lays out a settlement that respects the terrain under it.
 *
 * Cells too steep or too low are marked @ref SettlementCell::Blocked before
 * anything is placed, so roads route around them and no building stands on a
 * cliff or in a lake.
 *
 * @param params    Size, seed and layout knobs (dimensions must match @p terrain).
 * @param terrain   Ground the settlement sits on.
 * @param outPlots  Receives the building footprints (may be null).
 * @param outReport Receives the layout statistics (may be null).
 * @return The settlement map.
 */
[[nodiscard]] SettlementMap generateSettlementOnTerrain(const SettlementParams &params, const Heightfield &terrain,
                                                        lpl::pmr::vector<BuildingPlot> *outPlots = nullptr,
                                                        SettlementReport *outReport = nullptr);

/**
 * @brief Is every road cell reachable from every other?
 * @param map Settlement to test.
 * @return true when the street network is a single connected piece.
 */
[[nodiscard]] bool areRoadsConnected(const SettlementMap &map);

/**
 * @brief FNV-1a fold of a settlement map, for determinism checks.
 * @param map Map to fold.
 * @return The 32-bit signature.
 */
[[nodiscard]] core::u32 foldSettlement(const SettlementMap &map);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_SETTLEMENT_HPP
