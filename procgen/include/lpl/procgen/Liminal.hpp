/**
 * @file Liminal.hpp
 * @brief Spaces that are recognisably built and impossible to justify.
 *
 * The horror of a "backrooms" space is not a monster, it is a floor plan the
 * brain accepts as human construction and cannot find a use for. Office
 * partitions that enclose nothing, a corridor with a door onto a wall, a room
 * whose proportions are almost right. That effect has a specific technical
 * requirement, and it is the reason this file exists rather than another dungeon
 * generator: **no single algorithm can produce it**.
 *
 * A pure BSP gives offices too perfect to unsettle. A pure cellular automaton
 * gives caves, which read as nature and therefore as safe. What produces unease
 * is the *layering*: rigid order laid down first, gnawed at by a process that
 * knows nothing about it, then dressed by a solver that makes every local detail
 * connect flawlessly. Global nonsense, local perfection — a space that looks
 * designed, by someone who was not thinking.
 *
 * So the pipeline is fixed and its order is the point:
 *
 *   1. **Zone** — low-frequency noise decides what KIND of space each area is
 *      (tight corridors, open-plan offices, cavernous halls, tiled pools). This
 *      is what stops an infinite space from being uniformly infinite.
 *   2. **Partition** — a BSP per zone, with the zone dictating room size.
 *   3. **Erode** — cellular automata and asymmetric merges chew the partitions,
 *      and misaligned pillars destroy the vanishing point.
 *   4. **Repair** — a breadth-first pass guarantees every open cell is reachable.
 *      Non-negotiable: an unreachable pocket in an oppressive space is not
 *      atmosphere, it is a softlock.
 *   5. **Dress** — the eroded volume becomes a hard constraint on a tile solve.
 *
 * Infinity is by chunk, on absolute world coordinates, so two players in
 * opposite sectors agree without a message passing between them.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_LIMINAL_HPP
#    define LPL_PROCGEN_LIMINAL_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/procgen/Dungeon.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/procgen/QualityGate.hpp>

namespace lpl::procgen {

/**
 * @enum LiminalZone
 * @brief What kind of space an area is, before a single wall is placed.
 *
 * The atmospheric "super-biome" of the survey. Each one sets the partition's room
 * size and how hard the erosion bites, which is enough for the four to feel like
 * different buildings rather than different colours.
 */
enum class LiminalZone : core::u8 {
    Corridor = 0, ///< Tight, branching, claustrophobic.
    Office,       ///< Regular cubicle grid: the uncanny baseline.
    Hall,         ///< Cavernous open space, few supports.
    Pool,         ///< Wide tiled rooms with shallow partitions.
    Count         ///< Number of zones; never a classification.
};

/// Per-cell zone assignment.
using LiminalZoneMap = Grid<LiminalZone>;

/**
 * @struct LiminalParams
 * @brief Size, seed, and how hard each stage bites.
 */
struct LiminalParams {
    core::u32 width{64u};  ///< Cells along X.
    core::u32 depth{64u};  ///< Cells along Z.
    core::u32 seed{1337u}; ///< Determinism anchor.

    core::f32 zoneBelts{2.5f};   ///< Zone features across the map's longer axis.
    core::u32 zoneOctaves{2u};   ///< fBm octaves of the zoning field.
    core::u32 zoneSeed{0x5AFEu}; ///< Seed of the zoning field.

    core::u32 bspDepth{4u};           ///< Partition recursion depth.
    core::f32 erosionStrength{0.18f}; ///< Share of boundary cells the automata flip.
    core::f32 mergeStrength{0.22f};   ///< Share of party walls dissolved.
    core::f32 pillarDensity{0.02f};   ///< Share of open cells that become a stray pillar.

    core::u32 hotPathEvents{8u}; ///< Event sites to place along the critical path.
    core::u32 secretSites{3u};   ///< Reward sites to place in the deepest dead ends.
};

/**
 * @struct LiminalSpace
 * @brief Everything one liminal sector is.
 */
struct LiminalSpace {
    DungeonMap map;               ///< Walls and floor.
    LiminalZoneMap zones;         ///< What kind of space each cell belongs to.
    core::u32 openCells{0u};      ///< Walkable cells.
    core::u32 wallsBroken{0u};    ///< Cells the connectivity repair had to open.
    core::u32 wallsDissolved{0u}; ///< Cells the asymmetric merge opened.
    core::u32 pillars{0u};        ///< Stray pillars placed.
    bool connected{false};        ///< Whether every open cell is reachable.

    /// Cells the critical path runs through: where events belong.
    lpl::pmr::vector<core::u32> eventSites;
    /// Cells furthest from the critical path: where a reward belongs.
    lpl::pmr::vector<core::u32> secretSites;
};

/**
 * @brief Assigns a zone to every cell from a low-frequency field.
 *
 * Frequency is relative to the map, not fixed in cells, for the reason every
 * other field in this module is: a zone is meant to be a region a player walks
 * across, and a frequency in cells would keep the zone's *width* constant so a
 * bigger map would just have more of them.
 *
 * @param params Sizes and the zoning field's shape.
 * @return The zone map.
 */
[[nodiscard]] LiminalZoneMap zoneMap(const LiminalParams &params);

/**
 * @brief Builds one liminal sector, running the whole pipeline in order.
 * @param params Generation parameters.
 * @return The finished sector.
 */
[[nodiscard]] LiminalSpace generateLiminal(const LiminalParams &params);

/**
 * @brief Builds one sector of an unbounded liminal space.
 *
 * Everything derives from the world seed and the chunk's coordinates, so a sector
 * can be generated, discarded and regenerated identically, in any order, on any
 * machine.
 *
 * @param params World parameters (its @c width is the chunk edge).
 * @param coord  Which sector.
 * @return The sector.
 */
[[nodiscard]] LiminalSpace generateLiminalChunk(const LiminalParams &params, ChunkCoord coord);

/**
 * @brief Builds a solver preset that pins every masked cell to one tile.
 *
 * The "mask-first" solve the survey describes: the eroded geometry becomes an
 * immovable input to the tile solve rather than something applied afterwards. The
 * solver then has to make its dressing agree with a shape it did not choose,
 * which is what produces the effect — flawless local detail over a global plan
 * that makes no sense.
 *
 * @param mask       Non-zero marks a cell to pin.
 * @param pinnedTile Tile index the marked cells take.
 * @return A preset grid, pinned where masked and free elsewhere.
 */
[[nodiscard]] TileGrid presetFromMask(const Grid<core::u8> &mask, core::u8 pinnedTile);

/// @brief Human-readable zone name, for debugging and reports.
[[nodiscard]] const char *liminalZoneName(LiminalZone zone) noexcept;

/// @brief FNV-1a fold of the map and its zones, for determinism checks.
[[nodiscard]] core::u32 foldLiminal(const LiminalSpace &space);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_LIMINAL_HPP
