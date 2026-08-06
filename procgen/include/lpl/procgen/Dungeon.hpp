/**
 * @file Dungeon.hpp
 * @brief Underground space: ordered by construction, or organic by accretion.
 *
 * Two opposite families, because they answer different needs. Top-down
 * partitioning (BSP) produces rooms and corridors — legible, navigable, and
 * exactly right for a built structure. Bottom-up accretion (cellular automata,
 * a wandering digger) produces caves — irregular, surprising, and right for
 * anything that was not designed by anyone.
 *
 * The distinction that matters in practice is connectivity. BSP guarantees it
 * structurally: corridors are traced by walking the tree bottom-up, sibling to
 * sibling then parent to parent, so no leaf can be orphaned. The bottom-up
 * methods guarantee nothing at all — a cellular automaton routinely produces
 * sealed pockets, and a level with an unreachable room is worse than a boring
 * one. So every generator here ends with the same post-process:
 * @ref connectRegions floods the grid, discards pockets below a threshold, and
 * digs straight links between whatever survives.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_DUNGEON_HPP
#    define LPL_PROCGEN_DUNGEON_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/// Cell states of a dungeon map.
enum class DungeonCell : core::u8 {
    Wall = 0, ///< Solid rock.
    Floor,    ///< Walkable.
    Door      ///< Walkable, and marks a room boundary.
};

/// A dungeon map.
using DungeonMap = Grid<DungeonCell>;

/**
 * @brief Can a body stand here?
 *
 * Public because it is a property of the cell type, not of any one pass, and
 * because it was previously spelled out twice — once in the dungeon generators
 * and once in the quality gate. Two answers to "what counts as walkable" is one
 * too many for a predicate that decides what *connected* means.
 *
 * @param cell Cell to test.
 */
[[nodiscard]] constexpr bool isWalkable(DungeonCell cell) noexcept { return cell != DungeonCell::Wall; }

/**
 * @enum CaveKind
 * @brief Which underground generator a recipe asks for.
 *
 * Four generators existed and a recipe could name exactly one of them, so the other
 * three were reachable only by writing @ref WorldBuilder calls by hand — which is
 * what a viewer did, and why its world could not be saved, baked, replayed in ring 0
 * or asked for by an intelligence. A director who cannot name what it wants is not a
 * director.
 *
 * @c Cellular stays the default because it is what @ref parityWorldRecipe bakes and
 * what every existing document therefore means.
 */
enum class CaveKind : core::u32 {
    Cellular = 0u, ///< Automaton smoothing with connectivity repair. The default.
    Bsp = 1u,      ///< Recursive partition into rooms joined by corridors.
    Dla = 2u,      ///< Diffusion-limited aggregation: thin, branching, organic.

    /**
     * @brief A stack of plans joined by shafts, at least one reaching the surface.
     *
     * The gate judges this one in three dimensions — see @ref evaluateCaveSystem. It
     * did not, for a while: @ref GateCriteria was asked of the flat @c DungeonMap,
     * which this generator leaves empty, so a layered recipe reported zero open cells
     * and failed a world that was perfectly navigable, and a document had to switch
     * @c checkPlayability off to use the generator at all.
     *
     * Worth keeping in mind because the first test written for it passed for exactly
     * that reason and proved nothing: the layered cave "differed from the default" by
     * reporting nothing at all.
     */
    Layered = 3u,

    /**
     * @brief Let the PLACE decide. The default a document gets when it says nothing.
     *
     * The other four are the developer's answer, and an editor must be able to give
     * one — a director who cannot name what it wants is not a director, which is the
     * whole reason this enum exists. But naming one for every world is also how every
     * cave in a world ends up identical, and a streamed world has thousands of them.
     *
     * So this value is not "random". It is @ref chooseCaveKind: the evidence a site
     * actually offers — were there people here, does the ground dissolve, how deep is
     * this layer — resolved into one of the four above. It is a pure function of that
     * evidence, so two targets asking about one place get one answer.
     *
     * @warning Never reaches a generator. Every consumer resolves it first, and
     *          @ref generateCaveSystem clamps it to @c Cellular if one forgets, rather
     *          than indexing a switch that has no case for it.
     */
    Auto = 4u
};

/**
 * @struct Room
 * @brief An axis-aligned rectangle carved out of the rock.
 */
struct Room {
    core::u32 x{0u};     ///< Left edge.
    core::u32 z{0u};     ///< Top edge.
    core::u32 width{0u}; ///< Extent along X.
    core::u32 depth{0u}; ///< Extent along Z.

    /// @return Centre column (rounded down).
    [[nodiscard]] core::u32 centerX() const noexcept { return x + width / 2u; }
    /// @return Centre row (rounded down).
    [[nodiscard]] core::u32 centerZ() const noexcept { return z + depth / 2u; }
};

/**
 * @struct BspDungeonParams
 * @brief Recursive partitioning into rooms joined by corridors.
 */
struct BspDungeonParams {
    core::u32 width{64u};        ///< Map width.
    core::u32 depth{64u};        ///< Map depth.
    core::u32 seed{1337u};       ///< Determinism anchor.
    core::u32 maxDepth{5u};      ///< Recursion levels.
    core::u32 minLeafSize{8u};   ///< A node smaller than this is not split.
    core::u32 roomPadding{1u};   ///< Rock kept between a room and its leaf edge.
    core::u32 corridorWidth{1u}; ///< Corridor thickness.
};

/**
 * @struct CaveParams
 * @brief Cellular-automaton cave carving (the 4-5 rule).
 */
struct CaveParams {
    core::u32 width{64u};             ///< Map width.
    core::u32 depth{64u};             ///< Map depth.
    core::u32 seed{1337u};            ///< Determinism anchor.
    core::f32 fillProbability{0.45f}; ///< Chance a cell starts as rock.
    core::u32 steps{5u};              ///< Automaton iterations.
    core::u32 birthLimit{5u};         ///< Rock neighbours that turn floor to rock.
    core::u32 survivalLimit{4u};      ///< Rock neighbours a rock cell needs to stay rock.
    core::u32 minRegionSize{24u};     ///< Pockets smaller than this are filled in.
};

/**
 * @struct DrunkardParams
 * @brief A digger on a confined random walk.
 */
struct DrunkardParams {
    core::u32 width{64u};           ///< Map width.
    core::u32 depth{64u};           ///< Map depth.
    core::u32 seed{1337u};          ///< Determinism anchor.
    core::u32 diggers{4u};          ///< Independent walks.
    core::u32 stepsPerDigger{400u}; ///< Steps each walk takes.
    core::f32 targetFill{0.4f};     ///< Stop early once this share is floor.
    core::u32 margin{2u};           ///< Rock border the walk may not cross.
};

/**
 * @struct DungeonReport
 * @brief What a generator produced, and whether it is playable.
 */
struct DungeonReport {
    core::u32 floorCells{0u};    ///< Walkable cells.
    core::u32 regions{0u};       ///< Disconnected areas found before linking.
    core::u32 pocketsFilled{0u}; ///< Sub-threshold areas turned back to rock.
    core::u32 linksDug{0u};      ///< Corridors added to join surviving areas.
    bool connected{false};       ///< Is every floor cell reachable from every other?
};

/**
 * @brief Carves rooms into BSP leaves and joins them bottom-up.
 * @param params   Size, seed and partitioning limits.
 * @param outRooms Receives the carved rooms, in leaf order (may be null).
 * @return The map.
 */
[[nodiscard]] DungeonMap generateBspDungeon(const BspDungeonParams &params, lpl::pmr::vector<Room> *outRooms = nullptr);

/**
 * @brief Grows caves from noise with the 4-5 rule.
 * @param params Size, seed, fill and iteration counts.
 * @return The map (already connected; see @ref connectRegions).
 */
[[nodiscard]] DungeonMap generateCellularCave(const CaveParams &params);

/**
 * @brief Digs winding galleries with confined random walkers.
 * @param params Size, seed, digger count and budget.
 * @return The map.
 */
[[nodiscard]] DungeonMap generateDrunkardWalk(const DrunkardParams &params);

/**
 * @brief Fills small pockets and links what remains into one navigable space.
 *
 * The guarantee the bottom-up generators cannot make for themselves. Regions are
 * found by flood fill, anything under @p minRegionSize becomes rock again, and
 * the survivors are joined by digging straight L-corridors between their nearest
 * representative cells.
 *
 * @param map            Map to repair in place.
 * @param minRegionSize  Smallest area worth keeping.
 * @return A report of what was found and done.
 */
DungeonReport connectRegions(DungeonMap &map, core::u32 minRegionSize);

/**
 * @brief Checks that every floor cell is reachable from every other.
 * @param map Map to test.
 * @return true when the walkable space is a single connected region.
 */
[[nodiscard]] bool isFullyConnected(const DungeonMap &map);

/**
 * @brief Roughens room edges so a built structure reads as a ruin.
 *
 * The standard answer to BSP's rigidity: perturb the boundaries with noise so
 * the rectangles stop being rectangles, without touching the connectivity the
 * partitioning guaranteed.
 *
 * @param map      Map to erode in place.
 * @param seed     Determinism anchor.
 * @param strength Chance an eligible edge cell flips, in [0, 1].
 */
void erodeEdges(DungeonMap &map, core::u32 seed, core::f32 strength);

/**
 * @brief Breaks walls until every open cell is reachable from every other.
 *
 * The anti-softlock guarantee. Any generator that carves — cellular automata,
 * drunkard's walk, a BSP whose corridors were later eroded — can leave a pocket
 * with no way in, and a player who spawns in one has no game. Detecting it is not
 * enough: a generator that reports "not connected" and stops has only moved the
 * problem to its caller.
 *
 * So: flood the largest component, then for each orphan find the THINNEST wall
 * separating it from anything already connected, and break exactly that. Thinnest
 * rather than nearest, because a one-cell breach reads as a doorway while a long
 * gouge reads as a mistake.
 *
 * @param map  Map to repair in place.
 * @param seed Stream for tie-breaking between equally thin walls.
 * @return Number of wall cells broken (0 when it was already connected).
 */
core::u32 forceConnectivity(DungeonMap &map, core::u32 seed);

/**
 * @brief Merges neighbouring rooms by dissolving the wall between them.
 *
 * A BSP dungeon's rooms are all rectangles of similar size, which reads as a
 * floor plan rather than as a place. Dissolving a share of the party walls
 * produces asymmetric spaces no single split could have made — the "erosion of
 * perfect partitions" a liminal generator needs.
 *
 * @param map      Map to modify in place.
 * @param seed     Stream for which walls go.
 * @param strength Share of interior wall cells to dissolve, in [0, 1].
 * @return Number of cells opened.
 */
core::u32 mergeRoomsAsymmetric(DungeonMap &map, core::u32 seed, core::f32 strength);

/**
 * @brief Scatters single-cell pillars off the lattice the rooms were built on.
 *
 * Aligned pillars give a corridor a vanishing point, and a vanishing point is
 * what tells a player where they are. Misaligning them is the cheapest way to
 * take that away, which is the entire architectural trick of a liminal space.
 *
 * @param map     Map to modify in place.
 * @param seed    Stream for placement.
 * @param density Share of open cells that become a pillar, in [0, 1].
 * @return Number of pillars placed.
 */
core::u32 misalignPillars(DungeonMap &map, core::u32 seed, core::f32 density);

/**
 * @brief FNV-1a fold of a dungeon map, for determinism checks.
 * @param map Map to fold.
 * @return The 32-bit signature.
 */
[[nodiscard]] core::u32 foldDungeon(const DungeonMap &map);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_DUNGEON_HPP
