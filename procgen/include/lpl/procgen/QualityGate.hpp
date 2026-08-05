/**
 * @file QualityGate.hpp
 * @brief Asking a generated level whether it is actually playable.
 *
 * Generation can always produce something; whether that something is worth
 * shipping is a separate question, and one nobody is around to answer at
 * runtime. So the answer has to be computed. A Dijkstra distance map over the
 * walkable space gives it: whether the exit is reachable, how far it is, how
 * much of the level lies on the way there, and whether the level is a corridor
 * or a space.
 *
 * These are the properties a designer would check by playing. Computing them
 * turns "the seed produced a bad level" from something a player discovers into
 * something the generator rejects — a seed that fails its gate is simply not
 * used, and the next one is tried.
 *
 * The distance map is also directly useful beyond validation: it IS a flow
 * field, so anything that needs to move toward the exit can descend it without
 * pathfinding.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_QUALITYGATE_HPP
#    define LPL_PROCGEN_QUALITYGATE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Dungeon.hpp>
#    include <lpl/procgen/Grid.hpp>

namespace lpl::procgen {

/// Distance in steps from a source; @ref kUnreachable when no path exists.
using DistanceMap = Grid<core::u32>;

/// Marks a cell no path reaches.
inline constexpr core::u32 kUnreachable = 0xFFFFFFFFu;

/**
 * @struct LevelQuality
 * @brief What the gate measured.
 */
struct LevelQuality {
    bool goalReachable{false};     ///< Is the exit connected to the entrance?
    bool fullyConnected{false};    ///< Is every walkable cell reachable from the entrance?
    core::u32 walkableCells{0u};   ///< Total walkable cells.
    core::u32 reachableCells{0u};  ///< Cells the entrance can actually reach.
    core::u32 pathLength{0u};      ///< Steps from entrance to exit.
    core::u32 deadEnds{0u};        ///< Walkable cells with exactly one walkable neighbour.
    core::u32 junctions{0u};       ///< Walkable cells with three or more.
    core::u32 longestDistance{0u}; ///< Farthest reachable cell, in steps.
};

/**
 * @struct GateCriteria
 * @brief What a level has to satisfy to pass.
 *
 * Defaults are deliberately mild: a gate that rejects most seeds is a slow way
 * to generate nothing.
 */
struct GateCriteria {
    bool requireGoalReachable{true};  ///< The exit must be reachable.
    bool requireFullyConnected{true}; ///< No isolated pockets.
    core::u32 minPathLength{8u};      ///< Reject levels where the exit is next to the entrance.
    core::u32 minWalkableCells{32u};  ///< Reject levels too small to play in.
    core::u32 maxDeadEndRatio{60u};   ///< Reject mazes: max share of dead ends, in percent.
};

/**
 * @brief Breadth-first distances from one cell over the walkable space.
 * @param map     Level to traverse.
 * @param startX  Source column.
 * @param startZ  Source row.
 * @return A distance map; unreachable cells hold @ref kUnreachable.
 */
[[nodiscard]] DistanceMap computeDistanceMap(const DungeonMap &map, core::u32 startX, core::u32 startZ);

/**
 * @brief Breadth-first distances from many sources at once.
 *
 * Flooding from a whole set rather than a point is what turns a distance map into
 * a measurement of "how far off the beaten track is this?": seed it with every
 * cell of the critical path and each cell's value becomes its detour depth.
 *
 * @param map     Level to traverse.
 * @param sources Cell flags; every non-zero cell is a source at distance 0.
 * @return A distance map; unreachable cells hold @ref kUnreachable.
 */
[[nodiscard]] DistanceMap computeDistanceMapFrom(const DungeonMap &map, const Grid<core::u8> &sources);

/// A signed field an agent descends; unlike @ref DistanceMap it may go negative.
using DesireMap = Grid<core::i32>;

/**
 * @struct DesireTerm
 * @brief One want, and how much of it to take.
 */
struct DesireTerm {
    const DistanceMap *map{nullptr}; ///< Distances to the things this want is about.
    core::i32 weight{1};             ///< Scaled by 16: 16 is "×1", -16 is "flee", 8 is "half".
};

/**
 * @brief Blends several distance maps into one field an agent can descend.
 *
 * The report's "algebraic manipulation of desires", and the reason it is worth
 * having: an agent with three wants does not need three searches and an arbiter.
 * Scale each map by how much that want matters, add them, and descending the sum
 * satisfies all three in proportion — hungry pulls toward food, afraid pushes
 * away from the threat, and the agent walks the compromise without anything
 * having to decide between them.
 *
 * @warning A weighted sum can create local minima that none of its inputs had,
 *          and no amount of rescanning removes them: rescanning only enforces
 *          "no cell exceeds a neighbour by more than one step", which a plateau
 *          satisfies without offering a way down. That is the same gap that made
 *          the published flee-map recipe fail its own test here, so the honest
 *          contract is: a desire map is a good heuristic, not a guaranteed
 *          descent. When the guarantee is what matters — an escape that must
 *          work — use @ref computeFleeMap, which is a breadth-first search from
 *          the havens and therefore monotone by construction.
 *
 * @param map   The level, for walkability.
 * @param terms The wants to blend.
 * @param count How many terms.
 * @return The blended field; unreachable cells are left at 0.
 */
[[nodiscard]] DesireMap combineDesires(const DungeonMap &map, const DesireTerm *terms, core::u32 count);

/**
 * @brief The neighbouring cell a descent should step to, if any.
 * @param map     The level, for walkability.
 * @param desires The field to descend.
 * @param x       Current column.
 * @param z       Current row.
 * @param outX    Receives the next column.
 * @param outZ    Receives the next row.
 * @return false when no neighbour is strictly lower (a minimum, local or not).
 */
[[nodiscard]] bool descendDesire(const DungeonMap &map, const DesireMap &desires, core::u32 x, core::u32 z,
                                 core::u32 &outX, core::u32 &outZ);

/**
 * @struct HotPathAnalysis
 * @brief What the critical path through a level looks like, and what hangs off it.
 */
struct HotPathAnalysis {
    Grid<core::u8> onPath;        ///< 1 for cells on the shortest entrance-to-exit route.
    DistanceMap detour;           ///< Steps from the nearest path cell.
    core::u32 pathCells{0u};      ///< Length of the route, in cells.
    core::u32 deepestDetour{0u};  ///< Farthest any cell lies from the route.
    core::u32 excessiveCells{0u}; ///< Cells beyond @c detourLimit from the route.
    core::u32 farthestCell{0u};   ///< Flat index of the deepest cell — where a secret goes.
    bool valid{false};            ///< False when entrance and exit are not connected.
};

/**
 * @brief Extracts the critical path and measures everything that branches off it.
 *
 * The third step of the level-validation method in the knowledge base, and the one
 * that turns a distance map from a navigation aid into a design critique. Trace
 * the shortest route from entrance to exit — the "hot path", the spine of the
 * player's experience — then flood from the whole spine at once. Every other cell
 * now carries its detour depth.
 *
 * That single number answers two questions a generator otherwise cannot. A cell
 * far past the limit is an architectural excrescence: a long spur that exhausts a
 * player for nothing, and a reason to reject or repair the layout. And the very
 * deepest cell is, for exactly the same reason, the best place in the level to hide
 * something worth finding — the same measurement condemns the geometry and
 * furnishes it.
 *
 * @param map          Level to analyse.
 * @param startX       Entrance column.
 * @param startZ       Entrance row.
 * @param goalX        Exit column.
 * @param goalZ        Exit row.
 * @param detourLimit  Detour depth beyond which a cell counts as excessive.
 * @return The analysis; check @c valid first.
 */
[[nodiscard]] HotPathAnalysis analyseHotPath(const DungeonMap &map, core::u32 startX, core::u32 startZ, core::u32 goalX,
                                             core::u32 goalZ, core::u32 detourLimit);

/// What a spot on a level is FOR.
enum class PlacementRole : core::u8 {
    Encounter = 0, ///< On the spine: the player will pass through here.
    Reward = 1,    ///< Off it, deep: the player has to choose to come here.
};

/**
 * @struct Placement
 * @brief One spot a level furnishes, and why it was chosen.
 */
struct Placement {
    core::u32 x{0u};
    core::u32 z{0u};
    PlacementRole role{PlacementRole::Encounter};
    core::u32 detour{0u};   ///< Steps off the critical path; 0 for an encounter.
    core::u32 progress{0u}; ///< Steps from the entrance — how far in this is.
};

/**
 * @struct PlacementParams
 * @brief How much to put where.
 */
struct PlacementParams {
    core::u32 encounters{4u};     ///< Spots wanted on the spine.
    core::u32 rewards{2u};        ///< Spots wanted off it.
    core::u32 minSpacing{3u};     ///< Cells between two spots of the same role.
    core::u32 rewardMinDetour{2u}; ///< A reward must lie at least this far off the path.
};

/**
 * @brief Furnishes a level from its own measurements: events on the spine, prizes off it.
 *
 * @ref analyseHotPath already knows both answers and neither was ever used. The
 * critical path is where the player will certainly go, so that is where an
 * encounter is guaranteed to be met rather than missed; and the same analysis names
 * the cell farthest off that path, which is the best place in the level to hide
 * something for exactly the reason that made it an architectural excrescence — a
 * player only ever reaches it on purpose.
 *
 * No randomness. Encounters are spaced by PROGRESS along the spine, not by index
 * into a flat grid, so they land evenly through the player's route instead of
 * evenly through memory; rewards are the deepest dead ends, deepest first. Ties
 * break by ascending flat index, which is what makes the result foldable.
 *
 * A reward must be a DEAD END, not merely a deep cell: a deep cell in the middle of
 * a wide room is passed through, and a prize the player walks over by accident is
 * not a reward, it is litter.
 *
 * @param map        The level.
 * @param analysis   Its hot-path analysis; @c valid must be true.
 * @param startX     Entrance column, for measuring progress.
 * @param startZ     Entrance row.
 * @param params     How much to place.
 * @param out        Receives the placements, encounters first.
 * @param capacity   How many @p out can hold.
 * @return How many placements were written; fewer than asked when the level has no
 *         room for them, which is a fact about the level and not an error.
 */
[[nodiscard]] core::u32 placeAlongHotPath(const DungeonMap &map, const HotPathAnalysis &analysis, core::u32 startX,
                                          core::u32 startZ, const PlacementParams &params, Placement *out,
                                          core::u32 capacity);

/**
 * @brief Turns a danger map into one an agent can flee along.
 *
 * Fleeing is not "walk uphill on the danger map". An agent doing that reliably
 * wedges itself into the nearest corner, because a dead end is a local maximum:
 * every neighbour is closer to the threat, so it stops.
 *
 * The map returned here holds each cell's distance to the nearest place of safety,
 * so descending it always moves away from the threat and always arrives somewhere.
 * Its guarantee is structural rather than incidental — see the implementation note
 * for why the usual "negate and re-relax" recipe does not have it.
 *
 * @param map          Level the distances were taken over.
 * @param danger       Distances from whatever is being fled.
 * @param safeDistance Danger distance at or beyond which a cell is safety; 0 means
 *                     "wherever is furthest from the threat".
 * @return A map to descend to move away from the source.
 */
[[nodiscard]] DistanceMap computeFleeMap(const DungeonMap &map, const DistanceMap &danger, core::u32 safeDistance = 0u);

/**
 * @brief Measures a level's playability properties.
 * @param map     Level to measure.
 * @param startX  Entrance column.
 * @param startZ  Entrance row.
 * @param goalX   Exit column.
 * @param goalZ   Exit row.
 * @return The measurements.
 */
[[nodiscard]] LevelQuality evaluateLevel(const DungeonMap &map, core::u32 startX, core::u32 startZ, core::u32 goalX,
                                         core::u32 goalZ);

/**
 * @brief Does a level meet the criteria?
 * @param quality  Measurements from @ref evaluateLevel.
 * @param criteria What it has to satisfy.
 * @return true when every criterion holds.
 */
[[nodiscard]] bool passesGate(const LevelQuality &quality, const GateCriteria &criteria);

/**
 * @brief Finds the two walkable cells farthest apart, and uses them as endpoints.
 *
 * A generator rarely knows where its entrance and exit should be. Taking the
 * two ends of the longest path (the graph's diameter, approximated by the usual
 * double sweep) puts them where the level is most worth crossing, rather than
 * at arbitrary corners that might be adjacent.
 *
 * @param map     Level to inspect.
 * @param outStartX Receives the entrance column.
 * @param outStartZ Receives the entrance row.
 * @param outGoalX  Receives the exit column.
 * @param outGoalZ  Receives the exit row.
 * @return false when the level has no walkable cell.
 */
[[nodiscard]] bool findFarthestPair(const DungeonMap &map, core::u32 &outStartX, core::u32 &outStartZ,
                                    core::u32 &outGoalX, core::u32 &outGoalZ);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_QUALITYGATE_HPP
