/**
 * @file AiMap.hpp
 * @brief Navigation that knows a body has a shape and a facing.
 *
 * Ordinary grid pathfinding treats a cell as a point: you are in it or you are
 * not, and every way out costs the same. That is fine for a dot. It is wrong for
 * a lizard five segments long in a tunnel, and wrong in a specific, visible way —
 * the shortest route tells it to reverse, and reversing means folding its own
 * body through itself.
 *
 * The fix from the Rain World study is to make the search state richer than the
 * cell: **(cell, direction of arrival)**. A step that reverses the arrival
 * direction then has its own cost, and can be made expensive without making
 * anything else expensive. An agent that is cornered will *prefer to go forward
 * to an intersection where it can turn around* — which is exactly what a real
 * long-bodied animal does, and it emerges from the cost table rather than from a
 * special case.
 *
 * The second axis is **capability**. A cell is not simply passable: it is
 * swimmable, climbable, crawlable. A bitmask per cell and a bitmask per creature
 * means one map serves every species, and a fish never routes across dry land
 * without a single test that mentions fish.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_AIMAP_HPP
#    define LPL_AI_AIMAP_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ai {

/**
 * @enum Locomotion
 * @brief How a body may move through a cell. Combine as a mask.
 */
enum class Locomotion : core::u8 {
    None = 0,
    Walk = 1u << 0,  ///< Ordinary ground.
    Swim = 1u << 1,  ///< Water.
    Climb = 1u << 2, ///< Vertical surface.
    Crawl = 1u << 3, ///< A gap too tight to stand in.
    Fly = 1u << 4    ///< Open air.
};

[[nodiscard]] constexpr core::u8 operator|(Locomotion a, Locomotion b) noexcept
{
    return static_cast<core::u8>(static_cast<core::u8>(a) | static_cast<core::u8>(b));
}

/// Direction count: the eight neighbours, plus "arrived from nowhere" for a start.
inline constexpr core::u32 kDirectionCount = 8u;
inline constexpr core::u32 kNoIncoming = 8u;

/**
 * @struct AiMapParams
 * @brief What each kind of move costs.
 */
struct AiMapParams {
    core::u32 baseCost{16u};     ///< Cost of a cardinal step.
    core::u32 diagonalCost{23u}; ///< Cost of a diagonal step (16 * sqrt(2), rounded).

    /**
     * @brief Extra cost for reversing the direction of arrival.
     *
     * The heart of the whole file. Large enough that a body would rather walk to
     * an intersection than back up, small enough that it will still reverse when
     * there is genuinely no alternative — an agent that CANNOT reverse gets stuck
     * in a dead end forever, which is a worse failure than an ugly turn.
     */
    core::u32 reverseCost{160u};

    core::u32 turnCost{4u}; ///< Small penalty per change of heading, so paths are smooth.
};

/**
 * @class AiMap
 * @brief Per-cell capability masks, and pathfinding over (cell, facing).
 */
class AiMap {
public:
    AiMap() = default;

    /**
     * @brief Allocates a map where every cell is impassable.
     * @param width Cells along X.
     * @param depth Cells along Z.
     */
    AiMap(core::u32 width, core::u32 depth);

    [[nodiscard]] core::u32 width() const noexcept { return _capability.width(); }
    [[nodiscard]] core::u32 depth() const noexcept { return _capability.depth(); }
    [[nodiscard]] bool empty() const noexcept { return _capability.empty(); }

    /// @brief Sets which locomotion modes a cell admits.
    void setCapability(core::u32 x, core::u32 z, core::u8 mask);

    /// @return The cell's capability mask, or 0 when out of range.
    [[nodiscard]] core::u8 capability(core::u32 x, core::u32 z) const;

    /// @return Whether a body with @p mask may occupy the cell.
    [[nodiscard]] bool passable(core::u32 x, core::u32 z, core::u8 mask) const
    {
        return (capability(x, z) & mask) != 0u;
    }

    /**
     * @brief Finds a route, charging for reversals.
     *
     * A* over states of (cell, incoming direction). The heuristic is the Chebyshev
     * distance scaled by @c baseCost, which never overestimates because no step
     * costs less than that — so the result is a genuine shortest path under the
     * cost model rather than a plausible one.
     *
     * @param startX  Start column.
     * @param startZ  Start row.
     * @param goalX   Goal column.
     * @param goalZ   Goal row.
     * @param mask    The body's locomotion modes.
     * @param params  Costs.
     * @param outPath Receives the route as flat cell indices, start first.
     * @return Total cost, or @ref kNoPath when unreachable.
     */
    [[nodiscard]] core::u32 findPath(core::u32 startX, core::u32 startZ, core::u32 goalX, core::u32 goalZ,
                                     core::u8 mask, const AiMapParams &params,
                                     lpl::pmr::vector<core::u32> &outPath) const;

    /// Returned by @ref findPath when no route exists.
    static constexpr core::u32 kNoPath = 0xFFFFFFFFu;

private:
    procgen::Grid<core::u8> _capability;
};

/**
 * @brief Does this path ever fold back on itself within @p bodyLength cells?
 *
 * The measurable version of "a long body does not pass through itself": take the
 * trailing @p bodyLength cells as the body's current occupancy and check the head
 * never enters one. A path that passes this is one a segmented creature can
 * physically follow.
 *
 * @param path       The route, as flat cell indices.
 * @param count      Entries in @p path.
 * @param bodyLength Segments behind the head.
 * @return Number of self-intersections (0 is what a valid path scores).
 */
[[nodiscard]] core::u32 countSelfIntersections(const core::u32 *path, core::u32 count, core::u32 bodyLength);

} // namespace lpl::ai

#endif // LPL_AI_AIMAP_HPP
