/**
 * @file ITerrainQuery.hpp
 * @brief What a system may ask the ground it is standing on.
 *
 * The seam this file closes was named, not smuggled: the creature systems could
 * deposit scent and steer by it, but not decide whether an animal may stand where
 * it is going, nor what it eats where it stands. Both need the terrain, and a
 * terrain was state a World held privately — so those two behaviours stayed inside
 * a 2000-line sample, reachable only as lambdas the sample passed down.
 *
 * A lambda would have worked and would have been worse. A system that takes a
 * callback from its owner cannot be reordered against the thing the callback
 * touches, cannot be given a fake in a test, and cannot say in its descriptor what
 * it depends on. An interface can: @c ecs::ResourceId::Terrain and
 * @c ecs::ResourceId::Vegetation name the two halves, and the scheduler builds the
 * same ordering edges for them as it does for components.
 *
 * Deliberately NOT here:
 *  - @c groundAt / the drawn height. That is a float, it is non-authoritative, and
 *    it belongs to the render path (engine::TerrainRenderer already takes it as a
 *    callable). Letting a simulation system read it would put a float on the
 *    authoritative path, which the determinism contract forbids.
 *  - anything that resembles a query language. Two questions, both answered in
 *    O(1) by every implementation, because a system asks them once per animal per
 *    tick.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-04
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_ITERRAINQUERY_HPP
#    define LPL_ENGINE_ITERRAINQUERY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::engine {

/**
 * @class ITerrainQuery
 * @brief The ground, as the two questions a walking animal asks it.
 *
 * Implemented by whatever generated the world — a bounded heightfield, a streamed
 * one, or a test's flat plane. A World implements it because a World is what knows
 * how its own cells map to world units.
 */
class ITerrainQuery {
public:
    virtual ~ITerrainQuery() = default;

    /**
     * @brief May a body occupy this spot?
     *
     * World units, authoritative Fixed32: this decides a position, so it may not
     * go through a float. Callers test the two axes SEPARATELY — an animal that
     * cannot move diagonally into a corner can still slide along the wall, which
     * is what stops a herd piling up against a cliff and vibrating.
     *
     * @param x World X.
     * @param z World Z.
     * @return True when a body may stand there.
     */
    [[nodiscard]] virtual bool standable(math::Fixed32 x, math::Fixed32 z) const = 0;

    /**
     * @brief A forager eats one plant at this world cell, if there is one.
     *
     * ONE plant per call and then return: eating everything in reach turns a
     * meadow into bare rock in a single tick, which reads as a bug even though
     * every step of it is correct.
     *
     * @param worldX Integer world cell, X.
     * @param worldZ Integer world cell, Z.
     * @return True when something was actually eaten.
     */
    virtual bool consumePlantAt(core::i32 worldX, core::i32 worldZ) = 0;

    /**
     * @brief Which way is out, for a body standing where it must not.
     *
     * The third question only a world can answer, and the one that made the
     * difference between a herd and a permanent traffic accident. A body CAN end up
     * somewhere unstandable — the flocking rules push it over a border, a pass
     * raises rock under it, a spawn lands badly — and from there every direction is
     * refused, so it refuses two steps a tick for the rest of the run. A few
     * escapees then look, from inside a counter, like a herd in constant collision.
     *
     * A bounded map answers "toward the middle", because that is where its ground
     * is. A streamed one has no middle and answers with the focus it streams
     * around. Neither answer is derivable from @ref standable, which only ever says
     * no.
     *
     * The default gives up, which is honest: a world that cannot say where its
     * ground is leaves the body to reverse and try again.
     *
     * @param x     Where the body is standing, world X.
     * @param z     Where the body is standing, world Z.
     * @param outX  Receives a unit-ish direction, X.
     * @param outZ  Receives a unit-ish direction, Z.
     * @return false when this world has no advice to offer.
     */
    [[nodiscard]] virtual bool recoveryDirection(math::Fixed32 x, math::Fixed32 z, math::Fixed32 &outX,
                                                 math::Fixed32 &outZ) const
    {
        (void) x;
        (void) z;
        (void) outX;
        (void) outZ;
        return false;
    }
};

} // namespace lpl::engine

#endif // LPL_ENGINE_ITERRAINQUERY_HPP
