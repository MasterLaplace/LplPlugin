/**
 * @file Swarm.hpp
 * @brief Many agents moving well together, at the cost of one thinking well.
 *
 * The split the swarm literature calls the Dynamic Particle Chain, and it is the
 * reason a thousand creatures cost less than ten pathfinders:
 *
 *  - **Macro** — which corridor to take — is read from a stigmergy field. No
 *    search, no per-agent state: look at the gradient under your feet and go.
 *  - **Micro** — how to move inside it without piling up — is three local rules
 *    over the nearest neighbours. Separation, alignment, cohesion.
 *
 * Neither half works alone. Gradient-following alone puts every agent on exactly
 * the same cell, because they all read the same number. Boids alone have no idea
 * where anything is. Together they produce flanking, queueing at chokepoints and
 * splitting around obstacles, none of which is written down anywhere.
 *
 * The exploration policy matters as much as either. A colony that only follows
 * the strongest trail cannot discover that the trail is now blocked — every agent
 * piles into the obstruction and the pheromone keeps reinforcing it. An
 * ε-greedy split (a share of agents ignore the field and wander) is what turns a
 * stuck colony into one that finds the detour, and it is measurable: with ε at
 * zero, a severed route stays severed.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_SWARM_HPP
#    define LPL_AI_SWARM_HPP

#    include <lpl/ai/StigmergyField.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ai {

/**
 * @struct BoidParams
 * @brief Weights of the three local rules, and how far a boid can see.
 */
struct BoidParams {
    math::Fixed32 separationRadius{math::Fixed32::fromRaw(2 << 16)}; ///< Below this, push apart.
    math::Fixed32 neighbourRadius{math::Fixed32::fromRaw(6 << 16)};  ///< Beyond this, ignore.
    core::f32 separationWeight{1.5f}; ///< Avoiding overlap. The strongest, deliberately.
    core::f32 alignmentWeight{0.6f};  ///< Matching the local heading.
    core::f32 cohesionWeight{0.3f};   ///< Drifting toward the local centre.
    core::f32 maxSpeed{1.0f};         ///< Speed cap, so the flock cannot accelerate forever.
};

/**
 * @struct Boid
 * @brief One agent's kinematic state, in world units.
 */
struct Boid {
    math::Fixed32 x{};
    math::Fixed32 z{};
    math::Fixed32 vx{};
    math::Fixed32 vz{};
};

/**
 * @brief Applies separation, alignment and cohesion to every boid.
 *
 * Double-buffered like the field, and for the same reason: a boid that reads a
 * neighbour the same sweep already moved is reacting to a future the others
 * cannot see, so the flock's shape would depend on iteration order.
 *
 * @param boids   Agents to steer, updated in place.
 * @param count   Entries in @p boids.
 * @param params  Weights and radii.
 */
/**
 * @brief Advances a flock by one step of @p dt seconds.
 *
 * @warning @p dt is explicit and has no default, deliberately. This function used
 *          to integrate `x + vx` — an implicit dt of one — which reads as harmless
 *          and means "one world unit per call". Driven from a 60 Hz fixed loop the
 *          flock therefore moved at sixty units a second, crossed a 128-cell map in
 *          two seconds, and jumped a full cell between ticks so that no caller
 *          could keep it out of a wall: every terrain check downstream was reading
 *          a position the boid had already teleported past. A default value would
 *          have preserved exactly that trap for the next caller.
 *
 * @param boids  Flock, updated in place.
 * @param count  Entries in @p boids.
 * @param params Weights, radii and the speed cap (in units per second).
 * @param dt     Duration of this step, in seconds.
 */
void stepBoids(Boid *boids, core::u32 count, const BoidParams &params, math::Fixed32 dt);

/**
 * @struct AntParams
 * @brief The colony's exploration/exploitation balance.
 */
struct AntParams {
    /**
     * @brief Share of moves that ignore the pheromone entirely, in sixteenths.
     *
     * The whole resilience of the colony lives here. At 0 the swarm follows the
     * strongest trail into whatever is now blocking it, and reinforces the trail
     * while doing so. A handful of agents wandering is what finds the detour.
     */
    core::u32 explore16{2u};

    math::Fixed32 depositQuality{math::Fixed32::fromRaw(64 << 16)}; ///< @f$Q@f$ in @f$Q/L@f$.
    core::u32 channel{0u}; ///< Which stigmergy channel carries the trail.
};

/**
 * @brief Chooses one agent's next move: follow the trail, or explore.
 *
 * @param field    The pheromone field.
 * @param params   Exploration balance.
 * @param x        Agent column.
 * @param z        Agent row.
 * @param stream   Per-agent, per-tick random stream state (advanced in place).
 * @param outExplored Set to true when the agent ignored the field this move.
 * @return A direction index into kNeighbor8, or @ref StigmergyField::kNoDirection.
 */
[[nodiscard]] core::u32 chooseAntMove(const StigmergyField &field, const AntParams &params, core::u32 x, core::u32 z,
                                      core::u32 &stream, bool &outExplored);

/**
 * @brief Seeds a pheromone field from distance to the known goals.
 *
 * The adaptive initialisation the ACO literature reports halves convergence
 * time: instead of starting uniform and letting random walks discover the
 * geometry, the field starts already sloping toward where the goals are. It is
 * a hint, not an answer — the trails still have to be walked — but it stops the
 * colony spending its first hundred ticks rediscovering that the map has a
 * shape.
 *
 * @param field   Field to seed.
 * @param channel Channel to write.
 * @param goals   Goal cells, as flat indices.
 * @param count   Entries in @p goals.
 * @param strength Peak value at a goal.
 */
void seedPheromoneField(StigmergyField &field, core::u32 channel, const core::u32 *goals, core::u32 count,
                        math::Fixed32 strength);

} // namespace lpl::ai

#endif // LPL_AI_SWARM_HPP
