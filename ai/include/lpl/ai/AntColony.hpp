/**
 * @file AntColony.hpp
 * @brief The agents an ACO field needs, and the one rule that closes a trail.
 *
 * This module already had @ref AntParams, @ref chooseAntMove, @ref depositTrail and
 * @ref seedPheromoneField — every *mechanism* of ant colony optimisation and not
 * one agent to run them. So the only colony in the project lived inside a viewer's
 * `main.cpp`: fifty lines holding the agent positions, the nest, and the rule that
 * makes the whole thing work.
 *
 * That rule is the part worth rescuing, because it is not obvious and it is not in
 * the mechanisms: **an agent that wandered past its forage range goes home.**
 * Without it the colony diffuses outward forever and the trail never closes into a
 * route — which is the difference between a pheromone field and a stain. Nothing in
 * `chooseAntMove` can express it, because the choice is local and the rule is about
 * the colony.
 *
 * Where the nest goes is the CALLER's, and that is not an omission: the viewer put
 * it on the town's plaza when the world had one, so the colony reads as something
 * happening in the world rather than as a demo running beside it. This module knows
 * nothing about settlements and should not — but it does insist the cell be one an
 * agent can stand on, because a nest in a lake produces a colony that never leaves
 * it.
 *
 * Deliberately NOT a system: the colony has no entities. It is agents on a grid,
 * and a caller that wants them to be entities has @c ecs and the creature systems.
 * Making this an `ISystem` would force a component for something whose whole state
 * is two integers per agent.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-04
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_ANT_COLONY_HPP
#    define LPL_AI_ANT_COLONY_HPP

#    include <lpl/ai/StigmergyField.hpp>
#    include <lpl/ai/Swarm.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ai {

/**
 * @struct AntColonyParams
 * @brief How many agents, how far they range, and on which channel.
 */
struct AntColonyParams {
    core::u32 agents{48u}; ///< Foragers walking the field.

    /**
     * @brief Chebyshev-ish range, in cells, before an agent is sent home.
     *
     * The colony's closing rule. Too large and the trail never closes; too small
     * and the agents never reach anything worth marking. It is a distance rather
     * than a tick count because what matters is where an agent IS, not how long it
     * has been out.
     */
    core::u32 forageRange{40u};

    core::u32 seed{0xA57Eu}; ///< Anchors the per-agent random streams.
    AntParams ants{};        ///< Exploration balance, deposit quality, channel.
};

/**
 * @class AntColony
 * @brief Foragers on a stigmergy field, with a nest they come back to.
 *
 * Holds no field: the field belongs to whoever else is depositing on it. Six
 * channels are named in @ref ScentChannel and the ants use exactly one — the one
 * the enumeration already calls @c Pheromone — so a colony and a food web can share
 * one field without reading each other's marks.
 */
class AntColony {
public:
    /**
     * @brief Places the nest and the agents on it, and seeds the field.
     *
     * Seeds the channel from the nest with @ref seedPheromoneField, which is the
     * adaptive initialisation the literature reports halves convergence: the field
     * starts already sloping toward home instead of spending its first hundred ticks
     * rediscovering that the map has a shape.
     *
     * @param field  The field to seed and later deposit on.
     * @param width  Grid width, in cells.
     * @param depth  Grid depth, in cells.
     * @param params Colony size, range and channel.
     * @param nestX  Nest column. Must be somewhere an agent can stand.
     * @param nestZ  Nest row.
     */
    void reset(StigmergyField &field, core::u32 width, core::u32 depth, const AntColonyParams &params, core::u32 nestX,
               core::u32 nestZ);

    /**
     * @brief One tick: every agent chooses, moves, may go home, and deposits.
     *
     * Deposit happens after the move and after the homing check, so a returning
     * agent marks the nest rather than the far cell it just left — the trail then
     * has both ends.
     *
     * @param field The field to read and deposit on.
     */
    void step(StigmergyField &field);

    [[nodiscard]] core::u32 agentCount() const noexcept { return static_cast<core::u32>(_x.size()); }
    [[nodiscard]] core::u32 agentX(core::u32 index) const noexcept { return _x[index]; }
    [[nodiscard]] core::u32 agentZ(core::u32 index) const noexcept { return _z[index]; }
    [[nodiscard]] core::u32 nestX() const noexcept { return _nestX; }
    [[nodiscard]] core::u32 nestZ() const noexcept { return _nestZ; }

    /// How many agents have been sent home since @ref reset. A colony whose returns
    /// stay at zero is not foraging, it is sitting on its nest.
    [[nodiscard]] core::u32 returns() const noexcept { return _returns; }

    /// How many moves ignored the field on the last @ref step — the exploration that
    /// finds a detour when the best route is blocked.
    [[nodiscard]] core::u32 explored() const noexcept { return _explored; }

private:
    lpl::pmr::vector<core::u32> _x;
    lpl::pmr::vector<core::u32> _z;

    /**
     * @brief ONE random stream for the whole colony, advanced agent by agent.
     *
     * Not per-agent, which would be tidier and would change every trail this has
     * ever produced. Determinism is unaffected — the agents are visited in index
     * order, so the sequence is fixed — but it does mean the colony is not
     * parallelisable across agents without re-baselining what it draws.
     */
    core::u32 _stream{0u};

    AntColonyParams _params{};
    core::u32 _width{0u};
    core::u32 _depth{0u};
    core::u32 _nestX{0u};
    core::u32 _nestZ{0u};
    core::u32 _returns{0u};
    core::u32 _explored{0u};
};

} // namespace lpl::ai

#    include <lpl/ai/AntColony.inl>

#endif // LPL_AI_ANT_COLONY_HPP
