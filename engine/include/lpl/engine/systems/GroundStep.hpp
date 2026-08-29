/**
 * @file GroundStep.hpp
 * @brief One step across ground that may refuse it, written once.
 *
 * @warning **Extracted rather than copied, and the copy was about to be written.** The rule lived
 * inside `LocomotionSystem` -- look a body-length ahead, turn to the free neighbour closest to the
 * heading, then move testing the axes apart -- and a historical traveller needed exactly the same
 * thing. Two steppers would have been two answers to "may a body go there", free to disagree
 * about corners; and the corner case is precisely the one that took three attempts to get right
 * the first time.
 *
 * @warning Parameterised by a PACE and a REACH rather than by a herd. `LocomotionSystem` derives both
 * from a genome and a personality, a traveller derives them from how far a man walks in a year --
 * neither of which the step has any business knowing. Passing `HerdParams` here would have made
 * a walking scholar a kind of animal.
 *
 * **What it does NOT do, deliberately.** Nothing about roads. A road is cheap ground, not a
 * corridor: `procgen::RoutingParams::reuseDiscount` already makes bodies converge on worn paths at
 * PLANNING time, which is where a preference belongs. Turning the edge of a road into an obstacle
 * would be a wall that never existed -- ancient roads were rarely paved and never fenced -- and,
 * worse, a body that cannot leave the road cannot depart from the plan, so a run could no longer
 * disagree with the corpus it is measured against. The only hard constraint here is the ground
 * itself.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_GROUNDSTEP_HPP
#    define LPL_ENGINE_SYSTEMS_GROUNDSTEP_HPP

#    include <lpl/engine/ITerrainQuery.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/math/Vec3.hpp>

namespace lpl::engine::systems {

/**
 * @struct GroundStepResult
 * @brief What the step had to do about the ground.
 */
struct GroundStepResult {
    bool avoided{false};  ///< The heading was turned to get round something.
    bool cornered{false}; ///< Nothing was free; the heading was reversed instead of freezing.
    bool moved{false};    ///< The position changed at all.
};

/**
 * @brief Moves a body one step, turning aside from ground it cannot stand on.
 *
 * @warning The look-ahead is a full @p reach rather than the fraction one tick covers: looking only as
 * far as the next step means the turn happens with the obstacle already underfoot, which is a
 * collision reported as an intention.
 *
 * @warning The axes are tested SEPARATELY and the diagonal as well. Testing the two apart and then
 * moving along both walks the corner between two free cells into the blocked one they share,
 * which puts the body inside the rock that the next tick has to rescue it from.
 *
 * @warning When nothing is free the HEADING is reversed rather than the velocity zeroed: zeroing
 * destroys the state the flocking rules accumulate, and a herd of those shudders in place.
 *
 * @param terrain  What may be stood on.
 * @param position Moved in place.
 * @param headingX Unit facing, x. Turned in place when the way is blocked.
 * @param headingZ Unit facing, z.
 * @param pace     Distance this step covers.
 * @param reach    How far ahead to look before committing. @warning Raised to @p pace when shorter: a
 *                 body that looks less far than it steps walks into what it never checked.
 * @return What it had to do.
 */
GroundStepResult stepOnGround(const ITerrainQuery &terrain, math::Vec3<math::Fixed32> &position,
                              math::Fixed32 &headingX, math::Fixed32 &headingZ, math::Fixed32 pace,
                              math::Fixed32 reach);

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_GROUNDSTEP_HPP
