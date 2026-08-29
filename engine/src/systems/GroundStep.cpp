/**
 * @file GroundStep.cpp
 * @brief Implementation of one step across ground that may refuse it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/engine/systems/GroundStep.hpp>

#include <lpl/math/FixedMath.hpp>
#include <lpl/procgen/Grid.hpp>

namespace lpl::engine::systems {

GroundStepResult stepOnGround(const ITerrainQuery &terrain, math::Vec3<math::Fixed32> &position,
                              math::Fixed32 &headingX, math::Fixed32 &headingZ, math::Fixed32 pace,
                              math::Fixed32 reach)
{
    GroundStepResult result;

    // @warning The look-ahead must cover what the step will actually traverse, and a caller passing a
    // shorter one is a body that walks into things it never saw. Measured: a traveller looking
    // eight units ahead while stepping thirty stopped dead against a boulder with `clear` true --
    // it checked a point it would fly past, then put its foot inside the rock, and the axis test
    // refused a move it had no reason to expect. Clamped HERE rather than documented, because a
    // rule every caller must remember is a rule one caller will forget; `LocomotionSystem` only
    // satisfied it because its animals are slow.
    if (reach < pace)
        reach = pace;

    // -- Avoidance, before the move rather than after refusing it --
    if (!terrain.standable(position.x + headingX * reach, position.z + headingZ * reach))
    {
        math::Fixed32 bestX = headingX;
        math::Fixed32 bestZ = headingZ;
        math::Fixed32 bestDot = math::Fixed32::fromInt(-2);
        bool found = false;
        for (core::u32 n = 0u; n < 8u; ++n)
        {
            const math::Fixed32 candidateX = math::Fixed32::fromInt(procgen::kNeighbor8X[n]) *
                                             (n < 4u ? math::Fixed32::one() : math::kInvSqrt2);
            const math::Fixed32 candidateZ = math::Fixed32::fromInt(procgen::kNeighbor8Z[n]) *
                                             (n < 4u ? math::Fixed32::one() : math::kInvSqrt2);
            if (!terrain.standable(position.x + candidateX * reach, position.z + candidateZ * reach))
                continue;
            // Closest to the current heading: turning is cheap, reversing is not, and a body that
            // takes the first free direction in array order makes every herd drift east.
            const math::Fixed32 dot = candidateX * headingX + candidateZ * headingZ;
            if (dot > bestDot)
            {
                bestDot = dot;
                bestX = candidateX;
                bestZ = candidateZ;
                found = true;
            }
        }
        if (found)
        {
            headingX = bestX;
            headingZ = bestZ;
            result.avoided = true;
        }
    }

    const math::Fixed32 stepX = headingX * pace;
    const math::Fixed32 stepZ = headingZ * pace;
    const math::Fixed32 tryX = position.x + stepX;
    const math::Fixed32 tryZ = position.z + stepZ;

    // Axes tested separately, DIAGONAL included: testing the two axes apart and then moving along
    // both walks the corner between two free cells into the blocked one they share, which puts the
    // body inside the rock the next tick has to rescue it from.
    const bool freeX = terrain.standable(tryX, position.z);
    const bool freeZ = terrain.standable(position.x, tryZ);
    if (freeX && freeZ && terrain.standable(tryX, tryZ))
    {
        position.x = tryX;
        position.z = tryZ;
        result.moved = true;
    }
    else if (freeX)
    {
        position.x = tryX;
        result.moved = true;
    }
    else if (freeZ)
    {
        position.z = tryZ;
        result.moved = true;
    }
    else
    {
        // Cornered: turn around rather than freeze. The HEADING is reversed, not the velocity --
        // zeroing that destroys the very state the flocking rules accumulate, and a herd of those
        // shudders in place.
        headingX = math::Fixed32{} - headingX;
        headingZ = math::Fixed32{} - headingZ;
        result.cornered = true;
    }
    return result;
}

} // namespace lpl::engine::systems
