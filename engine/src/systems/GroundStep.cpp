#include <lpl/engine/systems/GroundStep.hpp>

#include <lpl/math/FixedMath.hpp>
#include <lpl/procgen/Grid.hpp>

namespace lpl::engine::systems {

GroundStepResult stepOnGround(const ITerrainQuery &terrain, math::Vec3<math::Fixed32> &position,
                              math::Fixed32 &headingX, math::Fixed32 &headingZ, math::Fixed32 pace, math::Fixed32 reach)
{
    GroundStepResult result;

    if (reach < pace)
        reach = pace;

    if (!terrain.standable(position.x + headingX * reach, position.z + headingZ * reach))
    {
        math::Fixed32 bestX = headingX;
        math::Fixed32 bestZ = headingZ;
        math::Fixed32 bestDot = math::Fixed32::fromInt(-2);
        bool found = false;
        for (core::u32 n = 0u; n < 8u; ++n)
        {
            const math::Fixed32 candidateX =
                math::Fixed32::fromInt(procgen::kNeighbor8X[n]) * (n < 4u ? math::Fixed32::one() : math::kInvSqrt2);
            const math::Fixed32 candidateZ =
                math::Fixed32::fromInt(procgen::kNeighbor8Z[n]) * (n < 4u ? math::Fixed32::one() : math::kInvSqrt2);
            if (!terrain.standable(position.x + candidateX * reach, position.z + candidateZ * reach))
                continue;
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
        headingX = math::Fixed32{} - headingX;
        headingZ = math::Fixed32{} - headingZ;
        result.cornered = true;
    }
    return result;
}

} // namespace lpl::engine::systems
