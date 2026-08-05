/**
 * @file Swarm.cpp
 * @brief Implementation of boids and the ant-colony move policy.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ai/Swarm.hpp>

#include <lpl/math/FixedMath.hpp>

namespace lpl::ai {

void stepBoids(Boid *boids, core::u32 count, const BoidParams &params, math::Fixed32 dt)
{
    if (boids == nullptr || count == 0u || dt.raw() <= 0)
        return;

    const math::Fixed32 separationWeight = math::Fixed32::fromFloat(params.separationWeight);
    const math::Fixed32 alignmentWeight = math::Fixed32::fromFloat(params.alignmentWeight);
    const math::Fixed32 cohesionWeight = math::Fixed32::fromFloat(params.cohesionWeight);
    const math::Fixed32 maxSpeed = math::Fixed32::fromFloat(params.maxSpeed);

    // A snapshot of the flock as it was at the start of the tick. Steering off
    // partially-updated neighbours makes the result depend on the iteration
    // order, which on a deterministic simulation is a desynchronisation and not
    // a stylistic choice.
    lpl::pmr::vector<Boid> before(count, Boid{});
    for (core::u32 i = 0u; i < count; ++i)
        before[i] = boids[i];

    for (core::u32 i = 0u; i < count; ++i)
    {
        const Boid &self = before[i];

        math::Fixed32 sepX{};
        math::Fixed32 sepZ{};
        math::Fixed32 aliX{};
        math::Fixed32 aliZ{};
        math::Fixed32 cohX{};
        math::Fixed32 cohZ{};
        core::u32 neighbours = 0u;
        core::u32 crowd = 0u;

        for (core::u32 j = 0u; j < count; ++j)
        {
            if (j == i)
                continue;
            const Boid &other = before[j];
            const math::Fixed32 dx = other.x - self.x;
            const math::Fixed32 dz = other.z - self.z;

            // Chebyshev radius: no square root, and the neighbourhood is a box,
            // which is what a spatial hash returns anyway — so the test agrees
            // with the query rather than trimming its results.
            const math::Fixed32 adx = dx.abs();
            const math::Fixed32 adz = dz.abs();
            const math::Fixed32 distance = adx > adz ? adx : adz;

            if (distance > params.neighbourRadius)
                continue;

            ++neighbours;
            aliX = aliX + other.vx;
            aliZ = aliZ + other.vz;
            cohX = cohX + other.x;
            cohZ = cohZ + other.z;

            if (distance < params.separationRadius && distance.raw() != 0)
            {
                // Repulsion falls off with distance, so a boid pushes hardest
                // against whatever is closest. Dividing by the distance rather
                // than its square keeps the force finite at contact — an inverse
                // square here launches overlapping boids across the map.
                const math::Fixed32 push = (params.separationRadius - distance) / params.separationRadius;
                sepX = sepX - dx * push;
                sepZ = sepZ - dz * push;
                ++crowd;
            }
        }

        math::Fixed32 ax{};
        math::Fixed32 az{};

        if (neighbours != 0u)
        {
            const math::Fixed32 n = math::Fixed32::fromInt(static_cast<core::i32>(neighbours));
            aliX = aliX / n;
            aliZ = aliZ / n;
            cohX = cohX / n - self.x;
            cohZ = cohZ / n - self.z;

            ax = ax + (aliX - self.vx) * alignmentWeight + cohX * cohesionWeight;
            az = az + (aliZ - self.vz) * alignmentWeight + cohZ * cohesionWeight;
        }
        if (crowd != 0u)
        {
            ax = ax + sepX * separationWeight;
            az = az + sepZ * separationWeight;
        }

        // Steering is an ACCELERATION, so it is scaled by the step as well: a flock
        // that turns as hard per tick as per second changes shape with the tick rate.
        math::Fixed32 vx = self.vx + ax * dt;
        math::Fixed32 vz = self.vz + az * dt;

        // Speed cap on the Chebyshev norm, for the same reason as above: no
        // square root in an authoritative path.
        const math::Fixed32 speed = vx.abs() > vz.abs() ? vx.abs() : vz.abs();
        if (speed > maxSpeed && speed.raw() != 0)
        {
            vx = vx * maxSpeed / speed;
            vz = vz * maxSpeed / speed;
        }

        boids[i].vx = vx;
        boids[i].vz = vz;
        boids[i].x = self.x + vx * dt;
        boids[i].z = self.z + vz * dt;
    }
}

core::u32 chooseAntMove(const StigmergyField &field, const AntParams &params, core::u32 x, core::u32 z,
                        core::u32 &stream, bool &outExplored)
{
    // Advance the caller's stream in place: the agent owns its randomness, so two
    // agents stepping in a different order still each get their own sequence.
    stream ^= stream << 13;
    stream ^= stream >> 17;
    stream ^= stream << 5;
    if (stream == 0u)
        stream = 0x9E3779B9u;

    const core::u32 roll = stream & 0xFu;
    outExplored = roll < params.explore16;

    if (outExplored)
    {
        // An explorer ignores the field completely. Not "prefers a weaker trail" —
        // ignores it, because a weighted choice still concentrates on whatever the
        // colony already believes, and the point is to look where nobody has.
        return (stream >> 8) % 8u;
    }
    return field.gradientDirection(params.channel, x, z, true);
}

void seedPheromoneField(StigmergyField &field, core::u32 channel, const core::u32 *goals, core::u32 count,
                        math::Fixed32 strength)
{
    if (goals == nullptr || count == 0u || field.empty())
        return;

    const core::u32 width = field.width();
    const core::u32 depth = field.depth();
    const core::u32 longAxis = width > depth ? width : depth;
    if (longAxis == 0u)
        return;

    for (core::u32 z = 0u; z < depth; ++z)
    {
        for (core::u32 x = 0u; x < width; ++x)
        {
            // Nearest goal by Chebyshev distance, then a linear falloff. Linear
            // rather than inverse-square: this is a hint about where to start
            // looking, and a sharp falloff would pin the colony to the goal's
            // immediate surroundings — which is the search it was supposed to
            // avoid having to do.
            core::u32 nearest = longAxis;
            for (core::u32 g = 0u; g < count; ++g)
            {
                const core::u32 gx = goals[g] % width;
                const core::u32 gz = goals[g] / width;
                const core::u32 dx = gx > x ? gx - x : x - gx;
                const core::u32 dz = gz > z ? gz - z : z - gz;
                const core::u32 distance = dx > dz ? dx : dz;
                if (distance < nearest)
                    nearest = distance;
            }

            const math::Fixed32 t = math::Fixed32::one() - math::Fixed32::fromInt(static_cast<core::i32>(nearest)) /
                                                               math::Fixed32::fromInt(static_cast<core::i32>(longAxis));
            if (t.raw() > 0)
                field.deposit(channel, x, z, strength * t);
        }
    }
}

} // namespace lpl::ai
