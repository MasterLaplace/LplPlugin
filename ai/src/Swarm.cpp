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

        math::Fixed32 vx = self.vx + ax * dt;
        math::Fixed32 vz = self.vz + az * dt;

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
    stream ^= stream << 13;
    stream ^= stream >> 17;
    stream ^= stream << 5;
    if (stream == 0u)
        stream = 0x9E3779B9u;

    const core::u32 roll = stream & 0xFu;
    outExplored = roll < params.explore16;

    if (outExplored)
    {
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
