/**
 * @file SpringBody.cpp
 * @brief Implementation of the soft-body integrator and the two-bone solve.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ai/SpringBody.hpp>

#include <lpl/math/FixedMath.hpp>

namespace lpl::ai {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

} // namespace

core::u32 SpringBody::addChunk(const BodyChunk &chunk)
{
    _chunks.push_back(chunk);
    return static_cast<core::u32>(_chunks.size()) - 1u;
}

void SpringBody::connect(core::u32 a, core::u32 b, math::Fixed32 stiffness)
{
    if (a >= _chunks.size() || b >= _chunks.size() || a == b)
        return;

    const math::Fixed32 dx = _chunks[b].x - _chunks[a].x;
    const math::Fixed32 dz = _chunks[b].z - _chunks[a].z;
    // Rest length is the CURRENT separation: a body is defined by the pose it was
    // built in, so assembling one never starts it fighting itself.
    const math::Fixed32 rest = math::fixedSqrt(dx * dx + dz * dz);

    BodyChunkConnection link;
    link.a = a;
    link.b = b;
    link.restLength = rest;
    link.stiffness = stiffness;
    _links.push_back(link);
}

void SpringBody::step(const SpringBodyParams &params)
{
    // ── Integrate ───────────────────────────────────────────────────────────
    for (core::u32 i = 0u; i < _chunks.size(); ++i)
    {
        BodyChunk &chunk = _chunks[i];
        if (chunk.inverseMass.raw() == 0)
            continue;

        chunk.vz = chunk.vz + params.gravityZ;
        chunk.vx = chunk.vx * params.damping;
        chunk.vz = chunk.vz * params.damping;

        // Chebyshev clamp: no square root, and a stiff spring that has just
        // received a large correction cannot fling a chunk across the world.
        const math::Fixed32 speed = chunk.vx.abs() > chunk.vz.abs() ? chunk.vx.abs() : chunk.vz.abs();
        if (speed > params.maxSpeed && speed.raw() != 0)
        {
            chunk.vx = chunk.vx * params.maxSpeed / speed;
            chunk.vz = chunk.vz * params.maxSpeed / speed;
        }

        chunk.x = chunk.x + chunk.vx;
        chunk.z = chunk.z + chunk.vz;
    }

    // ── Relax the constraints ───────────────────────────────────────────────
    //
    // Position-based, iterated a fixed number of times. Fixed rather than
    // "until converged": a variable iteration count makes the cost of a tick
    // depend on the configuration, and a real-time budget cannot absorb that.
    for (core::u32 iteration = 0u; iteration < params.relaxations; ++iteration)
    {
        for (core::u32 l = 0u; l < _links.size(); ++l)
        {
            const BodyChunkConnection &link = _links[l];
            BodyChunk &a = _chunks[link.a];
            BodyChunk &b = _chunks[link.b];

            const math::Fixed32 dx = b.x - a.x;
            const math::Fixed32 dz = b.z - a.z;
            const math::Fixed32 distance = math::fixedSqrt(dx * dx + dz * dz);
            if (distance.raw() == 0)
                continue;

            const math::Fixed32 error = distance - link.restLength;
            const math::Fixed32 totalInverse = a.inverseMass + b.inverseMass;
            if (totalInverse.raw() == 0)
                continue;

            // Hooke, applied as a positional correction split by inverse mass —
            // so a pinned chunk (inverse mass 0) takes none of it and the moving
            // one takes all, which is the same convention the physics solver uses
            // for immovable bodies.
            const math::Fixed32 correction = error * link.stiffness / totalInverse;
            const math::Fixed32 nx = dx / distance;
            const math::Fixed32 nz = dz / distance;

            a.x = a.x + nx * correction * a.inverseMass;
            a.z = a.z + nz * correction * a.inverseMass;
            b.x = b.x - nx * correction * b.inverseMass;
            b.z = b.z - nz * correction * b.inverseMass;
        }
    }
}

void SpringBody::pull(core::u32 chunk, math::Fixed32 targetX, math::Fixed32 targetZ, math::Fixed32 strength)
{
    if (chunk >= _chunks.size())
        return;
    BodyChunk &c = _chunks[chunk];
    c.vx = c.vx + (targetX - c.x) * strength;
    c.vz = c.vz + (targetZ - c.z) * strength;
}

math::Fixed32 SpringBody::strainEnergy() const
{
    math::Fixed32 total{};
    for (core::u32 l = 0u; l < _links.size(); ++l)
    {
        const BodyChunkConnection &link = _links[l];
        const math::Fixed32 dx = _chunks[link.b].x - _chunks[link.a].x;
        const math::Fixed32 dz = _chunks[link.b].z - _chunks[link.a].z;
        const math::Fixed32 error = math::fixedSqrt(dx * dx + dz * dz) - link.restLength;
        total = total + error * error * link.stiffness;
    }
    return total;
}

core::u32 SpringBody::fold() const
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < _chunks.size(); ++i)
    {
        const core::u32 x = static_cast<core::u32>(_chunks[i].x.raw());
        const core::u32 z = static_cast<core::u32>(_chunks[i].z.raw());
        hash = (hash ^ (x & 0xFFFFu)) * kFnv1aPrime;
        hash = (hash ^ (x >> 16)) * kFnv1aPrime;
        hash = (hash ^ (z & 0xFFFFu)) * kFnv1aPrime;
        hash = (hash ^ (z >> 16)) * kFnv1aPrime;
    }
    return hash;
}

TwoBoneSolution solveTwoBone(math::Fixed32 rootX, math::Fixed32 rootZ, math::Fixed32 targetX, math::Fixed32 targetZ,
                             math::Fixed32 upper, math::Fixed32 lower, bool flip)
{
    TwoBoneSolution solution;

    const math::Fixed32 dx = targetX - rootX;
    const math::Fixed32 dz = targetZ - rootZ;
    const math::Fixed32 distanceSquared = dx * dx + dz * dz;
    const math::Fixed32 distance = math::fixedSqrt(distanceSquared);

    // Reported, not clamped. A limb that silently snaps to full extension puts a
    // foot where there is no ground, and the body then walks on it.
    if (distance > upper + lower || distance.raw() == 0 || distance + lower < upper)
    {
        solution.reachable = false;
        // Still return the straightest plausible joint, so a caller that ignores
        // the flag gets a limb pointing at the target rather than at the origin.
        const math::Fixed32 nx = distance.raw() != 0 ? dx / distance : math::Fixed32::one();
        const math::Fixed32 nz = distance.raw() != 0 ? dz / distance : math::Fixed32::zero();
        solution.jointX = rootX + nx * upper;
        solution.jointZ = rootZ + nz * upper;
        return solution;
    }

    // Circle-circle intersection, in the projection form. The textbook spelling
    // takes an arc cosine of the law of cosines; that would need libm, which the
    // determinism contract forbids in an authoritative path. This form needs the
    // same information and only one square root — the hardware instruction the
    // contract does allow.
    //
    //   a = (d² + u² − l²) / 2d      distance from root to the joint's projection
    //   h = sqrt(u² − a²)            offset perpendicular to the root-target line
    const math::Fixed32 a = (distanceSquared + upper * upper - lower * lower) / (distance * math::Fixed32::fromInt(2));
    math::Fixed32 hSquared = upper * upper - a * a;
    if (hSquared < math::Fixed32::zero())
        hSquared = math::Fixed32::zero(); // Rounding at the reach limit, not a failure.
    const math::Fixed32 h = math::fixedSqrt(hSquared);

    const math::Fixed32 nx = dx / distance;
    const math::Fixed32 nz = dz / distance;
    const math::Fixed32 baseX = rootX + nx * a;
    const math::Fixed32 baseZ = rootZ + nz * a;

    // The perpendicular is (-nz, nx); flipping it picks the mirror solution,
    // which is the difference between a knee that bends forward and one that
    // bends back. Exposed rather than chosen, because which one is correct
    // depends on the creature.
    const math::Fixed32 sign = flip ? -math::Fixed32::one() : math::Fixed32::one();
    solution.jointX = baseX - nz * h * sign;
    solution.jointZ = baseZ + nx * h * sign;
    solution.reachable = true;
    return solution;
}

} // namespace lpl::ai
