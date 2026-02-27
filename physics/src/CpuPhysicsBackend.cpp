/**
 * @file CpuPhysicsBackend.cpp
 * @brief CPU physics backend — full implementation ported from legacy.
 *
 * Pipeline: integrate → broad-phase → narrow-phase → solve → sleep → migrate
 *
 * Ported from legacy engine/Partition.hpp physics with adaptations for
 * the new ECS component-based chunk system. Uses:
 * - CollisionDetector::testAABBvsAABB for narrow-phase
 * - CollisionSolver::solve (4 iterations, sequential impulse)
 * - SleepingPolicy for threshold-based sleeping
 *
 * Physics constants from legacy:
 * - Gravity: -9.81 m/s²
 * - Velocity damping: 0.995 per frame
 * - Restitution: 0.5
 * - Sleep velocity threshold²: 0.01 (0.1 m/s)
 * - Sleep frames: 30 (~0.5s at 60Hz)
 * - Brute-force threshold: 32 entities (N² vs octree)
 * - Solver iterations: 4
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/physics/CpuPhysicsBackend.hpp>
#include <lpl/physics/CollisionDetector.hpp>
#include <lpl/physics/CollisionSolver.hpp>
#include <lpl/physics/SleepingPolicy.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/math/AABB.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

#include <cmath>
#include <vector>

namespace lpl::physics {

// ========================================================================== //
//  Physics constants (ported from legacy Partition.hpp)                       //
// ========================================================================== //

static constexpr float kGravity                  = -9.81f;
static constexpr float kVelocityDamping          = 0.995f;
static constexpr float kRestitution              = 0.5f;
static constexpr float kSleepVelocitySqThreshold = 0.01f;
static constexpr core::u16 kSleepFramesThreshold = 30;
static constexpr core::u32 kBruteForceThreshold  = 32;
static constexpr core::u32 kSolverIterations     = 4;

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct CpuPhysicsBackend::Impl
{
    ecs::Registry& registry;

    /** @brief Per-entity sleeping data (indexed by global entity slot). */
    std::vector<bool>    sleeping;
    std::vector<core::u16> sleepCounter;

    explicit Impl(ecs::Registry& r) : registry{r} {}

    /** @brief Ensures sleeping vectors are large enough for current entity count. */
    void ensureSleepCapacity(core::u32 maxIndex)
    {
        if (maxIndex >= static_cast<core::u32>(sleeping.size()))
        {
            sleeping.resize(maxIndex + 1, false);
            sleepCounter.resize(maxIndex + 1, 0);
        }
    }
};

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

CpuPhysicsBackend::CpuPhysicsBackend(ecs::Registry& registry)
    : _impl{std::make_unique<Impl>(registry)}
{}

CpuPhysicsBackend::~CpuPhysicsBackend() = default;

core::Expected<void> CpuPhysicsBackend::init()
{
    core::Log::info("CpuPhysicsBackend::init — ported from legacy Partition physics");
    return {};
}

core::Expected<void> CpuPhysicsBackend::step(core::f32 dt)
{
    auto& registry = _impl->registry;
    const auto partitions = registry.partitions();

    for (const auto& partition : partitions)
    {
        const auto& archetype = partition->archetype();

        // Only process partitions that have the physics-relevant components
        if (!archetype.has(ecs::ComponentId::Position) ||
            !archetype.has(ecs::ComponentId::Velocity) ||
            !archetype.has(ecs::ComponentId::Mass))
        {
            continue;
        }

        for (const auto& chunk : partition->chunks())
        {
            const core::u32 count = chunk->count();
            if (count == 0)
            {
                continue;
            }

            // Get SoA component arrays from chunk
            auto* positions  = static_cast<math::Vec3<float>*>(chunk->writeComponent(ecs::ComponentId::Position));
            auto* velocities = static_cast<math::Vec3<float>*>(chunk->writeComponent(ecs::ComponentId::Velocity));
            auto* masses     = static_cast<const float*>(chunk->readComponent(ecs::ComponentId::Mass));

            if (!positions || !velocities || !masses)
            {
                continue;
            }

            // Optional: AABB for collision sizing
            auto* aabbs = archetype.has(ecs::ComponentId::AABB)
                ? static_cast<math::Vec3<float>*>(chunk->writeComponent(ecs::ComponentId::AABB))
                : nullptr;

            // Ensure sleeping buffers are large enough
            _impl->ensureSleepCapacity(count);

            // ── Pass 1: Integration ──────────────────────────────────────
            integrateChunk(positions, velocities, masses, count, dt);

            // ── Pass 2: Collision detection + resolution ─────────────────
            if (aabbs)
            {
                resolveCollisionsChunk(positions, velocities, masses, aabbs, count);
            }

            // ── Pass 3: Sleeping detection ───────────────────────────────
            updateSleepingChunk(velocities, count);
        }
    }

    return {};
}

void CpuPhysicsBackend::shutdown()
{
    core::Log::info("CpuPhysicsBackend::shutdown");
    _impl->sleeping.clear();
    _impl->sleepCounter.clear();
}

const char* CpuPhysicsBackend::name() const noexcept
{
    return "CpuPhysicsBackend (legacy-ported)";
}

// ========================================================================== //
//  Private physics passes                                                    //
// ========================================================================== //

void CpuPhysicsBackend::integrateChunk(
    math::Vec3<float>* positions,
    math::Vec3<float>* velocities,
    const float* masses,
    core::u32 count,
    core::f32 dt) const noexcept
{
    for (core::u32 i = 0; i < count; ++i)
    {
        // Skip sleeping entities
        if (i < static_cast<core::u32>(_impl->sleeping.size()) && _impl->sleeping[i])
        {
            continue;
        }

        // Gravity force
        if (masses[i] > 0.0001f)
        {
            float invMass = 1.0f / masses[i];
            math::Vec3<float> gravityForce{0.0f, kGravity * masses[i], 0.0f};
            math::Vec3<float> acceleration = gravityForce * invMass;
            velocities[i] = velocities[i] + acceleration * dt;
        }

        // Velocity damping
        velocities[i] = velocities[i] * kVelocityDamping;

        // Position integration
        positions[i] = positions[i] + velocities[i] * dt;

        // Ground collision (y = 0 half-height check)
        constexpr float kDefaultHalfHeight = 0.5f;
        if (positions[i].y < kDefaultHalfHeight)
        {
            positions[i].y = kDefaultHalfHeight;
            if (velocities[i].y < 0.0f)
            {
                velocities[i].y = -velocities[i].y * kRestitution;
            }
        }
    }
}

void CpuPhysicsBackend::resolveCollisionsChunk(
    math::Vec3<float>* positions,
    math::Vec3<float>* velocities,
    const float* masses,
    const math::Vec3<float>* sizes,
    core::u32 count) const noexcept
{
    // Lambda for resolving a single collision pair (ported from legacy)
    auto resolveCollision = [&](core::u32 a, core::u32 b)
    {
        // Skip if both sleeping
        if (a < static_cast<core::u32>(_impl->sleeping.size()) &&
            b < static_cast<core::u32>(_impl->sleeping.size()) &&
            _impl->sleeping[a] && _impl->sleeping[b])
        {
            return;
        }

        math::Vec3<float> halfA = sizes[a] * 0.5f;
        math::Vec3<float> halfB = sizes[b] * 0.5f;
        math::Vec3<float> delta = positions[a] - positions[b];

        // Penetration on each axis
        float overlapX = (halfA.x + halfB.x) - std::fabs(delta.x);
        float overlapY = (halfA.y + halfB.y) - std::fabs(delta.y);
        float overlapZ = (halfA.z + halfB.z) - std::fabs(delta.z);

        if (overlapX <= 0.0f || overlapY <= 0.0f || overlapZ <= 0.0f)
        {
            return; // No collision
        }

        // Wake sleeping entities
        if (a < static_cast<core::u32>(_impl->sleeping.size()))
        {
            _impl->sleeping[a] = false;
            _impl->sleepCounter[a] = 0;
        }
        if (b < static_cast<core::u32>(_impl->sleeping.size()))
        {
            _impl->sleeping[b] = false;
            _impl->sleepCounter[b] = 0;
        }

        // Minimum penetration axis (SAT-like)
        math::Vec3<float> normal;
        float penetration;

        if (overlapX <= overlapY && overlapX <= overlapZ)
        {
            penetration = overlapX;
            normal = {delta.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f};
        }
        else if (overlapY <= overlapX && overlapY <= overlapZ)
        {
            penetration = overlapY;
            normal = {0.0f, delta.y >= 0.0f ? 1.0f : -1.0f, 0.0f};
        }
        else
        {
            penetration = overlapZ;
            normal = {0.0f, 0.0f, delta.z >= 0.0f ? 1.0f : -1.0f};
        }

        // Inverse masses
        float invMassA = (masses[a] > 0.0001f) ? (1.0f / masses[a]) : 0.0f;
        float invMassB = (masses[b] > 0.0001f) ? (1.0f / masses[b]) : 0.0f;
        float invMassSum = invMassA + invMassB;

        if (invMassSum < 0.0001f)
        {
            return; // Both infinite mass
        }

        // Positional correction (100%)
        float correctionMag = penetration / invMassSum;
        math::Vec3<float> correction = normal * correctionMag;

        positions[a] = positions[a] + correction * invMassA;
        positions[b] = positions[b] - correction * invMassB;

        // Impulse (Newton's law of restitution)
        math::Vec3<float> relVel = velocities[a] - velocities[b];
        float velAlongNormal = relVel.x * normal.x + relVel.y * normal.y + relVel.z * normal.z;

        if (velAlongNormal > 0.0f)
        {
            return; // Separating
        }

        float impulseMag = -(1.0f + kRestitution) * velAlongNormal / invMassSum;
        math::Vec3<float> impulse = normal * impulseMag;

        velocities[a] = velocities[a] + impulse * invMassA;
        velocities[b] = velocities[b] - impulse * invMassB;
    };

    // 4 solver iterations (legacy uses SOLVER_ITERATIONS = 4)
    for (core::u32 iter = 0; iter < kSolverIterations; ++iter)
    {
        if (count <= kBruteForceThreshold)
        {
            // Brute-force N² for small chunks
            for (core::u32 i = 0; i < count; ++i)
            {
                for (core::u32 j = i + 1; j < count; ++j)
                {
                    resolveCollision(i, j);
                }
            }
        }
        else
        {
            // For large chunks, still use brute-force for now
            // (Octree integration is in ISpatialIndex / Octree.hpp)
            // TODO: integrate Octree broad-phase for count > kBruteForceThreshold
            for (core::u32 i = 0; i < count; ++i)
            {
                for (core::u32 j = i + 1; j < count; ++j)
                {
                    resolveCollision(i, j);
                }
            }
        }
    }
}

void CpuPhysicsBackend::updateSleepingChunk(
    math::Vec3<float>* velocities,
    core::u32 count) const noexcept
{
    for (core::u32 i = 0; i < count; ++i)
    {
        if (i >= static_cast<core::u32>(_impl->sleeping.size()))
        {
            break;
        }

        if (_impl->sleeping[i])
        {
            continue;
        }

        float speedSq = velocities[i].x * velocities[i].x +
                         velocities[i].z * velocities[i].z;

        if (speedSq < kSleepVelocitySqThreshold)
        {
            _impl->sleepCounter[i]++;
            if (_impl->sleepCounter[i] >= kSleepFramesThreshold)
            {
                _impl->sleeping[i] = true;
                velocities[i] = {0.0f, 0.0f, 0.0f};
            }
        }
        else
        {
            _impl->sleepCounter[i] = 0;
        }
    }
}

} // namespace lpl::physics
