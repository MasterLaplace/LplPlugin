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

#include <lpl/core/Assert.hpp>
#include <lpl/core/EventBus.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/AABB.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Simd.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/physics/CollisionDetector.hpp>
#include <lpl/physics/CollisionEvent.hpp>
#include <lpl/physics/CollisionSolver.hpp>
#include <lpl/physics/CpuPhysicsBackend.hpp>
#include <lpl/physics/Octree.hpp>
#include <lpl/physics/SleepingPolicy.hpp>

#include <cmath>
#include <vector>

namespace lpl::physics {

// ========================================================================== //
//  Physics constants (ported from legacy Partition.hpp)                       //
// ========================================================================== //

static constexpr float kGravity = -9.81f;
static constexpr float kVelocityDamping = 0.995f;
static constexpr float kRestitution = 0.5f;
static constexpr float kSleepVelocitySqThreshold = 0.01f;
static constexpr core::u16 kSleepFramesThreshold = 30;
static constexpr core::u32 kBruteForceThreshold = 32;
static constexpr core::u32 kSolverIterations = 4;

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct CpuPhysicsBackend::Impl {
    ecs::Registry &registry;

    /** @brief Per-entity sleeping data (indexed by entity slot, not chunk-local). */
    std::vector<bool> sleeping;
    std::vector<core::u16> sleepCounter;

    explicit Impl(ecs::Registry &r) : registry{r} {}

    /** @brief Ensures sleeping vectors are large enough for a given entity slot. */
    void ensureSleepCapacity(core::u32 entitySlot)
    {
        if (entitySlot >= static_cast<core::u32>(sleeping.size()))
        {
            sleeping.resize(entitySlot + 256, false);
            sleepCounter.resize(entitySlot + 256, 0);
        }
    }

    /** @brief Check if a given entity slot is sleeping. */
    [[nodiscard]] bool isSleeping(core::u32 slot) const noexcept
    {
        return slot < static_cast<core::u32>(sleeping.size()) && sleeping[slot];
    }

    /** @brief Wake an entity by its slot. */
    void wake(core::u32 slot) noexcept
    {
        if (slot < static_cast<core::u32>(sleeping.size()))
        {
            sleeping[slot] = false;
            sleepCounter[slot] = 0;
        }
    }
};

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

CpuPhysicsBackend::CpuPhysicsBackend(ecs::Registry &registry) : _impl{std::make_unique<Impl>(registry)} {}

CpuPhysicsBackend::~CpuPhysicsBackend() = default;

core::Expected<void> CpuPhysicsBackend::init()
{
    core::Log::info("CpuPhysicsBackend::init — ported from legacy Partition physics");
    return {};
}

core::Expected<void> CpuPhysicsBackend::step(core::f32 dt)
{
    auto &registry = _impl->registry;
    const auto partitions = registry.partitions();

    for (const auto &partition : partitions)
    {
        const auto &archetype = partition->archetype();

        // Only process partitions that have the physics-relevant components
        if (!archetype.has(ecs::ComponentId::Position) || !archetype.has(ecs::ComponentId::Velocity) ||
            !archetype.has(ecs::ComponentId::Mass))
        {
            continue;
        }

        for (const auto &chunk : partition->chunks())
        {
            const core::u32 count = chunk->count();
            if (count == 0)
            {
                continue;
            }

            // Get SoA component arrays from chunk
            auto *positions = static_cast<math::Vec3<float> *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *velocities = static_cast<math::Vec3<float> *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            auto *masses = static_cast<const float *>(chunk->readComponent(ecs::ComponentId::Mass));

            if (!positions || !velocities || !masses)
            {
                continue;
            }

            // Optional: AABB for collision sizing
            auto *aabbs = archetype.has(ecs::ComponentId::AABB) ?
                              static_cast<math::Vec3<float> *>(chunk->writeComponent(ecs::ComponentId::AABB)) :
                              nullptr;

            // Ensure sleeping buffers cover the entity slots in this chunk
            const ecs::EntityId *entities = chunk->entities().data();
            for (core::u32 i = 0; i < count; ++i)
            {
                _impl->ensureSleepCapacity(entities[i].slot());
            }

            // ── Pass 1: Integration ──────────────────────────────────────
            integrateChunk(entities, positions, velocities, masses, count, dt);

            // ── Pass 2: Collision detection + resolution ─────────────────
            if (aabbs)
            {
                resolveCollisionsChunk(entities, positions, velocities, masses, aabbs, count);
            }

            // ── Pass 3: Sleeping detection ───────────────────────────────
            updateSleepingChunk(entities, velocities, count);
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

const char *CpuPhysicsBackend::name() const noexcept { return "CpuPhysicsBackend (legacy-ported)"; }

// ========================================================================== //
//  Private physics passes                                                    //
// ========================================================================== //

void CpuPhysicsBackend::integrateChunk(const ecs::EntityId *entities, math::Vec3<float> *positions,
                                       math::Vec3<float> *velocities, const float *masses, core::u32 count,
                                       core::f32 dt) const noexcept
{
    using namespace math::simd;

    const SimdFloat4 vDamping = SimdFloat4::splat(kVelocityDamping);
    const SimdFloat4 vDt = SimdFloat4::splat(dt);

    core::u32 i = 0;

    // Vectorized loop (4 entities at a time)
    // Only process a batch if all 4 are awake for simplicity
    for (; i + 3 < count; i += 4)
    {
        bool anySleeping = false;
        for (core::u32 j = 0; j < 4; ++j)
        {
            if (_impl->isSleeping(entities[i + j].slot()))
            {
                anySleeping = true;
                break;
            }
        }
        if (anySleeping)
        {
            // Process this block of 4 in scalar since some are sleeping
            for (core::u32 j = 0; j < 4; ++j)
            {
                const core::u32 k = i + j;
                if (_impl->isSleeping(entities[k].slot()))
                    continue;

                if (masses[k] > 0.0001f)
                {
                    velocities[k].y += kGravity * dt;
                }

                velocities[k] = velocities[k] * kVelocityDamping;
                positions[k] = positions[k] + velocities[k] * dt;

                constexpr float kDefaultHalfHeight = 0.5f;
                if (positions[k].y < kDefaultHalfHeight)
                {
                    positions[k].y = kDefaultHalfHeight;
                    if (velocities[k].y < 0.0f)
                    {
                        velocities[k].y = -velocities[k].y * kRestitution;
                    }
                }
            }
            continue;
        }

        // --- Load AoS -> SoA ---
        float px[4] = {positions[i].x, positions[i + 1].x, positions[i + 2].x, positions[i + 3].x};
        float py[4] = {positions[i].y, positions[i + 1].y, positions[i + 2].y, positions[i + 3].y};
        float pz[4] = {positions[i].z, positions[i + 1].z, positions[i + 2].z, positions[i + 3].z};
        float vx[4] = {velocities[i].x, velocities[i + 1].x, velocities[i + 2].x, velocities[i + 3].x};
        float vy[4] = {velocities[i].y, velocities[i + 1].y, velocities[i + 2].y, velocities[i + 3].y};
        float vz[4] = {velocities[i].z, velocities[i + 1].z, velocities[i + 2].z, velocities[i + 3].z};
        float ms[4] = {masses[i], masses[i + 1], masses[i + 2], masses[i + 3]};

        SimdFloat4 posX = SimdFloat4::load(px);
        SimdFloat4 posY = SimdFloat4::load(py);
        SimdFloat4 posZ = SimdFloat4::load(pz);
        SimdFloat4 velX = SimdFloat4::load(vx);
        SimdFloat4 velY = SimdFloat4::load(vy);
        SimdFloat4 velZ = SimdFloat4::load(vz);

        // Gravity force — active if mass > 0.0001f
        for (int j = 0; j < 4; ++j)
        {
            if (ms[j] > 0.0001f)
            {
                vy[j] += kGravity * dt;
            }
        }
        velY = SimdFloat4::load(vy);

        // Damping
        velX = velX * vDamping;
        velY = velY * vDamping;
        velZ = velZ * vDamping;

        // Position integration
        posX = SimdFloat4::fma(velX, vDt, posX);
        posY = SimdFloat4::fma(velY, vDt, posY);
        posZ = SimdFloat4::fma(velZ, vDt, posZ);

        // Unpack SoA -> AoS
        posX.store(px);
        posY.store(py);
        posZ.store(pz);
        velX.store(vx);
        velY.store(vy);
        velZ.store(vz);

        for (int j = 0; j < 4; ++j)
        {
            positions[i + j] = {px[j], py[j], pz[j]};
            velocities[i + j] = {vx[j], vy[j], vz[j]};

            // Ground collision
            constexpr float kDefaultHalfHeight = 0.5f;
            if (positions[i + j].y < kDefaultHalfHeight)
            {
                positions[i + j].y = kDefaultHalfHeight;
                if (velocities[i + j].y < 0.0f)
                {
                    velocities[i + j].y = -velocities[i + j].y * kRestitution;
                }
            }
        }
    }

    // Scalar fallback/tail
    for (; i < count; ++i)
    {
        if (_impl->isSleeping(entities[i].slot()))
            continue;

        if (masses[i] > 0.0001f)
        {
            velocities[i].y += kGravity * dt;
        }

        velocities[i] = velocities[i] * kVelocityDamping;
        positions[i] = positions[i] + velocities[i] * dt;

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

void CpuPhysicsBackend::resolveCollisionsChunk(const ecs::EntityId *entities, math::Vec3<float> *positions,
                                               math::Vec3<float> *velocities, const float *masses,
                                               const math::Vec3<float> *sizes, core::u32 count) const noexcept
{
    // Lambda for resolving a single collision pair (ported from legacy)
    auto resolveCollision = [&](core::u32 a, core::u32 b) {
        const core::u32 slotA = entities[a].slot();
        const core::u32 slotB = entities[b].slot();

        // Skip if both sleeping
        if (_impl->isSleeping(slotA) && _impl->isSleeping(slotB))
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
        _impl->wake(slotA);
        _impl->wake(slotB);

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

        if (impulseMag > 0.0f)
        {
            core::EventBus::instance().publish(CollisionEvent{entities[a], entities[b], normal, impulseMag});
        }
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
            // Octree broad-phase for larger chunks (O(N log N) vs O(N²))
            // Build world-bounds from all entities
            math::Vec3<math::Fixed32> wMin{math::Fixed32::fromFloat(-500.0f), math::Fixed32::fromFloat(-500.0f),
                                           math::Fixed32::fromFloat(-500.0f)};
            math::Vec3<math::Fixed32> wMax{math::Fixed32::fromFloat(500.0f), math::Fixed32::fromFloat(500.0f),
                                           math::Fixed32::fromFloat(500.0f)};
            Octree octree(math::AABB<math::Fixed32>{wMin, wMax});

            // Insert all entities into the octree
            for (core::u32 i = 0; i < count; ++i)
            {
                math::Vec3<float> halfSz = sizes[i] * 0.5f;
                math::AABB<math::Fixed32> aabb{
                    {math::Fixed32::fromFloat(positions[i].x - halfSz.x),
                     math::Fixed32::fromFloat(positions[i].y - halfSz.y),
                     math::Fixed32::fromFloat(positions[i].z - halfSz.z)},
                    {math::Fixed32::fromFloat(positions[i].x + halfSz.x),
                     math::Fixed32::fromFloat(positions[i].y + halfSz.y),
                     math::Fixed32::fromFloat(positions[i].z + halfSz.z)}
                };
                octree.insert(i, aabb);
            }
            octree.rebuild();

            // Query each entity's expanded AABB for candidate pairs
            for (core::u32 i = 0; i < count; ++i)
            {
                math::Vec3<float> halfSz = sizes[i] * 0.5f;
                // Slightly expanded query region
                math::AABB<math::Fixed32> queryRegion{
                    {math::Fixed32::fromFloat(positions[i].x - halfSz.x),
                     math::Fixed32::fromFloat(positions[i].y - halfSz.y),
                     math::Fixed32::fromFloat(positions[i].z - halfSz.z)},
                    {math::Fixed32::fromFloat(positions[i].x + halfSz.x),
                     math::Fixed32::fromFloat(positions[i].y + halfSz.y),
                     math::Fixed32::fromFloat(positions[i].z + halfSz.z)}
                };

                octree.query(queryRegion, [&](core::u32 j) {
                    if (j > i) // avoid duplicate pairs
                    {
                        resolveCollision(i, j);
                    }
                });
            }
        }
    }
}

void CpuPhysicsBackend::updateSleepingChunk(const ecs::EntityId *entities, math::Vec3<float> *velocities,
                                            core::u32 count) const noexcept
{
    for (core::u32 i = 0; i < count; ++i)
    {
        const core::u32 slot = entities[i].slot();

        if (_impl->isSleeping(slot))
        {
            continue;
        }

        // Include all 3 axes (X+Y+Z) — legacy bug ignored Y causing
        // entities in free-fall to freeze
        const float speedSq =
            velocities[i].x * velocities[i].x + velocities[i].y * velocities[i].y + velocities[i].z * velocities[i].z;

        if (speedSq < kSleepVelocitySqThreshold)
        {
            _impl->sleepCounter[slot]++;
            if (_impl->sleepCounter[slot] >= kSleepFramesThreshold)
            {
                _impl->sleeping[slot] = true;
                velocities[i] = {0.0f, 0.0f, 0.0f};
            }
        }
        else
        {
            _impl->sleepCounter[slot] = 0;
        }
    }
}

} // namespace lpl::physics
