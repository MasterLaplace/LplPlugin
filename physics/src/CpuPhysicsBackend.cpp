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
 * DETERMINISM: authoritative Position/Velocity/AABB/Mass are Fixed32 (Q16.16).
 * All integration and collision math is scalar Fixed32 — bit-identical between
 * the Linux oracle and the i686 kernel. The former float SIMD (SimdFloat4) fast
 * path is removed: float vector math is non-deterministic across targets (FMA
 * contraction / rounding). A deterministic integer SIMD (SimdFixed4/8) path may
 * be reintroduced later, parity-gated against this scalar reference.
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
 * @version 0.2.0
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
#include <lpl/std/vector.hpp>

namespace lpl::physics {

using math::Fixed32;
using FVec3 = math::Vec3<Fixed32>;

// ========================================================================== //
//  Physics constants (ported from legacy Partition.hpp, now Fixed32)          //
// ========================================================================== //

static constexpr Fixed32 kGravity = Fixed32::fromFloat(-9.81f);
static constexpr Fixed32 kVelocityDamping = Fixed32::fromFloat(0.995f);
static constexpr Fixed32 kRestitution = Fixed32::fromFloat(0.5f);
static constexpr Fixed32 kSleepVelocitySqThreshold = Fixed32::fromFloat(0.01f);
static constexpr Fixed32 kMassEpsilon = Fixed32::fromFloat(0.0001f);
static constexpr Fixed32 kHalf = Fixed32::fromFloat(0.5f);
static constexpr Fixed32 kDefaultHalfHeight = Fixed32::fromFloat(0.5f);
static constexpr core::u16 kSleepFramesThreshold = 30;
static constexpr core::u32 kBruteForceThreshold = 32;
static constexpr core::u32 kSolverIterations = 4;

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct CpuPhysicsBackend::Impl {
    ecs::Registry &registry;

    /** @brief Per-entity sleeping data (indexed by entity slot, not chunk-local). */
    lpl::pmr::vector<bool> sleeping;
    lpl::pmr::vector<core::u16> sleepCounter;

    explicit Impl(ecs::Registry &r) : registry{r} {}

    /**
     * @brief Ensures sleeping vectors are large enough for a given entity slot.
     */
    void ensureSleepCapacity(core::u32 entitySlot)
    {
        if (entitySlot >= static_cast<core::u32>(sleeping.size()))
        {
            sleeping.resize(entitySlot + 256, false);
            sleepCounter.resize(entitySlot + 256, 0);
        }
    }

    /**
     * @brief Check if a given entity slot is sleeping.
     */
    [[nodiscard]] bool isSleeping(core::u32 slot) const noexcept
    {
        return slot < static_cast<core::u32>(sleeping.size()) && sleeping[slot];
    }

    /**
     * @brief Wake an entity by its slot.
     */
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

CpuPhysicsBackend::CpuPhysicsBackend(ecs::Registry &registry) : _impl{lpl::pmr::make_unique<Impl>(registry)} {}

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

            // Get SoA component arrays from chunk (authoritative Fixed32)
            auto *positions = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *velocities = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            auto *masses = static_cast<const Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));

            if (!positions || !velocities || !masses)
            {
                continue;
            }

            // Optional: AABB for collision sizing
            auto *aabbs = archetype.has(ecs::ComponentId::AABB) ?
                              static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::AABB)) :
                              nullptr;

            // Ensure sleeping buffers cover the entity slots in this chunk
            const ecs::EntityId *entities = chunk->entities().data();
            for (core::u32 i = 0; i < count; ++i)
            {
                _impl->ensureSleepCapacity(entities[i].slot());
            }

            // ── Pass 1: Integration ──────────────────────────────────────
            // Quantize the (deterministic, fixed) timestep once per chunk.
            const Fixed32 fdt = Fixed32::fromFloat(dt);
            integrateChunk(entities, positions, velocities, masses, count, fdt);

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

void CpuPhysicsBackend::integrateChunk(const ecs::EntityId *entities, FVec3 *positions, FVec3 *velocities,
                                       const Fixed32 *masses, core::u32 count, Fixed32 dt) const noexcept
{
    using math::simd::SimdFixed4;

    // Semi-implicit Euler in deterministic Fixed32. Gravity (per-lane mass
    // branch) and the ground bounce (per-lane branch) stay scalar; the damping
    // and position integration are vectorised with SimdFixed4 for blocks of 4
    // fully-awake entities. SimdFixed4 is bit-identical to the scalar path
    // (see test_simd_fixed_parity), so the fold is invariant to the width.
    const SimdFixed4 vDamping = SimdFixed4::splat(kVelocityDamping.raw());
    const SimdFixed4 vDt = SimdFixed4::splat(dt.raw());

    // Scalar update of a single entity (shared by the SIMD tail and mixed blocks).
    auto integrateScalar = [&](core::u32 k) {
        if (masses[k] > kMassEpsilon)
            velocities[k].y += kGravity * dt;
        velocities[k] = velocities[k] * kVelocityDamping;
        positions[k] = positions[k] + velocities[k] * dt;
        if (positions[k].y < kDefaultHalfHeight)
        {
            positions[k].y = kDefaultHalfHeight;
            if (velocities[k].y < Fixed32::zero())
                velocities[k].y = -velocities[k].y * kRestitution;
        }
    };

    core::u32 i = 0;
    for (; i + 3 < count; i += 4)
    {
        bool anySleeping = false;
        for (core::u32 j = 0; j < 4; ++j)
            anySleeping |= _impl->isSleeping(entities[i + j].slot());
        if (anySleeping)
        {
            for (core::u32 j = 0; j < 4; ++j)
                if (!_impl->isSleeping(entities[i + j].slot()))
                    integrateScalar(i + j);
            continue;
        }

        // Gravity first (per-lane mass branch), matching the scalar order.
        for (core::u32 j = 0; j < 4; ++j)
            if (masses[i + j] > kMassEpsilon)
                velocities[i + j].y += kGravity * dt;

        // Gather AoS Fixed32 → planar raw SoA.
        core::i32 px[4], py[4], pz[4], vx[4], vy[4], vz[4];
        for (core::u32 j = 0; j < 4; ++j)
        {
            px[j] = positions[i + j].x.raw();
            py[j] = positions[i + j].y.raw();
            pz[j] = positions[i + j].z.raw();
            vx[j] = velocities[i + j].x.raw();
            vy[j] = velocities[i + j].y.raw();
            vz[j] = velocities[i + j].z.raw();
        }

        // vel *= damping ; pos += vel * dt  (same op order as the scalar path).
        SimdFixed4 velX = SimdFixed4::load(vx) * vDamping;
        SimdFixed4 velY = SimdFixed4::load(vy) * vDamping;
        SimdFixed4 velZ = SimdFixed4::load(vz) * vDamping;
        SimdFixed4 posX = SimdFixed4::load(px) + velX * vDt;
        SimdFixed4 posY = SimdFixed4::load(py) + velY * vDt;
        SimdFixed4 posZ = SimdFixed4::load(pz) + velZ * vDt;
        velX.store(vx);
        velY.store(vy);
        velZ.store(vz);
        posX.store(px);
        posY.store(py);
        posZ.store(pz);

        // Scatter back + scalar ground bounce.
        for (core::u32 j = 0; j < 4; ++j)
        {
            velocities[i + j] = {Fixed32::fromRaw(vx[j]), Fixed32::fromRaw(vy[j]), Fixed32::fromRaw(vz[j])};
            positions[i + j] = {Fixed32::fromRaw(px[j]), Fixed32::fromRaw(py[j]), Fixed32::fromRaw(pz[j])};
            if (positions[i + j].y < kDefaultHalfHeight)
            {
                positions[i + j].y = kDefaultHalfHeight;
                if (velocities[i + j].y < Fixed32::zero())
                    velocities[i + j].y = -velocities[i + j].y * kRestitution;
            }
        }
    }

    // Scalar tail.
    for (; i < count; ++i)
    {
        if (_impl->isSleeping(entities[i].slot()))
            continue;
        integrateScalar(i);
    }
}

void CpuPhysicsBackend::resolveCollisionsChunk(const ecs::EntityId *entities, FVec3 *positions, FVec3 *velocities,
                                               const Fixed32 *masses, const FVec3 *sizes,
                                               core::u32 count) const noexcept
{
    const Fixed32 kZero = Fixed32::zero();
    const Fixed32 kOne = Fixed32::one();

    // Lambda for resolving a single collision pair (ported from legacy, Fixed32)
    auto resolveCollision = [&](core::u32 a, core::u32 b) {
        const core::u32 slotA = entities[a].slot();
        const core::u32 slotB = entities[b].slot();

        // Skip if both sleeping
        if (_impl->isSleeping(slotA) && _impl->isSleeping(slotB))
        {
            return;
        }

        FVec3 halfA = sizes[a] * kHalf;
        FVec3 halfB = sizes[b] * kHalf;
        FVec3 delta = positions[a] - positions[b];

        // Penetration on each axis
        Fixed32 overlapX = (halfA.x + halfB.x) - delta.x.abs();
        Fixed32 overlapY = (halfA.y + halfB.y) - delta.y.abs();
        Fixed32 overlapZ = (halfA.z + halfB.z) - delta.z.abs();

        if (overlapX <= kZero || overlapY <= kZero || overlapZ <= kZero)
        {
            return; // No collision
        }

        // Wake sleeping entities
        _impl->wake(slotA);
        _impl->wake(slotB);

        // Minimum penetration axis (SAT-like)
        FVec3 normal;
        Fixed32 penetration;

        if (overlapX <= overlapY && overlapX <= overlapZ)
        {
            penetration = overlapX;
            normal = {delta.x >= kZero ? kOne : -kOne, kZero, kZero};
        }
        else if (overlapY <= overlapX && overlapY <= overlapZ)
        {
            penetration = overlapY;
            normal = {kZero, delta.y >= kZero ? kOne : -kOne, kZero};
        }
        else
        {
            penetration = overlapZ;
            normal = {kZero, kZero, delta.z >= kZero ? kOne : -kOne};
        }

        // Inverse masses
        Fixed32 invMassA = (masses[a] > kMassEpsilon) ? (kOne / masses[a]) : kZero;
        Fixed32 invMassB = (masses[b] > kMassEpsilon) ? (kOne / masses[b]) : kZero;
        Fixed32 invMassSum = invMassA + invMassB;

        if (invMassSum < kMassEpsilon)
        {
            return; // Both infinite mass
        }

        // Positional correction (100%)
        Fixed32 correctionMag = penetration / invMassSum;
        FVec3 correction = normal * correctionMag;

        positions[a] = positions[a] + correction * invMassA;
        positions[b] = positions[b] - correction * invMassB;

        // Impulse (Newton's law of restitution)
        FVec3 relVel = velocities[a] - velocities[b];
        Fixed32 velAlongNormal = relVel.dot(normal);

        if (velAlongNormal > kZero)
        {
            return; // Separating
        }

        Fixed32 impulseMag = -(kOne + kRestitution) * velAlongNormal / invMassSum;
        FVec3 impulse = normal * impulseMag;

        velocities[a] = velocities[a] + impulse * invMassA;
        velocities[b] = velocities[b] - impulse * invMassB;

        if (impulseMag > kZero)
        {
            // CollisionEvent is a non-authoritative gameplay signal: float is fine.
            math::Vec3<float> fNormal{normal.x.toFloat(), normal.y.toFloat(), normal.z.toFloat()};
            core::EventBus::instance().publish(CollisionEvent{entities[a], entities[b], fNormal, impulseMag.toFloat()});
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
            // Build world-bounds from all entities (already Fixed32)
            FVec3 wMin{Fixed32::fromFloat(-500.0f), Fixed32::fromFloat(-500.0f), Fixed32::fromFloat(-500.0f)};
            FVec3 wMax{Fixed32::fromFloat(500.0f), Fixed32::fromFloat(500.0f), Fixed32::fromFloat(500.0f)};
            Octree octree(math::AABB<Fixed32>{wMin, wMax});

            // Insert all entities into the octree
            for (core::u32 i = 0; i < count; ++i)
            {
                FVec3 halfSz = sizes[i] * kHalf;
                math::AABB<Fixed32> aabb{
                    {positions[i].x - halfSz.x, positions[i].y - halfSz.y, positions[i].z - halfSz.z},
                    {positions[i].x + halfSz.x, positions[i].y + halfSz.y, positions[i].z + halfSz.z}
                };
                octree.insert(i, aabb);
            }
            octree.rebuild();

            // Query each entity's expanded AABB for candidate pairs
            for (core::u32 i = 0; i < count; ++i)
            {
                FVec3 halfSz = sizes[i] * kHalf;
                // Slightly expanded query region
                math::AABB<Fixed32> queryRegion{
                    {positions[i].x - halfSz.x, positions[i].y - halfSz.y, positions[i].z - halfSz.z},
                    {positions[i].x + halfSz.x, positions[i].y + halfSz.y, positions[i].z + halfSz.z}
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

void CpuPhysicsBackend::updateSleepingChunk(const ecs::EntityId *entities, FVec3 *velocities,
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
        const Fixed32 speedSq = velocities[i].lengthSquared();

        if (speedSq < kSleepVelocitySqThreshold)
        {
            _impl->sleepCounter[slot]++;
            if (_impl->sleepCounter[slot] >= kSleepFramesThreshold)
            {
                _impl->sleeping[slot] = true;
                velocities[i] = {Fixed32::zero(), Fixed32::zero(), Fixed32::zero()};
            }
        }
        else
        {
            _impl->sleepCounter[slot] = 0;
        }
    }
}

} // namespace lpl::physics
