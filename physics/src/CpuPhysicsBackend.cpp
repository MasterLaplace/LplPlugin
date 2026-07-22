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
    /**
     * @brief One physics-relevant entity's pointers into its chunk's SoA
     *        buffers, valid for the duration of a single step().
     *
     * Chunks are grouped by archetype and capped at @c ecs::Chunk::kChunkCapacity
     * (256), not by spatial region: two entities can be in different chunks
     * (different archetype, or the same archetype overflowed into a second
     * chunk) and still occupy the same world position. Collision must see every
     * entity together, so this flat, world-wide index is what the broad-phase
     * and the resolver iterate — never a single chunk's arrays in isolation.
     */
    struct PhysicsEntityRef {
        ecs::EntityId id;
        FVec3 *position;
        FVec3 *velocity;
        const Fixed32 *mass;
        FVec3 *size;
    };

    ecs::Registry &registry;

    /** @brief Per-entity sleeping data (indexed by entity slot, not chunk-local). */
    lpl::pmr::vector<bool> sleeping;
    lpl::pmr::vector<core::u16> sleepCounter;

    /**
     * @brief World-wide flat index of physics entities, rebuilt (not
     *        reallocated on a warm step) at the top of Pass 2 every step.
     */
    lpl::pmr::vector<PhysicsEntityRef> physicsEntities;

    /**
     * @brief Broad-phase index, kept across steps and refilled in place.
     *
     * Built once, then cleared and refilled every step from @c physicsEntities.
     * A per-step local Octree instead re-grew every buffer it owns on every
     * tick — the single largest source of heap traffic in the authoritative
     * step.
     *
     * Shared across the whole step, like @c sleeping and @c sleepCounter: step()
     * walks partitions and chunks sequentially. Processing chunks in parallel
     * would need one index per worker, not one per backend.
     */
    lpl::pmr::unique_ptr<Octree> broadPhase;

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
    const Fixed32 fdt = Fixed32::fromFloat(dt);

    // ── Pass 1: Integration ─────────────────────────────────────────────
    // Purely per-entity (gravity, damping, ground bounce): no cross-chunk
    // data needed, so this stays scoped to each chunk's own arrays.
    for (const auto &partition : partitions)
    {
        const auto &archetype = partition->archetype();
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

            auto *positions = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *velocities = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            auto *masses = static_cast<const Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
            if (!positions || !velocities || !masses)
            {
                continue;
            }

            const ecs::EntityId *entities = chunk->entities().data();
            for (core::u32 i = 0; i < count; ++i)
            {
                _impl->ensureSleepCapacity(entities[i].slot());
            }

            integrateChunk(entities, positions, velocities, masses, count, fdt);
        }
    }

    // ── Pass 2: Collision detection + resolution ────────────────────────
    // Gathered across every chunk of every physics-relevant partition first:
    // chunks are grouped by archetype and capped at 256 entities, not by
    // spatial region, so two entities in different chunks can still occupy
    // the same world position and must be resolved against each other.
    _impl->physicsEntities.clear();
    for (const auto &partition : partitions)
    {
        const auto &archetype = partition->archetype();
        if (!archetype.has(ecs::ComponentId::Position) || !archetype.has(ecs::ComponentId::Velocity) ||
            !archetype.has(ecs::ComponentId::Mass) || !archetype.has(ecs::ComponentId::AABB))
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

            auto *positions = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *velocities = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            auto *masses = static_cast<const Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
            auto *aabbs = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::AABB));
            if (!positions || !velocities || !masses || !aabbs)
            {
                continue;
            }

            const ecs::EntityId *entities = chunk->entities().data();
            for (core::u32 i = 0; i < count; ++i)
            {
                _impl->physicsEntities.push_back(Impl::PhysicsEntityRef{
                    entities[i], &positions[i], &velocities[i], &masses[i], &aabbs[i]});
            }
        }
    }
    resolveCollisionsWorld();

    // ── Pass 3: Sleeping detection ───────────────────────────────────────
    // Position-independent, so it stays scoped to the chunk it walks.
    for (const auto &partition : partitions)
    {
        const auto &archetype = partition->archetype();
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

            auto *velocities = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            if (!velocities)
            {
                continue;
            }

            const ecs::EntityId *entities = chunk->entities().data();
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

void CpuPhysicsBackend::resolveCollisionsWorld() const noexcept
{
    auto &refs = _impl->physicsEntities;
    const core::u32 count = static_cast<core::u32>(refs.size());
    if (count == 0)
    {
        return;
    }

    const Fixed32 kZero = Fixed32::zero();
    const Fixed32 kOne = Fixed32::one();

    // Lambda for resolving a single collision pair (ported from legacy, Fixed32)
    auto resolveCollision = [&](core::u32 a, core::u32 b) {
        FVec3 *positions_a = refs[a].position;
        FVec3 *positions_b = refs[b].position;
        FVec3 *velocities_a = refs[a].velocity;
        FVec3 *velocities_b = refs[b].velocity;
        const Fixed32 &mass_a = *refs[a].mass;
        const Fixed32 &mass_b = *refs[b].mass;
        const FVec3 &size_a = *refs[a].size;
        const FVec3 &size_b = *refs[b].size;

        const core::u32 slotA = refs[a].id.slot();
        const core::u32 slotB = refs[b].id.slot();

        // Skip if both sleeping
        if (_impl->isSleeping(slotA) && _impl->isSleeping(slotB))
        {
            return;
        }

        FVec3 halfA = size_a * kHalf;
        FVec3 halfB = size_b * kHalf;
        FVec3 delta = *positions_a - *positions_b;

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
        Fixed32 invMassA = (mass_a > kMassEpsilon) ? (kOne / mass_a) : kZero;
        Fixed32 invMassB = (mass_b > kMassEpsilon) ? (kOne / mass_b) : kZero;
        Fixed32 invMassSum = invMassA + invMassB;

        if (invMassSum < kMassEpsilon)
        {
            return; // Both infinite mass
        }

        // Positional correction (100%)
        Fixed32 correctionMag = penetration / invMassSum;
        FVec3 correction = normal * correctionMag;

        *positions_a = *positions_a + correction * invMassA;
        *positions_b = *positions_b - correction * invMassB;

        // Impulse (Newton's law of restitution)
        FVec3 relVel = *velocities_a - *velocities_b;
        Fixed32 velAlongNormal = relVel.dot(normal);

        if (velAlongNormal > kZero)
        {
            return; // Separating
        }

        Fixed32 impulseMag = -(kOne + kRestitution) * velAlongNormal / invMassSum;
        FVec3 impulse = normal * impulseMag;

        *velocities_a = *velocities_a + impulse * invMassA;
        *velocities_b = *velocities_b - impulse * invMassB;

        if (impulseMag > kZero)
        {
            // CollisionEvent is a non-authoritative gameplay signal: float is fine.
            math::Vec3<float> fNormal{normal.x.toFloat(), normal.y.toFloat(), normal.z.toFloat()};
            core::EventBus::instance().publish(CollisionEvent{refs[a].id, refs[b].id, fNormal, impulseMag.toFloat()});
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
            // World bounds are fixed, so the index is built once and thereafter
            // only cleared and refilled — no allocation on a warm step.
            if (!_impl->broadPhase)
                _impl->broadPhase = lpl::pmr::make_unique<Octree>(math::AABB<Fixed32>{wMin, wMax});
            Octree &octree = *_impl->broadPhase;
            octree.clear();

            // Insert all entities into the octree
            for (core::u32 i = 0; i < count; ++i)
            {
                const FVec3 &pos = *refs[i].position;
                FVec3 halfSz = *refs[i].size * kHalf;
                math::AABB<Fixed32> aabb{
                    {pos.x - halfSz.x, pos.y - halfSz.y, pos.z - halfSz.z},
                    {pos.x + halfSz.x, pos.y + halfSz.y, pos.z + halfSz.z}
                };
                octree.insert(i, aabb);
            }
            octree.rebuild();

            // Query each entity's expanded AABB for candidate pairs
            for (core::u32 i = 0; i < count; ++i)
            {
                const FVec3 &pos = *refs[i].position;
                FVec3 halfSz = *refs[i].size * kHalf;
                // Slightly expanded query region
                math::AABB<Fixed32> queryRegion{
                    {pos.x - halfSz.x, pos.y - halfSz.y, pos.z - halfSz.z},
                    {pos.x + halfSz.x, pos.y + halfSz.y, pos.z + halfSz.z}
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
