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

/**
 * @brief Bodies a broad-phase leaf holds before it splits.
 *
 * Over 600 ticks of the cube pile, 16 executes 4.6 % fewer instructions than the octree's
 * default of 32, and both 8 and 24 execute more. A search reports its boxes in the sorted order
 * of their keys whatever the tree's shape, so this changes no result.
 */
static constexpr core::u32 kBroadPhaseLeafCapacity = 16;

/**
 * @brief How far a body may move before the neighbour lists stop covering it: 1/64.
 *
 * The lists hold, from where the bodies stood when they were made, the pairs whose boxes come
 * within twice this distance of each other. While no body has moved further than this on any axis,
 * every pair whose boxes meet is in those lists, for as many steps as that lasts. A step that finds
 * a body further makes them again, and a search for a body that moves further during a step goes
 * to the tree, as it did before there were lists.
 *
 * Over 600 ticks of the cube pile, 1/128 and 1/32 execute within 1 % of 1/64 and 1/256 5 % more,
 * while 1/16 lists so many pairs that it executes 17 % more: 1/64 keeps a factor of four from that
 * edge.
 */
static constexpr Fixed32 kNeighbourMargin = Fixed32::fromRaw(1024);

/** @brief Neighbours reserved per body: a packed pile gives a body thirteen after it, at most. */
static constexpr core::u32 kNeighboursReserved = 32;

/**
 * @brief Sorts @p bodies by the key each is filed under, then by number: the order a broad-phase
 *        search reports them in. An insertion sort, since a body has a handful of neighbours.
 */
static void sortAsFiled(lpl::pmr::vector<core::u32> &bodies, const lpl::pmr::vector<core::u32> &keys) noexcept
{
    for (core::u32 placed = 1u; placed < static_cast<core::u32>(bodies.size()); ++placed)
    {
        const core::u32 body = bodies[placed];
        const core::u32 key = keys[body];
        core::u32 slot = placed;
        while (slot != 0u &&
               (keys[bodies[slot - 1u]] > key || (keys[bodies[slot - 1u]] == key && bodies[slot - 1u] > body)))
        {
            bodies[slot] = bodies[slot - 1u];
            --slot;
        }
        bodies[slot] = body;
    }
}

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct CpuPhysicsBackend::Impl {
    /**
     * @brief One physics-relevant entity for a single step(): pointers into its
     *        chunk's SoA buffers for what the step moves, and the values the
     *        resolver derives from what it only reads.
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
        FVec3 half;          /**< Half its size: a step changes no size, so it is computed once a step. */
        Fixed32 inverseMass; /**< One over its mass, or zero for a body too heavy to move: once a step too. */
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
    lpl::pmr::vector<core::u32> candidates; /**< What one search found, refilled per search; its capacity stays. */

    /**
     * @name Neighbour lists, made when a step finds them stale: see kNeighbourMargin.
     * @{
     */
    lpl::pmr::vector<ecs::EntityId> listedIds;        /**< Which entity each listed body was. */
    lpl::pmr::vector<FVec3> listedHalves;             /**< Each body's half size when the lists were made. */
    lpl::pmr::vector<FVec3> listedPositions;          /**< Where each body stood when the lists were made. */
    lpl::pmr::vector<core::u32> neighbourStarts;      /**< Where each body's list starts; one more than the bodies. */
    lpl::pmr::vector<core::u32> neighbours;           /**< The bodies after each one that came within two margins. */
    lpl::pmr::vector<math::AABB<Fixed32>> filedBoxes; /**< Each body's box as the iteration began: what a tree holds. */
    lpl::pmr::vector<core::u32> filedKeys;            /**< The key each of those boxes is filed under. */
    /** @} */

    explicit Impl(ecs::Registry &r) : registry{r} {}

    /**
     * @brief The box body @p body covers now, from its position and its half size.
     */
    [[nodiscard]] math::AABB<Fixed32> boxOf(core::u32 body) const noexcept
    {
        const FVec3 &pos = *physicsEntities[body].position;
        const FVec3 &half = physicsEntities[body].half;
        return {
            {pos.x - half.x, pos.y - half.y, pos.z - half.z},
            {pos.x + half.x, pos.y + half.y, pos.z + half.z}
        };
    }

    /**
     * @brief Whether body @p body is still within kNeighbourMargin of where the lists found it.
     */
    [[nodiscard]] bool stillListed(core::u32 body) const noexcept
    {
        const FVec3 moved = *physicsEntities[body].position - listedPositions[body];
        return moved.x.abs() <= kNeighbourMargin && moved.y.abs() <= kNeighbourMargin &&
               moved.z.abs() <= kNeighbourMargin;
    }

    /**
     * @brief Whether the lists made at an earlier step still cover every body: the same entities in
     *        the same order, of the same sizes, none further than kNeighbourMargin from where it was.
     */
    [[nodiscard]] bool listsStillHold() const noexcept
    {
        const core::u32 count = static_cast<core::u32>(physicsEntities.size());
        if (static_cast<core::u32>(listedIds.size()) != count)
            return false;
        for (core::u32 body = 0u; body < count; ++body)
        {
            if (physicsEntities[body].id != listedIds[body] || physicsEntities[body].half != listedHalves[body] ||
                !stillListed(body))
                return false;
        }
        return true;
    }

    /**
     * @brief Files every body's box as it is now in @c filedBoxes, and returns the box they span.
     */
    math::AABB<Fixed32> fileBoxes()
    {
        const core::u32 count = static_cast<core::u32>(physicsEntities.size());
        filedBoxes.resize(count);
        filedBoxes[0] = boxOf(0);
        math::AABB<Fixed32> span = filedBoxes[0];
        for (core::u32 body = 1u; body < count; ++body)
        {
            filedBoxes[body] = boxOf(body);
            span = span.merge(filedBoxes[body]);
        }
        return span;
    }

    /**
     * @brief Builds the broad-phase over @c filedBoxes, bounded by @p span.
     */
    Octree &buildTree(const math::AABB<Fixed32> &span)
    {
        if (!broadPhase)
            broadPhase = lpl::pmr::make_unique<Octree>(span, kBroadPhaseLeafCapacity);
        broadPhase->clear();
        broadPhase->setWorldBounds(span);
        for (core::u32 body = 0u; body < static_cast<core::u32>(filedBoxes.size()); ++body)
            broadPhase->insert(body, filedBoxes[body]);
        broadPhase->rebuild();
        return *broadPhase;
    }

    /**
     * @brief Lists, for every body, the bodies after it whose box comes within two margins of its
     *        own, from where they stand now.
     */
    void listNeighbours()
    {
        const core::u32 count = static_cast<core::u32>(physicsEntities.size());
        Octree &octree = buildTree(fileBoxes());
        listedIds.resize(count);
        listedHalves.resize(count);
        listedPositions.resize(count);
        neighbourStarts.resize(count + 1u);
        neighbours.clear();
        neighbours.reserve(static_cast<std::size_t>(count) * kNeighboursReserved);
        for (core::u32 body = 0u; body < count; ++body)
        {
            listedIds[body] = physicsEntities[body].id;
            listedHalves[body] = physicsEntities[body].half;
            listedPositions[body] = *physicsEntities[body].position;
            neighbourStarts[body] = static_cast<core::u32>(neighbours.size());
            octree.collect(filedBoxes[body].expand(kNeighbourMargin + kNeighbourMargin), body + 1u, neighbours);
        }
        neighbourStarts[count] = static_cast<core::u32>(neighbours.size());
    }

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
            auto *aabbs = static_cast<const FVec3 *>(chunk->latestComponent(ecs::ComponentId::AABB));
            if (!positions || !velocities || !masses || !aabbs)
            {
                continue;
            }

            const ecs::EntityId *entities = chunk->entities().data();
            for (core::u32 i = 0; i < count; ++i)
            {
                _impl->physicsEntities.push_back(
                    Impl::PhysicsEntityRef{entities[i], &positions[i], &velocities[i], aabbs[i] * kHalf,
                                           masses[i] > kMassEpsilon ? Fixed32::one() / masses[i] : Fixed32::zero()});
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
    bool anyOverlap = false;

    // Lambda for resolving a single collision pair (ported from legacy, Fixed32)
    auto resolveCollision = [&](core::u32 a, core::u32 b) {
        FVec3 *positions_a = refs[a].position;
        FVec3 *positions_b = refs[b].position;
        FVec3 *velocities_a = refs[a].velocity;
        FVec3 *velocities_b = refs[b].velocity;

        const core::u32 slotA = refs[a].id.slot();
        const core::u32 slotB = refs[b].id.slot();

        // Skip if both sleeping
        if (_impl->isSleeping(slotA) && _impl->isSleeping(slotB))
        {
            return;
        }

        const FVec3 &halfA = refs[a].half;
        const FVec3 &halfB = refs[b].half;
        FVec3 delta = *positions_a - *positions_b;

        // Penetration on each axis
        Fixed32 overlapX = (halfA.x + halfB.x) - delta.x.abs();
        Fixed32 overlapY = (halfA.y + halfB.y) - delta.y.abs();
        Fixed32 overlapZ = (halfA.z + halfB.z) - delta.z.abs();

        if (overlapX <= kZero || overlapY <= kZero || overlapZ <= kZero)
        {
            return; // No collision
        }
        anyOverlap = true;

        // Wake sleeping entities
        _impl->wake(slotA);
        _impl->wake(slotB);

        // Minimum penetration axis (SAT-like)
        Fixed32 FVec3::*axis;
        Fixed32 penetration;

        if (overlapX <= overlapY && overlapX <= overlapZ)
        {
            penetration = overlapX;
            axis = &FVec3::x;
        }
        else if (overlapY <= overlapX && overlapY <= overlapZ)
        {
            penetration = overlapY;
            axis = &FVec3::y;
        }
        else
        {
            penetration = overlapZ;
            axis = &FVec3::z;
        }
        const Fixed32 direction = delta.*axis >= kZero ? kOne : -kOne;

        // Inverse masses
        const Fixed32 invMassA = refs[a].inverseMass;
        const Fixed32 invMassB = refs[b].inverseMass;
        Fixed32 invMassSum = invMassA + invMassB;

        if (invMassSum < kMassEpsilon)
        {
            return; // Both infinite mass
        }

        // Positional correction (100%)
        Fixed32 correctionMag = penetration / invMassSum;
        const Fixed32 correction = direction * correctionMag;

        (*positions_a).*axis = (*positions_a).*axis + correction * invMassA;
        (*positions_b).*axis = (*positions_b).*axis - correction * invMassB;

        // Impulse (Newton's law of restitution)
        Fixed32 velAlongNormal = ((*velocities_a).*axis - (*velocities_b).*axis) * direction;

        if (velAlongNormal > kZero)
        {
            return; // Separating
        }

        Fixed32 impulseMag = -(kOne + kRestitution) * velAlongNormal / invMassSum;
        const Fixed32 impulse = direction * impulseMag;

        (*velocities_a).*axis = (*velocities_a).*axis + impulse * invMassA;
        (*velocities_b).*axis = (*velocities_b).*axis - impulse * invMassB;

        if (impulseMag > kZero)
        {
            // CollisionEvent is a non-authoritative gameplay signal: float is fine.
            FVec3 normal{kZero, kZero, kZero};
            normal.*axis = direction;
            math::Vec3<float> fNormal{normal.x.toFloat(), normal.y.toFloat(), normal.z.toFloat()};
            core::EventBus::instance().publish(CollisionEvent{refs[a].id, refs[b].id, fNormal, impulseMag.toFloat()});
        }
    };

    if (count > kBruteForceThreshold && !_impl->listsStillHold())
    {
        _impl->listNeighbours();
    }

    // 4 solver iterations (legacy uses SOLVER_ITERATIONS = 4)
    for (core::u32 iter = 0; iter < kSolverIterations; ++iter)
    {
        anyOverlap = false;
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
            const math::AABB<Fixed32> span = _impl->fileBoxes();
            bool listsHold = true;
            for (core::u32 j = 0; j < count && listsHold; ++j)
                listsHold = _impl->stillListed(j);

            Octree &octree = *_impl->broadPhase;
            octree.setWorldBounds(span);
            _impl->filedKeys.resize(count);
            for (core::u32 j = 0; j < count; ++j)
                _impl->filedKeys[j] = octree.keyOf(_impl->filedBoxes[j]);
            _impl->candidates.reserve(count);
            bool treeBuilt = false;

            for (core::u32 i = 0; i < count; ++i)
            {
                const math::AABB<Fixed32> queryRegion = _impl->boxOf(i);

                _impl->candidates.clear();
                if (listsHold && _impl->stillListed(i))
                {
                    for (core::u32 k = _impl->neighbourStarts[i]; k < _impl->neighbourStarts[i + 1u]; ++k)
                    {
                        const core::u32 j = _impl->neighbours[k];
                        if (_impl->filedBoxes[j].intersects(queryRegion))
                            _impl->candidates.push_back(j);
                    }
                    sortAsFiled(_impl->candidates, _impl->filedKeys);
                }
                else
                {
                    if (!treeBuilt)
                    {
                        _impl->buildTree(span);
                        treeBuilt = true;
                    }
                    octree.collect(queryRegion, i + 1u, _impl->candidates);
                }
                for (const core::u32 j : _impl->candidates)
                {
                    resolveCollision(i, j);
                }
            }
        }
        if (!anyOverlap)
        {
            break;
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
