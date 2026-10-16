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

using FVec3 = math::Vec3<math::Fixed32>;

static constexpr math::Fixed32 kGravity = math::Fixed32::fromFloat(-9.81f);
static constexpr math::Fixed32 kVelocityDamping = math::Fixed32::fromFloat(0.995f);
static constexpr math::Fixed32 kRestitution = math::Fixed32::fromFloat(0.5f);
static constexpr math::Fixed32 kSleepVelocitySqThreshold = math::Fixed32::fromFloat(0.01f);
static constexpr math::Fixed32 kMassEpsilon = math::Fixed32::fromFloat(0.0001f);
static constexpr math::Fixed32 kHalf = math::Fixed32::fromFloat(0.5f);
static constexpr math::Fixed32 kDefaultHalfHeight = math::Fixed32::fromFloat(0.5f);
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
static constexpr math::Fixed32 kNeighbourMargin = math::Fixed32::fromRaw(1024);

/** @brief Neighbours reserved per body: a packed pile gives a body thirteen after it, at most. */
static constexpr core::u32 kNeighboursReserved = 32;

/**
 * @brief Sorts @p bodies by their keys, @p keys holding the key of each in the same order, then by
 *        number: the order a broad-phase search reports them in. An insertion sort, since a body has
 *        a handful of neighbours.
 */
static void sortAsFiled(lpl::pmr::vector<core::u32> &bodies, lpl::pmr::vector<core::u32> &keys) noexcept
{
    for (core::u32 placed = 1u; placed < static_cast<core::u32>(bodies.size()); ++placed)
    {
        const core::u32 body = bodies[placed];
        const core::u32 key = keys[placed];
        core::u32 slot = placed;
        while (slot != 0u && (keys[slot - 1u] > key || (keys[slot - 1u] == key && bodies[slot - 1u] > body)))
        {
            bodies[slot] = bodies[slot - 1u];
            keys[slot] = keys[slot - 1u];
            --slot;
        }
        bodies[slot] = body;
        keys[slot] = key;
    }
}

/**
 * @brief What the backend keeps from one step to the next.
 *
 * The broad-phase tree is made once and refilled in place each step: a tree made per step re-grew
 * every buffer it owns at every tick, the largest source of heap traffic in the authoritative step.
 * One tree serves the whole step, which walks the chunks one after the other: chunks resolved in
 * parallel would need one tree per worker.
 */
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
        FVec3 half;                /**< Half its size: a step changes no size, so it is computed once a step. */
        math::Fixed32 inverseMass; /**< One over its mass, or zero for a body too heavy to move: once a step too. */
    };

    ecs::Registry &registry;

    lpl::pmr::vector<bool> sleeping;                    /**< Per entity slot, not per chunk: whether it sleeps. */
    lpl::pmr::vector<core::u16> sleepCounter;           /**< Per slot: the steps in a row under the sleep threshold. */
    lpl::pmr::vector<PhysicsEntityRef> physicsEntities; /**< Every body of the world, gathered again each step. */
    lpl::pmr::unique_ptr<Octree> broadPhase;            /**< Made once, refilled in place each step. */
    lpl::pmr::vector<core::u32> candidates; /**< What one search found, refilled per search; its capacity stays. */
    lpl::pmr::vector<math::AABB<math::Fixed32>> candidateBoxes; /**< The filed boxes of candidates to sort. */
    lpl::pmr::vector<core::u32> candidateKeys;                  /**< The key each of those boxes is filed under. */

    /**
     * @name Neighbour lists, made when a step finds them stale: see kNeighbourMargin.
     * @{
     */
    lpl::pmr::vector<ecs::EntityId> listedIds;   /**< Which entity each listed body was. */
    lpl::pmr::vector<FVec3> listedHalves;        /**< Each body's half size when the lists were made. */
    lpl::pmr::vector<FVec3> listedPositions;     /**< Where each body stood when the lists were made. */
    lpl::pmr::vector<core::u32> neighbourStarts; /**< Where each body's list starts; one more than the bodies. */
    lpl::pmr::vector<core::u32> neighbours;      /**< The bodies after each one that came within two margins. */
    lpl::pmr::vector<math::AABB<math::Fixed32>> filedBoxes; /**< The boxes as the iteration began: what a tree holds. */
    /** @} */

    explicit Impl(ecs::Registry &r) : registry{r} {}

    /**
     * @brief The box body @p body covers now, from its position and its half size.
     */
    [[nodiscard]] math::AABB<math::Fixed32> boxOf(core::u32 body) const noexcept
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
    math::AABB<math::Fixed32> fileBoxes()
    {
        const core::u32 count = static_cast<core::u32>(physicsEntities.size());
        filedBoxes.resize(count);
        filedBoxes[0] = boxOf(0);
        math::AABB<math::Fixed32> span = filedBoxes[0];
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
    Octree &buildTree(const math::AABB<math::Fixed32> &span)
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
     * @brief Puts @c candidates in the order a broad-phase search reports them, keying their filed
     *        boxes alone over the bounds @p octree was last given.
     *
     * Only a list of two or more needs keys, and a body's list seldom holds two bodies its box
     * meets: over 600 ticks of the cube pile at 144 Hz, 3974 of 2.09 million did, and none held
     * three. Keying every filed box at each iteration cost 128 million instructions for those.
     */
    void sortCandidatesAsFiled(const Octree &octree)
    {
        candidateBoxes.clear();
        for (const core::u32 body : candidates)
            candidateBoxes.push_back(filedBoxes[body]);
        octree.keysOf(candidateBoxes, candidateKeys);
        sortAsFiled(candidates, candidateKeys);
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
    const math::Fixed32 fdt = math::Fixed32::fromFloat(dt);

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
            auto *masses = static_cast<const math::Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
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
            auto *masses = static_cast<const math::Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
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
                                           masses[i] > kMassEpsilon ? masses[i].reciprocal() : math::Fixed32::zero()});
            }
        }
    }
    resolveCollisionsWorld();

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

void CpuPhysicsBackend::integrateChunk(const ecs::EntityId *entities, FVec3 *positions, FVec3 *velocities,
                                       const math::Fixed32 *masses, core::u32 count, math::Fixed32 dt) const noexcept
{
    const math::simd::SimdFixed4 vDamping = math::simd::SimdFixed4::splat(kVelocityDamping.raw());
    const math::simd::SimdFixed4 vDt = math::simd::SimdFixed4::splat(dt.raw());

    auto integrateScalar = [&](core::u32 k) {
        if (masses[k] > kMassEpsilon)
            velocities[k].y += kGravity * dt;
        velocities[k] = velocities[k] * kVelocityDamping;
        positions[k] = positions[k] + velocities[k] * dt;
        if (positions[k].y < kDefaultHalfHeight)
        {
            positions[k].y = kDefaultHalfHeight;
            if (velocities[k].y < math::Fixed32::zero())
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

        for (core::u32 j = 0; j < 4; ++j)
            if (masses[i + j] > kMassEpsilon)
                velocities[i + j].y += kGravity * dt;

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

        math::simd::SimdFixed4 velX = math::simd::SimdFixed4::load(vx) * vDamping;
        math::simd::SimdFixed4 velY = math::simd::SimdFixed4::load(vy) * vDamping;
        math::simd::SimdFixed4 velZ = math::simd::SimdFixed4::load(vz) * vDamping;
        math::simd::SimdFixed4 posX = math::simd::SimdFixed4::load(px) + velX * vDt;
        math::simd::SimdFixed4 posY = math::simd::SimdFixed4::load(py) + velY * vDt;
        math::simd::SimdFixed4 posZ = math::simd::SimdFixed4::load(pz) + velZ * vDt;
        velX.store(vx);
        velY.store(vy);
        velZ.store(vz);
        posX.store(px);
        posY.store(py);
        posZ.store(pz);

        for (core::u32 j = 0; j < 4; ++j)
        {
            velocities[i + j] = {math::Fixed32::fromRaw(vx[j]), math::Fixed32::fromRaw(vy[j]),
                                 math::Fixed32::fromRaw(vz[j])};
            positions[i + j] = {math::Fixed32::fromRaw(px[j]), math::Fixed32::fromRaw(py[j]),
                                math::Fixed32::fromRaw(pz[j])};
            if (positions[i + j].y < kDefaultHalfHeight)
            {
                positions[i + j].y = kDefaultHalfHeight;
                if (velocities[i + j].y < math::Fixed32::zero())
                    velocities[i + j].y = -velocities[i + j].y * kRestitution;
            }
        }
    }

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

    const math::Fixed32 kZero = math::Fixed32::zero();
    const math::Fixed32 kOne = math::Fixed32::one();
    bool anyOverlap = false;

    auto resolveCollision = [&](core::u32 a, core::u32 b) {
        FVec3 *positions_a = refs[a].position;
        FVec3 *positions_b = refs[b].position;
        FVec3 *velocities_a = refs[a].velocity;
        FVec3 *velocities_b = refs[b].velocity;

        const core::u32 slotA = refs[a].id.slot();
        const core::u32 slotB = refs[b].id.slot();

        if (_impl->isSleeping(slotA) && _impl->isSleeping(slotB))
        {
            return;
        }

        const FVec3 &halfA = refs[a].half;
        const FVec3 &halfB = refs[b].half;
        FVec3 delta = *positions_a - *positions_b;

        math::Fixed32 overlapX = (halfA.x + halfB.x) - delta.x.abs();
        math::Fixed32 overlapY = (halfA.y + halfB.y) - delta.y.abs();
        math::Fixed32 overlapZ = (halfA.z + halfB.z) - delta.z.abs();

        if (overlapX <= kZero || overlapY <= kZero || overlapZ <= kZero)
        {
            return;
        }
        anyOverlap = true;

        _impl->wake(slotA);
        _impl->wake(slotB);

        math::Fixed32 FVec3::*axis;
        math::Fixed32 penetration;

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
        const math::Fixed32 direction = delta.*axis >= kZero ? kOne : -kOne;

        const math::Fixed32 invMassA = refs[a].inverseMass;
        const math::Fixed32 invMassB = refs[b].inverseMass;
        math::Fixed32 invMassSum = invMassA + invMassB;

        if (invMassSum < kMassEpsilon)
        {
            return;
        }

        math::Fixed32 correctionMag = penetration / invMassSum;
        const math::Fixed32 correction = direction * correctionMag;

        (*positions_a).*axis = (*positions_a).*axis + correction * invMassA;
        (*positions_b).*axis = (*positions_b).*axis - correction * invMassB;

        math::Fixed32 velAlongNormal = ((*velocities_a).*axis - (*velocities_b).*axis) * direction;

        if (velAlongNormal > kZero)
        {
            return;
        }

        math::Fixed32 impulseMag = -(kOne + kRestitution) * velAlongNormal / invMassSum;
        const math::Fixed32 impulse = direction * impulseMag;

        (*velocities_a).*axis = (*velocities_a).*axis + impulse * invMassA;
        (*velocities_b).*axis = (*velocities_b).*axis - impulse * invMassB;

        if (impulseMag > kZero)
        {
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

    for (core::u32 iter = 0; iter < kSolverIterations; ++iter)
    {
        anyOverlap = false;
        if (count <= kBruteForceThreshold)
        {
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
            const math::AABB<math::Fixed32> span = _impl->fileBoxes();
            bool listsHold = true;
            for (core::u32 j = 0; j < count && listsHold; ++j)
                listsHold = _impl->stillListed(j);

            Octree &octree = *_impl->broadPhase;
            octree.setWorldBounds(span);
            _impl->candidates.reserve(count);
            _impl->candidateBoxes.reserve(count);
            _impl->candidateKeys.reserve(count);
            bool treeBuilt = false;

            for (core::u32 i = 0; i < count; ++i)
            {
                const math::AABB<math::Fixed32> queryRegion = _impl->boxOf(i);

                _impl->candidates.clear();
                if (listsHold && _impl->stillListed(i))
                {
                    for (core::u32 k = _impl->neighbourStarts[i]; k < _impl->neighbourStarts[i + 1u]; ++k)
                    {
                        const core::u32 j = _impl->neighbours[k];
                        if (_impl->filedBoxes[j].intersects(queryRegion))
                            _impl->candidates.push_back(j);
                    }
                    if (_impl->candidates.size() > 1u)
                        _impl->sortCandidatesAsFiled(octree);
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

        const math::Fixed32 speedSq = velocities[i].lengthSquared();

        if (speedSq < kSleepVelocitySqThreshold)
        {
            _impl->sleepCounter[slot]++;
            if (_impl->sleepCounter[slot] >= kSleepFramesThreshold)
            {
                _impl->sleeping[slot] = true;
                velocities[i] = {math::Fixed32::zero(), math::Fixed32::zero(), math::Fixed32::zero()};
            }
        }
        else
        {
            _impl->sleepCounter[slot] = 0;
        }
    }
}

} // namespace lpl::physics
