#include <lpl/core/Assert.hpp>
#include <lpl/core/EventBus.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/core/Platform.hpp>
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

/** @brief Passes over a step's contacts that set the velocities, before the bodies move. */
static constexpr core::u32 kVelocityIterations = 4;

/** @brief Passes over a step's contacts that remove the overlap left once the bodies have moved. */
static constexpr core::u32 kPositionIterations = 4;

/**
 * @brief The approach speed above which a collision bounces: one unit a second.
 *
 * Below it a collision is inelastic, as in Box2D (`b2WorldDef::restitutionThreshold`). Gravity adds
 * a sixth of a unit a second at each step of 1/60 s, so with restitution at every speed a body at rest
 * bounced on its support at every step and never fell below the sleep threshold.
 */
static constexpr math::Fixed32 kRestitutionThreshold = math::Fixed32::one();

/**
 * @brief Bodies a broad-phase leaf holds before it splits.
 *
 * Over 600 ticks of the cube pile, with the solver that searched the tree at every iteration, 16
 * executed 4.6 % fewer instructions than the octree's default of 32, and both 8 and 24 more. A search reports its boxes
 * in the sorted order of their keys whatever the tree's shape, so this changes no result.
 */
static constexpr core::u32 kBroadPhaseLeafCapacity = 16;

/**
 * @brief How far a body may move before the neighbour lists stop covering it: 1/64.
 *
 * The lists hold, from where the bodies stood when they were made, the pairs whose boxes come
 * within kListReach of each other. While no body has moved further than this on any axis, every pair
 * whose boxes come within kSpeculativeDistance is in those lists, for as many steps as that lasts. A
 * step that finds a body further makes them again before it looks for its contacts.
 *
 * Over 600 ticks of the cube pile, with the solver that searched again at every iteration, 1/128
 * and 1/32 executed within 1 % of 1/64 and 1/256 5 % more, while 1/16 listed so many pairs that it
 * executed 17 % more: 1/64 kept a factor of four from that edge. It is also the speculative
 * distance now, so changing it changes the results.
 */
static constexpr math::Fixed32 kNeighbourMargin = math::Fixed32::fromRaw(1024);

/**
 * @brief How far apart two boxes may be and still form a contact: 1/64.
 *
 * A contact across a gap is speculative: it lets the pair close the gap within the step and no more,
 * so a slow body comes to rest on its support instead of sinking into it and being pushed out. A
 * body that touches its support exactly, as the position correction leaves it, keeps its contact
 * from one step to the next. At 60 steps a second the gap lets a pair close at up to 0.94 units a
 * second, under kRestitutionThreshold: a contact made across a gap never has to bounce.
 */
static constexpr math::Fixed32 kSpeculativeDistance = kNeighbourMargin;

/** @brief How far the neighbour lists reach around a box: two margins, and the speculative distance. */
static constexpr math::Fixed32 kListReach = kNeighbourMargin + kNeighbourMargin + kSpeculativeDistance;

/** @brief Neighbours reserved per body: a packed pile gives a body thirteen after it, at most. */
static constexpr core::u32 kNeighboursReserved = 32;

/** @brief Cells per body past which the neighbour lists are made through the tree, not a grid. */
static constexpr core::u32 kMostGridCellsPerBody = 8;

/**
 * @brief Sorts the bodies of @p bodies from @p first to @p last, the last excluded, in increasing
 *        order: an insertion sort, since a body has a handful of neighbours.
 */
static void sortAscending(lpl::pmr::vector<core::u32> &bodies, core::u32 first, core::u32 last) noexcept
{
    for (core::u32 placed = first + 1u; placed < last; ++placed)
    {
        const core::u32 body = bodies[placed];
        core::u32 slot = placed;
        while (slot != first && bodies[slot - 1u] > body)
        {
            bodies[slot] = bodies[slot - 1u];
            --slot;
        }
        bodies[slot] = body;
    }
}

/**
 * @brief The speed a body that meets a support at @p approach, a negative speed along the normal,
 *        leaves it with: half of it back when it came faster than kRestitutionThreshold, and none
 *        otherwise.
 */
[[nodiscard]] static math::Fixed32 bounceOf(math::Fixed32 approach) noexcept
{
    return -approach > kRestitutionThreshold ? -approach * kRestitution : math::Fixed32::zero();
}

/**
 * @brief The normal speed a contact aims at: across a @p gap, the speed that closes it within the
 *        step, and touching or overlapping, the bounce of its @p approach.
 *
 * @param gap       The distance between the two boxes along the normal: negative when they overlap.
 * @param approach  Their speed along the normal as the step began: negative when they close.
 * @param inverseDt One over the step, in seconds.
 */
[[nodiscard]] static math::Fixed32 targetSpeed(math::Fixed32 gap, math::Fixed32 approach,
                                               math::Fixed32 inverseDt) noexcept
{
    if (gap > math::Fixed32::zero())
        return -(gap * inverseDt);
    return bounceOf(approach);
}

/**
 * @brief @p value along a normal that points up its axis when @p positive, and down it otherwise.
 */
[[nodiscard]] static math::Fixed32 alongNormal(math::Fixed32 value, bool positive) noexcept
{
    return positive ? value : -value;
}

/**
 * @brief Publishes the contact of @p a and @p b: its normal, @p direction along @p axis, and the
 *        magnitude @p impulse of its impulse.
 *
 * Out of line, so that the resolver holds no float. A CollisionEvent is a gameplay signal and not
 * authoritative state, and building its floats inside the resolver made every pair pay for them.
 * The resolver calls it only when something is subscribed to the bus, which the server never is.
 */
LPL_NOINLINE static void publishContact(ecs::EntityId a, ecs::EntityId b, math::Fixed32 FVec3::*axis,
                                        math::Fixed32 direction, math::Fixed32 impulse) noexcept
{
    FVec3 normal{math::Fixed32::zero(), math::Fixed32::zero(), math::Fixed32::zero()};
    normal.*axis = direction;
    const math::Vec3<float> floatNormal{normal.x.toFloat(), normal.y.toFloat(), normal.z.toFloat()};
    core::EventBus::instance().publish(CollisionEvent{a, b, floatNormal, impulse.toFloat()});
}

/**
 * @brief What the backend keeps from one step to the next.
 *
 * The broad-phase tree is made once, at the first listing, and refilled in place whenever a listing
 * searches it: a tree made per step re-grew every buffer it owns at every tick, the largest source
 * of heap traffic in the authoritative step. The contacts keep their capacity in the same way.
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
        math::Fixed32 mass;        /**< Its mass. */
        math::Fixed32 inverseMass; /**< One over its mass, or zero for a body too heavy to move: once a step too. */
    };

    /**
     * @brief One contact of a step: two bodies, or a body and the ground, and the axis they meet on.
     *
     * The contacts are found once a step and solved by sequential impulses (Catto, GDC 2006): each
     * velocity iteration sets the impulse a contact needs given what the others did, and what a
     * contact keeps is the sum of its impulses, held at zero or more. A later iteration can then take
     * back part of an earlier impulse, which holding each impulse at zero or more would forbid.
     */
    struct Contact {
        core::u32 a;                /**< The body the normal points toward. */
        core::u32 b;                /**< The other body, or kGround. */
        ecs::EntityId first;        /**< a's entity, so that the next step knows the contact again. */
        ecs::EntityId second;       /**< b's entity, or a's own for the ground. */
        math::Fixed32 FVec3::*axis; /**< The axis they meet on. */
        bool positive;              /**< Whether the normal, from b to a, points up that axis. */
        math::Fixed32 inverseMassA; /**< a's inverse mass in this contact: zero while it sleeps. */
        math::Fixed32 inverseMassB; /**< b's, zero for the ground or a sleeper. */
        math::Fixed32 mass;         /**< One over the sum of the two: the pair's mass along the normal. */
        math::Fixed32 target;       /**< The normal speed the iterations aim at: a bounce, zero, or a gap closed. */
        math::Fixed32 impulse;      /**< The impulse summed over the step's iterations, never negative. */
    };

    /** @brief What Contact::b holds for a contact with the ground. */
    static constexpr core::u32 kGround = 0xFFFFFFFFu;

    ecs::Registry &registry;

    lpl::pmr::vector<bool> sleeping;                    /**< Per entity slot, not per chunk: whether it sleeps. */
    lpl::pmr::vector<core::u16> sleepCounter;           /**< Per slot: the steps in a row under the sleep threshold. */
    lpl::pmr::vector<PhysicsEntityRef> physicsEntities; /**< Every body of the world, gathered again each step. */
    lpl::pmr::unique_ptr<Octree> broadPhase;            /**< Made once, refilled in place when a listing needs it. */
    lpl::pmr::vector<Contact> contacts;                 /**< The step's contacts, the ground's first. */
    lpl::pmr::vector<Contact> previousContacts;         /**< The last step's, whose impulses warm the new ones. */

    /**
     * @name Neighbour lists, made when a step finds them stale: see kNeighbourMargin.
     * @{
     */
    lpl::pmr::vector<ecs::EntityId> listedIds;   /**< Which entity each listed body was. */
    lpl::pmr::vector<FVec3> listedHalves;        /**< Each body's half size when the lists were made. */
    lpl::pmr::vector<FVec3> listedPositions;     /**< Where each body stood when the lists were made. */
    lpl::pmr::vector<core::u32> neighbourStarts; /**< Where each body's list starts; one more than the bodies. */
    lpl::pmr::vector<core::u32> neighbours;      /**< The bodies after each one within kListReach, in order. */
    lpl::pmr::vector<math::AABB<math::Fixed32>> filedBoxes; /**< The boxes as the listing began: what a tree holds. */
    lpl::pmr::vector<core::u32>
        gridCellOf; /**< The grid cell each body's box starts in, when listing through a grid. */
    lpl::pmr::vector<core::u32> gridStarts;  /**< Where each grid cell's bodies start; one more than the cells. */
    lpl::pmr::vector<core::u32> gridCursors; /**< Where the next body of each grid cell goes, while they are placed. */
    lpl::pmr::vector<core::u32> gridBodies;  /**< The bodies, cell after cell, each cell's in their order. */
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
     * @brief Files every body's box as it is now in @c filedBoxes.
     */
    void fileBoxes()
    {
        const core::u32 count = static_cast<core::u32>(physicsEntities.size());
        filedBoxes.resize(count);
        for (core::u32 body = 0u; body < count; ++body)
            filedBoxes[body] = boxOf(body);
    }

    /**
     * @brief The box the filed boxes span.
     *
     * The lowest corner first, then the highest: on i686, the six bounds of a merge do not fit the
     * registers together, and each box spilled them.
     */
    [[nodiscard]] math::AABB<math::Fixed32> filedSpan() const noexcept
    {
        const core::u32 count = static_cast<core::u32>(filedBoxes.size());
        FVec3 low = filedBoxes[0].min;
        for (core::u32 body = 1u; body < count; ++body)
        {
            const FVec3 &corner = filedBoxes[body].min;
            low.x = corner.x < low.x ? corner.x : low.x;
            low.y = corner.y < low.y ? corner.y : low.y;
            low.z = corner.z < low.z ? corner.z : low.z;
        }
        FVec3 high = filedBoxes[0].max;
        for (core::u32 body = 1u; body < count; ++body)
        {
            const FVec3 &corner = filedBoxes[body].max;
            high.x = corner.x > high.x ? corner.x : high.x;
            high.y = corner.y > high.y ? corner.y : high.y;
            high.z = corner.z > high.z ? corner.z : high.z;
        }
        return {low, high};
    }

    /**
     * @brief Builds the broad-phase over @c filedBoxes, bounded by the box they span.
     */
    Octree &buildTree()
    {
        const math::AABB<math::Fixed32> span = filedSpan();
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
     * @brief Lists, for every body, the bodies after it whose box comes within kListReach of its own,
     *        from where they stand now, in increasing order.
     *
     * The order is the order their contacts are solved in, so it is fixed whichever way the lists are
     * made: the tree and the grid find the same bodies, not in the same order.
     */
    void listNeighbours()
    {
        const core::u32 count = static_cast<core::u32>(physicsEntities.size());
        fileBoxes();
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
        }
        if (!listThroughGrid())
            listThroughTree();
        for (core::u32 body = 0u; body < count; ++body)
            sortAscending(neighbours, neighbourStarts[body], neighbourStarts[body + 1u]);
    }

    /**
     * @brief Lists every body's neighbours by searching the tree, built over the filed boxes, once
     *        per body.
     */
    void listThroughTree()
    {
        const core::u32 count = static_cast<core::u32>(filedBoxes.size());
        Octree &octree = buildTree();
        for (core::u32 body = 0u; body < count; ++body)
        {
            neighbourStarts[body] = static_cast<core::u32>(neighbours.size());
            octree.collect(filedBoxes[body].expand(kListReach), body + 1u, neighbours);
        }
        neighbourStarts[count] = static_cast<core::u32>(neighbours.size());
    }

    /**
     * @brief Lists every body's neighbours through a grid whose cells are as wide as the widest box
     *        and kListReach, when the filed boxes span few enough of them.
     *
     * Two boxes that come within kListReach of each other start at most one cell apart on each
     * axis, so a body's neighbours are among the bodies of the 27 cells around the one its box
     * starts in, and the test that decides is the one a search of the tree makes: the lists hold
     * the same bodies, in another order, which listNeighbours sorts. Building the tree only to
     * search it once per body cost more than placing the bodies in cells.
     *
     * @return False, having listed nothing, before the tree has been built once, when the boxes
     *         span more than kMostGridCellsPerBody cells per body, or when a cell would be wider
     *         than half the range of a word. The first listing builds the tree, so that its storage
     *         is sized before a search through it needs it.
     */
    [[nodiscard]] bool listThroughGrid()
    {
        if (!broadPhase)
            return false;
        const core::u32 count = static_cast<core::u32>(filedBoxes.size());
        core::u32 extents[3] = {0u, 0u, 0u};
        for (const auto &box : filedBoxes)
        {
            const core::u32 x = static_cast<core::u32>(box.max.x.raw()) - static_cast<core::u32>(box.min.x.raw());
            const core::u32 y = static_cast<core::u32>(box.max.y.raw()) - static_cast<core::u32>(box.min.y.raw());
            const core::u32 z = static_cast<core::u32>(box.max.z.raw()) - static_cast<core::u32>(box.min.z.raw());
            extents[0] = x > extents[0] ? x : extents[0];
            extents[1] = y > extents[1] ? y : extents[1];
            extents[2] = z > extents[2] ? z : extents[2];
        }

        const math::AABB<math::Fixed32> span = filedSpan();
        const core::u32 lows[3] = {static_cast<core::u32>(span.min.x.raw()), static_cast<core::u32>(span.min.y.raw()),
                                   static_cast<core::u32>(span.min.z.raw())};
        const core::u32 highs[3] = {static_cast<core::u32>(span.max.x.raw()), static_cast<core::u32>(span.max.y.raw()),
                                    static_cast<core::u32>(span.max.z.raw())};
        core::u32 widths[3];
        core::u32 cells[3];
        core::u64 cellCount = 1u;
        const core::u32 reach = static_cast<core::u32>(kListReach.raw());
        for (core::u32 axis = 0u; axis < 3u; ++axis)
        {
            if (extents[axis] > 0x7FFFFFFFu - reach)
                return false;
            widths[axis] = extents[axis] + reach;
            cells[axis] = (highs[axis] - lows[axis]) / widths[axis] + 1u;
            cellCount *= cells[axis];
            if (cellCount > static_cast<core::u64>(count) * kMostGridCellsPerBody)
                return false;
        }

        const core::u32 total = static_cast<core::u32>(cellCount);
        if (gridStarts.capacity() < static_cast<std::size_t>(count) * kMostGridCellsPerBody + 1u)
        {
            gridStarts.reserve(static_cast<std::size_t>(count) * kMostGridCellsPerBody + 1u);
            gridCursors.reserve(static_cast<std::size_t>(count) * kMostGridCellsPerBody);
        }
        gridStarts.resize(total + 1u);
        for (core::u32 &start : gridStarts)
            start = 0u;
        gridCellOf.resize(count);
        for (core::u32 body = 0u; body < count; ++body)
        {
            const math::AABB<math::Fixed32> &box = filedBoxes[body];
            const core::u32 x = (static_cast<core::u32>(box.min.x.raw()) - lows[0]) / widths[0];
            const core::u32 y = (static_cast<core::u32>(box.min.y.raw()) - lows[1]) / widths[1];
            const core::u32 z = (static_cast<core::u32>(box.min.z.raw()) - lows[2]) / widths[2];
            const core::u32 cell = (z * cells[1] + y) * cells[0] + x;
            gridCellOf[body] = cell;
            ++gridStarts[cell + 1u];
        }
        for (core::u32 cell = 0u; cell < total; ++cell)
            gridStarts[cell + 1u] += gridStarts[cell];
        gridCursors.resize(total);
        for (core::u32 cell = 0u; cell < total; ++cell)
            gridCursors[cell] = gridStarts[cell];
        gridBodies.resize(count);
        for (core::u32 body = 0u; body < count; ++body)
            gridBodies[gridCursors[gridCellOf[body]]++] = body;

        for (core::u32 body = 0u; body < count; ++body)
        {
            const math::AABB<math::Fixed32> region = filedBoxes[body].expand(kListReach);
            const core::u32 cell = gridCellOf[body];
            const core::u32 x = cell % cells[0];
            const core::u32 y = (cell / cells[0]) % cells[1];
            const core::u32 z = cell / cells[0] / cells[1];
            const core::u32 firstX = x == 0u ? 0u : x - 1u;
            const core::u32 lastX = x + 1u < cells[0] ? x + 1u : x;
            neighbourStarts[body] = static_cast<core::u32>(neighbours.size());
            for (core::u32 nearZ = z == 0u ? 0u : z - 1u; nearZ <= z + 1u && nearZ < cells[2]; ++nearZ)
            {
                for (core::u32 nearY = y == 0u ? 0u : y - 1u; nearY <= y + 1u && nearY < cells[1]; ++nearY)
                {
                    const core::u32 row = (nearZ * cells[1] + nearY) * cells[0];
                    for (core::u32 k = gridStarts[row + firstX]; k < gridStarts[row + lastX + 1u]; ++k)
                    {
                        const core::u32 other = gridBodies[k];
                        if (other > body && filedBoxes[other].intersects(region))
                            neighbours.push_back(other);
                    }
                }
            }
        }
        neighbourStarts[count] = static_cast<core::u32>(neighbours.size());
        return true;
    }

    /**
     * @brief Finds the step's contacts: every body near the ground first, then every pair of bodies
     *        whose boxes come within kSpeculativeDistance, each body's partners in increasing order.
     *
     * Up to kBruteForceThreshold bodies every pair is tried. Past it the pairs come from the neighbour
     * lists, made again first when a body has moved beyond kNeighbourMargin of where they found it.
     *
     * @param inverseDt One over the step, in seconds.
     */
    void findContacts(math::Fixed32 inverseDt)
    {
        const core::u32 count = static_cast<core::u32>(physicsEntities.size());
        const bool everyPair = count <= kBruteForceThreshold;
        if (!everyPair && !listsStillHold())
            listNeighbours();

        const std::size_t pairs = everyPair ? static_cast<std::size_t>(count) * count / 2u : neighbours.size();
        const std::size_t most = pairs + count > static_cast<std::size_t>(count) * (kNeighboursReserved + 1u) ?
                                     pairs + count :
                                     static_cast<std::size_t>(count) * (kNeighboursReserved + 1u);
        if (contacts.capacity() < most)
        {
            contacts.reserve(most);
            previousContacts.reserve(most);
        }
        contacts.swap(previousContacts);
        contacts.clear();

        for (core::u32 body = 0u; body < count; ++body)
            considerGround(body, inverseDt);
        if (everyPair)
        {
            for (core::u32 a = 0u; a < count; ++a)
                for (core::u32 b = a + 1u; b < count; ++b)
                    considerPair(a, b, inverseDt);
            return;
        }
        for (core::u32 a = 0u; a < count; ++a)
            for (core::u32 k = neighbourStarts[a]; k < neighbourStarts[a + 1u]; ++k)
                considerPair(a, neighbours[k], inverseDt);
    }

    /**
     * @brief Adds the contact of @p body with the ground when it stands within kSpeculativeDistance
     *        of it, awake and light enough to move.
     *
     * @param body      The body.
     * @param inverseDt One over the step, in seconds.
     */
    void considerGround(core::u32 body, math::Fixed32 inverseDt)
    {
        const PhysicsEntityRef &ref = physicsEntities[body];
        const math::Fixed32 gap = ref.position->y - kDefaultHalfHeight;
        if (gap > kSpeculativeDistance || ref.inverseMass < kMassEpsilon || isSleeping(ref.id.slot()))
            return;
        contacts.push_back(Contact{body, kGround, ref.id, ref.id, &FVec3::y, true, ref.inverseMass,
                                   math::Fixed32::zero(), ref.mass, targetSpeed(gap, ref.velocity->y, inverseDt),
                                   math::Fixed32::zero()});
    }

    /**
     * @brief Adds the contact of bodies @p a and @p b when their boxes come within
     *        kSpeculativeDistance of each other across one face, unless both sleep.
     *
     * The normal is the axis of least overlap, from @p b to @p a. Boxes that overlap wake each other;
     * across a gap, a sleeper stays asleep and holds still, with no inverse mass in the contact.
     *
     * @param a         The first body.
     * @param b         The second body, after it.
     * @param inverseDt One over the step, in seconds.
     */
    void considerPair(core::u32 a, core::u32 b, math::Fixed32 inverseDt)
    {
        const PhysicsEntityRef &refA = physicsEntities[a];
        const PhysicsEntityRef &refB = physicsEntities[b];
        bool asleepA = isSleeping(refA.id.slot());
        bool asleepB = isSleeping(refB.id.slot());
        if (asleepA && asleepB)
            return;

        const FVec3 delta = *refA.position - *refB.position;
        const math::Fixed32 overlapX = (refA.half.x + refB.half.x) - delta.x.abs();
        const math::Fixed32 overlapY = (refA.half.y + refB.half.y) - delta.y.abs();
        const math::Fixed32 overlapZ = (refA.half.z + refB.half.z) - delta.z.abs();

        math::Fixed32 FVec3::*axis = &FVec3::z;
        math::Fixed32 overlap = overlapZ;
        math::Fixed32 across = overlapX < overlapY ? overlapX : overlapY;
        if (overlapX <= overlapY && overlapX <= overlapZ)
        {
            axis = &FVec3::x;
            overlap = overlapX;
            across = overlapY < overlapZ ? overlapY : overlapZ;
        }
        else if (overlapY <= overlapZ)
        {
            axis = &FVec3::y;
            overlap = overlapY;
            across = overlapX < overlapZ ? overlapX : overlapZ;
        }
        if (across <= math::Fixed32::zero() || overlap < -kSpeculativeDistance)
            return;

        if (overlap > math::Fixed32::zero())
        {
            wake(refA.id.slot());
            wake(refB.id.slot());
            asleepA = false;
            asleepB = false;
        }
        const math::Fixed32 inverseMassA = asleepA ? math::Fixed32::zero() : refA.inverseMass;
        const math::Fixed32 inverseMassB = asleepB ? math::Fixed32::zero() : refB.inverseMass;
        const math::Fixed32 inverseMassSum = inverseMassA + inverseMassB;
        if (inverseMassSum < kMassEpsilon)
            return;

        const bool positive = delta.*axis >= math::Fixed32::zero();
        const math::Fixed32 approach = alongNormal(refA.velocity->*axis - refB.velocity->*axis, positive);
        contacts.push_back(Contact{a, b, refA.id, refB.id, axis, positive, inverseMassA, inverseMassB,
                                   math::Fixed32::one() / inverseMassSum, targetSpeed(-overlap, approach, inverseDt),
                                   math::Fixed32::zero()});
    }

    /**
     * @brief Gives each contact the impulse the same contact summed over the last step, and applies
     *        it: warm starting (Catto, GDC 2006; the contact cache of Jolt).
     *
     * A stack at rest needs the same impulses at every step, and iterations that start from zero pass
     * a body's weight down the stack one box per iteration: with four of them, the top of a stack of
     * four kept most of a step of gravity. Started from the last step's impulses, they only correct
     * what changed.
     *
     * A contact is the same when it joins the same two entities along the same axis and the same
     * way. Both lists are in the order findContacts makes them, the ground's contacts first by body,
     * then the pairs by their first body and their second, so one pass over each finds them.
     */
    void warmStart() noexcept
    {
        core::u32 cursor = 0u;
        const core::u32 previous = static_cast<core::u32>(previousContacts.size());
        for (Contact &contact : contacts)
        {
            while (cursor < previous && comesBefore(previousContacts[cursor], contact))
                ++cursor;
            if (cursor == previous)
                return;
            const Contact &old = previousContacts[cursor];
            if (old.a != contact.a || old.b != contact.b || old.first != contact.first ||
                old.second != contact.second || old.axis != contact.axis || old.positive != contact.positive)
                continue;
            contact.impulse = old.impulse;
            const math::Fixed32 push = alongNormal(contact.impulse, contact.positive);
            FVec3 &velocityA = *physicsEntities[contact.a].velocity;
            velocityA.*contact.axis = velocityA.*contact.axis + push * contact.inverseMassA;
            if (contact.b != kGround)
            {
                FVec3 &velocityB = *physicsEntities[contact.b].velocity;
                velocityB.*contact.axis = velocityB.*contact.axis - push * contact.inverseMassB;
            }
        }
    }

    /**
     * @brief Whether contact @p left comes before contact @p right in the order findContacts makes
     *        them: the ground's contacts first, by body, then the pairs, by first body then second.
     */
    [[nodiscard]] static bool comesBefore(const Contact &left, const Contact &right) noexcept
    {
        const bool leftIsPair = left.b != kGround;
        const bool rightIsPair = right.b != kGround;
        if (leftIsPair != rightIsPair)
            return rightIsPair;
        if (left.a != right.a)
            return left.a < right.a;
        return leftIsPair && left.b < right.b;
    }

    /**
     * @brief One velocity iteration over the contacts, in order.
     *
     * Each contact takes the impulse that brings its normal speed to its target, given what the
     * contacts before it did, then holds the sum of its impulses at zero or more: a contact pushes,
     * it never pulls.
     *
     * @return Whether any contact changed its impulse. When none did, the next iteration would not
     *         either.
     */
    [[nodiscard]] bool solveVelocities() noexcept
    {
        bool changed = false;
        for (Contact &contact : contacts)
        {
            FVec3 &velocityA = *physicsEntities[contact.a].velocity;
            FVec3 *velocityB = contact.b == kGround ? nullptr : physicsEntities[contact.b].velocity;
            const math::Fixed32 relative =
                velocityB == nullptr ? velocityA.*contact.axis : velocityA.*contact.axis - velocityB->*contact.axis;
            const math::Fixed32 speed = alongNormal(relative, contact.positive);
            const math::Fixed32 wanted = contact.impulse + contact.mass * (contact.target - speed);
            const math::Fixed32 summed = wanted > math::Fixed32::zero() ? wanted : math::Fixed32::zero();
            const math::Fixed32 impulse = summed - contact.impulse;
            if (impulse.raw() == 0)
                continue;
            contact.impulse = summed;
            changed = true;
            const math::Fixed32 push = alongNormal(impulse, contact.positive);
            velocityA.*contact.axis = velocityA.*contact.axis + push * contact.inverseMassA;
            if (velocityB != nullptr)
                velocityB->*contact.axis = velocityB->*contact.axis - push * contact.inverseMassB;
        }
        return changed;
    }

    /**
     * @brief One position iteration over the contacts, once the bodies have moved: a pair that still
     *        overlaps along its axis moves apart by the whole overlap, shared by inverse mass, and a
     *        body below the ground returns to it.
     *
     * @return Whether anything moved. When nothing did, the next iteration would not either.
     */
    [[nodiscard]] bool correctPositions() noexcept
    {
        bool moved = false;
        for (const Contact &contact : contacts)
        {
            FVec3 &positionA = *physicsEntities[contact.a].position;
            if (contact.b == kGround)
            {
                if (positionA.y < kDefaultHalfHeight)
                {
                    positionA.y = kDefaultHalfHeight;
                    moved = true;
                }
                continue;
            }
            FVec3 &positionB = *physicsEntities[contact.b].position;
            const math::Fixed32 reach =
                physicsEntities[contact.a].half.*contact.axis + physicsEntities[contact.b].half.*contact.axis;
            const math::Fixed32 overlap =
                reach - alongNormal(positionA.*contact.axis - positionB.*contact.axis, contact.positive);
            if (overlap <= math::Fixed32::zero())
                continue;
            const math::Fixed32 push = alongNormal(overlap * contact.mass, contact.positive);
            positionA.*contact.axis = positionA.*contact.axis + push * contact.inverseMassA;
            positionB.*contact.axis = positionB.*contact.axis - push * contact.inverseMassB;
            moved = true;
        }
        return moved;
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

            auto *velocities = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Velocity));
            auto *masses = static_cast<const math::Fixed32 *>(chunk->readComponent(ecs::ComponentId::Mass));
            if (!velocities || !masses)
            {
                continue;
            }

            const ecs::EntityId *entities = chunk->entities().data();
            for (core::u32 i = 0; i < count; ++i)
            {
                _impl->ensureSleepCapacity(entities[i].slot());
            }

            integrateVelocities(entities, velocities, masses, count, fdt);
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
                    Impl::PhysicsEntityRef{entities[i], &positions[i], &velocities[i], aabbs[i] * kHalf, masses[i],
                                           masses[i] > kMassEpsilon ? masses[i].reciprocal() : math::Fixed32::zero()});
            }
        }
    }
    resolveCollisionsWorld(math::Fixed32::one() / fdt);

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
            if (!positions || !velocities)
            {
                continue;
            }

            integratePositions(chunk->entities().data(), positions, velocities, count, fdt);
        }
    }
    for (core::u32 iteration = 0u; iteration < kPositionIterations; ++iteration)
    {
        if (!_impl->correctPositions())
            break;
    }

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

void CpuPhysicsBackend::integrateVelocities(const ecs::EntityId *entities, FVec3 *velocities,
                                            const math::Fixed32 *masses, core::u32 count,
                                            math::Fixed32 dt) const noexcept
{
    const math::simd::SimdFixed4 vDamping = math::simd::SimdFixed4::splat(kVelocityDamping.raw());

    auto integrateScalar = [&](core::u32 k) {
        if (masses[k] > kMassEpsilon)
            velocities[k].y += kGravity * dt;
        velocities[k] = velocities[k] * kVelocityDamping;
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

        core::i32 vx[4], vy[4], vz[4];
        for (core::u32 j = 0; j < 4; ++j)
        {
            vx[j] = velocities[i + j].x.raw();
            vy[j] = velocities[i + j].y.raw();
            vz[j] = velocities[i + j].z.raw();
        }

        (math::simd::SimdFixed4::load(vx) * vDamping).store(vx);
        (math::simd::SimdFixed4::load(vy) * vDamping).store(vy);
        (math::simd::SimdFixed4::load(vz) * vDamping).store(vz);

        for (core::u32 j = 0; j < 4; ++j)
            velocities[i + j] = {math::Fixed32::fromRaw(vx[j]), math::Fixed32::fromRaw(vy[j]),
                                 math::Fixed32::fromRaw(vz[j])};
    }

    for (; i < count; ++i)
    {
        if (_impl->isSleeping(entities[i].slot()))
            continue;
        integrateScalar(i);
    }
}

void CpuPhysicsBackend::integratePositions(const ecs::EntityId *entities, FVec3 *positions, FVec3 *velocities,
                                           core::u32 count, math::Fixed32 dt) const noexcept
{
    const math::simd::SimdFixed4 vDt = math::simd::SimdFixed4::splat(dt.raw());

    auto returnToTheGround = [&](core::u32 k) {
        if (positions[k].y < kDefaultHalfHeight)
        {
            positions[k].y = kDefaultHalfHeight;
            if (velocities[k].y < math::Fixed32::zero())
                velocities[k].y = bounceOf(velocities[k].y);
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
            {
                if (_impl->isSleeping(entities[i + j].slot()))
                    continue;
                positions[i + j] = positions[i + j] + velocities[i + j] * dt;
                returnToTheGround(i + j);
            }
            continue;
        }

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

        (math::simd::SimdFixed4::load(px) + math::simd::SimdFixed4::load(vx) * vDt).store(px);
        (math::simd::SimdFixed4::load(py) + math::simd::SimdFixed4::load(vy) * vDt).store(py);
        (math::simd::SimdFixed4::load(pz) + math::simd::SimdFixed4::load(vz) * vDt).store(pz);

        for (core::u32 j = 0; j < 4; ++j)
        {
            positions[i + j] = {math::Fixed32::fromRaw(px[j]), math::Fixed32::fromRaw(py[j]),
                                math::Fixed32::fromRaw(pz[j])};
            returnToTheGround(i + j);
        }
    }

    for (; i < count; ++i)
    {
        if (_impl->isSleeping(entities[i].slot()))
            continue;
        positions[i] = positions[i] + velocities[i] * dt;
        returnToTheGround(i);
    }
}

void CpuPhysicsBackend::resolveCollisionsWorld(math::Fixed32 inverseDt) const noexcept
{
    if (_impl->physicsEntities.empty())
    {
        _impl->contacts.clear();
        return;
    }

    _impl->findContacts(inverseDt);
    _impl->warmStart();
    for (core::u32 iteration = 0u; iteration < kVelocityIterations; ++iteration)
    {
        if (!_impl->solveVelocities())
            break;
    }

    if (!core::EventBus::instance().hasSubscribers())
        return;
    for (const Impl::Contact &contact : _impl->contacts)
    {
        if (contact.b == Impl::kGround || contact.impulse <= math::Fixed32::zero())
            continue;
        publishContact(_impl->physicsEntities[contact.a].id, _impl->physicsEntities[contact.b].id, contact.axis,
                       contact.positive ? math::Fixed32::one() : -math::Fixed32::one(), contact.impulse);
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
