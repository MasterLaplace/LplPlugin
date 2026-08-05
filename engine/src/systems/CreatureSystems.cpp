/**
 * @file CreatureSystems.cpp
 * @brief Implementation of the living world as systems over entities.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/engine/systems/CreatureSystems.hpp>

#include <lpl/ai/Personality.hpp>
#include <lpl/ecology/Genome.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
// CreaturePipeline builds every stage from the layer, so it needs the whole class;
// the header keeps only the forward declaration, which is what stops the cycle
// (LivingLayer.hpp includes this header for CreatureFieldView).
#include <lpl/engine/LivingLayer.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/FixedMath.hpp>
#include <lpl/procgen/Grid.hpp>

namespace lpl::engine::systems {

namespace {

/// What a creature is: somewhere to be, a kind, and a way to move.
struct CreatureChunk {
    math::Vec3<math::Fixed32> *positions{nullptr};
    math::Vec3<math::Fixed32> *velocities{nullptr};
    const core::u32 *creature{nullptr};     ///< Pairs of (species, id).
    const ecology::Genome *genome{nullptr}; ///< Five Fixed32 per body.
    math::Fixed32 *heading{nullptr};        ///< Pairs of (x, z).
    core::u32 count{0u};
};

/// What a system needs beyond Position and Creature, which are always required.
enum Need : core::u32 {
    kNeedVelocity = 1u << 0,
    kNeedGenome = 1u << 1,
    kNeedHeading = 1u << 2,
};

/// Resolves one chunk into the arrays a creature system asked for.
///
/// Asks the ARCHETYPE whether a component is there, never the pointer:
/// readComponent answers non-null for every id, allocated or not, and trusting it
/// is how entities came to read another component's bytes as their own.
///
/// EVERY pointer comes from the write side, readers included and immutable data
/// included, and that is not a detail. Chunks are double buffered and the swap is a phase callback the Engine
/// installs AFTER Physics — so a world whose systems are all PrePhysics never swaps
/// at all, and its front buffer holds whatever the components were born with.
/// Reading the front side there made every creature deposit its scent at the origin
/// for the whole run. World::stateHash folds the write side for the same reason,
/// with the same story behind it: the read side "would sit still for a whole tick".
template <typename Chunk, typename Partition>
bool viewCreatures(const Partition &part, Chunk &chunk, CreatureChunk &out, core::u32 needs)
{
    if (!part.archetype().has(ecs::ComponentId::Position) || !part.archetype().has(ecs::ComponentId::Creature))
        return false;
    if ((needs & kNeedVelocity) != 0u && !part.archetype().has(ecs::ComponentId::Velocity))
        return false;
    if ((needs & kNeedGenome) != 0u && !part.archetype().has(ecs::ComponentId::Genome))
        return false;
    if ((needs & kNeedHeading) != 0u && !part.archetype().has(ecs::ComponentId::Heading))
        return false;

    out.positions = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Position));
    out.creature = static_cast<const core::u32 *>(chunk.writeComponent(ecs::ComponentId::Creature));
    if ((needs & kNeedVelocity) != 0u)
        out.velocities = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Velocity));
    if ((needs & kNeedGenome) != 0u)
        out.genome = static_cast<const ecology::Genome *>(chunk.writeComponent(ecs::ComponentId::Genome));
    if ((needs & kNeedHeading) != 0u)
        out.heading = static_cast<math::Fixed32 *>(chunk.writeComponent(ecs::ComponentId::Heading));
    out.count = chunk.count();

    if (out.positions == nullptr || out.creature == nullptr)
        return false;
    if ((needs & kNeedVelocity) != 0u && out.velocities == nullptr)
        return false;
    if ((needs & kNeedGenome) != 0u && out.genome == nullptr)
        return false;
    return (needs & kNeedHeading) == 0u || out.heading != nullptr;
}

const ecs::ComponentAccess kDepositAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadOnly},
    {ecs::ComponentId::Creature, ecs::AccessMode::ReadOnly},
};
// The field is not entity state, so before ecs::ResourceId existed this system
// declared nothing it wrote — and the scheduler was free to run the evaporation
// pass in the same wave as the deposits it evaporates. It happened to come out in
// registration order under the inline job system; a real thread pool would have
// raced. Naming the resource is what turns that order into an edge.
const ecs::ResourceAccess kFieldWrite[] = {
    {ecs::ResourceId::ScentField, ecs::AccessMode::ReadWrite},
};
const ecs::ResourceAccess kFieldRead[] = {
    {ecs::ResourceId::ScentField, ecs::AccessMode::ReadOnly},
};
const ecs::SystemDescriptor kDepositDesc{"ScentDeposit", ecs::SchedulePhase::PrePhysics,
                                         std::span<const ecs::ComponentAccess>{kDepositAccesses},
                                         std::span<const ecs::ResourceAccess>{kFieldWrite}};

const ecs::SystemDescriptor kFieldDesc{"ScentField", ecs::SchedulePhase::PrePhysics,
                                       std::span<const ecs::ComponentAccess>{},
                                       std::span<const ecs::ResourceAccess>{kFieldWrite}};

// No Genome here even though an animal's nature scales the pull: personality is
// derived from the Creature component's (id, species), so declaring Genome would
// name a dependency this system does not have.
const ecs::ComponentAccess kSteerAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadOnly },
    {ecs::ComponentId::Creature, ecs::AccessMode::ReadOnly },
    {ecs::ComponentId::Velocity, ecs::AccessMode::ReadWrite},
};
const ecs::SystemDescriptor kSteerDesc{"ScentSteering", ecs::SchedulePhase::PrePhysics,
                                       std::span<const ecs::ComponentAccess>{kSteerAccesses},
                                       std::span<const ecs::ResourceAccess>{kFieldRead}};

const ecs::ComponentAccess kFlockAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadOnly },
    {ecs::ComponentId::Creature, ecs::AccessMode::ReadOnly },
    {ecs::ComponentId::Velocity, ecs::AccessMode::ReadWrite},
};
const ecs::SystemDescriptor kFlockDesc{"Flocking", ecs::SchedulePhase::PrePhysics,
                                       std::span<const ecs::ComponentAccess>{kFlockAccesses}};

const ecs::ComponentAccess kGrazeAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadOnly},
    {ecs::ComponentId::Creature, ecs::AccessMode::ReadOnly},
};
const ecs::ResourceAccess kGrazeResources[] = {
    {ecs::ResourceId::Vegetation, ecs::AccessMode::ReadWrite},
};
const ecs::SystemDescriptor kGrazeDesc{"Grazing", ecs::SchedulePhase::PrePhysics,
                                       std::span<const ecs::ComponentAccess>{kGrazeAccesses},
                                       std::span<const ecs::ResourceAccess>{kGrazeResources}};

// Velocity is READ-WRITE, not read-only: recovering a stray rewrites it, because
// leaving the flock's velocity intact walks the body straight back out of bounds on
// the next tick. A descriptor that understated this would be a lie the scheduler
// believes.
const ecs::ComponentAccess kWalkAccesses[] = {
    {ecs::ComponentId::Creature, ecs::AccessMode::ReadOnly },
    {ecs::ComponentId::Genome,   ecs::AccessMode::ReadOnly },
    {ecs::ComponentId::Velocity, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Position, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Heading,  ecs::AccessMode::ReadWrite},
};
const ecs::ResourceAccess kWalkResources[] = {
    {ecs::ResourceId::Terrain, ecs::AccessMode::ReadOnly},
};
const ecs::SystemDescriptor kWalkDesc{"Locomotion", ecs::SchedulePhase::PrePhysics,
                                      std::span<const ecs::ComponentAccess>{kWalkAccesses},
                                      std::span<const ecs::ResourceAccess>{kWalkResources}};

} // namespace

bool CreatureFieldView::toCell(core::i32 worldX, core::i32 worldZ, core::u32 &outX, core::u32 &outZ) const noexcept
{
    if (width == 0u || depth == 0u)
        return false;
    const core::i32 localX = worldX - originX;
    const core::i32 localZ = worldZ - originZ;
    if (localX < 0 || localZ < 0 || static_cast<core::u32>(localX) >= width || static_cast<core::u32>(localZ) >= depth)
        return false;
    outX = static_cast<core::u32>(localX);
    outZ = static_cast<core::u32>(localZ);
    return true;
}

// ── ScentDepositSystem ────────────────────────────────────────────────────────

ScentDepositSystem::ScentDepositSystem(ecs::Registry &registry, ai::StigmergyField &field,
                                       const ecology::HerdParams &params, const CreatureFieldView &view)
    : _registry(registry), _field(field), _params(params), _view(view)
{
}

const ecs::SystemDescriptor &ScentDepositSystem::descriptor() const noexcept { return kDepositDesc; }

void ScentDepositSystem::execute(core::f32 /*dt*/)
{
    _deposits = 0u;
    for (const auto &part : _registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            CreatureChunk view;
            if (!chunk || !viewCreatures(*part, *chunk, view, 0u))
                continue;

            for (core::u32 i = 0u; i < view.count; ++i)
            {
                const core::u32 species = view.creature[i * 2u];
                if (species >= ecology::kMaxHerdSpecies)
                    continue;
                const ecology::SpeciesScent &nose = _params.scent[species];
                if (nose.depositChannel == ecology::SpeciesScent::kNoDeposit)
                    continue;

                core::u32 cellX = 0u;
                core::u32 cellZ = 0u;
                if (!_view.toCell(view.positions[i].x.toInt(), view.positions[i].z.toInt(), cellX, cellZ))
                    continue;
                _field.deposit(nose.depositChannel, cellX, cellZ, nose.depositAmount);
                ++_deposits;
            }
        }
    }
}

// ── ScentFieldSystem ──────────────────────────────────────────────────────────

ScentFieldSystem::ScentFieldSystem(ai::StigmergyField &field, const ai::StigmergyParams &params)
    : _field(field), _params(params)
{
}

const ecs::SystemDescriptor &ScentFieldSystem::descriptor() const noexcept { return kFieldDesc; }

void ScentFieldSystem::execute(core::f32 /*dt*/) { _field.step(_params); }

// ── ScentSteeringSystem ───────────────────────────────────────────────────────

ScentSteeringSystem::ScentSteeringSystem(ecs::Registry &registry, const ai::StigmergyField &field,
                                         const ecology::HerdParams &params, const CreatureFieldView &view)
    : _registry(registry), _field(field), _params(params), _view(view)
{
}

const ecs::SystemDescriptor &ScentSteeringSystem::descriptor() const noexcept { return kSteerDesc; }

void ScentSteeringSystem::execute(core::f32 /*dt*/)
{
    _steered = 0u;
    for (const auto &part : _registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            CreatureChunk view;
            if (!chunk || !viewCreatures(*part, *chunk, view, kNeedVelocity))
                continue;

            for (core::u32 i = 0u; i < view.count; ++i)
            {
                const core::u32 species = view.creature[i * 2u];
                if (species >= ecology::kMaxHerdSpecies)
                    continue;
                const ai::ScentPalate &palate = _params.scent[species].palate;
                if (palate.count == 0u)
                    continue;

                core::u32 cellX = 0u;
                core::u32 cellZ = 0u;
                if (!_view.toCell(view.positions[i].x.toInt(), view.positions[i].z.toInt(), cellX, cellZ))
                    continue;

                const core::u32 direction = _field.palateDirection(palate, cellX, cellZ);
                if (direction == ai::StigmergyField::kNoDirection)
                    continue;

                // Scaled by the animal's own nature, not by the species average.
                // personalityOf is a pure function of (id, species), both of which
                // the Creature component carries — so the system reproduces exactly
                // what Herd::step used to do, without needing the herd. Dropping
                // this term flattened the pack's behaviour the first time round,
                // and the encirclement measurement caught it.
                const ai::PersonalityTraits traits = ai::personalityOf(view.creature[i * 2u + 1u], species);
                const math::Fixed32 pull = _params.scentPull * (math::Fixed32::half() + traits.energy);
                view.velocities[i].x =
                    view.velocities[i].x + math::Fixed32::fromInt(procgen::kNeighbor8X[direction]) * pull;
                view.velocities[i].z =
                    view.velocities[i].z + math::Fixed32::fromInt(procgen::kNeighbor8Z[direction]) * pull;
                ++_steered;
            }
        }
    }
}

// ── FlockingSystem ────────────────────────────────────────────────────────────

FlockingSystem::FlockingSystem(ecs::Registry &registry, const ecology::HerdParams &params)
    : _registry(registry), _params(params)
{
}

const ecs::SystemDescriptor &FlockingSystem::descriptor() const noexcept { return kFlockDesc; }

void FlockingSystem::execute(core::f32 /*dt*/)
{
    _flocked = 0u;
    for (core::u32 species = 0u; species < _params.speciesCount; ++species)
    {
        // Gathered per species, because a flock is a species: a deer does not
        // align with a wolf. clear() keeps the capacity, so this costs an
        // allocation on the first tick and none afterwards.
        _flock.clear();
        _sinks.clear();
        for (const auto &part : _registry.partitions())
        {
            if (!part)
                continue;
            for (const auto &chunk : part->chunks())
            {
                CreatureChunk view;
                if (!chunk || !viewCreatures(*part, *chunk, view, kNeedVelocity))
                    continue;
                for (core::u32 i = 0u; i < view.count; ++i)
                {
                    if (view.creature[i * 2u] != species)
                        continue;
                    ai::Boid boid;
                    boid.x = view.positions[i].x;
                    boid.z = view.positions[i].z;
                    boid.vx = view.velocities[i].x;
                    boid.vz = view.velocities[i].z;
                    _flock.push_back(boid);
                    _sinks.push_back(&view.velocities[i]);
                }
            }
        }
        if (_flock.empty())
            continue;

        const bool hunter = species == 1u;
        ai::BoidParams boids;
        boids.separationWeight = hunter ? _params.separationHunter : _params.separationGrazer;
        boids.alignmentWeight = hunter ? _params.alignmentHunter : _params.alignmentGrazer;
        boids.cohesionWeight = hunter ? _params.cohesionHunter : _params.cohesionGrazer;
        boids.neighbourRadius = math::Fixed32::fromInt(hunter ? _params.neighbourHunter : _params.neighbourGrazer);
        boids.separationRadius = hunter ? _params.separationHunterRadius : _params.separationGrazerRadius;

        // dt is the params' fixed step, not the frame's: this is authoritative, and
        // a wall clock on the authoritative path is a desync waiting for a slow
        // frame. The integration the flock performed is thrown away — only the
        // velocities come back, because terrain decides the position.
        ai::stepBoids(&_flock[0], static_cast<core::u32>(_flock.size()), boids, _params.step);

        for (core::usize i = 0u; i < _sinks.size(); ++i)
        {
            _sinks[i]->x = _flock[i].vx;
            _sinks[i]->z = _flock[i].vz;
        }
        _flocked += static_cast<core::u32>(_sinks.size());
    }
}

// ── GrazingSystem ─────────────────────────────────────────────────────────────

GrazingSystem::GrazingSystem(ecs::Registry &registry, const ecology::HerdParams &params, ITerrainQuery &terrain)
    : _registry(registry), _params(params), _terrain(terrain)
{
}

const ecs::SystemDescriptor &GrazingSystem::descriptor() const noexcept { return kGrazeDesc; }

void GrazingSystem::execute(core::f32 /*dt*/)
{
    _meals = 0u;
    for (const auto &part : _registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            CreatureChunk view;
            if (!chunk || !viewCreatures(*part, *chunk, view, 0u))
                continue;
            for (core::u32 i = 0u; i < view.count; ++i)
            {
                const core::u32 species = view.creature[i * 2u];
                if (species >= ecology::kMaxHerdSpecies || !ecology::isHerbivore(_params.scent[species]))
                    continue;
                if (_terrain.consumePlantAt(view.positions[i].x.toInt(), view.positions[i].z.toInt()))
                    ++_meals;
            }
        }
    }
}

// ── LocomotionSystem ──────────────────────────────────────────────────────────

LocomotionSystem::LocomotionSystem(ecs::Registry &registry, const ecology::HerdParams &params,
                                   const ITerrainQuery &terrain)
    : _registry(registry), _params(params), _terrain(terrain)
{
}

const ecs::SystemDescriptor &LocomotionSystem::descriptor() const noexcept { return kWalkDesc; }

void LocomotionSystem::execute(core::f32 /*dt*/)
{
    _cornered = 0u;
    _avoided = 0u;
    _strays = 0u;
    for (const auto &part : _registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            CreatureChunk view;
            if (!chunk ||
                !viewCreatures(*part, *chunk, view, kNeedVelocity | kNeedGenome | kNeedHeading))
                continue;

            for (core::u32 i = 0u; i < view.count; ++i)
            {
                const core::u32 species = view.creature[i * 2u];
                if (species >= ecology::kMaxHerdSpecies)
                    continue;
                const ai::PersonalityTraits traits = ai::personalityOf(view.creature[i * 2u + 1u], species);
                math::Vec3<math::Fixed32> &position = view.positions[i];
                math::Vec3<math::Fixed32> &velocity = view.velocities[i];
                math::Fixed32 &headingX = view.heading[i * 2u];
                math::Fixed32 &headingZ = view.heading[i * 2u + 1u];

                // ── Containment, before anything else ────────────────────────
                //
                // A body standing where it may not stand refuses every direction, so
                // it refuses two steps a tick for the rest of the run. Ask the world
                // which way is out and take that step; a world with no advice leaves
                // it to reverse, which at least changes the situation.
                if (!_terrain.standable(position.x, position.z))
                {
                    math::Fixed32 outX{};
                    math::Fixed32 outZ{};
                    if (!_terrain.recoveryDirection(position.x, position.z, outX, outZ))
                    {
                        outX = math::Fixed32{} - headingX;
                        outZ = math::Fixed32{} - headingZ;
                    }
                    position.x = position.x + outX;
                    position.z = position.z + outZ;
                    headingX = outX;
                    headingZ = outZ;
                    // The velocity goes with it. Leaving the flock's velocity intact
                    // walks the body straight back out of bounds on the next tick.
                    velocity.x = outX;
                    velocity.z = outZ;
                    ++_strays;
                    continue;
                }

                // Facing follows the velocity only when there IS one: normalising a
                // velocity near zero turns numerical noise into a direction, and an
                // animal standing still would spin. Keeping the last heading is the
                // point of storing one — standing still is a valid boid state and not
                // a valid animal one.
                const math::Fixed32 lengthSquared = velocity.x * velocity.x + velocity.z * velocity.z;
                const math::Fixed32 length = math::fixedSqrt(lengthSquared);
                if (length.raw() > 256)
                {
                    headingX = velocity.x / length;
                    headingZ = velocity.z / length;
                }

                // Pace from the genome and the personality, NOT from the velocity's
                // magnitude: the flock and the scent decide the direction, the genome
                // decides how fast this animal can possibly travel.
                const math::Fixed32 pace = view.genome[i].maxSpeed * _params.step *
                                           (math::Fixed32::fromFloat(0.7f) + traits.energy * math::Fixed32::half());

                // ── Avoidance, before the move rather than after refusing it ──
                //
                // A full body-length ahead, not the fraction one tick covers: looking
                // only as far as the next step means the turn happens with the
                // obstacle already underfoot, which is a collision reported as an
                // intention.
                const math::Fixed32 reach = math::Fixed32::one() + view.genome[i].size;
                if (!_terrain.standable(position.x + headingX * reach, position.z + headingZ * reach))
                {
                    math::Fixed32 bestX = headingX;
                    math::Fixed32 bestZ = headingZ;
                    math::Fixed32 bestDot = math::Fixed32::fromInt(-2);
                    bool found = false;
                    for (core::u32 n = 0u; n < 8u; ++n)
                    {
                        const math::Fixed32 candidateX = math::Fixed32::fromInt(procgen::kNeighbor8X[n]) *
                                                         (n < 4u ? math::Fixed32::one() : math::kInvSqrt2);
                        const math::Fixed32 candidateZ = math::Fixed32::fromInt(procgen::kNeighbor8Z[n]) *
                                                         (n < 4u ? math::Fixed32::one() : math::kInvSqrt2);
                        if (!_terrain.standable(position.x + candidateX * reach, position.z + candidateZ * reach))
                            continue;
                        // Closest to the current heading: turning is cheap, reversing
                        // is not, and a creature that takes the first free direction
                        // in array order makes every herd drift east.
                        const math::Fixed32 dot = candidateX * headingX + candidateZ * headingZ;
                        if (dot > bestDot)
                        {
                            bestDot = dot;
                            bestX = candidateX;
                            bestZ = candidateZ;
                            found = true;
                        }
                    }
                    if (found)
                    {
                        headingX = bestX;
                        headingZ = bestZ;
                        ++_avoided;
                    }
                }

                const math::Fixed32 stepX = headingX * pace;
                const math::Fixed32 stepZ = headingZ * pace;
                const math::Fixed32 tryX = position.x + stepX;
                const math::Fixed32 tryZ = position.z + stepZ;

                // Axes tested separately, DIAGONAL included: testing the two axes
                // apart and then moving along both walks the corner between two free
                // cells into the blocked one they share, which puts the body inside
                // the rock the next tick has to rescue it from.
                const bool freeX = _terrain.standable(tryX, position.z);
                const bool freeZ = _terrain.standable(position.x, tryZ);
                if (freeX && freeZ && _terrain.standable(tryX, tryZ))
                {
                    position.x = tryX;
                    position.z = tryZ;
                }
                else if (freeX)
                    position.x = tryX;
                else if (freeZ)
                    position.z = tryZ;
                else
                {
                    // Cornered: turn around rather than freeze. The HEADING is
                    // reversed, not the boid velocity — zeroing that destroys the very
                    // state the flocking rules accumulate, and a herd of those
                    // shudders in place.
                    headingX = math::Fixed32{} - headingX;
                    headingZ = math::Fixed32{} - headingZ;
                    ++_cornered;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CreaturePipeline — one statement of the order, two ways to drive it
// ─────────────────────────────────────────────────────────────────────────────

void CreaturePipeline::build(ecs::Registry &registry, LivingLayer &living, ITerrainQuery &terrain)
{
    // Constructed in CreatureStage order, and stored under their stage index, so
    // both drivers below get the order from the enumeration rather than from the
    // shape of the code around them.
    auto deposit = lpl::pmr::make_unique<ScentDepositSystem>(registry, living.scent().field(), living.herdParams(),
                                                            living.fieldView());
    _deposit = deposit.get();
    _stages[static_cast<core::u32>(CreatureStage::ScentDeposit)] = std::move(deposit);

    _stages[static_cast<core::u32>(CreatureStage::ScentField)] =
        lpl::pmr::make_unique<ScentFieldSystem>(living.scent().field(), living.recipe().stigmergy);

    auto steering = lpl::pmr::make_unique<ScentSteeringSystem>(registry, living.scent().field(), living.herdParams(),
                                                               living.fieldView());
    _steering = steering.get();
    _stages[static_cast<core::u32>(CreatureStage::ScentSteering)] = std::move(steering);

    auto flocking = lpl::pmr::make_unique<FlockingSystem>(registry, living.herdParams());
    _flocking = flocking.get();
    _stages[static_cast<core::u32>(CreatureStage::Flocking)] = std::move(flocking);

    auto grazing = lpl::pmr::make_unique<GrazingSystem>(registry, living.herdParams(), terrain);
    _grazing = grazing.get();
    _stages[static_cast<core::u32>(CreatureStage::Grazing)] = std::move(grazing);

    auto locomotion = lpl::pmr::make_unique<LocomotionSystem>(registry, living.herdParams(), terrain);
    _locomotion = locomotion.get();
    _stages[static_cast<core::u32>(CreatureStage::Locomotion)] = std::move(locomotion);
}

core::ExpectedVoid CreaturePipeline::registerOn(ecs::SystemScheduler &scheduler)
{
    for (core::u32 i = 0u; i < static_cast<core::u32>(CreatureStage::Count); ++i)
    {
        if (!_stages[i])
            continue;
        // The typed pointers survive this: ownership moves, the objects do not.
        if (auto registered = scheduler.registerSystem(std::move(_stages[i])); !registered)
            return registered;
    }
    return {};
}

void CreaturePipeline::step(core::f32 dt)
{
    for (core::u32 i = 0u; i < static_cast<core::u32>(CreatureStage::Count); ++i)
        if (_stages[i])
            _stages[i]->execute(dt);
}

void CreaturePipeline::setView(const CreatureFieldView &view) noexcept
{
    // Only two stages read the window; the others work in world units. Forwarding
    // to one and forgetting the other is how a herd deposits where it is and reads
    // the gradient somewhere else.
    if (_deposit != nullptr)
        _deposit->setView(view);
    if (_steering != nullptr)
        _steering->setView(view);
}

} // namespace lpl::engine::systems
