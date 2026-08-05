/**
 * @file CreatureSystems.hpp
 * @brief The living world, as systems over entities.
 *
 * Sixteen systems existed before these and not one of them was about anything
 * alive: physics, network, input, render — the engine's plumbing. Everything a
 * creature did lived in `samples/TerrainWorld.hpp`, which is 1952 lines, nearly
 * three quarters of all sample code. An animal was not an entity, its genome was
 * not a component, and none of it could be looked at through the registry, put in
 * a document, or asked for by an intelligence.
 *
 * The six of them are one tick of an animal's life, in the order the scheduler
 * derives from what each declares:
 *
 *   ScentDepositSystem   every creature marks the channel of its own kind
 *   ScentFieldSystem     the field evaporates and diffuses, once per tick
 *   ScentSteeringSystem  each creature reads its palate and steers
 *   FlockingSystem       velocity from the boid rules, per species
 *   GrazingSystem        a forager eats one plant where it stands
 *   LocomotionSystem     facing, pace from the genome, and the walk that terrain allows
 *
 * The last three are the ones that needed the ground. They get it through
 * engine::ITerrainQuery, which is a named interface rather than the lambdas the
 * sample used to hand down — see that header for why the difference is not
 * cosmetic. What is left in a World is generation, streaming, drawing and the
 * answers only it can give; what an animal DOES is here.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_CREATURESYSTEMS_HPP
#    define LPL_ENGINE_SYSTEMS_CREATURESYSTEMS_HPP

#    include <lpl/ai/StigmergyField.hpp>
#    include <lpl/ai/Swarm.hpp>
#    include <lpl/ecology/Genome.hpp>
#    include <lpl/ecology/Herd.hpp>
#    include <lpl/ecs/Partition.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/ecs/SystemScheduler.hpp>
#    include <lpl/engine/ITerrainQuery.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/std/memory.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::engine {
class LivingLayer;
}

namespace lpl::engine::systems {

/**
 * @struct CreatureFieldView
 * @brief How a creature's world position lands in the scent window.
 *
 * The one thing these systems cannot derive. A bounded map and a streamed one
 * place the same coordinate in different cells, and the window follows the
 * player, so the mapping moves. It is a small, explicit value rather than a
 * callback: an origin and a size are data, and data can be written down, folded
 * and replayed.
 */
struct CreatureFieldView {
    core::i32 originX{0}; ///< World cell the window's column 0 covers.
    core::i32 originZ{0};
    core::u32 width{0u};
    core::u32 depth{0u};

    /// @return false when @p worldX / @p worldZ fall outside the window.
    [[nodiscard]] bool toCell(core::i32 worldX, core::i32 worldZ, core::u32 &outX, core::u32 &outZ) const noexcept;
};

/**
 * @brief Visits every creature body in @p registry, one entity at a time.
 *
 * The per-entity counterpart of the per-chunk walk the systems below use internally.
 * Two granularities of one traversal, and the difference is deliberate: a system wants
 * the whole chunk's arrays because it touches every row, while a viewer or a diagnostic
 * wants one body at a time and does not care how they are stored. What is NOT
 * acceptable is a third copy of the walk, which is what a map viewer had.
 *
 * Asks the ARCHETYPE, never the pointer: @c readComponent answers for every component
 * id a chunk was built with, so a null test asks a different question than "does this
 * partition store creatures".
 *
 * Reads the WRITE side, like every creature system — that is where they write, and a
 * world whose systems never trigger a phase swap has nothing published on the other.
 *
 * @param registry Where the bodies live.
 * @param visit    Called with (species, identity, genome, position).
 */
template <typename Visit> void forEachCreature(const ecs::Registry &registry, Visit &&visit)
{
    for (const auto &partition : registry.partitions())
    {
        if (!partition || !partition->archetype().has(ecs::ComponentId::Position) ||
            !partition->archetype().has(ecs::ComponentId::Creature) ||
            !partition->archetype().has(ecs::ComponentId::Genome))
            continue;
        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            const auto *positions =
                static_cast<const math::Vec3<math::Fixed32> *>(chunk->writeComponent(ecs::ComponentId::Position));
            const auto *creature = static_cast<const core::u32 *>(chunk->writeComponent(ecs::ComponentId::Creature));
            const auto *genome = static_cast<const ecology::Genome *>(chunk->writeComponent(ecs::ComponentId::Genome));
            if (positions == nullptr || creature == nullptr || genome == nullptr)
                continue;
            for (core::u32 i = 0u; i < chunk->count(); ++i)
                visit(creature[i * 2u], creature[i * 2u + 1u], genome[i], positions[i]);
        }
    }
}

/**
 * @class ScentDepositSystem
 * @brief Every creature leaves the mark of its own kind on the field.
 *
 * Reads Position and Creature, writes the field — not a component, which is why
 * it declares only reads: two systems that both wrote Velocity would be a hazard
 * the scheduler must serialise, and this one touches no component at all.
 */
class ScentDepositSystem final : public ecs::ISystem {
public:
    ScentDepositSystem(ecs::Registry &registry, ai::StigmergyField &field, const ecology::HerdParams &params,
                       const CreatureFieldView &view);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

    /// The window moves with the player, so the view is refreshed, not fixed.
    void setView(const CreatureFieldView &view) noexcept { _view = view; }

    /// @return How many marks the last tick laid.
    [[nodiscard]] core::u32 deposits() const noexcept { return _deposits; }

private:
    ecs::Registry &_registry;
    ai::StigmergyField &_field;
    const ecology::HerdParams &_params;
    CreatureFieldView _view;
    core::u32 _deposits{0u};
};

/**
 * @class ScentFieldSystem
 * @brief One tick of evaporation and diffusion.
 *
 * Separate from the deposit on purpose, and scheduled after it: evaporating a
 * mark in the same tick it was laid is the classic way to make a scent field look
 * like noise. Stating the order as two systems makes the scheduler enforce what a
 * comment used to ask for.
 */
class ScentFieldSystem final : public ecs::ISystem {
public:
    ScentFieldSystem(ai::StigmergyField &field, const ai::StigmergyParams &params);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    ai::StigmergyField &_field;
    const ai::StigmergyParams &_params;
};

/**
 * @class ScentSteeringSystem
 * @brief Each creature reads what it is drawn to and what it flees, and turns.
 *
 * The behaviour a hard-coded channel index used to make impossible: a grazer and
 * a hunter standing on the same cell now read the SAME field and move in opposite
 * directions, because they weigh its channels differently.
 *
 * Adds an impulse to Velocity; it does not set a speed. Pace belongs to the
 * genome, and letting a chain of scent impulses accumulate into a bolt is exactly
 * what separating the two prevents.
 */
class ScentSteeringSystem final : public ecs::ISystem {
public:
    ScentSteeringSystem(ecs::Registry &registry, const ai::StigmergyField &field, const ecology::HerdParams &params,
                        const CreatureFieldView &view);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

    void setView(const CreatureFieldView &view) noexcept { _view = view; }

    /// @return How many creatures the last tick actually steered.
    [[nodiscard]] core::u32 steered() const noexcept { return _steered; }

private:
    ecs::Registry &_registry;
    const ai::StigmergyField &_field;
    const ecology::HerdParams &_params;
    CreatureFieldView _view;
    core::u32 _steered{0u};
};

/**
 * @class FlockingSystem
 * @brief Velocity from the boid rules, one flock per species.
 *
 * The flock's own integration is DISCARDED: only the velocities are taken back.
 * The boid rules decide where an animal WANTS to go; terrain decides where it may
 * stand, and taking the position back from the flock put bodies inside rock before
 * anything could refuse them.
 *
 * Registered after @ref ScentSteeringSystem, and the order is enforced rather than
 * hoped for: both declare Velocity read-write, so the scheduler puts an edge
 * between them. Running the other way round would let the boid pass overwrite the
 * scent impulse, and the pack would stop flanking.
 */
class FlockingSystem final : public ecs::ISystem {
public:
    FlockingSystem(ecs::Registry &registry, const ecology::HerdParams &params);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

    /// @return How many bodies the last tick flocked.
    [[nodiscard]] core::u32 flocked() const noexcept { return _flocked; }

private:
    ecs::Registry &_registry;
    const ecology::HerdParams &_params;
    /// Scratch, reused every tick: clear() keeps the capacity, so a steady herd
    /// allocates on the first tick and never again. test-tick-allocations is the
    /// instrument that makes that claim checkable rather than hopeful.
    lpl::pmr::vector<ai::Boid> _flock;
    lpl::pmr::vector<math::Vec3<math::Fixed32> *> _sinks;
    core::u32 _flocked{0u};
};

/**
 * @class GrazingSystem
 * @brief A forager eats one plant where it stands.
 *
 * Which species forages is read off its scent declaration (ecology::isHerbivore),
 * not off a species index: "species 0 is the grazer" is a convention several files
 * repeat and none enforces.
 *
 * Writes the vegetation, so it declares @c ResourceId::Vegetation read-write, and
 * reads Position — which is what orders it BEFORE @ref LocomotionSystem. An animal
 * eats where it stood at the start of the tick, not where the tick took it.
 */
class GrazingSystem final : public ecs::ISystem {
public:
    GrazingSystem(ecs::Registry &registry, const ecology::HerdParams &params, ITerrainQuery &terrain);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

    /// Meals taken on the LAST tick — transient, unlike the world's running total.
    [[nodiscard]] core::u32 meals() const noexcept { return _meals; }

private:
    ecs::Registry &_registry;
    const ecology::HerdParams &_params;
    ITerrainQuery &_terrain;
    core::u32 _meals{0u};
};

/**
 * @class LocomotionSystem
 * @brief Facing from the velocity, pace from the genome, and the walk terrain allows.
 *
 * Not engine::systems::MovementSystem, which turns a player's INPUT into a
 * velocity. This is the other half of the same idea for something with no player:
 * the direction comes from the flock and the scent, and the PACE comes from the
 * genome and the personality — never from the velocity's magnitude. That separation
 * is what keeps a chain of scent impulses from accumulating into a bolt.
 *
 * Three rules in the order they were each learned from a picture, all three merged
 * in from the map viewer's own copy of this loop — which had been fixed further
 * than the engine's while nobody noticed there were two:
 *
 *  - **Containment first.** A body standing somewhere unstandable refuses every
 *    direction, so it refuses two steps a tick forever. It asks the terrain which
 *    way is out (@ref ITerrainQuery::recoveryDirection) instead of grinding.
 *  - **Avoidance BEFORE the move, not after refusing it.** Steering around a rock
 *    and being stopped by one look identical in one frame and nothing alike over a
 *    second. It looks a full body-length ahead and, if that is blocked, takes the
 *    free 8-neighbour CLOSEST to where it was already going — turning is cheap,
 *    reversing is not, and picking the first free direction in array order makes
 *    every herd drift east.
 *  - **The terrain disposes, one axis at a time**, diagonal included: testing two
 *    axes separately and then moving along both walks the corner between two free
 *    cells into the blocked one they share.
 */
class LocomotionSystem final : public ecs::ISystem {
public:
    LocomotionSystem(ecs::Registry &registry, const ecology::HerdParams &params, const ITerrainQuery &terrain);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

    /// @return How many bodies the terrain refused outright on the last tick.
    [[nodiscard]] core::u32 cornered() const noexcept { return _cornered; }

    /// @return How many steered around an obstacle rather than into it.
    [[nodiscard]] core::u32 avoided() const noexcept { return _avoided; }

    /// @return How many were standing somewhere unstandable and had to be recovered.
    [[nodiscard]] core::u32 strays() const noexcept { return _strays; }

private:
    ecs::Registry &_registry;
    const ecology::HerdParams &_params;
    const ITerrainQuery &_terrain;
    core::u32 _cornered{0u};
    core::u32 _avoided{0u};
    core::u32 _strays{0u};
};

/**
 * @enum CreatureStage
 * @brief One animal's tick, in the order it must happen. Stated ONCE.
 *
 * The order was written down twice before this existed — as a registration
 * sequence in the sample World, and as six @c execute calls in the map viewer —
 * and a third copy was about to appear in the editor. Two of the three would have
 * been right; the interesting part is that a wrong one is nearly invisible. Put
 * flocking before steering and the boid pass overwrites the scent impulse: the
 * pack simply stops flanking, with no error and nothing in a signature to say why.
 *
 * Both drivers below read the order from this enumeration, so there is one place
 * to be right about.
 */
enum class CreatureStage : core::u8 {
    ScentDeposit = 0, ///< Every creature marks the channel of its own kind.
    ScentField,       ///< The field evaporates and diffuses — after the marks, never with them.
    ScentSteering,    ///< Each creature reads its palate and turns.
    Flocking,         ///< The boid rules, per species, on top of that impulse.
    Grazing,          ///< A forager eats where it STOOD, so before it moves.
    Locomotion,       ///< Facing, pace, and the walk terrain allows.
    Count
};

/**
 * @class CreaturePipeline
 * @brief The six systems of one animal's tick, built and ordered together.
 *
 * Two ways to drive them, because there are genuinely two kinds of caller:
 *  - @ref registerOn hands them to a scheduler, which is what a World does. The
 *    DAG then *enforces* the order rather than trusting it: within a phase the
 *    earlier-registered system wins a conflict, and every consecutive pair here
 *    shares a declared dependency, so each edge is real.
 *  - @ref step executes them in order, for a tool that has no World and no
 *    scheduler — a viewer or an editor panel watching the layer run.
 *
 * Ownership transfers on @ref registerOn, but the systems themselves do not move,
 * so the typed accessors stay valid: that is what lets a caller keep pushing a
 * fresh @ref CreatureFieldView into the two systems that need one.
 */
class CreaturePipeline {
public:
    /**
     * @brief Builds all six against @p living, in @ref CreatureStage order.
     *
     * Every constructor argument comes from the layer or from the terrain, which
     * is why this takes two objects rather than eight references: a caller cannot
     * hand one system a different herd from another's.
     *
     * @param registry Where the bodies live.
     * @param living   The layer that owns the field, the herd parameters and the recipe.
     * @param terrain  What the ground answers. Must outlive the pipeline.
     */
    void build(ecs::Registry &registry, LivingLayer &living, ITerrainQuery &terrain);

    /// Hands every stage to @p scheduler in order. @return an error if one is refused.
    [[nodiscard]] core::ExpectedVoid registerOn(ecs::SystemScheduler &scheduler);

    /// Executes every stage in order, for a caller with no scheduler.
    void step(core::f32 dt);

    /// Pushes a moved scent window into the two stages that read one.
    void setView(const CreatureFieldView &view) noexcept;

    [[nodiscard]] bool built() const noexcept { return _deposit != nullptr; }

    /// @return The deposit stage, or nullptr before @ref build.
    [[nodiscard]] ScentDepositSystem *deposit() const noexcept { return _deposit; }
    [[nodiscard]] ScentSteeringSystem *steering() const noexcept { return _steering; }
    [[nodiscard]] FlockingSystem *flocking() const noexcept { return _flocking; }
    [[nodiscard]] GrazingSystem *grazing() const noexcept { return _grazing; }
    [[nodiscard]] LocomotionSystem *locomotion() const noexcept { return _locomotion; }

private:
    /// Indexed by @ref CreatureStage, which IS the order.
    lpl::pmr::unique_ptr<ecs::ISystem> _stages[static_cast<core::u32>(CreatureStage::Count)];

    ScentDepositSystem *_deposit{nullptr};
    ScentSteeringSystem *_steering{nullptr};
    FlockingSystem *_flocking{nullptr};
    GrazingSystem *_grazing{nullptr};
    LocomotionSystem *_locomotion{nullptr};
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_CREATURESYSTEMS_HPP
