/**
 * @file HistorySystem.hpp
 * @brief The ECS system that applies constraints each step.
 *
 * Registered in the Physics/Logic phases like any other system, so it obeys the
 * same scheduler contract and the same zero-unbounded-allocation rule.
 *
 * It is a system rather than a hook on the World for the reason CubePileStepSystem
 * was: a step written by hand inside a World cannot be ordered against what it
 * touches, cannot be given a fake in a test, and cannot declare what it depends on.
 * A constraint that seeds a settlement has to run before whatever grows settlements,
 * and only the scheduler can be told that.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_HISTORYSYSTEM_HPP
#    define LPL_LPL_HISTORY_HISTORYSYSTEM_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/history/Chronicle.hpp>
#    include <lpl/history/Place.hpp>
#    include <lpl/history/Predicate.hpp>
#    include <lpl/history/Era.hpp>
#    include <lpl/history/Timeline.hpp>

namespace lpl::history {

/**
 * @class HistorySystem
 * @brief Applies the timeline's constraints as the era advances.
 *
 * Holds no world state of its own: the timeline is what it reads, the chronicle is
 * what it writes, and both are the caller's. A system that owned either would be a
 * second place where a run's history lives.
 */
class HistorySystem final : public ecs::ISystem {
public:
    /**
     * @brief Binds the timeline, the era and the chronicle this system works on.
     * @param timeline  The constraints to honour.
     * @param era       The gearing between ticks and years.
     * @param chronicle Where events are recorded.
     */
    HistorySystem(const Timeline &timeline, const Era &era, Chronicle &chronicle) noexcept
        : _timeline(&timeline), _era(era), _chronicle(&chronicle)
    {
    }

    /**
     * @brief Lets the constraints ACT, rather than only be recorded.
     *
     * @warning Without this the system was inert in the way that matters: it wrote a chronicle and
     * touched no entity, so a `Seed` made nobody exist and a `Force` moved nothing. The corpus
     * could say Herodotus was born at Halicarnassus and no body in the world was him.
     *
     * @warning Optional, and the two-argument constructor above is kept, because gate P13 runs this
     * system with no world at all — a timeline of claims about a king and two settlements needs
     * no registry, and requiring one would have made a folded signature depend on a world.
     *
     * @param registry  Where bodies are made.
     * @param resolver  Where places are.
     * @param archetype What a seeded body carries.
     */
    void bindWorld(ecs::Registry &registry, const IPlaceResolver &resolver,
                   const ecs::Archetype &archetype) noexcept;

    /**
     * @brief How many bodies the constraints brought into the world.
     *
     * @return The count.
     */
    [[nodiscard]] core::u32 seeded() const noexcept { return _seeded; }

    /**
     * @brief How many times a constraint moved a body to where it says it was.
     *
     * @return The count.
     */
    [[nodiscard]] core::u32 forced() const noexcept { return _forced; }

    /**
     * @brief Constraints that named a place the gazetteer could not put anywhere.
     *
     * @warning Counted rather than ignored: a corpus claim about a place nobody has located is a real
     * and common thing, and a run that quietly dropped those would look like a run whose corpus
     * said less than it does.
     *
     * @return The count.
     */
    [[nodiscard]] core::u32 unplaceable() const noexcept { return _unplaceable; }

    /**
     * @brief The body standing for a subject, if one was seeded.
     *
     * @param subject The corpus identifier.
     * @param out     Receives the entity.
     * @return false when nothing stands for it.
     */
    [[nodiscard]] bool bodyOf(core::u32 subject, ecs::EntityId &out) const noexcept;

    /**
     * @brief Advances one tick, applying whatever this year carries.
     * @param dt Ignored: an era's clock is ticks, not seconds.
     */
    void execute(core::f32 dt) override;

    /**
     * @brief What this system reads and writes.
     * @return Its descriptor.
     */
    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;

    /**
     * @brief Ticks retired so far.
     * @return The count.
     */
    [[nodiscard]] core::u32 tick() const noexcept { return _tick; }

    /**
     * @brief Constraints applied so far.
     * @return The count.
     */
    [[nodiscard]] core::u32 applied() const noexcept { return _applied; }

private:
    /**
     * @brief Makes or finds the body standing for a subject.
     *
     * @param subject The corpus identifier.
     * @param fact    The claim being applied, for its dates.
     * @param out     Receives the entity.
     * @return false when no body could be made.
     */
    [[nodiscard]] bool bodyFor(core::u32 subject, const Fact &fact, ecs::EntityId &out,
                               bool &outCreated);

    /**
     * @brief Puts a body at a place.
     *
     * @param body  The entity.
     * @param place Where.
     */
    void placeBody(ecs::EntityId body, const Place &place);

    struct Binding {
        core::u32 subject{0u};
        ecs::EntityId body{};
    };

    ecs::Registry *_registry{nullptr};
    const IPlaceResolver *_resolver{nullptr};
    ecs::Archetype _archetype{};
    static constexpr core::u32 kMaxBodies = 64u;
    Binding _bindings[kMaxBodies]{};
    core::u32 _bindingCount{0u};
    core::u32 _seeded{0u};
    core::u32 _forced{0u};
    core::u32 _unplaceable{0u};

    const Timeline *_timeline{nullptr};
    Era _era{};
    Chronicle *_chronicle{nullptr};
    core::u32 _tick{0u};
    core::u32 _applied{0u};
};

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_HISTORYSYSTEM_HPP
