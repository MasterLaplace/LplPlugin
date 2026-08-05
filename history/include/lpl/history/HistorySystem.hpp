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

#    include <lpl/ecs/System.hpp>
#    include <lpl/history/Chronicle.hpp>
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
    const Timeline *_timeline{nullptr};
    Era _era{};
    Chronicle *_chronicle{nullptr};
    core::u32 _tick{0u};
    core::u32 _applied{0u};
};

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_HISTORYSYSTEM_HPP
