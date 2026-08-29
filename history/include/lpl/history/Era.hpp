/**
 * @file Era.hpp
 * @brief A span of days, and the rate the simulation crosses it at.
 *
 * @warning **The gearing and the RESOLUTION are different things, and conflating them cost the corpus
 * its precision.** This used to be a span of years crossed at `ticksPerYear`, which quietly made
 * the year both the clock's step and the finest thing a fact could say -- so a source that knew
 * a date to the day had nowhere to put it. They are now separate: @ref Fact speaks days, and
 * @ref daysPerTick says how fast this era is crossed.
 *
 * **Slow down, speed up, skip -- as a sequence of eras rather than a schedule.** A documented
 * decade is an era with `daysPerTick` of 1; a silent century is an era with `daysPerTick` of
 * 36525. Keeping the rate uniform WITHIN an era is what leaves @ref dayOfTick a closed form, so
 * two targets agree on which day a tick is without replaying the run to find out.
 *
 * @warning **There is no "boundary" any more, and that is the point.** Constraints used to fire when
 * the current year EQUALLED a fact's year, which is correct only while the clock advances one
 * year at a time: skip a century and every constraint inside it is silently never applied, with
 * a chronicle that looks complete. Every tick now COVERS a span (@ref spanOfTick), and nothing
 * can fall between two of them.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_ERA_HPP
#    define LPL_LPL_HISTORY_ERA_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/history/Calendar.hpp>

namespace lpl::history {

/**
 * @struct Era
 * @brief A span of days, and the rate the simulation crosses it at.
 */
struct Era {
    core::i32 startDay{0};     ///< First day simulated, inclusive.
    core::i32 endDay{0};       ///< Last day simulated, inclusive.
    core::u32 daysPerTick{1u}; ///< Days one scheduler step advances. 1 is day by day.

    /**
     * @brief Builds an era from whole years, at a rate given in ticks per year.
     *
     * @warning A convenience for the many callers whose corpus really is year-resolution, and NOT a
     * second representation: it converts once and stores days like everything else. Writing the
     * old fields alongside the new ones would be two answers to "when does this era start".
     *
     * @param firstYear   First year simulated, inclusive.
     * @param lastYear    Last year simulated, inclusive.
     * @param ticksPerYear Steps per year; zero is treated as one.
     * @return The era.
     */
    [[nodiscard]] static constexpr Era ofYears(core::i32 firstYear, core::i32 lastYear,
                                               core::u32 ticksPerYear) noexcept
    {
        Era era;
        era.startDay = firstDayOfYear(firstYear);
        era.endDay = lastDayOfYear(lastYear);
        // 365 rather than 365.2425: an era's gearing is a rate, not a date, so it may be
        // approximate -- and it must be an integer, because a fractional step would make the day
        // a tick lands on depend on how the division rounded on that target.
        era.daysPerTick = ticksPerYear == 0u ? 365u : (365u / ticksPerYear == 0u ? 1u : 365u / ticksPerYear);
        return era;
    }

    /**
     * @brief Days the era covers.
     *
     * @return endDay - startDay + 1, or 0 when the span is empty.
     */
    [[nodiscard]] constexpr core::u32 days() const noexcept
    {
        return endDay < startDay ? 0u : static_cast<core::u32>(endDay - startDay) + 1u;
    }

    /**
     * @brief Ticks the whole era takes.
     *
     * @warning Rounded UP, never down: a final tick that covers only part of a step still has to
     * happen, or the last few days of every era go unsimulated and whatever a source dated to
     * them never fires.
     *
     * @return The count.
     */
    [[nodiscard]] constexpr core::u32 totalTicks() const noexcept
    {
        const core::u32 step = daysPerTick == 0u ? 1u : daysPerTick;
        return (days() + step - 1u) / step;
    }

    /**
     * @brief Which day a tick starts on.
     *
     * @param tick Index from the start of the era.
     * @return The day; clamped to the era when the tick is past its end.
     */
    [[nodiscard]] constexpr core::i32 dayOfTick(core::u32 tick) const noexcept
    {
        const core::u32 step = daysPerTick == 0u ? 1u : daysPerTick;
        const core::i64 day = static_cast<core::i64>(startDay) + static_cast<core::i64>(tick) * step;
        return day > endDay ? endDay : static_cast<core::i32>(day);
    }

    /**
     * @brief The span of days a tick covers.
     *
     * @warning **What replaces the year boundary.** A tick is not an instant, it is the stretch of
     * time the simulation crossed while taking it, and a constraint fires when its window starts
     * inside that stretch. Nothing can be skipped, because consecutive spans are contiguous by
     * construction -- and nothing fires twice, because a window has exactly one start.
     *
     * @param tick   Index from the start of the era.
     * @param outFrom Receives the first day covered.
     * @param outTo   Receives the last day covered, inclusive.
     */
    constexpr void spanOfTick(core::u32 tick, core::i32 &outFrom, core::i32 &outTo) const noexcept
    {
        const core::u32 step = daysPerTick == 0u ? 1u : daysPerTick;
        const core::i64 from = static_cast<core::i64>(startDay) + static_cast<core::i64>(tick) * step;
        const core::i64 to = from + static_cast<core::i64>(step) - 1;
        outFrom = from > endDay ? endDay : static_cast<core::i32>(from);
        outTo = to > endDay ? endDay : static_cast<core::i32>(to);
    }

    /**
     * @brief The first tick whose span contains a day.
     *
     * @param day A day, clamped to the era.
     * @return The tick index.
     */
    [[nodiscard]] constexpr core::u32 firstTickOfDay(core::i32 day) const noexcept
    {
        const core::u32 step = daysPerTick == 0u ? 1u : daysPerTick;
        const core::i32 clamped = day < startDay ? startDay : (day > endDay ? endDay : day);
        return static_cast<core::u32>(clamped - startDay) / step;
    }

    /**
     * @brief Which year a tick falls in.
     *
     * Kept because plenty of a chronicle's readers think in years and always will; it is derived
     * from the day rather than stored beside it, so the two cannot disagree.
     *
     * @param tick Index from the start of the era.
     * @return The year.
     */
    [[nodiscard]] constexpr core::i32 yearOfTick(core::u32 tick) const noexcept
    {
        return yearOfDay(dayOfTick(tick));
    }
};

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_ERA_HPP
