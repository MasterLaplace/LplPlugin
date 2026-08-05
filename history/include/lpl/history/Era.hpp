/**
 * @file Era.hpp
 * @brief The mapping between simulation ticks and historical time.
 *
 * A century cannot be simulated at 60 Hz. An era declares how many years a tick
 * carries and which systems are stepped at that scale, so the same world can be
 * walked at human speed and fast-forwarded across centuries without two codebases.
 *
 * Ticks per YEAR rather than years per tick, even though the fast-forward case wants
 * the second: an integer number of ticks per year keeps a year boundary landing
 * exactly on a tick, and a constraint dated to a year has to be applied on a tick or
 * it is applied on neither side of one.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_ERA_HPP
#    define LPL_LPL_HISTORY_ERA_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::history {

/**
 * @struct Era
 * @brief A span of years, and the rate the simulation crosses it at.
 */
struct Era {
    core::i32 startYear{0};     ///< First year simulated, inclusive.
    core::i32 endYear{0};       ///< Last year simulated, inclusive.
    core::u32 ticksPerYear{1u}; ///< Steps the scheduler takes per year.

    /**
     * @brief Years the era covers.
     * @return endYear - startYear + 1, or 0 when the span is empty.
     */
    [[nodiscard]] core::u32 years() const noexcept
    {
        return endYear < startYear ? 0u : static_cast<core::u32>(endYear - startYear) + 1u;
    }

    /**
     * @brief Ticks the whole era takes.
     * @return years() * ticksPerYear.
     */
    [[nodiscard]] core::u32 totalTicks() const noexcept { return years() * ticksPerYear; }

    /**
     * @brief Which year a tick falls in.
     * @param tick Index from the start of the era.
     * @return The year; clamped to the era when the tick is past its end.
     */
    [[nodiscard]] core::i32 yearOfTick(core::u32 tick) const noexcept
    {
        if (ticksPerYear == 0u)
            return startYear;
        const core::i32 offset = static_cast<core::i32>(tick / ticksPerYear);
        const core::i32 year = startYear + offset;
        return year > endYear ? endYear : year;
    }

    /**
     * @brief The first tick of a year.
     * @param year A year, clamped to the era.
     * @return The tick index.
     */
    [[nodiscard]] core::u32 firstTickOfYear(core::i32 year) const noexcept
    {
        const core::i32 clamped = year < startYear ? startYear : (year > endYear ? endYear : year);
        return static_cast<core::u32>(clamped - startYear) * ticksPerYear;
    }

    /**
     * @brief Is @p tick the first one of its year?
     *
     * The question a constraint asks: a fact dated to a year must be applied once, on
     * a definite tick, or two runs at different rates would apply it a different
     * number of times and fold differently.
     *
     * @param tick Index from the start of the era.
     * @return true on a year boundary.
     */
    [[nodiscard]] bool isYearBoundary(core::u32 tick) const noexcept
    {
        return ticksPerYear != 0u && (tick % ticksPerYear) == 0u;
    }
};

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_ERA_HPP
