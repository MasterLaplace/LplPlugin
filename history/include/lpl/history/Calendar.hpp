/**
 * @file Calendar.hpp
 * @brief Days, because a year throws away what the sources actually say.
 *
 * @warning **The unit was the defect, and it was in the CORPUS rather than in the clock.** @ref Fact
 * carried `fromYear`/`toYear`, so a diary entry dated 18 June 1815 was stored as "1815" and the
 * day was discarded at ingestion, irreversibly -- no gearing of the simulation could get it back.
 * Ancient material genuinely is year-resolution; modern material is not, and a type that cannot
 * tell them apart makes every consumer treat a ship's log and a chronicle as equally precise.
 *
 * **Precision needs no new field: it is the WIDTH of the interval @ref Fact already carries.**
 * In days, "1815" is [1 Jan, 31 Dec] -- 365 wide, which is exactly what "we only know the year"
 * means -- and "18 June 1815" is one day. That is why this is a change of unit and not a change
 * of structure.
 *
 * **Proleptic Gregorian, day 0 = 1 January 1 CE.** Julian Day Numbers are the interchange
 * standard and would work equally well, but their offset turns every fixture in this repository
 * into a seven-digit constant; with this epoch the SIGN still means what the year's sign meant,
 * so a date reads the way the corpus reads. @ref julianDayNumber converts, so interop costs one
 * addition.
 *
 * @warning **A stored value is a DAY, not a rendering, and the calendar that rendered it is the
 * ingester's problem.** Sources before 1582 print Julian dates, which differ from proleptic
 * Gregorian by ten days in 1582 and by more the further back one goes. Converting a Julian date
 * with @ref dayOfDate silently shifts it. That is a real historiographic hazard, so it is named
 * here rather than discovered later: an ingester that knows its source is Julian must convert,
 * and one that does not know must say the window is wide rather than guess it is narrow.
 *
 * Integer throughout, no libm, freestanding: this is corpus arithmetic and it runs in ring 0.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_CALENDAR_HPP
#    define LPL_LPL_HISTORY_CALENDAR_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::history {

/**
 * Days from this epoch to 1 January 1970, the origin the conversion below is stated for.
 *
 * Only used to convert to and from anything that speaks Unix days; the shift the algorithm
 * itself needs is @ref kEraShift, and confusing the two is exactly the mistake this file's test
 * caught on its first run.
 */
inline constexpr core::i32 kUnixEpochDay = 719162;

/**
 * The shift that moves the algorithm's origin onto 1 January 1 CE.
 *
 * @warning Two different constants live here and they are one apart in NAME only. The day-count
 * algorithm is written around 719468 -- days from the shifted era's origin (1 March of year 0,
 * which is what makes the leap day the last day of a shifted year and removes every special
 * case) to 1 January 1970. Moving from there to 1 January 1 CE means also stepping back over
 * the 719162 days between 1 CE and 1970, so the net shift is their DIFFERENCE. Substituting one
 * for the other, which a first version did, moves every date in the corpus by 306 days.
 */
inline constexpr core::i32 kEraShift = 719468 - kUnixEpochDay;

/**
 * Day number the Gregorian calendar was adopted (15 October 1582).
 *
 * Exposed so an ingester can ASK rather than hard-code the boundary it has to reason about. See
 * the file header: a source printing a date before this is printing a Julian one.
 */
inline constexpr core::i32 kGregorianAdoptionDay = 577735;

/**
 * Offset from this epoch to Julian Day Number.
 *
 * JDN 1721426 is 1 January 1 CE, so a day number here plus this is a JDN.
 */
inline constexpr core::i32 kJulianDayNumberOfEpoch = 1721426;

/**
 * @brief Whether a year has 366 days in the proleptic Gregorian calendar.
 *
 * @param year The year; negative is BCE, and there IS a year zero here (1 BCE), because the
 *             arithmetic below needs one and astronomers use the same convention.
 * @return true when it is a leap year.
 */
[[nodiscard]] constexpr bool isLeapYear(core::i32 year) noexcept
{
    return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
}

/**
 * @brief The day number of a calendar date.
 *
 * Exact integer arithmetic over the whole range, negative years included. The shift by two
 * months makes the leap day the LAST day of the shifted year, which is what removes every
 * special case from the day count -- the trick is Hinnant's and it is used here rather than
 * re-derived because a hand-rolled calendar is a well-known way to be wrong once a century.
 *
 * @param year  The year; negative is BCE with a year zero.
 * @param month 1..12. Out-of-range values are NOT clamped: a caller that has month 13 has a
 *              parsing bug, and clamping it would turn that into a plausible date.
 * @param day   1..31, same rule.
 * @return Days since 1 January 1 CE; negative before it.
 */
[[nodiscard]] constexpr core::i32 dayOfDate(core::i32 year, core::u32 month, core::u32 day) noexcept
{
    const core::i32 shifted = year - (month <= 2u ? 1 : 0);
    const core::i32 era = (shifted >= 0 ? shifted : shifted - 399) / 400;
    const core::u32 yearOfEra = static_cast<core::u32>(shifted - era * 400);
    const core::u32 dayOfYear =
        (153u * (month + (month > 2u ? -3u : 9u)) + 2u) / 5u + day - 1u;
    const core::u32 dayOfEra = yearOfEra * 365u + yearOfEra / 4u - yearOfEra / 100u + dayOfYear;
    return era * 146097 + static_cast<core::i32>(dayOfEra) - kEraShift;
}

/**
 * @brief The first day of a year.
 *
 * @param year The year.
 * @return Its 1 January.
 */
[[nodiscard]] constexpr core::i32 firstDayOfYear(core::i32 year) noexcept
{
    return dayOfDate(year, 1u, 1u);
}

/**
 * @brief The last day of a year.
 *
 * @warning Computed as the day before the next year's first, never as "first + 364": that would be
 * wrong in every leap year, which is one year in four -- often enough to look like a rounding
 * quirk rather than a bug.
 *
 * @param year The year.
 * @return Its 31 December.
 */
[[nodiscard]] constexpr core::i32 lastDayOfYear(core::i32 year) noexcept
{
    return dayOfDate(year + 1, 1u, 1u) - 1;
}

/**
 * @brief The year a day falls in.
 *
 * @param day Day number.
 * @return The year; negative before 1 CE.
 */
[[nodiscard]] constexpr core::i32 yearOfDay(core::i32 day) noexcept
{
    const core::i32 shifted = day + kEraShift;
    const core::i32 era = (shifted >= 0 ? shifted : shifted - 146096) / 146097;
    const core::u32 dayOfEra = static_cast<core::u32>(shifted - era * 146097);
    const core::u32 yearOfEra =
        (dayOfEra - dayOfEra / 1460u + dayOfEra / 36524u - dayOfEra / 146096u) / 365u;
    const core::i32 candidate = static_cast<core::i32>(yearOfEra) + era * 400;
    const core::u32 dayOfYear =
        dayOfEra - (365u * yearOfEra + yearOfEra / 4u - yearOfEra / 100u);
    // The shifted year runs March to February, so January and February belong to the next one.
    const core::u32 shiftedMonth = (5u * dayOfYear + 2u) / 153u;
    return shiftedMonth >= 10u ? candidate + 1 : candidate;
}

/**
 * @brief The Julian Day Number of a day.
 *
 * @param day Day number in this epoch.
 * @return The JDN, for interchange with anything that speaks it.
 */
[[nodiscard]] constexpr core::i32 julianDayNumber(core::i32 day) noexcept
{
    return day + kJulianDayNumberOfEpoch;
}

/**
 * @brief Whether two closed day intervals share any day.
 *
 * @warning **The test that makes a variable clock safe.** A constraint used to fire when the current
 * year EQUALLED a fact's year, which is correct only while the clock advances one year at a
 * time. The moment an era is allowed to skip a silent century -- which is the whole point of
 * having a gearing -- every constraint inside the skipped span is simply never applied: not
 * slower or faster, but quietly missing, with a chronicle that looks complete. Asking whether
 * the step CROSSED the window instead of landing on it has no such failure.
 *
 * @param firstFrom  Start of the first interval, inclusive.
 * @param firstTo    End of the first interval, inclusive.
 * @param secondFrom Start of the second.
 * @param secondTo   End of the second.
 * @return true when they overlap.
 */
[[nodiscard]] constexpr bool daysOverlap(core::i32 firstFrom, core::i32 firstTo, core::i32 secondFrom,
                                         core::i32 secondTo) noexcept
{
    return firstFrom <= secondTo && secondFrom <= firstTo;
}

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_CALENDAR_HPP
