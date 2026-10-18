#include <lpl/history/Calendar.hpp>

#include <cstdio>

namespace {

int gChecks = 0;
int gFailures = 0;

/**
 * @brief Records one assertion.
 *
 * @param what Description.
 * @param ok   Whether it held.
 */
void check(const char *what, bool ok)
{
    ++gChecks;
    if (ok)
        return;
    ++gFailures;
    std::printf("  (fail) %s\n", what);
}

/**
 * @brief Anchors, computed elsewhere: absolute day numbers and their Julian Day Numbers.
 *
 * @details The interchange constant is checked the way it is used rather than restated.
 */
void checkAnchorsComputedElsewhere()
{
    std::printf("-- anchors, computed elsewhere\n");
    check("1 January 1 CE is day zero", lpl::history::dayOfDate(1, 1u, 1u) == 0);
    check("1 January 1970 is day 719162", lpl::history::dayOfDate(1970, 1u, 1u) == 719162);
    check("15 October 1582 is day 577735",
          lpl::history::dayOfDate(1582, 10u, 15u) == lpl::history::kGregorianAdoptionDay);
    check("18 June 1815 is day 662717", lpl::history::dayOfDate(1815, 6u, 18u) == 662717);
    check("1 January 1815 is day 662549", lpl::history::dayOfDate(1815, 1u, 1u) == 662549);
    check("31 December 1815 is day 662913", lpl::history::dayOfDate(1815, 12u, 31u) == 662913);
    check("day zero is Julian Day Number 1721426", lpl::history::julianDayNumber(0) == 1721426);
    check("and 1 January 1970 is JDN 2440588",
          lpl::history::julianDayNumber(lpl::history::dayOfDate(1970, 1u, 1u)) == 2440588);
}

/**
 * @brief Precision is the WIDTH of the interval, which is the whole point.
 *
 * @details What "we only know the year" means, said in the unit the type now carries. The distinction the old unit
 *          could not express at all: a precise day and its vague year were both "1815".
 */
void checkPrecisionIsTheWidthOfTheInterval()
{
    std::printf("-- precision is the WIDTH of the interval, which is the whole point\n");
    const lpl::core::i32 yearFrom = lpl::history::firstDayOfYear(1815);
    const lpl::core::i32 yearTo = lpl::history::lastDayOfYear(1815);
    check("a year-resolution claim spans 365 days", yearTo - yearFrom + 1 == 365);

    const lpl::core::i32 exact = lpl::history::dayOfDate(1815, 6u, 18u);
    check("a day-resolution claim spans one", exact - exact + 1 == 1);
    check("and the precise day lies inside the vague year", lpl::history::daysOverlap(exact, exact, yearFrom, yearTo));
    check("while a different year does not",
          !lpl::history::daysOverlap(exact, exact, lpl::history::firstDayOfYear(1816),
                                     lpl::history::lastDayOfYear(1816)));
}

/**
 * @brief Leap years, including the ones that catch a hand-rolled calendar.
 *
 * @details The length of a year is computed as the day before the next new year, never as first + 364: that is
 *          wrong in one year out of four, often enough to read as a rounding quirk rather than a bug.
 */
void checkLeapYearsIncludingTheOnesThatCatchAHandRolledCalendar()
{
    std::printf("-- leap years, including the ones that catch a hand-rolled calendar\n");
    check("1900 is not a leap year", !lpl::history::isLeapYear(1900));
    check("2000 is", lpl::history::isLeapYear(2000));
    check("2024 is", lpl::history::isLeapYear(2024));
    check("2023 is not", !lpl::history::isLeapYear(2023));
    check("a leap year is 366 days long",
          lpl::history::lastDayOfYear(2000) - lpl::history::firstDayOfYear(2000) + 1 == 366);
    check("and 29 February exists in it",
          lpl::history::dayOfDate(2000, 2u, 29u) == lpl::history::dayOfDate(2000, 3u, 1u) - 1);
}

/**
 * @brief Structure, over a range no calendar tool will anchor.
 *
 * @details There is a year zero here (1 BCE) because the arithmetic needs one and astronomers use the same
 *          convention; an ingester that reads "44 BCE" off a source must write -43.
 *
 * The sign still means what the year's sign meant, which is why this epoch was chosen over Julian Day Numbers: a
 * date reads the way the corpus reads.
 *
 * @warning The range is DERIVED from the corpus, not chosen. A first version swept -3000 to +2200 on the assumption
 *          that a historical corpus starts around the first cities -- and the data said otherwise: Pleiades dates
 *          14 places at or before 100 000 BCE and its earliest, Franchthi Cave among them, at 2 600 000 BCE. Places
 *          have Palaeolithic occupation layers, so a gazetteer of them reaches into deep prehistory whatever a
 *          library of TEXTS does. Sweeping only the era I expected would have left the arithmetic that actually
 *          runs on the corpus untested.
 */
void checkStructureOverARangeNoCalendarToolWillAnchor()
{
    std::printf("-- structure, over a range no calendar tool will anchor\n");
    bool everyGapMatchesTheLeapRule = true;
    bool everyNewYearReportsItsYear = true;
    bool everyLastDayReportsItsYear = true;
    lpl::core::i32 checked = 0;

    for (lpl::core::i32 year = -2600000; year <= 2200; ++year)
    {
        const lpl::core::i32 first = lpl::history::firstDayOfYear(year);
        const lpl::core::i32 next = lpl::history::firstDayOfYear(year + 1);
        const lpl::core::i32 expected = lpl::history::isLeapYear(year) ? 366 : 365;
        if (next - first != expected)
            everyGapMatchesTheLeapRule = false;
        if (lpl::history::yearOfDay(first) != year)
            everyNewYearReportsItsYear = false;
        if (lpl::history::yearOfDay(lpl::history::lastDayOfYear(year)) != year)
            everyLastDayReportsItsYear = false;
        ++checked;
    }

    check("2 602 201 consecutive years were exercised", checked == 2602201);
    check("each is exactly as long as the leap rule says", everyGapMatchesTheLeapRule);
    check("every new year reports the year it was built from", everyNewYearReportsItsYear);
    check("and so does every 31 December", everyLastDayReportsItsYear);
    check("dates before 1 CE are negative", lpl::history::firstDayOfYear(-44) < 0);
    check("and dates after it are not", lpl::history::firstDayOfYear(44) > 0);
}

/**
 * @brief How much of the range the corpus actually needs.
 *
 * @details Roughly 5.88 million years each way: the year whose first day still fits.
 *
 * @warning A day number is i32, so this format spans about 5.88 million years either side of 1 CE. That is not a
 *          comfortable margin dressed up as a limit -- it is the limit, and the deepest date the corpus carries
 *          today uses 44 % of it. Stated here so the day somebody wants geological time, the refusal is a known
 *          bound rather than a signature that quietly stops matching.
 */
void checkHowMuchOfTheRangeTheCorpusActuallyNeeds()
{
    std::printf("-- how much of the range the corpus actually needs\n");
    const lpl::core::i32 deepest = lpl::history::firstDayOfYear(-2600000);
    check("the deepest date the corpus carries is representable", deepest < 0 && deepest > -1000000000);
    check("and it reports its own year back", lpl::history::yearOfDay(deepest) == -2600000);
    check("five million years back still fits", lpl::history::firstDayOfYear(-5000000) < 0);
    check("and reports its year", lpl::history::yearOfDay(lpl::history::firstDayOfYear(-5000000)) == -5000000);
}

/**
 * @brief Overlap, which is what makes a variable clock safe.
 *
 * @details A step that CROSSES a window must catch it. Landing on it exactly is the only case the old equality test
 *          handled, and it is the one a skipping clock never produces.
 */
void checkOverlapMakesAVariableClockSafe()
{
    std::printf("-- overlap, which is what makes a variable clock safe\n");
    const lpl::core::i32 windowFrom = lpl::history::dayOfDate(1815, 6u, 18u);
    const lpl::core::i32 windowTo = windowFrom;
    check("a step that lands on the day catches it",
          lpl::history::daysOverlap(windowFrom, windowFrom, windowFrom, windowTo));
    check("a step that jumps over it still catches it",
          lpl::history::daysOverlap(lpl::history::dayOfDate(1815, 1u, 1u), lpl::history::dayOfDate(1815, 12u, 31u),
                                    windowFrom, windowTo));
    check("a step that stops short does not",
          !lpl::history::daysOverlap(lpl::history::dayOfDate(1814, 1u, 1u), lpl::history::dayOfDate(1814, 12u, 31u),
                                     windowFrom, windowTo));
    check("nor does one that starts after",
          !lpl::history::daysOverlap(lpl::history::dayOfDate(1816, 1u, 1u), lpl::history::dayOfDate(1816, 12u, 31u),
                                     windowFrom, windowTo));
    check("touching at one end counts", lpl::history::daysOverlap(0, 10, 10, 20));
    check("a gap of one day does not", !lpl::history::daysOverlap(0, 10, 11, 20));
}

} // namespace

/**
 * @brief Date arithmetic, checked against anchors computed somewhere else.
 *
 * @warning Every absolute day number below was computed with an INDEPENDENT tool and pasted, never
 * read off this implementation's own output. An expected value taken from the code under test
 * asserts only that the code agrees with itself -- and this file exists because writing one of
 * these constants by hand had already produced an off-by-one (the Gregorian adoption day, 577736
 * where the answer is 577735).
 *
 * The negative years cannot be anchored that way -- the tool refuses years before 1 CE -- so they
 * are checked by STRUCTURE instead: the gap between consecutive new years must be 365 or 366
 * exactly as the leap rule says, and every day must report the year it was built from. Those are
 * invariants rather than a second implementation, which is the point: a second implementation
 * would be a second thing to get wrong the same way.
 */
int main()
{
    checkAnchorsComputedElsewhere();

    checkPrecisionIsTheWidthOfTheInterval();

    checkLeapYearsIncludingTheOnesThatCatchAHandRolledCalendar();

    checkStructureOverARangeNoCalendarToolWillAnchor();

    checkHowMuchOfTheRangeTheCorpusActuallyNeeds();

    checkOverlapMakesAVariableClockSafe();

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
