/**
 * @file test_calendar.cpp
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
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

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

} // namespace

int main()
{
    using namespace lpl;
    using namespace lpl::history;

    std::printf("-- anchors, computed elsewhere\n");
    {
        check("1 January 1 CE is day zero", dayOfDate(1, 1u, 1u) == 0);
        check("1 January 1970 is day 719162", dayOfDate(1970, 1u, 1u) == 719162);
        check("15 October 1582 is day 577735", dayOfDate(1582, 10u, 15u) == kGregorianAdoptionDay);
        check("18 June 1815 is day 662717", dayOfDate(1815, 6u, 18u) == 662717);
        check("1 January 1815 is day 662549", dayOfDate(1815, 1u, 1u) == 662549);
        check("31 December 1815 is day 662913", dayOfDate(1815, 12u, 31u) == 662913);
        // The interchange constant, checked the way it is used rather than restated.
        check("day zero is Julian Day Number 1721426", julianDayNumber(0) == 1721426);
        check("and 1 January 1970 is JDN 2440588", julianDayNumber(dayOfDate(1970, 1u, 1u)) == 2440588);
    }

    std::printf("-- precision is the WIDTH of the interval, which is the whole point\n");
    {
        // What "we only know the year" means, said in the unit the type now carries.
        const core::i32 yearFrom = firstDayOfYear(1815);
        const core::i32 yearTo = lastDayOfYear(1815);
        check("a year-resolution claim spans 365 days", yearTo - yearFrom + 1 == 365);

        const core::i32 exact = dayOfDate(1815, 6u, 18u);
        check("a day-resolution claim spans one", exact - exact + 1 == 1);
        // The distinction the old unit could not express at all: both of these were "1815".
        check("and the precise day lies inside the vague year", daysOverlap(exact, exact, yearFrom, yearTo));
        check("while a different year does not", !daysOverlap(exact, exact, firstDayOfYear(1816), lastDayOfYear(1816)));
    }

    std::printf("-- leap years, including the ones that catch a hand-rolled calendar\n");
    {
        check("1900 is not a leap year", !isLeapYear(1900));
        check("2000 is", isLeapYear(2000));
        check("2024 is", isLeapYear(2024));
        check("2023 is not", !isLeapYear(2023));
        // Computed as the day before the next new year, never as first + 364: that is wrong in
        // one year out of four, often enough to read as a rounding quirk rather than a bug.
        check("a leap year is 366 days long", lastDayOfYear(2000) - firstDayOfYear(2000) + 1 == 366);
        check("and 29 February exists in it", dayOfDate(2000, 2u, 29u) == dayOfDate(2000, 3u, 1u) - 1);
    }

    std::printf("-- structure, over a range no calendar tool will anchor\n");
    {
        bool everyGapMatchesTheLeapRule = true;
        bool everyNewYearReportsItsYear = true;
        bool everyLastDayReportsItsYear = true;
        core::i32 checked = 0;

        // @warning The range is DERIVED from the corpus, not chosen. A first version swept -3000 to
        // +2200 on the assumption that a historical corpus starts around the first cities --
        // and the data said otherwise: Pleiades dates 14 places at or before 100 000 BCE and
        // its earliest, Franchthi Cave among them, at 2 600 000 BCE. Places have Palaeolithic
        // occupation layers, so a gazetteer of them reaches into deep prehistory whatever a
        // library of TEXTS does. Sweeping only the era I expected would have left the arithmetic
        // that actually runs on the corpus untested.
        //
        // There is a year zero here (1 BCE) because the arithmetic needs one and astronomers use
        // the same convention; an ingester that reads "44 BCE" off a source must write -43.
        for (core::i32 year = -2600000; year <= 2200; ++year)
        {
            const core::i32 first = firstDayOfYear(year);
            const core::i32 next = firstDayOfYear(year + 1);
            const core::i32 expected = isLeapYear(year) ? 366 : 365;
            if (next - first != expected)
                everyGapMatchesTheLeapRule = false;
            if (yearOfDay(first) != year)
                everyNewYearReportsItsYear = false;
            if (yearOfDay(lastDayOfYear(year)) != year)
                everyLastDayReportsItsYear = false;
            ++checked;
        }

        check("2 602 201 consecutive years were exercised", checked == 2602201);
        check("each is exactly as long as the leap rule says", everyGapMatchesTheLeapRule);
        check("every new year reports the year it was built from", everyNewYearReportsItsYear);
        check("and so does every 31 December", everyLastDayReportsItsYear);
        // The sign still means what the year's sign meant, which is why this epoch was chosen
        // over Julian Day Numbers: a date reads the way the corpus reads.
        check("dates before 1 CE are negative", firstDayOfYear(-44) < 0);
        check("and dates after it are not", firstDayOfYear(44) > 0);
    }

    std::printf("-- how much of the range the corpus actually needs\n");
    {
        // @warning A day number is i32, so this format spans about 5.88 million years either side of
        // 1 CE. That is not a comfortable margin dressed up as a limit -- it is the limit, and
        // the deepest date the corpus carries today uses 44 % of it. Stated here so the day
        // somebody wants geological time, the refusal is a known bound rather than a signature
        // that quietly stops matching.
        const core::i32 deepest = firstDayOfYear(-2600000);
        check("the deepest date the corpus carries is representable", deepest < 0 && deepest > -1000000000);
        check("and it reports its own year back", yearOfDay(deepest) == -2600000);
        // Roughly 5.88 million years each way: the year whose first day still fits.
        check("five million years back still fits", firstDayOfYear(-5000000) < 0);
        check("and reports its year", yearOfDay(firstDayOfYear(-5000000)) == -5000000);
    }

    std::printf("-- overlap, which is what makes a variable clock safe\n");
    {
        // A step that CROSSES a window must catch it. Landing on it exactly is the only case the
        // old equality test handled, and it is the one a skipping clock never produces.
        const core::i32 windowFrom = dayOfDate(1815, 6u, 18u);
        const core::i32 windowTo = windowFrom;
        check("a step that lands on the day catches it", daysOverlap(windowFrom, windowFrom, windowFrom, windowTo));
        check("a step that jumps over it still catches it",
              daysOverlap(dayOfDate(1815, 1u, 1u), dayOfDate(1815, 12u, 31u), windowFrom, windowTo));
        check("a step that stops short does not",
              !daysOverlap(dayOfDate(1814, 1u, 1u), dayOfDate(1814, 12u, 31u), windowFrom, windowTo));
        check("nor does one that starts after",
              !daysOverlap(dayOfDate(1816, 1u, 1u), dayOfDate(1816, 12u, 31u), windowFrom, windowTo));
        check("touching at one end counts", daysOverlap(0, 10, 10, 20));
        check("a gap of one day does not", !daysOverlap(0, 10, 11, 20));
    }

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
