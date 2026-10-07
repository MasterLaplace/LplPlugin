#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Geo.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(geo);

/**
 * @brief On a world closed at its seam, two places either side of it are neighbours: a body that
 *        measured with a plain subtraction would go round the planet the long way.
 */
LPL_TEST(the_near_place_is_across_the_seam)
{
    const lpl::math::Fixed32 width = lpl::math::Fixed32::fromFloat(1000.0f);
    const lpl::math::Fixed32 west = lpl::math::Fixed32::fromFloat(995.0f);
    const lpl::math::Fixed32 east = lpl::math::Fixed32::fromFloat(5.0f);
    const lpl::math::Fixed32 inland = lpl::math::Fixed32::fromFloat(120.0f);
    const lpl::math::Fixed32 closedGap = lpl::math::foldOntoShorterWay(width, east - west);
    lpl::math::GlobeWrap globe{};
    lpl::core::i32 cellDx = 0;
    lpl::core::i32 cellDz = 0;

    test.check(lpl::math::foldOntoShorterWay(lpl::math::Fixed32{}, east - west) <
                   lpl::math::Fixed32::fromFloat(-900.0f),
               "an open world sees them far apart");
    test.check(closedGap > lpl::math::Fixed32::zero() && closedGap < lpl::math::Fixed32::fromFloat(11.0f),
               "a closed world sees them as neighbours");
    test.check(lpl::math::foldOntoShorterWay(width, west - east) < lpl::math::Fixed32::zero(),
               "and the crossing has a direction");
    test.check(lpl::math::foldOntoShorterWay(width, inland).raw() == inland.raw(),
               "an ordinary separation is left alone");

    globe.columns = 1000;
    globe.rows = 400;
    lpl::math::shortestDelta(globe, 995, 0, 5, 0, cellDx, cellDz);
    test.check(cellDx == 10 && closedGap.toInt() == 10, "cells and world units agree");
}

/**
 * @brief A separation wider than the world folds all the way down, not once: 300 across a world
 *        120 wide is 60, where folding once left 180, most of the way round.
 */
LPL_TEST(a_gap_wider_than_the_world_folds_all_the_way)
{
    const lpl::math::Fixed32 narrow = lpl::math::Fixed32::fromFloat(120.0f);
    const lpl::math::Fixed32 threeLaps = lpl::math::foldOntoShorterWay(narrow, lpl::math::Fixed32::fromFloat(300.0f));

    test.check(threeLaps.toInt() == 60 && threeLaps > lpl::math::Fixed32::zero(), "it folds to 60 and keeps its side");
    test.check(lpl::math::foldOntoShorterWay(narrow, lpl::math::Fixed32::fromFloat(-300.0f)).toInt() == -60,
               "the same going the other way");
}
