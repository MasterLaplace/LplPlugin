#include <lpl/engine/CaveParity.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(caves);

/**
 * @brief Gate P19 caves: a body walking at a cave's mouth ends up under rock, and the same walk at
 *        a mouth filled with rock is stopped, with the same signatures on both targets.
 *
 * @details The sealed run is the control: a collider that let everything through would put a body
 *          inside the cave, and through a mountain as well.
 */
LPL_TEST(a_body_walks_in_and_rock_stops_it)
{
    const lpl::engine::CaveFoldResult open = lpl::engine::foldCaveParity();
    const lpl::engine::CaveFoldResult sealed = lpl::engine::foldSealedCaveParity();

    test.check(open.warrenSignature != 0u, "the gate finds a cave to walk into");
    test.check(open.enclosedTicks > 0u, "a body walking at the mouth ends up under rock");
    test.check(sealed.enclosedTicks == 0u, "and a mouth filled with rock lets nobody in");
    test.check(sealed.blocked > open.blocked, "the sealed walk is stopped, not merely slower");
    test.check(open.walkSignature != sealed.walkSignature, "the two walks are different runs");
    test.check(open.spanSignature != sealed.spanSignature, "and disagree about where the rock is");

    test.measureHexadecimal("warren_signature", open.warrenSignature);
    test.measureHexadecimal("walk_signature", open.walkSignature);
    test.measureHexadecimal("span_signature", open.spanSignature);
    test.measureHexadecimal("sealed_walk_signature", sealed.walkSignature);
    test.measure("covered_columns", open.coveredColumns);
    test.measure("open_cells", open.openCells);
    test.measure("reachable_cells", open.reachableCells);
    test.measure("aperture_cells", open.apertureCells);
    test.measure("path_length", open.pathLength);
    test.measure("enclosed_ticks", open.enclosedTicks);
    test.measure("descended_levels", open.descendedLevels);
    test.measure("blocked", open.blocked);
    test.measure("head_bumps", open.headBumps);
    test.measure("navigable", open.navigable);
    test.measure("kind", open.kind);
    test.measure("sealed_blocked", sealed.blocked);
}
