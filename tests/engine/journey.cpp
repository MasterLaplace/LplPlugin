#include <lpl/engine/systems/Journey.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(journey);

/**
 * @brief Gate P20 journey: a body seeded by a constraint walks on its own, along attested roads and
 *        around ground it cannot cross, and earns some of the timeline's claims but not all.
 *
 * @details The fixture holds a place nearer than the attested destination but unlinked, a road
 *          that doglegs, a boulder on the way, and a ridge whose shortest way round is 57 cells:
 *          each number below separates a walk that read the corpus from one that only measured
 *          distances, which every signature would fold just as stably.
 */
LPL_TEST(a_body_walks_the_attested_roads)
{
    lpl::engine::systems::JourneyFoldResult journey{};

    lpl::engine::systems::foldJourneyState(journey);
    test.check(journey.seeded == 1u, "a constraint seeds a body");
    test.check(journey.arrivals >= 2u, "and the body goes places on its own");
    test.check(journey.earned >= 1u && journey.earned < journey.scoredClaims,
               "it earns some of the scored claims, and not all");
    test.check(journey.unplaceable == 0u, "it never walks to a place the gazetteer cannot locate");
    test.check(journey.firstArrival == 302u, "it follows the attested road, not the nearest place");
    test.check(journey.routedLegs >= 2u, "along the road's waypoints rather than the straight line");
    test.check(journey.avoided >= 1u, "and turns aside from ground it cannot cross");
    test.check(journey.scoredClaims != 0u &&
                   journey.divergenceScore == (65536u * journey.earned) / journey.scoredClaims,
               "the divergence score is the earned fraction");

    test.measureHexadecimal("chronicle_signature", journey.chronicleSignature);
    test.measureHexadecimal("position_signature", journey.positionSignature);
    test.measureHexadecimal("deed_signature", journey.deedSignature);
    test.measure("forced", journey.forced);
    test.measure("arrivals", journey.arrivals);
    test.measure("scored_claims", journey.scoredClaims);
    test.measure("earned", journey.earned);
    test.measureHexadecimal("divergence_score", journey.divergenceScore);
    test.measure("routed_legs", journey.routedLegs);
    test.measure("avoided", journey.avoided);
}

/**
 * @brief The attested link becomes one road, the shortest one round the ridge, and closing or
 *        cascading the routing grid changes the road only where it should.
 *
 * @details 57 cells is derived from the fixture: the ridge fills columns 0 to 39 of its row, so
 *          the road goes 28 columns east, crosses, and comes 28 back. A free climb lays 28 cells
 *          straight through the mountain, and would fold just as stably.
 */
LPL_TEST(roads_take_the_shortest_way_round)
{
    lpl::engine::systems::JourneyFoldResult journey{};

    lpl::engine::systems::foldJourneyState(journey);
    test.check(journey.roadPairs == 1u, "the attested link becomes exactly one road");
    test.check(journey.roadCells == 57u, "the shortest one that goes round the ridge");
    test.check(journey.roadSignature != 0u && journey.waypointSignature != 0u, "the road and its waypoints fold");
    test.check(journey.wrappedRoadCells > 0u && journey.wrappedRoadCells < journey.roadCells,
               "a grid closed at its seam lays a shorter road");
    test.check(journey.polarRoadCells <= journey.wrappedRoadCells, "and offering the poles never makes it longer");
    test.check(journey.cascadeRoadSignature == journey.roadSignature && journey.cascadeRoadCells == journey.roadCells,
               "a road planned coarse and refined fine is the road found flat, cell for cell");
    test.check(journey.cascadeCoarseExpanded > 0u && journey.cascadeCorridorCells > 0u,
               "because a coarse plan ran and opened a corridor");

    test.measureHexadecimal("road_signature", journey.roadSignature);
    test.measureHexadecimal("waypoint_signature", journey.waypointSignature);
    test.measure("road_cells", journey.roadCells);
    test.measure("wrapped_road_cells", journey.wrappedRoadCells);
    test.measure("polar_road_cells", journey.polarRoadCells);
    test.measure("cascade_coarse_expanded", journey.cascadeCoarseExpanded);
    test.measure("cascade_corridor_cells", journey.cascadeCorridorCells);
}

/**
 * @brief One more admitted source, or a world closed at its seam, gives a body a different life:
 *        "possible worlds" is a mechanism, not a label.
 */
LPL_TEST(other_worlds_are_walked_differently)
{
    lpl::engine::systems::JourneyFoldResult journey{};

    lpl::engine::systems::foldJourneyState(journey);
    test.check(journey.alternateChronicle != 0u && journey.alternateChronicle != journey.chronicleSignature,
               "the world according to one source is another world");
    test.check(journey.alternateArrivals >= 1u, "where the body still goes somewhere");
    test.check(journey.closedChronicle != journey.chronicleSignature,
               "a closed world folds differently from an open one");
    test.check(journey.closedArrivals >= 1u, "and its body goes somewhere too");

    test.measureHexadecimal("alternate_chronicle_signature", journey.alternateChronicle);
    test.measure("alternate_arrivals", journey.alternateArrivals);
    test.measure("alternate_first_arrival", journey.alternateFirst);
    test.measureHexadecimal("closed_chronicle_signature", journey.closedChronicle);
    test.measure("closed_arrivals", journey.closedArrivals);
    test.measure("closed_first_arrival", journey.closedFirst);
}
