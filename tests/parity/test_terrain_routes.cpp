#include <lpl/engine/systems/TerrainRoutes.hpp>
#include <lpl/history/Place.hpp>
#include <lpl/procgen/Heightfield.hpp>

#include <cstdio>

namespace {

constexpr lpl::core::u32 kSize = 64u;
constexpr lpl::core::u32 kRidgeRow = 32u;
constexpr lpl::core::u32 kPassFrom = 56u;

/**
 * @name Where the two places either side of the ridge stand
 *
 * Cell centres, so a place sits in the middle of its cell rather than on a boundary where two cells
 * would both be defensible answers.
 * @{
 */
constexpr float kWestX = -19.5f;
constexpr float kSouthZ = -9.5f;
constexpr float kNorthZ = 10.5f;
/** @} */

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
 * @class TableResolver
 * @brief A gazetteer of a handful of places, and the links a corpus states between them.
 *
 * @warning Links are stored in BOTH directions, mirroring what the real reader does: the source data
 * is directed (A lists B, B does not necessarily list A) but a road is walked either way, and a
 * consumer forced to check two orders ends up checking one.
 */
class TableResolver final : public lpl::history::IPlaceResolver {
public:
    /**
     * @brief Adds a place.
     *
     * @param id      Its identifier.
     * @param x       World X.
     * @param z       World Z.
     * @param located Whether anyone has found it on the ground.
     */
    void add(lpl::core::u32 id, float x, float z, bool located = true)
    {
        lpl::history::Place place;
        place.id = id;
        place.x = lpl::math::Fixed32::fromFloat(x);
        place.z = lpl::math::Fixed32::fromFloat(z);
        place.located = located;
        _places[_count++] = place;
    }

    /**
     * @brief States that a corpus says these two are connected.
     *
     * @param a One place.
     * @param b The other.
     */
    void link(lpl::core::u32 a, lpl::core::u32 b)
    {
        _links[_linkCount][0] = a;
        _links[_linkCount][1] = b;
        ++_linkCount;
        _links[_linkCount][0] = b;
        _links[_linkCount][1] = a;
        ++_linkCount;
    }

    /**
     * @brief Looks a place up.
     *
     * @param id  The identifier.
     * @param out Receives it.
     * @return false when this table does not carry it.
     */
    [[nodiscard]] bool resolve(lpl::core::u32 id, lpl::history::Place &out) const override
    {
        for (lpl::core::u32 i = 0u; i < _count; ++i)
        {
            if (_places[i].id != id)
                continue;
            out = _places[i];
            return true;
        }
        return false;
    }

    /**
     * @brief Collects what a corpus says this place is connected to.
     *
     * @param id       The place.
     * @param out      Receives the neighbours.
     * @param capacity Room in @p out.
     * @return How many were written.
     */
    [[nodiscard]] lpl::core::u32 linkedPlaces(lpl::core::u32 id, lpl::core::u32 *out,
                                              lpl::core::u32 capacity) const override
    {
        lpl::core::u32 written = 0u;
        for (lpl::core::u32 i = 0u; i < _linkCount && written < capacity; ++i)
        {
            if (_links[i][0] == id)
                out[written++] = _links[i][1];
        }
        return written;
    }

private:
    lpl::history::Place _places[16]{};
    lpl::core::u32 _count{0u};
    lpl::core::u32 _links[32][2]{};
    lpl::core::u32 _linkCount{0u};
};

/**
 * @brief A plain with a ridge across it, and one pass at the eastern end.
 *
 * The plain sits ABOVE the routing model's water level, which defaults to zero: a field left at
 * height zero would be entirely sea, so every cell would pay the water penalty and the fixture
 * would be measuring a uniform surcharge rather than a relief.
 *
 * @param size      Cells on a side.
 * @param ridgeRow  Row the ridge runs along.
 * @param passFrom  First column of the gap.
 * @return The field.
 */
[[nodiscard]] lpl::procgen::Heightfield ridgedPlain(lpl::core::u32 size, lpl::core::u32 ridgeRow,
                                                    lpl::core::u32 passFrom)
{
    lpl::procgen::Heightfield field{size, size, lpl::math::Fixed32::fromFloat(1.0f)};
    for (lpl::core::u32 x = 0u; x < passFrom; ++x)
        field.at(x, ridgeRow) = lpl::math::Fixed32::fromFloat(40.0f);
    return field;
}

/**
 * @brief A road goes round a ridge, not through it.
 *
 * @details The pass is the eastern end of the ridge; a road that used it has to get there. The last
 *          waypoint is the goal's own cell, so the walk ends where it was sent rather than at the last
 *          bend before it.
 *
 * @param field       The ridged plain.
 * @param places      The gazetteer, whose first two places stand either side of the ridge.
 * @param detourCount Receives how many waypoints the road has.
 * @param detour      Receives the road's waypoints, which the checks after this one compare against.
 */
void checkRoadGoesRoundARidge(const lpl::procgen::Heightfield &field, const TableResolver &places,
                              lpl::core::u32 &detourCount, lpl::history::RouteLeg (&detour)[12])
{
    std::printf("-- a road goes round a ridge, not through it\n");
    lpl::engine::systems::TerrainRouteParams params;
    params.cost.slopePenalty = 6.0f;
    lpl::engine::systems::TerrainRoutes routes;
    routes.bind(field, places, params);

    detourCount = routes.route(1u, 2u, detour, 12u);
    check("a road is found across the ridge line", detourCount > 0u);

    lpl::math::Fixed32 furthestEast = lpl::math::Fixed32::fromFloat(-1000.0f);
    for (lpl::core::u32 i = 0u; i < detourCount; ++i)
    {
        if (detour[i].x > furthestEast)
            furthestEast = detour[i].x;
    }
    check("it bends, rather than running straight", detourCount > 1u);
    check("and it bends toward the pass", furthestEast > lpl::math::Fixed32::fromFloat(20.0f));

    lpl::history::Place goal;
    check("the gazetteer carries the goal", places.resolve(2u, goal));
    const lpl::math::Fixed32 lastX = detour[detourCount - 1u].x;
    const lpl::math::Fixed32 lastZ = detour[detourCount - 1u].z;
    check("the last waypoint is the destination cell", lastX == goal.x && lastZ == goal.z);
}

/**
 * @brief With the climb made free, the same road is straight.
 *
 * @details The control, and the reason the fixture has a ridge at all: "the road avoided the ridge" is
 *          also what a straight line does when nothing is in the way. Only a run where the climb costs
 *          nothing shows that the detour was bought by the relief.
 *
 * @param field       The ridged plain.
 * @param places      The gazetteer, whose first two places stand either side of the ridge.
 * @param detourCount How many waypoints the road relief produced has.
 */
void checkFreeClimbLeavesTheRoadStraight(const lpl::procgen::Heightfield &field, const TableResolver &places,
                                         lpl::core::u32 detourCount)
{
    std::printf("-- with the climb made free, the same road is straight\n");
    lpl::engine::systems::TerrainRouteParams params;
    params.cost.slopePenalty = 0.0f;
    lpl::engine::systems::TerrainRoutes routes;
    routes.bind(field, places, params);

    lpl::history::RouteLeg straight[12]{};
    const lpl::core::u32 count = routes.route(1u, 2u, straight, 12u);
    check("a free climb is crossed head-on", count == 1u);
    if (count == 1u)
        check("so the only waypoint is the destination", straight[0].x < lpl::math::Fixed32::zero());
    else
        check("so the only waypoint is the destination", false);
    check("and it is a different road from the one relief produced", count != detourCount);
}

/**
 * @brief A road is summarised as its corners, strided rather than truncated.
 *
 * @details The claim that separates striding from truncating: truncation keeps the FIRST corners, so
 *          the last slot would hold an early bend and the body would strike out straight from the
 *          middle of the ridge -- across precisely what the road existed to go round.
 *
 * @param field       The ridged plain.
 * @param places      The gazetteer, whose first two places stand either side of the ridge.
 * @param detourCount How many waypoints the road relief produced has.
 * @param detour      The waypoints of that road.
 */
void checkRoadIsSummarisedAsStridedCorners(const lpl::procgen::Heightfield &field, const TableResolver &places,
                                           lpl::core::u32 detourCount, const lpl::history::RouteLeg (&detour)[12])
{
    std::printf("-- a road is summarised as its corners, strided rather than truncated\n");
    lpl::engine::systems::TerrainRouteParams params;
    params.cost.slopePenalty = 6.0f;
    lpl::engine::systems::TerrainRoutes routes;
    routes.bind(field, places, params);

    check("the detour has more corners than two", detourCount > 2u);

    lpl::history::RouteLeg tight[2]{};
    const lpl::core::u32 count = routes.route(1u, 2u, tight, 2u);
    check("a caller with two slots gets two waypoints", count == 2u);
    check("and the last of them is still the destination",
          count == 2u && tight[1].x == detour[detourCount - 1u].x && tight[1].z == detour[detourCount - 1u].z);
    check("not the road's first bend", count == 2u && !(tight[1].x == detour[0].x && tight[1].z == detour[0].z));
}

/**
 * @brief The two cells either side of the origin are not one cell.
 *
 * @details Truncating division maps -3 and 2 both to cell 0 when four units make a cell, so the cell
 *          straddling the origin would be half as wide as every other and two places five units apart
 *          would share it. They would then route to themselves, which returns nothing -- a walk that
 *          quietly goes straight only near the middle of the map.
 *
 * @param field The ridged plain.
 */
void checkCellsEitherSideOfTheOriginAreTwoCells(const lpl::procgen::Heightfield &field)
{
    std::printf("-- the two cells either side of the origin are not one cell\n");
    TableResolver straddle;
    straddle.add(10u, -2.5f, 0.5f);
    straddle.add(11u, 2.5f, 0.5f);

    lpl::engine::systems::TerrainRouteParams params;
    params.cellSize = 4u;
    lpl::engine::systems::TerrainRoutes routes;
    routes.bind(field, straddle, params);

    lpl::history::RouteLeg legs[4]{};
    const lpl::core::u32 count = routes.route(10u, 11u, legs, 4u);
    check("two places five units apart do not share a four-unit cell", count >= 1u);
}

/**
 * @brief A place nobody has found gets no road.
 *
 * @details Neither an unlocated place nor one off the grid counts as a planned route: the search never
 *          ran, and reporting it as an unreachable goal would blame the terrain for a gap in the
 *          gazetteer.
 *
 * @param field  The ridged plain.
 * @param places The gazetteer, whose third place is off the grid and whose fourth was never found.
 */
void checkPlaceNobodyHasFoundGetsNoRoad(const lpl::procgen::Heightfield &field, const TableResolver &places)
{
    std::printf("-- a place nobody has found gets no road\n");
    lpl::engine::systems::TerrainRouteParams params;
    lpl::engine::systems::TerrainRoutes routes;
    routes.bind(field, places, params);

    lpl::history::RouteLeg legs[8]{};
    check("an unlocated place is not routed to", routes.route(1u, 4u, legs, 8u) == 0u);
    check("nor is one off the grid", routes.route(1u, 3u, legs, 8u) == 0u);
    check("and neither is blamed on the terrain", routes.planned() == 0u && routes.unreachable() == 0u);
}

/**
 * @brief An attested network grows along its own trunk.
 *
 * @details Two roads a corpus attests, running parallel two cells apart on flat ground. With an existing
 *          road discounted, the second should join the first and share it; with the discount removed,
 *          it has no reason to and lays a second line beside it.
 *
 *          The resolver hands out four directed links for two attested roads, so a count of four pairs
 *          would mean each road was laid twice -- the second time taking its own discount, which is a
 *          parallel road nobody built. Both roads must still connect: a merge that lost one of them
 *          would also pave fewer cells, so the saving means nothing without that beside it.
 */
void checkAttestedNetworkGrowsAlongItsOwnTrunk()
{
    std::printf("-- an attested network grows along its own trunk\n");
    const lpl::procgen::Heightfield flat{kSize, kSize, lpl::math::Fixed32::fromFloat(1.0f)};

    TableResolver network;
    network.add(1u, -19.5f, -19.5f);
    network.add(2u, -19.5f, 20.5f);
    network.add(3u, -17.5f, -19.5f);
    network.add(4u, -17.5f, 20.5f);
    network.link(1u, 2u);
    network.link(3u, 4u);
    const lpl::core::u32 order[4] = {1u, 2u, 3u, 4u};

    lpl::engine::systems::TerrainRouteParams merging;
    merging.cost.reuseDiscount = 0.9f;
    lpl::engine::systems::TerrainRoutes merged;
    merged.bind(flat, network, merging);
    const lpl::core::u32 pavedMerged = merged.paveAttested(order, 4u);

    lpl::engine::systems::TerrainRouteParams parallel;
    parallel.cost.reuseDiscount = 0.0f;
    lpl::engine::systems::TerrainRoutes apart;
    apart.bind(flat, network, parallel);
    const lpl::core::u32 pavedApart = apart.paveAttested(order, 4u);

    check("both attested links become road", pavedMerged > 0u && pavedApart > 0u);
    check("a discounted road is joined rather than doubled", pavedMerged < pavedApart);
    check("each attested pair is laid once, not once per direction", merged.pairs() == 2u && apart.pairs() == 2u);
    check("and neither road was lost to the merge", merged.unreachable() == 0u && apart.unreachable() == 0u);
}

/**
 * @brief A road crosses the antimeridian instead of going round the planet.
 *
 * @details Flat ground, so the ONLY thing deciding the route is the shape of the world. The two places
 *          stand five cells west of the seam and five cells east of it: ten cells apart the short way,
 *          one hundred and ninety the long way.
 *
 * @warning THE assertion is that the closed world's road crosses the seam. Both routes exist, both are
 *          valid, and the long one is a perfectly ordinary road -- which is why nothing downstream could
 *          tell it was the wrong answer.
 *
 * @warning The heuristic is the half that fails SILENTLY. An unwrapped one overestimates by nearly a
 *          circumference at the seam, so A* settles the whole planet before conceding the short way --
 *          it still returns a road, just after doing enormously more work. Expanding fewer cells than
 *          the long route has is what says the heuristic actually pointed at the seam rather than being
 *          rescued by exhaustion.
 *
 * The route must actually step ACROSS the boundary, not merely be short. A wrap that does not match the
 * grid is ignored rather than half-applied: a router that wrapped its neighbours at one width and its
 * heuristic at another would be inadmissible in a way no single line of it looks wrong.
 */
void checkRoadCrossesTheAntimeridian()
{
    std::printf("-- a road crosses the antimeridian instead of going round the planet\n");
    constexpr lpl::core::u32 kW = 200u;
    constexpr lpl::core::u32 kH = 24u;
    lpl::procgen::Heightfield flat{kW, kH, lpl::math::Fixed32::zero()};

    lpl::procgen::RoutingParams params{};
    params.waterPenalty = 0.0f;
    params.reuseDiscount = 0.0f;

    const lpl::core::u32 startX = kW - 5u;
    const lpl::core::u32 goalX = 5u;

    lpl::procgen::RoutingParams open = params;
    const lpl::procgen::RoutedPath around = lpl::procgen::routeLeastCost(flat, nullptr, startX, 12u, goalX, 12u, open);
    check("an open world still finds a road", around.found);
    check("but it goes the long way round", around.cells.size() > 100u);

    lpl::procgen::RoutingParams closed = params;
    closed.wrapColumns = kW;
    const lpl::procgen::RoutedPath through =
        lpl::procgen::routeLeastCost(flat, nullptr, startX, 12u, goalX, 12u, closed);
    check("a closed world finds one too", through.found);

    check("and it crosses the seam instead", through.cells.size() <= 12u);
    check("which is far shorter than going round", through.cells.size() * 5u < around.cells.size());
    check("and it costs less", through.cost < around.cost);

    check("without searching the whole world to find it", through.expanded < around.cells.size());
    std::printf("     seam route %zu cells, %u expanded; the long way %zu cells, %u expanded\n", through.cells.size(),
                through.expanded, around.cells.size(), around.expanded);

    bool steppedOverSeam = false;
    for (std::size_t i = 1u; i < through.cells.size(); ++i)
    {
        const lpl::core::i32 previous = static_cast<lpl::core::i32>(through.cells[i - 1u] % kW);
        const lpl::core::i32 here = static_cast<lpl::core::i32>(through.cells[i] % kW);
        if ((previous > here ? previous - here : here - previous) > 1)
            steppedOverSeam = true;
    }
    check("and the path really steps over the boundary", steppedOverSeam);

    lpl::procgen::RoutingParams mismatched = params;
    mismatched.wrapColumns = kW + 1u;
    const lpl::procgen::RoutedPath ignored =
        lpl::procgen::routeLeastCost(flat, nullptr, startX, 12u, goalX, 12u, mismatched);
    check("a wrap that does not match the grid is ignored", ignored.cells.size() == around.cells.size());
}

/**
 * @brief And over the pole, which is a shortcut and not a wrap.
 *
 * @details Flat ground again: the only thing deciding the route is the shape of the world. Two places at
 *          high latitude on OPPOSITE meridians: three rows from the top, half a world apart. Going round
 *          the sheet is sixty columns; going over the pole is about seven rows.
 *
 * @warning The whole point: two points at high latitude on opposite meridians really ARE closer over the
 *          pole, and a router that cannot cross one lays a road all the way round instead -- a perfectly
 *          valid, perfectly long road that nothing downstream can question.
 *
 * The road must actually TOUCH the top row and jump half a world there, rather than merely being short
 * by some other means.
 *
 * @warning **A pole is not a wrap.** Going north off the top must come back on the TOP row, not the
 *          bottom: the other reading joins the Arctic to the Antarctic, and a route computed that way
 *          would look like an ordinary road through a place nobody can walk.
 *
 * Two points near the EQUATOR of this sheet are not helped by a pole, and the router must not invent a
 * detour: the estimate has to stay the direct one where the direct one wins.
 *
 * @warning The heuristic must stay ADMISSIBLE, and a polar road that never costs more is what says so:
 *          with poles enabled the router must never return a road MORE expensive than the same search
 *          without the shortcut. An estimate that overestimated a polar route would discard it
 *          unevaluated and hand back the long way while looking entirely healthy.
 *
 * And a pole is refused on a grid that does not close east-west, because half a world of nothing is
 * nothing.
 */
void checkRoadCrossesThePole()
{
    std::printf("-- and over the pole, which is a shortcut and not a wrap\n");
    constexpr lpl::core::u32 kW = 120u;
    constexpr lpl::core::u32 kH = 40u;
    lpl::procgen::Heightfield flat{kW, kH, lpl::math::Fixed32::zero()};

    lpl::procgen::RoutingParams closed{};
    closed.waterPenalty = 0.0f;
    closed.reuseDiscount = 0.0f;
    closed.wrapColumns = kW;

    lpl::procgen::RoutingParams polar = closed;
    polar.wrapPoles = true;

    const lpl::core::u32 startX = 10u;
    const lpl::core::u32 goalX = 10u + kW / 2u;
    const lpl::core::u32 row = 3u;

    const lpl::procgen::RoutedPath around =
        lpl::procgen::routeLeastCost(flat, nullptr, startX, row, goalX, row, closed);
    const lpl::procgen::RoutedPath over = lpl::procgen::routeLeastCost(flat, nullptr, startX, row, goalX, row, polar);
    check("a world without poles still finds a road", around.found);
    check("and so does one with them", over.found);

    check("the polar route is shorter", over.cells.size() < around.cells.size());
    check("and it costs less", over.cost < around.cost);
    std::printf("     over the pole %zu cells (%u expanded), round the sheet %zu (%u expanded)\n", over.cells.size(),
                over.expanded, around.cells.size(), around.expanded);

    bool touchedPole = false;
    for (std::size_t i = 1u; i < over.cells.size(); ++i)
    {
        const lpl::core::i32 previousZ = static_cast<lpl::core::i32>(over.cells[i - 1u] / kW);
        const lpl::core::i32 hereZ = static_cast<lpl::core::i32>(over.cells[i] / kW);
        const lpl::core::i32 previousX = static_cast<lpl::core::i32>(over.cells[i - 1u] % kW);
        const lpl::core::i32 hereX = static_cast<lpl::core::i32>(over.cells[i] % kW);
        const lpl::core::i32 jump = previousX > hereX ? previousX - hereX : hereX - previousX;
        if (previousZ == 0 && hereZ == 0 && jump > 1)
            touchedPole = true;
    }
    check("the road really crosses the pole", touchedPole);

    bool reachedFarSide = false;
    for (lpl::core::u32 cell : over.cells)
        if (cell / kW >= kH - 2u)
            reachedFarSide = true;
    check("and never comes out at the other pole", !reachedFarSide);

    const lpl::core::u32 mid = kH / 2u;
    const lpl::procgen::RoutedPath flatRun = lpl::procgen::routeLeastCost(flat, nullptr, 10u, mid, 30u, mid, polar);
    check("a short hop mid-sheet is still direct", flatRun.found && flatRun.cells.size() <= 22u);

    check("enabling poles never makes a road worse", over.cost <= around.cost);

    lpl::procgen::RoutingParams openSheet{};
    openSheet.waterPenalty = 0.0f;
    openSheet.reuseDiscount = 0.0f;
    openSheet.wrapPoles = true;
    const lpl::procgen::RoutedPath refused =
        lpl::procgen::routeLeastCost(flat, nullptr, startX, row, goalX, row, openSheet);
    check("poles alone do nothing on an open sheet",
          refused.cells.size() == around.cells.size() || refused.cells.size() > over.cells.size());
}

/**
 * @brief How a block is reduced decides whether the plan can see the pass.
 *
 * @warning **This measurement exists because the tree disagreed with itself.** The cascade's own
 *          documentation says a coarse cell is an AVERAGE; the only code that ever built one took the
 *          MAXIMUM and argued against the mean in a comment. Both were reasoning, neither was measured,
 *          and the fixture that measured nothing was a one-cell ridge with a pass nine cells wide --
 *          wider than the block, and aligned to it, so every rule found it.
 *
 * A real range is thick and its pass is narrow, and neither is aligned to a lattice chosen for a memory
 * budget. That is the case each rule has to survive: the range starts on a row deliberately not a
 * multiple of the ratio, and its pass is three cells wide, inside one block.
 *
 * The flat search is the ground truth: what the road IS, at full resolution, with no summary involved.
 * Every rule is judged against it and against nothing chosen here. Where a road crosses the range is the
 * whole question: through the pass, or over the wall.
 *
 * @warning **Both rivals throw away one of the two facts the plan needs, in opposite directions.** The
 *          maximum keeps the wall and erases the gap: a block holding a three-cell pass also holds five
 *          cells of ridge, so it reads as solid ridge. The minimum keeps the gap and erases the wall: a
 *          block straddling the range's edge also holds valley floor, so it reads as valley and the plan
 *          crosses wherever it likes. Only the mean is sensitive to the PROPORTION, which is the one
 *          thing that distinguishes "a wall with a door in it" from either a wall or a door.
 *
 * Not "close to" the flat road: the SAME road. A corridor that contains the optimum lets A* return the
 * optimum, and stating it exactly is what would notice a corridor that clipped a corner off it.
 *
 * @warning **The lesson that is easy to miss, and the reason this is measured at all.** Neither rival
 *          WIDENED. The recover-once-and-report path only catches a plan that turns out to be
 *          impassable; a plan that merely hides a cheaper way is passable, so the fine search finds a
 *          road, reports success, and hands back something 2.65 times the cost of the real one. A bad
 *          summary does not fail loudly -- it succeeds quietly.
 */
void checkBlockReductionDecidesWhetherThePlanSeesThePass()
{
    std::printf("-- how a block is reduced decides whether the plan can see the pass\n");
    constexpr lpl::core::u32 kRatio = 8u;
    constexpr lpl::core::u32 kSize = 192u;
    constexpr lpl::core::u32 kRidgeTop = 90u;
    constexpr lpl::core::u32 kRidgeBottom = 97u;
    constexpr lpl::core::u32 kGateFrom = 149u;
    constexpr lpl::core::u32 kGateTo = 151u;

    lpl::procgen::Heightfield fine{kSize, kSize, lpl::math::Fixed32::fromFloat(1.0f)};
    for (lpl::core::u32 z = kRidgeTop; z <= kRidgeBottom; ++z)
        for (lpl::core::u32 x = 0u; x < kSize; ++x)
            if (x < kGateFrom || x > kGateTo)
                fine.at(x, z) = lpl::math::Fixed32::fromFloat(60.0f);

    const lpl::core::u32 coarseWidth = (kSize + kRatio - 1u) / kRatio;
    const auto reduceBy = [&](int rule) {
        lpl::procgen::Heightfield out{coarseWidth, coarseWidth, lpl::math::Fixed32::zero()};
        for (lpl::core::u32 cz = 0u; cz < coarseWidth; ++cz)
            for (lpl::core::u32 cx = 0u; cx < coarseWidth; ++cx)
            {
                lpl::math::Fixed32 hi = lpl::math::Fixed32::min();
                lpl::math::Fixed32 lo = lpl::math::Fixed32::max();
                for (lpl::core::u32 fz = 0u; fz < kRatio; ++fz)
                    for (lpl::core::u32 fx = 0u; fx < kRatio; ++fx)
                    {
                        const lpl::math::Fixed32 here = fine.at(cx * kRatio + fx, cz * kRatio + fz);
                        if (here > hi)
                            hi = here;
                        if (here < lo)
                            lo = here;
                    }
                out.at(cx, cz) = rule == 0 ? hi : lo;
            }
        return out;
    };

    lpl::procgen::RoutingParams cost{};
    cost.waterPenalty = 0.0f;
    cost.reuseDiscount = 0.0f;

    constexpr lpl::core::u32 kStartX = 20u;
    constexpr lpl::core::u32 kStartZ = 30u;
    constexpr lpl::core::u32 kGoalZ = 160u;

    const lpl::procgen::RoutedPath truth =
        lpl::procgen::routeLeastCost(fine, nullptr, kStartX, kStartZ, kStartX, kGoalZ, cost);
    check("the flat search crosses the range", truth.found);

    const auto crossesAtGate = [&](const lpl::procgen::RoutedPath &path) {
        for (const lpl::core::u32 cell : path.cells)
        {
            const lpl::core::u32 x = cell % kSize;
            const lpl::core::u32 z = cell / kSize;
            if (z >= kRidgeTop && z <= kRidgeBottom && x >= kGateFrom && x <= kGateTo)
                return true;
        }
        return false;
    };
    check("and it crosses at the pass", crossesAtGate(truth));

    std::printf("     flat: %zu cells, cost %.1f, %u expanded\n", truth.cells.size(), truth.cost.toFloat(),
                truth.expanded);

    const char *names[3] = {"max ", "min ", "mean"};
    bool sawGate[3] = {false, false, false};
    bool optimal[3] = {false, false, false};
    bool widened[3] = {false, false, false};
    for (int rule = 0; rule < 3; ++rule)
    {
        const lpl::procgen::Heightfield summary =
            rule == 2 ? lpl::procgen::reduceHeightfield(fine, kRatio) : reduceBy(rule);
        const lpl::procgen::HierarchicalRoute cascaded =
            lpl::procgen::routeAcrossWorld(summary, fine, kRatio, nullptr, kStartX, kStartZ, kStartX, kGoalZ, cost);
        sawGate[rule] = cascaded.fine.found && crossesAtGate(cascaded.fine);
        optimal[rule] = cascaded.fine.found && cascaded.fine.cost.raw() == truth.cost.raw();
        widened[rule] = cascaded.widened;
        std::printf("     %s: %s, %zu cells, cost %.1f, %u+%u expanded%s, %s\n", names[rule],
                    cascaded.fine.found ? "found" : "NO ROAD", cascaded.fine.cells.size(), cascaded.fine.cost.toFloat(),
                    cascaded.coarseExpanded, cascaded.fine.expanded, cascaded.widened ? ", WIDENED" : "",
                    sawGate[rule] ? "through the pass" : "OVER THE WALL");
    }

    check("the mean summary routes through the pass", sawGate[2]);
    check("the maximum summary does not", !sawGate[0]);
    check("nor does the minimum", !sawGate[1]);

    check("and the road it returns is the flat road, exactly", optimal[2]);
    check("while the rivals return a costlier one", !optimal[0] && !optimal[1]);

    check("and neither of them widened, so nothing reported the mistake", !widened[0] && !widened[1]);
}

/**
 * @brief A world too large to search flat is searched twice.
 *
 * @warning **The measurement that forces the shape.** A* keeps cost, parent and a settled flag per cell
 *          -- about twelve bytes. A global grid at thirty-metre cells is 890 733 444 400 cells, roughly
 *          ten TERABYTES; at 7.7 km it is 163 MB. A planet cannot be searched flat.
 *
 * A ridge runs across the middle with one pass through it, so the route has a shape to find rather than
 * a straight line to walk: a corridor around a straight line would confine nothing and the comparison
 * would be vacuous.
 *
 * @warning This used to build the summary by hand, taking the MAXIMUM of each block on the stated grounds
 *          that "a mean would let a wall average away into a slope".
 *          checkBlockReductionDecidesWhetherThePlanSeesThePass() measures that claim on a range a
 *          summary can actually get wrong, and it is false: the maximum erases the pass and returns a
 *          road 2.65 times the cost of the real one, without widening, so nothing reports it. One
 *          reducer in the tree, and it is the mean.
 *
 * @warning **The whole justification.** If the cascade settled as many cells as the flat search, it would
 *          be the same search with extra steps -- and the ten terabytes above would still be ten
 *          terabytes. The corridor is what makes a planet affordable.
 *
 * And the road must be worth having. A corridor that squeezed the route into a detour would buy its
 * cheapness with a road nobody would drive; within a quarter is the bound, and it is measured against
 * the flat answer rather than against a number chosen here.
 *
 * @warning A corridor constrains the SEARCH, never the ground. A goal outside it must make the search
 *          FAIL rather than wander out and return a road nobody planned -- which is why this is a
 *          separate field and not a very large cost.
 *
 * A corridor that does not match the grid is ignored rather than half applied.
 *
 * @warning **A cascade on a CLOSED world, which is what the coarse plan's own wrap is for.** The coarse
 *          grid is narrower than the fine one by the cell ratio, so a coarse heuristic folding at the
 *          FINE width folds at a width its grid does not have -- it stops being admissible exactly at
 *          the seam, and the plan comes back round the long way while looking like any other plan. The
 *          two places sit either side of it.
 *
 * @warning **The number that catches a coarse heuristic folding at the wrong width.** Both widths find the
 *          same road here -- the grid is small enough that exhaustion rescues a bad estimate -- so the
 *          road alone says nothing. What moves is the WORK: measured, 4 coarse expansions with the
 *          coarse grid's own width against 30 with the fine one's. The bound is derived rather than
 *          tuned: a coarse plan that pointed at the seam settles on the order of the road's length,
 *          never more than it.
 *
 * A ratio of one is a plain route, not a cascade pretending to be one.
 */
void checkWorldTooLargeToSearchFlatIsSearchedTwice()
{
    std::printf("-- a world too large to search flat is searched twice\n");
    constexpr lpl::core::u32 kRatio = 8u;
    constexpr lpl::core::u32 kFine = 256u;
    constexpr lpl::core::u32 kCoarse = kFine / kRatio;

    lpl::procgen::Heightfield fine{kFine, kFine, lpl::math::Fixed32::fromFloat(1.0f)};
    constexpr lpl::core::u32 kRidge = 128u;
    constexpr lpl::core::u32 kGate = 200u;
    for (lpl::core::u32 x = 0u; x < kFine; ++x)
        if (x < kGate || x > kGate + 8u)
            fine.at(x, kRidge) = lpl::math::Fixed32::fromFloat(60.0f);

    const lpl::procgen::Heightfield coarse = lpl::procgen::reduceHeightfield(fine, kRatio);
    check("the summary is one cell per block", coarse.width() == kCoarse && coarse.depth() == kCoarse);

    lpl::procgen::RoutingParams cost{};
    cost.waterPenalty = 0.0f;
    cost.reuseDiscount = 0.0f;

    const lpl::procgen::RoutedPath flat = lpl::procgen::routeLeastCost(fine, nullptr, 20u, 20u, 220u, 240u, cost);
    const lpl::procgen::HierarchicalRoute cascaded =
        lpl::procgen::routeAcrossWorld(coarse, fine, kRatio, nullptr, 20u, 20u, 220u, 240u, cost);

    check("the flat search finds a road", flat.found);
    check("and so does the cascade", cascaded.fine.found);
    check("the coarse plan existed", cascaded.coarseFound);

    std::printf("     flat: %zu cells, %u expanded | cascade: %zu cells, %u+%u expanded, "
                "%u corridor cells%s\n",
                flat.cells.size(), flat.expanded, cascaded.fine.cells.size(), cascaded.coarseExpanded,
                cascaded.fine.expanded, cascaded.corridorCells, cascaded.widened ? ", WIDENED" : "");

    check("the cascade settles far fewer cells", cascaded.coarseExpanded + cascaded.fine.expanded < flat.expanded);

    check("and the road it finds is comparable",
          cascaded.fine.cost <= flat.cost + flat.cost * lpl::math::Fixed32::fromFloat(0.25f));

    lpl::procgen::Grid<lpl::core::u8> narrow{kFine, kFine, lpl::core::u8{0}};
    for (lpl::core::u32 z = 0u; z < 8u; ++z)
        for (lpl::core::u32 x = 0u; x < 8u; ++x)
            narrow.at(x, z) = 1u;
    lpl::procgen::RoutingParams boxed = cost;
    boxed.corridor = &narrow;
    const lpl::procgen::RoutedPath refused = lpl::procgen::routeLeastCost(fine, nullptr, 1u, 1u, 220u, 240u, boxed);
    check("a goal outside the corridor is refused, not approximated", !refused.found);

    lpl::procgen::Grid<lpl::core::u8> wrongSize{16u, 16u, lpl::core::u8{1}};
    lpl::procgen::RoutingParams mismatched = cost;
    mismatched.corridor = &wrongSize;
    const lpl::procgen::RoutedPath ignored =
        lpl::procgen::routeLeastCost(fine, nullptr, 20u, 20u, 220u, 240u, mismatched);
    check("a corridor of the wrong shape is ignored", ignored.found);

    lpl::procgen::RoutingParams closedCost = cost;
    closedCost.wrapColumns = kFine;
    const lpl::procgen::HierarchicalRoute wrapped =
        lpl::procgen::routeAcrossWorld(coarse, fine, kRatio, nullptr, kFine - 12u, 40u, 12u, 40u, closedCost);
    check("a closed world cascades too", wrapped.fine.found);
    check("and crosses the seam rather than going round", wrapped.fine.cells.size() <= 40u);
    std::printf("     closed cascade: %zu cells, %u+%u expanded%s\n", wrapped.fine.cells.size(), wrapped.coarseExpanded,
                wrapped.fine.expanded, wrapped.widened ? ", WIDENED" : "");

    check("and the coarse plan pointed at the seam rather than searching for it",
          wrapped.coarseExpanded <= wrapped.fine.cells.size());

    const lpl::procgen::HierarchicalRoute plainRatio =
        lpl::procgen::routeAcrossWorld(fine, fine, 1u, nullptr, 20u, 20u, 220u, 240u, cost);
    check("a ratio of one is just a route", plainRatio.fine.found && plainRatio.coarseExpanded == 0u);
}

/**
 * @brief The corridor is narrower at the seam, and it turns out not to matter.
 *
 * @warning **A gap that was looked for and could not be made to bite.** `paintCorridor` walks the plan's
 *          cells and adds slack either side; a coarse column past the last, or before the first, is
 *          skipped rather than wrapped. So on a closed world the corridor really is narrower at the seam
 *          than anywhere else, and a road squeezed by a corridor does not fail -- it comes back longer,
 *          which is the quiet failure this file keeps finding.
 *
 * Three fixtures were built to catch it and none did, for a reason worth writing down: the coarse plan
 * is drawn on the SAME terrain the fine road crosses, so wherever the fine road wants the far side of
 * the seam, the plan has already gone there -- and the plan's own cells do wrap. The slack is lost
 * exactly where neither of them had a reason to be. That is an argument, not a proof, so this stays as a
 * fixture: a road alongside the seam, with a wall the average absorbs, which must come out as cheap as
 * the flat search finds it.
 *
 * Widening the painter would be three lines and could only help -- a superset of allowed cells never
 * yields a worse road. It is not done because it is not measured: it would buy expansions everywhere for
 * a case nobody can demonstrate.
 *
 * The wall is thin enough for the average to absorb: one row, so its coarse cell reads 8.4 against a
 * floor of 1 and the plan goes straight through it rather than round. A shelf on the far side of the
 * seam is cheap enough for a fine road to prefer stepping onto it, expensive enough that the coarse plan
 * will not route along it. The corridor must not cost the road anything: a road that comes back more
 * expensive than the flat one is the quiet failure, found, valid, and worse.
 */
void checkCorridorNarrowerAtTheSeamCostsTheRoadNothing()
{
    std::printf("-- the corridor is narrower at the seam, and it turns out not to matter\n");
    constexpr lpl::core::u32 kRatio = 8u;
    constexpr lpl::core::u32 kSize = 64u;

    lpl::procgen::Heightfield fine{kSize, kSize, lpl::math::Fixed32::fromFloat(1.0f)};
    for (lpl::core::u32 x = 0u; x < 16u; ++x)
        fine.at(x, 36u) = lpl::math::Fixed32::fromFloat(60.0f);
    for (lpl::core::u32 z = 0u; z < kSize; ++z)
        for (lpl::core::u32 x = 56u; x < kSize; ++x)
            fine.at(x, z) = lpl::math::Fixed32::fromFloat(20.0f);

    const lpl::procgen::Heightfield summary = lpl::procgen::reduceHeightfield(fine, kRatio);
    lpl::procgen::RoutingParams cost{};
    cost.waterPenalty = 0.0f;
    cost.reuseDiscount = 0.0f;
    cost.wrapColumns = kSize;

    const lpl::procgen::HierarchicalRoute seam =
        lpl::procgen::routeAcrossWorld(summary, fine, kRatio, nullptr, 0u, 8u, 0u, 56u, cost);
    const lpl::procgen::RoutedPath flat = lpl::procgen::routeLeastCost(fine, nullptr, 0u, 8u, 0u, 56u, cost);

    std::printf("     flat: %zu cells, cost %.1f | cascade: %zu cells, cost %.1f, %u corridor%s\n", flat.cells.size(),
                flat.cost.toFloat(), seam.fine.cells.size(), seam.fine.cost.toFloat(), seam.corridorCells,
                seam.widened ? ", WIDENED" : "");

    check("the flat search gets past the wall", flat.found);
    check("and so does the cascade", seam.fine.found);
    check("and the corridor did not make the road worse", seam.fine.cost.raw() == flat.cost.raw());
}

/**
 * @brief An attested network is laid by the cascade, and it is the same network.
 *
 * @warning **The wiring this section exists for.** `routeAcrossWorld` had exactly one caller and it was
 *          the test above -- which is the orphan pattern this repository keeps finding, one step short
 *          of the usual form: written, documented, measured, and reachable by nothing a world runs. A
 *          corpus laying its attested roads is the caller it was waiting for.
 *
 * The places are centred on the origin, the convention TerrainRoutes shares with GridTerrain: cell c is
 * world unit c - size/2.
 *
 * The same network, cell for cell -- not merely the same length. Two roads of equal length through
 * different valleys would satisfy a count and would still mean the body walks ground the corpus never
 * routed it over.
 *
 * @warning **Without its counters the section proves nothing.** A cascade that silently fell back to a
 *          flat search would pave an identical network and pass every other check. These counters are
 *          zero unless a coarse plan actually ran and opened a corridor.
 *
 * That the cascade settles far fewer cells is the point of the whole arrangement. A ratio of zero or one
 * leaves the summary empty, so nothing about a small world changes by declaring a cascade it does not
 * need.
 *
 * @warning A body reading the same network must get the same waypoints either way, because `route()` and
 *          `paveAttested` go through one planner. They used to be two call sites, and the day one of them
 *          cascaded and the other did not, a body would walk a road laid by a different search than the
 *          one that laid the network it reads.
 */
void checkCascadeLaysTheSameAttestedNetwork()
{
    std::printf("-- an attested network is laid by the cascade, and it is the same network\n");
    constexpr lpl::core::u32 kRatio = 8u;
    constexpr lpl::core::u32 kSize = 192u;
    constexpr lpl::core::u32 kHalf = kSize / 2u;

    lpl::procgen::Heightfield fine{kSize, kSize, lpl::math::Fixed32::fromFloat(1.0f)};
    for (lpl::core::u32 z = 90u; z <= 97u; ++z)
        for (lpl::core::u32 x = 0u; x < kSize; ++x)
            if (x < 149u || x > 151u)
                fine.at(x, z) = lpl::math::Fixed32::fromFloat(60.0f);

    TableResolver places;
    places.add(1u, 20.0f - kHalf, 30.0f - kHalf);
    places.add(2u, 20.0f - kHalf, 160.0f - kHalf);
    places.link(1u, 2u);
    const lpl::core::u32 order[2] = {1u, 2u};

    lpl::engine::systems::TerrainRouteParams flatParams;
    flatParams.cost.waterPenalty = 0.0f;
    flatParams.cost.reuseDiscount = 0.0f;
    lpl::engine::systems::TerrainRoutes flatRoutes;
    flatRoutes.bind(fine, places, flatParams);
    const lpl::core::u32 flatPaved = flatRoutes.paveAttested(order, 2u);

    lpl::engine::systems::TerrainRouteParams cascadeParams = flatParams;
    cascadeParams.coarseRatio = kRatio;
    lpl::engine::systems::TerrainRoutes cascadeRoutes;
    cascadeRoutes.bind(fine, places, cascadeParams);
    const lpl::core::u32 cascadePaved = cascadeRoutes.paveAttested(order, 2u);

    std::printf("     flat: %u cells paved, %u expanded | cascade: %u paved, %u+%u expanded, "
                "%u corridor, %u widened\n",
                flatPaved, flatRoutes.expanded(), cascadePaved, cascadeRoutes.coarseExpanded(),
                cascadeRoutes.expanded(), cascadeRoutes.corridorCells(), cascadeRoutes.widened());

    check("both lay the attested road", flatPaved > 0u && flatRoutes.pairs() == 1u);
    check("and the cascade lays the same one", cascadePaved == flatPaved);

    bool identical = true;
    for (lpl::core::u32 cell = 0u; cell < fine.cellCount() && identical; ++cell)
        identical = flatRoutes.roads()[cell] == cascadeRoutes.roads()[cell];
    check("cell for cell, not merely the same length", identical);

    check("a coarse plan actually ran", cascadeRoutes.coarseExpanded() > 0u);
    check("and it opened a corridor", cascadeRoutes.corridorCells() > 0u);
    check("which the fine search fitted inside", cascadeRoutes.widened() == 0u);

    check("the cascade settles far fewer cells",
          cascadeRoutes.coarseExpanded() + cascadeRoutes.expanded() < flatRoutes.expanded());

    lpl::engine::systems::TerrainRouteParams noneParams = flatParams;
    noneParams.coarseRatio = 1u;
    lpl::engine::systems::TerrainRoutes noneRoutes;
    noneRoutes.bind(fine, places, noneParams);
    check("a ratio of one routes flat", noneRoutes.coarse().empty());
    check("and a ratio of zero does too", flatRoutes.coarse().empty());

    lpl::history::RouteLeg flatLegs[8]{};
    lpl::history::RouteLeg cascadeLegs[8]{};
    const lpl::core::u32 flatCount = flatRoutes.route(1u, 2u, flatLegs, 8u);
    const lpl::core::u32 cascadeCount = cascadeRoutes.route(1u, 2u, cascadeLegs, 8u);
    bool sameLegs = flatCount == cascadeCount && flatCount > 0u;
    for (lpl::core::u32 i = 0u; i < flatCount && sameLegs; ++i)
        sameLegs = flatLegs[i].x.raw() == cascadeLegs[i].x.raw() && flatLegs[i].z.raw() == cascadeLegs[i].z.raw();
    check("and a traveller gets the same waypoints either way", sameLegs);
}

} // namespace

/**
 * @brief The roads a corpus attests, laid across relief that refuses the straight line.
 *
 * @details The gazetteer of the first checks holds two places either side of the ridge, a third off the
 *          grid entirely, and a fourth known from texts, never found.
 *
 * @warning Written with the code it tests, because `procgen::routeLeastCost` had gone from written to
 *          documented to forgotten without ever acquiring a caller -- and a pass with no caller has no
 *          test either. Every claim here is one a straight line would also satisfy on flat ground, so
 *          every fixture puts something in the way: a ridge between the endpoints, a second road beside
 *          the first, a cell boundary through the origin.
 */
int main()
{
    const lpl::procgen::Heightfield field = ridgedPlain(kSize, kRidgeRow, kPassFrom);

    TableResolver places;
    places.add(1u, kWestX, kSouthZ);
    places.add(2u, kWestX, kNorthZ);
    places.add(3u, 900.0f, 900.0f);
    places.add(4u, 0.5f, 0.5f, false);

    lpl::core::u32 detourCount = 0u;
    lpl::history::RouteLeg detour[12]{};
    checkRoadGoesRoundARidge(field, places, detourCount, detour);
    checkFreeClimbLeavesTheRoadStraight(field, places, detourCount);
    checkRoadIsSummarisedAsStridedCorners(field, places, detourCount, detour);
    checkCellsEitherSideOfTheOriginAreTwoCells(field);
    checkPlaceNobodyHasFoundGetsNoRoad(field, places);
    checkAttestedNetworkGrowsAlongItsOwnTrunk();
    checkRoadCrossesTheAntimeridian();
    checkRoadCrossesThePole();
    checkBlockReductionDecidesWhetherThePlanSeesThePass();
    checkWorldTooLargeToSearchFlatIsSearchedTwice();
    checkCorridorNarrowerAtTheSeamCostsTheRoadNothing();
    checkCascadeLaysTheSameAttestedNetwork();

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
