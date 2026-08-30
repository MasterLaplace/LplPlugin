/**
 * @file test_terrain_routes.cpp
 * @brief The roads a corpus attests, laid across relief that refuses the straight line.
 *
 * @warning Written with the code it tests, because `procgen::routeLeastCost` had gone from written to
 * documented to forgotten without ever acquiring a caller -- and a pass with no caller has no
 * test either. Every claim here is one a straight line would also satisfy on flat ground, so
 * every fixture puts something in the way: a ridge between the endpoints, a second road beside
 * the first, a cell boundary through the origin.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/engine/systems/TerrainRoutes.hpp>
#include <lpl/history/Place.hpp>
#include <lpl/procgen/Heightfield.hpp>

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

} // namespace

int main()
{
    using namespace lpl;

    constexpr core::u32 kSize = 64u;
    constexpr core::u32 kRidgeRow = 32u;
    constexpr core::u32 kPassFrom = 56u;

    // Cell centres, so a place sits in the middle of its cell rather than on a boundary where
    // two cells would both be defensible answers.
    constexpr float kWestX = -19.5f;
    constexpr float kSouthZ = -9.5f;
    constexpr float kNorthZ = 10.5f;

    const procgen::Heightfield field = ridgedPlain(kSize, kRidgeRow, kPassFrom);

    TableResolver places;
    places.add(1u, kWestX, kSouthZ);
    places.add(2u, kWestX, kNorthZ);
    places.add(3u, 900.0f, 900.0f);    // off the grid entirely
    places.add(4u, 0.5f, 0.5f, false); // known from texts, never found

    std::printf("-- a road goes round a ridge, not through it\n");
    core::u32 detourCount = 0u;
    history::RouteLeg detour[12]{};
    {
        engine::systems::TerrainRouteParams params;
        params.cost.slopePenalty = 6.0f;
        engine::systems::TerrainRoutes routes;
        routes.bind(field, places, params);

        detourCount = routes.route(1u, 2u, detour, 12u);
        check("a road is found across the ridge line", detourCount > 0u);

        math::Fixed32 furthestEast = math::Fixed32::fromFloat(-1000.0f);
        for (core::u32 i = 0u; i < detourCount; ++i)
        {
            if (detour[i].x > furthestEast)
                furthestEast = detour[i].x;
        }
        // The pass is the eastern end of the ridge; a road that used it has to get there.
        check("it bends, rather than running straight", detourCount > 1u);
        check("and it bends toward the pass", furthestEast > math::Fixed32::fromFloat(20.0f));

        history::Place goal;
        check("the gazetteer carries the goal", places.resolve(2u, goal));
        // The last waypoint is the goal's own cell, so the walk ends where it was sent rather
        // than at the last bend before it.
        const math::Fixed32 lastX = detour[detourCount - 1u].x;
        const math::Fixed32 lastZ = detour[detourCount - 1u].z;
        check("the last waypoint is the destination cell", lastX == goal.x && lastZ == goal.z);
    }

    std::printf("-- with the climb made free, the same road is straight\n");
    {
        // The control, and the reason the fixture has a ridge at all: "the road avoided the
        // ridge" is also what a straight line does when nothing is in the way. Only a run where
        // the climb costs nothing shows that the detour was bought by the relief.
        engine::systems::TerrainRouteParams params;
        params.cost.slopePenalty = 0.0f;
        engine::systems::TerrainRoutes routes;
        routes.bind(field, places, params);

        history::RouteLeg straight[12]{};
        const core::u32 count = routes.route(1u, 2u, straight, 12u);
        check("a free climb is crossed head-on", count == 1u);
        if (count == 1u)
            check("so the only waypoint is the destination", straight[0].x < math::Fixed32::zero());
        else
            check("so the only waypoint is the destination", false);
        check("and it is a different road from the one relief produced", count != detourCount);
    }

    std::printf("-- a road is summarised as its corners, strided rather than truncated\n");
    {
        engine::systems::TerrainRouteParams params;
        params.cost.slopePenalty = 6.0f;
        engine::systems::TerrainRoutes routes;
        routes.bind(field, places, params);

        check("the detour has more corners than two", detourCount > 2u);

        history::RouteLeg tight[2]{};
        const core::u32 count = routes.route(1u, 2u, tight, 2u);
        check("a caller with two slots gets two waypoints", count == 2u);
        // The claim that separates striding from truncating: truncation keeps the FIRST corners,
        // so the last slot would hold an early bend and the body would strike out straight from
        // the middle of the ridge -- across precisely what the road existed to go round.
        check("and the last of them is still the destination",
              count == 2u && tight[1].x == detour[detourCount - 1u].x && tight[1].z == detour[detourCount - 1u].z);
        check("not the road's first bend", count == 2u && !(tight[1].x == detour[0].x && tight[1].z == detour[0].z));
    }

    std::printf("-- the two cells either side of the origin are not one cell\n");
    {
        // Truncating division maps -3 and 2 both to cell 0 when four units make a cell, so the
        // cell straddling the origin would be half as wide as every other and two places five
        // units apart would share it. They would then route to themselves, which returns nothing
        // -- a walk that quietly goes straight only near the middle of the map.
        TableResolver straddle;
        straddle.add(10u, -2.5f, 0.5f);
        straddle.add(11u, 2.5f, 0.5f);

        engine::systems::TerrainRouteParams params;
        params.cellSize = 4u;
        engine::systems::TerrainRoutes routes;
        routes.bind(field, straddle, params);

        history::RouteLeg legs[4]{};
        const core::u32 count = routes.route(10u, 11u, legs, 4u);
        check("two places five units apart do not share a four-unit cell", count >= 1u);
    }

    std::printf("-- a place nobody has found gets no road\n");
    {
        engine::systems::TerrainRouteParams params;
        engine::systems::TerrainRoutes routes;
        routes.bind(field, places, params);

        history::RouteLeg legs[8]{};
        check("an unlocated place is not routed to", routes.route(1u, 4u, legs, 8u) == 0u);
        check("nor is one off the grid", routes.route(1u, 3u, legs, 8u) == 0u);
        // Neither counts as a planned route: the search never ran, and reporting it as an
        // unreachable goal would blame the terrain for a gap in the gazetteer.
        check("and neither is blamed on the terrain", routes.planned() == 0u && routes.unreachable() == 0u);
    }

    std::printf("-- an attested network grows along its own trunk\n");
    {
        // Two roads a corpus attests, running parallel two cells apart on flat ground. With an
        // existing road discounted, the second should join the first and share it; with the
        // discount removed, it has no reason to and lays a second line beside it.
        const procgen::Heightfield flat{kSize, kSize, math::Fixed32::fromFloat(1.0f)};

        TableResolver network;
        network.add(1u, -19.5f, -19.5f);
        network.add(2u, -19.5f, 20.5f);
        network.add(3u, -17.5f, -19.5f);
        network.add(4u, -17.5f, 20.5f);
        network.link(1u, 2u);
        network.link(3u, 4u);
        const core::u32 order[4] = {1u, 2u, 3u, 4u};

        engine::systems::TerrainRouteParams merging;
        merging.cost.reuseDiscount = 0.9f;
        engine::systems::TerrainRoutes merged;
        merged.bind(flat, network, merging);
        const core::u32 pavedMerged = merged.paveAttested(order, 4u);

        engine::systems::TerrainRouteParams parallel;
        parallel.cost.reuseDiscount = 0.0f;
        engine::systems::TerrainRoutes apart;
        apart.bind(flat, network, parallel);
        const core::u32 pavedApart = apart.paveAttested(order, 4u);

        check("both attested links become road", pavedMerged > 0u && pavedApart > 0u);
        check("a discounted road is joined rather than doubled", pavedMerged < pavedApart);
        // The resolver hands out four directed links for two attested roads, so a count of four
        // here would mean each road was laid twice -- the second time taking its own discount,
        // which is a parallel road nobody built.
        check("each attested pair is laid once, not once per direction", merged.pairs() == 2u && apart.pairs() == 2u);
        // Both roads still connect: a merge that lost one of them would also pave fewer cells,
        // so the saving above means nothing without this beside it.
        check("and neither road was lost to the merge", merged.unreachable() == 0u && apart.unreachable() == 0u);
    }

    std::printf("-- a road crosses the antimeridian instead of going round the planet\n");
    {
        // Flat ground, so the ONLY thing deciding the route is the shape of the world.
        constexpr core::u32 kW = 200u;
        constexpr core::u32 kH = 24u;
        procgen::Heightfield flat{kW, kH, math::Fixed32::zero()};

        procgen::RoutingParams params{};
        params.waterPenalty = 0.0f;
        params.reuseDiscount = 0.0f;

        // Five cells west of the seam to five cells east of it: ten cells apart the short way,
        // one hundred and ninety the long way.
        const core::u32 startX = kW - 5u;
        const core::u32 goalX = 5u;

        procgen::RoutingParams open = params;
        const procgen::RoutedPath around = procgen::routeLeastCost(flat, nullptr, startX, 12u, goalX, 12u, open);
        check("an open world still finds a road", around.found);
        check("but it goes the long way round", around.cells.size() > 100u);

        procgen::RoutingParams closed = params;
        closed.wrapColumns = kW;
        const procgen::RoutedPath through = procgen::routeLeastCost(flat, nullptr, startX, 12u, goalX, 12u, closed);
        check("a closed world finds one too", through.found);

        // @warning THE assertion. Both routes exist, both are valid, and the long one is a perfectly
        // ordinary road -- which is why nothing downstream could tell it was the wrong answer.
        check("and it crosses the seam instead", through.cells.size() <= 12u);
        check("which is far shorter than going round", through.cells.size() * 5u < around.cells.size());
        check("and it costs less", through.cost < around.cost);

        // @warning The heuristic is the half that fails SILENTLY. An unwrapped one overestimates by
        // nearly a circumference at the seam, so A* settles the whole planet before conceding the
        // short way -- it still returns a road, just after doing enormously more work. Expanding
        // fewer cells than the long route has is what says the heuristic actually pointed at the
        // seam rather than being rescued by exhaustion.
        check("without searching the whole world to find it", through.expanded < around.cells.size());
        std::printf("     seam route %zu cells, %u expanded; the long way %zu cells, %u expanded\n",
                    through.cells.size(), through.expanded, around.cells.size(), around.expanded);

        // The route must actually step ACROSS the boundary, not merely be short.
        bool steppedOverSeam = false;
        for (std::size_t i = 1u; i < through.cells.size(); ++i)
        {
            const core::i32 previous = static_cast<core::i32>(through.cells[i - 1u] % kW);
            const core::i32 here = static_cast<core::i32>(through.cells[i] % kW);
            if ((previous > here ? previous - here : here - previous) > 1)
                steppedOverSeam = true;
        }
        check("and the path really steps over the boundary", steppedOverSeam);

        // A wrap that does not match the grid is ignored rather than half-applied: a router that
        // wrapped its neighbours at one width and its heuristic at another would be inadmissible
        // in a way no single line of it looks wrong.
        procgen::RoutingParams mismatched = params;
        mismatched.wrapColumns = kW + 1u;
        const procgen::RoutedPath ignored = procgen::routeLeastCost(flat, nullptr, startX, 12u, goalX, 12u, mismatched);
        check("a wrap that does not match the grid is ignored", ignored.cells.size() == around.cells.size());
    }

    std::printf("-- and over the pole, which is a shortcut and not a wrap\n");
    {
        // Flat ground again: the only thing deciding the route is the shape of the world.
        constexpr core::u32 kW = 120u;
        constexpr core::u32 kH = 40u;
        procgen::Heightfield flat{kW, kH, math::Fixed32::zero()};

        procgen::RoutingParams closed{};
        closed.waterPenalty = 0.0f;
        closed.reuseDiscount = 0.0f;
        closed.wrapColumns = kW;

        procgen::RoutingParams polar = closed;
        polar.wrapPoles = true;

        // Two places at high latitude on OPPOSITE meridians: three rows from the top, half a world
        // apart. Going round the sheet is sixty columns; going over the pole is about seven rows.
        const core::u32 startX = 10u;
        const core::u32 goalX = 10u + kW / 2u;
        const core::u32 row = 3u;

        const procgen::RoutedPath around = procgen::routeLeastCost(flat, nullptr, startX, row, goalX, row, closed);
        const procgen::RoutedPath over = procgen::routeLeastCost(flat, nullptr, startX, row, goalX, row, polar);
        check("a world without poles still finds a road", around.found);
        check("and so does one with them", over.found);

        // @warning The whole point: two points at high latitude on opposite meridians really ARE
        // closer over the pole, and a router that cannot cross one lays a road all the way round
        // instead -- a perfectly valid, perfectly long road that nothing downstream can question.
        check("the polar route is shorter", over.cells.size() < around.cells.size());
        check("and it costs less", over.cost < around.cost);
        std::printf("     over the pole %zu cells (%u expanded), round the sheet %zu (%u expanded)\n",
                    over.cells.size(), over.expanded, around.cells.size(), around.expanded);

        // It must actually TOUCH the top row and jump half a world there, rather than merely being
        // short by some other means.
        bool touchedPole = false;
        for (std::size_t i = 1u; i < over.cells.size(); ++i)
        {
            const core::i32 previousZ = static_cast<core::i32>(over.cells[i - 1u] / kW);
            const core::i32 hereZ = static_cast<core::i32>(over.cells[i] / kW);
            const core::i32 previousX = static_cast<core::i32>(over.cells[i - 1u] % kW);
            const core::i32 hereX = static_cast<core::i32>(over.cells[i] % kW);
            const core::i32 jump = previousX > hereX ? previousX - hereX : hereX - previousX;
            if (previousZ == 0 && hereZ == 0 && jump > 1)
                touchedPole = true;
        }
        check("the road really crosses the pole", touchedPole);

        // @warning **A pole is not a wrap.** Going north off the top must come back on the TOP row,
        // not the bottom: the other reading joins the Arctic to the Antarctic, and a route computed
        // that way would look like an ordinary road through a place nobody can walk.
        bool reachedFarSide = false;
        for (core::u32 cell : over.cells)
            if (cell / kW >= kH - 2u)
                reachedFarSide = true;
        check("and never comes out at the other pole", !reachedFarSide);

        // Two points near the EQUATOR of this sheet are not helped by a pole, and the router must
        // not invent a detour: the estimate has to stay the direct one where the direct one wins.
        const core::u32 mid = kH / 2u;
        const procgen::RoutedPath flatRun = procgen::routeLeastCost(flat, nullptr, 10u, mid, 30u, mid, polar);
        check("a short hop mid-sheet is still direct", flatRun.found && flatRun.cells.size() <= 22u);

        // @warning The heuristic must stay ADMISSIBLE, and this is what says so: with poles enabled
        // the router must never return a road MORE expensive than the same search without the
        // shortcut. An estimate that overestimated a polar route would discard it unevaluated and
        // hand back the long way while looking entirely healthy.
        check("enabling poles never makes a road worse", over.cost <= around.cost);

        // And a pole is refused on a grid that does not close east-west, because half a world of
        // nothing is nothing.
        procgen::RoutingParams openSheet{};
        openSheet.waterPenalty = 0.0f;
        openSheet.reuseDiscount = 0.0f;
        openSheet.wrapPoles = true; // no wrapColumns
        const procgen::RoutedPath refused = procgen::routeLeastCost(flat, nullptr, startX, row, goalX, row, openSheet);
        check("poles alone do nothing on an open sheet",
              refused.cells.size() == around.cells.size() || refused.cells.size() > over.cells.size());
    }

    std::printf("-- how a block is reduced decides whether the plan can see the pass\n");
    {
        // @warning **This measurement exists because the tree disagreed with itself.** The cascade's
        // own documentation says a coarse cell is an AVERAGE; the only code that ever built one took
        // the MAXIMUM and argued against the mean in a comment. Both were reasoning, neither was
        // measured, and the fixture that measured nothing was a one-cell ridge with a pass nine
        // cells wide -- wider than the block, and aligned to it, so every rule found it.
        //
        // A real range is thick and its pass is narrow, and neither is aligned to a lattice chosen
        // for a memory budget. That is the case each rule has to survive.
        constexpr core::u32 kRatio = 8u;
        constexpr core::u32 kSize = 192u;
        constexpr core::u32 kRidgeTop = 90u; // deliberately not a multiple of the ratio
        constexpr core::u32 kRidgeBottom = 97u;
        constexpr core::u32 kGateFrom = 149u; // three cells wide, inside one block
        constexpr core::u32 kGateTo = 151u;

        procgen::Heightfield fine{kSize, kSize, math::Fixed32::fromFloat(1.0f)};
        for (core::u32 z = kRidgeTop; z <= kRidgeBottom; ++z)
            for (core::u32 x = 0u; x < kSize; ++x)
                if (x < kGateFrom || x > kGateTo)
                    fine.at(x, z) = math::Fixed32::fromFloat(60.0f);

        const core::u32 coarseWidth = (kSize + kRatio - 1u) / kRatio;
        const auto reduceBy = [&](int rule) {
            procgen::Heightfield out{coarseWidth, coarseWidth, math::Fixed32::zero()};
            for (core::u32 cz = 0u; cz < coarseWidth; ++cz)
                for (core::u32 cx = 0u; cx < coarseWidth; ++cx)
                {
                    math::Fixed32 hi = math::Fixed32::min();
                    math::Fixed32 lo = math::Fixed32::max();
                    for (core::u32 fz = 0u; fz < kRatio; ++fz)
                        for (core::u32 fx = 0u; fx < kRatio; ++fx)
                        {
                            const math::Fixed32 here = fine.at(cx * kRatio + fx, cz * kRatio + fz);
                            if (here > hi)
                                hi = here;
                            if (here < lo)
                                lo = here;
                        }
                    out.at(cx, cz) = rule == 0 ? hi : lo;
                }
            return out;
        };

        procgen::RoutingParams cost{};
        cost.waterPenalty = 0.0f;
        cost.reuseDiscount = 0.0f;

        constexpr core::u32 kStartX = 20u;
        constexpr core::u32 kStartZ = 30u;
        constexpr core::u32 kGoalZ = 160u;

        // The flat search is the ground truth: what the road IS, at full resolution, with no
        // summary involved. Every rule below is judged against it and against nothing chosen here.
        const procgen::RoutedPath truth =
            procgen::routeLeastCost(fine, nullptr, kStartX, kStartZ, kStartX, kGoalZ, cost);
        check("the flat search crosses the range", truth.found);

        // Where a road crosses the range is the whole question: through the pass, or over the wall.
        const auto crossesAtGate = [&](const procgen::RoutedPath &path) {
            for (const core::u32 cell : path.cells)
            {
                const core::u32 x = cell % kSize;
                const core::u32 z = cell / kSize;
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
            const procgen::Heightfield summary = rule == 2 ? procgen::reduceHeightfield(fine, kRatio) : reduceBy(rule);
            const procgen::HierarchicalRoute cascaded =
                procgen::routeAcrossWorld(summary, fine, kRatio, nullptr, kStartX, kStartZ, kStartX, kGoalZ, cost);
            sawGate[rule] = cascaded.fine.found && crossesAtGate(cascaded.fine);
            optimal[rule] = cascaded.fine.found && cascaded.fine.cost.raw() == truth.cost.raw();
            widened[rule] = cascaded.widened;
            std::printf("     %s: %s, %zu cells, cost %.1f, %u+%u expanded%s, %s\n", names[rule],
                        cascaded.fine.found ? "found" : "NO ROAD", cascaded.fine.cells.size(),
                        cascaded.fine.cost.toFloat(), cascaded.coarseExpanded, cascaded.fine.expanded,
                        cascaded.widened ? ", WIDENED" : "", sawGate[rule] ? "through the pass" : "OVER THE WALL");
        }

        // @warning **Both rivals throw away one of the two facts the plan needs, in opposite
        // directions.** The maximum keeps the wall and erases the gap: a block holding a
        // three-cell pass also holds five cells of ridge, so it reads as solid ridge. The minimum
        // keeps the gap and erases the wall: a block straddling the range's edge also holds valley
        // floor, so it reads as valley and the plan crosses wherever it likes. Only the mean is
        // sensitive to the PROPORTION, which is the one thing that distinguishes "a wall with a
        // door in it" from either a wall or a door.
        check("the mean summary routes through the pass", sawGate[2]);
        check("the maximum summary does not", !sawGate[0]);
        check("nor does the minimum", !sawGate[1]);

        // Not "close to" the flat road: the SAME road. A corridor that contains the optimum lets
        // A* return the optimum, and stating it exactly is what would notice a corridor that
        // clipped a corner off it.
        check("and the road it returns is the flat road, exactly", optimal[2]);
        check("while the rivals return a costlier one", !optimal[0] && !optimal[1]);

        // @warning **The lesson that is easy to miss, and the reason this is measured at all.**
        // Neither rival WIDENED. The recover-once-and-report path only catches a plan that turns
        // out to be impassable; a plan that merely hides a cheaper way is passable, so the fine
        // search finds a road, reports success, and hands back something 2.65 times the cost of
        // the real one. A bad summary does not fail loudly -- it succeeds quietly.
        check("and neither of them widened, so nothing reported the mistake", !widened[0] && !widened[1]);
    }

    std::printf("-- a world too large to search flat is searched twice\n");
    {
        // @warning **The measurement that forces the shape.** A* keeps cost, parent and a settled flag
        // per cell -- about twelve bytes. A global grid at thirty-metre cells is 890 733 444 400
        // cells, roughly ten TERABYTES; at 7.7 km it is 163 MB. A planet cannot be searched flat.
        constexpr core::u32 kRatio = 8u;
        constexpr core::u32 kFine = 256u;
        constexpr core::u32 kCoarse = kFine / kRatio;

        // A ridge across the middle with one pass through it, so the route has a shape to find
        // rather than a straight line to walk: a corridor around a straight line would confine
        // nothing and the comparison below would be vacuous.
        procgen::Heightfield fine{kFine, kFine, math::Fixed32::fromFloat(1.0f)};
        constexpr core::u32 kRidge = 128u;
        constexpr core::u32 kGate = 200u;
        for (core::u32 x = 0u; x < kFine; ++x)
            if (x < kGate || x > kGate + 8u)
                fine.at(x, kRidge) = math::Fixed32::fromFloat(60.0f);

        // @warning This used to build the summary by hand, taking the MAXIMUM of each block on the
        // stated grounds that "a mean would let a wall average away into a slope". The section
        // above measures that claim on a range a summary can actually get wrong, and it is false:
        // the maximum erases the pass and returns a road 2.65 times the cost of the real one,
        // without widening, so nothing reports it. One reducer in the tree, and it is the mean.
        const procgen::Heightfield coarse = procgen::reduceHeightfield(fine, kRatio);
        check("the summary is one cell per block", coarse.width() == kCoarse && coarse.depth() == kCoarse);

        procgen::RoutingParams cost{};
        cost.waterPenalty = 0.0f;
        cost.reuseDiscount = 0.0f;

        const procgen::RoutedPath flat = procgen::routeLeastCost(fine, nullptr, 20u, 20u, 220u, 240u, cost);
        const procgen::HierarchicalRoute cascaded =
            procgen::routeAcrossWorld(coarse, fine, kRatio, nullptr, 20u, 20u, 220u, 240u, cost);

        check("the flat search finds a road", flat.found);
        check("and so does the cascade", cascaded.fine.found);
        check("the coarse plan existed", cascaded.coarseFound);

        std::printf("     flat: %zu cells, %u expanded | cascade: %zu cells, %u+%u expanded, "
                    "%u corridor cells%s\n",
                    flat.cells.size(), flat.expanded, cascaded.fine.cells.size(), cascaded.coarseExpanded,
                    cascaded.fine.expanded, cascaded.corridorCells, cascaded.widened ? ", WIDENED" : "");

        // @warning **The whole justification.** If the cascade settled as many cells as the flat search,
        // it would be the same search with extra steps -- and the ten terabytes above would still
        // be ten terabytes. The corridor is what makes a planet affordable.
        check("the cascade settles far fewer cells", cascaded.coarseExpanded + cascaded.fine.expanded < flat.expanded);

        // And the road must be worth having. A corridor that squeezed the route into a detour would
        // buy its cheapness with a road nobody would drive; within a quarter is the bound, and it is
        // measured against the flat answer rather than against a number chosen here.
        check("and the road it finds is comparable",
              cascaded.fine.cost <= flat.cost + flat.cost * math::Fixed32::fromFloat(0.25f));

        // @warning A corridor constrains the SEARCH, never the ground. A goal outside it must make the
        // search FAIL rather than wander out and return a road nobody planned -- which is why this
        // is a separate field and not a very large cost.
        procgen::Grid<core::u8> narrow{kFine, kFine, core::u8{0}};
        for (core::u32 z = 0u; z < 8u; ++z)
            for (core::u32 x = 0u; x < 8u; ++x)
                narrow.at(x, z) = 1u;
        procgen::RoutingParams boxed = cost;
        boxed.corridor = &narrow;
        const procgen::RoutedPath refused = procgen::routeLeastCost(fine, nullptr, 1u, 1u, 220u, 240u, boxed);
        check("a goal outside the corridor is refused, not approximated", !refused.found);

        // A corridor that does not match the grid is ignored rather than half applied.
        procgen::Grid<core::u8> wrongSize{16u, 16u, core::u8{1}};
        procgen::RoutingParams mismatched = cost;
        mismatched.corridor = &wrongSize;
        const procgen::RoutedPath ignored = procgen::routeLeastCost(fine, nullptr, 20u, 20u, 220u, 240u, mismatched);
        check("a corridor of the wrong shape is ignored", ignored.found);

        // @warning **A cascade on a CLOSED world, which is what the coarse plan's own wrap is for.**
        // The coarse grid is narrower than the fine one by the cell ratio, so a coarse heuristic
        // folding at the FINE width folds at a width its grid does not have -- it stops being
        // admissible exactly at the seam, and the plan comes back round the long way while looking
        // like any other plan. The two places sit either side of it.
        procgen::RoutingParams closedCost = cost;
        closedCost.wrapColumns = kFine;
        const procgen::HierarchicalRoute wrapped =
            procgen::routeAcrossWorld(coarse, fine, kRatio, nullptr, kFine - 12u, 40u, 12u, 40u, closedCost);
        check("a closed world cascades too", wrapped.fine.found);
        check("and crosses the seam rather than going round", wrapped.fine.cells.size() <= 40u);
        std::printf("     closed cascade: %zu cells, %u+%u expanded%s\n", wrapped.fine.cells.size(),
                    wrapped.coarseExpanded, wrapped.fine.expanded, wrapped.widened ? ", WIDENED" : "");

        // @warning **The number that catches a coarse heuristic folding at the wrong width.** Both
        // widths find the same road here -- the grid is small enough that exhaustion rescues a bad
        // estimate -- so the road alone says nothing. What moves is the WORK: measured, 4 coarse
        // expansions with the coarse grid's own width against 30 with the fine one's. The bound is
        // derived rather than tuned: a coarse plan that pointed at the seam settles on the order of
        // the road's length, never more than it.
        check("and the coarse plan pointed at the seam rather than searching for it",
              wrapped.coarseExpanded <= wrapped.fine.cells.size());

        // A ratio of one is a plain route, not a cascade pretending to be one.
        const procgen::HierarchicalRoute plainRatio =
            procgen::routeAcrossWorld(fine, fine, 1u, nullptr, 20u, 20u, 220u, 240u, cost);
        check("a ratio of one is just a route", plainRatio.fine.found && plainRatio.coarseExpanded == 0u);
    }

    std::printf("-- the corridor is narrower at the seam, and it turns out not to matter\n");
    {
        // @warning **A gap that was looked for and could not be made to bite.** `paintCorridor` walks
        // the plan's cells and adds slack either side; a coarse column past the last, or before the
        // first, is skipped rather than wrapped. So on a closed world the corridor really is
        // narrower at the seam than anywhere else, and a road squeezed by a corridor does not fail
        // -- it comes back longer, which is the quiet failure this file keeps finding.
        //
        // Three fixtures were built to catch it and none did, for a reason worth writing down: the
        // coarse plan is drawn on the SAME terrain the fine road crosses, so wherever the fine road
        // wants the far side of the seam, the plan has already gone there -- and the plan's own
        // cells do wrap. The slack is lost exactly where neither of them had a reason to be. That is
        // an argument, not a proof, so this stays as a fixture: a road alongside the seam, with a
        // wall the average absorbs, which must come out as cheap as the flat search finds it.
        //
        // Widening the painter would be three lines and could only help -- a superset of allowed
        // cells never yields a worse road. It is not done because it is not measured: it would buy
        // expansions everywhere for a case nobody can demonstrate.
        constexpr core::u32 kRatio = 8u;
        constexpr core::u32 kSize = 64u;

        procgen::Heightfield fine{kSize, kSize, math::Fixed32::fromFloat(1.0f)};
        // A wall thin enough for the average to absorb: one row, so its coarse cell reads 8.4
        // against a floor of 1 and the plan goes straight through it rather than round.
        for (core::u32 x = 0u; x < 16u; ++x)
            fine.at(x, 36u) = math::Fixed32::fromFloat(60.0f);
        // And a shelf on the far side of the seam: cheap enough for a fine road to prefer stepping
        // onto it, expensive enough that the coarse plan will not route along it.
        for (core::u32 z = 0u; z < kSize; ++z)
            for (core::u32 x = 56u; x < kSize; ++x)
                fine.at(x, z) = math::Fixed32::fromFloat(20.0f);

        const procgen::Heightfield summary = procgen::reduceHeightfield(fine, kRatio);
        procgen::RoutingParams cost{};
        cost.waterPenalty = 0.0f;
        cost.reuseDiscount = 0.0f;
        cost.wrapColumns = kSize;

        const procgen::HierarchicalRoute seam =
            procgen::routeAcrossWorld(summary, fine, kRatio, nullptr, 0u, 8u, 0u, 56u, cost);
        const procgen::RoutedPath flat = procgen::routeLeastCost(fine, nullptr, 0u, 8u, 0u, 56u, cost);

        std::printf("     flat: %zu cells, cost %.1f | cascade: %zu cells, cost %.1f, %u corridor%s\n",
                    flat.cells.size(), flat.cost.toFloat(), seam.fine.cells.size(), seam.fine.cost.toFloat(),
                    seam.corridorCells, seam.widened ? ", WIDENED" : "");

        check("the flat search gets past the wall", flat.found);
        check("and so does the cascade", seam.fine.found);
        // The corridor must not cost the road anything. A road that comes back more expensive than
        // the flat one is the quiet failure: found, valid, and worse.
        check("and the corridor did not make the road worse", seam.fine.cost.raw() == flat.cost.raw());
    }

    std::printf("-- an attested network is laid by the cascade, and it is the same network\n");
    {
        // @warning **The wiring this section exists for.** `routeAcrossWorld` had exactly one caller
        // and it was the test above -- which is the orphan pattern this repository keeps finding,
        // one step short of the usual form: written, documented, measured, and reachable by nothing
        // a world runs. A corpus laying its attested roads is the caller it was waiting for.
        constexpr core::u32 kRatio = 8u;
        constexpr core::u32 kSize = 192u;
        constexpr core::u32 kHalf = kSize / 2u;

        procgen::Heightfield fine{kSize, kSize, math::Fixed32::fromFloat(1.0f)};
        for (core::u32 z = 90u; z <= 97u; ++z)
            for (core::u32 x = 0u; x < kSize; ++x)
                if (x < 149u || x > 151u)
                    fine.at(x, z) = math::Fixed32::fromFloat(60.0f);

        // Centred on the origin, the convention TerrainRoutes shares with GridTerrain: cell c is
        // world unit c - size/2.
        TableResolver places;
        places.add(1u, 20.0f - kHalf, 30.0f - kHalf);
        places.add(2u, 20.0f - kHalf, 160.0f - kHalf);
        places.link(1u, 2u);
        const core::u32 order[2] = {1u, 2u};

        engine::systems::TerrainRouteParams flatParams;
        flatParams.cost.waterPenalty = 0.0f;
        flatParams.cost.reuseDiscount = 0.0f;
        engine::systems::TerrainRoutes flatRoutes;
        flatRoutes.bind(fine, places, flatParams);
        const core::u32 flatPaved = flatRoutes.paveAttested(order, 2u);

        engine::systems::TerrainRouteParams cascadeParams = flatParams;
        cascadeParams.coarseRatio = kRatio;
        engine::systems::TerrainRoutes cascadeRoutes;
        cascadeRoutes.bind(fine, places, cascadeParams);
        const core::u32 cascadePaved = cascadeRoutes.paveAttested(order, 2u);

        std::printf("     flat: %u cells paved, %u expanded | cascade: %u paved, %u+%u expanded, "
                    "%u corridor, %u widened\n",
                    flatPaved, flatRoutes.expanded(), cascadePaved, cascadeRoutes.coarseExpanded(),
                    cascadeRoutes.expanded(), cascadeRoutes.corridorCells(), cascadeRoutes.widened());

        check("both lay the attested road", flatPaved > 0u && flatRoutes.pairs() == 1u);
        check("and the cascade lays the same one", cascadePaved == flatPaved);

        // The same network, cell for cell -- not merely the same length. Two roads of equal length
        // through different valleys would satisfy a count and would still mean the body walks
        // ground the corpus never routed it over.
        bool identical = true;
        for (core::u32 cell = 0u; cell < fine.cellCount() && identical; ++cell)
            identical = flatRoutes.roads()[cell] == cascadeRoutes.roads()[cell];
        check("cell for cell, not merely the same length", identical);

        // @warning **Without this the section proves nothing.** A cascade that silently fell back to a
        // flat search would pave an identical network and pass every check above. These counters are
        // zero unless a coarse plan actually ran and opened a corridor.
        check("a coarse plan actually ran", cascadeRoutes.coarseExpanded() > 0u);
        check("and it opened a corridor", cascadeRoutes.corridorCells() > 0u);
        check("which the fine search fitted inside", cascadeRoutes.widened() == 0u);

        // The point of the whole arrangement.
        check("the cascade settles far fewer cells",
              cascadeRoutes.coarseExpanded() + cascadeRoutes.expanded() < flatRoutes.expanded());

        // A ratio of zero or one leaves the summary empty, so nothing about a small world changes
        // by declaring a cascade it does not need.
        engine::systems::TerrainRouteParams noneParams = flatParams;
        noneParams.coarseRatio = 1u;
        engine::systems::TerrainRoutes noneRoutes;
        noneRoutes.bind(fine, places, noneParams);
        check("a ratio of one routes flat", noneRoutes.coarse().empty());
        check("and a ratio of zero does too", flatRoutes.coarse().empty());

        // @warning A body reading the same network must get the same waypoints either way, because
        // `route()` and `paveAttested` go through one planner. They used to be two call sites, and
        // the day one of them cascaded and the other did not, a body would walk a road laid by a
        // different search than the one that laid the network it reads.
        history::RouteLeg flatLegs[8]{};
        history::RouteLeg cascadeLegs[8]{};
        const core::u32 flatCount = flatRoutes.route(1u, 2u, flatLegs, 8u);
        const core::u32 cascadeCount = cascadeRoutes.route(1u, 2u, cascadeLegs, 8u);
        bool sameLegs = flatCount == cascadeCount && flatCount > 0u;
        for (core::u32 i = 0u; i < flatCount && sameLegs; ++i)
            sameLegs = flatLegs[i].x.raw() == cascadeLegs[i].x.raw() && flatLegs[i].z.raw() == cascadeLegs[i].z.raw();
        check("and a traveller gets the same waypoints either way", sameLegs);
    }

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
