/**
 * @file test_procgen_shapegrammar.cpp
 * @brief Probes for the split grammar, its string form, and the town it raises.
 *
 * The claims worth testing:
 *
 *  1. **A malformed grammar is refused**, not half-parsed. A grammar that parses
 *     as far as it goes produces a building that looks plausible and is wrong.
 *  2. **Weights mean what they say.** `:2` really is twice as likely as `:1`, or
 *     the notation is decoration.
 *  3. **No building stands in its own street**, and none escapes its plot. Both
 *     have happened in this module already.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/ShapeGrammar.hpp>
#include <lpl/procgen/WorldBuilder.hpp>

#include <lpl/ecs/Registry.hpp>

#include <cstdio>

namespace {

using namespace lpl;

int gChecks = 0;
int gFailures = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    if (!condition)
    {
        ++gFailures;
        std::printf("  FAIL %s\n", what);
    }
}

void testParser()
{
    std::printf("the grammar string parses, and refuses what it cannot read\n");

    procgen::SequenceGrammar grammar;
    check(procgen::parseSequenceGrammar("{[A,P]:2,[BL,P]:1,[BS,P]:1}*,[G,P]", grammar),
          "the survey's own example parses");
    check(grammar.alternativeCount == 3u, "three alternatives");
    check(grammar.totalWeight == 4u, "weights sum to 2+1+1");
    check(grammar.hasTerminator, "the trailing module is a terminator");
    check(grammar.alternatives[0].material == procgen::materialForSymbol('A'), "the first symbol picks the material");
    check(grammar.alternatives[0].height == 2u, "[A,P] is two symbols tall");

    // Every one of these is a plausible typo, and every one must be refused
    // rather than parsed as far as it goes.
    const char *malformed[] = {
        "",                       // nothing
        "{[A]}",                  // no repeat operator
        "{[A]*",                  // unclosed block
        "[A]*",                   // no block
        "{[A]:0}*",               // a zero weight is not a weight
        "{[A]:}*",                // a colon with no number
        "{[]}*",                  // an empty module
        "{[A]}*,",                // a comma promising a terminator that is not there
        "{[A]}* junk",            // trailing garbage
        "{[A],}*",                // a dangling separator
    };
    for (const char *text : malformed)
    {
        procgen::SequenceGrammar rejected;
        const bool parsed = procgen::parseSequenceGrammar(text, rejected);
        if (parsed)
            std::printf("    accepted malformed grammar: \"%s\"\n", text);
        check(!parsed, "a malformed grammar is refused");
    }
}

void testWeightsAreHonoured()
{
    std::printf("declared weights are the actual frequencies\n");

    procgen::SequenceGrammar grammar;
    check(procgen::parseSequenceGrammar("{[A]:3,[B]:1}*", grammar), "a weighted grammar parses");

    lpl::pmr::vector<procgen::GrammarModule> modules;
    const core::u32 slots = 4000u;
    procgen::applySequence(grammar, slots, 20260728u, modules);

    core::u32 a = 0u;
    for (core::u32 i = 0u; i < modules.size(); ++i)
        if (modules[i].material == procgen::materialForSymbol('A'))
            ++a;

    const float share = static_cast<float>(a) / static_cast<float>(slots);
    std::printf("    [A]:3 against [B]:1 gave %.3f (expected 0.750)\n", share);
    check(share > 0.72f && share < 0.78f, "the 3:1 weighting is honoured");

    // Same seed, same sequence — the whole point of a keyed stream.
    lpl::pmr::vector<procgen::GrammarModule> twin;
    procgen::applySequence(grammar, slots, 20260728u, twin);
    bool identical = twin.size() == modules.size();
    for (core::u32 i = 0u; identical && i < twin.size(); ++i)
        identical = twin[i].material == modules[i].material;
    check(identical, "the sequence is reproducible from its seed");
}

void testBuildingsStayOnTheirPlots()
{
    std::printf("buildings stay on their plots, and off their streets\n");

    procgen::WorldBuilder builder{7331u};
    procgen::SettlementParams town;
    town.districtSize = 10u;

    builder.terrain(96u, 96u).normalize(-2.0f, 6.0f).settlement(town);

    procgen::BuildingGrammarParams grammar;
    grammar.minFloors = 1u;
    grammar.maxFloors = 5u;
    grammar.roofHeight = 2u;
    builder.buildings(grammar);

    const procgen::VoxelVolume &volume = builder.townVolume();
    const procgen::SettlementMap &map = builder.settlementMap();
    check(!volume.empty(), "the town has volume");

    core::u32 solid = 0u;
    core::u32 onRoad = 0u;
    core::u32 offPlot = 0u;
    core::u32 tallestPlot = 0u;
    core::u32 tallestRoad = 0u;

    for (core::u32 z = 0u; z < volume.depth; ++z)
    {
        for (core::u32 x = 0u; x < volume.width; ++x)
        {
            core::u32 column = 0u;
            for (core::u32 y = 0u; y < volume.levels; ++y)
                if (volume.at(x, y, z) != 0u)
                {
                    ++solid;
                    column = y + 1u;
                }
            if (column == 0u)
                continue;

            const procgen::SettlementCell cell = map.at(x, z);
            if (cell == procgen::SettlementCell::Road || cell == procgen::SettlementCell::Plaza)
            {
                ++onRoad;
                if (column > tallestRoad)
                    tallestRoad = column;
            }
            else if (cell != procgen::SettlementCell::Plot)
            {
                ++offPlot;
            }
            else if (column > tallestPlot)
            {
                tallestPlot = column;
            }
        }
    }

    std::printf("    %u solid voxels, tallest plot column %u, tallest road column %u\n", solid, tallestPlot,
                tallestRoad);
    check(solid > 0u, "something was actually raised");
    check(onRoad == 0u, "no building occupies a street");
    check(offPlot == 0u, "no building occupies ground the map does not call a plot");
    check(tallestPlot > 1u, "buildings have more than one storey");

    // Adding a plot must not redraw the town: the per-plot stream is keyed to the
    // plot's position, not to its index in the list.
    procgen::WorldBuilder twin{7331u};
    twin.terrain(96u, 96u).normalize(-2.0f, 6.0f).settlement(town).buildings(grammar);
    check(procgen::foldVolume(twin.townVolume()) == procgen::foldVolume(volume), "the town is reproducible");
}

void testRoadsideDecoration()
{
    std::printf("the same grammar decorates a line\n");

    procgen::WorldBuilder builder{4242u};
    procgen::SettlementParams town;
    town.districtSize = 12u;
    builder.terrain(80u, 80u).normalize(-2.0f, 6.0f).settlement(town).roads();

    const procgen::BuiltWorldStats before = builder.bakeGrids();
    builder.roadside("{[A]:3,[B]:1}*,[G]", 2u);
    const procgen::BuiltWorldStats after = builder.bakeGrids();

    std::printf("    %u road cells, %u modules placed\n", before.roadCells, after.roadsideModules);
    check(before.roadCells > 0u, "there are roads to decorate");
    check(after.roadsideModules > 0u, "modules were placed along them");
    check(after.roadsideModules <= before.roadCells, "no more modules than there are road cells");

    // A refused grammar must decorate nothing rather than decorate partially.
    procgen::WorldBuilder broken{4242u};
    broken.terrain(80u, 80u).normalize(-2.0f, 6.0f).settlement(town).roads().roadside("{[A]:3", 2u);
    check(broken.bakeGrids().roadsideModules == 0u, "a refused grammar places nothing");
}

} // namespace

int main()
{
    std::printf("== procgen shape grammar ==\n");
    testParser();
    testWeightsAreHonoured();
    testBuildingsStayOnTheirPlots();
    testRoadsideDecoration();

    if (gFailures == 0)
        std::printf("\nALL PASS (0 failures, %d checks)\n", gChecks);
    else
        std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
