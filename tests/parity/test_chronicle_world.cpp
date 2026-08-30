/**
 * @file test_chronicle_world.cpp
 * @brief The corpus, the roads and the walk, on real ground, without a window.
 *
 * @warning Written with the sample, because the tree already knows what happens otherwise: the
 * creature loop in `apps/mapview/main.cpp` drifted in BOTH directions -- it learned two things
 * the engine had not, and never learned that a scent channel means something -- and nothing saw
 * it, because code in an app's `main.cpp` has no test target at all. `ChronicleWorld` runs the
 * newest and least-exercised things in the tree (a hierarchical route, an attested network, a
 * body walking a corpus's roads), so it gets one on the day it is written rather than after it
 * has quietly stopped working.
 *
 * It runs the World for real -- `LinuxPlatform` provides a framebuffer nobody looks at, which is
 * exactly what a headless check of a drawing world wants.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <cstdio>

#include <lpl/engine/Config.hpp>
#include <lpl/engine/ResourceManager.hpp>
#include <lpl/memory/ArenaAllocator.hpp>
#include <lpl/platform/linux/LinuxPlatform.hpp>
#include <lpl/procgen/WorldRecipe.hpp>
#include <lpl/samples/ChronicleWorld.hpp>

using namespace lpl;

namespace {

int gChecks = 0;
int gFailures = 0;

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
    std::printf("== the chronicle world ==\n");

    // ── The gazetteer puts places where they really are ──────────────────────
    std::printf("-- a coordinate is a fact about the world, a cell is one about a lattice\n");
    {
        procgen::WorldRecipe recipe = procgen::parityWorldRecipe();
        recipe.width = 64u;
        recipe.depth = 64u;
        const math::ReliefField field = samples::makeRealReliefField(recipe, recipe.terrain.amplitude);

        samples::ManiGazetteer gazetteer;
        gazetteer.bind(field, 64u);

        // @warning Checked against the PROJECTION rather than against remembered cell numbers. A
        // pinned cell index would pass for the survey it was written on and silently point at a
        // different valley the day the window moves -- which is the whole reason places are
        // carried as latitude and longitude.
        core::u32 inside = 0u;
        core::u32 outside = 0u;
        for (core::u32 i = 0u; i < samples::kManiPlaceCount; ++i)
        {
            const samples::ChroniclePlace &place = samples::kManiPlaces[i];
            const core::i32 expectedX = field.projection.cellX(math::Fixed32::fromFloat(place.longitude).raw());
            const core::i32 expectedZ = field.projection.cellZ(math::Fixed32::fromFloat(place.latitude).raw());

            core::i32 cellX = 0;
            core::i32 cellZ = 0;
            if (gazetteer.mapCell(i, cellX, cellZ))
            {
                ++inside;
                if (cellX != expectedX || cellZ != expectedZ)
                    check("a place sits where its coordinates put it", false);
            }
            else
            {
                ++outside;
            }

            // Every one of them is LOCATED, whether or not this grid can hold it: the two are
            // different questions and answering them with one flag is how "nobody knows where
            // this was" turns into a position somebody walks to.
            history::Place resolved;
            check("the gazetteer knows every place it lists", gazetteer.resolve(place.id, resolved));
            check("and reports it as found on the ground", resolved.located);
            (void) expectedX;
            (void) expectedZ;
        }

        std::printf("     %u places on this survey, %u located elsewhere\n", inside, outside);
        // @warning Both halves asserted. A gazetteer entirely inside its own map never exercises the
        // off-grid branch; one entirely outside would pass a test that only counted the inside.
        check("some places fall on this survey", inside >= 4u);
        check("and some are located and off it", outside >= 1u);

        // The attested network is symmetric: a road is walked either way, and a table that stated
        // one direction would leave the other half of every journey unroutable.
        core::u32 pairs = 0u;
        for (core::u32 i = 0u; i < samples::kManiPlaceCount; ++i)
        {
            core::u32 neighbours[8];
            const core::u32 count = gazetteer.linkedPlaces(samples::kManiPlaces[i].id, neighbours, 8u);
            for (core::u32 n = 0u; n < count; ++n)
            {
                core::u32 back[8];
                const core::u32 backCount = gazetteer.linkedPlaces(neighbours[n], back, 8u);
                bool mutual = false;
                for (core::u32 b = 0u; b < backCount; ++b)
                    mutual = mutual || back[b] == samples::kManiPlaces[i].id;
                check("every attested link is stated both ways", mutual);
                ++pairs;
            }
        }
        check("the corpus attests a network rather than a pair", pairs >= 8u);
    }

    // ── The world runs, and the people in it go places ───────────────────────
    std::printf("-- a corpus seeds bodies and they walk roads nobody handed them\n");
    {
        procgen::WorldRecipe recipe = procgen::parityWorldRecipe();
        platform::linux_host::LinuxPlatform host{256u, 256u};
        engine::ResourceManager resources;
        const engine::Config config = engine::Config::Builder{}.tickRate(60u).build();
        std::vector<core::u8> arenaBytes(1u << 20);
        memory::ArenaAllocator arena{arenaBytes.data(), arenaBytes.size()};

        samples::ChronicleWorld world{recipe};
        engine::WorldContext context{host, resources, arena, config, nullptr};

        check("the world initialises", world.onInit(context).has_value());
        check("and its scheduler is well formed", world.scheduler().buildGraph().has_value());

        // Long enough for a walk of a hundred years at thirty days a tick.
        constexpr core::u32 kTicks = 1400u;
        for (core::u32 tick = 0u; tick < kTicks; ++tick)
        {
            world.onFixedStep(1.0f / 60.0f);
            world.registry().swapAllBuffers();
        }

        std::printf("     %u road cells, %u pairs, %u coarse expansions, %u arrivals, %u chronicle events\n",
                    world.pavedCells(), world.routes().pairs(), world.routes().coarseExpanded(), world.arrivals(),
                    world.chronicle().size());

        check("a road network was laid", world.pavedCells() > 0u);
        check("between more than one pair", world.routes().pairs() >= 2u);
        // @warning Zero coarse expansions would mean the route fell back to a flat search, which
        // lays an identical network -- so without this the section passes against a cascade that
        // never ran.
        check("planned coarse and refined fine", world.routes().coarseExpanded() > 0u);
        check("and the fine search fitted inside the corridor", world.routes().widened() == 0u);

        // @warning The claim of the sample: bodies the corpus SEEDED, going places it only SCORED.
        // A world where nothing arrives passes every check above while proving that a map was
        // drawn.
        check("the corpus put people in the world", world.seeded() >= 2u);
        check("and they arrived somewhere on their own", world.arrivals() >= 1u);
        check("which the chronicle recorded", world.chronicle().size() >= world.arrivals());
    }

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
