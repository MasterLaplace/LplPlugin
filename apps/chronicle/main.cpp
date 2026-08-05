/**
 * @file main.cpp
 * @brief Run a constrained world across centuries, headless.
 *
 * Where the demon's score comes from. procgen builds a plausible world; the
 * timeline constrains it to be ours; the chronicle records what actually
 * happened; divergence measures the gap. A prediction is worth exactly what the
 * reconstruction of a known past is worth, so this number exists before any
 * claim about the future is made.
 *
 * The body below is the API written from the caller's side, before the callee
 * exists — the cheapest way to find out whether a library is pleasant to use. It
 * is fenced out until the modules it names are implemented, and the entry point
 * fails loudly rather than returning success it has not earned.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>

#include <lpl/engine/Engine.hpp>
#include <lpl/history/Chronicle.hpp>
#include <lpl/history/Divergence.hpp>
#include <lpl/history/Era.hpp>
#include <lpl/history/PossibleWorld.hpp>
#include <lpl/history/Timeline.hpp>
#include <lpl/pack/Cartridge.hpp>

int main(int argc, char **argv)
{
    (void) argc;
    (void) argv;

#if 0 // ── intended usage ─────────────────────────────────────────────────────
    // One cartridge answers three questions: what the world IS (the recipe), what
    // it LOOKS like (the view profile), and what it REMEMBERS (the timeline).
    const lpl::pack::Cartridge cartridge = lpl::pack::Cartridge::open(argv[1]);
    LPL_VERIFY(cartridge.valid());

    const lpl::history::Timeline timeline = lpl::history::Timeline::from(cartridge.history());

    // A century cannot be simulated at 60 Hz. An era is a gearing, not a different
    // engine: the same systems, stepped at a scale where populations move.
    const lpl::history::Era era = lpl::history::Era::from(1000).to(1400).ticksPerYear(4);

    // One hypothesis is one World. Running "according to this source" is a mode,
    // not a fork of the code.
    lpl::history::PossibleWorld world{cartridge.recipe(), timeline};

    lpl::engine::Engine engine{lpl::engine::HostProfile::dedicated().config(), world.build()};
    engine.init();

    lpl::history::Chronicle chronicle;
    for (const auto year : era)
        engine.stepYear(year, chronicle);

    const lpl::history::Divergence verdict = lpl::history::Divergence::measure(chronicle, timeline);
    lpl::core::log::info("chronicle: {} events, divergence {}", chronicle.size(), verdict.score());

    engine.shutdown();
    return verdict.acceptable() ? 0 : 1;
#endif // ─────────────────────────────────────────────────────────────────────

    LPL_NOT_IMPLEMENTED("lpl-chronicle");
}
