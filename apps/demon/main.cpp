/**
 * @file main.cpp
 * @brief `lpl-demon` — an intelligence directing a deterministic world.
 *
 * The inversion this project is built on, in one entry point: the demon decides
 * WHAT, the engine decides HOW, and the engine is the only thing that touches the
 * world. Every act goes through the command journal, so a session is undoable and
 * replayable; every act is validated against a grammar derived from the component
 * registry, so an act the engine cannot perform cannot be uttered.
 *
 * Usage:
 * @code
 *   lpl-demon [--intent "<sentence>"] [--turns N] [--out world.lplscene]
 *             [--shot frame.ppm] [--trace transcript.json] [--grammar out.gbnf]
 * @endcode
 *
 * ── What changed from the sketch that used to live here ──────────────────────
 *
 * This file was written caller-first, before any of the callee existed, which is
 * exactly what it was for. Two things it assumed turned out not to be true, and
 * the sketch is what gave way:
 *
 *   - `HostProfile::dedicated().withInference(...)` — HostProfile is an ENUM
 *     (Ring0Client / Ring0Server / DedicatedServer) applied to a Config::Builder,
 *     not a factory. And a budget expressed as a fraction of a frame is a wall
 *     clock in a replayable path, so InferenceBudget counts TURNS instead.
 *   - `while (engine.running()) { engine.tick(); demon.think(...); }` — Engine
 *     owns its loop (`run()`), and engine::Boot owns the whole boot. Opening the
 *     loop publicly to drive it from here would have duplicated `run()`. The
 *     demon attaches to a world, not to a hand-rolled loop.
 *
 * The demon here directs a world headlessly, which is the shape that is testable
 * today (`test-agent-loop`). Hosting one INSIDE a running Engine is the same
 * DemonHost with its `think()` called from the frame, and is what Config's
 * built-in-system flags will select.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/agent/Grammar.hpp>
#include <lpl/agent/Observation.hpp>
#include <lpl/agent/Planner.hpp>
#include <lpl/agent/ToolRegistry.hpp>
#include <lpl/agent/Vision.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/CommandJournal.hpp>
#include <lpl/editor/CommandProcessor.hpp>
#include <lpl/editor/SceneSerializer.hpp>
#include <lpl/engine/DemonHost.hpp>
#include <lpl/engine/InferenceBudget.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct Options {
    std::string intent{"make a world worth walking through"};
    std::string scenePath;
    std::string shotPath;
    std::string tracePath;
    std::string grammarPath;
    lpl::core::u32 turns{8u};
};

void usage()
{
    std::puts("lpl-demon — an intelligence directing a deterministic world");
    std::puts("  --intent \"<sentence>\"   what to ask for");
    std::puts("  --turns N               how many reason-act-observe steps (default 8)");
    std::puts("  --out    <file>         write the resulting .lplscene");
    std::puts("  --shot   <file>         write a PPM of the world");
    std::puts("  --trace  <file>         write the session transcript as JSON");
    std::puts("  --grammar <file>        write the GBNF a model would be constrained to");
}

/// Reads @p argv into @p out; returns false when the arguments make no sense.
bool parseArguments(int argc, char **argv, Options &out)
{
    const auto value = [&](int &i) -> const char * { return (i + 1 < argc) ? argv[++i] : nullptr; };
    for (int i = 1; i < argc; ++i)
    {
        const char *flag = argv[i];
        if (std::strcmp(flag, "--help") == 0 || std::strcmp(flag, "-h") == 0)
            return false;
        const char *given = nullptr;
        if (std::strcmp(flag, "--intent") == 0 && (given = value(i)) != nullptr)
            out.intent = given;
        else if (std::strcmp(flag, "--out") == 0 && (given = value(i)) != nullptr)
            out.scenePath = given;
        else if (std::strcmp(flag, "--shot") == 0 && (given = value(i)) != nullptr)
            out.shotPath = given;
        else if (std::strcmp(flag, "--trace") == 0 && (given = value(i)) != nullptr)
            out.tracePath = given;
        else if (std::strcmp(flag, "--grammar") == 0 && (given = value(i)) != nullptr)
            out.grammarPath = given;
        else if (std::strcmp(flag, "--turns") == 0 && (given = value(i)) != nullptr)
            out.turns = static_cast<lpl::core::u32>(std::atoi(given));
        else
            return false;
    }
    return true;
}

bool writeFile(const std::string &path, const std::string &contents)
{
    std::FILE *handle = std::fopen(path.c_str(), "wb");
    if (handle == nullptr)
        return false;
    const std::size_t written = std::fwrite(contents.data(), 1u, contents.size(), handle);
    std::fclose(handle);
    return written == contents.size();
}

const char *outcomeName(lpl::engine::ThinkOutcome outcome)
{
    switch (outcome)
    {
    case lpl::engine::ThinkOutcome::Concluded: return "concluded";
    case lpl::engine::ThinkOutcome::BudgetExhausted: return "budget exhausted";
    case lpl::engine::ThinkOutcome::Stuck: return "stuck, repeating itself";
    case lpl::engine::ThinkOutcome::NoLegalMove: return "no legal move";
    }
    return "?";
}

} // namespace

int main(int argc, char **argv)
{
    Options options;
    if (!parseArguments(argc, argv, options))
    {
        usage();
        return 1;
    }

    lpl::ecs::Registry world;
    lpl::editor::CommandJournal journal{world};
    lpl::agent::CorrectionPlanner planner;
    lpl::engine::DemonHost demon{world, journal, planner};

    // Dialogue is deliberately not a tool. Tools are how the demon acts on its
    // world; this is how it addresses the sovereign, who is outside it.
    lpl::agent::Dialogue &channel = demon.openDialogue();
    channel.offer(options.intent);
    if (const auto intent = channel.poll())
        demon.consider(*intent);

    std::printf("lpl-demon — planner \"%s\", budget %u turns\n", planner.name(), options.turns);
    std::printf("  intent: %s\n\n", options.intent.c_str());

    const lpl::engine::ThinkResult result = demon.think(lpl::engine::InferenceBudget::ofTurns(options.turns));

    for (const lpl::agent::Turn &turn : demon.transcript().turns())
    {
        std::printf("  [%u] %-20s %s\n", turn.index, turn.tool.c_str(), turn.ok ? "ok" : "REFUSED");
        if (!turn.thought.empty())
            std::printf("      why: %s\n", turn.thought.c_str());
        if (!turn.ok)
            std::printf("      %s\n", turn.observation.c_str());
    }

    std::printf("\n  %s after %u turn(s): %u act(s), %u refusal(s)\n", outcomeName(result.outcome), result.turns,
                result.acts, result.refusals);
    std::printf("  defects %u -> %u, %u entities\n", result.defectsBefore, result.defectsAfter,
                lpl::editor::entityCount(world));

    // Whatever the demon could not fix, say so plainly rather than reporting a
    // success the world does not have.
    const lpl::agent::Observations remaining = demon.observe();
    for (const lpl::agent::Finding &finding : remaining.findings)
        std::printf("  still wrong: [%s] %s\n", finding.code.c_str(), finding.message.c_str());

    if (!options.scenePath.empty())
        std::printf("  scene   -> %s%s\n", options.scenePath.c_str(),
                    writeFile(options.scenePath, lpl::editor::toLplScene(world)) ? "" : "  (FAILED)");

    if (!options.tracePath.empty())
        std::printf("  trace   -> %s%s\n", options.tracePath.c_str(),
                    writeFile(options.tracePath, demon.transcript().toJson()) ? "" : "  (FAILED)");

    if (!options.grammarPath.empty())
    {
        const lpl::agent::ToolRegistry tools = lpl::agent::ToolRegistry::forWorld(world);
        std::printf("  grammar -> %s (%u tools offered)%s\n", options.grammarPath.c_str(), tools.size(),
                    writeFile(options.grammarPath, lpl::agent::emitGbnf(tools)) ? "" : "  (FAILED)");
    }

    if (!options.shotPath.empty())
    {
        const auto shot = lpl::agent::captureToFile(world, options.shotPath, 480u, 300u, lpl::agent::CameraPose{});
        if (shot.has_value())
            std::printf("  shot    -> %s (%u entities, frame signature 0x%08X)\n", options.shotPath.c_str(),
                        shot.value().entitiesDrawn, shot.value().fold);
        else
            std::printf("  shot    -> %s  (FAILED)\n", options.shotPath.c_str());
    }

    // A world the critics still object to is not a success, and saying so is the
    // difference between a tool that reports outcomes and one that reports effort.
    return result.sound() ? 0 : 2;
}
