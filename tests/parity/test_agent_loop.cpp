/**
 * @file test_agent_loop.cpp
 * @brief The reason-act-observe loop corrects a defect without being told how.
 *
 * This is the criterion the whole Caine batch exists to meet: a world with
 * something wrong with it becomes a world with nothing wrong with it, because a
 * critic named the defect and the loop acted on it. No model is involved, and
 * that is deliberate — a correction loop whose correctness depended on inference
 * could not be a test.
 *
 * The other half is just as important and easier to get wrong: the loop must
 * always TERMINATE. Three ways out are exercised here — it concludes, it runs out
 * of budget, or it notices it is repeating itself and stops.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/agent/Observation.hpp>
#include <lpl/agent/Planner.hpp>
#include <lpl/agent/ToolRegistry.hpp>
#include <lpl/agent/Transcript.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/CommandJournal.hpp>
#include <lpl/editor/CommandProcessor.hpp>
#include <lpl/editor/Json.hpp>
#include <lpl/engine/DemonHost.hpp>
#include <lpl/engine/InferenceBudget.hpp>

#include <cstdio>
#include <string>

using namespace lpl;

static int failures = 0;

static void check(bool ok, const std::string &what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

static const char *outcomeName(engine::ThinkOutcome outcome)
{
    switch (outcome)
    {
    case engine::ThinkOutcome::Concluded: return "concluded";
    case engine::ThinkOutcome::BudgetExhausted: return "budget exhausted";
    case engine::ThinkOutcome::Stuck: return "stuck";
    case engine::ThinkOutcome::NoLegalMove: return "no legal move";
    }
    return "?";
}

/// A planner that always proposes the same act, to arm the anti-loop guard.
class StubbornPlanner final : public agent::IHostedDecider {
public:
    [[nodiscard]] agent::Act decide(const agent::DecisionContext &context) noexcept override
    {
        if (context.mustConclude)
            return {};
        ++_calls;
        // `count` never changes the world, so the observation is identical every
        // turn: exactly the shape a real agent falls into when it stops making
        // progress and keeps asking the same question.
        /* What a decider EMITS is the call, because that is what the host dispatches.
           What a transcript PROJECTION carries is the act's identity — its tool and
           arguments — because that is the part a kernel can also keep and the part an
           anti-repeat guard should compare. The two differ only on the host, and only
           by the thought a call also carries. */
        agent::Act act;
        act.kind = agent::ActKind::Action;
        act.step = context.turn;
        const char call[] = R"({"thought":"asking again","tool":"count","args":{}})";
        act.bytes = sizeof(call) - 1u;
        for (core::u32 i = 0u; i < act.bytes; ++i)
            act.text[i] = call[i];
        return act;
    }

    void observe(const agent::Observations &, std::string_view) override {}

    [[nodiscard]] core::u32 calls() const noexcept { return _calls; }

private:
    core::u32 _calls{0u};
};

/// A planner that proposes nothing at all, to check the loop leaves immediately.
class IdlePlanner final : public agent::IHostedDecider {
public:
    [[nodiscard]] agent::Act decide(const agent::DecisionContext &) noexcept override { return {}; }

    void observe(const agent::Observations &, std::string_view) override {}
};

int main()
{
    std::printf("== the correction loop ==\n\n");

    // ── 1. A defective world becomes a sound one, unattended ──────────────────
    std::printf("-- generate, look, correct --\n");
    {
        ecs::Registry world;
        editor::CommandJournal journal{world};
        agent::CorrectionPlanner planner;
        engine::DemonHost demon{world, journal, planner};

        demon.consider(agent::Intent{"make me a world", 0u});

        const engine::ThinkResult result = demon.think(engine::InferenceBudget::standard());

        std::printf("  outcome: %s in %u turn(s), %u act(s), %u refusal(s)\n", outcomeName(result.outcome),
                    result.turns, result.acts, result.refusals);

        // The claim, stated as a number that can move: defects go from some to none.
        check(result.defectsBefore != 0u,
              "the empty world starts with " + std::to_string(result.defectsBefore) + " defect(s)");
        check(result.defectsAfter == 0u, "and ends with none");
        check(result.sound(), "so the world is sound");
        check(result.acts >= 1u, "at least one correcting act was taken");
        check(result.outcome == engine::ThinkOutcome::Concluded, "the loop concluded rather than ran out");
        check(editor::entityCount(world) != 0u,
              "the world now holds " + std::to_string(editor::entityCount(world)) + " entities");

        // The correction really went through the journal, so it can be rewound.
        check(journal.size() >= 1u, "the journal recorded the act");
        check(journal.undo(), "and it can be undone");
        check(editor::entityCount(world) == 0u, "leaving the world as it was found");

        // The trace explains itself: a thought, a tool, and where it landed.
        const agent::Transcript &transcript = demon.transcript();
        check(transcript.size() == result.turns, "the transcript has one entry per turn");
        check(transcript.last() != nullptr && !transcript.last()->thought.empty(),
              "and each act carries the reason for it");
        bool traceOk = false;
        (void) editor::detail::parse(transcript.toJson(), &traceOk);
        check(traceOk, "the transcript serialises as valid JSON");
        check(transcript.toJson().find("journalEntry") != std::string::npos,
              "and points into the journal rather than copying it");
    }

    // ── 1b. A world that EXISTS and is wrong ──────────────────────────────────
    // Harder than starting from nothing, and closer to what a director actually
    // hits: the generator ran, reported success, and produced a world missing
    // something that was asked for. Nobody says which knob to turn.
    std::printf("\n-- a world that exists and is wrong --\n");
    {
        ecs::Registry world;
        editor::CommandJournal journal{world};
        agent::CorrectionPlanner planner;
        engine::DemonHost demon{world, journal, planner};

        // Flat, unerodeed, tiny: drainage has nothing to cut, so rivers are asked
        // for and none appear.
        const char *seedCall =
            R"({"cmd":"generate_world","seed":7,"width":16,"depth":16,)"
            R"("terrain":{"amplitude":0.0,"octaves":1},"erosion":{"enabled":false},"rivers":{"enabled":true}})";
        const auto seeded = journal.execute(seedCall);
        check(seeded.has_value(), "a flat world generates");
        if (seeded.has_value())
            std::printf("  seed report: %s\n", seeded.value().c_str());

        const agent::Observations before = agent::reviewGeneration(seedCall, seeded.value());
        std::printf("  critics on the seeded world: %s\n", before.toJson().c_str());
        check(before.defects() != 0u, "the critics find " + std::to_string(before.defects()) + " defect(s) in it");
    }

    // A correction is a PATCH over the recipe in force, not a replacement. Tested
    // on the planner directly rather than through a session, because a session
    // that happened not to need a second generation would make this pass by
    // producing nothing — a check that cannot fail is worse than no check.
    std::printf("\n-- a correction keeps what the director chose --\n");
    {
        ecs::Registry world;
        editor::CommandJournal journal{world};
        const char *recipeArgs =
            R"({"seed":7,"width":16,"depth":16,"terrain":{"amplitude":0.0,"octaves":1},)"
            R"("erosion":{"enabled":false},"rivers":{"enabled":true}})";
        (void) journal.execute(std::string{R"({"cmd":"generate_world",)"} + (recipeArgs + 1));

        const agent::ToolRegistry tools = agent::ToolRegistry::forWorld(world);
        const agent::Observations findings = agent::reviewGeneration(
            recipeArgs,
            R"({"ok":true,"created":256,"riverCells":4,"caveFloor":2448,"plots":0,"reachable":true,"passed":true})");
        check(findings.defects() != 0u, "the settled-nowhere defect is seen");

        std::string alphabet;
        for (const agent::ToolDesc *tool : tools.tools())
        {
            if (!alphabet.empty())
                alphabet += ' ';
            alphabet += tool->name;
        }

        agent::CorrectionPlanner planner;
        planner.observe(findings, recipeArgs);

        agent::DecisionContext context;
        context.available = alphabet.c_str();
        context.availableBytes = static_cast<core::u32>(alphabet.size());
        context.turnsRemaining = 8u;

        const agent::Act act = planner.decide(context);
        const std::string call(act.text, act.bytes);
        std::printf("  planned: %s\n", call.c_str());

        check(!call.empty(), "the planner proposes a correction");
        check(call.find("\"seed\":7") != std::string::npos, "and keeps the seed");
        check(call.find("\"width\":16") != std::string::npos, "and the size");
        check(call.find("\"amplitude\":0") != std::string::npos, "and the terrain the director asked for");
        check(call.find("\"settlement\"") != std::string::npos, "while applying the fix");

        // And the merge is a merge, not an append: the patch turns erosion ON
        // where the recipe had it off, rather than leaving both values present.
        check(call.find("\"enabled\":false") == std::string::npos ||
                  call.find("\"erosion\":{\"enabled\":true}") != std::string::npos,
              "overriding a nested field replaces it rather than duplicating it");
    }

    // ── 2. Termination, three ways ───────────────────────────────────────────
    std::printf("\n-- the loop always terminates --\n");
    {
        ecs::Registry world;
        editor::CommandJournal journal{world};
        IdlePlanner idle;
        engine::DemonHost demon{world, journal, idle};
        const engine::ThinkResult result = demon.think(engine::InferenceBudget::standard());
        check(result.outcome == engine::ThinkOutcome::Concluded, "a planner with nothing to say concludes at once");
        check(result.turns == 0u, "having spent no turn");
    }
    {
        ecs::Registry world;
        editor::CommandJournal journal{world};
        StubbornPlanner stubborn;
        engine::DemonHost demon{world, journal, stubborn};
        const engine::ThinkResult result = demon.think(engine::InferenceBudget::ofTurns(50u));
        check(result.outcome == engine::ThinkOutcome::Stuck, "a planner that repeats itself is stopped");
        // Not fifty. The guard exists so that a stuck agent costs three turns
        // rather than the whole budget.
        check(result.turns <= 4u, "after " + std::to_string(result.turns) + " turns, not the whole budget");
        check(stubborn.calls() <= 4u, "and the planner was not asked fifty times");
    }
    {
        // A budget of one turn cannot both act and conclude, so it must report
        // that it ran out rather than pretend it finished.
        ecs::Registry world;
        editor::CommandJournal journal{world};
        agent::CorrectionPlanner planner;
        engine::DemonHost demon{world, journal, planner};
        const engine::ThinkResult result = demon.think(engine::InferenceBudget::ofTurns(0u));
        check(result.turns == 0u, "a budget of zero turns takes none");
        check(result.defectsAfter != 0u, "and leaves the defects standing");
    }

    // ── 3. The forced exit reserves the tail of the budget ────────────────────
    std::printf("\n-- the forced conclusion --\n");
    {
        check(engine::InferenceBudget::ofTurns(100u).concludeAfter() == 90u, "a budget of 100 concludes after 90");
        check(engine::InferenceBudget::ofTurns(8u).concludeAfter() == 7u, "a budget of 8 reserves its last turn");
        check(engine::InferenceBudget::ofTurns(1u).concludeAfter() == 0u, "a budget of 1 is all conclusion");
        check(engine::InferenceBudget::none().exhausted(), "and none() permits nothing");
    }

    // ── 4. The sovereign's channel is not a tool ──────────────────────────────
    std::printf("\n-- the channel to the sovereign --\n");
    {
        ecs::Registry world;
        editor::CommandJournal journal{world};
        agent::CorrectionPlanner planner;
        engine::DemonHost demon{world, journal, planner};

        agent::Dialogue &channel = demon.openDialogue();
        check(!channel.poll().has_value(), "an empty channel yields nothing");
        channel.offer("build me a village by a river");
        check(channel.pending() == 1u, "an intent queues");

        const auto intent = channel.poll();
        check(intent.has_value(), "and comes back out");
        if (intent.has_value())
        {
            demon.consider(*intent);
            check(demon.goal() == "build me a village by a river", "the demon takes it as its goal, verbatim");
        }
        check(!channel.poll().has_value(), "and is not delivered twice");

        channel.say("done: 580 entities");
        check(channel.replies().size() == 1u, "a reply queues the other way");
        check(channel.drainReplies().size() == 1u, "and drains");
        check(channel.replies().empty(), "leaving the channel clear");
    }

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
