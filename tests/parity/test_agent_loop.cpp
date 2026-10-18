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

static int failures = 0;

static void check(bool ok, const std::string &what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

static const char *outcomeName(lpl::engine::ThinkOutcome outcome)
{
    switch (outcome)
    {
    case lpl::engine::ThinkOutcome::Concluded: return "concluded";
    case lpl::engine::ThinkOutcome::BudgetExhausted: return "budget exhausted";
    case lpl::engine::ThinkOutcome::Stuck: return "stuck";
    case lpl::engine::ThinkOutcome::NoLegalMove: return "no legal move";
    }
    return "?";
}

/**
 * @brief A planner that always proposes the same act, to arm the anti-loop guard.
 */
class StubbornPlanner final : public lpl::agent::IHostedDecider {
public:
    /**
     * @brief Proposes the same `count` call every turn, until the loop must conclude.
     *
     * @details `count` never changes the world, so the observation is identical every turn: exactly the shape a real
     *          agent falls into when it stops making progress and keeps asking the same question.
     *
     * What a decider EMITS is the call, because that is what the host dispatches. What a transcript PROJECTION
     * carries is the act's identity — its tool and arguments — because that is the part a kernel can also keep and
     * the part an anti-repeat guard should compare. The two differ only on the host, and only by the thought a call
     * also carries.
     */
    [[nodiscard]] lpl::agent::Act decide(const lpl::agent::DecisionContext &context) noexcept override
    {
        if (context.mustConclude)
            return {};
        ++_calls;
        lpl::agent::Act act;
        act.kind = lpl::agent::ActKind::Action;
        act.step = context.turn;
        const char call[] = R"({"thought":"asking again","tool":"count","args":{}})";
        act.bytes = sizeof(call) - 1u;
        for (lpl::core::u32 i = 0u; i < act.bytes; ++i)
            act.text[i] = call[i];
        return act;
    }

    void observe(const lpl::agent::Observations &, std::string_view) override {}

    [[nodiscard]] lpl::core::u32 calls() const noexcept { return _calls; }

private:
    lpl::core::u32 _calls{0u};
};

/**
 * @brief A planner that proposes nothing at all, to check the loop leaves immediately.
 */
class IdlePlanner final : public lpl::agent::IHostedDecider {
public:
    [[nodiscard]] lpl::agent::Act decide(const lpl::agent::DecisionContext &) noexcept override { return {}; }

    void observe(const lpl::agent::Observations &, std::string_view) override {}
};

/**
 * @brief A defective world becomes a sound one, unattended: generate, look, correct.
 *
 * @details The claim is stated as a number that can move: defects go from some to none. The correction really goes
 *          through the journal, so it can be rewound. The trace explains itself: a thought, a tool, and where it
 *          landed.
 */
static void checkADefectiveWorldBecomesASoundOneUnattended()
{
    std::printf("-- generate, look, correct --\n");
    lpl::ecs::Registry world;
    lpl::editor::CommandJournal journal{world};
    lpl::agent::CorrectionPlanner planner;
    lpl::engine::DemonHost demon{world, journal, planner};

    demon.consider(lpl::agent::Intent{"make me a world", 0u});

    const lpl::engine::ThinkResult result = demon.think(lpl::engine::InferenceBudget::standard());

    std::printf("  outcome: %s in %u turn(s), %u act(s), %u refusal(s)\n", outcomeName(result.outcome), result.turns,
                result.acts, result.refusals);

    check(result.defectsBefore != 0u,
          "the empty world starts with " + std::to_string(result.defectsBefore) + " defect(s)");
    check(result.defectsAfter == 0u, "and ends with none");
    check(result.sound(), "so the world is sound");
    check(result.acts >= 1u, "at least one correcting act was taken");
    check(result.outcome == lpl::engine::ThinkOutcome::Concluded, "the loop concluded rather than ran out");
    check(lpl::editor::entityCount(world) != 0u,
          "the world now holds " + std::to_string(lpl::editor::entityCount(world)) + " entities");

    check(journal.size() >= 1u, "the journal recorded the act");
    check(journal.undo(), "and it can be undone");
    check(lpl::editor::entityCount(world) == 0u, "leaving the world as it was found");

    const lpl::agent::Transcript &transcript = demon.transcript();
    check(transcript.size() == result.turns, "the transcript has one entry per turn");
    check(transcript.last() != nullptr && !transcript.last()->thought.empty(),
          "and each act carries the reason for it");
    bool traceOk = false;
    (void) lpl::editor::detail::parse(transcript.toJson(), &traceOk);
    check(traceOk, "the transcript serialises as valid JSON");
    check(transcript.toJson().find("journalEntry") != std::string::npos,
          "and points into the journal rather than copying it");
}

/**
 * @brief A world that EXISTS and is wrong.
 *
 * @details Harder than starting from nothing, and closer to what a director actually hits: the generator ran,
 *          reported success, and produced a world missing something that was asked for. Nobody says which knob to
 *          turn.
 *
 * The seeded world is flat, unerodeed, tiny: drainage has nothing to cut, so rivers are asked for and none appear.
 */
static void checkTheCriticsFindTheDefectsOfAWorldThatExists()
{
    std::printf("\n-- a world that exists and is wrong --\n");
    lpl::ecs::Registry world;
    lpl::editor::CommandJournal journal{world};
    lpl::agent::CorrectionPlanner planner;
    lpl::engine::DemonHost demon{world, journal, planner};

    const char *seedCall =
        R"({"cmd":"generate_world","seed":7,"width":16,"depth":16,)"
        R"("terrain":{"amplitude":0.0,"octaves":1},"erosion":{"enabled":false},"rivers":{"enabled":true}})";
    const auto seeded = journal.execute(seedCall);
    check(seeded.has_value(), "a flat world generates");
    if (seeded.has_value())
        std::printf("  seed report: %s\n", seeded.value().c_str());

    const lpl::agent::Observations before = lpl::agent::reviewGeneration(seedCall, seeded.value());
    std::printf("  critics on the seeded world: %s\n", before.toJson().c_str());
    check(before.defects() != 0u, "the critics find " + std::to_string(before.defects()) + " defect(s) in it");
}

/**
 * @brief A correction keeps what the director chose.
 *
 * @details A correction is a PATCH over the recipe in force, not a replacement. Tested on the planner directly
 *          rather than through a session, because a session that happened not to need a second generation would
 *          make this pass by producing nothing — a check that cannot fail is worse than no check.
 *
 * And the merge is a merge, not an append: the patch turns erosion ON where the recipe had it off, rather than
 * leaving both values present.
 */
static void checkACorrectionKeepsWhatTheDirectorChose()
{
    std::printf("\n-- a correction keeps what the director chose --\n");
    lpl::ecs::Registry world;
    lpl::editor::CommandJournal journal{world};
    const char *recipeArgs = R"({"seed":7,"width":16,"depth":16,"terrain":{"amplitude":0.0,"octaves":1},)"
                             R"("erosion":{"enabled":false},"rivers":{"enabled":true}})";
    (void) journal.execute(std::string{R"({"cmd":"generate_world",)"} + (recipeArgs + 1));

    const lpl::agent::ToolRegistry tools = lpl::agent::ToolRegistry::forWorld(world);
    const lpl::agent::Observations findings = lpl::agent::reviewGeneration(
        recipeArgs,
        R"({"ok":true,"created":256,"riverCells":4,"caveFloor":2448,"plots":0,"reachable":true,"passed":true})");
    check(findings.defects() != 0u, "the settled-nowhere defect is seen");

    std::string alphabet;
    for (const lpl::agent::ToolDesc *tool : tools.tools())
    {
        if (!alphabet.empty())
            alphabet += ' ';
        alphabet += tool->name;
    }

    lpl::agent::CorrectionPlanner planner;
    planner.observe(findings, recipeArgs);

    lpl::agent::DecisionContext context;
    context.available = alphabet.c_str();
    context.availableBytes = static_cast<lpl::core::u32>(alphabet.size());
    context.turnsRemaining = 8u;

    const lpl::agent::Act act = planner.decide(context);
    const std::string call(act.text, act.bytes);
    std::printf("  planned: %s\n", call.c_str());

    check(!call.empty(), "the planner proposes a correction");
    check(call.find("\"seed\":7") != std::string::npos, "and keeps the seed");
    check(call.find("\"width\":16") != std::string::npos, "and the size");
    check(call.find("\"amplitude\":0") != std::string::npos, "and the terrain the director asked for");
    check(call.find("\"settlement\"") != std::string::npos, "while applying the fix");

    check(call.find("\"enabled\":false") == std::string::npos ||
              call.find("\"erosion\":{\"enabled\":true}") != std::string::npos,
          "overriding a nested field replaces it rather than duplicating it");
}

/**
 * @brief A planner with nothing to say concludes at once, having spent no turn.
 */
static void checkAPlannerWithNothingToSayConcludesAtOnce()
{
    lpl::ecs::Registry world;
    lpl::editor::CommandJournal journal{world};
    IdlePlanner idle;
    lpl::engine::DemonHost demon{world, journal, idle};
    const lpl::engine::ThinkResult result = demon.think(lpl::engine::InferenceBudget::standard());
    check(result.outcome == lpl::engine::ThinkOutcome::Concluded, "a planner with nothing to say concludes at once");
    check(result.turns == 0u, "having spent no turn");
}

/**
 * @brief A planner that repeats itself is stopped, and not after fifty turns.
 *
 * @details The guard exists so that a stuck agent costs three turns rather than the whole budget.
 */
static void checkAPlannerThatRepeatsItselfIsStopped()
{
    lpl::ecs::Registry world;
    lpl::editor::CommandJournal journal{world};
    StubbornPlanner stubborn;
    lpl::engine::DemonHost demon{world, journal, stubborn};
    const lpl::engine::ThinkResult result = demon.think(lpl::engine::InferenceBudget::ofTurns(50u));
    check(result.outcome == lpl::engine::ThinkOutcome::Stuck, "a planner that repeats itself is stopped");
    check(result.turns <= 4u, "after " + std::to_string(result.turns) + " turns, not the whole budget");
    check(stubborn.calls() <= 4u, "and the planner was not asked fifty times");
}

/**
 * @brief A budget of zero turns takes none, and leaves the defects standing.
 */
static void checkABudgetOfZeroTurnsTakesNone()
{
    lpl::ecs::Registry world;
    lpl::editor::CommandJournal journal{world};
    lpl::agent::CorrectionPlanner planner;
    lpl::engine::DemonHost demon{world, journal, planner};
    const lpl::engine::ThinkResult result = demon.think(lpl::engine::InferenceBudget::ofTurns(0u));
    check(result.turns == 0u, "a budget of zero turns takes none");
    check(result.defectsAfter != 0u, "and leaves the defects standing");
}

/**
 * @brief Termination, three ways: the loop always terminates.
 */
static void checkTheLoopAlwaysTerminates()
{
    std::printf("\n-- the loop always terminates --\n");
    checkAPlannerWithNothingToSayConcludesAtOnce();
    checkAPlannerThatRepeatsItselfIsStopped();
    checkABudgetOfZeroTurnsTakesNone();
}

/**
 * @brief The forced exit reserves the tail of the budget.
 */
static void checkTheForcedExitReservesTheTailOfTheBudget()
{
    std::printf("\n-- the forced conclusion --\n");
    check(lpl::engine::InferenceBudget::ofTurns(100u).concludeAfter() == 90u, "a budget of 100 concludes after 90");
    check(lpl::engine::InferenceBudget::ofTurns(8u).concludeAfter() == 7u, "a budget of 8 reserves its last turn");
    check(lpl::engine::InferenceBudget::ofTurns(1u).concludeAfter() == 0u, "a budget of 1 is all conclusion");
    check(lpl::engine::InferenceBudget::none().exhausted(), "and none() permits nothing");
}

/**
 * @brief The sovereign's channel is not a tool.
 */
static void checkTheSovereignsChannelIsNotATool()
{
    std::printf("\n-- the channel to the sovereign --\n");
    lpl::ecs::Registry world;
    lpl::editor::CommandJournal journal{world};
    lpl::agent::CorrectionPlanner planner;
    lpl::engine::DemonHost demon{world, journal, planner};

    lpl::agent::Dialogue &channel = demon.openDialogue();
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

/**
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
 */
int main()
{
    std::printf("== the correction loop ==\n\n");

    checkADefectiveWorldBecomesASoundOneUnattended();

    checkTheCriticsFindTheDefectsOfAWorldThatExists();

    checkACorrectionKeepsWhatTheDirectorChose();

    checkTheLoopAlwaysTerminates();

    checkTheForcedExitReservesTheTailOfTheBudget();

    checkTheSovereignsChannelIsNotATool();

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
