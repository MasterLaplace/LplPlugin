/**
 * @file DemonHost.cpp
 * @brief Implementation of hosting an intelligence inside the engine loop.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/engine/DemonHost.hpp>

#include <lpl/agent/Observation.hpp>
#include <lpl/agent/ToolCall.hpp>
#include <lpl/agent/ToolRegistry.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/CommandJournal.hpp>

namespace lpl::engine {

namespace {

/// How many identical, identically-answered turns in a row means "stuck".
constexpr core::u32 kRepeatsBeforeAbort = 2u;

/// Merges the world critics with the asked-for-versus-got critics.
agent::Observations combine(agent::Observations world, const agent::Observations &generation)
{
    for (const agent::Finding &finding : generation.findings)
        world.findings.push_back(finding);
    world.total += generation.total;
    world.truncated = world.truncated || generation.truncated;
    return world;
}

} // namespace

DemonHost::DemonHost(ecs::Registry &registry, editor::CommandJournal &journal, agent::IHostedDecider &decider)
    : _registry(registry), _journal(journal), _decider(decider), _dispatcher(journal, registry)
{
}

void DemonHost::consider(const agent::Intent &intent) { _goal = intent.text; }

agent::Observations DemonHost::observe() const
{
    agent::Observations findings = agent::inspectWorld(_registry);
    if (!_lastGenerationReport.empty())
        findings = combine(std::move(findings), agent::reviewGeneration(_lastRecipe, _lastGenerationReport));
    return findings;
}

/// Turns the projection can carry. The decision only ever reads the tail of a
/// session, and a bounded view is the whole point of the shared act.
constexpr core::u32 kProjectedTurns = 16u;

/**
 * @brief Projects the tail of a rich transcript into bounded acts.
 *
 * Each turn becomes the pair the shared seam knows about — the call, then what came
 * back — so a decider reading `context.transcript` sees the same alternation whether
 * it is running here or in ring 0.
 *
 * @param transcript The record.
 * @param out        Receives the acts.
 * @param capacity   Room in @p out.
 * @return Acts written.
 */
core::u32 projectTranscript(const agent::Transcript &transcript, agent::Act *out, core::u32 capacity)
{
    const auto &turns = transcript.turns();
    core::u32 written = 0u;
    const core::usize first = turns.size() * 2u > capacity ? turns.size() - capacity / 2u : 0u;

    for (core::usize i = first; i < turns.size() && written + 1u < capacity; ++i)
    {
        const agent::Turn &turn = turns[i];

        agent::Act call;
        call.kind = agent::ActKind::Action;
        call.step = turn.index;
        const std::string text = turn.tool + turn.args;
        call.bytes = static_cast<core::u32>(text.size() < agent::kActBytes ? text.size() : agent::kActBytes);
        for (core::u32 b = 0u; b < call.bytes; ++b)
            call.text[b] = text[b];
        out[written++] = call;

        agent::Act observation;
        observation.kind = agent::ActKind::Observation;
        observation.step = turn.index;
        observation.bytes = static_cast<core::u32>(
            turn.observation.size() < agent::kActBytes ? turn.observation.size() : agent::kActBytes);
        for (core::u32 b = 0u; b < observation.bytes; ++b)
            observation.text[b] = turn.observation[b];
        out[written++] = observation;
    }
    return written;
}

ThinkResult DemonHost::think(const InferenceBudget &budget)
{
    ThinkResult result;
    result.defectsBefore = observe().defects();
    result.defectsAfter = result.defectsBefore;

    const core::u32 concludeAfter = budget.concludeAfter();

    for (core::u32 turn = 0u; turn < budget.turns(); ++turn)
    {
        // Everything the planner sees is recomputed here, every turn. That is the
        // point of regenerating the grammar each step: a plan made against a stale
        // picture of the world is a plan that earns a refusal.
        const agent::ToolRegistry tools = agent::ToolRegistry::forWorld(_registry);
        const agent::Observations findings = observe();
        result.defectsAfter = findings.defects();

        /* The rich transcript stays the record; what crosses the seam is a bounded
           PROJECTION of it. That is marshalling and not duplication — the same shape
           as SocketTransport turning a net::Endpoint into a sockaddr at the boundary
           and nowhere else — and it is what lets one decider serve a host that keeps
           journal indices and a kernel that keeps 512 bytes a line. */
        agent::Act view[kProjectedTurns]{};
        const core::u32 lines = projectTranscript(_transcript, view, kProjectedTurns);

        std::string alphabet;
        for (const agent::ToolDesc *tool : tools.tools())
        {
            if (!alphabet.empty())
                alphabet += ' ';
            alphabet += tool->name;
        }

        _decider.observe(findings, _lastRecipe);

        agent::DecisionContext context;
        context.available = alphabet.c_str();
        context.availableBytes = static_cast<core::u32>(alphabet.size());
        context.transcript = view;
        context.transcriptLines = lines;
        context.goal = _goal.c_str();
        context.goalBytes = static_cast<core::u32>(_goal.size());
        context.turn = turn;
        context.turnsRemaining = budget.turns() - turn;
        context.mustConclude = turn >= concludeAfter;
        context.satisfied = findings.defects() == 0u;

        const agent::Act decided = _decider.decide(context);
        if (decided.kind != agent::ActKind::Action || decided.bytes == 0u)
        {
            result.outcome = ThinkOutcome::Concluded;
            return result;
        }
        const std::string call(decided.text, decided.bytes);

        ++result.turns;

        // Validate before acting, and record the refusal rather than swallowing
        // it: an agent that is told WHY it was refused can do better next turn,
        // and one that is silently ignored cannot.
        auto validated = agent::parseToolCall(call, tools);
        if (!validated.has_value())
        {
            ++result.refusals;
            _transcript.recordRefusal(call, std::string{validated.error().message().c_str()});
            continue;
        }

        auto answer = _dispatcher.dispatch(validated.value());
        const bool accepted = answer.has_value();
        std::string observation = accepted ? answer.value() : std::string{answer.error().message().c_str()};

        if (accepted)
            ++result.acts;
        else
            ++result.refusals;

        // Keep the generation pair, so the asked-for-versus-got critics have
        // something to judge. Reading them off the transcript instead would work
        // until somebody trimmed it.
        if (validated.value().tool->name == "generate_world" && accepted)
        {
            _lastRecipe = validated.value().args;
            _lastGenerationReport = observation;
        }

        _transcript.record(validated.value(), std::move(observation), accepted, _journal.size());

        if (_transcript.trailingRepeats() >= kRepeatsBeforeAbort)
        {
            result.defectsAfter = observe().defects();
            result.outcome = ThinkOutcome::Stuck;
            return result;
        }
    }

    result.defectsAfter = observe().defects();
    // A budget spent with nothing left to do is not exhaustion; it is a
    // conclusion that happened to land on the last turn.
    result.outcome = result.defectsAfter == 0u ? ThinkOutcome::Concluded : ThinkOutcome::BudgetExhausted;
    return result;
}

} // namespace lpl::engine
