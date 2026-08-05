/**
 * @file Planner.cpp
 * @brief Implementation of the deterministic correction planner.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/agent/Planner.hpp>

#include <lpl/editor/Json.hpp>

namespace lpl::agent {

namespace {

void appendEscaped(std::string &out, std::string_view s)
{
    for (const char c : s)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        default: out += c; break;
        }
    }
}

/// The arguments to send: the suggested patch, laid over the recipe in force.
std::string argumentsFor(const Finding &finding, std::string_view lastRecipe)
{
    const std::string patch = finding.suggestedArgs.empty() ? "{}" : finding.suggestedArgs;
    // Only generation carries a standing recipe worth preserving; every other
    // tool's arguments stand alone.
    if (lastRecipe.empty() || finding.suggestedTool != "generate_world")
        return patch;

    bool baseOk = false;
    bool patchOk = false;
    editor::detail::JVal base = editor::detail::parse(lastRecipe, &baseOk);
    const editor::detail::JVal delta = editor::detail::parse(patch, &patchOk);
    if (!baseOk || !patchOk)
        return patch;
    editor::detail::overlay(base, delta);
    return editor::detail::emit(base);
}

/// Builds one call object from a finding's suggestion.
std::string callFor(const Finding &finding, std::string_view lastRecipe)
{
    std::string out = "{\"thought\":\"";
    appendEscaped(out, finding.message);
    out += "\",\"tool\":\"";
    appendEscaped(out, finding.suggestedTool);
    out += "\",\"args\":";
    out += argumentsFor(finding, lastRecipe);
    out += '}';
    return out;
}

} // namespace

void CorrectionPlanner::observe(const Observations &findings, std::string_view lastRecipe)
{
    _findings = &findings;
    _lastRecipe.assign(lastRecipe);
}

Act CorrectionPlanner::decide(const DecisionContext &context) noexcept
{
    Act act;
    act.step = context.turn;
    act.kind = ActKind::Answer;

    if (context.mustConclude || _findings == nullptr)
        return act;

    for (const Finding &finding : _findings->findings)
    {
        if (finding.severity != Severity::Defect)
            continue;
        if (finding.suggestedTool.empty())
            continue;
        // A capability that is not offered right now cannot be the fix, whatever
        // the critic hoped: acting on it would earn a refusal and burn a turn.
        if (!alphabetOffers(context.available, context.availableBytes, finding.suggestedTool.c_str(),
                            static_cast<core::u32>(finding.suggestedTool.size())))
            continue;

        const std::string call = callFor(finding, _lastRecipe);
        if (call.size() > kActBytes)
            continue; // A call that cannot be stated in a shared act cannot be made.

        /* Skip a suggestion that is exactly what was just tried. Repeating it would
           leave the anti-loop guard to end the session, which is a worse outcome than
           trying the next defect on the list.

           Compared on the ACT'S IDENTITY — the tool and its arguments — and never on
           the serialised call, which also carries a thought the critic phrases anew
           each turn. Comparing whole calls would make the guard depend on JSON
           escaping and on wording, so it would silently stop matching and the loop
           would spend its budget re-issuing one move. That is the same pair the
           previous version compared, and it is what a transcript projection can carry
           without a parser. */
        const std::string identity = finding.suggestedTool + argumentsFor(finding, _lastRecipe);
        bool repeatsLast = false;
        for (core::u32 i = context.transcriptLines; i > 0u && !repeatsLast; --i)
        {
            const Act &previous = context.transcript[i - 1u];
            if (previous.kind != ActKind::Action)
                continue;
            repeatsLast = previous.bytes == identity.size();
            for (core::u32 b = 0u; b < previous.bytes && repeatsLast; ++b)
                repeatsLast = previous.text[b] == identity[b];
            break; // Only the LAST act counts, exactly as before.
        }
        if (repeatsLast)
            continue;

        act.kind = ActKind::Action;
        act.bytes = static_cast<core::u32>(call.size());
        for (core::u32 i = 0u; i < act.bytes; ++i)
            act.text[i] = call[i];
        return act;
    }

    // Nothing actionable is left. Concluding here rather than idling is what makes
    // "the loop terminates" a property instead of a hope.
    return act;
}

} // namespace lpl::agent
