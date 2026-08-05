/**
 * @file Transcript.cpp
 * @brief Implementation of the record of a reason-act-observe loop.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/agent/Transcript.hpp>

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
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
}

void appendString(std::string &out, std::string_view s)
{
    out += '"';
    appendEscaped(out, s);
    out += '"';
}

} // namespace

void Transcript::record(const ToolCall &call, std::string observation, bool ok, core::u32 journalEntry)
{
    Turn turn;
    turn.index = static_cast<core::u32>(_turns.size());
    turn.thought = call.thought;
    turn.tool = std::string{call.tool->name};
    turn.args = call.args;
    turn.observation = std::move(observation);
    turn.ok = ok;
    turn.journalEntry = journalEntry;
    _turns.push_back(std::move(turn));
}

void Transcript::recordRefusal(std::string_view attempted, std::string reason)
{
    Turn turn;
    turn.index = static_cast<core::u32>(_turns.size());
    turn.tool = std::string{attempted};
    turn.observation = std::move(reason);
    turn.ok = false;
    _turns.push_back(std::move(turn));
}

core::u32 Transcript::trailingRepeats() const noexcept
{
    if (_turns.size() < 2u)
        return 0u;
    const Turn &last = _turns.back();
    core::u32 repeats = 0u;
    for (std::size_t i = _turns.size() - 1u; i-- > 0u;)
    {
        // The observation is part of the identity: reissuing a call and getting a
        // DIFFERENT answer is progress, however slow. Only the same call answered
        // the same way means nothing is moving.
        if (_turns[i].tool != last.tool || _turns[i].args != last.args || _turns[i].observation != last.observation)
            break;
        ++repeats;
    }
    return repeats;
}

std::string Transcript::toJson() const
{
    std::string out = "{\"format\":\"lpltranscript/1\",\"turns\":[";
    bool first = true;
    for (const Turn &turn : _turns)
    {
        if (!first)
            out += ',';
        first = false;
        out += "{\"index\":" + std::to_string(turn.index);
        out += ",\"tool\":";
        appendString(out, turn.tool);
        if (!turn.thought.empty())
        {
            out += ",\"thought\":";
            appendString(out, turn.thought);
        }
        if (!turn.args.empty())
        {
            out += ",\"args\":";
            out += turn.args;
        }
        out += ",\"ok\":";
        out += turn.ok ? "true" : "false";
        // The link to the world's real history. A reader that wants to REBUILD
        // follows this into the journal; a reader that wants to UNDERSTAND stays
        // here. Two documents, one history.
        if (turn.journalEntry != kNotJournalled)
            out += ",\"journalEntry\":" + std::to_string(turn.journalEntry);
        out += ",\"observation\":";
        appendString(out, turn.observation);
        out += '}';
    }
    out += "]}";
    return out;
}

} // namespace lpl::agent
