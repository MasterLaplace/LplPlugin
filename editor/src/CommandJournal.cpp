/**
 * @file CommandJournal.cpp
 * @brief Implementation of the replay-based command journal.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/editor/CommandJournal.hpp>

#include <lpl/editor/CommandProcessor.hpp>
#include <lpl/editor/Json.hpp>

namespace lpl::editor {

namespace {

constexpr const char *kJournalFormat = "lplcommands/1";

/// Commands that only read. Running one changes nothing, so recording it would
/// make an undo depend on how often somebody looked at the world.
bool isInspectionCommand(std::string_view cmd)
{
    return cmd == "count" || cmd == "get_world_stats" || cmd == "query_entities" || cmd == "save_scene";
}

/// True when every command in @p root (object or array) is read-only.
bool isPurelyInspection(const detail::JVal &root)
{
    const auto commandName = [](const detail::JVal &object) -> std::string_view {
        const detail::JVal *cmd = object.find("cmd");
        return (cmd != nullptr && cmd->t == detail::JVal::T::Str) ? std::string_view{cmd->str} : std::string_view{};
    };

    if (root.t == detail::JVal::T::Arr)
    {
        for (const detail::JVal &element : root.arr)
            if (!isInspectionCommand(commandName(element)))
                return false;
        return true;
    }
    return isInspectionCommand(commandName(root));
}

} // namespace

core::Expected<std::string> CommandJournal::execute(std::string_view json)
{
    // Parse before running: a malformed command must not land in the journal,
    // or every later replay would fail on it.
    bool ok = false;
    const detail::JVal root = detail::parse(json, &ok);
    if (!ok)
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"malformed command JSON"});

    CommandProcessor processor{_registry};
    auto report = processor.execute(json);
    if (!report)
        return report;

    if (!isPurelyInspection(root))
    {
        _entries.emplace_back(json);
        // A new action invalidates the redo branch, as in every editor: the
        // future you had undone is no longer reachable from here.
        _undone.clear();
    }
    return report;
}

void CommandJournal::rebuild()
{
    (void) destroyAllEntities(_registry);

    CommandProcessor processor{_registry};
    for (const std::string &entry : _entries)
        (void) processor.execute(entry);
}

bool CommandJournal::undo()
{
    if (_entries.empty())
        return false;

    _undone.emplace_back(std::move(_entries.back()));
    _entries.pop_back();
    rebuild();
    return true;
}

bool CommandJournal::redo()
{
    if (_undone.empty())
        return false;

    _entries.emplace_back(std::move(_undone.back()));
    _undone.pop_back();

    // Only the newly restored command needs running: the world already reflects
    // every entry before it.
    CommandProcessor processor{_registry};
    (void) processor.execute(_entries.back());
    return true;
}

std::string CommandJournal::toJson() const
{
    std::string out = "{\"format\":\"";
    out += kJournalFormat;
    out += "\",\"commands\":[";
    for (core::usize i = 0u; i < _entries.size(); ++i)
    {
        if (i != 0u)
            out += ",";
        // Entries are stored as the JSON text they arrived as, so they embed
        // verbatim — no re-encoding, hence no chance of a round-trip changing
        // a number and therefore the world it rebuilds.
        out += _entries[i];
    }
    out += "]}";
    return out;
}

core::Expected<core::u32> CommandJournal::replay(std::string_view json)
{
    bool ok = false;
    const detail::JVal root = detail::parse(json, &ok);
    if (!ok || root.t != detail::JVal::T::Obj)
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"malformed journal root"});

    const detail::JVal *format = root.find("format");
    if (format == nullptr || format->t != detail::JVal::T::Str || format->str != kJournalFormat)
        return core::makeError(core::ErrorCode::kNotSupported, lpl::pmr::string{"unsupported journal format"});

    const detail::JVal *commands = root.find("commands");
    if (commands == nullptr || commands->t != detail::JVal::T::Arr)
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"missing commands array"});

    reset();

    // Re-serialise each element rather than slicing the source text: the parser
    // does not record byte spans, and a command must be replayable on its own.
    for (const detail::JVal &element : commands->arr)
        _entries.emplace_back(detail::emit(element));

    rebuild();
    return static_cast<core::u32>(_entries.size());
}

void CommandJournal::reset()
{
    _entries.clear();
    _undone.clear();
    (void) destroyAllEntities(_registry);
}

} // namespace lpl::editor
