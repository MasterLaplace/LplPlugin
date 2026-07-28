/**
 * @file CommandJournal.hpp
 * @brief The record of what was done, which is also how it is undone.
 *
 * The research report (§5.3) asks for this in one sentence: "chaque outil = une
 * commande sérialisable → le journal de commandes donne gratuitement undo/redo,
 * replay déterministe, et un slice de parité". This is that journal.
 *
 * The trick is that nothing here has to know how to reverse a command. Every
 * editor command is deterministic and starts from a known state, so undo is not
 * an inverse operation — it is "replay everything except the last entry, from an
 * empty world". That costs more than an inverse would, but it cannot drift: an
 * inverse has to be written and maintained per command, and the first one that
 * is subtly wrong corrupts a world in a way no test would notice. Replay is
 * correct by construction for any command that is added later, including ones
 * whose inverse is not expressible (a procedural pass that destroys entities it
 * did not create).
 *
 * The same property makes a journal a document: serialised, it IS the recipe
 * that rebuilds the world, and replaying it on two targets must fold the same
 * signature — the parity slice the report asks for.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_EDITOR_COMMANDJOURNAL_HPP
#    define LPL_EDITOR_COMMANDJOURNAL_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>

#    include <string>
#    include <string_view>
#    include <vector>

namespace lpl::ecs {
class Registry;
}

namespace lpl::editor {

/**
 * @class CommandJournal
 * @brief Executes editor commands against a registry, remembering the mutating
 *        ones so the world can be rewound or rebuilt.
 *
 * Non-owning: the registry belongs to the caller (an EditorSession, a test, a
 * future AI bridge). Inspection commands run but are not recorded — replaying a
 * query would change nothing, and keeping them would make an undo depend on how
 * often someone looked at the world.
 */
class CommandJournal {
public:
    /**
     * @brief Binds a journal to the world it will drive.
     * @param registry World to mutate; must outlive the journal.
     */
    explicit CommandJournal(ecs::Registry &registry) : _registry(registry) {}

    /**
     * @brief Runs @p json, recording it when it mutates the world.
     * @param json A command object, or an array of them (a batch records as one
     *             entry, so undoing a batch undoes all of it — a batch is a
     *             single intention).
     * @return The command's JSON report, or a parse error.
     */
    [[nodiscard]] core::Expected<std::string> execute(std::string_view json);

    /**
     * @brief Rewinds the last recorded command.
     * @return false when the journal is empty (nothing to undo).
     */
    bool undo();

    /**
     * @brief Re-applies the last undone command.
     * @return false when nothing has been undone since the last execute().
     */
    bool redo();

    /// @return Number of commands currently applied to the world.
    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_entries.size()); }

    /// @return Number of undone commands available to @ref redo.
    [[nodiscard]] core::u32 redoSize() const noexcept { return static_cast<core::u32>(_undone.size()); }

    /**
     * @brief Serialises the journal as a replayable document.
     * @return `{"format":"lplcommands/1","commands":[ ... ]}`.
     */
    [[nodiscard]] std::string toJson() const;

    /**
     * @brief Rebuilds the world from a serialised journal, discarding the
     *        current one.
     * @param json A document produced by @ref toJson.
     * @return Number of commands replayed, or a parse error.
     */
    [[nodiscard]] core::Expected<core::u32> replay(std::string_view json);

    /// Empties the world and forgets every recorded command.
    void reset();

private:
    /// Empties the world, then re-applies every entry in order.
    void rebuild();

    ecs::Registry &_registry;
    std::vector<std::string> _entries;
    std::vector<std::string> _undone;
};

} // namespace lpl::editor

#endif // LPL_EDITOR_COMMANDJOURNAL_HPP
