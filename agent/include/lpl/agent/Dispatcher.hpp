/**
 * @file Dispatcher.hpp
 * @brief The abstract execution seam.
 *
 * Where a call actually lands: the host editor's command processor, or the ring-0
 * engine directly. The protocol is identical either way, which is the whole point;
 * MCP, a local socket and a ring buffer are transports, not dialects.
 *
 * It dispatches through @c editor::CommandJournal rather than
 * @c editor::CommandProcessor, and that one choice buys undo, redo and replay for
 * nothing: every act of an intelligence becomes a journal entry, so a bad turn is
 * rewound instead of argued with. The journal already declines to record
 * inspection commands, so looking at the world does not lengthen the history.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_DISPATCHER_HPP
#    define LPL_LPL_AGENT_DISPATCHER_HPP

#    include <lpl/agent/ToolCall.hpp>
#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>

#    include <string>

namespace lpl::ecs {
class Registry;
}

namespace lpl::editor {
class CommandJournal;
}

namespace lpl::agent {

/**
 * @class Dispatcher
 * @brief Runs validated calls against a world, through its journal.
 *
 * Non-owning: the journal belongs to the caller, which also owns the registry it
 * drives.
 */
class Dispatcher {
public:
    /**
     * @brief Binds a dispatcher to the journal it will drive.
     *
     * The registry comes in alongside the journal rather than being reached
     * through it, because @c ToolHost::Agent capabilities read the world directly
     * — a screenshot is not a command, it is a look.
     *
     * @param journal  Command journal over the world to mutate; must outlive this.
     * @param registry The world that journal drives; must outlive this.
     */
    Dispatcher(editor::CommandJournal &journal, ecs::Registry &registry) : _journal(journal), _registry(registry) {}

    /**
     * @brief Executes @p call and returns the command's JSON report.
     * @return The report, or the reason the command refused it.
     */
    [[nodiscard]] core::Expected<std::string> dispatch(const ToolCall &call);

    /**
     * @brief Parses, validates and executes in one step.
     *
     * The refusal path a caller normally wants: a malformed or forbidden call
     * never reaches the world, and the reason comes back in the same shape as a
     * command error, so the loop above has one thing to read.
     */
    [[nodiscard]] core::Expected<std::string> dispatchJson(std::string_view json, const ToolRegistry &registry);

    /// @return How many calls have mutated the world since this dispatcher was bound.
    [[nodiscard]] core::u32 mutations() const noexcept { return _mutations; }

    /// Rewinds the last mutating call. @return false when there is nothing to undo.
    bool undo();

private:
    /// The @c ToolHost::Agent half: capabilities editor/ cannot host.
    [[nodiscard]] core::Expected<std::string> dispatchHere(const ToolCall &call);

    editor::CommandJournal &_journal;
    ecs::Registry &_registry;
    core::u32 _mutations{0u};
};

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_DISPATCHER_HPP
