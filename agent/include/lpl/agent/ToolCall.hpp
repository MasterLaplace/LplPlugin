/**
 * @file ToolCall.hpp
 * @brief A parsed, validated invocation.
 *
 * Validation happens before execution and returns a reason, because an agent that
 * learns from a rejection is worth more than one that is silently ignored.
 *
 * The wire shape is one object per step:
 * @code
 * {"thought": "the world is empty, so generate one",
 *  "tool": "generate_world",
 *  "args": {"seed": 42, "width": 64, "depth": 64}}
 * @endcode
 * The thought travels with the act on purpose: a transcript entry that carried
 * only the act would let the reasoning and the action drift apart when replayed.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_TOOLCALL_HPP
#    define LPL_LPL_AGENT_TOOLCALL_HPP

#    include <lpl/agent/ToolRegistry.hpp>
#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>

#    include <string>

namespace lpl::agent {

/**
 * @struct ToolCall
 * @brief One validated invocation, ready to dispatch.
 */
struct ToolCall {
    const ToolDesc *tool{nullptr}; ///< Never null once validated.
    std::string thought;           ///< Optional; empty when the model gave none.
    std::string args;              ///< The argument object, re-emitted as JSON text.

    /**
     * @brief The call as an @c editor::CommandProcessor command object.
     *
     * The translation is a rename, not a transformation: @c args already uses the
     * command's own parameter names, because the tool table was declared from the
     * command surface rather than beside it.
     */
    [[nodiscard]] std::string toCommandJson() const;
};

/**
 * @brief Parses and validates one call against what @p registry offers.
 *
 * Refuses, with a reason, when: the JSON is malformed; @c tool is missing or not
 * a string; the named tool is unknown OR not currently offered; @c args is
 * missing or not an object; a required parameter is absent; a parameter is not
 * declared; a value has the wrong JSON type; a number falls outside its declared
 * bounds; or a string is outside its closed set.
 *
 * Bounds are checked here rather than in the grammar because a context-free
 * grammar cannot express a numeric range — see Grammar.hpp.
 *
 * @param json     One call object.
 * @param registry The capabilities offered at this instant.
 * @return The validated call, or the reason it was refused.
 */
[[nodiscard]] core::Expected<ToolCall> parseToolCall(std::string_view json, const ToolRegistry &registry);

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_TOOLCALL_HPP
