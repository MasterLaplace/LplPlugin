/**
 * @file Grammar.hpp
 * @brief GBNF emission for constrained decoding.
 *
 * The grammar is regenerated at each step and exposes only the currently valid
 * actions and enum values. Hallucinating an API becomes physically impossible
 * rather than merely unlikely — the argument that makes a small local model usable.
 *
 * What the grammar CAN enforce: the set of tool names, the set of parameter names
 * per tool, the closed sets of enum strings, and JSON well-formedness. What it
 * cannot: numeric ranges, which a context-free grammar has no way to say. Bounds
 * therefore live in the JSON-Schema (for the model to read) and in
 * @c parseToolCall (for the engine to enforce). Two places, one declaration —
 * @ref ToolParam — so they cannot disagree.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_GRAMMAR_HPP
#    define LPL_LPL_AGENT_GRAMMAR_HPP

#    include <lpl/agent/ToolRegistry.hpp>
#    include <lpl/core/Types.hpp>

#    include <string>

namespace lpl::agent {

/**
 * @brief A GBNF grammar accepting exactly the calls @p registry offers.
 *
 * The emitted root accepts one object: an optional @c thought, a @c tool name
 * drawn from the offered set, and an @c args object whose members are drawn from
 * that tool's parameters.
 *
 * Two argument shapes are emitted, for a reason worth stating: when a tool has
 * required parameters the grammar pins them in declared order, so the model
 * cannot omit one; when it has none, the grammar accepts any ordered selection of
 * known members. A context-free grammar cannot say "these three in any order,
 * all present", and forcing an order on fifteen optional parameters would make
 * the common case unreachable.
 */
[[nodiscard]] std::string emitGbnf(const ToolRegistry &registry);

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_GRAMMAR_HPP
