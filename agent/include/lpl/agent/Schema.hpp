/**
 * @file Schema.hpp
 * @brief JSON-Schema emission from the reflection registry.
 *
 * One source of truth, three consumers: the engine validates with it, the editor
 * renders it, and Grammar compiles it into a sampler constraint.
 *
 * This is the fourth use DESIGN §3 promised of a single `constexpr` declaration —
 * layout, named (de)serialisation, JSON-Schema, and the model's grammar. The first
 * three shipped in July 2026; this file is the fourth, and the reason
 * @c ecs::FieldDesc has carried @c hasBounds / @c minRaw / @c maxRaw unused ever
 * since.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_SCHEMA_HPP
#    define LPL_LPL_AGENT_SCHEMA_HPP

#    include <lpl/agent/Tool.hpp>
#    include <lpl/agent/ToolRegistry.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/ComponentReflection.hpp>

#    include <string>

namespace lpl::agent {

/**
 * @brief The JSON-Schema of one component, derived from its field table.
 *
 * Moved here from @c tests/parity/test_reflection.cpp, which had written it as a
 * file-local static and pinned its output. The test now calls this function, so
 * the schema it verifies is the schema the grammar is built from — one emitter,
 * not two that agree today.
 *
 * Bounds follow @c FieldDesc's encoding: raw Q16.16 for @c Fixed32, the integer
 * itself for integer types, and a float bit pattern for @c F32 (see
 * @c ecs::floatBits), which is why the float case is decoded before printing.
 */
[[nodiscard]] std::string emitJsonSchema(const ecs::ComponentSchema &schema);

/**
 * @brief The JSON-Schema of one tool's argument object.
 *
 * @c required lists the parameters without a default. A @c DynamicEnum parameter
 * is expanded here rather than declared, so adding a component widens the schema
 * with no edit to the tool table.
 */
[[nodiscard]] std::string emitJsonSchema(const ToolDesc &tool);

/**
 * @brief The JSON-Schema of everything callable right now.
 *
 * A @c oneOf over the offered tools, each branch pinning @c tool to one literal
 * name. Shape chosen so that the grammar (Grammar.hpp) and this document describe
 * the same set: a call this accepts is a call the grammar can emit.
 */
[[nodiscard]] std::string emitJsonSchema(const ToolRegistry &registry);

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_SCHEMA_HPP
