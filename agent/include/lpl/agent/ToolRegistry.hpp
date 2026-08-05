/**
 * @file ToolRegistry.hpp
 * @brief The set of tools valid in the current world state.
 *
 * Not a fixed list: the registry answers what is callable NOW, so that spawning
 * cannot be offered before a world exists. That state-dependence is what makes a
 * constrained grammar able to forbid a bad call outright.
 *
 * The precondition is EVALUATED from the registry, never carried as a flag a
 * caller has to remember to update. A flag would be a second answer to "is there
 * a world", and this repository has already paid for two answers to one question
 * (§33: the eroded field and the raw noise both claimed to be the ground height,
 * and props floated above the ridges).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_TOOLREGISTRY_HPP
#    define LPL_LPL_AGENT_TOOLREGISTRY_HPP

#    include <lpl/agent/Tool.hpp>
#    include <lpl/core/Types.hpp>

#    include <vector>

namespace lpl::ecs {
class Registry;
}

namespace lpl::agent {

/**
 * @struct WorldState
 * @brief The facts a tool's precondition is allowed to depend on.
 *
 * Small on purpose. Every field here widens the grammar's dependency surface, so
 * a fact earns its place by gating a tool, not by being interesting.
 */
struct WorldState {
    core::u32 entityCount{0u};
    bool hasWorld{false}; ///< The world holds at least one entity.

    /** @brief Reads the state out of a live registry. */
    [[nodiscard]] static WorldState observe(const ecs::Registry &registry);
};

/**
 * @class ToolRegistry
 * @brief The gated view of @ref kTools for one world state.
 */
class ToolRegistry {
public:
    /**
     * @brief The capabilities available for a world in @p state.
     * @param state World facts the preconditions read.
     */
    [[nodiscard]] static ToolRegistry forState(WorldState state);

    /**
     * @brief The capabilities available for the world @p registry holds.
     *
     * "World" rather than "registry" because that is what it is on this side of
     * the fence: editor::CommandProcessor's entire world IS an ecs::Registry, and
     * `apps/demon/main.cpp` was written against this spelling before any of it
     * existed. It deliberately does NOT take an engine::World — agent/ must not
     * depend on engine/, or the engine could not host a demon without a cycle.
     */
    [[nodiscard]] static ToolRegistry forWorld(const ecs::Registry &registry);

    /**
     * @brief Every capability, gating ignored. For documentation, not for a model.
     * @return A registry that offers every tool, regardless of world state.
     */
    [[nodiscard]] static ToolRegistry ungated();

    /// The offered capabilities, in @ref kTools order.
    [[nodiscard]] const std::vector<const ToolDesc *> &tools() const noexcept { return _tools; }

    /**
     * @brief Finds a tool by name.
     * @param name The name of the tool to find.
     * @return A pointer to the tool if found, or nullptr otherwise.
     */
    [[nodiscard]] const ToolDesc *find(std::string_view name) const noexcept;

    /**
     * @brief Checks if a tool is offered.
     * @param name The name of the tool to check.
     * @return True if the tool is offered, false otherwise.
     */
    [[nodiscard]] bool offers(std::string_view name) const noexcept { return find(name) != nullptr; }

    [[nodiscard]] const WorldState &state() const noexcept { return _state; }
    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_tools.size()); }

    /**
     * @brief Checks if a tool gate is satisfied by a world state.
     * @param gate The tool gate to check.
     * @param state The world state to check against.
     * @return True if the gate is satisfied, false otherwise.
     */
    [[nodiscard]] static bool satisfied(ToolGate gate, const WorldState &state) noexcept;

private:
    WorldState _state{};
    std::vector<const ToolDesc *> _tools;
};

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_TOOLREGISTRY_HPP
