/**
 * @file ToolRegistry.cpp
 * @brief Implementation of the set of tools valid in the current world state.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/agent/ToolRegistry.hpp>

#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/CommandProcessor.hpp>

namespace lpl::agent {

WorldState WorldState::observe(const ecs::Registry &registry)
{
    WorldState state;
    // editor::entityCount is the one live count in the tree; a second walk over
    // partitions and chunks here would be a second answer to the same question.
    state.entityCount = editor::entityCount(registry);
    state.hasWorld = state.entityCount != 0u;
    return state;
}

bool ToolRegistry::satisfied(ToolGate gate, const WorldState &state) noexcept
{
    switch (gate)
    {
    case ToolGate::Always: return true;
    case ToolGate::RequiresWorld: return state.hasWorld;
    case ToolGate::RequiresEmptyWorld: return !state.hasWorld;
    }
    return false;
}

ToolRegistry ToolRegistry::forState(WorldState state)
{
    ToolRegistry registry;
    registry._state = state;
    registry._tools.reserve(kToolCount);
    for (const ToolDesc &tool : kTools)
        if (satisfied(tool.gate, state))
            registry._tools.push_back(&tool);
    return registry;
}

ToolRegistry ToolRegistry::forWorld(const ecs::Registry &registry) { return forState(WorldState::observe(registry)); }

ToolRegistry ToolRegistry::ungated()
{
    ToolRegistry registry;
    registry._tools.reserve(kToolCount);
    for (const ToolDesc &tool : kTools)
        registry._tools.push_back(&tool);
    // A state that satisfies nothing in particular: this view exists to document
    // the whole surface, and a caller that dispatched from it would earn refusals.
    registry._state = WorldState{};
    return registry;
}

const ToolDesc *ToolRegistry::find(std::string_view name) const noexcept
{
    for (const ToolDesc *tool : _tools)
        if (tool->name == name)
            return tool;
    return nullptr;
}

} // namespace lpl::agent
