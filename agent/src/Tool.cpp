#include <lpl/agent/Tool.hpp>

namespace lpl::agent {

namespace {

/**
 * @brief Every tool has a name and a brief, and no two share a name.
 */
constexpr bool toolsAreNamed() noexcept
{
    for (core::u32 i = 0u; i < kToolCount; ++i)
    {
        if (kTools[i].name.empty() || kTools[i].brief.empty())
            return false;
        for (core::u32 j = i + 1u; j < kToolCount; ++j)
            if (kTools[i].name == kTools[j].name)
                return false;
    }
    return true;
}

/**
 * @brief Every parameter is named, uniquely within its tool, and its bounds are ordered.
 *
 * A closed set of strings only means something for a string, so only a string
 * parameter may declare one.
 */
constexpr bool paramsAreWellFormed() noexcept
{
    for (const ToolDesc &tool : kTools)
    {
        for (std::size_t i = 0u; i < tool.params.size(); ++i)
        {
            const ToolParam &p = tool.params[i];
            if (p.name.empty())
                return false;
            if (p.hasBounds && !(p.minValue <= p.maxValue))
                return false;
            if (p.choices != DynamicEnum::None && p.kind != ParamKind::String)
                return false;
            for (std::size_t j = i + 1u; j < tool.params.size(); ++j)
                if (p.name == tool.params[j].name)
                    return false;
        }
    }
    return true;
}

/**
 * @brief A tool that only looks at the world must not be journalled.
 *
 * Replaying a query changes nothing, and recording one would make undo depend on
 * how often somebody looked. This encodes the rule CommandJournal already follows.
 */
constexpr bool inspectionToolsDoNotMutate() noexcept
{
    for (const ToolDesc &tool : kTools)
    {
        const bool inspects = tool.name == "count" || tool.name == "save_scene" || tool.name == "get_world_stats" ||
                              tool.name == "query_entities" || tool.name == "take_screenshot" ||
                              tool.name == "diff_scenes";
        if (inspects && tool.mutates)
            return false;
    }
    return true;
}

/**
 * @brief The single-mutation-surface rule, made structural.
 *
 * A capability may be hosted by agent/ only if it does not change the world.
 * The moment a mutating tool were served outside editor::CommandJournal, there
 * would be world changes no replay could reproduce and no undo could rewind —
 * and it would be discovered much later, by a divergence rather than by a build
 * error. So it is a build error.
 */
constexpr bool mutationsGoThroughTheJournal() noexcept
{
    for (const ToolDesc &tool : kTools)
        if (tool.mutates && tool.host != ToolHost::Journal)
            return false;
    return true;
}

static_assert(toolsAreNamed(), "a tool is unnamed, undocumented, or declared twice");
static_assert(paramsAreWellFormed(), "a tool parameter is unnamed, duplicated, or has inverted bounds");
static_assert(inspectionToolsDoNotMutate(), "an inspection tool is marked as mutating the world");
static_assert(mutationsGoThroughTheJournal(), "a mutating tool bypasses the command journal");

static_assert(paramKindOf(ecs::FieldType::Fixed32) == ParamKind::Integer,
              "authoritative fields travel as raw integers");
static_assert(paramKindOf(ecs::FieldType::F32) == ParamKind::Number, "cosmetic floats travel as numbers");
static_assert(paramKindOf(ecs::FieldType::Vec3Fixed) == ParamKind::Object, "composites are objects");
static_assert(jsonTypeName(ParamKind::Integer) == "integer", "JSON-Schema spelling");

} // namespace

} // namespace lpl::agent
