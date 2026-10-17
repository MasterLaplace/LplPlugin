#ifndef LPL_AGENT_TOOL_INL
#define LPL_AGENT_TOOL_INL

namespace lpl::agent {

constexpr ParamKind paramKindOf(ecs::FieldType t) noexcept
{
    switch (t)
    {
    case ecs::FieldType::F32: return ParamKind::Number;
    case ecs::FieldType::Fixed32:
    case ecs::FieldType::I32:
    case ecs::FieldType::U32:
    case ecs::FieldType::U16:
    case ecs::FieldType::U8: return ParamKind::Integer;
    default: return ParamKind::Object;
    }
}

constexpr std::string_view jsonTypeName(ParamKind k) noexcept
{
    switch (k)
    {
    case ParamKind::Number: return "number";
    case ParamKind::Integer: return "integer";
    case ParamKind::String: return "string";
    case ParamKind::Bool: return "boolean";
    case ParamKind::Array: return "array";
    case ParamKind::Object: return "object";
    }
    return "object";
}

template <typename Sink> void forEachChoice(DynamicEnum choices, Sink &&sink)
{
    switch (choices)
    {
    case DynamicEnum::ComponentName:
        for (const ecs::ComponentSchema &schema : ecs::allSchemas())
            sink(schema.name);
        return;
    case DynamicEnum::CaveKind:
        for (core::u32 i = 0u; i <= static_cast<core::u32>(procgen::CaveKind::Auto); ++i)
            sink(std::string_view{procgen::caveKindName(static_cast<procgen::CaveKind>(i))});
        return;
    case DynamicEnum::None: return;
    }
}

inline bool inChoices(DynamicEnum choices, std::string_view text)
{
    if (choices == DynamicEnum::None)
        return true;
    bool found = false;
    forEachChoice(choices, [&](std::string_view word) { found = found || word == text; });
    return found;
}

constexpr const ToolDesc *findTool(std::string_view name) noexcept
{
    for (const ToolDesc &tool : kTools)
        if (tool.name == name)
            return &tool;
    return nullptr;
}

} // namespace lpl::agent

#endif // LPL_AGENT_TOOL_INL
