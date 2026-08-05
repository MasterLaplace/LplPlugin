/**
 * @file ToolCall.cpp
 * @brief Implementation of a parsed, validated invocation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/agent/ToolCall.hpp>

#include <lpl/core/Error.hpp>
#include <lpl/ecs/ComponentReflection.hpp>
#include <lpl/editor/Json.hpp>

#include <cstdio>

namespace lpl::agent {

namespace {

core::Error refuse(core::ErrorCode code, std::string_view what)
{
    return core::makeError(code, lpl::pmr::string{what.data(), what.size()}).error();
}

/// The JSON type a parsed value actually has, for comparison against a kind.
bool matches(ParamKind kind, const editor::detail::JVal &value)
{
    using T = editor::detail::JVal::T;
    switch (kind)
    {
    case ParamKind::Number: return value.t == T::Num;
    // JSON has no integer type; an integer is a number with nothing after the
    // point. Checking the value rather than the token is what makes 3.0 and 3
    // both acceptable while 3.5 is not.
    case ParamKind::Integer:
        return value.t == T::Num && value.num == static_cast<double>(static_cast<long long>(value.num));
    case ParamKind::String: return value.t == T::Str;
    case ParamKind::Bool: return value.t == T::Bool;
    case ParamKind::Object: return value.t == T::Obj;
    case ParamKind::Array: return value.t == T::Arr;
    }
    return false;
}

} // namespace

std::string ToolCall::toCommandJson() const
{
    // Rebuilt through the JSON layer rather than by splicing braces: args came
    // from a parser and goes to a parser, and a hand-rolled concatenation would
    // be a third JSON implementation waiting to disagree with the other two.
    bool ok = false;
    editor::detail::JVal parsed = editor::detail::parse(args, &ok);
    editor::detail::JVal command;
    command.t = editor::detail::JVal::T::Obj;

    editor::detail::JVal name;
    name.t = editor::detail::JVal::T::Str;
    name.str = std::string{tool->name};
    command.obj.emplace_back("cmd", name);

    if (ok && parsed.t == editor::detail::JVal::T::Obj)
        for (const auto &member : parsed.obj)
            command.obj.push_back(member);

    return editor::detail::emit(command);
}

core::Expected<ToolCall> parseToolCall(std::string_view json, const ToolRegistry &registry)
{
    bool ok = false;
    const editor::detail::JVal root = editor::detail::parse(json, &ok);
    if (!ok || root.t != editor::detail::JVal::T::Obj)
        return std::unexpected(refuse(core::ErrorCode::kDeserializationFailed, "the call is not a JSON object"));

    const editor::detail::JVal *name = root.find("tool");
    if (name == nullptr || name->t != editor::detail::JVal::T::Str)
        return std::unexpected(refuse(core::ErrorCode::kInvalidArgument, "the call has no \"tool\" string"));

    const ToolDesc *tool = registry.find(name->str);
    if (tool == nullptr)
    {
        // Two different refusals, because they teach different things: a tool
        // that does not exist is a hallucination, a tool that exists but is not
        // offered is a precondition the caller has not met yet.
        std::string message = findTool(name->str) != nullptr
                                  ? "tool \"" + name->str + "\" is not available in this world state"
                                  : "unknown tool \"" + name->str + "\"";
        return std::unexpected(refuse(core::ErrorCode::kNotFound, message));
    }

    const editor::detail::JVal *args = root.find("args");
    if (args == nullptr || args->t != editor::detail::JVal::T::Obj)
        return std::unexpected(refuse(core::ErrorCode::kInvalidArgument, "the call has no \"args\" object"));

    for (const ToolParam &param : tool->params)
    {
        const editor::detail::JVal *value = args->find(param.name);
        if (value == nullptr)
        {
            if (!param.required)
                continue;
            return std::unexpected(
                refuse(core::ErrorCode::kInvalidArgument,
                       "\"" + std::string{tool->name} + "\" requires \"" + std::string{param.name} + "\""));
        }
        if (!matches(param.kind, *value))
            return std::unexpected(refuse(core::ErrorCode::kInvalidArgument,
                                          "\"" + std::string{param.name} + "\" must be a " +
                                              std::string{jsonTypeName(param.kind)}));
        if (param.hasBounds && (value->num < param.minValue || value->num > param.maxValue))
        {
            char range[128];
            std::snprintf(range, sizeof(range), "\" is out of range [%g, %g]", param.minValue, param.maxValue);
            return std::unexpected(
                refuse(core::ErrorCode::kOutOfRange, "\"" + std::string{param.name} + std::string{range}));
        }
        if (value->t == editor::detail::JVal::T::Str && !inChoices(param.choices, value->str))
            return std::unexpected(refuse(core::ErrorCode::kInvalidArgument,
                                          "\"" + value->str + "\" is not one of the accepted values for \"" +
                                              std::string{param.name} + "\""));
    }

    // A parameter nobody declared is a typo, and a typo that silently succeeds is
    // worse than one refused with a reason: the caller would keep making it.
    for (const auto &member : args->obj)
    {
        bool declared = false;
        for (const ToolParam &param : tool->params)
            declared = declared || param.name == member.first;
        if (!declared)
            return std::unexpected(refuse(core::ErrorCode::kInvalidArgument,
                                          "\"" + std::string{tool->name} + "\" has no parameter \"" + member.first +
                                              "\""));
    }

    ToolCall call;
    call.tool = tool;
    if (const editor::detail::JVal *thought = root.find("thought");
        thought != nullptr && thought->t == editor::detail::JVal::T::Str)
        call.thought = thought->str;
    call.args = editor::detail::emit(*args);
    return call;
}

} // namespace lpl::agent
