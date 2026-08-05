/**
 * @file Dispatcher.cpp
 * @brief Implementation of the abstract execution seam.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/agent/Dispatcher.hpp>

#include <lpl/agent/Vision.hpp>
#include <lpl/core/Error.hpp>
#include <lpl/editor/CommandJournal.hpp>
#include <lpl/editor/Json.hpp>

namespace lpl::agent {

namespace {

std::string reportError(std::string_view tool, std::string_view message)
{
    std::string out = "{\"cmd\":\"";
    out.append(tool);
    out += "\",\"ok\":false,\"error\":\"";
    for (const char c : message)
    {
        if (c == '"' || c == '\\')
            out += '\\';
        out += c;
    }
    out += "\"}";
    return out;
}

} // namespace

core::Expected<std::string> Dispatcher::dispatchHere(const ToolCall &call)
{
    bool ok = false;
    const editor::detail::JVal args = editor::detail::parse(call.args, &ok);
    if (!ok || args.t != editor::detail::JVal::T::Obj)
        return reportError(call.tool->name, "arguments did not parse");

    if (call.tool->name == "take_screenshot")
    {
        const editor::detail::JVal *path = args.find("path");
        if (path == nullptr || path->t != editor::detail::JVal::T::Str)
            return reportError("take_screenshot", "missing \"path\"");

        CameraPose pose;
        pose.yawDeg = static_cast<core::f32>(args.numOr("yawDeg", pose.yawDeg));
        pose.pitchDeg = static_cast<core::f32>(args.numOr("pitchDeg", pose.pitchDeg));
        pose.distance = static_cast<core::f32>(args.numOr("distance", 0.0));
        const auto width = static_cast<core::u32>(args.numOr("width", 480.0));
        const auto height = static_cast<core::u32>(args.numOr("height", 300.0));

        auto shot = captureToFile(_registry, path->str, width, height, pose);
        if (!shot.has_value())
            return reportError("take_screenshot", shot.error().message().c_str());

        std::string out = "{\"cmd\":\"take_screenshot\",\"ok\":true,\"path\":\"";
        out += path->str;
        out += "\",\"width\":" + std::to_string(shot.value().width);
        out += ",\"height\":" + std::to_string(shot.value().height);
        out += ",\"entities\":" + std::to_string(shot.value().entitiesDrawn);
        out += ",\"triangles\":" + std::to_string(shot.value().triangles);
        // The perceptual signature, not a parity signature: it answers "did the
        // picture change", which is the question a correction loop asks.
        out += ",\"frameSignature\":" + std::to_string(shot.value().fold);
        out += "}";
        return out;
    }

    return reportError(call.tool->name, "no agent-side handler for this tool");
}

core::Expected<std::string> Dispatcher::dispatch(const ToolCall &call)
{
    if (call.tool->host == ToolHost::Agent)
        return dispatchHere(call);

    auto answer = _journal.execute(call.toCommandJson());
    if (!answer.has_value())
        return answer;
    if (call.tool->mutates)
        ++_mutations;
    return answer;
}

core::Expected<std::string> Dispatcher::dispatchJson(std::string_view json, const ToolRegistry &registry)
{
    auto call = parseToolCall(json, registry);
    if (!call.has_value())
        return std::unexpected(std::move(call.error()));
    return dispatch(call.value());
}

bool Dispatcher::undo()
{
    if (!_journal.undo())
        return false;
    if (_mutations != 0u)
        --_mutations;
    return true;
}

} // namespace lpl::agent
