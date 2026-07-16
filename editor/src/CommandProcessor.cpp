/**
 * @file CommandProcessor.cpp
 * @brief Implementation of the JSON command interface.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <lpl/editor/CommandProcessor.hpp>

#include <lpl/core/Error.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/SceneSerializer.hpp>
#include <lpl/procgen/HeightfieldGenerator.hpp>
#include <lpl/procgen/PlayabilityGate.hpp>
#include <lpl/procgen/PoissonScatter.hpp>
#include <lpl/editor/Json.hpp>

#include <cstdio>

namespace lpl::editor {


namespace {

// Escapes a string for embedding as a JSON string value.
void appendEscaped(std::string &out, std::string_view s)
{
    for (const char c : s)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
}

std::string reportError(std::string_view cmd, std::string_view msg)
{
    std::string out = "{\"cmd\":\"";
    appendEscaped(out, cmd);
    out += "\",\"ok\":false,\"error\":\"";
    appendEscaped(out, msg);
    out += "\"}";
    return out;
}

// Executes a single command object; always returns a JSON report object.
std::string executeOne(ecs::Registry &registry, const detail::JVal &cmdObj)
{
    const detail::JVal *cmdVal = cmdObj.find("cmd");
    if (cmdVal == nullptr || cmdVal->t != detail::JVal::T::Str)
        return reportError("?", "missing \"cmd\" field");
    const std::string &cmd = cmdVal->str;

    if (cmd == "generate_heightfield")
    {
        procgen::HeightfieldParams p;
        p.seed = static_cast<core::u32>(cmdObj.numOr("seed", p.seed));
        p.cols = static_cast<core::u32>(cmdObj.numOr("cols", p.cols));
        p.rows = static_cast<core::u32>(cmdObj.numOr("rows", p.rows));
        p.spacing = static_cast<core::f32>(cmdObj.numOr("spacing", p.spacing));
        p.noiseScale = static_cast<core::f32>(cmdObj.numOr("noiseScale", p.noiseScale));
        p.amplitude = static_cast<core::f32>(cmdObj.numOr("amplitude", p.amplitude));
        p.octaves = static_cast<core::u32>(cmdObj.numOr("octaves", p.octaves));
        p.cubeHalf = static_cast<core::f32>(cmdObj.numOr("cubeHalf", p.cubeHalf));
        const core::u32 created = procgen::generateHeightfield(registry, p);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "{\"cmd\":\"generate_heightfield\",\"ok\":true,\"created\":%u}", created);
        return buf;
    }
    if (cmd == "scatter_poisson")
    {
        procgen::PoissonScatterParams p;
        p.seed = static_cast<core::u32>(cmdObj.numOr("seed", p.seed));
        p.width = static_cast<core::f32>(cmdObj.numOr("width", p.width));
        p.depth = static_cast<core::f32>(cmdObj.numOr("depth", p.depth));
        p.radius = static_cast<core::f32>(cmdObj.numOr("radius", p.radius));
        p.propHalf = static_cast<core::f32>(cmdObj.numOr("propHalf", p.propHalf));
        p.maxPoints = static_cast<core::u32>(cmdObj.numOr("maxPoints", p.maxPoints));
        const core::u32 created = procgen::scatterPoisson(registry, p);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "{\"cmd\":\"scatter_poisson\",\"ok\":true,\"created\":%u}", created);
        return buf;
    }
    if (cmd == "check_playability")
    {
        procgen::PlayabilityParams p;
        p.seed = static_cast<core::u32>(cmdObj.numOr("seed", p.seed));
        p.cols = static_cast<core::u32>(cmdObj.numOr("cols", p.cols));
        p.rows = static_cast<core::u32>(cmdObj.numOr("rows", p.rows));
        p.noiseScale = static_cast<core::f32>(cmdObj.numOr("noiseScale", p.noiseScale));
        p.octaves = static_cast<core::u32>(cmdObj.numOr("octaves", p.octaves));
        p.wallThreshold = static_cast<core::f32>(cmdObj.numOr("wallThreshold", p.wallThreshold));
        p.startCol = static_cast<core::u32>(cmdObj.numOr("startCol", p.startCol));
        p.startRow = static_cast<core::u32>(cmdObj.numOr("startRow", p.startRow));
        p.goalCol = static_cast<core::u32>(cmdObj.numOr("goalCol", p.goalCol));
        p.goalRow = static_cast<core::u32>(cmdObj.numOr("goalRow", p.goalRow));
        const procgen::PlayabilityResult v = procgen::evaluateReachability(p);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "{\"cmd\":\"check_playability\",\"ok\":true,\"reachable\":%s,\"pathCostRaw\":%d,\"visited\":%u}",
                      v.reachable ? "true" : "false", v.pathCost.raw(), v.visited);
        return buf;
    }
    if (cmd == "load_scene")
    {
        const detail::JVal *scene = cmdObj.find("scene");
        if (scene == nullptr || scene->t != detail::JVal::T::Str)
            return reportError("load_scene", "missing \"scene\" string");
        const auto loaded = fromLplScene(scene->str, registry);
        if (!loaded.has_value())
            return reportError("load_scene", "fromLplScene failed");
        char buf[80];
        std::snprintf(buf, sizeof(buf), "{\"cmd\":\"load_scene\",\"ok\":true,\"created\":%u}", loaded.value());
        return buf;
    }
    if (cmd == "save_scene")
    {
        const std::string doc = toLplScene(registry);
        std::string out = "{\"cmd\":\"save_scene\",\"ok\":true,\"scene\":\"";
        appendEscaped(out, doc);
        out += "\"}";
        return out;
    }
    if (cmd == "count")
    {
        char buf[72];
        std::snprintf(buf, sizeof(buf), "{\"cmd\":\"count\",\"ok\":true,\"entities\":%u}", entityCount(registry));
        return buf;
    }
    return reportError(cmd, "unknown command");
}

} // namespace

core::u32 entityCount(const ecs::Registry &registry)
{
    core::u32 total = 0;
    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
            if (chunk)
                total += chunk->count();
    }
    return total;
}

core::Expected<std::string> CommandProcessor::execute(std::string_view json)
{
    detail::Parser parser{json, 0, true};
    const detail::JVal root = parser.value();
    if (!parser.ok)
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"malformed command JSON"});

    if (root.t == detail::JVal::T::Obj)
        return executeOne(registry_, root);

    if (root.t == detail::JVal::T::Arr)
    {
        std::string out = "[";
        bool first = true;
        for (const detail::JVal &cmd : root.arr)
        {
            if (cmd.t != detail::JVal::T::Obj)
                continue;
            if (!first)
                out += ',';
            first = false;
            out += executeOne(registry_, cmd);
        }
        out += ']';
        return out;
    }
    return core::makeError(core::ErrorCode::kNotSupported, lpl::pmr::string{"command must be an object or array"});
}

} // namespace lpl::editor
