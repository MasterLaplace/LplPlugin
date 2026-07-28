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
#include <lpl/ecs/ComponentReflection.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/GamePackBaker.hpp>
#include <lpl/editor/Json.hpp>
#include <lpl/editor/SceneSerializer.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cstdio>
#include <string>
#include <vector>

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

/// Aggregate the inspection commands report on.
struct WorldStats {
    core::u32 entityCount{0u};
    core::u32 archetypeCount{0u};
    core::u32 chunkCount{0u};
    core::u32 perComponent[static_cast<core::u32>(ecs::ComponentId::Count)]{};
    math::Fixed32 minX{}, minY{}, minZ{}, maxX{}, maxY{}, maxZ{};
};

WorldStats collectWorldStats(const ecs::Registry &registry)
{
    WorldStats stats;
    bool seeded = false;

    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        ++stats.archetypeCount;
        for (const auto &chunk : part->chunks())
        {
            if (!chunk)
                continue;
            ++stats.chunkCount;
            const core::u32 n = chunk->count();
            stats.entityCount += n;

            for (const auto &schema : ecs::allSchemas())
                if (chunk->readComponent(schema.id) != nullptr)
                    stats.perComponent[static_cast<core::u32>(schema.id)] += n;

            const auto *pos =
                static_cast<const math::Vec3<math::Fixed32> *>(chunk->readComponent(ecs::ComponentId::Position));
            if (pos == nullptr)
                continue;
            for (core::u32 i = 0u; i < n; ++i)
            {
                if (!seeded)
                {
                    stats.minX = stats.maxX = pos[i].x;
                    stats.minY = stats.maxY = pos[i].y;
                    stats.minZ = stats.maxZ = pos[i].z;
                    seeded = true;
                    continue;
                }
                if (pos[i].x < stats.minX)
                    stats.minX = pos[i].x;
                if (pos[i].x > stats.maxX)
                    stats.maxX = pos[i].x;
                if (pos[i].y < stats.minY)
                    stats.minY = pos[i].y;
                if (pos[i].y > stats.maxY)
                    stats.maxY = pos[i].y;
                if (pos[i].z < stats.minZ)
                    stats.minZ = pos[i].z;
                if (pos[i].z > stats.maxZ)
                    stats.maxZ = pos[i].z;
            }
        }
    }
    return stats;
}

// Executes a single command object; always returns a JSON report object.
std::string executeOne(ecs::Registry &registry, const detail::JVal &cmdObj)
{
    const detail::JVal *cmdVal = cmdObj.find("cmd");
    if (cmdVal == nullptr || cmdVal->t != detail::JVal::T::Str)
        return reportError("?", "missing \"cmd\" field");
    const std::string &cmd = cmdVal->str;

    if (cmd == "generate_world")
    {
        // One command for the whole pipeline, because a world IS one pipeline:
        // erosion needs the terrain it erodes, moisture needs the drainage the
        // rivers left, biomes need the moisture. Exposing them as separate verbs
        // would only let a caller ask for them in an order that cannot work, and
        // the per-pass "enabled" switches already give the same control.
        //
        // The command object IS a "procedural" block, so it is read by the very
        // reader a `.lplscene` goes through — one schema, not a parallel one that
        // drifts. Anything an editor can build, a document can carry.
        detail::JVal format;
        format.t = detail::JVal::T::Str;
        format.str = "lplscene/1";
        detail::JVal document;
        document.t = detail::JVal::T::Obj;
        document.obj.emplace_back("format", format);
        document.obj.emplace_back("procedural", cmdObj);

        procgen::WorldRecipe recipe{};
        if (const auto parsed = parseSceneRecipe(detail::emit(document), recipe); !parsed)
            return reportError("generate_world", "the recipe is not valid");

        const procgen::WorldRecipeResult baked = procgen::bakeWorld(registry, recipe);
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "{\"cmd\":\"generate_world\",\"ok\":true,\"created\":%u,\"stateSignature\":%u,"
                      "\"heightSignature\":%u,\"biomeSignature\":%u,\"riverCells\":%u,\"caveFloor\":%u,"
                      "\"plots\":%u,\"reachable\":%s,\"pathLength\":%u,\"passed\":%s}",
                      baked.entityCount, baked.stateSignature, baked.heightSignature, baked.biomeSignature,
                      baked.riverCells, baked.dungeonFloor, baked.settlementPlots,
                      baked.gateReachable != 0u ? "true" : "false", baked.gatePathLength,
                      baked.ok != 0u ? "true" : "false");
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
    if (cmd == "spawn_from_template")
    {
        // The pointwise counterpart of the procedural passes (§5.3): place one
        // instance of a named prefab. Implemented by handing SceneSerializer a
        // one-entity document rather than by a second instantiation path, so
        // template chains, field overrides and validation behave identically
        // here and when a whole scene loads.
        const detail::JVal *templates = cmdObj.find("templates");
        const detail::JVal *name = cmdObj.find("name");
        if (name == nullptr || name->t != detail::JVal::T::Str)
            return reportError("spawn_from_template", "missing \"name\" string");
        if (templates == nullptr || templates->t != detail::JVal::T::Obj)
            return reportError("spawn_from_template", "missing \"templates\" object");

        detail::JVal entity;
        entity.t = detail::JVal::T::Obj;
        entity.obj.emplace_back("$use", *name);
        // Optional per-instance overrides, merged field-wise like any entity.
        if (const detail::JVal *overrides = cmdObj.find("overrides");
            overrides != nullptr && overrides->t == detail::JVal::T::Obj)
        {
            for (const auto &component : overrides->obj)
                entity.obj.push_back(component);
        }

        detail::JVal entities;
        entities.t = detail::JVal::T::Arr;
        const core::u32 count = static_cast<core::u32>(cmdObj.numOr("count", 1.0));
        for (core::u32 i = 0u; i < count; ++i)
            entities.arr.push_back(entity);

        detail::JVal document;
        document.t = detail::JVal::T::Obj;
        detail::JVal format;
        format.t = detail::JVal::T::Str;
        format.str = "lplscene/1";
        document.obj.emplace_back("format", format);
        document.obj.emplace_back("templates", *templates);
        document.obj.emplace_back("entities", entities);

        const auto spawned = fromLplScene(detail::emit(document), registry);
        if (!spawned.has_value())
            return reportError("spawn_from_template", "instantiation failed");
        char buf[88];
        std::snprintf(buf, sizeof(buf), "{\"cmd\":\"spawn_from_template\",\"ok\":true,\"created\":%u}",
                      spawned.value());
        return buf;
    }
    if (cmd == "clear_world")
    {
        const core::u32 removed = destroyAllEntities(registry);
        char buf[80];
        std::snprintf(buf, sizeof(buf), "{\"cmd\":\"clear_world\",\"ok\":true,\"destroyed\":%u}", removed);
        return buf;
    }
    // ── Inspection ──────────────────────────────────────────────────────────
    // The report calls this "la moitié oubliée par tous les designs": a caller
    // that can only mutate is flying blind. These are also the first calls an
    // AI bridge needs — look before you act — which is why they belong to the
    // plan rather than to the agentic phase.
    if (cmd == "get_world_stats")
    {
        WorldStats stats = collectWorldStats(registry);
        std::string out = "{\"cmd\":\"get_world_stats\",\"ok\":true";
        out += ",\"entities\":" + std::to_string(stats.entityCount);
        out += ",\"archetypes\":" + std::to_string(stats.archetypeCount);
        out += ",\"chunks\":" + std::to_string(stats.chunkCount);
        out += ",\"stateSignature\":" + std::to_string(procgen::foldWorldState(registry));
        if (stats.entityCount != 0u)
        {
            // Bounds in raw Q16.16: the wire stays integral, so a reader
            // reconstructs the exact same box the engine measured.
            out += ",\"boundsMinRaw\":{\"x\":" + std::to_string(stats.minX.raw()) +
                   ",\"y\":" + std::to_string(stats.minY.raw()) + ",\"z\":" + std::to_string(stats.minZ.raw()) + "}";
            out += ",\"boundsMaxRaw\":{\"x\":" + std::to_string(stats.maxX.raw()) +
                   ",\"y\":" + std::to_string(stats.maxY.raw()) + ",\"z\":" + std::to_string(stats.maxZ.raw()) + "}";
        }
        out += ",\"components\":{";
        bool first = true;
        for (const auto &schema : ecs::allSchemas())
        {
            const core::u32 n = stats.perComponent[static_cast<core::u32>(schema.id)];
            if (n == 0u)
                continue;
            if (!first)
                out += ",";
            first = false;
            out += "\"";
            out.append(schema.name.data(), schema.name.size());
            out += "\":" + std::to_string(n);
        }
        out += "}}";
        return out;
    }
    if (cmd == "query_entities")
    {
        // Filter: an optional component name every match must carry, plus an
        // optional axis-aligned box in human units. Absent filter = everything.
        ecs::ComponentId required = ecs::ComponentId::Position;
        bool hasRequired = false;
        if (const detail::JVal *withComponent = cmdObj.find("with");
            withComponent != nullptr && withComponent->t == detail::JVal::T::Str)
        {
            // componentIdByName returns ComponentId::Count for an unknown name.
            const ecs::ComponentId resolved = ecs::componentIdByName(withComponent->str);
            if (resolved == ecs::ComponentId::Count)
                return reportError("query_entities", "unknown component in \"with\"");
            required = resolved;
            hasRequired = true;
        }

        const bool hasBox = cmdObj.find("minX") != nullptr || cmdObj.find("maxX") != nullptr ||
                            cmdObj.find("minY") != nullptr || cmdObj.find("maxY") != nullptr ||
                            cmdObj.find("minZ") != nullptr || cmdObj.find("maxZ") != nullptr;

        // An absent bound means "unbounded on that side", and the sentinel for
        // that is the raw Q16.16 extreme — NOT a huge float. Fixed32 saturates
        // around +-32767, so fromFloat(1e9) does not mean "very large", it means
        // whatever the conversion happens to land on. Same trap as the
        // Fixed32{10} raw-vs-value confusion.
        const auto bound = [&cmdObj](const char *key, math::Fixed32 fallback) {
            const detail::JVal *value = cmdObj.find(key);
            if (value == nullptr || value->t != detail::JVal::T::Num)
                return fallback;
            return math::Fixed32::fromFloat(static_cast<core::f32>(value->num));
        };
        const math::Fixed32 kLowest = math::Fixed32::fromRaw(-2147483647 - 1);
        const math::Fixed32 kHighest = math::Fixed32::fromRaw(2147483647);
        const math::Fixed32 minX = bound("minX", kLowest);
        const math::Fixed32 maxX = bound("maxX", kHighest);
        const math::Fixed32 minY = bound("minY", kLowest);
        const math::Fixed32 maxY = bound("maxY", kHighest);
        const math::Fixed32 minZ = bound("minZ", kLowest);
        const math::Fixed32 maxZ = bound("maxZ", kHighest);

        // A dump of every match would be unbounded, and the report's CCR note
        // is explicit that bulky tool returns must be skimmed, not streamed
        // whole. So: always the count, and at most `limit` sample indices.
        const core::u32 limit = static_cast<core::u32>(cmdObj.numOr("limit", 16.0));

        core::u32 matched = 0u;
        core::u32 flatIndex = 0u;
        std::string samples;
        for (const auto &part : registry.partitions())
        {
            if (!part)
                continue;
            for (const auto &chunk : part->chunks())
            {
                if (!chunk)
                    continue;
                const core::u32 n = chunk->count();
                const bool carries = !hasRequired || chunk->readComponent(required) != nullptr;
                const auto *pos =
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk->readComponent(ecs::ComponentId::Position));
                for (core::u32 i = 0u; i < n; ++i, ++flatIndex)
                {
                    if (!carries)
                        continue;
                    if (hasBox)
                    {
                        if (pos == nullptr)
                            continue;
                        const auto &p = pos[i];
                        if (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY || p.z < minZ || p.z > maxZ)
                            continue;
                    }
                    if (matched < limit)
                    {
                        if (!samples.empty())
                            samples += ",";
                        samples += std::to_string(flatIndex);
                    }
                    ++matched;
                }
            }
        }
        std::string out = "{\"cmd\":\"query_entities\",\"ok\":true,\"matched\":" + std::to_string(matched);
        out += ",\"truncated\":" + std::string(matched > limit ? "true" : "false");
        out += ",\"indices\":[" + samples + "]}";
        return out;
    }
    return reportError(cmd, "unknown command");
}

} // namespace

core::u32 destroyAllEntities(ecs::Registry &registry)
{
    std::vector<ecs::EntityId> ids;
    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
            if (chunk)
                for (const ecs::EntityId id : chunk->entities())
                    ids.push_back(id);
    }
    for (const ecs::EntityId id : ids)
        (void) registry.destroyEntity(id);
    return static_cast<core::u32>(ids.size());
}

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
