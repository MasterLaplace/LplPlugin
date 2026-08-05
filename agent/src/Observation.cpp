/**
 * @file Observation.cpp
 * @brief Implementation of the deterministic critics.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/agent/Observation.hpp>

#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/Json.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>

#include <unordered_map>

namespace lpl::agent {

namespace {

/// Escapes @p s into @p out as a JSON string body (no quotes).
void appendEscaped(std::string &out, std::string_view s)
{
    for (const char c : s)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        default: out += c; break;
        }
    }
}

std::string_view severityName(Severity severity) noexcept
{
    switch (severity)
    {
    case Severity::Info: return "info";
    case Severity::Warning: return "warning";
    case Severity::Defect: return "defect";
    }
    return "info";
}

/// Adds a finding, honouring the bound but always counting.
void record(Observations &out, core::u32 limit, Severity severity, std::string code, std::string message,
            std::string tool = {}, std::string args = {})
{
    ++out.total;
    if (out.findings.size() >= limit)
    {
        out.truncated = true;
        return;
    }
    out.findings.push_back(Finding{severity, std::move(code), std::move(message), std::move(tool), std::move(args)});
}

/// Packs a Q16.16 XZ position into a key, for spotting entities stacked on a spot.
core::u64 positionKey(const math::Vec3<math::Fixed32> &p) noexcept
{
    const auto x = static_cast<core::u64>(static_cast<core::u32>(p.x.raw()));
    const auto z = static_cast<core::u64>(static_cast<core::u32>(p.z.raw()));
    return (x << 32u) | z;
}

/// Reads an integer field out of a report object.
core::i64 field(const editor::detail::JVal &object, std::string_view key, core::i64 fallback)
{
    const editor::detail::JVal *value = object.find(key);
    if (value == nullptr || value->t != editor::detail::JVal::T::Num)
        return fallback;
    return static_cast<core::i64>(value->num);
}

/// Whether a `generate_world` block enabled a pass; an unstated pass runs.
bool passEnabled(const editor::detail::JVal &recipe, std::string_view block)
{
    const editor::detail::JVal *node = recipe.find(block);
    if (node == nullptr || node->t != editor::detail::JVal::T::Obj)
        return true; // Absent block = engine defaults, and the defaults run it.
    const editor::detail::JVal *enabled = node->find("enabled");
    if (enabled == nullptr || enabled->t != editor::detail::JVal::T::Bool)
        return true;
    return enabled->b;
}

} // namespace

core::u32 Observations::defects() const noexcept
{
    core::u32 count = 0u;
    for (const Finding &finding : findings)
        count += finding.severity == Severity::Defect ? 1u : 0u;
    return count;
}

std::string Observations::toJson() const
{
    std::string out = "{\"findings\":" + std::to_string(total);
    out += ",\"defects\":" + std::to_string(defects());
    out += ",\"truncated\":" + std::string(truncated ? "true" : "false");
    out += ",\"items\":[";
    bool first = true;
    for (const Finding &finding : findings)
    {
        if (!first)
            out += ',';
        first = false;
        out += "{\"severity\":\"";
        out += severityName(finding.severity);
        out += "\",\"code\":\"";
        appendEscaped(out, finding.code);
        out += "\",\"message\":\"";
        appendEscaped(out, finding.message);
        out += "\"";
        if (!finding.suggestedTool.empty())
        {
            out += ",\"suggestedTool\":\"";
            appendEscaped(out, finding.suggestedTool);
            out += "\"";
        }
        if (!finding.suggestedArgs.empty())
        {
            out += ",\"suggestedArgs\":";
            out += finding.suggestedArgs;
        }
        out += '}';
    }
    out += "]}";
    return out;
}

Observations inspectWorld(const ecs::Registry &registry, core::u32 limit)
{
    Observations out;

    core::u32 entities = 0u;
    core::u32 degenerateExtents = 0u;
    core::u32 saturated = 0u;
    bool seeded = false;
    math::Fixed32 minY{};
    math::Fixed32 maxY{};
    std::unordered_map<core::u64, core::u32> occupancy;
    core::u32 mostStacked = 0u;

    // Q16.16 saturates a little past ±32767. A coordinate up against that is not
    // "far away", it is a value the arithmetic can no longer represent — the same
    // trap that made an unbounded query box mean nothing.
    constexpr core::i32 kSaturationGuard = 32000 << 16;

    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        // readComponent answers a pointer for EVERY id, allocated or not, so the
        // archetype is what must be asked whether a component is really there.
        const bool hasPosition = part->archetype().has(ecs::ComponentId::Position);
        const bool hasAabb = part->archetype().has(ecs::ComponentId::AABB);
        for (const auto &chunk : part->chunks())
        {
            if (!chunk)
                continue;
            const core::u32 count = chunk->count();
            entities += count;
            if (!hasPosition)
                continue;

            const auto *positions =
                static_cast<const math::Vec3<math::Fixed32> *>(chunk->readComponent(ecs::ComponentId::Position));
            const auto *extents =
                hasAabb ? static_cast<const math::Vec3<math::Fixed32> *>(chunk->readComponent(ecs::ComponentId::AABB))
                        : nullptr;
            if (positions == nullptr)
                continue;

            for (core::u32 i = 0u; i < count; ++i)
            {
                const math::Vec3<math::Fixed32> &p = positions[i];
                if (!seeded)
                {
                    minY = p.y;
                    maxY = p.y;
                    seeded = true;
                }
                minY = p.y < minY ? p.y : minY;
                maxY = p.y > maxY ? p.y : maxY;

                if (p.x.raw() > kSaturationGuard || p.x.raw() < -kSaturationGuard || p.y.raw() > kSaturationGuard ||
                    p.y.raw() < -kSaturationGuard || p.z.raw() > kSaturationGuard || p.z.raw() < -kSaturationGuard)
                    ++saturated;

                const core::u32 here = ++occupancy[positionKey(p)];
                mostStacked = here > mostStacked ? here : mostStacked;

                if (extents != nullptr &&
                    (extents[i].x.raw() <= 0 || extents[i].y.raw() <= 0 || extents[i].z.raw() <= 0))
                    ++degenerateExtents;
            }
        }
    }

    if (entities == 0u)
    {
        record(out, limit, Severity::Defect, "empty-world", "The world holds no entities: nothing has been generated.",
               "generate_world", "{}");
        return out;
    }

    if (saturated != 0u)
        record(out, limit, Severity::Defect, "saturated-coordinates",
               std::to_string(saturated) +
                   " entities sit at a coordinate Q16.16 can barely represent; a position past about 32767 is not far "
                   "away, it is undefined.",
               "query_entities", "{}");

    if (degenerateExtents != 0u)
        record(out, limit, Severity::Defect, "degenerate-extents",
               std::to_string(degenerateExtents) +
                   " entities have a zero or negative AABB half-extent, so nothing can ever collide with them.");

    // The Fixed32{10} signature: a scatter whose spacing was read as a raw Q16.16
    // word collapses every instance onto one square. A threshold rather than
    // "any", because a small stack is a legitimate pile.
    if (mostStacked > 16u)
        record(out, limit, Severity::Warning, "stacked-entities",
               std::to_string(mostStacked) +
                   " entities share one exact position; a scatter that collapses to a point usually means a distance "
                   "was built from a raw fixed-point word instead of a value.",
               "get_world_stats", "{}");

    if (seeded && minY.raw() == maxY.raw())
        record(out, limit, Severity::Info, "flat-world",
               "Every entity sits at the same height; the terrain has no relief.", "generate_world",
               "{\"terrain\":{\"amplitude\":12.0}}");

    return out;
}

Observations reviewGeneration(std::string_view recipeJson, std::string_view reportJson, core::u32 limit)
{
    Observations out;

    bool recipeOk = false;
    bool reportOk = false;
    const editor::detail::JVal recipe = editor::detail::parse(recipeJson, &recipeOk);
    const editor::detail::JVal answer = editor::detail::parse(reportJson, &reportOk);
    if (!recipeOk || !reportOk || recipe.t != editor::detail::JVal::T::Obj ||
        answer.t != editor::detail::JVal::T::Obj)
    {
        record(out, limit, Severity::Warning, "unreadable-report",
               "The recipe or the generation report did not parse, so nothing can be judged from them.");
        return out;
    }

    // Asked-for versus got. Every one of these is a real failure mode: a pass
    // that runs, succeeds and produces nothing looks exactly like a pass that
    // worked, until somebody goes and looks.
    if (field(answer, "created", 0) == 0)
        record(out, limit, Severity::Defect, "no-entities", "Generation reported zero entities created.",
               "generate_world", "{\"materializeGround\":true}");

    if (passEnabled(recipe, "rivers") && field(answer, "riverCells", -1) == 0)
        record(out, limit, Severity::Defect, "no-rivers",
               "Rivers were enabled and no river cell was carved; the drainage found nothing to route.",
               "generate_world", "{\"rivers\":{\"density\":0.35},\"erosion\":{\"enabled\":true}}");

    if (passEnabled(recipe, "caves") && field(answer, "caveFloor", -1) == 0)
        record(out, limit, Severity::Defect, "no-caves",
               "Caves were enabled and the underground layer is solid rock; the automaton closed every cell.",
               "generate_world", "{\"caves\":{\"fillProbability\":0.45,\"steps\":4}}");

    if (passEnabled(recipe, "settlement") && field(answer, "plots", -1) == 0)
        record(out, limit, Severity::Defect, "no-settlement",
               "A settlement was enabled and no plot was laid out; the terrain offered nowhere flat enough.",
               "generate_world", "{\"settlement\":{\"enabled\":true},\"erosion\":{\"enabled\":true}}");

    // The playability gate is the one judgement the generator already makes.
    // Re-deriving it here would be a second answer to a question procgen has
    // answered, so this reads its verdict instead.
    const editor::detail::JVal *passed = answer.find("passed");
    const editor::detail::JVal *reachable = answer.find("reachable");
    if (reachable != nullptr && reachable->t == editor::detail::JVal::T::Bool && !reachable->b)
        record(out, limit, Severity::Defect, "goal-unreachable",
               "The underground's goal cannot be reached from its entrance: the level is not playable.",
               "generate_world", "{\"caves\":{\"minRegionSize\":24}}");
    else if (passed != nullptr && passed->t == editor::detail::JVal::T::Bool && !passed->b)
        record(out, limit, Severity::Defect, "gate-failed",
               "The world did not pass its own playability gate (path length " +
                   std::to_string(field(answer, "pathLength", 0)) + ").",
               "generate_world", "{\"gate\":{}}");

    return out;
}

} // namespace lpl::agent
