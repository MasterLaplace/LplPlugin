/**
 * @file Schema.cpp
 * @brief Implementation of JSON-Schema emission from the reflection registry.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/agent/Schema.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace lpl::agent {

namespace {

/// Appends @p s as a JSON string literal, quotes included.
void appendString(std::string &out, std::string_view s)
{
    out += '"';
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
    out += '"';
}

/// Decodes a @c FieldType::F32 default/bound, which @c FieldDesc stores as bits.
float rawToFloat(core::i64 raw)
{
    const std::uint32_t bits = static_cast<std::uint32_t>(static_cast<core::i32>(raw));
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

/// The closed set of strings a @ref DynamicEnum stands for, as a JSON array.
void appendChoices(std::string &out, DynamicEnum choices)
{
    if (choices == DynamicEnum::None)
        return;
    out += ",\"enum\":[";
    bool first = true;
    forEachChoice(choices, [&](std::string_view word) {
        if (!first)
            out += ',';
        first = false;
        appendString(out, word);
    });
    out += ']';
}

/// A bound in the units the parameter is declared in.
void appendBound(std::string &out, const char *key, ParamKind kind, double value)
{
    char buf[64];
    if (kind == ParamKind::Integer)
        std::snprintf(buf, sizeof(buf), ",\"%s\":%lld", key, static_cast<long long>(value));
    else
        std::snprintf(buf, sizeof(buf), ",\"%s\":%g", key, value);
    out += buf;
}

} // namespace

std::string emitJsonSchema(const ecs::ComponentSchema &schema)
{
    // Byte-for-byte the shape test_reflection.cpp has pinned since 2026-07-16.
    // Changing it means changing that assertion in the same commit, deliberately.
    std::string out = "{\"type\":\"object\",\"title\":\"";
    out += schema.name;
    out += "\",\"properties\":{";
    bool first = true;
    for (const ecs::FieldDesc &f : schema.fields)
    {
        if (!first)
            out += ",";
        first = false;
        out += "\"";
        out += f.name;
        out += "\":{\"type\":\"";
        out += jsonTypeName(paramKindOf(f.type));
        out += "\"";
        if (f.hasBounds)
        {
            char b[96];
            if (f.type == ecs::FieldType::F32)
                std::snprintf(b, sizeof(b), ",\"minimum\":%g,\"maximum\":%g", rawToFloat(f.minRaw),
                              rawToFloat(f.maxRaw));
            else
                std::snprintf(b, sizeof(b), ",\"minimum\":%lld,\"maximum\":%lld", static_cast<long long>(f.minRaw),
                              static_cast<long long>(f.maxRaw));
            out += b;
        }
        out += "}";
    }
    out += "}}";
    return out;
}

std::string emitJsonSchema(const ToolDesc &tool)
{
    std::string out = "{\"type\":\"object\",\"title\":";
    appendString(out, tool.name);
    out += ",\"description\":";
    appendString(out, tool.brief);
    out += ",\"properties\":{";

    bool first = true;
    for (const ToolParam &p : tool.params)
    {
        if (!first)
            out += ',';
        first = false;
        appendString(out, p.name);
        out += ":{\"type\":";
        appendString(out, jsonTypeName(p.kind));
        if (!p.brief.empty())
        {
            out += ",\"description\":";
            appendString(out, p.brief);
        }
        if (p.hasBounds)
        {
            appendBound(out, "minimum", p.kind, p.minValue);
            appendBound(out, "maximum", p.kind, p.maxValue);
        }
        appendChoices(out, p.choices);
        out += '}';
    }
    out += "},\"required\":[";
    first = true;
    for (const ToolParam &p : tool.params)
    {
        if (!p.required)
            continue;
        if (!first)
            out += ',';
        first = false;
        appendString(out, p.name);
    }
    // Closed: an argument nobody declared is a typo, and a typo that silently
    // succeeds is worse than one that is refused with a reason.
    out += "],\"additionalProperties\":false}";
    return out;
}

std::string emitJsonSchema(const ToolRegistry &registry)
{
    // A call is one object: an optional thought, the tool's name, its arguments.
    // Keeping the thought inside the call is what makes a transcript entry
    // self-contained — the reasoning and the act cannot drift apart later.
    std::string out = "{\"type\":\"object\",\"title\":\"tool_call\",\"oneOf\":[";
    bool first = true;
    for (const ToolDesc *tool : registry.tools())
    {
        if (!first)
            out += ',';
        first = false;
        out += "{\"type\":\"object\",\"properties\":{\"thought\":{\"type\":\"string\"},\"tool\":{\"const\":";
        appendString(out, tool->name);
        out += "},\"args\":";
        out += emitJsonSchema(*tool);
        out += "},\"required\":[\"tool\",\"args\"],\"additionalProperties\":false}";
    }
    out += "]}";
    return out;
}

} // namespace lpl::agent
