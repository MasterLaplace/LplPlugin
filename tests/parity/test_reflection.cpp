/**
 * @file test_reflection.cpp
 * @brief Slice-1 proof for the component reflection registry.
 *
 * Proves that a SINGLE constexpr declaration per component drives every
 * consumer:
 *   1. the derived {size, alignment} equals the hand-written defaultLayout()
 *      for all 12 components — i.e. the reflection table can replace the
 *      hard-coded switch (the exact anti-pattern the whole effort removes);
 *   2. a component's raw bytes emit a named JSON object (Fixed32 as raw int);
 *   3. the same schema emits a JSON-Schema fragment (the input to validation
 *      and to the AI tool grammar / GBNF).
 *
 * Host-only. The JSON emitters live here for slice 1 and will move into the
 * editor module. Compile directly (like the cubepile oracle):
 *   g++ -std=gnu++23 -I core/include -I math/include -I ecs/include \
 *       tests/parity/test_reflection.cpp -o /tmp/test_reflection
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/ComponentReflection.hpp>

using namespace lpl;
using ecs::FieldType;

static int failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

// --- Host-side derivations from the schema (slice 1) ----------------------- //

static float rawToFloat(core::i64 raw)
{
    std::uint32_t bits = static_cast<std::uint32_t>(static_cast<core::i32>(raw));
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Emit one field's value from raw component bytes.
static std::string emitValue(const ecs::FieldDesc &f, const core::byte *data)
{
    auto readI32 = [&](core::u32 off) {
        core::i32 v;
        std::memcpy(&v, data + off, sizeof(v));
        return v;
    };
    auto readF32 = [&](core::u32 off) {
        float v;
        std::memcpy(&v, data + off, sizeof(v));
        return v;
    };
    char buf[128];
    switch (f.type)
    {
    case FieldType::F32: std::snprintf(buf, sizeof(buf), "%g", readF32(f.offset)); break;
    case FieldType::I32: std::snprintf(buf, sizeof(buf), "%d", readI32(f.offset)); break;
    case FieldType::U32:
        std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(readI32(f.offset)));
        break;
    case FieldType::U16: {
        std::uint16_t v;
        std::memcpy(&v, data + f.offset, sizeof(v));
        std::snprintf(buf, sizeof(buf), "%u", v);
        break;
    }
    case FieldType::U8: std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(data[f.offset])); break;
    case FieldType::Fixed32:
        // Authoritative: emit the RAW integer (deterministic, kernel<->oracle).
        std::snprintf(buf, sizeof(buf), "%d", readI32(f.offset));
        break;
    case FieldType::Vec3F:
        std::snprintf(buf, sizeof(buf), "{\"x\":%g,\"y\":%g,\"z\":%g}", readF32(f.offset),
                      readF32(f.offset + 4), readF32(f.offset + 8));
        break;
    case FieldType::Vec3Fixed:
        std::snprintf(buf, sizeof(buf), "{\"x\":%d,\"y\":%d,\"z\":%d}", readI32(f.offset),
                      readI32(f.offset + 4), readI32(f.offset + 8));
        break;
    case FieldType::QuatF:
        std::snprintf(buf, sizeof(buf), "{\"x\":%g,\"y\":%g,\"z\":%g,\"w\":%g}", readF32(f.offset),
                      readF32(f.offset + 4), readF32(f.offset + 8), readF32(f.offset + 12));
        break;
    }
    return buf;
}

static std::string emitJson(const ecs::ComponentSchema &s, const core::byte *data)
{
    std::string out = "{";
    bool first = true;
    for (const ecs::FieldDesc &f : s.fields)
    {
        if (!first)
            out += ",";
        first = false;
        out += "\"";
        out += f.name;
        out += "\":";
        out += emitValue(f, data);
    }
    out += "}";
    return out;
}

static const char *jsonType(FieldType t)
{
    switch (t)
    {
    case FieldType::F32: return "number";
    case FieldType::Fixed32: return "integer"; // raw
    case FieldType::I32:
    case FieldType::U32:
    case FieldType::U16:
    case FieldType::U8: return "integer";
    default: return "object";
    }
}

static std::string emitJsonSchema(const ecs::ComponentSchema &s)
{
    std::string out = "{\"type\":\"object\",\"title\":\"";
    out += s.name;
    out += "\",\"properties\":{";
    bool first = true;
    for (const ecs::FieldDesc &f : s.fields)
    {
        if (!first)
            out += ",";
        first = false;
        out += "\"";
        out += f.name;
        out += "\":{\"type\":\"";
        out += jsonType(f.type);
        out += "\"";
        if (f.hasBounds)
        {
            char b[96];
            if (f.type == FieldType::F32)
                std::snprintf(b, sizeof(b), ",\"minimum\":%g,\"maximum\":%g", rawToFloat(f.minRaw),
                              rawToFloat(f.maxRaw));
            else
                std::snprintf(b, sizeof(b), ",\"minimum\":%lld,\"maximum\":%lld",
                              static_cast<long long>(f.minRaw), static_cast<long long>(f.maxRaw));
            out += b;
        }
        out += "}";
    }
    out += "}}";
    return out;
}

int main()
{
    std::printf("== component reflection parity (slice 1) ==\n\n");

    // 1. Derived layout must equal the hand-written defaultLayout() for all 12.
    std::printf("-- derived layout == defaultLayout() --\n");
    for (const ecs::ComponentSchema &s : ecs::allSchemas())
    {
        const ecs::DerivedLayout d = ecs::computeLayout(s);
        const ecs::ComponentLayout h = ecs::defaultLayout(s.id);
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%-16.*s size %u==%u  align %u==%u",
                      static_cast<int>(s.name.size()), s.name.data(), d.size, h.size, d.alignment,
                      h.alignment);
        check(d.size == h.size && d.alignment == h.alignment, msg);
    }

    // 2. Name lookup round-trips.
    std::printf("\n-- name <-> id --\n");
    check(ecs::componentIdByName("Health") == ecs::ComponentId::Health, "componentIdByName(\"Health\")");
    check(ecs::componentIdByName("Nonexistent") == ecs::ComponentId::Count, "unknown name -> Count");

    // 3. JSON emission from raw bytes, driven by the schema.
    std::printf("\n-- JSON from bytes (schema-driven) --\n");
    {
        // A Position = Vec3<Fixed32>{1.5, -2.0, 3.25} → raw Q16.16 ints
        // 1.5=0x18000=98304, -2.0=-131072, 3.25=0x34000=212992
        alignas(4) core::byte buf[16] = {};
        const core::i32 p[3] = {98304, -131072, 212992};
        std::memcpy(buf, p, sizeof(p));
        const std::string js = emitJson(ecs::schemaOf(ecs::ComponentId::Position), buf);
        std::printf("  Position -> %s\n", js.c_str());
        check(js == "{\"value\":{\"x\":98304,\"y\":-131072,\"z\":212992}}", "Position JSON (Fixed32 raw) matches");
    }
    {
        // A Health = i32{ 350 }
        alignas(4) core::byte buf[4] = {};
        const core::i32 hp = 350;
        std::memcpy(buf, &hp, sizeof(hp));
        const std::string js = emitJson(ecs::schemaOf(ecs::ComponentId::Health), buf);
        std::printf("  Health   -> %s\n", js.c_str());
        check(js == "{\"points\":350}", "Health JSON matches");
    }

    // 4. JSON-Schema emission (the GBNF / validation input) from the SAME decl.
    std::printf("\n-- JSON-Schema from the same declaration --\n");
    std::printf("  Health   -> %s\n", emitJsonSchema(ecs::schemaOf(ecs::ComponentId::Health)).c_str());
    std::printf("  Mass     -> %s\n", emitJsonSchema(ecs::schemaOf(ecs::ComponentId::Mass)).c_str());
    std::printf("  Position -> %s\n", emitJsonSchema(ecs::schemaOf(ecs::ComponentId::Position)).c_str());
    check(emitJsonSchema(ecs::schemaOf(ecs::ComponentId::Health)) ==
              "{\"type\":\"object\",\"title\":\"Health\",\"properties\":"
              "{\"points\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":1000000}}}",
          "Health JSON-Schema matches");

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
