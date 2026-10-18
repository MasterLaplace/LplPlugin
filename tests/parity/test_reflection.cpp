#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <lpl/agent/Schema.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/ComponentReflection.hpp>

static int failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

/**
 * @name Host-side derivations from the schema (slice 1)
 * @{
 */

/**
 * @brief Emits one field's value from raw component bytes.
 *
 * @details A Fixed32 field is authoritative: it is emitted as the RAW integer (deterministic, kernel<->oracle).
 */
static std::string emitValue(const lpl::ecs::FieldDesc &f, const lpl::core::byte *data)
{
    auto readI32 = [&](lpl::core::u32 off) {
        lpl::core::i32 v;
        std::memcpy(&v, data + off, sizeof(v));
        return v;
    };
    auto readF32 = [&](lpl::core::u32 off) {
        float v;
        std::memcpy(&v, data + off, sizeof(v));
        return v;
    };
    char buf[128];
    switch (f.type)
    {
    case lpl::ecs::FieldType::F32: std::snprintf(buf, sizeof(buf), "%g", readF32(f.offset)); break;
    case lpl::ecs::FieldType::I32: std::snprintf(buf, sizeof(buf), "%d", readI32(f.offset)); break;
    case lpl::ecs::FieldType::U32:
        std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(readI32(f.offset)));
        break;
    case lpl::ecs::FieldType::U16: {
        std::uint16_t v;
        std::memcpy(&v, data + f.offset, sizeof(v));
        std::snprintf(buf, sizeof(buf), "%u", v);
        break;
    }
    case lpl::ecs::FieldType::U8: std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(data[f.offset])); break;
    case lpl::ecs::FieldType::Fixed32: std::snprintf(buf, sizeof(buf), "%d", readI32(f.offset)); break;
    case lpl::ecs::FieldType::Vec3F:
        std::snprintf(buf, sizeof(buf), "{\"x\":%g,\"y\":%g,\"z\":%g}", readF32(f.offset), readF32(f.offset + 4),
                      readF32(f.offset + 8));
        break;
    case lpl::ecs::FieldType::Vec3Fixed:
        std::snprintf(buf, sizeof(buf), "{\"x\":%d,\"y\":%d,\"z\":%d}", readI32(f.offset), readI32(f.offset + 4),
                      readI32(f.offset + 8));
        break;
    case lpl::ecs::FieldType::QuatF:
        std::snprintf(buf, sizeof(buf), "{\"x\":%g,\"y\":%g,\"z\":%g,\"w\":%g}", readF32(f.offset),
                      readF32(f.offset + 4), readF32(f.offset + 8), readF32(f.offset + 12));
        break;
    }
    return buf;
}

static std::string emitJson(const lpl::ecs::ComponentSchema &s, const lpl::core::byte *data)
{
    std::string out = "{";
    bool first = true;
    for (const lpl::ecs::FieldDesc &f : s.fields)
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

/** @} */

/**
 * @brief The derived layout must equal the hand-written defaultLayout(), for every component.
 */
static void checkDerivedLayoutEqualsDefaultLayout()
{
    std::printf("-- derived layout == defaultLayout() --\n");
    for (const lpl::ecs::ComponentSchema &s : lpl::ecs::allSchemas())
    {
        const lpl::ecs::DerivedLayout d = lpl::ecs::computeLayout(s);
        const lpl::ecs::ComponentLayout h = lpl::ecs::defaultLayout(s.id);
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%-16.*s size %u==%u  align %u==%u", static_cast<int>(s.name.size()),
                      s.name.data(), d.size, h.size, d.alignment, h.alignment);
        check(d.size == h.size && d.alignment == h.alignment, msg);
    }
}

/**
 * @brief Name lookup round-trips.
 */
static void checkNameLookupRoundTrips()
{
    std::printf("\n-- name <-> id --\n");
    check(lpl::ecs::componentIdByName("Health") == lpl::ecs::ComponentId::Health, "componentIdByName(\"Health\")");
    check(lpl::ecs::componentIdByName("Nonexistent") == lpl::ecs::ComponentId::Count, "unknown name -> Count");
}

/**
 * @brief A Position emits its Fixed32 lanes as raw integers.
 *
 * @details A Position = Vec3<Fixed32>{1.5, -2.0, 3.25} → raw Q16.16 ints 1.5=0x18000=98304, -2.0=-131072,
 *          3.25=0x34000=212992.
 */
static void checkPositionJsonCarriesTheRawFixed32()
{
    alignas(4) lpl::core::byte buf[16] = {};
    const lpl::core::i32 p[3] = {98304, -131072, 212992};
    std::memcpy(buf, p, sizeof(p));
    const std::string js = emitJson(lpl::ecs::schemaOf(lpl::ecs::ComponentId::Position), buf);
    std::printf("  Position -> %s\n", js.c_str());
    check(js == "{\"value\":{\"x\":98304,\"y\":-131072,\"z\":212992}}", "Position JSON (Fixed32 raw) matches");
}

/**
 * @brief A Health = i32{ 350 } emits its points.
 */
static void checkHealthJsonCarriesItsPoints()
{
    alignas(4) lpl::core::byte buf[4] = {};
    const lpl::core::i32 hp = 350;
    std::memcpy(buf, &hp, sizeof(hp));
    const std::string js = emitJson(lpl::ecs::schemaOf(lpl::ecs::ComponentId::Health), buf);
    std::printf("  Health   -> %s\n", js.c_str());
    check(js == "{\"points\":350}", "Health JSON matches");
}

/**
 * @brief JSON emission from raw bytes, driven by the schema.
 */
static void checkJsonEmissionFromRawBytes()
{
    std::printf("\n-- JSON from bytes (schema-driven) --\n");
    checkPositionJsonCarriesTheRawFixed32();
    checkHealthJsonCarriesItsPoints();
}

/**
 * @brief JSON-Schema emission (the GBNF / validation input) from the SAME declaration.
 *
 * @details emitJsonSchema used to live here as a file-local static, and this file's header once said the emitters "will
 *          move into the editor module". It moved to agent/, because that is where the second consumer turned up: a
 *          JSON-Schema is only ever wanted in order to constrain a model. The assertion is unchanged and now pins
 *          the emitter the GBNF grammar is actually built from — two emitters that agreed today would be the
 *          mixColours duplication of §33 waiting to happen.
 */
static void checkJsonSchemaEmissionFromTheSameDeclaration()
{
    std::printf("\n-- JSON-Schema from the same declaration --\n");
    std::printf("  Health   -> %s\n",
                lpl::agent::emitJsonSchema(lpl::ecs::schemaOf(lpl::ecs::ComponentId::Health)).c_str());
    std::printf("  Mass     -> %s\n",
                lpl::agent::emitJsonSchema(lpl::ecs::schemaOf(lpl::ecs::ComponentId::Mass)).c_str());
    std::printf("  Position -> %s\n",
                lpl::agent::emitJsonSchema(lpl::ecs::schemaOf(lpl::ecs::ComponentId::Position)).c_str());
    check(lpl::agent::emitJsonSchema(lpl::ecs::schemaOf(lpl::ecs::ComponentId::Health)) ==
              "{\"type\":\"object\",\"title\":\"Health\",\"properties\":"
              "{\"points\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":1000000}}}",
          "Health JSON-Schema matches");
}

/**
 * @brief Slice-1 proof for the component reflection registry.
 *
 * Proves that a SINGLE constexpr declaration per component drives every
 * consumer:
 *   1. the derived {size, alignment} equals the hand-written defaultLayout()
 *      for every component — i.e. the reflection table can replace the
 *      hard-coded switch (the exact anti-pattern the whole effort removes);
 *   2. a component's raw bytes emit a named JSON object (Fixed32 as raw int);
 *   3. the same schema emits a JSON-Schema fragment (the input to validation
 *      and to the AI tool grammar / GBNF).
 *
 * Host-only. The JSON value emitter lives here; the JSON-Schema emitter is agent's.
 */
int main()
{
    std::printf("== component reflection parity (slice 1) ==\n\n");

    checkDerivedLayoutEqualsDefaultLayout();

    checkNameLookupRoundTrips();

    checkJsonEmissionFromRawBytes();

    checkJsonSchemaEmissionFromTheSameDeclaration();

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
