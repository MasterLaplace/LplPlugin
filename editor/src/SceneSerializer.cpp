/**
 * @file SceneSerializer.cpp
 * @brief Implementation of reflection-driven `.lplscene` (de)serialization.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <lpl/editor/SceneSerializer.hpp>

#include <lpl/core/Error.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/ComponentReflection.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>

#include <bit>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace lpl::editor {

using ecs::FieldDesc;
using ecs::FieldType;

namespace {

// --- Minimal, exception-free JSON value + parser --------------------------- //

struct JVal {
    enum class T { Null, Bool, Num, Str, Arr, Obj };
    T t{T::Null};
    bool b{false};
    double num{0.0};
    std::string str;
    std::vector<JVal> arr;
    std::vector<std::pair<std::string, JVal>> obj;

    [[nodiscard]] const JVal *find(std::string_view key) const
    {
        for (const auto &kv : obj)
            if (kv.first == key)
                return &kv.second;
        return nullptr;
    }
};

struct Parser {
    std::string_view s;
    std::size_t i{0};
    bool ok{true};

    void ws()
    {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
            ++i;
    }
    bool eat(char c)
    {
        ws();
        if (i < s.size() && s[i] == c)
        {
            ++i;
            return true;
        }
        return false;
    }

    JVal value()
    {
        ws();
        if (i >= s.size())
        {
            ok = false;
            return {};
        }
        const char c = s[i];
        if (c == '{')
            return object();
        if (c == '[')
            return array();
        if (c == '"')
        {
            JVal v;
            v.t = JVal::T::Str;
            v.str = string();
            return v;
        }
        if (c == 't' || c == 'f')
            return boolean();
        if (c == 'n')
        {
            i += 4; // "null"
            return {};
        }
        return number();
    }

    std::string string()
    {
        std::string out;
        if (!eat('"'))
        {
            ok = false;
            return out;
        }
        while (i < s.size() && s[i] != '"')
        {
            char c = s[i++];
            if (c == '\\' && i < s.size())
            {
                const char e = s[i++];
                switch (e)
                {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                default: out += e; break;
                }
            }
            else
            {
                out += c;
            }
        }
        if (i < s.size() && s[i] == '"')
            ++i;
        else
            ok = false;
        return out;
    }

    JVal number()
    {
        const std::size_t start = i;
        while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '-' || s[i] == '+' ||
                                s[i] == '.' || s[i] == 'e' || s[i] == 'E'))
            ++i;
        JVal v;
        v.t = JVal::T::Num;
        const std::string tok{s.substr(start, i - start)};
        v.num = std::strtod(tok.c_str(), nullptr);
        return v;
    }

    JVal boolean()
    {
        JVal v;
        v.t = JVal::T::Bool;
        if (s.compare(i, 4, "true") == 0)
        {
            v.b = true;
            i += 4;
        }
        else if (s.compare(i, 5, "false") == 0)
        {
            v.b = false;
            i += 5;
        }
        else
        {
            ok = false;
        }
        return v;
    }

    JVal array()
    {
        JVal v;
        v.t = JVal::T::Arr;
        eat('[');
        if (eat(']'))
            return v;
        while (ok)
        {
            v.arr.push_back(value());
            if (eat(','))
                continue;
            if (eat(']'))
                break;
            ok = false;
        }
        return v;
    }

    JVal object()
    {
        JVal v;
        v.t = JVal::T::Obj;
        eat('{');
        if (eat('}'))
            return v;
        while (ok)
        {
            ws();
            std::string k = string();
            if (!eat(':'))
            {
                ok = false;
                break;
            }
            v.obj.emplace_back(std::move(k), value());
            if (eat(','))
                continue;
            if (eat('}'))
                break;
            ok = false;
        }
        return v;
    }
};

// --- Byte helpers ---------------------------------------------------------- //

core::i32 getI32(const core::byte *p)
{
    core::i32 v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
float getF32(const core::byte *p)
{
    float v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
void putI32(core::byte *p, core::i32 v) { std::memcpy(p, &v, sizeof(v)); }
void putF32(core::byte *p, float v) { std::memcpy(p, &v, sizeof(v)); }

double laneOr(const JVal *o, const char *k, double d)
{
    if (o == nullptr)
        return d;
    const JVal *v = o->find(k);
    return v != nullptr ? v->num : d;
}

// --- Reflection-driven field emit / read ----------------------------------- //

void emitField(std::string &out, const FieldDesc &f, const core::byte *comp)
{
    const core::byte *p = comp + f.offset;
    out += '"';
    out += f.name;
    out += "\":";
    char buf[192];
    switch (f.type)
    {
    case FieldType::F32: std::snprintf(buf, sizeof(buf), "%g", getF32(p)); break;
    case FieldType::I32:
    case FieldType::Fixed32: std::snprintf(buf, sizeof(buf), "%d", getI32(p)); break;
    case FieldType::U32:
        std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(getI32(p)));
        break;
    case FieldType::U16: {
        std::uint16_t v;
        std::memcpy(&v, p, sizeof(v));
        std::snprintf(buf, sizeof(buf), "%u", v);
        break;
    }
    case FieldType::U8: std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(p[0])); break;
    case FieldType::Vec3F:
        std::snprintf(buf, sizeof(buf), "{\"x\":%g,\"y\":%g,\"z\":%g}", getF32(p), getF32(p + 4), getF32(p + 8));
        break;
    case FieldType::Vec3Fixed:
        std::snprintf(buf, sizeof(buf), "{\"x\":%d,\"y\":%d,\"z\":%d}", getI32(p), getI32(p + 4), getI32(p + 8));
        break;
    case FieldType::QuatF:
        std::snprintf(buf, sizeof(buf), "{\"x\":%g,\"y\":%g,\"z\":%g,\"w\":%g}", getF32(p), getF32(p + 4),
                      getF32(p + 8), getF32(p + 12));
        break;
    }
    out += buf;
}

void readField(const JVal &compObj, const FieldDesc &f, core::byte *comp)
{
    core::byte *p = comp + f.offset;
    const JVal *fv = compObj.find(f.name);
    const float defF = std::bit_cast<float>(static_cast<std::uint32_t>(static_cast<core::i32>(f.defaultRaw)));
    switch (f.type)
    {
    case FieldType::F32: putF32(p, fv != nullptr ? static_cast<float>(fv->num) : defF); break;
    case FieldType::I32:
    case FieldType::Fixed32:
        putI32(p, fv != nullptr ? static_cast<core::i32>(fv->num) : static_cast<core::i32>(f.defaultRaw));
        break;
    case FieldType::U32: {
        const std::uint32_t v =
            fv != nullptr ? static_cast<std::uint32_t>(fv->num) : static_cast<std::uint32_t>(f.defaultRaw);
        std::memcpy(p, &v, sizeof(v));
        break;
    }
    case FieldType::U16: {
        const std::uint16_t v =
            fv != nullptr ? static_cast<std::uint16_t>(fv->num) : static_cast<std::uint16_t>(f.defaultRaw);
        std::memcpy(p, &v, sizeof(v));
        break;
    }
    case FieldType::U8:
        p[0] = static_cast<core::byte>(fv != nullptr ? static_cast<unsigned>(fv->num)
                                                     : static_cast<unsigned>(f.defaultRaw));
        break;
    case FieldType::Vec3F: {
        const double d = static_cast<double>(defF);
        putF32(p, static_cast<float>(laneOr(fv, "x", d)));
        putF32(p + 4, static_cast<float>(laneOr(fv, "y", d)));
        putF32(p + 8, static_cast<float>(laneOr(fv, "z", d)));
        break;
    }
    case FieldType::Vec3Fixed: {
        const double d = static_cast<double>(static_cast<core::i32>(f.defaultRaw));
        putI32(p, static_cast<core::i32>(laneOr(fv, "x", d)));
        putI32(p + 4, static_cast<core::i32>(laneOr(fv, "y", d)));
        putI32(p + 8, static_cast<core::i32>(laneOr(fv, "z", d)));
        break;
    }
    case FieldType::QuatF: {
        const double d = static_cast<double>(defF);
        putF32(p, static_cast<float>(laneOr(fv, "x", d)));
        putF32(p + 4, static_cast<float>(laneOr(fv, "y", d)));
        putF32(p + 8, static_cast<float>(laneOr(fv, "z", d)));
        putF32(p + 12, static_cast<float>(laneOr(fv, "w", d)));
        break;
    }
    }
}

void emitComponent(std::string &out, const ecs::ComponentSchema &schema, const core::byte *comp)
{
    out += '{';
    bool first = true;
    for (const FieldDesc &f : schema.fields)
    {
        if (!first)
            out += ',';
        first = false;
        emitField(out, f, comp);
    }
    out += '}';
}

} // namespace

std::string toLplScene(const ecs::Registry &registry)
{
    std::string out = "{\"format\":\"lplscene/1\",\"entities\":[";
    bool firstEntity = true;
    for (const auto &partition : registry.partitions())
    {
        if (!partition)
            continue;
        const ecs::Archetype &arch = partition->archetype();
        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            const core::u32 count = chunk->count();
            for (core::u32 li = 0; li < count; ++li)
            {
                if (!firstEntity)
                    out += ',';
                firstEntity = false;
                out += '{';
                bool firstComp = true;
                for (const ecs::ComponentSchema &schema : ecs::allSchemas())
                {
                    if (!arch.has(schema.id))
                        continue;
                    const auto *base = static_cast<const core::byte *>(chunk->readComponent(schema.id));
                    if (base == nullptr)
                        continue;
                    if (!firstComp)
                        out += ',';
                    firstComp = false;
                    out += '"';
                    out += schema.name;
                    out += "\":";
                    const core::u32 size = ecs::computeLayout(schema).size;
                    emitComponent(out, schema, base + static_cast<std::size_t>(li) * size);
                }
                out += '}';
            }
        }
    }
    out += "]}";
    return out;
}

core::Expected<core::u32> fromLplScene(std::string_view text, ecs::Registry &registry)
{
    Parser parser{text, 0, true};
    const JVal root = parser.value();
    if (!parser.ok || root.t != JVal::T::Obj)
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"malformed .lplscene root"});

    const JVal *format = root.find("format");
    if (format == nullptr || format->t != JVal::T::Str || format->str != "lplscene/1")
        return core::makeError(core::ErrorCode::kNotSupported, lpl::pmr::string{"unsupported .lplscene format"});

    const JVal *entities = root.find("entities");
    if (entities == nullptr || entities->t != JVal::T::Arr)
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"missing entities array"});

    core::u32 created = 0;
    for (const JVal &ent : entities->arr)
    {
        if (ent.t != JVal::T::Obj)
            continue;

        // Build the archetype from the component names present on this entity.
        std::vector<ecs::ComponentId> ids;
        ids.reserve(ent.obj.size());
        for (const auto &[name, val] : ent.obj)
        {
            const ecs::ComponentId id = ecs::componentIdByName(name);
            if (id != ecs::ComponentId::Count)
                ids.push_back(id);
        }
        if (ids.empty())
            continue;

        const ecs::Archetype arch{std::span<const ecs::ComponentId>{ids}};
        auto idResult = registry.createEntity(arch);
        if (!idResult.has_value())
            return core::makeError(core::ErrorCode::kInternalError, lpl::pmr::string{"createEntity failed"});
        auto refResult = registry.resolve(idResult.value());
        if (!refResult.has_value())
            return core::makeError(core::ErrorCode::kInternalError, lpl::pmr::string{"resolve failed"});
        const ecs::EntityRef ref = refResult.value();

        ecs::Partition &partition = registry.getOrCreatePartition(arch);
        const auto chunks = partition.chunks();
        if (ref.chunkIndex >= chunks.size() || !chunks[ref.chunkIndex])
            return core::makeError(core::ErrorCode::kInternalError, lpl::pmr::string{"bad chunk index"});
        ecs::Chunk &chunk = *chunks[ref.chunkIndex];

        for (const ecs::ComponentId id : ids)
        {
            const ecs::ComponentSchema &schema = ecs::schemaOf(id);
            const JVal *compObj = ent.find(schema.name);
            if (compObj == nullptr || compObj->t != JVal::T::Obj)
                continue;
            const core::u32 size = ecs::computeLayout(schema).size;
            const std::size_t byteOffset = static_cast<std::size_t>(ref.localIndex) * size;

            // Write both buffers: the write (back) buffer the integrator uses and
            // the read (front) buffer some systems read (e.g. Mass).
            if (auto *wb = static_cast<core::byte *>(chunk.writeComponent(id)))
                for (const FieldDesc &f : schema.fields)
                    readField(*compObj, f, wb + byteOffset);
            if (auto *rb = static_cast<core::byte *>(const_cast<void *>(chunk.readComponent(id))))
                for (const FieldDesc &f : schema.fields)
                    readField(*compObj, f, rb + byteOffset);
        }
        ++created;
    }
    return created;
}

} // namespace lpl::editor
