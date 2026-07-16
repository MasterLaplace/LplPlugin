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
#include <lpl/editor/Json.hpp>

#include <bit>
#include <cstdio>
#include <cstring>
#include <vector>

namespace lpl::editor {


namespace {

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

double laneOr(const detail::JVal *o, const char *k, double d)
{
    if (o == nullptr)
        return d;
    const detail::JVal *v = o->find(k);
    return v != nullptr ? v->num : d;
}

// --- Reflection-driven field emit / read ----------------------------------- //

void emitField(std::string &out, const ecs::FieldDesc &f, const core::byte *comp)
{
    const core::byte *p = comp + f.offset;
    out += '"';
    out += f.name;
    out += "\":";
    char buf[192];
    switch (f.type)
    {
    case ecs::FieldType::F32: std::snprintf(buf, sizeof(buf), "%g", getF32(p)); break;
    case ecs::FieldType::I32:
    case ecs::FieldType::Fixed32: std::snprintf(buf, sizeof(buf), "%d", getI32(p)); break;
    case ecs::FieldType::U32: std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(getI32(p))); break;
    case ecs::FieldType::U16: {
        std::uint16_t v;
        std::memcpy(&v, p, sizeof(v));
        std::snprintf(buf, sizeof(buf), "%u", v);
        break;
    }
    case ecs::FieldType::U8: std::snprintf(buf, sizeof(buf), "%u", static_cast<unsigned>(p[0])); break;
    case ecs::FieldType::Vec3F:
        std::snprintf(buf, sizeof(buf), "{\"x\":%g,\"y\":%g,\"z\":%g}", getF32(p), getF32(p + 4), getF32(p + 8));
        break;
    case ecs::FieldType::Vec3Fixed:
        std::snprintf(buf, sizeof(buf), "{\"x\":%d,\"y\":%d,\"z\":%d}", getI32(p), getI32(p + 4), getI32(p + 8));
        break;
    case ecs::FieldType::QuatF:
        std::snprintf(buf, sizeof(buf), "{\"x\":%g,\"y\":%g,\"z\":%g,\"w\":%g}", getF32(p), getF32(p + 4),
                      getF32(p + 8), getF32(p + 12));
        break;
    }
    out += buf;
}

void readField(const detail::JVal &compObj, const ecs::FieldDesc &f, core::byte *comp)
{
    core::byte *p = comp + f.offset;
    const detail::JVal *fv = compObj.find(f.name);
    const float defF = std::bit_cast<float>(static_cast<std::uint32_t>(static_cast<core::i32>(f.defaultRaw)));
    switch (f.type)
    {
    case ecs::FieldType::F32: putF32(p, fv != nullptr ? static_cast<float>(fv->num) : defF); break;
    case ecs::FieldType::I32:
    case ecs::FieldType::Fixed32:
        putI32(p, fv != nullptr ? static_cast<core::i32>(fv->num) : static_cast<core::i32>(f.defaultRaw));
        break;
    case ecs::FieldType::U32: {
        const std::uint32_t v =
            fv != nullptr ? static_cast<std::uint32_t>(fv->num) : static_cast<std::uint32_t>(f.defaultRaw);
        std::memcpy(p, &v, sizeof(v));
        break;
    }
    case ecs::FieldType::U16: {
        const std::uint16_t v =
            fv != nullptr ? static_cast<std::uint16_t>(fv->num) : static_cast<std::uint16_t>(f.defaultRaw);
        std::memcpy(p, &v, sizeof(v));
        break;
    }
    case ecs::FieldType::U8:
        p[0] = static_cast<core::byte>(fv != nullptr ? static_cast<unsigned>(fv->num) :
                                                       static_cast<unsigned>(f.defaultRaw));
        break;
    case ecs::FieldType::Vec3F: {
        const double d = static_cast<double>(defF);
        putF32(p, static_cast<float>(laneOr(fv, "x", d)));
        putF32(p + 4, static_cast<float>(laneOr(fv, "y", d)));
        putF32(p + 8, static_cast<float>(laneOr(fv, "z", d)));
        break;
    }
    case ecs::FieldType::Vec3Fixed: {
        const double d = static_cast<double>(static_cast<core::i32>(f.defaultRaw));
        putI32(p, static_cast<core::i32>(laneOr(fv, "x", d)));
        putI32(p + 4, static_cast<core::i32>(laneOr(fv, "y", d)));
        putI32(p + 8, static_cast<core::i32>(laneOr(fv, "z", d)));
        break;
    }
    case ecs::FieldType::QuatF: {
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
    for (const ecs::FieldDesc &f : schema.fields)
    {
        if (!first)
            out += ',';
        first = false;
        emitField(out, f, comp);
    }
    out += '}';
}

// --- Template (prefab) resolution ------------------------------------------ //

// Overlays entity component @p src onto @p dst at the field level: fields present
// in @p src replace those in @p dst, other fields keep the template's value.
void overlayComponent(detail::JVal &dst, const detail::JVal &src)
{
    if (dst.t != detail::JVal::T::Obj || src.t != detail::JVal::T::Obj)
    {
        dst = src;
        return;
    }
    for (const auto &field : src.obj)
    {
        detail::JVal *existing = nullptr;
        for (auto &d : dst.obj)
            if (d.first == field.first)
            {
                existing = &d.second;
                break;
            }
        if (existing == nullptr)
            dst.obj.push_back(field);
        else
            *existing = field.second;
    }
}

// Produces the effective component map for @p entity: if it carries a "$use"
// reference, the named template's components are laid down first (deep-merged),
// then the entity's own components override them field-by-field. This is the
// Flakkari prefab pattern — a template graph flattened at instantiation time.
detail::JVal resolveEntity(const detail::JVal &entity, const detail::JVal *templates)
{
    detail::JVal eff;
    eff.t = detail::JVal::T::Obj;

    if (templates != nullptr)
    {
        const detail::JVal *use = entity.find("$use");
        if (use != nullptr && use->t == detail::JVal::T::Str)
        {
            const detail::JVal *tmpl = templates->find(use->str);
            if (tmpl != nullptr && tmpl->t == detail::JVal::T::Obj)
                for (const auto &comp : tmpl->obj)
                {
                    if (comp.first == "$use")
                        continue;
                    eff.obj.push_back(comp);
                }
        }
    }

    for (const auto &comp : entity.obj)
    {
        if (comp.first == "$use")
            continue;
        detail::JVal *existing = nullptr;
        for (auto &e : eff.obj)
            if (e.first == comp.first)
            {
                existing = &e.second;
                break;
            }
        if (existing == nullptr)
            eff.obj.push_back(comp);
        else
            overlayComponent(*existing, comp.second);
    }
    return eff;
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
    detail::Parser parser{text, 0, true};
    const detail::JVal root = parser.value();
    if (!parser.ok || root.t != detail::JVal::T::Obj)
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"malformed .lplscene root"});

    const detail::JVal *format = root.find("format");
    if (format == nullptr || format->t != detail::JVal::T::Str || format->str != "lplscene/1")
        return core::makeError(core::ErrorCode::kNotSupported, lpl::pmr::string{"unsupported .lplscene format"});

    const detail::JVal *entities = root.find("entities");
    if (entities == nullptr || entities->t != detail::JVal::T::Arr)
        return core::makeError(core::ErrorCode::kDeserializationFailed, lpl::pmr::string{"missing entities array"});

    // Optional prefab graph: entities may reference these by "$use" and override.
    const detail::JVal *templates = root.find("templates");
    if (templates != nullptr && templates->t != detail::JVal::T::Obj)
        templates = nullptr;

    core::u32 created = 0;
    for (const detail::JVal &rawEnt : entities->arr)
    {
        if (rawEnt.t != detail::JVal::T::Obj)
            continue;

        // Flatten any "$use" template reference into a concrete component map.
        const detail::JVal ent = resolveEntity(rawEnt, templates);

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
            const detail::JVal *compObj = ent.find(schema.name);
            if (compObj == nullptr || compObj->t != detail::JVal::T::Obj)
                continue;
            const core::u32 size = ecs::computeLayout(schema).size;
            const std::size_t byteOffset = static_cast<std::size_t>(ref.localIndex) * size;

            // Write both buffers: the write (back) buffer the integrator uses and
            // the read (front) buffer some systems read (e.g. Mass).
            if (auto *wb = static_cast<core::byte *>(chunk.writeComponent(id)))
                for (const ecs::FieldDesc &f : schema.fields)
                    readField(*compObj, f, wb + byteOffset);
            if (auto *rb = static_cast<core::byte *>(const_cast<void *>(chunk.readComponent(id))))
                for (const ecs::FieldDesc &f : schema.fields)
                    readField(*compObj, f, rb + byteOffset);
        }
        ++created;
    }
    return created;
}

} // namespace lpl::editor
