/**
 * @file EditorSession.cpp
 * @brief Implementation of the headless editing model.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <lpl/editor/EditorSession.hpp>

#include <lpl/ecs/ComponentReflection.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/editor/CommandProcessor.hpp>
#include <lpl/editor/SceneSerializer.hpp>
#include <lpl/math/FixedPoint.hpp>

#include <cstring>
#include <vector>

namespace lpl::editor {

namespace {

// Locates the ecs::FieldDesc named @p field within component @p id, or nullptr.
const ecs::FieldDesc *findField(ecs::ComponentId id, std::string_view field)
{
    const ecs::ComponentSchema &schema = ecs::schemaOf(id);
    for (const ecs::FieldDesc &f : schema.fields)
        if (field == f.name)
            return &f;
    return nullptr;
}

// Reads a single 4-byte-or-less lane at @p p as a human-unit double.
double readLane(const core::byte *p, ecs::FieldType type)
{
    switch (type)
    {
    case ecs::FieldType::F32:
    case ecs::FieldType::Vec3F:
    case ecs::FieldType::QuatF: {
        float v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<double>(v);
    }
    case ecs::FieldType::Fixed32:
    case ecs::FieldType::Vec3Fixed: {
        core::i32 raw;
        std::memcpy(&raw, p, sizeof(raw));
        return static_cast<double>(math::Fixed32::fromRaw(raw).toFloat());
    }
    case ecs::FieldType::I32: {
        core::i32 v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<double>(v);
    }
    case ecs::FieldType::U32: {
        core::u32 v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<double>(v);
    }
    case ecs::FieldType::U16: {
        core::u16 v;
        std::memcpy(&v, p, sizeof(v));
        return static_cast<double>(v);
    }
    case ecs::FieldType::U8: return static_cast<double>(static_cast<unsigned char>(p[0]));
    }
    return 0.0;
}

// Writes a human-unit double into a single lane at @p p (math::Fixed32 is quantised).
void writeLane(core::byte *p, ecs::FieldType type, double value)
{
    switch (type)
    {
    case ecs::FieldType::F32:
    case ecs::FieldType::Vec3F:
    case ecs::FieldType::QuatF: {
        const float v = static_cast<float>(value);
        std::memcpy(p, &v, sizeof(v));
        break;
    }
    case ecs::FieldType::Fixed32:
    case ecs::FieldType::Vec3Fixed: {
        const core::i32 raw = math::Fixed32::fromFloat(static_cast<float>(value)).raw();
        std::memcpy(p, &raw, sizeof(raw));
        break;
    }
    case ecs::FieldType::I32: {
        const core::i32 v = static_cast<core::i32>(value);
        std::memcpy(p, &v, sizeof(v));
        break;
    }
    case ecs::FieldType::U32: {
        const core::u32 v = static_cast<core::u32>(value);
        std::memcpy(p, &v, sizeof(v));
        break;
    }
    case ecs::FieldType::U16: {
        const core::u16 v = static_cast<core::u16>(value);
        std::memcpy(p, &v, sizeof(v));
        break;
    }
    case ecs::FieldType::U8: p[0] = static_cast<core::byte>(static_cast<unsigned>(value)); break;
    }
}

// Lane byte stride within a composite (0 for scalars).
core::u32 laneStride(ecs::FieldType type)
{
    return (type == ecs::FieldType::Vec3F || type == ecs::FieldType::Vec3Fixed || type == ecs::FieldType::QuatF) ? 4u :
                                                                                                                   0u;
}

} // namespace

core::u32 EditorSession::laneCount(ecs::FieldType type) noexcept
{
    switch (type)
    {
    case ecs::FieldType::Vec3F:
    case ecs::FieldType::Vec3Fixed: return 3u;
    case ecs::FieldType::QuatF: return 4u;
    default: return 1u;
    }
}

core::u32 EditorSession::entityCount() const { return editor::entityCount(registry_); }

EntityLocation EditorSession::locate(core::u32 flatIndex) const
{
    core::u32 seen = 0;
    for (const auto &part : registry_.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            if (!chunk)
                continue;
            const core::u32 n = chunk->count();
            if (flatIndex < seen + n)
                return EntityLocation{chunk.get(), flatIndex - seen};
            seen += n;
        }
    }
    return EntityLocation{};
}

bool EditorSession::getField(core::u32 flatIndex, ecs::ComponentId id, std::string_view field, core::u32 lane,
                             double &out) const
{
    const EntityLocation loc = locate(flatIndex);
    if (!loc.valid() || !loc.chunk->archetype().has(id))
        return false;
    const ecs::FieldDesc *f = findField(id, field);
    if (f == nullptr || lane >= laneCount(f->type))
        return false;

    const auto *base = static_cast<const core::byte *>(loc.chunk->readComponent(id));
    if (base == nullptr)
        return false;
    const core::u32 size = ecs::computeLayout(ecs::schemaOf(id)).size;
    const core::byte *comp = base + static_cast<std::size_t>(loc.localIndex) * size + f->offset;
    out = readLane(comp + static_cast<std::size_t>(lane) * laneStride(f->type), f->type);
    return true;
}

bool EditorSession::setField(core::u32 flatIndex, ecs::ComponentId id, std::string_view field, core::u32 lane,
                             double value)
{
    const EntityLocation loc = locate(flatIndex);
    if (!loc.valid() || !loc.chunk->archetype().has(id))
        return false;
    const ecs::FieldDesc *f = findField(id, field);
    if (f == nullptr || lane >= laneCount(f->type))
        return false;

    const core::u32 size = ecs::computeLayout(ecs::schemaOf(id)).size;
    const std::size_t byteOffset = static_cast<std::size_t>(loc.localIndex) * size + f->offset +
                                   static_cast<std::size_t>(lane) * laneStride(f->type);

    // Write both buffers so integrators (back) and readers (front) agree.
    bool wrote = false;
    if (auto *wb = static_cast<core::byte *>(loc.chunk->writeComponent(id)))
    {
        writeLane(wb + byteOffset, f->type, value);
        wrote = true;
    }
    if (auto *rb = static_cast<core::byte *>(const_cast<void *>(loc.chunk->readComponent(id))))
    {
        writeLane(rb + byteOffset, f->type, value);
        wrote = true;
    }
    return wrote;
}

void EditorSession::clear()
{
    // Collect every live id first: destroying while iterating chunks mutates them.
    std::vector<ecs::EntityId> ids;
    for (const auto &part : registry_.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
            if (chunk)
                for (const ecs::EntityId id : chunk->entities())
                    ids.push_back(id);
    }
    for (const ecs::EntityId id : ids)
        (void) registry_.destroyEntity(id);
    clearSelection();
}

core::Expected<std::string> EditorSession::command(std::string_view json)
{
    CommandProcessor processor(registry_);
    return processor.execute(json);
}

std::string EditorSession::save() const { return toLplScene(registry_); }

core::Expected<core::u32> EditorSession::load(std::string_view text) { return fromLplScene(text, registry_); }

} // namespace lpl::editor
