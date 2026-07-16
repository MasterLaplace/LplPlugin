/**
 * @file ComponentReflection.hpp
 * @brief Single-source-of-truth reflection metadata for ECS components.
 *
 * One @c constexpr declaration per component drives every consumer: the ECS
 * layout table, named (de)serialization, the JSON-Schema fed to validation and
 * to the AI tool grammar (GBNF), and the editor inspector. This eliminates the
 * duplicated hard-coded component knowledge that plagued every prior engine
 * (Flakkari string dispatch, the R-Type editor @c getDefaultComponentValue
 * if/else, @c drawComponent<T>) and now @ref defaultLayout here.
 *
 * The determination class is carried by the field @b type: @c Fixed32 /
 * @c Vec3Fixed fields are authoritative (raw-int, bit-identical kernel<->oracle)
 * while @c F32 / @c Vec3F / @c QuatF are render-only. Migrating a component to
 * determinism is a type change; layout/JSON/validation follow automatically.
 *
 * Freestanding-safe (no exceptions, no heap): usable from the kernel parity
 * path. Host-only derivations (JSON emitters) live outside this header.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_COMPONENTREFLECTION_HPP
#    define LPL_ECS_COMPONENTREFLECTION_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Component.hpp>

#    include <bit>
#    include <span>
#    include <string_view>

namespace lpl::ecs {

/**
 * @enum FieldType
 * @brief Primitive and composite field types a component can be built from.
 *
 * The type also fixes the determination class: @c Fixed32 and @c Vec3Fixed are
 * authoritative and serialize as raw integers; the float variants are
 * render-only and must never feed authoritative state.
 */
enum class FieldType : core::u8 {
    F32 = 0,   ///< 32-bit float (render-only / cosmetic).
    I32,       ///< 32-bit signed integer (meta, e.g. health).
    U32,       ///< 32-bit unsigned integer (meta, e.g. network id).
    U16,       ///< 16-bit unsigned integer.
    U8,        ///< 8-bit unsigned integer (tags, flags).
    Fixed32,   ///< Q16.16 fixed-point, raw i32 (AUTHORITATIVE, deterministic).
    Vec3F,     ///< 3x F32 (render-only).
    Vec3Fixed, ///< 3x Fixed32 (AUTHORITATIVE).
    QuatF      ///< 4x F32 (render-only).
};

/** @brief Byte size of a single field of the given type. */
[[nodiscard]] constexpr core::u32 fieldSize(FieldType t) noexcept
{
    switch (t)
    {
    case FieldType::F32:
    case FieldType::I32:
    case FieldType::U32:
    case FieldType::Fixed32: return 4;
    case FieldType::U16: return 2;
    case FieldType::U8: return 1;
    case FieldType::Vec3F:
    case FieldType::Vec3Fixed: return 12;
    case FieldType::QuatF: return 16;
    }
    return 0;
}

/** @brief Byte alignment of a single field of the given type. */
[[nodiscard]] constexpr core::u32 fieldAlign(FieldType t) noexcept
{
    switch (t)
    {
    case FieldType::U8: return 1;
    case FieldType::U16: return 2;
    default: return 4;
    }
}

/** @brief True if the field carries authoritative (deterministic) state. */
[[nodiscard]] constexpr bool isAuthoritative(FieldType t) noexcept
{
    return t == FieldType::Fixed32 || t == FieldType::Vec3Fixed;
}

/** @brief Reinterpret a float's bit pattern as an i64 (for @c defaultRaw). */
[[nodiscard]] constexpr core::i64 floatBits(float f) noexcept
{
    return static_cast<core::i64>(static_cast<core::i32>(std::bit_cast<core::u32>(f)));
}

/**
 * @struct FieldDesc
 * @brief One named field within a component.
 *
 * @c defaultRaw is interpreted per @c type: Fixed32 raw value, integer value,
 * or the bit pattern of a float (see @ref floatBits). For composites the
 * default is applied to every lane (slice-1 simplification).
 */
struct FieldDesc {
    std::string_view name;
    FieldType type;
    core::u32 offset;         ///< Byte offset within the component.
    core::i64 defaultRaw{0};  ///< Default value, interpreted per @c type.
    bool hasBounds{false};    ///< Whether @c minRaw / @c maxRaw constrain it.
    core::i64 minRaw{0};      ///< Inclusive minimum (same encoding as defaultRaw).
    core::i64 maxRaw{0};      ///< Inclusive maximum.
};

/**
 * @struct ComponentSchema
 * @brief The single declaration of a component: its id, name and fields.
 */
struct ComponentSchema {
    ComponentId id;
    std::string_view name;
    std::span<const FieldDesc> fields;
};

/**
 * @struct DerivedLayout
 * @brief Size and alignment computed from a schema's fields.
 */
struct DerivedLayout {
    core::u32 size;
    core::u32 alignment;
};

/**
 * @brief Computes the byte size and alignment implied by a schema's fields.
 *
 * Size is the highest @c (offset + fieldSize) rounded up to the alignment;
 * alignment is the maximum field alignment. Must match @ref defaultLayout for
 * every registered component — that equality is what lets the reflection table
 * replace the hand-written @c defaultLayout switch.
 */
[[nodiscard]] constexpr DerivedLayout computeLayout(const ComponentSchema &schema) noexcept
{
    core::u32 size = 0;
    core::u32 align = 1;
    for (const FieldDesc &f : schema.fields)
    {
        const core::u32 end = f.offset + fieldSize(f.type);
        if (end > size)
            size = end;
        const core::u32 a = fieldAlign(f.type);
        if (a > align)
            align = a;
    }
    if (align != 0 && (size % align) != 0)
        size += align - (size % align);
    return {size, align};
}

namespace detail {

// --- Field tables (one contiguous array per component) -------------------- //
// Offsets/types mirror the concrete data types documented in Component.hpp.

// Position/Velocity/AABB/Mass are authoritative → Fixed32 (raw i32 defaults).
inline constexpr FieldDesc kPositionFields[] = {
    {"value", FieldType::Vec3Fixed, 0, 0},
};
inline constexpr FieldDesc kVelocityFields[] = {
    {"value", FieldType::Vec3Fixed, 0, 0},
};
inline constexpr FieldDesc kRotationFields[] = {
    // Rotation stays float for now — Fixed32 quaternion (CORDIC) is a later slice.
    {"value", FieldType::QuatF, 0, floatBits(0.0f)},
};
inline constexpr FieldDesc kAngularVelocityFields[] = {
    // Not yet consumed by the deterministic path; migrate with Rotation.
    {"value", FieldType::Vec3F, 0, floatBits(0.0f)},
};
inline constexpr FieldDesc kMassFields[] = {
    // Fixed32 default 1.0 = raw 0x10000; bounds in raw Q16.16.
    {"kilograms", FieldType::Fixed32, 0, 0x10000, true, 0, static_cast<core::i64>(1000) << 16},
};
inline constexpr FieldDesc kAabbFields[] = {
    // Fixed32 half-extents; default 0.5 = raw 0x8000 per lane.
    {"halfExtents", FieldType::Vec3Fixed, 0, 0x8000},
};
inline constexpr FieldDesc kHealthFields[] = {
    {"points", FieldType::I32, 0, 100, true, 0, 1000000},
};
inline constexpr FieldDesc kNetworkSyncFields[] = {
    {"id", FieldType::U32, 0, 0},
};
inline constexpr FieldDesc kInputSnapshotFields[] = {
    {"index", FieldType::U32, 0, 0},
};
inline constexpr FieldDesc kPlayerTagFields[] = {
    {"team", FieldType::U8, 0, 0, true, 0, 255},
};
inline constexpr FieldDesc kSleepStateFields[] = {
    {"asleep", FieldType::U8, 0, 0, true, 0, 1},
};
inline constexpr FieldDesc kBciInputFields[] = {
    {"alpha", FieldType::F32, 0, floatBits(0.0f)},
    {"beta", FieldType::F32, 4, floatBits(0.0f)},
    {"concentration", FieldType::F32, 8, floatBits(0.0f)},
};

// Indexed by static_cast<usize>(ComponentId). Order MUST match the enum.
inline constexpr ComponentSchema kSchemas[] = {
    {ComponentId::Position, "Position", kPositionFields},
    {ComponentId::Velocity, "Velocity", kVelocityFields},
    {ComponentId::Rotation, "Rotation", kRotationFields},
    {ComponentId::AngularVelocity, "AngularVelocity", kAngularVelocityFields},
    {ComponentId::Mass, "Mass", kMassFields},
    {ComponentId::AABB, "AABB", kAabbFields},
    {ComponentId::Health, "Health", kHealthFields},
    {ComponentId::NetworkSync, "NetworkSync", kNetworkSyncFields},
    {ComponentId::InputSnapshot, "InputSnapshot", kInputSnapshotFields},
    {ComponentId::PlayerTag, "PlayerTag", kPlayerTagFields},
    {ComponentId::SleepState, "SleepState", kSleepStateFields},
    {ComponentId::BciInput, "BciInput", kBciInputFields},
};

static_assert(sizeof(kSchemas) / sizeof(kSchemas[0]) == static_cast<core::usize>(ComponentId::Count),
              "reflection table must cover every ComponentId");

} // namespace detail

/** @brief Returns the reflection schema for a known component. */
[[nodiscard]] constexpr const ComponentSchema &schemaOf(ComponentId id) noexcept
{
    return detail::kSchemas[static_cast<core::usize>(id)];
}

/** @brief Read-only view of every component schema. */
[[nodiscard]] constexpr std::span<const ComponentSchema> allSchemas() noexcept
{
    return {detail::kSchemas, static_cast<core::usize>(ComponentId::Count)};
}

/**
 * @brief Resolves a component name to its id.
 * @return The matching ComponentId, or ComponentId::Count if unknown.
 */
[[nodiscard]] constexpr ComponentId componentIdByName(std::string_view name) noexcept
{
    for (const ComponentSchema &s : detail::kSchemas)
        if (s.name == name)
            return s.id;
    return ComponentId::Count;
}

} // namespace lpl::ecs

#endif // LPL_ECS_COMPONENTREFLECTION_HPP
