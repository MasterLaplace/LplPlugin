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
 * @warning **ComponentReflection.cpp exists because a check nobody compiles is not a check.** This
 * header asserts that every schema tiles its component exactly -- that the offsets and widths the
 * editor and the baker read a component THROUGH agree with the struct every system reads it AS. A
 * `static_assert` in a header only fires in translation units that include it, and the header was
 * included by `agent/`, `editor/` and one app: none of them is the module that owns the components.
 * Measured, before that file: an offset deliberately broken from 8 to 12 compiled clean. It defines
 * nothing and only includes this header, so the agreement is verified wherever `lpl-ecs` is built,
 * which is every target including ring 0 -- and it fails at build time rather than as a component
 * that serialises to something no system ever wrote.
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
    F32 = 0,   /**< 32-bit float (render-only / cosmetic). */
    I32,       /**< 32-bit signed integer (meta, e.g. health). */
    U32,       /**< 32-bit unsigned integer (meta, e.g. network id). */
    U16,       /**< 16-bit unsigned integer. */
    U8,        /**< 8-bit unsigned integer (tags, flags). */
    Fixed32,   /**< Q16.16 fixed-point, raw i32 (AUTHORITATIVE, deterministic). */
    Vec3F,     /**< 3x F32 (render-only). */
    Vec3Fixed, /**< 3x Fixed32 (AUTHORITATIVE). */
    QuatF      /**< 4x F32 (render-only). */
};

/**
 * @brief Byte size of a single field of the given type.
 * @param t The field type.
 * @return The byte size of the field.
 */
[[nodiscard]] constexpr core::u32 fieldSize(FieldType t) noexcept;

/**
 * @brief Byte alignment of a single field of the given type.
 * @param t The field type.
 * @return The byte alignment of the field.
 */
[[nodiscard]] constexpr core::u32 fieldAlign(FieldType t) noexcept;

/**
 * @brief True if the field carries authoritative (deterministic) state.
 * @param t The field type.
 * @return True if the field is authoritative, false otherwise.
 */
[[nodiscard]] constexpr bool isAuthoritative(FieldType t) noexcept
{
    return t == FieldType::Fixed32 || t == FieldType::Vec3Fixed;
}

/**
 * @brief Reinterpret a float's bit pattern as an i64 (for @c defaultRaw).
 * @param f The float to reinterpret.
 * @return The i64 representation of the float's bits.
 */
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
    core::u32 offset;        /**< Byte offset within the component. */
    core::i64 defaultRaw{0}; /**< Default value, interpreted per @c type. */
    bool hasBounds{false};   /**< Whether @c minRaw / @c maxRaw constrain it. */
    core::i64 minRaw{0};     /**< Inclusive minimum (same encoding as defaultRaw). */
    core::i64 maxRaw{0};     /**< Inclusive maximum. */
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
 * every registered component -- that equality is what lets the reflection table
 * replace the hand-written @c defaultLayout switch.
 */
[[nodiscard]] constexpr DerivedLayout computeLayout(const ComponentSchema &schema) noexcept;

namespace detail {

/**
 * @name Field tables
 *
 * One contiguous array per component. Offsets/types mirror the concrete data types documented in
 * Component.hpp. Position/Velocity/AABB/Mass are authoritative -> Fixed32 (raw i32 defaults).
 * @{
 */
inline constexpr FieldDesc kPositionFields[] = {
    {"value", FieldType::Vec3Fixed, 0, 0},
};
inline constexpr FieldDesc kVelocityFields[] = {
    {"value", FieldType::Vec3Fixed, 0, 0},
};
/** @brief Rotation stays float for now -- Fixed32 quaternion (CORDIC) is a later slice. */
inline constexpr FieldDesc kRotationFields[] = {
    {"value", FieldType::QuatF, 0, floatBits(0.0f)},
};
/** @brief Not yet consumed by the deterministic path; migrate with Rotation. */
inline constexpr FieldDesc kAngularVelocityFields[] = {
    {"value", FieldType::Vec3F, 0, floatBits(0.0f)},
};
/** @brief Fixed32 default 1.0 = raw 0x10000; bounds in raw Q16.16. */
inline constexpr FieldDesc kMassFields[] = {
    {"kilograms", FieldType::Fixed32, 0, 0x10000, true, 0, static_cast<core::i64>(1000) << 16},
};
/** @brief Fixed32 half-extents; default 0.5 = raw 0x8000 per lane. */
inline constexpr FieldDesc kAabbFields[] = {
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
/**
 * @brief The animal's heritable traits.
 *
 * Authoritative -- a genome multiplies into speed and damage, and breeding compounds it, so a float
 * would let two machines disagree about a population after a few generations.
 */
inline constexpr FieldDesc kGenomeFields[] = {
    {"maxSpeed",   FieldType::Fixed32, 0,  4 << 16},
    {"vision",     FieldType::Fixed32, 4,  8 << 16},
    {"strength",   FieldType::Fixed32, 8,  5 << 16},
    {"absorption", FieldType::Fixed32, 12, 1 << 16},
    {"size",       FieldType::Fixed32, 16, 1 << 16},
};
/**
 * @brief Trophic role and identity.
 *
 * Bounded because a species index outside the food web is not a rare creature, it is an
 * out-of-range read.
 */
inline constexpr FieldDesc kCreatureFields[] = {
    {"species", FieldType::U32, 0, 0, true, 0, 15},
    {"id", FieldType::U32, 4, 0},
};
/**
 * @brief Unit facing on the ground plane.
 *
 * Bounded to [-1, 1] in raw Q16.16 because a facing longer than one is not a fast animal, it is a pace
 * multiplier hidden in a direction -- and the pace belongs to the genome.
 */
inline constexpr FieldDesc kHeadingFields[] = {
    {"x", FieldType::Fixed32, 0, 1 << 16, true, -(1 << 16), 1 << 16},
    {"z", FieldType::Fixed32, 4, 0,       true, -(1 << 16), 1 << 16},
};
/**
 * @brief Who this entity is, in a corpus.
 *
 * @warning `subject` is NOT bounded: it is an identifier interned by whatever curated the corpus, so any
 * value is a legitimate one and a range check here would refuse real people. The years are unbounded
 * for the same reason in the other direction -- a corpus that reaches back to the palaeolithic has
 * legitimate values a game would call absurd.
 */
inline constexpr FieldDesc kHistoricalFields[] = {
    {"subject", FieldType::U32, 0, 0},
    {"bornDay", FieldType::I32, 4, 0},
    {"diedDay", FieldType::I32, 8, 0},
};
/**
 * @brief Which chunk a Position is relative to.
 *
 * @warning Unbounded on purpose: a chunk index is how far the world reaches, and any bound here would be
 * a hard edge on a world whose whole point is not having one.
 */
inline constexpr FieldDesc kWorldCellFields[] = {
    {"chunkX", FieldType::I32, 0, 0},
    {"chunkZ", FieldType::I32, 4, 0},
};
inline constexpr FieldDesc kBciInputFields[] = {
    {"alpha",         FieldType::F32, 0, floatBits(0.0f)},
    {"beta",          FieldType::F32, 4, floatBits(0.0f)},
    {"concentration", FieldType::F32, 8, floatBits(0.0f)},
};
/** @} */

/** @brief Indexed by static_cast<usize>(ComponentId). Order MUST match the enum. */
inline constexpr ComponentSchema kSchemas[] = {
    {ComponentId::Position,        "Position",        kPositionFields       },
    {ComponentId::Velocity,        "Velocity",        kVelocityFields       },
    {ComponentId::Rotation,        "Rotation",        kRotationFields       },
    {ComponentId::AngularVelocity, "AngularVelocity", kAngularVelocityFields},
    {ComponentId::Mass,            "Mass",            kMassFields           },
    {ComponentId::AABB,            "AABB",            kAabbFields           },
    {ComponentId::Health,          "Health",          kHealthFields         },
    {ComponentId::NetworkSync,     "NetworkSync",     kNetworkSyncFields    },
    {ComponentId::InputSnapshot,   "InputSnapshot",   kInputSnapshotFields  },
    {ComponentId::PlayerTag,       "PlayerTag",       kPlayerTagFields      },
    {ComponentId::SleepState,      "SleepState",      kSleepStateFields     },
    {ComponentId::BciInput,        "BciInput",        kBciInputFields       },
    {ComponentId::Genome,          "Genome",          kGenomeFields         },
    {ComponentId::Creature,        "Creature",        kCreatureFields       },
    {ComponentId::Heading,         "Heading",         kHeadingFields        },
    {ComponentId::Historical,      "Historical",      kHistoricalFields     },
    {ComponentId::WorldCell,       "WorldCell",       kWorldCellFields      },
};

static_assert(sizeof(kSchemas) / sizeof(kSchemas[0]) == static_cast<core::usize>(ComponentId::Count),
              "reflection table must cover every ComponentId");

/**
 * @brief Bytes a field occupies, from its type alone.
 *
 * @param type The field type.
 * @return Its width.
 */
[[nodiscard]] constexpr core::u32 fieldWidth(FieldType type) noexcept;

/**
 * @brief Whether a schema's fields tile its component without gap or overhang.
 *
 * @warning **The check nothing performed, and the one that matters most.** The reflection table is what
 * bake and the editor read a component THROUGH -- offsets, widths, bounds -- while the C++ struct is
 * what every system reads it AS. Nothing tied the two together: a field added to one and not the
 * other, or an offset off by four, produces a component that serialises to something a system
 * never wrote, silently and plausibly. The existing assertion only checked that the table has a
 * row per `ComponentId`, which is presence, not agreement.
 *
 * Fields must be declared in offset order and must exactly cover the layout `defaultLayout`
 * declares -- a gap means a byte nothing describes, an overhang means a read past the component.
 *
 * @param schema The schema.
 * @return true when the fields tile the component exactly.
 */
[[nodiscard]] constexpr bool schemaTilesComponent(const ComponentSchema &schema) noexcept;

/**
 * @brief Whether every schema tiles its component.
 *
 * @warning Components with no fields declared yet are skipped rather than failed: a schema that
 * describes nothing is an honest "not described yet", where one that describes the wrong
 * thing is a silent misread. Only the second is a defect.
 *
 * @return true when they all do.
 */
[[nodiscard]] constexpr bool everySchemaTilesItsComponent() noexcept;

} // namespace detail

/**
 * @brief Returns the reflection schema for a known component.
 * @param id The component ID.
 * @return The matching ComponentSchema.
 */
[[nodiscard]] constexpr const ComponentSchema &schemaOf(ComponentId id) noexcept
{
    return detail::kSchemas[static_cast<core::usize>(id)];
}

/**
 * @brief Returns a read-only view of every component schema.
 * @return A span containing all component schemas.
 */
[[nodiscard]] constexpr std::span<const ComponentSchema> allSchemas() noexcept
{
    return {detail::kSchemas, static_cast<core::usize>(ComponentId::Count)};
}

/**
 * @brief Resolves a component name to its id.
 * @param name The component name.
 * @return The matching ComponentId, or ComponentId::Count if unknown.
 */
[[nodiscard]] constexpr ComponentId componentIdByName(std::string_view name) noexcept;

} // namespace lpl::ecs

#    include "ComponentReflection.inl"

#endif // LPL_ECS_COMPONENTREFLECTION_HPP
