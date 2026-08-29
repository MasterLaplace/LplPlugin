/**
 * @file Component.hpp
 * @brief Component metadata types for the ECS.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_COMPONENT_HPP
#    define LPL_ECS_COMPONENT_HPP

#    include <lpl/core/Types.hpp>

#    include <cstddef>
#    include <cstdint>

namespace lpl::ecs {

/**
 * @enum ComponentId
 * @brief Compile-time enumeration of all known component types.
 *
 * Adding a new component requires appending to this enum and registering
 * its layout in the Partition's component table.
 */
enum class ComponentId : core::u16 {
    Position = 0,
    Velocity = 1,
    Rotation = 2,
    AngularVelocity = 3,
    Mass = 4,
    AABB = 5,
    Health = 6,
    NetworkSync = 7,
    InputSnapshot = 8,
    PlayerTag = 9,
    SleepState = 10,
    BciInput = 11,
    /// Heritable traits: 5x Fixed32 (speed, vision, strength, absorption, size).
    Genome = 12,
    /// What KIND of creature this is: a trophic role, and a stable identity.
    Creature = 13,
    /// Unit facing on the ground plane: 2x Fixed32 (x, z).
    ///
    /// AUTHORITATIVE, and that is the whole reason it is a component: a walker
    /// advances along its facing at a pace its genome fixes, so the facing is an
    /// input to the next position. It used to be two fields inside
    /// ecology::HerdMember, which meant the state that decides where a body goes
    /// lived outside the registry the body lives in.
    Heading = 14,
    /// Who this entity IS, in a corpus: 1x u32 identifier + 2x i32 years.
    ///
    /// @warning The component that lets a historical claim and a walking body be the
    /// same thing. Until it existed, a `history::Fact` had a subject and a body
    /// had an `EntityId` and NOTHING mapped one to the other -- so a corpus could
    /// say Herodotus was born at Halicarnassus and no entity could be him.
    ///
    /// AUTHORITATIVE, and not because of arithmetic: the identifier decides which
    /// constraints apply to this body and the years decide whether it is alive to
    /// act at all, so two targets disagreeing here would run two different lives.
    /// The years are stored rather than derived because a source DATES a birth;
    /// deriving them from a tick would make the gearing part of the biography.
    Historical = 15,

    /// Which chunk this entity's Position is relative to: 2x i32.
    ///
    /// @warning The component that lets a world be bigger than 32 kilometres. Q16.16
    /// spans +-32768 units, so a flat world position cannot address a planet --
    /// and widening the type is the wrong fix twice over: it doubles every
    /// position in the game, and a float or a wider fixed word still loses
    /// precision far from the origin, which is the "jitter at the edge of the
    /// map" every large world has.
    ///
    /// `procgen/Chunking.hpp` already argues this for terrain, in as many words:
    /// coordinates stay integral because far from the origin a float cannot
    /// distinguish adjacent cells. This is the same decision for BODIES.
    ///
    /// The convention: an entity carrying this has a Position LOCAL to its
    /// chunk; one without keeps the old meaning. Precision is then constant
    /// everywhere -- 1/65536 of a unit at the origin and a billion kilometres
    /// out -- and no product of two world-scale quantities can occur, because
    /// a local coordinate is bounded by the chunk.
    ///
    /// AUTHORITATIVE: it is half of where a body IS.
    WorldCell = 16,

    Count
};

/**
 * @enum AccessMode
 * @brief Describes how a System accesses a component (read-only or
 *        read-write).  Used by the SystemScheduler DAG builder.
 */
enum class AccessMode : core::u8 {
    ReadOnly = 0,
    ReadWrite = 1
};

/**
 * @struct ComponentAccess
 * @brief Pair of component ID + access mode used in system descriptors.
 */
struct ComponentAccess {
    ComponentId id;
    AccessMode mode;
};

/**
 * @struct ComponentLayout
 * @brief Describes the size and alignment of a single component type.
 */
struct ComponentLayout {
    ComponentId id;
    core::u32 size;
    core::u32 alignment;
};

/**
 * @brief Returns the default size and alignment for a known component type.
 *
 * This maps ComponentId to concrete data types:
 *   Position/Velocity/AABB -> Vec3<Fixed32> (12, 4) -- authoritative, deterministic
 *   AngularVelocity  -> Vec3<float> (12, 4) -- migrates with Rotation
 *   Rotation         -> Quat<float> (16, 4) -- Fixed32 quaternion is a later slice
 *   Mass             -> Fixed32 (4, 4) -- authoritative
 *   Health           -> i32 (4, 4)
 *   NetworkSync      -> u32 (4, 4)
 *   InputSnapshot    -> u32 (4, 4) -- entity-level input index
 *   PlayerTag        -> u8 (1, 1)
 *   SleepState       -> u8 (1, 1)
 *   BciInput         -> 3xfloat (12, 4) -- alpha, beta, concentration
 *   Genome           -> 5xFixed32 (20, 4) -- AUTHORITATIVE: these multiply into
 *                      speeds and damage, so a float here would desynchronise a
 *                      population after a few generations of breeding
 *   Creature         -> 2xu32 (8, 4) -- species index, stable id
 *   Heading          -> 2xFixed32 (8, 4) -- AUTHORITATIVE: a walker's next position
 *                      is its facing times its pace, so a float facing would let
 *                      two machines walk the same animal to different cells
 */
[[nodiscard]] constexpr ComponentLayout defaultLayout(ComponentId id) noexcept
{
    switch (id)
    {
    case ComponentId::Position: return {id, 12, 4};
    case ComponentId::Velocity: return {id, 12, 4};
    case ComponentId::Rotation: return {id, 16, 4};
    case ComponentId::AngularVelocity: return {id, 12, 4};
    case ComponentId::Mass: return {id, 4, 4};
    case ComponentId::AABB: return {id, 12, 4};
    case ComponentId::Health: return {id, 4, 4};
    case ComponentId::NetworkSync: return {id, 4, 4};
    case ComponentId::InputSnapshot: return {id, 4, 4};
    case ComponentId::PlayerTag: return {id, 1, 1};
    case ComponentId::SleepState: return {id, 1, 1};
    case ComponentId::BciInput: return {id, 12, 4};
    case ComponentId::Genome: return {id, 20, 4};
    case ComponentId::Creature: return {id, 8, 4};
    case ComponentId::Heading: return {id, 8, 4};
    case ComponentId::Historical: return {id, 12, 4};
    case ComponentId::WorldCell: return {id, 8, 4};
    default: return {id, 4, 4};
    }
}

} // namespace lpl::ecs

#endif // LPL_ECS_COMPONENT_HPP
