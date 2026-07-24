/**
 * @file EntityDelta.hpp
 * @brief Field-masked entity delta codec (book §6.2.5, §6.3.3).
 *
 * The area-of-interest broadcast already sends a client only its neighbours, as
 * spawn / delta / destroy. This codec makes the *delta* itself cheap: an entity
 * that stayed in range is encoded as a one-byte field mask followed by only the
 * fields that changed against what the server last sent that client. An entity
 * that did not move at all costs its id plus one empty mask byte — the "unchanged
 * field costs one bit of absence" of Quake III / DOOM III's acked-baseline model.
 *
 * The wire form for one entity is
 *   [id:u32][mask:u8]( [field:4B] for each set mask bit, in bit order )
 * so it stays byte-aligned and needs no baseline to *parse* — only to *apply*:
 * the receiver merges the present fields onto the state it already holds. A
 * keyframe (mask = all) periodically re-sends everything so a lost delta
 * self-heals within the keyframe interval; without a reliable channel that is
 * what bounds staleness.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-24
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_PROTOCOL_ENTITYDELTA_HPP
#    define LPL_NET_PROTOCOL_ENTITYDELTA_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/net/protocol/Bitstream.hpp>

namespace lpl::net::protocol {

/**
 * @struct EntitySnapshot
 * @brief The replicated, non-authoritative transform of one entity on the wire.
 *
 * Float, not Fixed32: this is the render-side representation the broadcast emits
 * at the wire boundary, never fed back into authoritative state.
 */
struct EntitySnapshot {
    core::u32 id{0};
    float px{0.0f}, py{0.0f}, pz{0.0f};
    float sx{1.0f}, sy{1.0f}, sz{1.0f};
    core::i32 hp{100};
};

/**
 * @enum EntityField
 * @brief One bit per replicated field, in wire (bit) order.
 */
enum EntityField : core::u8 {
    FieldPosX = 1u << 0,
    FieldPosY = 1u << 1,
    FieldPosZ = 1u << 2,
    FieldSizeX = 1u << 3,
    FieldSizeY = 1u << 4,
    FieldSizeZ = 1u << 5,
    FieldHp = 1u << 6,

    FieldAll = FieldPosX | FieldPosY | FieldPosZ | FieldSizeX | FieldSizeY | FieldSizeZ | FieldHp
};

/**
 * @brief Bitmask of fields that differ between @p prev and @p cur.
 * @return 0 when identical (a dormant entity), else the changed-field mask.
 */
[[nodiscard]] core::u8 computeFieldMask(const EntitySnapshot &prev, const EntitySnapshot &cur) noexcept;

/**
 * @brief Writes one entity as [id][mask]( changed fields ).
 * @param stream Destination (must be byte-aligned; it stays so).
 * @param cur    The entity to encode.
 * @param mask   Which fields to write. Callers pass @ref computeFieldMask, or
 *               @ref FieldAll for a keyframe / spawn.
 */
void writeEntityDelta(Bitstream &stream, const EntitySnapshot &cur, core::u8 mask);

/**
 * @brief Reads one entity delta, merging present fields onto @p inOut.
 *
 * Absent fields are left untouched, so @p inOut must carry the receiver's
 * current belief about the entity (the merge baseline). @p outId and @p outMask
 * report what arrived, so a caller with no prior belief still learns the id.
 *
 * @return The bit-read may fail (truncated packet); on success @p inOut holds
 *         the merged state.
 */
[[nodiscard]] core::Expected<void> readEntityDelta(Bitstream &stream, EntitySnapshot &inOut, core::u32 &outId,
                                                   core::u8 &outMask);

/**
 * @brief Writes one entity delta with its position quantized (§6.2.6).
 *
 * The precision half of network LOD: a far entity's position is not worth a full
 * float. Each present position axis is quantized to @p posBits bits over
 * [-extent, extent] (so a 2000 m world on 16 bits keeps ~3 cm), while size and hp
 * stay full. @p posBits must be a multiple of 8 so the stream stays byte-aligned
 * and mixes cleanly with the full-width size/hp writes.
 */
void writeEntityDeltaQuantized(Bitstream &stream, const EntitySnapshot &cur, core::u8 mask, float extent,
                               core::u32 posBits);

/** @brief Reads an entity written by @ref writeEntityDeltaQuantized (same params). */
[[nodiscard]] core::Expected<void> readEntityDeltaQuantized(Bitstream &stream, EntitySnapshot &inOut, core::u32 &outId,
                                                            core::u8 &outMask, float extent, core::u32 posBits);

} // namespace lpl::net::protocol

#endif // LPL_NET_PROTOCOL_ENTITYDELTA_HPP
