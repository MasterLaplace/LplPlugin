/**
 * @file AoiBroadcastSystem.hpp
 * @brief Area-of-interest server broadcast: each client gets only its neighbours.
 *
 * Runs in the Network phase (after physics and the buffer swap), like
 * BroadcastSystem — but instead of sending every client the whole world, it
 * sends each client only the entities inside its interest radius, and as a
 * per-session delta: EntitySpawn for entities that just entered the radius,
 * EntityDestroy for those that left, StateDelta for those that stayed. That
 * breaks the O(clients × N) broadcast wall (see the book's §2.6).
 *
 * Non-authoritative: it changes only what each client receives, never the
 * folded authoritative state. The wire snapshot is float, converted from the
 * authoritative Fixed32 transform at this boundary.
 *
 * Server-side only. Selected over BroadcastSystem when Config::interestRadius is
 * positive; a zero radius keeps the full-broadcast fallback.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-23
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_AOIBROADCASTSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_AOIBROADCASTSYSTEM_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/net/session/SessionManager.hpp>
#    include <lpl/net/transport/ITransport.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class AoiBroadcastSystem
 * @brief Per-client interest-managed state broadcast (server).
 */
class AoiBroadcastSystem final : public ecs::ISystem {
public:
    /**
     * @param sessionManager Active sessions; each session's bound entity is its
     *        centre of interest.
     * @param transport      Transport for sending (batched at end of tick).
     * @param world          Spatial index the radius query runs against.
     * @param registry       Registry the entity snapshots are read from.
     * @param interestRadius Radius around each client's avatar to send (world
     *        units). Must be positive; a zero radius means AOI is off and the
     *        Server registers BroadcastSystem instead.
     */
    AoiBroadcastSystem(net::session::SessionManager &sessionManager,
                       std::shared_ptr<net::transport::ITransport> transport, ecs::WorldPartition &world,
                       ecs::Registry &registry, math::Fixed32 interestRadius);

    /**
     * @param keyframeInterval Ticks between full re-sends of an in-range entity.
     *        Between keyframes an entity is sent as a field delta carrying only
     *        what changed against the last state the client was told (§6.2.5); the
     *        keyframe periodically re-sends everything so a lost delta self-heals.
     *        1 (or 0) means "always full" — delta compression off.
     */
    AoiBroadcastSystem(net::session::SessionManager &sessionManager,
                       std::shared_ptr<net::transport::ITransport> transport, ecs::WorldPartition &world,
                       ecs::Registry &registry, math::Fixed32 interestRadius, core::u32 keyframeInterval);

    /**
     * @param budgetBytes Per-client, per-tick byte budget for the delta stream
     *        (§6.2.7). The in-range entities that changed are ordered by a
     *        relevancy priority (proximity + staleness) and only the top ones up
     *        to this many bytes are sent; the rest age and win a later tick, so
     *        nothing starves. 0 means unlimited (send every due entity).
     */
    AoiBroadcastSystem(net::session::SessionManager &sessionManager,
                       std::shared_ptr<net::transport::ITransport> transport, ecs::WorldPartition &world,
                       ecs::Registry &registry, math::Fixed32 interestRadius, core::u32 keyframeInterval,
                       core::u32 budgetBytes);
    ~AoiBroadcastSystem() override;

    /**
     * @brief Enables network LOD: entities beyond @p nearRadius update less often.
     *
     * The cadence half of the concentric-ring model (§6.2.6). Entities within
     * @p nearRadius of a client's avatar keep the full per-tick rate; those
     * farther (but still inside the interest radius) are sent once every
     * @p farInterval ticks, their field deltas batching the change between sends.
     * A zero @p nearRadius (the default) disables LOD — every in-range entity is
     * full-rate.
     *
     * @param nearRadius  Radius of the full-rate near ring, world units.
     * @param farInterval Update interval in ticks for the far ring (clamped >= 1).
     */
    void setNetworkLod(math::Fixed32 nearRadius, core::u32 farInterval) noexcept;

    /**
     * @brief Selects the strict acked-baseline delta model (§6.2.5).
     *
     * When enabled, the delta baseline for a client advances only when that client
     * acknowledges the sequence (SnapshotAck), so an unconfirmed change keeps being
     * resent until confirmed — reliable at the cost of the ack traffic and some
     * redundancy. Disabled (the default), the baseline advances on send and a
     * keyframe self-heals a lost delta. Kept as a choice, per Config::reliableBaseline.
     */
    void setReliableBaseline(bool enabled) noexcept;

    /**
     * @brief Enables far-ring position quantization — the precision half of LOD (§6.2.6).
     *
     * With a positive @p worldExtent and network LOD on, an entity in the far ring
     * has each replicated position axis quantized to @p posBits bits over
     * [-worldExtent, worldExtent] instead of a full float, in a StateDelta marked
     * Compressed. Near-ring entities keep full precision. @p posBits must be a
     * multiple of 8 (else it falls back to 16): byte-aligned only, so quantized
     * positions mix cleanly with full-width fields. A zero @p worldExtent disables it.
     */
    void setPrecisionLod(math::Fixed32 worldExtent, core::u32 posBits) noexcept;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;

    /**
     * @brief Sends each connected client what is near its avatar, as a delta against
     *        what that client was last told about.
     *
     * One snapshot pass over the whole world; each session then filters it by its own
     * radius. This is what turns O(clients × N) serialisation into O(N) +
     * O(clients × neighbours). A session with no avatar yet, or whose avatar is not in
     * the world this tick, has nothing to centre the query on and gets nothing. The
     * neighbours are those within the radius — grid-cell granular, an interest query
     * rather than exact geometry — that have a record this tick, and they are diffed
     * against what the client already knows:
     *
     * - EntitySpawn (0x20), the entities newly inside the radius: [u16 count] then 32
     *   bytes each, the StateSnapshot layout
     *   [id:4][posX:4][posY:4][posZ:4][sizeX:4][sizeY:4][sizeZ:4][hp:4]. A spawn is a
     *   full snapshot. In the optimistic model it becomes the delta baseline
     *   immediately; in the strict model the baseline waits for the client's ack, so it
     *   is recorded as "sent at this sequence".
     * - StateDelta (0x12), the entities still inside: [u16 count] then, per entity,
     *   [id:4][mask:1] and the fields its mask names, as field-masked deltas; a far-ring
     *   group carries the Compressed flag and quantized positions.
     * - EntityDestroy (0x21), the entities that left the radius: [u16 count][u32]*.
     *   They are no longer replicated to that client, and leave its baseline.
     *
     * The neighbour set becomes what the client now knows: swapped, not copied, since
     * the scratch is cleared at the top of the next session anyway. The per-client
     * memory (known set AND delta baseline) of clients that are gone is dropped, so a
     * disconnected player's state does not linger — and a recycled playerId does not
     * inherit a stale set or baseline. Every datagram of the tick is handed to the
     * transport in one sendBatch, the same batching win broadcastState takes.
     */
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_AOIBROADCASTSYSTEM_HPP
