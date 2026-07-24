/**
 * @file Relevancy.hpp
 * @brief Per-entity replication priority (book §6.2.7).
 *
 * Area-of-interest (§6.2.6) decides *who* is visible to a client; relevancy
 * decides, among the visible, *how much* of the bandwidth budget each entity
 * deserves this tick. Mirrors Unreal Engine's actor relevancy + priority: an
 * entity accrues priority from proximity (closer matters more) and from
 * staleness (the longer it has gone unsent, the more it is owed), and the server
 * sends the highest-priority entities until a measured byte budget is spent. The
 * staleness term is what makes the scheme starvation-free: an entity skipped for
 * being far or over budget rises every tick until it is finally sent.
 *
 * Pure and header-only: no state, no I/O, so it is trivially testable and can be
 * reused by any broadcast (AOI grid, future query-based interest, server mesh).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-24
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_RELEVANCY_RELEVANCY_HPP
#    define LPL_NET_RELEVANCY_RELEVANCY_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::net::relevancy {

/// Weight of the proximity term relative to one tick of staleness. A close entity
/// (proximity ~1) can jump ahead of another that has waited up to this many ticks;
/// beyond that, staleness wins and nothing starves.
inline constexpr float kProximityWeight = 8.0f;

/**
 * @brief Replication priority of one entity for one client, higher = send sooner.
 *
 * @param distanceSq         Squared distance from the client's avatar to the
 *                           entity, in world units². Only the ordering matters,
 *                           so the square (no sqrt) is fine and deterministic.
 * @param ticksSinceLastSent Ticks since the entity was last sent to this client.
 *                           Grows without bound while an entity waits, so a
 *                           long-starved entity eventually outranks any close one.
 * @return A monotonic score: strictly increasing in staleness, and (for equal
 *         staleness) strictly decreasing in distance.
 */
[[nodiscard]] inline float priority(float distanceSq, core::u32 ticksSinceLastSent) noexcept
{
    const float proximity = 1.0f / (1.0f + distanceSq);             // (0, 1], 1 at the avatar
    const float staleness = static_cast<float>(ticksSinceLastSent); // unbounded, anti-starvation
    return kProximityWeight * proximity + staleness;
}

/**
 * @brief Whether an entity is due for a network update this tick.
 *
 * NetUpdateFrequency, expressed as an interval: an entity that changed is always
 * due; an unchanged (dormant) one is due only on its keyframe, so a resting
 * entity consumes no traffic between keyframes (Unreal's dormancy).
 *
 * @param changed          True if any replicated field differs from the baseline.
 * @param keyframe         True on a keyframe tick (forces a full re-send).
 * @return True if the entity should be considered for sending this tick.
 */
[[nodiscard]] inline bool isDue(bool changed, bool keyframe) noexcept { return changed || keyframe; }

/**
 * @brief Network-LOD update interval for an entity, by distance (book §6.2.6).
 *
 * Interest management is not all-or-nothing: inside the radius, replication
 * fidelity should fall off with distance. This is the cadence half of that — the
 * concentric-ring model (Unreal's NetUpdateFrequency by relevancy distance):
 * entities within @p nearRadiusSq of the client update every tick, entities
 * beyond it (but still inside the interest radius) update once every
 * @p farInterval ticks. A far entity therefore costs a fraction of the bandwidth
 * of a near one, the field delta between its sparser updates simply batching its
 * accumulated change.
 *
 * @param distanceSq   Squared distance from the client's avatar, world units².
 * @param nearRadiusSq Squared radius of the full-rate near ring; <= 0 disables
 *                     LOD (every entity is near, interval 1).
 * @param farInterval  Update interval (ticks) for the far ring; clamped to >= 1.
 * @return 1 for the near ring, else @p farInterval.
 */
[[nodiscard]] inline core::u32 lodUpdateInterval(float distanceSq, float nearRadiusSq, core::u32 farInterval) noexcept
{
    if (nearRadiusSq <= 0.0f || distanceSq <= nearRadiusSq)
        return 1;
    return farInterval < 1 ? 1 : farInterval;
}

} // namespace lpl::net::relevancy

#endif // LPL_NET_RELEVANCY_RELEVANCY_HPP
