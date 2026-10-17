/**
 * @file LagCompensation.hpp
 * @brief Server-side rewind for fair hit validation (book §6.2.9).
 *
 * In a server-authoritative game a client's aim reaches the server one round trip
 * late: by the time the server processes "I shot at that spot", the targets have
 * moved. Lag compensation, formalised by Yahn Bernier for Valve, rewinds the
 * world to the instant the client actually saw — the client's render time — tests
 * the shot against where entities *were* then, and resumes the present. Without
 * it, a player must lead every moving target by their own latency.
 *
 * It keeps a short history of each entity's position (the same few dozen ticks
 * the desync detector and rollback already retain, §6.2.2, §6.4) and samples it
 * at an arbitrary past time. Built on SnapshotInterpolator, so a rewind between
 * two recorded ticks interpolates rather than snapping to the nearest.
 *
 * Positions are non-authoritative float used only to arbitrate a hit; the shot's
 * *effect* still runs through the deterministic authoritative state.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-24
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_NETCODE_LAGCOMPENSATION_HPP
#    define LPL_NET_NETCODE_LAGCOMPENSATION_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/net/netcode/Interpolation.hpp>

#    include <unordered_map>

namespace lpl::net::netcode {

/**
 * @class LagCompensator
 * @brief Per-entity position history with rewind-to-past-time queries.
 */
class LagCompensator {
public:
    /**
     * @param historyDepth Ticks of position history to retain per entity.
     */
    explicit LagCompensator(core::u32 historyDepth = 32) : _depth{historyDepth < 2 ? 2 : historyDepth} {}

    /**
     * @brief Records @p entityId at @p pos at server time @p time.
     */
    void record(core::u32 entityId, core::f64 time, const math::Vec3<float> &pos);

    /**
     * @brief Forgets an entity (e.g. it despawned).
     */
    void forget(core::u32 entityId) { _history.erase(entityId); }

    /**
     * @brief Where @p entityId was at past time @p time (interpolated).
     * @param outFound Set false if the entity has no history.
     */
    [[nodiscard]] math::Vec3<float> positionAt(core::u32 entityId, core::f64 time, bool &outFound) const;

    /**
     * @brief Whether a ray hits @p entityId as it stood at past time @p time.
     *
     * Rewinds the entity to @p time and tests the ray against a sphere of
     * @p radius around that historical position (closest approach of the ray to
     * the centre: the projection of the centre onto @p dir, clamped to the
     * forward ray). @p dir need not be normalised; a degenerate (zero) @p dir is
     * treated as a point at @p origin.
     *
     * @return true if the shot, validated in the past the client saw, connects.
     */
    [[nodiscard]] bool rayHitsAt(core::u32 entityId, core::f64 time, const math::Vec3<float> &origin,
                                 const math::Vec3<float> &dir, float radius) const;

    [[nodiscard]] core::usize trackedCount() const noexcept { return _history.size(); }

private:
    core::u32 _depth;
    std::unordered_map<core::u32, SnapshotInterpolator> _history;
};

} // namespace lpl::net::netcode

#    include "LagCompensation.inl"

#endif // LPL_NET_NETCODE_LAGCOMPENSATION_HPP
