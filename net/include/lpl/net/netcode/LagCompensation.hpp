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
    /** @param historyDepth Ticks of position history to retain per entity. */
    explicit LagCompensator(core::u32 historyDepth = 32) : _depth{historyDepth < 2 ? 2 : historyDepth} {}

    /** @brief Records @p entityId at @p pos at server time @p time. */
    void record(core::u32 entityId, core::f64 time, const math::Vec3<float> &pos)
    {
        auto it = _history.find(entityId);
        if (it == _history.end())
            it = _history.emplace(entityId, SnapshotInterpolator{_depth}).first;
        it->second.addSample(time, pos);
    }

    /** @brief Forgets an entity (e.g. it despawned). */
    void forget(core::u32 entityId) { _history.erase(entityId); }

    /**
     * @brief Where @p entityId was at past time @p time (interpolated).
     * @param outFound Set false if the entity has no history.
     */
    [[nodiscard]] math::Vec3<float> positionAt(core::u32 entityId, core::f64 time, bool &outFound) const
    {
        auto it = _history.find(entityId);
        if (it == _history.end() || it->second.count() == 0)
        {
            outFound = false;
            return {};
        }
        outFound = true;
        return it->second.sample(time);
    }

    /**
     * @brief Whether a ray hits @p entityId as it stood at past time @p time.
     *
     * Rewinds the entity to @p time and tests the ray against a sphere of
     * @p radius around that historical position (closest approach of the ray to
     * the centre). @p dir need not be normalised.
     *
     * @return true if the shot, validated in the past the client saw, connects.
     */
    [[nodiscard]] bool rayHitsAt(core::u32 entityId, core::f64 time, const math::Vec3<float> &origin,
                                 const math::Vec3<float> &dir, float radius) const
    {
        bool found = false;
        const math::Vec3<float> center = positionAt(entityId, time, found);
        if (!found)
            return false;

        const float dd = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
        if (dd <= 0.0f)
        {
            // Degenerate ray: treat as a point at the origin.
            const float ex = center.x - origin.x, ey = center.y - origin.y, ez = center.z - origin.z;
            return (ex * ex + ey * ey + ez * ez) <= radius * radius;
        }

        // t* = projection of (center - origin) onto dir, clamped to the forward ray.
        const float ox = center.x - origin.x, oy = center.y - origin.y, oz = center.z - origin.z;
        float t = (ox * dir.x + oy * dir.y + oz * dir.z) / dd;
        if (t < 0.0f)
            t = 0.0f;
        const float cx = origin.x + dir.x * t, cy = origin.y + dir.y * t, cz = origin.z + dir.z * t;
        const float dx = center.x - cx, dy = center.y - cy, dz = center.z - cz;
        return (dx * dx + dy * dy + dz * dz) <= radius * radius;
    }

    [[nodiscard]] core::usize trackedCount() const noexcept { return _history.size(); }

private:
    core::u32 _depth;
    std::unordered_map<core::u32, SnapshotInterpolator> _history;
};

} // namespace lpl::net::netcode

#endif // LPL_NET_NETCODE_LAGCOMPENSATION_HPP
