/**
 * @file Interpolation.hpp
 * @brief Snapshot interpolation + dead reckoning for remote entities (§6.2.4).
 *
 * A client predicts only its own avatar (it knows its own inputs). Every other
 * entity it learns about through periodic authoritative snapshots, which arrive
 * spaced out and sometimes out of order or missing. Rendering them at the raw
 * snapshot positions would stutter; this smooths them two ways, as formalised by
 * Yahn Bernier for Valve's Source engine:
 *
 *   - Interpolation: the client renders the recent PAST, at `now - delay`, and
 *     linearly interpolates between the two snapshots that bracket that time.
 *     A late or dropped packet is hidden as long as the delay covers the gap.
 *   - Dead reckoning: when the render time runs past the newest snapshot (the
 *     client got ahead, or a packet is overdue), it extrapolates along the last
 *     known velocity instead of freezing, then eases back when the next snapshot
 *     lands.
 *
 * Render-side and non-authoritative: everything here is float and never flows
 * back into the deterministic Fixed32 state. Header-only so the client can use it
 * without a translation unit.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-24
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_NETCODE_INTERPOLATION_HPP
#    define LPL_NET_NETCODE_INTERPOLATION_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/Vec3.hpp>

#    include <vector>

namespace lpl::net::netcode {

/**
 * @class SnapshotInterpolator
 * @brief Per-entity ring of timestamped snapshots with interpolated sampling.
 *
 * Feed it authoritative snapshots as they arrive (@ref addSample), then ask for
 * the render position at a chosen time (@ref sample), usually `now - delay`.
 */
class SnapshotInterpolator {
public:
    /** @brief One timestamped authoritative sample of a remote entity. */
    struct Sample {
        core::f64 time{0.0};
        math::Vec3<float> pos{};
        math::Vec3<float> vel{}; /**< For dead reckoning; zero if the wire omits velocity. */
    };

    /**
     * @param capacity Max snapshots retained (older ones are dropped).
     */
    explicit SnapshotInterpolator(core::u32 capacity = 32) : _capacity{capacity < 2 ? 2 : capacity} {}

    /**
     * @brief Records a snapshot. Samples are expected roughly in time order; an
     *        out-of-order (older than newest) or duplicate sample is dropped, the
     *        way a late UDP packet for a tick already superseded would be: the
     *        newer snapshot already stands.
     */
    void addSample(core::f64 time, const math::Vec3<float> &pos, const math::Vec3<float> &vel = {});

    /**
     * @brief Render position at @p renderTime.
     *
     * Before the oldest sample it clamps to the oldest; between two samples it
     * lerps; past the newest it dead-reckons along the last velocity.
     */
    [[nodiscard]] math::Vec3<float> sample(core::f64 renderTime) const;

    /**
     * @brief True when @p renderTime is past the newest sample (dead reckoning).
     */
    [[nodiscard]] bool extrapolating(core::f64 renderTime) const noexcept
    {
        return !_samples.empty() && renderTime > _samples.back().time;
    }

    [[nodiscard]] core::usize count() const noexcept { return _samples.size(); }
    [[nodiscard]] core::f64 newestTime() const noexcept { return _samples.empty() ? 0.0 : _samples.back().time; }
    void clear() noexcept { _samples.clear(); }

private:
    [[nodiscard]] math::Vec3<float> frontOr(core::f64) const { return _samples.front().pos; }

    core::u32 _capacity;
    std::vector<Sample> _samples; /**< Oldest at front, newest at back. */
};

} // namespace lpl::net::netcode

#    include "Interpolation.inl"

#endif // LPL_NET_NETCODE_INTERPOLATION_HPP
