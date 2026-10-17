#ifndef LPL_NET_NETCODE_INTERPOLATION_INL
#define LPL_NET_NETCODE_INTERPOLATION_INL

namespace lpl::net::netcode {

inline void SnapshotInterpolator::addSample(core::f64 time, const math::Vec3<float> &pos, const math::Vec3<float> &vel)
{
    if (!_samples.empty() && time <= _samples.back().time)
        return;
    _samples.push_back(Sample{time, pos, vel});
    if (static_cast<core::u32>(_samples.size()) > _capacity)
        _samples.erase(_samples.begin());
}

inline math::Vec3<float> SnapshotInterpolator::sample(core::f64 renderTime) const
{
    if (_samples.empty())
        return {};
    if (_samples.size() == 1 || renderTime <= _samples.front().time)
        return frontOr(renderTime);

    const Sample &newest = _samples.back();
    if (renderTime >= newest.time)
    {
        const auto dt = static_cast<float>(renderTime - newest.time);
        return math::Vec3<float>{newest.pos.x + newest.vel.x * dt, newest.pos.y + newest.vel.y * dt,
                                 newest.pos.z + newest.vel.z * dt};
    }

    for (core::usize i = 1; i < _samples.size(); ++i)
    {
        const Sample &s1 = _samples[i];
        if (renderTime <= s1.time)
        {
            const Sample &s0 = _samples[i - 1];
            const core::f64 span = s1.time - s0.time;
            const float t = (span > 0.0) ? static_cast<float>((renderTime - s0.time) / span) : 0.0f;
            return math::Vec3<float>{s0.pos.x + (s1.pos.x - s0.pos.x) * t, s0.pos.y + (s1.pos.y - s0.pos.y) * t,
                                     s0.pos.z + (s1.pos.z - s0.pos.z) * t};
        }
    }
    return newest.pos;
}

} // namespace lpl::net::netcode

#endif // LPL_NET_NETCODE_INTERPOLATION_INL
