#ifndef LPL_NET_NETCODE_LAGCOMPENSATION_INL
#define LPL_NET_NETCODE_LAGCOMPENSATION_INL

namespace lpl::net::netcode {

inline void LagCompensator::record(core::u32 entityId, core::f64 time, const math::Vec3<float> &pos)
{
    auto it = _history.find(entityId);
    if (it == _history.end())
        it = _history.emplace(entityId, SnapshotInterpolator{_depth}).first;
    it->second.addSample(time, pos);
}

inline math::Vec3<float> LagCompensator::positionAt(core::u32 entityId, core::f64 time, bool &outFound) const
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

inline bool LagCompensator::rayHitsAt(core::u32 entityId, core::f64 time, const math::Vec3<float> &origin,
                                      const math::Vec3<float> &dir, float radius) const
{
    bool found = false;
    const math::Vec3<float> center = positionAt(entityId, time, found);
    if (!found)
        return false;

    const float dd = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
    if (dd <= 0.0f)
    {
        const float ex = center.x - origin.x, ey = center.y - origin.y, ez = center.z - origin.z;
        return (ex * ex + ey * ey + ez * ez) <= radius * radius;
    }

    const float ox = center.x - origin.x, oy = center.y - origin.y, oz = center.z - origin.z;
    float t = (ox * dir.x + oy * dir.y + oz * dir.z) / dd;
    if (t < 0.0f)
        t = 0.0f;
    const float cx = origin.x + dir.x * t, cy = origin.y + dir.y * t, cz = origin.z + dir.z * t;
    const float dx = center.x - cx, dy = center.y - cy, dz = center.z - cz;
    return (dx * dx + dy * dy + dz * dz) <= radius * radius;
}

} // namespace lpl::net::netcode

#endif // LPL_NET_NETCODE_LAGCOMPENSATION_INL
