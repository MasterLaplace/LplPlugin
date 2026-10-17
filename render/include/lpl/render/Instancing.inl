#ifndef LPL_RENDER_INSTANCING_INL
#define LPL_RENDER_INSTANCING_INL

namespace lpl::render {

inline void InstanceSet::add(math::Fixed32 x, math::Fixed32 y, math::Fixed32 z, math::Fixed32 scale)
{
    _px.push_back(x);
    _py.push_back(y);
    _pz.push_back(z);
    _scale.push_back(scale);
}

inline void InstanceSet::clear()
{
    _px.clear();
    _py.clear();
    _pz.clear();
    _scale.clear();
}

inline Frustum Frustum::fromViewProjection(const math::Mat4<core::f32> &vp) noexcept
{
    Frustum f{};
    const core::f32 r0[4] = {vp(0, 0), vp(0, 1), vp(0, 2), vp(0, 3)};
    const core::f32 r1[4] = {vp(1, 0), vp(1, 1), vp(1, 2), vp(1, 3)};
    const core::f32 r2[4] = {vp(2, 0), vp(2, 1), vp(2, 2), vp(2, 3)};
    const core::f32 r3[4] = {vp(3, 0), vp(3, 1), vp(3, 2), vp(3, 3)};

    const core::f32 src[6][4] = {
        {r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]},
        {r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]},
        {r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]},
        {r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]},
        {r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2], r3[3] + r2[3]},
        {r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]},
    };
    for (core::u32 i = 0; i < 6u; ++i)
    {
        const core::f32 len = lpl::pmr::sqrt(src[i][0] * src[i][0] + src[i][1] * src[i][1] + src[i][2] * src[i][2]);
        const core::f32 inv = (len > 0.0f) ? (1.0f / len) : 1.0f;
        f.planes[i] = Plane{src[i][0] * inv, src[i][1] * inv, src[i][2] * inv, src[i][3] * inv};
    }
    return f;
}

inline bool Frustum::containsSphere(core::f32 x, core::f32 y, core::f32 z, core::f32 radius) const noexcept
{
    for (core::u32 i = 0; i < 6u; ++i)
        if (planes[i].distance(x, y, z) < -radius)
            return false;
    return true;
}

inline void frustumCull(const InstanceSet &set, const Frustum &frustum, core::f32 localRadius,
                        pmr::vector<core::u32> &outVisible)
{
    outVisible.clear();
    const core::u32 n = set.count();
    for (core::u32 i = 0; i < n; ++i)
    {
        const core::f32 wx = set.x(i).toFloat();
        const core::f32 wy = set.y(i).toFloat();
        const core::f32 wz = set.z(i).toFloat();
        const core::f32 radius = set.scale(i).toFloat() * localRadius;
        if (frustum.containsSphere(wx, wy, wz, radius))
            outVisible.push_back(i);
    }
}

} // namespace lpl::render

#endif // LPL_RENDER_INSTANCING_INL
