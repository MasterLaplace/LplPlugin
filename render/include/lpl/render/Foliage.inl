#ifndef LPL_RENDER_FOLIAGE_INL
#define LPL_RENDER_FOLIAGE_INL

namespace lpl::render {

namespace detail {

inline void cross3(core::f32 ax, core::f32 ay, core::f32 az, core::f32 bx, core::f32 by, core::f32 bz, core::f32 &ox,
                   core::f32 &oy, core::f32 &oz) noexcept
{
    ox = ay * bz - az * by;
    oy = az * bx - ax * bz;
    oz = ax * by - ay * bx;
}

} // namespace detail

inline core::u32 drawFoliage(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, const CameraBasis &basis,
                             const FoliageMesh &mesh, const FoliageInstance &instance,
                             const FoliageStyle &style) noexcept
{
    if (mesh.segments == nullptr || mesh.segmentCount == 0u)
        return 0u;

    const core::f32 cy = OrbitCamera::cosOf(instance.yaw);
    const core::f32 sy = OrbitCamera::sinOf(instance.yaw);
    core::u32 triangles = 0u;

    const auto toWorldX = [&](core::f32 mx, core::f32 mz) { return instance.x + (mx * cy + mz * sy) * instance.scale; };
    const auto toWorldZ = [&](core::f32 mx, core::f32 mz) { return instance.z + (mz * cy - mx * sy) * instance.scale; };
    const auto toWorldY = [&](core::f32 my) { return instance.y + my * instance.scale; };

    const auto fog = [&style](core::u32 colour, core::f32 distance) {
        if (style.fogDensity <= 0.0f)
            return colour;
        return applyAerialPerspective(colour, style.hazeTint, distance, style.fogDensity);
    };

    for (core::u32 i = 0u; i < mesh.segmentCount; ++i)
    {
        const FoliageSegment &segment = mesh.segments[i];
        if (segment.depth > style.maxDepth)
            continue;

        const core::f32 ax = toWorldX(segment.x0, segment.z0);
        const core::f32 ay = toWorldY(segment.y0);
        const core::f32 az = toWorldZ(segment.x0, segment.z0);
        const core::f32 bx = toWorldX(segment.x1, segment.z1);
        const core::f32 by = toWorldY(segment.y1);
        const core::f32 bz = toWorldZ(segment.x1, segment.z1);

        core::f32 wx = 0.0f;
        core::f32 wy = 0.0f;
        core::f32 wz = 0.0f;
        detail::cross3(bx - ax, by - ay, bz - az, basis.forward.x, basis.forward.y, basis.forward.z, wx, wy, wz);
        const core::f32 lengthSquared = wx * wx + wy * wy + wz * wz;
        if (lengthSquared < 1.0e-9f)
            continue;
        const core::f32 inverse = inverseSqrtNewton(lengthSquared);
        wx *= inverse;
        wy *= inverse;
        wz *= inverse;

        const core::f32 r0 = segment.radius0 * instance.scale;
        const core::f32 r1 = segment.radius1 * instance.scale;
        const core::f32 distance = approximateLength(ax - basis.eye.x, az - basis.eye.z);
        const core::f32 shaded = style.light * (1.0f - 0.06f * static_cast<core::f32>(segment.depth));
        const core::u32 colour = fog(modulate(style.bark, shaded < 0.25f ? 0.25f : shaded), distance);

        const core::f32 quad[12] = {ax - wx * r0, ay - wy * r0, az - wz * r0, ax + wx * r0, ay + wy * r0, az + wz * r0,
                                    bx + wx * r1, by + wy * r1, bz + wz * r1, bx - wx * r1, by - wy * r1, bz - wz * r1};
        triangles += fillPolygonClipped(rt, mvp, quad, 4u, colour);
    }

    const core::f32 instanceDistance = approximateLength(instance.x - basis.eye.x, instance.z - basis.eye.z);
    for (core::u32 i = 0u; instanceDistance >= style.spriteMinDistance && i < mesh.spriteCount; ++i)
    {
        const FoliageSprite &sprite = mesh.sprites[i];
        if (sprite.depth > style.maxDepth)
            continue;

        const core::f32 px = toWorldX(sprite.x, sprite.z);
        const core::f32 py = toWorldY(sprite.y);
        const core::f32 pz = toWorldZ(sprite.x, sprite.z);
        const core::f32 half = sprite.size * instance.scale;
        const core::f32 distance = approximateLength(px - basis.eye.x, pz - basis.eye.z);
        const core::f32 tone = (i & 1u) != 0u ? 1.12f : 0.86f;
        const core::u32 colour = fog(modulate(style.leaf, style.light * tone), distance);

        const core::f32 rx = basis.right.x * half;
        const core::f32 ry = basis.right.y * half;
        const core::f32 rz = basis.right.z * half;
        const core::f32 ux = basis.up.x * half;
        const core::f32 uy = basis.up.y * half;
        const core::f32 uz = basis.up.z * half;
        const core::f32 quad[12] = {px - rx - ux, py - ry - uy, pz - rz - uz, px + rx - ux, py + ry - uy, pz + rz - uz,
                                    px + rx + ux, py + ry + uy, pz + rz + uz, px - rx + ux, py - ry + uy, pz - rz + uz};
        triangles += fillPolygonClipped(rt, mvp, quad, 4u, colour);
    }

    return triangles;
}

} // namespace lpl::render

#endif // LPL_RENDER_FOLIAGE_INL
