#ifndef LPL_RENDER_BOX_INL
#define LPL_RENDER_BOX_INL

namespace lpl::render {

template <typename Emit>
void forEachBoxFace(core::f32 x0, core::f32 y0, core::f32 z0, core::f32 x1, core::f32 y1, core::f32 z1, Emit &&emit)
{
    const core::f32 faces[6][12] = {
        {x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1},
        {x0, y0, z1, x1, y0, z1, x1, y0, z0, x0, y0, z0},
        {x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1},
        {x1, y0, z0, x0, y0, z0, x0, y1, z0, x1, y1, z0},
        {x1, y0, z1, x1, y0, z0, x1, y1, z0, x1, y1, z1},
        {x0, y0, z0, x0, y0, z1, x0, y1, z1, x0, y1, z0},
    };
    const core::f32 normals[6][3] = {
        {0.0f,  1.0f,  0.0f },
        {0.0f,  -1.0f, 0.0f },
        {0.0f,  0.0f,  1.0f },
        {0.0f,  0.0f,  -1.0f},
        {1.0f,  0.0f,  0.0f },
        {-1.0f, 0.0f,  0.0f },
    };
    for (core::u32 face = 0u; face < 6u; ++face)
        emit(faces[face], normals[face][0], normals[face][1], normals[face][2]);
}

inline core::u32 drawBox(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, core::f32 x0, core::f32 y0,
                         core::f32 z0, core::f32 x1, core::f32 y1, core::f32 z1, core::u32 colour,
                         const math::Vec3<core::f32> &sun, core::f32 ambient)
{
    core::u32 triangles = 0u;
    forEachBoxFace(x0, y0, z0, x1, y1, z1, [&](const core::f32 *quad, core::f32 nx, core::f32 ny, core::f32 nz) {
        const core::f32 lambert = nx * sun.x + ny * sun.y + nz * sun.z;
        const core::f32 lit = ambient + (1.0f - ambient) * (lambert > 0.0f ? lambert : 0.0f);
        triangles += fillPolygonClipped(rt, mvp, quad, 4u, modulate(colour, lit));
    });
    return triangles;
}

} // namespace lpl::render

#endif // LPL_RENDER_BOX_INL
