#ifndef LPL_RENDER_SKY_DOME_INL
#define LPL_RENDER_SKY_DOME_INL

namespace lpl::render {

inline void drawSky(const RenderTarget &rt, const CameraBasis &basis, const SunState &sun, const SkyParams &params,
                    core::f32 tanHalfFov, core::u32 block) noexcept
{
    const core::f32 aspect = static_cast<core::f32>(rt.width) / static_cast<core::f32>(rt.height);
    const core::u32 step = block == 0u ? 1u : block;

    for (core::u32 y = 0u; y < rt.height; y += step)
    {
        const core::f32 ndcY = 1.0f - 2.0f * (static_cast<core::f32>(y) + 0.5f) / static_cast<core::f32>(rt.height);
        for (core::u32 x = 0u; x < rt.width; x += step)
        {
            const core::f32 ndcX = 2.0f * (static_cast<core::f32>(x) + 0.5f) / static_cast<core::f32>(rt.width) - 1.0f;
            const core::f32 h = ndcX * tanHalfFov * aspect;
            const core::f32 v = ndcY * tanHalfFov;

            core::f32 dx = basis.forward.x + basis.right.x * h + basis.up.x * v;
            core::f32 dy = basis.forward.y + basis.right.y * h + basis.up.y * v;
            core::f32 dz = basis.forward.z + basis.right.z * h + basis.up.z * v;
            const core::f32 inverse = inverseSqrtNewton(dx * dx + dy * dy + dz * dz);
            dx *= inverse;
            dy *= inverse;
            dz *= inverse;

            const core::u32 colour = skyColour(dx, dy, dz, sun, params);

            const core::u32 maxY = (y + step) < rt.height ? (y + step) : rt.height;
            const core::u32 maxX = (x + step) < rt.width ? (x + step) : rt.width;
            for (core::u32 by = y; by < maxY; ++by)
                for (core::u32 bx = x; bx < maxX; ++bx)
                {
                    rt.color[by * rt.width + bx] = colour;
                    rt.depth[by * rt.width + bx] = 1.0e30f;
                }
        }
    }
}

} // namespace lpl::render

#endif // LPL_RENDER_SKY_DOME_INL
