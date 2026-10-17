#ifndef LPL_RENDER_HEIGHTFIELD_PATCH_INL
#define LPL_RENDER_HEIGHTFIELD_PATCH_INL

namespace lpl::render {

template <typename HeightAt, typename ShadeAt, typename ColourAt, typename Shader, typename SkipAt>
core::u32 drawHeightfieldPatch(const RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                               const HeightfieldPatchParams &params, const SunState &sun, HeightAt &&heightAt,
                               ShadeAt &&shadeAt, ColourAt &&colourAt, Shader &&shade, bool perPixel, SkipAt &&skipAt)
{
    const core::u32 stride = params.stride == 0u ? 1u : params.stride;
    const core::f32 strideF = static_cast<core::f32>(stride);
    core::u32 triangles = 0u;

    for (core::u32 z = 0u; z + stride <= params.size; z += stride)
    {
        for (core::u32 x = 0u; x + stride <= params.size; x += stride)
        {
            if (skipAt(x, z))
                continue;
            const core::f32 y00 = heightAt(x, z);
            const core::f32 y10 = heightAt(x + stride, z);
            const core::f32 y11 = heightAt(x + stride, z + stride);
            const core::f32 y01 = heightAt(x, z + stride);

            const core::f32 x0 = static_cast<core::f32>(params.originX + static_cast<core::i32>(x));
            const core::f32 x1 = x0 + strideF;
            const core::f32 z0 = static_cast<core::f32>(params.originZ + static_cast<core::i32>(z));
            const core::f32 z1 = z0 + strideF;

            const core::f32 nx = (y00 - y10) / strideF;
            const core::f32 nz = (y00 - y01) / strideF;
            const core::f32 normalScale = 1.0f / (1.0f + nx * nx + nz * nz);
            core::f32 ndl = (nx * sun.x + sun.y + nz * sun.z) * normalScale;
            ndl = ndl < 0.0f ? 0.0f : ndl;

            const core::f32 occlusion = shadeAt(x, z);
            const core::f32 lit =
                OrbitCamera::clamp(params.ambient + (1.0f - params.ambient) * ndl * sun.intensity * (1.0f - occlusion),
                                   0.0f, params.maxLight);
            const core::u32 base = colourAt(x, z);

            core::f32 cy00 = y00, cy10 = y10, cy11 = y11, cy01 = y01;
            if (params.curvatureFactor != 0.0f)
            {
                const auto drop = [&](core::f32 wx, core::f32 wz) noexcept -> core::f32 {
                    const core::f32 dx = wx - params.cameraX;
                    const core::f32 dz = wz - params.cameraZ;
                    return (dx * dx + dz * dz) * params.curvatureFactor;
                };
                cy00 -= drop(x0, z0);
                cy10 -= drop(x1, z0);
                cy11 -= drop(x1, z1);
                cy01 -= drop(x0, z1);
            }

            const core::f32 quad[12] = {x0, cy00, z0, x1, cy10, z0, x1, cy11, z1, x0, cy01, z1};
            if (perPixel)
                triangles += fillPolygonShadedClipped(rt, mvp, quad, 4u, [&](core::f32 wx, core::f32 wy, core::f32 wz) {
                    (void) wy;
                    return shade(wx, wz, base, lit, nx, nz, occlusion);
                });
            else
                triangles += fillPolygonClipped(rt, mvp, quad, 4u, shade(x0, z0, base, lit, nx, nz, occlusion));
        }
    }
    return triangles;
}

template <typename HeightAt, typename CliffAt>
core::u32 drawPatchSkirts(const RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                          const HeightfieldPatchParams &params, core::f32 drop, HeightAt &&heightAt, CliffAt &&cliffAt)
{
    const core::u32 last = params.size - 1u;
    core::u32 triangles = 0u;

    for (core::u32 i = 0u; i + 1u < params.size; ++i)
    {
        const core::f32 wx = static_cast<core::f32>(params.originX + static_cast<core::i32>(i));
        const core::f32 wz = static_cast<core::f32>(params.originZ + static_cast<core::i32>(i));
        const core::f32 x0 = static_cast<core::f32>(params.originX);
        const core::f32 x1 = static_cast<core::f32>(params.originX + static_cast<core::i32>(last));
        const core::f32 z0 = static_cast<core::f32>(params.originZ);
        const core::f32 z1 = static_cast<core::f32>(params.originZ + static_cast<core::i32>(last));

        triangles +=
            drawSkirtQuad(rt, mvp, wx, heightAt(i, 0u), z0, wx + 1.0f, heightAt(i + 1u, 0u), z0, drop, cliffAt(i, 0u));
        triangles += drawSkirtQuad(rt, mvp, wx, heightAt(i, last), z1, wx + 1.0f, heightAt(i + 1u, last), z1, drop,
                                   cliffAt(i, last));
        triangles +=
            drawSkirtQuad(rt, mvp, x0, heightAt(0u, i), wz, x0, heightAt(0u, i + 1u), wz + 1.0f, drop, cliffAt(0u, i));
        triangles += drawSkirtQuad(rt, mvp, x1, heightAt(last, i), wz, x1, heightAt(last, i + 1u), wz + 1.0f, drop,
                                   cliffAt(last, i));
    }
    return triangles;
}

} // namespace lpl::render

#endif // LPL_RENDER_HEIGHTFIELD_PATCH_INL
