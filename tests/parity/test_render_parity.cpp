/**
 * @file test_render_parity.cpp
 * @brief Parity test: deterministic 3D camera/projection pipeline.
 *
 * Projects a Fixed32-authored unit cube (CORDIC model rotation) through a
 * perspective camera and folds the resulting screen coordinates + depths. The
 * geometry/rotation is authoritative Fixed32; the view/projection/divide is
 * float (SSE, -ffp-contract=off) which is bit-identical host vs kernel. The
 * folded signatures are the cross-target authority for the in-kernel smoke.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#include <cstdio>
#include <lpl/render/RenderParity.hpp>
#include <lpl/render/SoftwareRasterizer.hpp>
#include <lpl/render/Texture.hpp>

using namespace lpl;

static int failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

int main()
{
    std::printf("== 3D projection parity ==\n");

    // Identity rotation: cube is axis-aligned, centred in front of the camera.
    const auto r0 = render::projectParityCube(math::Fixed32::fromInt(0), 1280u, 800u);
    check(r0.in_front_count == 8u, "all 8 vertices in front of camera (angle 0)");
    check(r0.vertex0_x > 0 && r0.vertex0_x < 1280, "vertex0 x within viewport");
    check(r0.vertex0_y > 0 && r0.vertex0_y < 800, "vertex0 y within viewport");

    // A quarter-turn rotation must change the screen fold but keep all in front.
    const auto rq = render::projectParityCube(math::Fixed32::fromFloat(0.78539816f), 1280u, 800u);
    check(rq.in_front_count == 8u, "all 8 vertices in front of camera (angle pi/4)");
    check(rq.screen_signature != r0.screen_signature, "rotation changes screen signature");

    // Report the raw signatures so the kernel smoke can be compared bit-for-bit.
    std::printf("== signatures (must match kernel smoke) ==\n");
    std::printf("  angle0 screen_sig = 0x%08X\n", r0.screen_signature);
    std::printf("  angle0 depth_sig  = 0x%08X\n", r0.depth_signature);
    std::printf("  angle0 vertex0    = (%d, %d)\n", r0.vertex0_x, r0.vertex0_y);
    std::printf("  pi/4   screen_sig = 0x%08X\n", rq.screen_signature);
    std::printf("  pi/4   depth_sig  = 0x%08X\n", rq.depth_signature);

    std::printf("== software 3D rasterizer (depth-buffered cube) ==\n");
    constexpr core::u32 kW = 96u;
    constexpr core::u32 kH = 64u;
    static core::u32 colorBuf[kW * kH];
    static core::f32 depthBuf[kW * kH];
    render::RenderTarget rt{colorBuf, depthBuf, kW, kH};

    render::renderCube(rt, math::Fixed32::fromInt(0));
    const core::u32 cubeSig0 = render::foldTarget(rt);
    // The cube must cover some pixels (signature differs from a cleared buffer).
    render::clearTarget(rt, 0x00102030u);
    const core::u32 clearSig = render::foldTarget(rt);
    check(cubeSig0 != clearSig, "rasterized cube writes pixels (depth test)");

    render::renderCube(rt, math::Fixed32::fromFloat(0.78539816f));
    const core::u32 cubeSigQ = render::foldTarget(rt);
    check(cubeSigQ != cubeSig0, "rotation changes rasterized cube");

    std::printf("  cube angle0 sig = 0x%08X\n", cubeSig0);
    std::printf("  cube pi/4   sig = 0x%08X\n", cubeSigQ);

    std::printf("== instancing + frustum cull ==\n");
    const auto cull = render::cullParityInstanceGrid(1280u, 800u);
    check(cull.total == 49u, "instance grid has 49 entries (7x7)");
    check(cull.visible > 0u && cull.visible < cull.total, "frustum culls some but not all instances");
    std::printf("  total=%u visible=%u visible_sig=0x%08X\n", cull.total, cull.visible, cull.visible_signature);

    std::printf("== texture sampling (integer-deterministic) ==\n");
    const auto tex = render::Texture::makeChecker(64u, 64u, 0x00FF0000u, 0x000000FFu, 8u);
    // Nearest at a cell corner; bilinear at a cell boundary blends the two.
    const core::u32 n0 = tex.sampleNearest(0u, 0u);
    const core::u32 nMid = tex.sampleNearest(32768u, 0u);            // u=0.5 -> cell 4 (even) -> colorA
    const core::u32 nOdd = tex.sampleNearest(9u * 65536u / 64u, 0u); // x=9 -> cell 1 (odd) -> colorB
    check(n0 == 0x00FF0000u, "nearest (0,0) = colorA");
    check(nMid == 0x00FF0000u, "nearest (0.5,0) = colorA (cell 4, even)");
    check(nOdd == 0x000000FFu, "nearest (cell 1) = colorB (odd)");
    // Fold a row of bilinear samples across the texture.
    core::u32 texSig = 0x811C9DC5u;
    for (core::u32 i = 0; i < 64u; ++i)
    {
        const core::u32 uq = (i * 65536u) / 64u;
        texSig = render::detail::fnv1aStep(texSig, tex.sampleBilinear(uq, uq));
    }
    check(texSig != 0x811C9DC5u, "bilinear sample fold non-trivial");

    constexpr core::u32 tW = 96u;
    constexpr core::u32 tH = 64u;
    static core::u32 texColor[tW * tH];
    static core::f32 texDepth[tW * tH];
    render::RenderTarget trt{texColor, texDepth, tW, tH};
    render::renderTexturedCube(trt, math::Fixed32::fromInt(0), tex);
    const core::u32 texturedCubeSig = render::foldTarget(trt);
    check(texturedCubeSig != cubeSig0, "textured cube differs from flat-shaded cube");
    std::printf("  tex_sample_sig = 0x%08X\n", texSig);
    std::printf("  textured_cube_sig = 0x%08X\n", texturedCubeSig);

    std::printf("== classical lighting ==\n");
    {
        render::Material mat;
        mat.albedo = render::Vec3f(0.8f, 0.7f, 0.6f);
        mat.shininess = 32u;
        render::Light dir;
        dir.type = render::LightType::Directional;
        dir.direction = render::Vec3f(-0.4f, -0.7f, -0.6f);
        const render::Vec3f N(0.0f, 0.0f, 1.0f);
        const render::Vec3f frag(0.0f, 0.0f, 1.0f);
        const render::Vec3f eye(0.0f, 0.0f, 5.0f);
        const core::u32 lamb = render::shadeToRgb(render::ShadingModel::Lambert, mat, &dir, 1u, N, frag, eye);
        const core::u32 phong = render::shadeToRgb(render::ShadingModel::Phong, mat, &dir, 1u, N, frag, eye);
        const core::u32 blinn = render::shadeToRgb(render::ShadingModel::BlinnPhong, mat, &dir, 1u, N, frag, eye);
        check(lamb != 0u, "Lambert shade non-black");
        check(phong != lamb || blinn != lamb, "specular models differ from Lambert");
        std::printf("  lambert=0x%06X phong=0x%06X blinn=0x%06X\n", lamb, phong, blinn);

        constexpr core::u32 lW = 96u, lH = 64u;
        static core::u32 litColor[lW * lH];
        static core::f32 litDepth[lW * lH];
        render::RenderTarget lrt{litColor, litDepth, lW, lH};
        render::renderLitCube(lrt, math::Fixed32::fromInt(0), render::ShadingModel::BlinnPhong);
        const core::u32 litCubeSig = render::foldTarget(lrt);
        check(litCubeSig != cubeSig0, "lit cube differs from flat cube");
        std::printf("  lit_cube_sig = 0x%08X\n", litCubeSig);
    }

    std::printf("== multi-viewport + render-to-texture ==\n");
    {
        constexpr core::u32 mW = 128u, mH = 96u;
        static core::u32 mvColor[mW * mH];
        static core::f32 mvDepth[mW * mH];
        render::RenderTarget mrt{mvColor, mvDepth, mW, mH};
        render::renderMultiViewport(mrt);
        const core::u32 mvSig = render::foldTarget(mrt);
        check(mvSig != clearSig, "multi-viewport composite writes pixels");

        static core::u32 rttColor[96 * 64];
        static core::f32 rttDepth[96 * 64];
        render::RenderTarget rrt{rttColor, rttDepth, 96u, 64u};
        render::renderToTextureCube(rrt, math::Fixed32::fromInt(0));
        const core::u32 rttSig = render::foldTarget(rrt);
        check(rttSig != texturedCubeSig, "render-to-texture cube differs from checker cube");
        std::printf("  multiviewport_sig = 0x%08X\n", mvSig);
        std::printf("  render_to_texture_sig = 0x%08X\n", rttSig);
    }

    std::printf("%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
