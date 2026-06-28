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

    std::printf("%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
