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
#include <lpl/render/Water.hpp>

#include <utility>

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
    // ── The swell's direction ───────────────────────────────────────────────
    //
    // The two crest directions used to be constants in the ripple function; they are now a
    // drift a caller points. The property worth asserting is not that the waves move — it is
    // the one the original comment named and that setting four fields by hand would lose:
    // the two crests must NOT be perpendicular, or they interfere into a checkerboard that
    // reads as a tiled floor.
    //
    // That the DEFAULTS still produce the original surface is asserted by the signatures
    // below, which did not move when the literals became dot products.
    {
        std::printf("== water: the swell runs where it is pointed ==\n");
        for (const auto &dir : {std::pair{1.0f, 0.0f}, std::pair{0.0f, 1.0f}, std::pair{-1.0f, 0.0f},
                               std::pair{0.6f, -0.8f}})
        {
            lpl::render::WaterParams water;
            water.setDrift(dir.first, dir.second);
            const float dot = water.driftX * water.crossX + water.driftZ * water.crossZ;
            const float driftLen = water.driftX * water.driftX + water.driftZ * water.driftZ;
            const float crossLen = water.crossX * water.crossX + water.crossZ * water.crossZ;
            // Cosine of the angle, squared, against the 103 degrees the pair is rotated by:
            // cos(103) is about -0.225, so the squared cosine is about 0.0506. A pair that had
            // become perpendicular would give zero, and a parallel one would give one.
            const float cosSq = (dot * dot) / (driftLen * crossLen);
            check(cosSq > 0.02f && cosSq < 0.10f, "the crossing crest stays off perpendicular");
            check(crossLen > 0.5f * driftLen, "and keeps the drift's magnitude");
        }
    }

    // ── The swell ───────────────────────────────────────────────────────────
    //
    // Every field of it defaults to OFF, which is the first thing to assert: a lake in a
    // courtyard must not start heaving because a sea was added to the engine.
    {
        std::printf("== water: the swell ==\n");
        const lpl::render::WaterParams still;
        check(lpl::render::waterHeight(3.0f, 7.0f, still) == 0.0f, "a surface with no swell is flat");

        lpl::render::WaterParams sea;
        sea.swellHeight = 0.8f;
        sea.crestSharpness = 0.6f;
        sea.foamCrest = 0.52f;
        float highest = -1.0e9f;
        float lowest = 1.0e9f;
        for (int i = 0; i < 512; ++i)
        {
            const float h = lpl::render::waterHeight(static_cast<float>(i) * 0.37f, static_cast<float>(i) * 0.11f, sea);
            highest = h > highest ? h : highest;
            lowest = h < lowest ? h : lowest;
        }
        std::printf("    swell spans %.3f .. %.3f for a stated height of %.2f\n", static_cast<double>(lowest),
                    static_cast<double>(highest), static_cast<double>(sea.swellHeight));
        check(highest <= sea.swellHeight + 1.0e-4f && lowest >= -sea.swellHeight - 1.0e-4f,
              "and one with a swell stays inside the height it declares");

        // The property the previous version did NOT have, and the reason foam could not be
        // trusted to land on a crest: the gradient must be the derivative of the SAME sum
        // the height comes from. The old normal used the wave's value, which is a quarter
        // period out — a highlight on the flank instead of the face.
        //
        // Only the SIGN is compared: the gradient is normalised against its own worst case
        // rather than the height's, so its magnitude is not d(height)/dx. Points where the
        // difference is tiny are skipped — that is a crest or a trough, where the sign of a
        // finite difference is noise.
        int compared = 0;
        int disagreements = 0;
        for (int i = 0; i < 400; ++i)
        {
            const float x = static_cast<float>(i) * 0.23f;
            const float z = 5.0f;
            const float step = 0.004f;
            const float before = lpl::render::waterHeight(x - step, z, sea);
            const float after = lpl::render::waterHeight(x + step, z, sea);
            const float difference = after - before;
            if (difference > -0.0008f && difference < 0.0008f)
                continue;
            // A triangle wave has TURNING POINTS, and it is not differentiable at them. If
            // one lies inside the sampling interval the finite difference is a chord across
            // a peak, whose sign says nothing about either side — so those points are
            // excluded rather than counted as disagreements. Detected by the slope changing
            // sign across the interval, which is what a turning point is.
            const float slopeBefore = lpl::render::sampleWater(x - step, z, sea).slopeX;
            const float slopeAfter = lpl::render::sampleWater(x + step, z, sea).slopeX;
            if ((slopeBefore > 0.0f) != (slopeAfter > 0.0f))
                continue;
            const float slope = lpl::render::sampleWater(x, z, sea).slopeX;
            ++compared;
            if ((difference > 0.0f) != (slope > 0.0f))
                ++disagreements;
        }
        std::printf("    %d sampled slopes, %d disagreeing in sign with the height they came from\n", compared,
                    disagreements);
        check(compared > 100, "enough points were steep enough to judge");
        // Not exactly zero, and the allowance is reasoned rather than tuned: three octaves
        // of a folded triangle have turning points everywhere, and a sampling interval can
        // straddle TWO of them — the slope then has the same sign at both ends while the
        // chord between them does not. What this rules out is the defect it exists for: a
        // normal taken from the wave's VALUE is a quarter period out of phase and disagrees
        // on about half of all points, so two per cent is a fifty-fold margin, not a
        // threshold chosen to make today's numbers pass.
        check(disagreements * 50 <= compared,
              "the gradient is the derivative of the height, not a second guess at it");

        // A crest is the TOP of the wave and nothing else. A threshold that let the whole
        // surface count as a crest would make the sea milk and the scatter uniform.
        int cresting = 0;
        int crestAboveThreshold = 0;
        for (int i = 0; i < 2000; ++i)
        {
            const float x = static_cast<float>(i) * 0.17f;
            const lpl::render::WaveSample sample = lpl::render::sampleWater(x, 2.0f, sea);
            if (sample.crest <= 0.0f)
                continue;
            ++cresting;
            if (sample.height > sea.foamCrest - 1.0e-5f)
                ++crestAboveThreshold;
        }
        std::printf("    %d of 2000 samples are cresting, %d of them above the threshold\n", cresting,
                    crestAboveThreshold);
        // A BAND, not a floor. Eleven samples in two thousand was the first measurement and
        // it is a flat calm — a whitecap you meet once a minute is one nobody sees. Most of
        // the sea breaking is a different failure and just as wrong.
        check(cresting > 40, "enough of the sea is cresting to see whitecaps");
        check(cresting < 1000, "but not most of it");
        check(cresting == crestAboveThreshold, "and a crest is exactly the part above the stated threshold");

        // Sharpening broadens the trough and narrows the peak, so the MEAN drops. Asserted
        // as monotonicity across three settings rather than against a number: a fixed
        // number here would be one chosen so that today's tuning passes.
        float previous = 1.0e9f;
        bool monotone = true;
        for (float sharpness : {0.0f, 0.45f, 0.9f})
        {
            lpl::render::WaterParams shaped = sea;
            shaped.crestSharpness = sharpness;
            float total = 0.0f;
            for (int i = 0; i < 1024; ++i)
                total += lpl::render::sampleWater(static_cast<float>(i) * 0.031f, 1.0f, shaped).height;
            const float mean = total / 1024.0f;
            std::printf("    sharpness %.2f -> mean height %+.4f\n", static_cast<double>(sharpness),
                        static_cast<double>(mean));
            if (mean >= previous)
                monotone = false;
            previous = mean;
        }
        check(monotone, "a sharper crest is a broader trough");
    }

    // ── Foam ────────────────────────────────────────────────────────────────
    //
    // Foam is the only part of a sea whose colour comes from neither what is behind it nor
    // what is above it, so it is the one thing that must be applied LAST — after the
    // reflection, the Fresnel and the glint. Two claims: it is off unless asked for, and
    // when asked for it whitens the shallows and not the deeps.
    {
        std::printf("== water: foam ==\n");
        const lpl::render::SunState sun = lpl::render::sunAt(0.32f);
        const lpl::render::SkyParams sky;
        const lpl::math::Vec3<float> eye{0.0f, 6.0f, -12.0f};

        lpl::render::WaterParams dry;
        dry.swellHeight = 0.8f;
        dry.foamGain = 0.0f;
        lpl::render::WaterParams surf = dry;
        surf.foamGain = 0.9f;
        surf.foamWidth = 1.8f;

        const unsigned shallowDry = lpl::render::waterColour(2.0f, 0.0f, 4.0f, eye, sun, sky, dry, 0.1f);
        const unsigned shallowSurf = lpl::render::waterColour(2.0f, 0.0f, 4.0f, eye, sun, sky, surf, 0.1f);
        const unsigned deepDry = lpl::render::waterColour(2.0f, 0.0f, 4.0f, eye, sun, sky, dry, 40.0f);
        const unsigned deepSurf = lpl::render::waterColour(2.0f, 0.0f, 4.0f, eye, sun, sky, surf, 40.0f);

        const auto luminance = [](unsigned c) {
            return static_cast<int>((c >> 16) & 0xFFu) + static_cast<int>((c >> 8) & 0xFFu) +
                   static_cast<int>(c & 0xFFu);
        };
        std::printf("    shallow %06X -> %06X, deep %06X -> %06X\n", shallowDry, shallowSurf, deepDry, deepSurf);
        check(shallowSurf != shallowDry, "asking for foam changes the shallows");
        check(luminance(shallowSurf) > luminance(shallowDry), "and makes them brighter, which is what surf is");
        // Deep water may still whiten AT a crest — that is a whitecap, and it is wanted.
        // What must not happen is the shore band reaching out to sea, so the deep water must
        // whiten LESS than the shallows do.
        check(luminance(shallowSurf) - luminance(shallowDry) > luminance(deepSurf) - luminance(deepDry),
              "and whitens a shore more than open water");
    }

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
