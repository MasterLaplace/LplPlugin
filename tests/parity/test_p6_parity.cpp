/**
 * @file test_p6_parity.cpp
 * @brief Parity test for P6: topology, ray tracing, PBR, immutable command
 *        buffers with late-latching, and foveated rasterization.
 *
 * Every slice authors authoritative state in Fixed32 and runs the render-side
 * math in float (SSE, -ffp-contract=off), bit-identical host vs kernel. The
 * folded signatures printed here are the cross-target authority for the in-kernel
 * smoke (libengine p6).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#include <cstdio>
#include <lpl/render/CommandBuffer.hpp>
#include <lpl/render/Foveated.hpp>
#include <lpl/render/Pbr.hpp>
#include <lpl/render/RayTracer.hpp>
#include <lpl/render/Topology.hpp>

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
    std::printf("== P6.1 topology ==\n");
    {
        // Catmull-Rom loop through a Fixed32 square + apex.
        const math::Fixed32 ctrl[5][3] = {
            {math::Fixed32::fromInt(-2), math::Fixed32::fromInt(0), math::Fixed32::fromInt(-2)},
            {math::Fixed32::fromInt(2), math::Fixed32::fromInt(0), math::Fixed32::fromInt(-2)},
            {math::Fixed32::fromInt(2), math::Fixed32::fromInt(0), math::Fixed32::fromInt(2)},
            {math::Fixed32::fromInt(-2), math::Fixed32::fromInt(0), math::Fixed32::fromInt(2)},
            {math::Fixed32::fromInt(0), math::Fixed32::fromInt(3), math::Fixed32::fromInt(0)},
        };
        const auto loop = render::tessellateCatmullLoop(ctrl, 5u, 8u);
        check(loop.sample_count == 40u, "catmull loop emits 5*8 samples");

        const auto saddle = render::tessellateSaddle(16u);
        check(saddle.sample_count == 17u * 17u, "saddle tessellation emits (res+1)^2 verts");

        // Delaunay of a Fixed32 point cloud.
        const math::Fixed32 pts[6][3] = {
            {math::Fixed32::fromInt(0), math::Fixed32::fromInt(0), math::Fixed32::fromInt(0)},
            {math::Fixed32::fromInt(4), math::Fixed32::fromInt(0), math::Fixed32::fromInt(0)},
            {math::Fixed32::fromInt(4), math::Fixed32::fromInt(4), math::Fixed32::fromInt(0)},
            {math::Fixed32::fromInt(0), math::Fixed32::fromInt(4), math::Fixed32::fromInt(0)},
            {math::Fixed32::fromInt(2), math::Fixed32::fromInt(2), math::Fixed32::fromInt(0)},
            {math::Fixed32::fromInt(1), math::Fixed32::fromInt(3), math::Fixed32::fromInt(0)},
        };
        const auto del = render::delaunay2D(pts, 6u);
        check(del.triangle_count > 0u, "delaunay produces triangles");
        std::printf("  catmull_sig=0x%08X saddle_sig=0x%08X\n", loop.sample_signature, saddle.sample_signature);
        std::printf("  delaunay tris=%u tri_sig=0x%08X\n", del.triangle_count, del.triangle_signature);
    }

    std::printf("== P6.2 software ray tracing ==\n");
    {
        constexpr core::u32 W = 96u, H = 72u;
        static core::u32 img[W * H];
        const auto rt = render::rayTraceScene(img, W, H, 3u);
        check(rt.hit_count > 0u, "ray tracer hits geometry");
        check(rt.image_signature != 0x811C9DC5u, "ray traced image non-trivial");
        std::printf("  hits=%u image_sig=0x%08X\n", rt.hit_count, rt.image_signature);
    }

    std::printf("== P6.3 PBR metallic/roughness + HDRI tone map ==\n");
    {
        render::PbrMaterial gold;
        gold.albedo = render::Vec3f(1.0f, 0.77f, 0.34f);
        gold.metallic = 1.0f;
        gold.roughness = 0.25f;
        render::PbrMaterial plastic;
        plastic.albedo = render::Vec3f(0.2f, 0.6f, 0.9f);
        plastic.metallic = 0.0f;
        plastic.roughness = 0.6f;

        render::Light key;
        key.type = render::LightType::Directional;
        key.direction = render::Vec3f(-0.5f, -0.8f, -0.6f);
        key.intensity = 3.0f;
        const render::Vec3f N(0.0f, 0.0f, 1.0f);
        const render::Vec3f frag(0.0f, 0.0f, 0.0f);
        const render::Vec3f eye(0.0f, 0.0f, 3.0f);
        const render::Vec3f hdri(0.12f, 0.14f, 0.18f);

        const core::u32 goldRein = render::pbrShadeToRgb(gold, &key, 1u, N, frag, eye, hdri, render::ToneMap::Reinhard);
        const core::u32 goldAces = render::pbrShadeToRgb(gold, &key, 1u, N, frag, eye, hdri, render::ToneMap::Aces);
        const core::u32 plasticAces =
            render::pbrShadeToRgb(plastic, &key, 1u, N, frag, eye, hdri, render::ToneMap::Aces);
        check(goldRein != 0u && goldAces != 0u, "PBR shade non-black");
        check(goldAces != plasticAces, "metal vs dielectric differ");
        check(goldRein != goldAces, "tone-map operators differ");
        std::printf("  gold_reinhard=0x%06X gold_aces=0x%06X plastic_aces=0x%06X\n", goldRein, goldAces, plasticAces);
    }

    std::printf("== P6.4/P6.5 command buffer + late-latching ==\n");
    {
        render::CommandBuffer cb;
        for (core::u32 i = 0; i < 4u; ++i)
            cb.record(render::DrawCommand{0x1000u + i * 0x100u, 0x9000u + i * 0x40u, 36u, 1u, i, i & 1u});
        cb.finalize();
        cb.record(render::DrawCommand{}); // rejected (sealed)
        check(cb.count() == 4u, "sealed command buffer rejects further records");
        const core::u32 recSig = cb.recordingSignature();

        // Pose buffer, then advance the sim and re-submit the SAME recording.
        render::Pose poses[4];
        for (core::u32 i = 0; i < 4u; ++i)
            poses[i].x = math::Fixed32::fromInt(static_cast<core::i32>(i));
        const auto s0 = render::submitLateLatched(cb, poses, 4u);

        // Late latch: mutate poses, recording unchanged -> signature must change.
        for (core::u32 i = 0; i < 4u; ++i)
            poses[i].x = poses[i].x + math::Fixed32::fromFloat(0.5f);
        const auto s1 = render::submitLateLatched(cb, poses, 4u);
        check(s0.draws == 4u && s1.draws == 4u, "all draws submitted");
        check(s0.latched_signature != s1.latched_signature, "late-latch reflects new poses, same recording");
        std::printf("  recording_sig=0x%08X latched0=0x%08X latched1=0x%08X\n", recSig, s0.latched_signature,
                    s1.latched_signature);
    }

    std::printf("== P6.6 foveated rasterizer ==\n");
    {
        constexpr core::u32 W = 128u, H = 96u;
        static core::u32 img[W * H];
        const auto fov = render::foveatedShade(img, W, H, 64u, 48u);
        check(fov.shaded_fragments < fov.full_fragments, "foveation shades fewer fragments than full rate");
        check(fov.shaded_fragments > 0u, "foveation shades something");
        std::printf("  shaded=%u/%u image_sig=0x%08X\n", fov.shaded_fragments, fov.full_fragments, fov.image_signature);
    }

    std::printf("%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
