/**
 * @file test_irradiance.cpp
 * @brief The environment term has to DEPEND on the normal, or it is not lighting.
 *
 * The physically based path shipped with the sky read straight up and used for every
 * surface. That is a defensible placeholder and an indefensible result: a wall
 * facing the sunset and a wall facing away from it received identical ambient light,
 * so half of what a sky is for never reached the image.
 *
 * The failure mode is invisible in a render — a slightly flat valley looks like a
 * choice — which is why it is worth an assertion rather than an eye. What is checked
 * here is not a colour but the properties any diffuse image-based lighting must
 * have, each of which the constant version fails or trivially passes:
 *
 *  - A CONSTANT environment must reconstruct to that same constant. This is the
 *    sanity check on the whole projection-and-convolution chain: get the band
 *    weights or the basis normalisation wrong and a grey sky comes back darker or
 *    brighter than grey, in a way no other test would notice.
 *  - A real sky must give DIFFERENT answers for different normals, and the up
 *    direction must receive more than the down direction, because that is where the
 *    sky is.
 *  - No component may come back negative. A band-limited reconstruction of an
 *    environment with a sharp horizon dips below zero, and negative light is black
 *    pixels rather than a subtlety.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-31
 * @copyright MIT License
 */

#include <cstdio>
#include <lpl/render/Irradiance.hpp>

using namespace lpl;

static int failures = 0;

static void check(bool condition, const char *what)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++failures;
}

static bool near(core::f32 a, core::f32 b, core::f32 tolerance) noexcept
{
    const core::f32 d = a - b;
    return (d < 0.0f ? -d : d) <= tolerance;
}

int main()
{
    std::printf("== sky irradiance (diffuse image-based lighting) ==\n");

    // 1. A constant environment must come back constant, whatever the normal.
    {
        const render::IrradianceProbe probe = render::IrradianceProbe::uniform(render::Vec3f(0.5f, 0.5f, 0.5f));
        const render::Vec3f up = render::evaluateIrradiance(probe, 0.0f, 1.0f, 0.0f);
        const render::Vec3f side = render::evaluateIrradiance(probe, 1.0f, 0.0f, 0.0f);
        const render::Vec3f down = render::evaluateIrradiance(probe, 0.0f, -1.0f, 0.0f);
        std::printf("  uniform 0.5 -> up %.4f  side %.4f  down %.4f\n", up.x, side.x, down.x);
        check(near(up.x, 0.5f, 0.02f), "a constant environment reconstructs to itself");
        check(near(up.x, side.x, 0.001f) && near(up.x, down.x, 0.001f), "and does so in every direction");
    }

    // 2. A real sky must be directional, and brighter above than below.
    {
        render::SkyParams params{};
        const render::SunState noon = render::sunAt(0.5f);
        const render::IrradianceProbe probe = render::projectSky(noon, params);

        const render::Vec3f up = render::evaluateIrradiance(probe, 0.0f, 1.0f, 0.0f);
        const render::Vec3f down = render::evaluateIrradiance(probe, 0.0f, -1.0f, 0.0f);
        const render::Vec3f east = render::evaluateIrradiance(probe, 1.0f, 0.0f, 0.0f);
        std::printf("  noon sky  -> up (%.3f %.3f %.3f)  down (%.3f %.3f %.3f)  east (%.3f %.3f %.3f)\n", up.x, up.y,
                    up.z, down.x, down.y, down.z, east.x, east.y, east.z);

        const core::f32 upLuma = up.x + up.y + up.z;
        const core::f32 downLuma = down.x + down.y + down.z;
        check(upLuma > downLuma * 1.10f, "a normal facing the sky receives more than one facing the ground");
        check(!near(upLuma, east.x + east.y + east.z, 0.01f), "the answer depends on the normal at all");
        check(up.z > up.x, "the sky it integrates is still blue");
        check(up.x >= 0.0f && up.y >= 0.0f && up.z >= 0.0f && down.x >= 0.0f && down.y >= 0.0f && down.z >= 0.0f,
              "no component is negative");
    }

    // 3. A low sun must bias the probe towards the colour it turns the sky.
    {
        render::SkyParams params{};
        const render::IrradianceProbe noon = render::projectSky(render::sunAt(0.5f), params);
        const render::IrradianceProbe dusk = render::projectSky(render::sunAt(0.26f), params);

        const render::Vec3f noonUp = render::evaluateIrradiance(noon, 0.0f, 1.0f, 0.0f);
        const render::Vec3f duskUp = render::evaluateIrradiance(dusk, 0.0f, 1.0f, 0.0f);
        const core::f32 noonWarmth = noonUp.x - noonUp.z;
        const core::f32 duskWarmth = duskUp.x - duskUp.z;
        std::printf("  warmth (red - blue): noon %.4f  dusk %.4f\n", noonWarmth, duskWarmth);
        check(duskWarmth > noonWarmth, "a low sun warms the ambient light it casts");
    }

    // 4. Resolution: a coarse projection must agree with a fine one. If it does not,
    //    the texel solid-angle weighting is wrong and the corners are dominating.
    {
        render::SkyParams params{};
        const render::SunState sun = render::sunAt(0.4f);
        const render::Vec3f coarse = render::evaluateIrradiance(render::projectSky(sun, params, 4u), 0.3f, 0.9f, 0.2f);
        const render::Vec3f fine = render::evaluateIrradiance(render::projectSky(sun, params, 24u), 0.3f, 0.9f, 0.2f);
        std::printf("  coarse (%.4f %.4f %.4f) vs fine (%.4f %.4f %.4f)\n", coarse.x, coarse.y, coarse.z, fine.x,
                    fine.y, fine.z);
        check(near(coarse.x, fine.x, 0.02f) && near(coarse.y, fine.y, 0.02f) && near(coarse.z, fine.z, 0.02f),
              "eight texels a face is already converged");
    }

    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\n%d FAILURE(S)\n", failures);
    return failures == 0 ? 0 : 1;
}
