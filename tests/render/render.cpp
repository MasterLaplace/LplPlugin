#include <lpl/render/Lighting.hpp>
#include <lpl/render/RenderParity.hpp>
#include <lpl/render/SoftwareRasterizer.hpp>
#include <lpl/render/Texture.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(render);

namespace {

constexpr lpl::core::u32 kWidth = 96u;
constexpr lpl::core::u32 kHeight = 64u;
constexpr lpl::core::u32 kViewportsWidth = 128u;
constexpr lpl::core::u32 kViewportsHeight = 96u;
constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;

/**
 * @brief The targets the cubes are drawn into, static because a kernel stack cannot hold them.
 */
lpl::core::u32 gColour[kWidth * kHeight];
lpl::core::f32 gDepth[kWidth * kHeight];
lpl::core::u32 gViewportsColour[kViewportsWidth * kViewportsHeight];
lpl::core::f32 gViewportsDepth[kViewportsWidth * kViewportsHeight];

[[nodiscard]] lpl::render::RenderTarget cubeTarget() { return {gColour, gDepth, kWidth, kHeight}; }

[[nodiscard]] lpl::render::Texture checker()
{
    return lpl::render::Texture::makeChecker(64u, 64u, 0x00FF0000u, 0x000000FFu, 8u);
}

[[nodiscard]] lpl::core::u32 foldFlatCube(lpl::math::Fixed32 angle)
{
    lpl::render::RenderTarget target = cubeTarget();

    lpl::render::renderCube(target, angle);
    return lpl::render::foldTarget(target);
}

} // namespace

/**
 * @brief Gate P5 render, projection: a Fixed32 cube turned by CORDIC and projected in float lands
 *        on the same pixels and depths on both targets.
 */
LPL_TEST(cube_projects_inside_the_viewport)
{
    const auto straight = lpl::render::projectParityCube(lpl::math::Fixed32::fromInt(0), 1280u, 800u);
    const auto turned = lpl::render::projectParityCube(lpl::math::Fixed32::fromFloat(0.78539816f), 1280u, 800u);

    test.check(straight.in_front_count == 8u, "the eight corners are in front of the camera");
    test.check(straight.vertex0_x > 0 && straight.vertex0_x < 1280 && straight.vertex0_y > 0 &&
                   straight.vertex0_y < 800,
               "the first corner lands inside the viewport");
    test.check(turned.in_front_count == 8u, "turned by a quarter pi, they are still in front");
    test.check(turned.screen_signature != straight.screen_signature, "and the turn moves them on screen");

    test.measureHexadecimal("screen_signature", straight.screen_signature);
    test.measureHexadecimal("depth_signature", straight.depth_signature);
    test.measure("first_corner_x", straight.vertex0_x);
    test.measure("first_corner_y", straight.vertex0_y);
    test.measureHexadecimal("turned_screen_signature", turned.screen_signature);
    test.measureHexadecimal("turned_depth_signature", turned.depth_signature);
}

LPL_TEST(rasterized_cube_writes_pixels)
{
    const lpl::core::u32 straight = foldFlatCube(lpl::math::Fixed32::fromInt(0));
    lpl::render::RenderTarget target = cubeTarget();

    lpl::render::clearTarget(target, 0x00102030u);

    const lpl::core::u32 cleared = lpl::render::foldTarget(target);
    const lpl::core::u32 turned = foldFlatCube(lpl::math::Fixed32::fromFloat(0.78539816f));

    test.check(straight != cleared, "the cube covers pixels of a cleared target");
    test.check(turned != straight, "and turning it changes them");

    test.measureHexadecimal("cube_signature", straight);
    test.measureHexadecimal("turned_cube_signature", turned);
}

LPL_TEST(frustum_culls_some_instances)
{
    const auto culled = lpl::render::cullParityInstanceGrid(1280u, 800u);

    test.check(culled.total == 49u, "the grid holds 7 by 7 instances");
    test.check(culled.visible > 0u && culled.visible < culled.total, "the frustum keeps some and not all");

    test.measure("visible_instances", culled.visible);
    test.measureHexadecimal("visible_signature", culled.visible_signature);
}

/**
 * @brief Gate P5 render, textures: integer sampling of a checker, and the cube it textures.
 */
LPL_TEST(checker_texture_samples_exactly)
{
    const lpl::render::Texture texture = checker();
    lpl::core::u32 diagonal = kFnv1aOffsetBasis;

    test.check(texture.sampleNearest(0u, 0u) == 0x00FF0000u, "the first cell has the first colour");
    test.check(texture.sampleNearest(32768u, 0u) == 0x00FF0000u, "so does cell 4, at u = 0.5");
    test.check(texture.sampleNearest(9u * 65536u / 64u, 0u) == 0x000000FFu, "cell 1 has the second colour");

    for (lpl::core::u32 sample = 0u; sample < 64u; ++sample)
    {
        const lpl::core::u32 coordinate = (sample * 65536u) / 64u;

        diagonal = lpl::render::detail::fnv1aStep(diagonal, texture.sampleBilinear(coordinate, coordinate));
    }

    lpl::render::RenderTarget target = cubeTarget();

    lpl::render::renderTexturedCube(target, lpl::math::Fixed32::fromInt(0), texture);

    const lpl::core::u32 textured = lpl::render::foldTarget(target);

    test.check(textured != foldFlatCube(lpl::math::Fixed32::fromInt(0)), "a textured cube differs from a flat one");

    test.measureHexadecimal("diagonal_signature", diagonal);
    test.measureHexadecimal("textured_cube_signature", textured);
}

/**
 * @brief Gate P5 render, lighting: one fragment under one directional light, and the lit cube.
 */
LPL_TEST(lighting_models_shade_the_fragment)
{
    lpl::render::Material material;
    lpl::render::Light light;
    const lpl::render::Vec3f normal(0.0f, 0.0f, 1.0f);
    const lpl::render::Vec3f fragment(0.0f, 0.0f, 1.0f);
    const lpl::render::Vec3f eye(0.0f, 0.0f, 5.0f);

    material.albedo = lpl::render::Vec3f(0.8f, 0.7f, 0.6f);
    material.shininess = 32u;
    light.type = lpl::render::LightType::Directional;
    light.direction = lpl::render::Vec3f(-0.4f, -0.7f, -0.6f);

    const lpl::core::u32 lambert =
        lpl::render::shadeToRgb(lpl::render::ShadingModel::Lambert, material, &light, 1u, normal, fragment, eye);
    const lpl::core::u32 phong =
        lpl::render::shadeToRgb(lpl::render::ShadingModel::Phong, material, &light, 1u, normal, fragment, eye);
    const lpl::core::u32 blinnPhong =
        lpl::render::shadeToRgb(lpl::render::ShadingModel::BlinnPhong, material, &light, 1u, normal, fragment, eye);
    lpl::render::RenderTarget target = cubeTarget();

    lpl::render::renderLitCube(target, lpl::math::Fixed32::fromInt(0), lpl::render::ShadingModel::BlinnPhong);

    const lpl::core::u32 litCube = lpl::render::foldTarget(target);

    test.check(lambert != 0u, "Lambert lights the fragment");
    test.check(phong != lambert || blinnPhong != lambert, "a specular model adds to it");
    test.check(litCube != foldFlatCube(lpl::math::Fixed32::fromInt(0)), "a lit cube differs from a flat one");

    test.measureHexadecimal("lambert_colour", lambert);
    test.measureHexadecimal("phong_colour", phong);
    test.measureHexadecimal("blinn_phong_colour", blinnPhong);
    test.measureHexadecimal("lit_cube_signature", litCube);
}

LPL_TEST(viewports_and_render_to_texture_write_pixels)
{
    lpl::render::RenderTarget viewports{gViewportsColour, gViewportsDepth, kViewportsWidth, kViewportsHeight};
    lpl::render::RenderTarget target = cubeTarget();

    lpl::render::clearTarget(viewports, 0x00102030u);

    const lpl::core::u32 cleared = lpl::render::foldTarget(viewports);

    lpl::render::renderMultiViewport(viewports);

    const lpl::core::u32 composite = lpl::render::foldTarget(viewports);

    lpl::render::renderTexturedCube(target, lpl::math::Fixed32::fromInt(0), checker());

    const lpl::core::u32 textured = lpl::render::foldTarget(target);

    lpl::render::renderToTextureCube(target, lpl::math::Fixed32::fromInt(0));

    const lpl::core::u32 renderedToTexture = lpl::render::foldTarget(target);

    test.check(composite != cleared, "the viewports write pixels");
    test.check(renderedToTexture != textured, "a cube textured with a render differs from the checker cube");

    test.measureHexadecimal("viewports_signature", composite);
    test.measureHexadecimal("render_to_texture_signature", renderedToTexture);
}
