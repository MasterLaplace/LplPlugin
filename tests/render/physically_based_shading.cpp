#include <lpl/render/Pbr.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(physically_based_shading);

/**
 * @brief Gate P6 shading: gold and plastic under one light and an ambient sky, tone-mapped two ways.
 */
LPL_TEST(metal_and_plastic_shade_apart)
{
    lpl::render::PbrMaterial gold;
    lpl::render::PbrMaterial plastic;
    lpl::render::Light key;
    const lpl::render::Vec3f normal(0.0f, 0.0f, 1.0f);
    const lpl::render::Vec3f fragment(0.0f, 0.0f, 0.0f);
    const lpl::render::Vec3f eye(0.0f, 0.0f, 3.0f);
    const lpl::render::Vec3f sky(0.12f, 0.14f, 0.18f);

    gold.albedo = lpl::render::Vec3f(1.0f, 0.77f, 0.34f);
    gold.metallic = 1.0f;
    gold.roughness = 0.25f;
    plastic.albedo = lpl::render::Vec3f(0.2f, 0.6f, 0.9f);
    plastic.metallic = 0.0f;
    plastic.roughness = 0.6f;
    key.type = lpl::render::LightType::Directional;
    key.direction = lpl::render::Vec3f(-0.5f, -0.8f, -0.6f);
    key.intensity = 3.0f;

    const lpl::core::u32 goldReinhard =
        lpl::render::pbrShadeToRgb(gold, &key, 1u, normal, fragment, eye, sky, lpl::render::ToneMap::Reinhard);
    const lpl::core::u32 goldAces =
        lpl::render::pbrShadeToRgb(gold, &key, 1u, normal, fragment, eye, sky, lpl::render::ToneMap::Aces);
    const lpl::core::u32 plasticAces =
        lpl::render::pbrShadeToRgb(plastic, &key, 1u, normal, fragment, eye, sky, lpl::render::ToneMap::Aces);

    test.check(goldReinhard != 0u && goldAces != 0u, "gold is lit");
    test.check(goldAces != plasticAces, "metal and plastic shade differently");
    test.check(goldReinhard != goldAces, "and so do the two tone maps");

    test.measureHexadecimal("gold_reinhard", goldReinhard);
    test.measureHexadecimal("gold_aces", goldAces);
    test.measureHexadecimal("plastic_aces", plasticAces);
}
