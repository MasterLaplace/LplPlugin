#include <lpl/math/Cordic.hpp>
#include <lpl/math/Mat4.hpp>
#include <lpl/render/Instancing.hpp>
#include <lpl/render/Lighting.hpp>
#include <lpl/render/Projection.hpp>
#include <lpl/render/RenderParity.hpp>
#include <lpl/render/SoftwareRasterizer.hpp>
#include <lpl/render/Texture.hpp>
#include <lpl/std/vector.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(render);

namespace {

constexpr lpl::core::u32 kWidth = 96u;
constexpr lpl::core::u32 kHeight = 64u;
constexpr lpl::core::u32 kViewportsWidth = 128u;
constexpr lpl::core::u32 kViewportsHeight = 96u;
constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kScreenWidth = 1280u;
constexpr lpl::core::u32 kScreenHeight = 800u;

/**
 * @brief Vertical field of view of both cameras: sixty degrees, in radians.
 */
constexpr lpl::core::f32 kFieldOfView = 1.04719755f;

/**
 * @brief Radius of the sphere around a unit cube, which the cull tests each instance against.
 */
constexpr lpl::core::f32 kUnitCubeCircumradius = 1.73205081f;

/**
 * @brief The corners of the unit cube gate P5 render projects, at plus and minus one.
 */
constexpr lpl::core::i32 kCubeCorners[8][3] = {
    {-1, -1, -1},
    {1,  -1, -1},
    {1,  1,  -1},
    {-1, 1,  -1},
    {-1, -1, 1 },
    {1,  -1, 1 },
    {1,  1,  1 },
    {-1, 1,  1 },
};

/**
 * @struct ProjectedCube
 * @brief The unit cube projected through a camera, as gate P5 render records it.
 */
struct ProjectedCube {
    lpl::core::u32 screenSignature{0u}; /**< FNV-1a fold of the eight floored screen positions. */
    lpl::core::u32 depthSignature{0u};  /**< FNV-1a fold of the eight quantized depths. */
    lpl::core::i32 firstCornerX{0};     /**< Floored screen X of the first corner. */
    lpl::core::i32 firstCornerY{0};     /**< Floored screen Y of the first corner. */
    lpl::core::u32 cornersInFront{0u};  /**< Corners in front of the camera, with w above zero. */
};

/**
 * @struct CulledGrid
 * @brief The instance grid culled against a camera, as gate P5 render records it.
 */
struct CulledGrid {
    lpl::core::u32 total{0u};            /**< Instances in the grid. */
    lpl::core::u32 visible{0u};          /**< Instances surviving the frustum cull. */
    lpl::core::u32 visibleSignature{0u}; /**< FNV-1a fold of the visible indices. */
};

/**
 * @struct ClipPosition
 * @brief A point in clip space, before the perspective divide.
 */
struct ClipPosition {
    lpl::core::f32 x; /**< Clip X. */
    lpl::core::f32 y; /**< Clip Y. */
    lpl::core::f32 z; /**< Clip Z. */
    lpl::core::f32 w; /**< The divisor of the perspective divide. */
};

[[nodiscard]] ClipPosition toClipSpace(const lpl::math::Mat4<lpl::core::f32> &viewProjection,
                                       const lpl::render::Vec3f &world)
{
    return {
        viewProjection(0, 0) * world.x + viewProjection(0, 1) * world.y + viewProjection(0, 2) * world.z +
            viewProjection(0, 3),
        viewProjection(1, 0) * world.x + viewProjection(1, 1) * world.y + viewProjection(1, 2) * world.z +
            viewProjection(1, 3),
        viewProjection(2, 0) * world.x + viewProjection(2, 1) * world.y + viewProjection(2, 2) * world.z +
            viewProjection(2, 3),
        viewProjection(3, 0) * world.x + viewProjection(3, 1) * world.y + viewProjection(3, 2) * world.z +
            viewProjection(3, 3),
    };
}

/**
 * @brief Projects the unit cube, turned about Y, through a camera at (0, 0, 5) looking at the
 *        origin, and folds its screen positions and depths.
 *
 * @details The corners and the turn are authored in Fixed32, through CORDIC: authoritative state.
 *          The rotation by that sine and cosine, the view, the projection and the divide run in
 *          float, which both targets compute bit for bit with SSE and no contraction.
 *
 * @param rotationAngle Turn about Y, Fixed32 radians.
 */
[[nodiscard]] ProjectedCube projectCube(lpl::math::Fixed32 rotationAngle)
{
    lpl::math::Fixed32 sine = lpl::math::Fixed32::fromInt(0);
    lpl::math::Fixed32 cosine = lpl::math::Fixed32::fromInt(0);

    lpl::math::Cordic::sincos(rotationAngle, sine, cosine);

    const auto view = lpl::math::Mat4<lpl::core::f32>::lookAt(lpl::render::Vec3f(0.0f, 0.0f, 5.0f),
                                                              lpl::render::Vec3f(0.0f, 0.0f, 0.0f),
                                                              lpl::render::Vec3f(0.0f, 1.0f, 0.0f));
    const lpl::core::f32 aspect =
        static_cast<lpl::core::f32>(kScreenWidth) / static_cast<lpl::core::f32>(kScreenHeight);
    const auto projection =
        lpl::render::perspectiveFov(lpl::math::Fixed32::fromFloat(kFieldOfView), aspect, 0.1f, 100.0f);
    const auto viewProjection = projection * view;
    const lpl::core::f32 cosineFloat = cosine.toFloat();
    const lpl::core::f32 sineFloat = sine.toFloat();
    ProjectedCube out{};
    lpl::core::u32 screenHash = kFnv1aOffsetBasis;
    lpl::core::u32 depthHash = kFnv1aOffsetBasis;

    for (lpl::core::u32 corner = 0u; corner < 8u; ++corner)
    {
        const lpl::core::f32 x = lpl::math::Fixed32::fromInt(kCubeCorners[corner][0]).toFloat();
        const lpl::core::f32 y = lpl::math::Fixed32::fromInt(kCubeCorners[corner][1]).toFloat();
        const lpl::core::f32 z = lpl::math::Fixed32::fromInt(kCubeCorners[corner][2]).toFloat();
        const lpl::render::Vec3f world(cosineFloat * x + sineFloat * z, y, -sineFloat * x + cosineFloat * z);
        const ClipPosition clip = toClipSpace(viewProjection, world);

        if (clip.w > 0.0f)
            ++out.cornersInFront;

        const lpl::core::f32 inverseW = (clip.w != 0.0f) ? (1.0f / clip.w) : 1.0f;
        const lpl::core::f32 ndcX = clip.x * inverseW;
        const lpl::core::f32 ndcY = clip.y * inverseW;
        const lpl::core::f32 ndcZ = clip.z * inverseW;
        const lpl::core::i32 screenX =
            static_cast<lpl::core::i32>((ndcX * 0.5f + 0.5f) * static_cast<lpl::core::f32>(kScreenWidth));
        const lpl::core::i32 screenY =
            static_cast<lpl::core::i32>((0.5f - ndcY * 0.5f) * static_cast<lpl::core::f32>(kScreenHeight));
        const lpl::core::i32 depth = static_cast<lpl::core::i32>(ndcZ * lpl::render::detail::kQ16FoldScale);

        screenHash = lpl::render::detail::fnv1aStep(screenHash, static_cast<lpl::core::u32>(screenX));
        screenHash = lpl::render::detail::fnv1aStep(screenHash, static_cast<lpl::core::u32>(screenY));
        depthHash = lpl::render::detail::fnv1aStep(depthHash, static_cast<lpl::core::u32>(depth));
        if (corner == 0u)
        {
            out.firstCornerX = screenX;
            out.firstCornerY = screenY;
        }
    }
    out.screenSignature = screenHash;
    out.depthSignature = depthHash;
    return out;
}

/**
 * @brief Culls a seven by seven grid of unit cubes, five units apart on the XZ plane, against a
 *        camera at (0, 8, 18) looking at the origin, and folds the indices it keeps.
 */
[[nodiscard]] CulledGrid cullInstanceGrid()
{
    lpl::render::InstanceSet set;

    for (lpl::core::i32 gridZ = -3; gridZ <= 3; ++gridZ)
    {
        for (lpl::core::i32 gridX = -3; gridX <= 3; ++gridX)
            set.add(lpl::math::Fixed32::fromInt(gridX * 5), lpl::math::Fixed32::fromInt(0),
                    lpl::math::Fixed32::fromInt(gridZ * 5), lpl::math::Fixed32::fromInt(1));
    }

    const auto view = lpl::math::Mat4<lpl::core::f32>::lookAt(lpl::render::Vec3f(0.0f, 8.0f, 18.0f),
                                                              lpl::render::Vec3f(0.0f, 0.0f, 0.0f),
                                                              lpl::render::Vec3f(0.0f, 1.0f, 0.0f));
    const lpl::core::f32 aspect =
        static_cast<lpl::core::f32>(kScreenWidth) / static_cast<lpl::core::f32>(kScreenHeight);
    const auto projection =
        lpl::render::perspectiveFov(lpl::math::Fixed32::fromFloat(kFieldOfView), aspect, 0.1f, 100.0f);
    const auto frustum = lpl::render::Frustum::fromViewProjection(projection * view);
    lpl::pmr::vector<lpl::core::u32> visible;
    CulledGrid out{};
    lpl::core::u32 hash = kFnv1aOffsetBasis;

    lpl::render::frustumCull(set, frustum, kUnitCubeCircumradius, visible);
    out.total = set.count();
    out.visible = static_cast<lpl::core::u32>(visible.size());
    for (lpl::core::u32 index = 0u; index < out.visible; ++index)
        hash = lpl::render::detail::fnv1aStep(hash, visible[index]);
    out.visibleSignature = hash;
    return out;
}

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
    const ProjectedCube straight = projectCube(lpl::math::Fixed32::fromInt(0));
    const ProjectedCube turned = projectCube(lpl::math::Fixed32::fromFloat(0.78539816f));

    test.check(straight.cornersInFront == 8u, "the eight corners are in front of the camera");
    test.check(straight.firstCornerX > 0 && straight.firstCornerX < 1280 && straight.firstCornerY > 0 &&
                   straight.firstCornerY < 800,
               "the first corner lands inside the viewport");
    test.check(turned.cornersInFront == 8u, "turned by a quarter pi, they are still in front");
    test.check(turned.screenSignature != straight.screenSignature, "and the turn moves them on screen");

    test.measureHexadecimal("screen_signature", straight.screenSignature);
    test.measureHexadecimal("depth_signature", straight.depthSignature);
    test.measure("first_corner_x", straight.firstCornerX);
    test.measure("first_corner_y", straight.firstCornerY);
    test.measureHexadecimal("turned_screen_signature", turned.screenSignature);
    test.measureHexadecimal("turned_depth_signature", turned.depthSignature);
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
    const CulledGrid culled = cullInstanceGrid();

    test.check(culled.total == 49u, "the grid holds 7 by 7 instances");
    test.check(culled.visible > 0u && culled.visible < culled.total, "the frustum keeps some and not all");

    test.measure("visible_instances", culled.visible);
    test.measureHexadecimal("visible_signature", culled.visibleSignature);
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
