#include <lpl/render/RayTracer.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(ray_tracer);

namespace {

constexpr lpl::core::u32 kWidth = 96u;
constexpr lpl::core::u32 kHeight = 72u;

/** The traced image, static because a kernel stack cannot hold it. */
lpl::core::u32 gImage[kWidth * kHeight];

} // namespace

/**
 * @brief Gate P6 ray tracing: the reference scene, traced with three bounces, gives the same image.
 */
LPL_TEST(reference_scene_traces_the_same_image)
{
    const auto traced = lpl::render::rayTraceScene(gImage, kWidth, kHeight, 3u);

    test.check(traced.hit_count > 0u && traced.hit_count < kWidth * kHeight, "rays hit the scene, and not every one");

    test.measure("hits", traced.hit_count);
    test.measureHexadecimal("image_signature", traced.image_signature);
}
