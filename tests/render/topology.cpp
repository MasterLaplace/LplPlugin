#include <lpl/render/Topology.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(topology);

namespace {

[[nodiscard]] lpl::math::Fixed32 units(lpl::core::i32 value) { return lpl::math::Fixed32::fromInt(value); }

} // namespace

/**
 * @brief Gate P6 topology: a Catmull-Rom loop through a square and its apex, a saddle, and a
 *        Delaunay triangulation, all in Fixed32.
 */
LPL_TEST(curves_surfaces_and_triangulations_fold_the_same)
{
    const lpl::math::Fixed32 loopPoints[5][3] = {
        {units(-2), units(0), units(-2)},
        {units(2),  units(0), units(-2)},
        {units(2),  units(0), units(2) },
        {units(-2), units(0), units(2) },
        {units(0),  units(3), units(0) },
    };
    const lpl::math::Fixed32 cloud[6][3] = {
        {units(0), units(0), units(0)},
        {units(4), units(0), units(0)},
        {units(4), units(4), units(0)},
        {units(0), units(4), units(0)},
        {units(2), units(2), units(0)},
        {units(1), units(3), units(0)},
    };
    const auto loop = lpl::render::tessellateCatmullLoop(loopPoints, 5u, 8u);
    const auto saddle = lpl::render::tessellateSaddle(16u);
    const auto triangulation = lpl::render::delaunay2D(cloud, 6u);

    test.check(loop.sample_count == 5u * 8u, "the loop has eight samples per control point");
    test.check(saddle.sample_count == 17u * 17u, "a saddle of resolution 16 has 17 by 17 vertices");
    test.check(triangulation.triangle_count > 0u, "the cloud triangulates");

    test.measureHexadecimal("loop_signature", loop.sample_signature);
    test.measureHexadecimal("saddle_signature", saddle.sample_signature);
    test.measure("triangles", triangulation.triangle_count);
    test.measureHexadecimal("triangle_signature", triangulation.triangle_signature);
}
