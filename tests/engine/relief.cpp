#include <lpl/engine/ReliefParity.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(relief);

/**
 * @brief Gate P21 relief: a world standing on measured ground comes out the same on both targets.
 *
 * @details The signatures alone would agree just as well if the survey were never read, so the
 *          counts show that it was, and the world that ignores it must differ.
 */
LPL_TEST(world_stands_on_measured_ground)
{
    const lpl::engine::ReliefFoldResult measured = lpl::engine::foldReliefParity();
    const lpl::engine::ReliefFoldResult invented = lpl::engine::foldInventedParity();

    test.check(measured.measuredCells > 0u, "measured ground was stood on");
    test.check(measured.inventedCells > 0u, "invented ground was reached beyond it");
    test.check(measured.blendedCells > 0u, "and the border band was crossed");
    test.check(measured.seaCells > 0u &&
                   measured.seaCells < measured.measuredCells + measured.inventedCells + measured.blendedCells,
               "the coast is inside the world");

    test.check(measured.heightSignature != invented.heightSignature, "the ground differs from the invented world's");
    test.check(measured.coastSignature != invented.coastSignature, "the coastline differs");
    test.check(measured.walkSignature != invented.walkSignature, "and the body goes somewhere else");
    test.check(measured.sampleSignature == invented.sampleSignature, "but the survey itself is one survey");

    test.check(measured.walkSteps >= 8u, "the body walked more than a handful of steps");
    test.check(measured.descended > 0, "and ended lower than it started");

    test.measureHexadecimal("sample_signature", measured.sampleSignature);
    test.measureHexadecimal("height_signature", measured.heightSignature);
    test.measureHexadecimal("walk_signature", measured.walkSignature);
    test.measureHexadecimal("coast_signature", measured.coastSignature);
    test.measure("measured_cells", measured.measuredCells);
    test.measure("invented_cells", measured.inventedCells);
    test.measure("blended_cells", measured.blendedCells);
    test.measure("sea_cells", measured.seaCells);
    test.measure("walk_steps", measured.walkSteps);
    test.measure("descended", measured.descended);
    test.measureHexadecimal("invented_height_signature", invented.heightSignature);
    test.measureHexadecimal("invented_walk_signature", invented.walkSignature);
}
