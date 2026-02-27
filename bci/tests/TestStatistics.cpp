/**
 * @file TestStatistics.cpp
 * @brief Unit tests for bci::math::Statistics.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "lpl/bci/math/Statistics.hpp"

namespace lpl::bci {

using namespace bci::math;
using Catch::Matchers::WithinAbs;

TEST_CASE("Statistics::integratePsd sums bins inclusively", "[math][statistics]")
{
    const std::vector<float> psd = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    SECTION("full range")
    {
        REQUIRE_THAT(Statistics::integratePsd(psd, 0, 4), WithinAbs(15.0f, 1e-5f));
    }

    SECTION("partial range")
    {
        REQUIRE_THAT(Statistics::integratePsd(psd, 1, 3), WithinAbs(9.0f, 1e-5f));
    }

    SECTION("single bin")
    {
        REQUIRE_THAT(Statistics::integratePsd(psd, 2, 2), WithinAbs(3.0f, 1e-5f));
    }

    SECTION("out of range returns zero")
    {
        REQUIRE_THAT(Statistics::integratePsd(psd, 3, 10), WithinAbs(0.0f, 1e-5f));
    }

    SECTION("empty span returns zero")
    {
        REQUIRE_THAT(Statistics::integratePsd({}, 0, 0), WithinAbs(0.0f, 1e-5f));
    }
}

TEST_CASE("Statistics::hzToBin converts frequency to bin index", "[math][statistics]")
{
    REQUIRE(Statistics::hzToBin(10.0f, 250.0f, 256) == 10);
    REQUIRE(Statistics::hzToBin(0.0f, 250.0f, 256) == 0);
    REQUIRE(Statistics::hzToBin(125.0f, 250.0f, 256) == 128);
}

TEST_CASE("Statistics::slidingWindowRms computes trailing RMS", "[math][statistics]")
{
    const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    SECTION("window smaller than data")
    {
        const float rms = Statistics::slidingWindowRms(data, 3);
        const float expected = std::sqrt((9.0f + 16.0f + 25.0f) / 3.0f);
        REQUIRE_THAT(rms, WithinAbs(expected, 1e-5f));
    }

    SECTION("window larger than data uses all data")
    {
        const float rms = Statistics::slidingWindowRms(data, 100);
        const float expected = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f + 25.0f) / 5.0f);
        REQUIRE_THAT(rms, WithinAbs(expected, 1e-5f));
    }

    SECTION("empty data returns zero")
    {
        REQUIRE_THAT(Statistics::slidingWindowRms({}, 5), WithinAbs(0.0f, 1e-5f));
    }
}

TEST_CASE("Statistics::computeBaseline computes mean and stddev", "[math][statistics]")
{
    SECTION("uniform data has zero stddev")
    {
        const std::vector<float> data = {5.0f, 5.0f, 5.0f, 5.0f};
        auto bl = Statistics::computeBaseline(data);
        REQUIRE_THAT(bl.mean, WithinAbs(5.0f, 1e-5f));
        REQUIRE_THAT(bl.stdDev, WithinAbs(0.0f, 1e-5f));
    }

    SECTION("known distribution")
    {
        const std::vector<float> data = {2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f};
        auto bl = Statistics::computeBaseline(data);
        REQUIRE_THAT(bl.mean, WithinAbs(5.0f, 1e-5f));
        REQUIRE_THAT(bl.stdDev, WithinAbs(2.0f, 1e-2f));
    }

    SECTION("empty data returns zeros")
    {
        auto bl = Statistics::computeBaseline({});
        REQUIRE_THAT(bl.mean, WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(bl.stdDev, WithinAbs(0.0f, 1e-5f));
    }
}

} // namespace lpl::bci
