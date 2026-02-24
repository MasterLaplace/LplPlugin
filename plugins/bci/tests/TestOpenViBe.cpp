/**
 * @file TestOpenViBe.cpp
 * @brief Unit tests for lpl::bci::openvibe processors.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "lpl/bci/openvibe/MuscleRelaxationBox.hpp"
#include "lpl/bci/openvibe/StabilityMonitorBox.hpp"

using namespace lpl::bci::openvibe;
using Catch::Matchers::WithinAbs;

TEST_CASE("MuscleRelaxationBox computes gamma ratio", "[openvibe][muscle]")
{
    MuscleRelaxationConfig config;
    config.sampleRate = 250.0f;
    config.fftSize = 256;
    config.alertThreshold = 0.5f;

    MuscleRelaxationBox box(config);

    std::vector<std::vector<float>> psd(2, std::vector<float>(129, 0.01f));

    auto result = box.compute(psd);
    REQUIRE(result.gammaRatio >= 0.0f);
}

TEST_CASE("MuscleRelaxationBox alerts on high gamma", "[openvibe][muscle]")
{
    MuscleRelaxationConfig config;
    config.alertThreshold = 0.1f;

    MuscleRelaxationBox box(config);

    std::vector<std::vector<float>> psd(1, std::vector<float>(129, 10.0f));

    auto result = box.compute(psd);
    REQUIRE(result.isAlert);
}

TEST_CASE("StabilityMonitorBox initial update returns default stability", "[openvibe][stability]")
{
    StabilityMonitorBox box;

    Eigen::MatrixXf spd(2, 2);
    spd << 3, 1,
           1, 3;

    auto result = box.update(spd);
    REQUIRE(result.has_value());
    REQUIRE_THAT(static_cast<double>(result->smoothedStability), WithinAbs(0.5, 0.1));
}

TEST_CASE("StabilityMonitorBox stability converges for identical matrices", "[openvibe][stability]")
{
    StabilityMonitorBox box;

    Eigen::MatrixXf spd(2, 2);
    spd << 3, 1,
           1, 3;

    for (int i = 0; i < 20; ++i) {
        auto res = box.update(spd);
        REQUIRE(res.has_value());
    }

    REQUIRE(box.stability() > 0.5f);
}

TEST_CASE("StabilityMonitorBox reset returns to default", "[openvibe][stability]")
{
    StabilityMonitorBox box;

    Eigen::MatrixXf spd(2, 2);
    spd << 3, 1,
           1, 3;

    auto res = box.update(spd);
    REQUIRE(res.has_value());
    box.reset();

    REQUIRE_THAT(static_cast<double>(box.stability()), WithinAbs(0.5, 1e-5));
}
