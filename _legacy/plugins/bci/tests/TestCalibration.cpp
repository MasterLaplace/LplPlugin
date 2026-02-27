/**
 * @file TestCalibration.cpp
 * @brief Unit tests for bci::calibration::Calibration state machine.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "calibration/Calibration.hpp"

using namespace bci::calibration;

TEST_CASE("Calibration starts in Idle state", "[calibration]")
{
    Calibration cal;
    REQUIRE(cal.state() == CalibrationState::kIdle);
    REQUIRE(cal.trialCount() == 0);
}

TEST_CASE("Calibration transitions Idle → Calibrating → Ready", "[calibration]")
{
    CalibrationConfig config;
    config.channelCount = 2;
    config.requiredTrials = 3;

    Calibration cal(config);

    SECTION("start transitions to Calibrating")
    {
        auto result = cal.start();
        REQUIRE(result.has_value());
        REQUIRE(cal.state() == CalibrationState::kCalibrating);
    }

    SECTION("adding trials transitions to Ready when complete")
    {
        auto res = cal.start();
        REQUIRE(res.has_value());

        std::vector<float> alpha = {1.0f, 2.0f};
        std::vector<float> beta  = {3.0f, 4.0f};

        auto res1 = cal.addTrial(alpha, beta);
        REQUIRE(res1.has_value());
        REQUIRE(cal.trialCount() == 1);

        auto res2 = cal.addTrial(alpha, beta);
        REQUIRE(res2.has_value());
        REQUIRE(cal.trialCount() == 2);

        auto res3 = cal.addTrial(alpha, beta);
        REQUIRE(res3.has_value());
        REQUIRE(cal.state() == CalibrationState::kReady);
    }
}

TEST_CASE("Calibration rejects operations in wrong state", "[calibration]")
{
    Calibration cal;

    SECTION("addTrial fails when idle")
    {
        std::vector<float> data = {1.0f};
        auto result = cal.addTrial(data, data);
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("start fails when already calibrating")
    {
        auto res_init = cal.start();
        REQUIRE(res_init.has_value());
        auto result = cal.start();
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("baselines unavailable before completion")
    {
        auto result = cal.baselines();
        REQUIRE_FALSE(result.has_value());
    }
}

TEST_CASE("Calibration produces valid baselines", "[calibration]")
{
    CalibrationConfig config;
    config.channelCount = 1;
    config.requiredTrials = 4;

    Calibration cal(config);
    auto res = cal.start();
    REQUIRE(res.has_value());

    std::vector<float> alpha = {10.0f};
    std::vector<float> beta  = {5.0f};

    for (std::size_t i = 0; i < 4; ++i) {
        auto res_trial = cal.addTrial(alpha, beta);
        REQUIRE(res_trial.has_value());
    }

    auto baselines = cal.baselines();
    REQUIRE(baselines.has_value());
    REQUIRE(baselines->size() == 1);

    REQUIRE_THAT(static_cast<double>((*baselines)[0].alpha.mean),
        Catch::Matchers::WithinAbs(10.0, 1e-5));

    REQUIRE_THAT(static_cast<double>((*baselines)[0].alpha.stdDev),
        Catch::Matchers::WithinAbs(0.0, 1e-5));
}

TEST_CASE("Calibration observer is notified", "[calibration]")
{
    CalibrationConfig config;
    config.channelCount = 1;
    config.requiredTrials = 1;

    Calibration cal(config);

    int callCount = 0;
    cal.onStateChange([&](CalibrationState, CalibrationState) {
        ++callCount;
    });

    auto res = cal.start();
    REQUIRE(res.has_value());
    REQUIRE(callCount == 1);

    std::vector<float> alpha = {1.0f};
    std::vector<float> beta = {2.0f};
    auto res_trial = cal.addTrial(alpha, beta);
    REQUIRE(res_trial.has_value());
    REQUIRE(callCount == 2);
}

TEST_CASE("Calibration reset returns to Idle", "[calibration]")
{
    CalibrationConfig config;
    config.channelCount = 1;
    config.requiredTrials = 1;

    Calibration cal(config);
    auto res_start = cal.start();
    REQUIRE(res_start.has_value());
    std::vector<float> alpha = {1.0f};
    std::vector<float> beta = {2.0f};
    auto res_trial = cal.addTrial(alpha, beta);
    REQUIRE(res_trial.has_value());
    REQUIRE(cal.state() == CalibrationState::kReady);

    cal.reset();
    REQUIRE(cal.state() == CalibrationState::kIdle);
    REQUIRE(cal.trialCount() == 0);
}
