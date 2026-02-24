/**
 * @file TestMetrics.cpp
 * @brief Unit tests for bci::metric (Signal, Neural, Stability).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "lpl/bci/metric/NeuralMetric.hpp"
#include "lpl/bci/metric/SignalMetric.hpp"
#include "lpl/bci/metric/StabilityMetric.hpp"

using namespace bci::metric;
using Catch::Matchers::WithinAbs;

TEST_CASE("SignalMetric computes Schumacher ratio", "[metric][signal]")
{
    SignalMetricConfig config;
    config.sampleRate = 250.0f;
    config.fftSize = 256;

    SignalMetric metric(config);

    std::vector<std::vector<float>> psd(2, std::vector<float>(129, 1.0f));

    auto results = metric.compute(psd);
    REQUIRE(results.size() == 2);

    for (const auto& r : results) {
        REQUIRE(r.thetaPower > 0.0f);
        REQUIRE(r.alphaPower > 0.0f);
        REQUIRE(r.betaPower > 0.0f);
        REQUIRE(r.ratio > 0.0f);
    }
}

TEST_CASE("SignalMetric mean is average of channel ratios", "[metric][signal]")
{
    SignalMetric metric;

    std::vector<std::vector<float>> psd(1, std::vector<float>(129, 1.0f));

    auto results = metric.compute(psd);
    float mean = metric.computeMean(psd);

    REQUIRE_THAT(static_cast<double>(mean), WithinAbs(static_cast<double>(results[0].ratio), 1e-5));
}

TEST_CASE("NeuralMetric normalizes to [0,1]", "[metric][neural]")
{
    NeuralMetric metric;

    std::vector<ChannelBaseline> baselines = {
        {.alpha = {.mean = 10.0f, .stdDev = 2.0f}, .beta = {.mean = 5.0f, .stdDev = 1.0f}},
        {.alpha = {.mean = 10.0f, .stdDev = 2.0f}, .beta = {.mean = 5.0f, .stdDev = 1.0f}},
    };
    metric.setBaselines(baselines);

    std::vector<float> alpha = {10.0f, 10.0f};
    std::vector<float> beta  = {5.0f, 5.0f};

    auto state = metric.compute(alpha, beta);
    REQUIRE(state.channelAlpha.size() == 2);

    for (const float v : state.channelAlpha) {
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
    }

    REQUIRE_THAT(static_cast<double>(state.alphaPower), WithinAbs(0.5, 0.1));
}

TEST_CASE("NeuralMetric clamps extremes", "[metric][neural]")
{
    NeuralMetric metric;

    std::vector<ChannelBaseline> baselines = {
        {.alpha = {.mean = 10.0f, .stdDev = 1.0f}, .beta = {.mean = 5.0f, .stdDev = 1.0f}},
    };
    metric.setBaselines(baselines);

    std::vector<float> highAlpha = {100.0f};
    std::vector<float> lowBeta   = {-100.0f};

    auto state = metric.compute(highAlpha, lowBeta);

    REQUIRE_THAT(static_cast<double>(state.channelAlpha[0]), WithinAbs(1.0, 1e-5));
    REQUIRE_THAT(static_cast<double>(state.channelBeta[0]), WithinAbs(0.0, 1e-5));
}

TEST_CASE("StabilityMetric first update returns stable", "[metric][stability]")
{
    StabilityMetric metric;

    Eigen::MatrixXf spd(2, 2);
    spd << 2, 0.5f,
           0.5f, 3;

    auto result = metric.update(spd);
    REQUIRE(result.has_value());
    REQUIRE(result->isStable);
    REQUIRE_THAT(static_cast<double>(result->currentDistance), WithinAbs(0.0, 1e-5));
}

TEST_CASE("StabilityMetric detects instability", "[metric][stability]")
{
    StabilityConfig config;
    config.historySize = 5;
    config.stableThreshold = 0.01f;
    StabilityMetric metric(config);

    Eigen::MatrixXf a(2, 2);
    a << 2, 0,
         0, 2;

    Eigen::MatrixXf b(2, 2);
    b << 10, 0,
         0,  10;

    auto res1 = metric.update(a);
    REQUIRE(res1.has_value());
    auto result = metric.update(b);

    REQUIRE(result.has_value());
    REQUIRE(result->currentDistance > 0.0f);
}

TEST_CASE("StabilityMetric builds history", "[metric][stability]")
{
    StabilityMetric metric;

    Eigen::MatrixXf spd(2, 2);
    spd << 3, 1,
           1, 3;

    auto res1 = metric.update(spd);
    REQUIRE(res1.has_value());
    auto res2 = metric.update(spd);
    REQUIRE(res2.has_value());
    auto res3 = metric.update(spd);
    REQUIRE(res3.has_value());

    REQUIRE(metric.historySize() == 2);
}
