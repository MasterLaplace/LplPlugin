/**
 * @file TestPipeline.cpp
 * @brief Unit tests for lpl::bci::dsp::Pipeline and DSP stages.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "lpl/bci/dsp/BandExtractor.hpp"
#include "lpl/bci/dsp/FftProcessor.hpp"
#include "lpl/bci/dsp/Pipeline.hpp"
#include "lpl/bci/dsp/Windowing.hpp"

using namespace lpl::bci;
using namespace lpl::bci::dsp;

static SignalBlock makeConstantBlock(std::size_t samples, std::size_t channels, float value)
{
    SignalBlock block;
    block.sampleRate = 250.0f;
    block.channelCount = channels;
    block.data.resize(samples);
    for (auto& sample : block.data) {
        sample.assign(channels, value);
    }
    return block;
}

TEST_CASE("Pipeline processes empty stage list as identity", "[dsp][pipeline]")
{
    auto pipeline = PipelineBuilder().build();
    auto block = makeConstantBlock(256, 2, 1.0f);
    auto result = pipeline.process(block);

    REQUIRE(result.has_value());
    REQUIRE(result->data.size() == 256);
}

TEST_CASE("HannWindow applies window coefficients", "[dsp][windowing]")
{
    HannWindow window(256);
    auto block = makeConstantBlock(256, 1, 1.0f);
    auto result = window.process(block);

    REQUIRE(result.has_value());
    REQUIRE_THAT(
        static_cast<double>(result->data[0][0]),
        Catch::Matchers::WithinAbs(0.0, 1e-3));
    REQUIRE(result->data[128][0] > 0.9f);
}

TEST_CASE("HannWindow rejects mismatched block size", "[dsp][windowing]")
{
    HannWindow window(256);
    auto block = makeConstantBlock(128, 1, 1.0f);
    auto result = window.process(block);

    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("FftProcessor produces half-spectrum PSD", "[dsp][fft]")
{
    FftProcessor fft(256);
    auto block = makeConstantBlock(256, 2, 0.0f);

    for (std::size_t i = 0; i < 256; ++i) {
        block.data[i][0] = std::sin(2.0f * 3.14159265f * 10.0f * static_cast<float>(i) / 250.0f);
        block.data[i][1] = 0.0f;
    }

    auto result = fft.process(block);
    REQUIRE(result.has_value());
    REQUIRE(result->data.size() == 129);
    REQUIRE(result->data[0].size() == 2);
}

TEST_CASE("FftProcessor rejects non-power-of-two size", "[dsp][fft]")
{
    FftProcessor fft(200);
    auto block = makeConstantBlock(200, 1, 1.0f);
    auto result = fft.process(block);

    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("BandExtractor extracts correct frequency bands", "[dsp][band]")
{
    std::vector<FrequencyBand> bands = {
        {.low = 8.0f, .high = 13.0f},
        {.low = 13.0f, .high = 30.0f},
    };

    BandExtractor extractor(bands, 250.0f, 256);

    SignalBlock block = makeConstantBlock(256, 1, 0.0f);
    block.data.assign(129, std::vector<float>(1, 1.0f));

    auto result = extractor.process(block);
    REQUIRE(result.has_value());
    REQUIRE(result->data.size() == 2);
}
