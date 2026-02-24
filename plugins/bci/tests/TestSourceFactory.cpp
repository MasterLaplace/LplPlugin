/**
 * @file TestSourceFactory.cpp
 * @brief Unit tests for lpl::bci::source::SourceFactory.
 */

#include <catch2/catch_test_macros.hpp>

#include "lpl/bci/source/SourceFactory.hpp"

using namespace lpl::bci;
using namespace lpl::bci::source;

TEST_CASE("SourceFactory creates SyntheticSource", "[source][factory]")
{
    SourceConfig config;
    config.mode = AcquisitionMode::kSynthetic;
    config.channelCount = 4;
    config.sampleRate = 250.0f;

    auto result = SourceFactory::create(config);
    REQUIRE(result.has_value());

    auto info = (*result)->info();
    REQUIRE(info.channelCount == 4);
}

TEST_CASE("SourceFactory creates CsvReplaySource", "[source][factory]")
{
    SourceConfig config;
    config.mode = AcquisitionMode::kCsvReplay;
    config.channelCount = 2;
    config.csvFilePath = "/nonexistent/file.csv";

    auto result = SourceFactory::create(config);
    REQUIRE(result.has_value());
}

TEST_CASE("SourceFactory rejects CsvReplay without path", "[source][factory]")
{
    SourceConfig config;
    config.mode = AcquisitionMode::kCsvReplay;
    config.csvFilePath = "";

    auto result = SourceFactory::create(config);
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("SourceFactory creates LslSource", "[source][factory]")
{
    SourceConfig config;
    config.mode = AcquisitionMode::kLsl;
    config.channelCount = 8;
    config.lslStreamName = "TestStream";

    auto result = SourceFactory::create(config);
    REQUIRE(result.has_value());
}
