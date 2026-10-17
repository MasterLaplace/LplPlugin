#include <catch2/catch_test_macros.hpp>

#include "lpl/bci/source/SourceFactory.hpp"

namespace lpl::bci {

TEST_CASE("SourceFactory creates SyntheticSource", "[source][factory]")
{
    lpl::bci::source::SourceConfig config;
    config.mode = AcquisitionMode::kSynthetic;
    config.channelCount = 4;
    config.sampleRate = 250.0f;

    auto result = lpl::bci::source::SourceFactory::create(config);
    REQUIRE(result.has_value());

    auto info = (*result)->info();
    REQUIRE(info.channelCount == 4);
}

TEST_CASE("SourceFactory creates CsvReplaySource", "[source][factory]")
{
    lpl::bci::source::SourceConfig config;
    config.mode = AcquisitionMode::kCsvReplay;
    config.channelCount = 2;
    config.csvFilePath = "/nonexistent/file.csv";

    auto result = lpl::bci::source::SourceFactory::create(config);
    REQUIRE(result.has_value());
}

TEST_CASE("SourceFactory rejects CsvReplay without path", "[source][factory]")
{
    lpl::bci::source::SourceConfig config;
    config.mode = AcquisitionMode::kCsvReplay;
    config.csvFilePath = "";

    auto result = lpl::bci::source::SourceFactory::create(config);
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("SourceFactory creates LslSource", "[source][factory]")
{
    lpl::bci::source::SourceConfig config;
    config.mode = AcquisitionMode::kLsl;
    config.channelCount = 8;
    config.lslStreamName = "TestStream";

    auto result = lpl::bci::source::SourceFactory::create(config);
    REQUIRE(result.has_value());
}

} // namespace lpl::bci
