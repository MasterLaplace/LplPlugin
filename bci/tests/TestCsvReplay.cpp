/**
 * @file TestCsvReplay.cpp
 * @brief Unit tests for bci::source::CsvReplaySource.
 */

#include <catch2/catch_test_macros.hpp>

#include "lpl/bci/source/CsvReplaySource.hpp"

#include <filesystem>
#include <fstream>

namespace lpl::bci {

using namespace bci::source;

namespace {

class TempCsvFile {
public:
    explicit TempCsvFile(const std::string& content)
        : _path(std::filesystem::temp_directory_path() / "lpl_bci_test.csv")
    {
        std::ofstream ofs(_path);
        ofs << content;
    }

    ~TempCsvFile() { std::filesystem::remove(_path); }

    [[nodiscard]] std::string path() const { return _path.string(); }

private:
    std::filesystem::path _path;
};

} // namespace

TEST_CASE("CsvReplaySource reads CSV data", "[source][csv]")
{
    TempCsvFile csv("1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n");

    CsvReplayConfig config;
    config.filePath = csv.path();
    config.channelCount = 3;
    config.burstMode = true;

    CsvReplaySource source(config);
    auto startResult = source.start();
    REQUIRE(startResult.has_value());

    auto info = source.info();
    REQUIRE(info.channelCount == 3);

    std::vector<bci::Sample> buffer(3);
    auto readResult = source.read(buffer);

    REQUIRE(readResult.has_value());
    REQUIRE(*readResult == 3);
    REQUIRE(buffer[0].channels[0] == 1.0f);
    REQUIRE(buffer[1].channels[1] == 5.0f);

    source.stop();
}

TEST_CASE("CsvReplaySource loops on exhaustion", "[source][csv]")
{
    TempCsvFile csv("1.0,2.0\n3.0,4.0\n");

    CsvReplayConfig config;
    config.filePath = csv.path();
    config.channelCount = 2;
    config.burstMode = true;

    CsvReplaySource source(config);
    auto res = source.start();
    REQUIRE(res.has_value());

    std::vector<bci::Sample> buffer(4);
    auto result1 = source.read(buffer);
    REQUIRE(result1.has_value());
    REQUIRE(*result1 == 2);

    auto result2 = source.read(buffer);
    REQUIRE(result2.has_value());
    REQUIRE(*result2 == 2);

    source.stop();
}

TEST_CASE("CsvReplaySource rejects invalid file", "[source][csv]")
{
    CsvReplayConfig config;
    config.filePath = "/nonexistent/file.csv";
    config.channelCount = 2;

    CsvReplaySource source(config);
    auto result = source.start();

    REQUIRE_FALSE(result.has_value());
}

} // namespace lpl::bci
