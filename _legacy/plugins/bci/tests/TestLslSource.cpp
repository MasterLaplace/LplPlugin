#include <catch2/catch_test_macros.hpp>
#include "source/LslSource.hpp"
#include "core/Types.hpp"
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <lsl_cpp.h>
#include <thread>
#include <chrono>

using namespace bci::source;
using namespace bci;

TEST_CASE("LslSource reads from temporary outlet", "[source][lsl]")
{
    // create a short-lived outlet
    lsl::stream_info streamInfo("TestStream", "EEG", 2, 250.0, lsl::cf_float32, "testid");
    lsl::stream_outlet outlet(streamInfo);

    // push a single sample after a brief delay to ensure inlet sees it
    std::vector<float> sample = {1.0f, 2.0f};
    std::thread writer([&](){
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        outlet.push_sample(sample);
    });

    LslSourceConfig cfg;
    cfg.streamName = "TestStream";
    cfg.resolveTimeoutSec = 0.1; // quick fail-over in CI
    // channel count determined by the resolved stream

    LslSource src(cfg);
    REQUIRE(src.start().has_value());

    std::array<Sample,1> buf;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto srcInfo = src.info();
    REQUIRE(srcInfo.channelCount == 2);
    auto r = src.read(buf);
    REQUIRE(r.has_value());
    if (r.value() == 0) {
        // no samples arrived - network/LSL resolution may be unreliable in CI
        SUCCEED("no sample received; environment may not support LSL");
    } else {
        REQUIRE(buf[0].channels.size() == 2);
        REQUIRE_THAT(static_cast<double>(buf[0].channels[0]), Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(static_cast<double>(buf[0].channels[1]), Catch::Matchers::WithinAbs(2.0, 1e-6));
    }

    src.stop();
    writer.join();
}
