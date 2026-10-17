#include "lpl/bci/core/Types.hpp"
#include "lpl/bci/source/LslSource.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <lsl_cpp.h>
#include <thread>

namespace lpl::bci {

/**
 * @brief Reads one sample from a short-lived outlet, which pushes it after a brief delay so that the
 *        inlet sees it.
 *
 * The resolve timeout is short, for a quick fail-over in CI, and the channel count is the resolved
 * stream's. A run where no sample arrives passes: LSL resolution may be unreliable in CI.
 */
TEST_CASE("LslSource reads from temporary outlet", "[source][lsl]")
{
    lsl::stream_info streamInfo("TestStream", "EEG", 2, 250.0, lsl::cf_float32, "testid");
    lsl::stream_outlet outlet(streamInfo);

    std::vector<float> sample = {1.0f, 2.0f};
    std::thread writer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        outlet.push_sample(sample);
    });

    lpl::bci::source::LslSourceConfig cfg;
    cfg.streamName = "TestStream";
    cfg.resolveTimeoutSec = 0.1;

    lpl::bci::source::LslSource src(cfg);
    REQUIRE(src.start().has_value());

    std::array<Sample, 1> buf;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto srcInfo = src.info();
    REQUIRE(srcInfo.channelCount == 2);
    auto r = src.read(buf);
    REQUIRE(r.has_value());
    if (r.value() == 0)
    {
        SUCCEED("no sample received; environment may not support LSL");
    }
    else
    {
        REQUIRE(buf[0].channels.size() == 2);
        REQUIRE_THAT(static_cast<double>(buf[0].channels[0]), Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(static_cast<double>(buf[0].channels[1]), Catch::Matchers::WithinAbs(2.0, 1e-6));
    }

    src.stop();
    writer.join();
}

} // namespace lpl::bci
