/**
 * @file TestOpenBciSource.cpp
 * @brief BCI component: TestOpenBciSource.cpp
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#ifdef __unix__
#include <catch2/catch_test_macros.hpp>
#include "lpl/bci/source/OpenBciSource.hpp"
#include "lpl/bci/core/Constants.hpp"
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <pty.h>
#include <unistd.h>
#include <array>
#include <thread>
#include <chrono>

namespace lpl::bci {

using namespace bci::source;
using namespace bci;

TEST_CASE("OpenBciSource reads one packet", "[source][openbci]")
{
    int master, slave;
    char slaveName[100];
    REQUIRE(openpty(&master, &slave, slaveName, nullptr, nullptr) == 0);

    OpenBciConfig cfg;
    cfg.port = slaveName;
    cfg.channelCount = 2;
    cfg.baudRate = kCytonBaudRate;

    OpenBciSource src(cfg);
    REQUIRE(src.start().has_value());

    std::array<uint8_t, kCytonPacketSize> pkt{};
    pkt[0] = kCytonHeaderByte;
    pkt[1] = 123;
    // remaining bytes left zero
    pkt[kCytonPacketSize - 1] = kCytonFooterByte;

    ::write(master, pkt.data(), pkt.size());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::array<Sample, 1> buf;
    auto r = src.read(buf);
    REQUIRE(r.has_value());
    REQUIRE(r.value() == 1);
    REQUIRE(buf[0].channels.size() == 2);
    REQUIRE_THAT(static_cast<double>(buf[0].channels[0]), Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(static_cast<double>(buf[0].channels[1]), Catch::Matchers::WithinAbs(0.0, 1e-6));

    src.stop();
    close(master);
    close(slave);
}
#endif

} // namespace lpl::bci
