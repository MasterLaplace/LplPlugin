#ifdef __unix__
#    include "lpl/bci/source/serial/SerialPort.hpp"
#    include <catch2/catch_test_macros.hpp>

#    include <array>
#    include <pty.h>
#    include <string>
#    include <unistd.h>

namespace lpl::bci {

/**
 * @brief Writes through the port and reads on the master side, then writes on the master side and
 *        reads through the port.
 */
TEST_CASE("SerialPort open/read/write/close", "[serial]")
{
    int master, slave;
    char slaveName[100];
    REQUIRE(openpty(&master, &slave, slaveName, nullptr, nullptr) == 0);

    lpl::bci::source::SerialPort port;
    lpl::bci::source::SerialConfig cfg{.portPath = slaveName, .baudRate = 115200};
    auto res = port.open(cfg);
    REQUIRE(res.has_value());
    REQUIRE(port.isOpen());

    std::string msg = "hello";
    auto w = port.write(std::span((const uint8_t *) msg.data(), msg.size()));
    REQUIRE(w.has_value());
    REQUIRE(w.value() == msg.size());

    std::array<char, 10> buf;
    auto n = ::read(master, buf.data(), buf.size());
    REQUIRE(n == (int) msg.size());
    REQUIRE(std::string(buf.data(), n) == msg);

    std::string msg2 = "world";
    ::write(master, msg2.data(), msg2.size());
    std::array<uint8_t, 10> buf2;
    auto r2 = port.read(buf2);
    REQUIRE(r2.has_value());
    REQUIRE(r2.value() == msg2.size());
    REQUIRE(std::string((char *) buf2.data(), r2.value()) == msg2);

    port.close();
    REQUIRE(!port.isOpen());

    close(master);
    close(slave);
}

TEST_CASE("SerialPort open fails on invalid path", "[serial]")
{
    lpl::bci::source::SerialPort port;
    lpl::bci::source::SerialConfig cfg{.portPath = "/nonexistent"};
    auto res = port.open(cfg);
    REQUIRE_FALSE(res.has_value());
}
#endif

} // namespace lpl::bci
