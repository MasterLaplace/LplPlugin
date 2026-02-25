#ifdef __unix__
#include <catch2/catch_test_macros.hpp>
#include "source/serial/SerialPort.hpp"

#include <pty.h>
#include <unistd.h>
#include <array>
#include <string>

using namespace bci::source;

TEST_CASE("SerialPort open/read/write/close", "[serial]")
{
    int master, slave;
    char slaveName[100];
    REQUIRE(openpty(&master, &slave, slaveName, nullptr, nullptr) == 0);

    SerialPort port;
    SerialConfig cfg{.portPath = slaveName, .baudRate = 115200};
    auto res = port.open(cfg);
    REQUIRE(res.has_value());
    REQUIRE(port.isOpen());

    std::string msg = "hello";
    auto w = port.write(std::span((const uint8_t *)msg.data(), msg.size()));
    REQUIRE(w.has_value());
    REQUIRE(w.value() == msg.size());

    std::array<char, 10> buf;
    auto n = ::read(master, buf.data(), buf.size());
    REQUIRE(n == (int)msg.size());
    REQUIRE(std::string(buf.data(), n) == msg);

    // test read via SerialPort by writing to master
    std::string msg2 = "world";
    ::write(master, msg2.data(), msg2.size());
    std::array<uint8_t, 10> buf2;
    auto r2 = port.read(buf2);
    REQUIRE(r2.has_value());
    REQUIRE(r2.value() == msg2.size());
    REQUIRE(std::string((char *)buf2.data(), r2.value()) == msg2);

    port.close();
    REQUIRE(!port.isOpen());

    close(master);
    close(slave);
}

TEST_CASE("SerialPort open fails on invalid path", "[serial]")
{
    SerialPort port;
    SerialConfig cfg{.portPath = "/nonexistent"};
    auto res = port.open(cfg);
    REQUIRE_FALSE(res.has_value());
}
#endif
