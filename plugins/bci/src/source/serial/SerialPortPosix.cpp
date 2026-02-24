/**
 * @file SerialPortPosix.cpp
 * @brief POSIX implementation of the SerialPort abstraction.
 * @author MasterLaplace
 *
 * Uses termios for configuration, O_RDWR | O_NOCTTY for opening,
 * and TIOCEXCL for exclusive access. Ported and hardened from the
 * V1 OpenBCIDriver serial setup.
 */

#include "lpl/bci/source/serial/SerialPort.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

namespace bci::source {

struct SerialPort::Impl {
    int fd = -1;
};

SerialPort::SerialPort()
    : _impl(new Impl)
{
}

SerialPort::~SerialPort()
{
    close();
    delete _impl;
}

SerialPort::SerialPort(SerialPort &&other) noexcept
    : _impl(other._impl)
{
    other._impl = new Impl;
}

SerialPort &SerialPort::operator=(SerialPort &&other) noexcept
{
    if (this == &other) {
        return *this;
    }
    close();
    delete _impl;
    _impl = other._impl;
    other._impl = new Impl;
    return *this;
}

ExpectedVoid SerialPort::open(const SerialConfig &config)
{
    if (_impl->fd >= 0) {
        return std::unexpected(
            Error::make(ErrorCode::kAlreadyRunning, "Serial port already open"));
    }

    _impl->fd = ::open(config.portPath.c_str(), O_RDWR | O_NOCTTY);
    if (_impl->fd < 0) {
        return std::unexpected(
            Error::make(ErrorCode::kSerialPortNotFound,
                config.portPath + ": " + std::strerror(errno)));
    }

    if (::ioctl(_impl->fd, TIOCEXCL, nullptr) < 0) {
        ::close(_impl->fd);
        _impl->fd = -1;
        return std::unexpected(
            Error::make(ErrorCode::kSerialPortConfigFailed,
                "Failed to set exclusive access on " + config.portPath));
    }

    struct termios tty{};
    if (::tcgetattr(_impl->fd, &tty) != 0) {
        ::close(_impl->fd);
        _impl->fd = -1;
        return std::unexpected(
            Error::make(ErrorCode::kSerialPortConfigFailed,
                std::string("tcgetattr: ") + std::strerror(errno)));
    }

    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;

    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~ONLCR;

    tty.c_cc[VMIN] = config.vmin;
    tty.c_cc[VTIME] = config.vtime;

    speed_t speed = B115200;
    switch (config.baudRate) {
        case 9600:   speed = B9600;   break;
        case 19200:  speed = B19200;  break;
        case 38400:  speed = B38400;  break;
        case 57600:  speed = B57600;  break;
        case 115200: speed = B115200; break;
        case 230400: speed = B230400; break;
        default:     speed = B115200; break;
    }
    ::cfsetispeed(&tty, speed);
    ::cfsetospeed(&tty, speed);

    if (::tcsetattr(_impl->fd, TCSANOW, &tty) != 0) {
        ::close(_impl->fd);
        _impl->fd = -1;
        return std::unexpected(
            Error::make(ErrorCode::kSerialPortConfigFailed,
                std::string("tcsetattr: ") + std::strerror(errno)));
    }

    return {};
}

Expected<std::size_t> SerialPort::read(std::span<std::uint8_t> buffer)
{
    if (_impl->fd < 0) {
        return std::unexpected(
            Error::make(ErrorCode::kNotInitialized, "Serial port not open"));
    }

    auto bytesRead = ::read(_impl->fd, buffer.data(), buffer.size());
    if (bytesRead < 0) {
        return std::unexpected(
            Error::make(ErrorCode::kSerialReadFailed,
                std::string("read: ") + std::strerror(errno)));
    }

    return static_cast<std::size_t>(bytesRead);
}

Expected<std::size_t> SerialPort::write(std::span<const std::uint8_t> data)
{
    if (_impl->fd < 0) {
        return std::unexpected(
            Error::make(ErrorCode::kNotInitialized, "Serial port not open"));
    }

    auto bytesWritten = ::write(_impl->fd, data.data(), data.size());
    if (bytesWritten < 0) {
        return std::unexpected(
            Error::make(ErrorCode::kSerialWriteFailed,
                std::string("write: ") + std::strerror(errno)));
    }

    return static_cast<std::size_t>(bytesWritten);
}

void SerialPort::close() noexcept
{
    if (_impl->fd >= 0) {
        ::close(_impl->fd);
        _impl->fd = -1;
    }
}

bool SerialPort::isOpen() const noexcept
{
    return _impl->fd >= 0;
}

} // namespace bci::source
