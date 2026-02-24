/**
 * @file SerialPortWin32.cpp
 * @brief Win32 implementation of the SerialPort abstraction.
 * @author MasterLaplace
 *
 * Uses CreateFile / ReadFile / WriteFile with DCB configuration.
 * Provides the same interface as the POSIX implementation.
 */

#include "lpl/bci/source/serial/SerialPort.hpp"

#ifdef _WIN32

#include <windows.h>

namespace bci::source {

struct SerialPort::Impl {
    HANDLE handle = INVALID_HANDLE_VALUE;
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
    if (_impl->handle != INVALID_HANDLE_VALUE) {
        return std::unexpected(
            Error::make(ErrorCode::kAlreadyRunning, "Serial port already open"));
    }

    std::string path = "\\\\.\\" + config.portPath;
    _impl->handle = CreateFileA(
        path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

    if (_impl->handle == INVALID_HANDLE_VALUE) {
        return std::unexpected(
            Error::make(ErrorCode::kSerialPortNotFound,
                "Cannot open " + config.portPath));
    }

    DCB dcb{};
    dcb.DCBlength = sizeof(DCB);
    if (!GetCommState(_impl->handle, &dcb)) {
        CloseHandle(_impl->handle);
        _impl->handle = INVALID_HANDLE_VALUE;
        return std::unexpected(
            Error::make(ErrorCode::kSerialPortConfigFailed, "GetCommState failed"));
    }

    dcb.BaudRate = config.baudRate;
    dcb.ByteSize = config.dataBits;
    dcb.StopBits = (config.stopBits == 2) ? TWOSTOPBITS : ONESTOPBIT;
    dcb.Parity = config.parity ? EVENPARITY : NOPARITY;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;

    if (!SetCommState(_impl->handle, &dcb)) {
        CloseHandle(_impl->handle);
        _impl->handle = INVALID_HANDLE_VALUE;
        return std::unexpected(
            Error::make(ErrorCode::kSerialPortConfigFailed, "SetCommState failed"));
    }

    COMMTIMEOUTS timeouts{};
    timeouts.ReadIntervalTimeout = 10;
    timeouts.ReadTotalTimeoutMultiplier = 1;
    timeouts.ReadTotalTimeoutConstant = 100;
    SetCommTimeouts(_impl->handle, &timeouts);

    return {};
}

Expected<std::size_t> SerialPort::read(std::span<std::uint8_t> buffer)
{
    if (_impl->handle == INVALID_HANDLE_VALUE) {
        return std::unexpected(
            Error::make(ErrorCode::kNotInitialized, "Serial port not open"));
    }

    DWORD bytesRead = 0;
    if (!ReadFile(_impl->handle, buffer.data(),
                  static_cast<DWORD>(buffer.size()), &bytesRead, nullptr)) {
        return std::unexpected(
            Error::make(ErrorCode::kSerialReadFailed, "ReadFile failed"));
    }

    return static_cast<std::size_t>(bytesRead);
}

Expected<std::size_t> SerialPort::write(std::span<const std::uint8_t> data)
{
    if (_impl->handle == INVALID_HANDLE_VALUE) {
        return std::unexpected(
            Error::make(ErrorCode::kNotInitialized, "Serial port not open"));
    }

    DWORD bytesWritten = 0;
    if (!WriteFile(_impl->handle, data.data(),
                   static_cast<DWORD>(data.size()), &bytesWritten, nullptr)) {
        return std::unexpected(
            Error::make(ErrorCode::kSerialWriteFailed, "WriteFile failed"));
    }

    return static_cast<std::size_t>(bytesWritten);
}

void SerialPort::close() noexcept
{
    if (_impl->handle != INVALID_HANDLE_VALUE) {
        CloseHandle(_impl->handle);
        _impl->handle = INVALID_HANDLE_VALUE;
    }
}

bool SerialPort::isOpen() const noexcept
{
    return _impl->handle != INVALID_HANDLE_VALUE;
}

} // namespace bci::source

#endif // _WIN32
