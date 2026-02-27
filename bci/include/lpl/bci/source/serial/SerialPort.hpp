/**
 * @file SerialPort.hpp
 * @brief Cross-platform RAII serial port abstraction.
 * @author MasterLaplace
 *
 * Provides a platform-independent interface for serial communication,
 * with separate implementations for POSIX (Linux/macOS) and Win32.
 * All operations return Expected<> for structured error handling.
 *
 * @see OpenBciSource
 */

#pragma once

#include "lpl/bci/core/Error.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace lpl::bci::source {

/**
 * @brief Configuration parameters for opening a serial port.
 */
struct SerialConfig {
    std::string portPath;
    std::uint32_t baudRate = 115200;
    std::uint8_t dataBits = 8;
    std::uint8_t stopBits = 1;
    bool parity = false;
    std::uint8_t vmin = 33;
    std::uint8_t vtime = 1;
    bool nonBlocking = false;
};

/**
 * @brief RAII serial port handle with cross-platform read/write.
 *
 * The destructor automatically closes the port. Move-only semantics
 * ensure exclusive ownership of the underlying OS handle.
 *
 * @code
 *   SerialPort port;
 *   auto result = port.open({.portPath = "/dev/ttyUSB0", .baudRate = 115200});
 *   if (result) {
 *       std::array<std::uint8_t, 33> buf;
 *       auto bytesRead = port.read(buf);
 *   }
 * @endcode
 */
class SerialPort {
public:
    SerialPort();
    ~SerialPort();

    SerialPort(const SerialPort &) = delete;
    SerialPort &operator=(const SerialPort &) = delete;
    SerialPort(SerialPort &&other) noexcept;
    SerialPort &operator=(SerialPort &&other) noexcept;

    /**
     * @brief Opens and configures the serial port.
     *
     * @param config Port path, baud rate, and framing parameters
     * @return void on success, or an Error
     */
    [[nodiscard]] ExpectedVoid open(const SerialConfig &config);

    /**
     * @brief Reads up to buffer.size() bytes from the port.
     *
     * @param buffer Destination buffer
     * @return Number of bytes actually read, or an Error
     */
    [[nodiscard]] Expected<std::size_t> read(std::span<std::uint8_t> buffer);

    /**
     * @brief Writes the given data to the port.
     *
     * @param data Source buffer
     * @return Number of bytes actually written, or an Error
     */
    [[nodiscard]] Expected<std::size_t> write(std::span<const std::uint8_t> data);

    /**
     * @brief Closes the serial port (idempotent).
     */
    void close() noexcept;

    /**
     * @brief Returns true if the port is currently open.
     */
    [[nodiscard]] bool isOpen() const noexcept;

private:
    struct Impl;
    Impl *_impl;
};

} // namespace lpl::bci::source
