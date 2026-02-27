/**
 * @file OpenBciSource.hpp
 * @brief Acquisition source for OpenBCI Cyton hardware via serial port.
 * @author MasterLaplace
 *
 * Implements the ISource interface using a dedicated worker thread
 * (std::jthread) that continuously reads 33-byte packets from the
 * Cyton's USB-serial interface. Parsed samples are buffered in a
 * lock-free SPSC ring buffer for consumption by the main thread.
 *
 * @see ISource, SerialPort, RingBuffer
 */

#pragma once

#include "ISource.hpp"
#include "serial/SerialPort.hpp"
#include "core/Constants.hpp"
#include "dsp/RingBuffer.hpp"

#include <stop_token>
#include <thread>

namespace bci::source {

/**
 * @brief Configuration for an OpenBCI Cyton serial source.
 */
struct OpenBciConfig {
    std::string port = "/dev/ttyUSB0";
    std::uint32_t baudRate = kCytonBaudRate;
    std::size_t channelCount = kDefaultChannelCount;
};

/**
 * @brief Acquires raw EEG samples from an OpenBCI Cyton via serial.
 *
 * The worker thread parses the Cyton's proprietary 33-byte packet format:
 * [0xA0][sampleIndex][8×3-byte channels][accelerometer][0xC0]
 *
 * Each 3-byte channel value is a 24-bit two's complement integer
 * scaled to microvolts using the Cyton ADC scale factor.
 */
class OpenBciSource final : public ISource {
public:
    explicit OpenBciSource(OpenBciConfig config = {});
    ~OpenBciSource() override;

    [[nodiscard]] ExpectedVoid start() override;
    [[nodiscard]] Expected<std::size_t> read(std::span<Sample> buffer) override;
    void stop() noexcept override;
    [[nodiscard]] SourceInfo info() const noexcept override;

private:
    void workerLoop(std::stop_token stopToken);
    static float parseChannel(const std::uint8_t *data);

    OpenBciConfig _config;
    SerialPort _serial;
    dsp::RingBuffer<Sample, kCytonRingSlots> _ring;
    std::jthread _worker;
    bool _started = false;
};

} // namespace bci::source
