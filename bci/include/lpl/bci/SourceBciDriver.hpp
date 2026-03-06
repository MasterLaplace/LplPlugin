/**
 * @file SourceBciDriver.hpp
 * @brief Adapter bridging ISource → IBciDriver (Adapter pattern).
 *
 * Wraps any ISource (SyntheticSource, CsvReplaySource, etc.) behind the
 * IBciDriver interface expected by BciAdapter.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_BCI_SOURCEBCIDRIVER_HPP
#    define LPL_BCI_SOURCEBCIDRIVER_HPP

#    include <lpl/bci/IBciDriver.hpp>
#    include <lpl/bci/core/Types.hpp>
#    include <lpl/bci/source/ISource.hpp>

#    include <memory>
#    include <span>

namespace lpl::bci {

/**
 * @brief Adapts any ISource to the IBciDriver interface.
 *
 * This enables BciAdapter to work with SyntheticSource, CsvReplaySource,
 * or any future data source without coupling to a specific hardware SDK.
 */
class SourceBciDriver final : public IBciDriver {
public:
    explicit SourceBciDriver(std::unique_ptr<source::ISource> source)
        : _source{std::move(source)}, _status{DriverStatus::Disconnected}
    {
    }

    ~SourceBciDriver() override
    {
        if (_status == DriverStatus::Streaming)
            stopStream();
        if (_status == DriverStatus::Connected)
            disconnect();
    }

    [[nodiscard]] core::Expected<void> connect() override
    {
        _status = DriverStatus::Connected;
        return {};
    }

    [[nodiscard]] core::Expected<void> startStream() override
    {
        auto res = _source->start();
        if (!res)
            return core::makeError(core::ErrorCode::kDeviceOpenFailed, "BCI source start failed");
        _status = DriverStatus::Streaming;
        return {};
    }

    [[nodiscard]] core::Expected<RawSample> poll() override
    {
        Sample buf{};
        std::span<Sample> span{&buf, 1};
        auto readResult = _source->read(span);
        if (!readResult.has_value() || readResult.value() == 0)
            return core::makeError(core::ErrorCode::kInvalidState, "No BCI data available");

        RawSample raw{};
        raw.channelCount = static_cast<core::u16>(buf.channelCount());
        for (core::u16 ch = 0; ch < raw.channelCount && ch < RawSample::kMaxChannels; ++ch)
            raw.channels[ch] = buf.channels[ch];
        raw.timestampUs = static_cast<core::u64>(buf.timestamp * 1e6);
        raw.sequence = _seq++;
        return raw;
    }

    void stopStream() override
    {
        _source->stop();
        _status = DriverStatus::Connected;
    }

    void disconnect() override { _status = DriverStatus::Disconnected; }

    [[nodiscard]] DriverStatus status() const noexcept override { return _status; }

    [[nodiscard]] const char *name() const noexcept override { return "SourceBciDriver"; }

private:
    std::unique_ptr<source::ISource> _source;
    DriverStatus _status;
    core::u32 _seq{0};
};

} // namespace lpl::bci

#endif // LPL_BCI_SOURCEBCIDRIVER_HPP
