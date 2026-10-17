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

    inline ~SourceBciDriver() override;

    [[nodiscard]] inline core::Expected<void> connect() override;

    [[nodiscard]] inline core::Expected<void> startStream() override;

    [[nodiscard]] inline core::Expected<RawSample> poll() override;

    inline void stopStream() override;

    void disconnect() override { _status = DriverStatus::Disconnected; }

    [[nodiscard]] DriverStatus status() const noexcept override { return _status; }

    [[nodiscard]] const char *name() const noexcept override { return "SourceBciDriver"; }

private:
    std::unique_ptr<source::ISource> _source;
    DriverStatus _status;
    core::u32 _seq{0};
};

} // namespace lpl::bci

#    include "SourceBciDriver.inl"

#endif // LPL_BCI_SOURCEBCIDRIVER_HPP
