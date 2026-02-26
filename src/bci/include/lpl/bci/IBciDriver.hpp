// /////////////////////////////////////////////////////////////////////////////
/// @file IBciDriver.hpp
/// @brief Abstract BCI driver interface (Strategy / Bridge pattern).
///
/// Concrete implementations wrap specific BCI hardware SDKs
/// (OpenBCI, Emotiv, NeuroSky, OpenViBE, custom kernel driver).
// /////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

namespace lpl::bci {

/// @brief Raw EEG sample from one acquisition cycle.
struct RawSample
{
    static constexpr core::usize kMaxChannels = 64;

    core::f32  channels[kMaxChannels]{};
    core::u16  channelCount{0};
    core::u64  timestampUs{0};
    core::u32  sequence{0};
};

/// @brief Driver status.
enum class DriverStatus : core::u8
{
    Disconnected,
    Connecting,
    Connected,
    Streaming,
    Error
};

/// @brief Abstract BCI hardware driver.
///
/// Strategy pattern: each hardware SDK provides a concrete driver.
/// The engine only depends on this interface via BciAdapter.
class IBciDriver
{
public:
    virtual ~IBciDriver() = default;

    /// @brief Connect to the device.
    [[nodiscard]] virtual core::Expected<void> connect() = 0;

    /// @brief Start the acquisition stream.
    [[nodiscard]] virtual core::Expected<void> startStream() = 0;

    /// @brief Poll one raw sample (non-blocking).
    /// @return Sample if available, error otherwise.
    [[nodiscard]] virtual core::Expected<RawSample> poll() = 0;

    /// @brief Stop the acquisition stream.
    virtual void stopStream() = 0;

    /// @brief Disconnect from the device.
    virtual void disconnect() = 0;

    /// @brief Current driver status.
    [[nodiscard]] virtual DriverStatus status() const noexcept = 0;

    /// @brief Human-readable driver / hardware name.
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace lpl::bci
