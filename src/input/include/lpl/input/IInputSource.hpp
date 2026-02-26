// /////////////////////////////////////////////////////////////////////////////
/// @file IInputSource.hpp
/// @brief Abstract input source interface (keyboard, gamepad, BCI, etc.).
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

namespace lpl::input {

// /////////////////////////////////////////////////////////////////////////////
/// @class IInputSource
/// @brief Strategy interface for polled input sources.
///
/// Concrete implementations provide keyboard, gamepad, VR controller, or
/// BCI neural input.
// /////////////////////////////////////////////////////////////////////////////
class IInputSource
{
public:
    virtual ~IInputSource() = default;

    /// @brief Initializes the input source.
    [[nodiscard]] virtual core::Expected<void> init() = 0;

    /// @brief Polls the device for new data.
    [[nodiscard]] virtual core::Expected<void> poll() = 0;

    /// @brief Shuts down the input source.
    virtual void shutdown() = 0;

    /// @brief Returns a human-readable name.
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace lpl::input
